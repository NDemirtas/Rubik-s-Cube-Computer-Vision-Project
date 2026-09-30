import glob
import threading
import time
import cv2
import numpy as np
import copy
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from ultralytics import YOLO

from collections import deque
import heapq
import queue
import traceback
import kociemba
from sklearn.cluster import KMeans

# --- FLASK + SOCKET.IO SETUP ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "rubiks_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


class SequenceBufferedTracker:
    """
    Temporal move-deduction engine.

    The important distinction from the old implementation is that the top
    candidate is NOT immediately committed to the authoritative cube state.
    Instead, several complete-cube hypotheses are kept alive across successive
    camera observations.  A candidate only becomes a real move/history entry
    once the competing explanations have been ruled out by later evidence.

    This is deliberately conservative because the camera only observes the
    visible front 3x3.  A hidden B/B' move can therefore be genuinely
    indistinguishable until a later move exposes its effect.
    """

    MOVE_COSTS = {
        "R": 1.0, "R'": 1.0,
        "L": 1.0, "L'": 1.0,
        "U": 1.0, "U'": 1.0,
        "D": 1.0, "D'": 1.0,
        "F": 1.0, "F'": 1.0,
        "B": 1.0, "B'": 1.0,
    }

    MOVE_INVERSES = {
        "R": "R'", "R'": "R",
        "L": "L'", "L'": "L",
        "U": "U'", "U'": "U",
        "D": "D'", "D'": "D",
        "F": "F'", "F'": "F",
        "B": "B'", "B'": "B",
    }

    # Number of complete-cube explanations retained at each observation.
    MAX_HYPOTHESES = 16

    # Normal observations should normally span 0-2 missed moves.  If a
    # hypothesis has no explanation within depth 2, we allow a deeper recovery
    # search for that hypothesis only.
    NORMAL_SEARCH_DEPTH = 2
    RECOVERY_SEARCH_DEPTH = 4

    # Keep several explanations for each possible first move so that a more
    # expensive but correct sequence such as B R cannot be discarded merely
    # because R is cheaper on the first observation.
    MAX_PER_FIRST_MOVE = 2

    # A multi-hypothesis decision must remain ambiguous until later evidence
    # resolves it.  We only allow a score-based fallback after several
    # observations and a substantial lead; the normal path is exact convergence.
    SCORE_FALLBACK_OBSERVATIONS = 3
    SCORE_FALLBACK_MARGIN = 2.0

    # Unresolved inference is deliberately local. After at most three observed
    # state changes without a decisive separation, choose the highest-confidence
    # surviving complete-cube hypothesis and discard the alternatives.
    CONFIDENCE_TEMPERATURE = 1.0

    # Depth-3/4 recovery is expensive; use it only on the strongest hypotheses.
    RECOVERY_HYPOTHESES = 6

    # Short rollback journal for transient classifier glitches.
    #
    # IMPORTANT: normal moves are still committed immediately. The journal is a
    # safety net, not a waiting/provisional queue. If several moves are inferred
    # in a very short burst and the exact pre-burst visible front returns, the
    # whole burst can be treated as a transient colour-detection glitch.
    MAX_ROLLBACK_ENTRIES = 12
    MAX_ROLLBACK_OBSERVATIONS = 8
    GLITCH_RETURN_MAX_MS = 450.0
    GLITCH_EPISODE_MAX_MS = 650.0
    MIN_ROLLBACK_MOVES = 3

    def __init__(self, cube_tracker):
        self.tracker = cube_tracker
        self.cube_tracker = cube_tracker
        self.sequence_queue = queue.Queue(maxsize=10)

        self.last_clean_grid = None
        self.last_queued_grid = None
        self.stable_frames = 0
        self.MIN_HOLD = 2

        # --- LEGACY RECOVERY STATE ---
        self.pending_reconciliation = False
        self.pre_glitch_state = None
        self.unresolved_front_scans = []

        # --- TEMPORAL HYPOTHESIS STATE ---
        self.hypotheses = []
        self._reset_hypotheses_from_tracker()

        # Per-frame deduction telemetry for replay debugging.
        self.last_deduction_debug = {
            "top_candidates": [],
            "chosen": "NONE",
            "recovery_mode": False,
            "status": "IDLE",
            "hypothesis_count": 1,
            "commit_ready": False,
            "commit_reason": "INITIALIZED",
            "first_move_groups": {},
            "best_confidence": 0.0,
            "second_confidence": 0.0,
            "confidence_lead": 0.0,
            "ambiguity_observations": 0,
            "max_pending_moves": 0,
        }
        self.deduction_history = {}
        self._deduction_history_lock = threading.Lock()
        self._hypothesis_lock = threading.Lock()

        # --- DIAGNOSTIC TELEMETRY (NO DEDUCTION BEHAVIOUR CHANGE) ---
        self.metric_lock = threading.Lock()
        self.metrics = {
            "enqueued": 0,
            "processed": 0,
            "dropped": 0,
            "last_queue_delay_ms": 0.0,
            "max_queue_delay_ms": 0.0,
            "last_deduction_ms": 0.0,
            "max_deduction_ms": 0.0,
            "last_queue_depth": 0,
            "max_queue_depth": 0,
            "last_processed_frame": None,
            "last_process_finished": 0.0,
            "last_process_status": "IDLE",
            "last_observation": None,
            "rollbacks": 0,
            "last_rollback_frame": None,
        }

        # --- TRANSIENT GLITCH ROLLBACK JOURNAL ---
        self.observation_serial = 0
        self.rollback_journal = []
        self.last_rollback_info = None
        self.last_authoritative_front_seen_at = None
        self.committed_move_event_count = 0

        self.consumer_thread = threading.Thread(
            target=self._sequence_consumer, daemon=True
        )
        self.consumer_thread.start()
        print("🚀 [SequenceTracker] Temporal Multi-Hypothesis Move Deduction Active...")

    def _normalize(self, grid_data):
        if grid_data is None:
            return []
        if isinstance(grid_data, list) and len(grid_data) > 0:
            if isinstance(grid_data[0], list):
                grid_data = [c for row in grid_data for c in row]
        return [str(x) for x in list(grid_data)]

    def _state_key(self, state):
        return tuple(self._normalize(state))

    def get_debug_for_frame(self, frame_index):
        if frame_index is None:
            return None
        with self._deduction_history_lock:
            data = self.deduction_history.get(frame_index)
            return copy.deepcopy(data) if data is not None else None

    def _save_debug_for_frame(self, frame_index):
        if frame_index is None:
            return
        with self._deduction_history_lock:
            self.deduction_history[frame_index] = copy.deepcopy(self.last_deduction_debug)
            if len(self.deduction_history) > 5000:
                oldest_key = min(self.deduction_history)
                del self.deduction_history[oldest_key]

    def _reset_hypotheses_from_tracker(self):
        """Start a fresh temporal inference chain from the committed cube state."""
        base_state = self.tracker.state.copy()
        self.hypotheses = [{
            "state": base_state,
            "score": 0.0,
            "history": [],
            "observations": 0,
            "last_transition": [],
        }]
        self.pending_reconciliation = False
        self.pre_glitch_state = None
        self.unresolved_front_scans.clear()
        self.ambiguity_window_observations = 0

    def reset_tracking(self):
        """Reset queue/history/hypotheses after the physical cube is reset."""
        with self._hypothesis_lock:
            self.last_clean_grid = None
            self.last_queued_grid = None
            self.stable_frames = 0
            self._reset_hypotheses_from_tracker()

        with self._deduction_history_lock:
            self.deduction_history.clear()

        self.last_deduction_debug = {
            "top_candidates": [],
            "chosen": "NONE",
            "recovery_mode": False,
            "status": "TRACKING RESET",
            "hypothesis_count": 1,
            "commit_ready": False,
            "commit_reason": "TRACKING RESET",
            "first_move_groups": {},
            "state_mutated": False,
        }

        with self.metric_lock:
            for key in self.metrics:
                if key in {"last_process_status", "last_observation"}:
                    continue
                self.metrics[key] = 0.0 if isinstance(self.metrics[key], (int, float)) else None
            self.metrics["last_process_status"] = "TRACKING RESET"

        self.observation_serial = 0
        self.rollback_journal.clear()
        self.last_rollback_info = None
        self.last_authoritative_front_seen_at = None
        self.committed_move_event_count = 0

        # Discard stale observations already waiting in the queue.
        while True:
            try:
                self.sequence_queue.get_nowait()
                self.sequence_queue.task_done()
            except queue.Empty:
                break

    def process_camera_frame(self, scanned_grid, frame_index=None):
        if scanned_grid is None:
            return

        flat_grid = self._normalize(scanned_grid)

        if flat_grid == self.last_clean_grid:
            self.stable_frames += 1
        else:
            self.last_clean_grid = flat_grid
            self.stable_frames = 0
            return

        if self.stable_frames >= self.MIN_HOLD:
            # A grid identical to the authoritative front is not a new visible
            # state. In particular, this prevents an untouched/stationary cube
            # from manufacturing B/B' hypotheses. Hidden moves are recovered
            # only when a later visible change arrives.
            authoritative_front = self._normalize(self.tracker.state[18:27])
            if flat_grid == authoritative_front:
                self.last_queued_grid = flat_grid
                return

            if flat_grid != self.last_queued_grid:
                self.last_queued_grid = flat_grid

                dropped_item = False
                dropped_frame = None
                if self.sequence_queue.full():
                    try:
                        dropped_item = True
                        dropped = self.sequence_queue.get_nowait()
                        dropped_frame = (
                            dropped.get("frame_index")
                            if isinstance(dropped, dict)
                            else None
                        )
                        self.sequence_queue.task_done()
                    except queue.Empty:
                        dropped_item = False

                queue_depth_before_put = self.sequence_queue.qsize()
                queued_at = time.perf_counter()
                self.sequence_queue.put({
                    "grid": flat_grid,
                    "frame_index": frame_index,
                    "queued_at": queued_at,
                    "queue_depth_at_enqueue": queue_depth_before_put,
                })

                with self.metric_lock:
                    self.metrics["enqueued"] += 1
                    self.metrics["last_queue_depth"] = self.sequence_queue.qsize()
                    self.metrics["max_queue_depth"] = max(
                        self.metrics["max_queue_depth"],
                        self.sequence_queue.qsize(),
                    )
                    self.metrics["last_observation"] = list(flat_grid)
                    if dropped_item:
                        self.metrics["dropped"] += 1

                if dropped_item:
                    print(
                        "⚠️ [QUEUE DROP] Consumer was behind; dropped queued "
                        f"observation frame={dropped_frame} to enqueue frame={frame_index}. "
                        f"queue_depth={queue_depth_before_put}"
                    )

    def get_diagnostic_metrics(self):
        with self.metric_lock:
            data = dict(self.metrics)
        data["queue_now"] = self.sequence_queue.qsize()
        return data

    def _search_front_matches(self, start_state, target_front, max_depth, preserve_shallow_ambiguity=False):
        """
        Find exact visible-front explanations.

        Normal tracking can contain genuine ambiguities where a direct one-move
        explanation and a hidden-two-move explanation are both possible, or where
        two hidden prefixes such as ``B R`` and ``B' R`` produce the same visible
        front. Those alternatives must reach the hypothesis pool together so later
        observations can distinguish their complete-cube states.

        When ``preserve_shallow_ambiguity`` is False, the old shortest-depth
        behavior is retained for expensive recovery searches. Normal depth-2
        searches use the flag so depth-1 and depth-2 exact matches are both kept.
        """
        target_front = self._normalize(target_front)

        if len(start_state) != 54 or len(target_front) != 9:
            return []

        initial_front = self._normalize(start_state[18:27])

        # A front observation that has not changed contains no information about
        # a hidden B/B' turn. Treat it as HOLD rather than manufacturing hidden
        # turns while the cube is stationary.
        if target_front == initial_front:
            return [(0.0, [], start_state.copy())]

        matches_by_state = {}

        for depth_limit in range(1, max_depth + 1):
            frontier = deque([(start_state.copy(), [])])
            visited_by_state_depth = set()

            while frontier:
                curr_state, path = frontier.popleft()
                depth = len(path)

                if depth >= depth_limit:
                    continue

                for move_name, perm in self.tracker.permutations.items():
                    if path and self.MOVE_INVERSES.get(path[-1]) == move_name:
                        continue

                    next_state = curr_state[perm]
                    new_path = path + [move_name]
                    next_front = self._normalize(next_state[18:27])

                    if len(new_path) == depth_limit and next_front == target_front:
                        verified_state = self.tracker.apply_move_sequence(
                            start_state, new_path
                        )

                        if verified_state is None:
                            continue

                        if not np.array_equal(verified_state, next_state):
                            print(
                                "⚠️ [INTERNAL CONSISTENCY] Search path/state mismatch; "
                                f"rejecting: {' '.join(new_path)}"
                            )
                            continue

                        if self._normalize(verified_state[18:27]) != target_front:
                            continue

                        result_key = self._state_key(verified_state)
                        if result_key not in matches_by_state:
                            matches_by_state[result_key] = (
                                float(depth_limit),
                                new_path,
                                verified_state.copy(),
                            )
                        continue

                    if len(new_path) < depth_limit:
                        state_depth_key = (
                            self._state_key(next_state),
                            len(new_path),
                        )
                        if state_depth_key not in visited_by_state_depth:
                            visited_by_state_depth.add(state_depth_key)
                            frontier.append((next_state, new_path))

            # In recovery mode we intentionally preserve the old shortest-depth
            # behavior to avoid turning an already-expensive recovery search into
            # a combinatorial explosion. Normal mode continues through depth 2 so
            # shallow hidden alternatives survive alongside direct moves.
            if matches_by_state and not preserve_shallow_ambiguity:
                break

        if not matches_by_state:
            return []

        matches = list(matches_by_state.values())

        # Retain complete-state diversity first. Then make sure each newly
        # proposed first move gets representation in the beam. This is important
        # for cases such as B R vs B' R, where the visible front is identical but
        # the hidden complete-cube states are different.
        grouped = {}
        for item in matches:
            first_move = item[1][0] if item[1] else "HOLD"
            grouped.setdefault(first_move, []).append(item)

        selected = []
        for first_move in sorted(grouped):
            group = sorted(
                grouped[first_move],
                key=lambda x: (x[0], len(x[1]), tuple(x[1]))
            )
            selected.extend(group[:self.MAX_PER_FIRST_MOVE])

        selected.sort(key=lambda x: (x[0], len(x[1]), tuple(x[1])))

        # Hidden back-face turns are uniquely important with a front-only camera:
        # B R and B' R can have the same visible front while producing different
        # complete-cube states. Never allow the global beam cap to evict BOTH
        # hidden branches when they were present in the exact-match set.
        priority_items = []
        for hidden_move in ("B", "B'"):
            hidden_candidates = grouped.get(hidden_move, [])
            if hidden_candidates:
                priority_items.append(
                    min(
                        hidden_candidates,
                        key=lambda x: (x[0], len(x[1]), tuple(x[1]))
                    )
                )

        if priority_items:
            existing_keys = {self._state_key(item[2]) for item in selected}
            for priority in priority_items:
                key = self._state_key(priority[2])
                if key in existing_keys:
                    continue

                if len(selected) >= self.MAX_HYPOTHESES:
                    selected[-1] = priority
                else:
                    selected.append(priority)
                existing_keys.add(key)

            selected.sort(key=lambda x: (x[0], len(x[1]), tuple(x[1])))

        return selected[:self.MAX_HYPOTHESES]

    def _generate_successors_for_hypothesis(
        self, hypothesis, target_front, allow_recovery=True
    ):
        """Use exact depth-1/2 first; optionally allow depth-3/4 recovery."""
        matches = self._search_front_matches(
            hypothesis["state"],
            target_front,
            self.NORMAL_SEARCH_DEPTH,
            preserve_shallow_ambiguity=True,
        )
        used_recovery = False

        if not matches and allow_recovery:
            matches = self._search_front_matches(
                hypothesis["state"],
                target_front,
                self.RECOVERY_SEARCH_DEPTH,
            )
            used_recovery = bool(matches)

        return matches, used_recovery

    def _prune_global_hypotheses(self, candidates):
        """Deduplicate by complete cube state while retaining first-move diversity."""
        by_state = {}
        for h in candidates:
            key = self._state_key(h["state"])
            previous = by_state.get(key)
            if previous is None or h["score"] < previous["score"] - 1e-9:
                by_state[key] = h

        deduped = list(by_state.values())
        deduped.sort(key=lambda h: (h["score"], len(h["history"]), tuple(h["history"])))

        # First preserve diversity by first move of the NEW transition.
        groups = {}
        for h in deduped:
            transition = h.get("last_transition") or []
            first_move = transition[0] if transition else "HOLD"
            groups.setdefault(first_move, []).append(h)

        selected = []
        for first_move in sorted(groups):
            selected.extend(groups[first_move][:self.MAX_PER_FIRST_MOVE])

        selected.sort(key=lambda h: (h["score"], len(h["history"]), tuple(h["history"])))
        return selected[:self.MAX_HYPOTHESES]

    def _longest_common_prefix(self, histories):
        """Return the longest move prefix shared by every surviving hypothesis."""
        if not histories:
            return []

        shortest = min(len(h) for h in histories)
        if shortest == 0:
            return []

        prefix = []
        for i in range(shortest):
            move = histories[0][i]
            if all(h[i] == move for h in histories[1:]):
                prefix.append(move)
            else:
                break
        return prefix

    def _compute_hypothesis_confidences(self, hypotheses=None):
        """
        Convert the current hypotheses' relative path costs into a bounded
        confidence estimate.

        This is a heuristic confidence estimate, not a calibrated probability.
        All surviving hypotheses already satisfy the observed front exactly, so
        the score is the engine's only remaining preference signal.
        """
        hypotheses = list(hypotheses or self.hypotheses)
        if not hypotheses:
            return []

        scores = np.array(
            [float(h.get("score", 0.0)) for h in hypotheses],
            dtype=float,
        )
        min_score = float(np.min(scores))
        weights = np.exp(
            -(scores - min_score) / max(self.CONFIDENCE_TEMPERATURE, 1e-9)
        )
        total = float(np.sum(weights))
        if total <= 0.0:
            weights[:] = 1.0
            total = float(len(weights))

        confidence_values = (weights / total).tolist()
        ranked = sorted(
            zip(hypotheses, confidence_values),
            key=lambda pair: (
                -pair[1],
                float(pair[0].get("score", 0.0)),
                len(pair[0].get("history", [])),
                tuple(pair[0].get("history", [])),
            ),
        )
        return ranked

    def _summarize_hypotheses(self):
        """Summarize only the current local inference window."""
        ordered = sorted(
            self.hypotheses,
            key=lambda h: (h["score"], len(h["history"]), tuple(h["history"])),
        )

        ranked_confidence = self._compute_hypothesis_confidences(ordered)

        top_candidates = [
            (
                float(h["score"]),
                list(h.get("history", [])),
            )
            for h, _ in ranked_confidence[:6]
        ]

        first_move_groups = {}
        for h in ordered:
            history = h.get("history", [])
            first_move = history[0] if history else "HOLD"
            current = first_move_groups.get(first_move)
            if current is None or h["score"] < current:
                first_move_groups[first_move] = float(h["score"])

        histories = [list(h.get("history", [])) for h in ordered]
        common_prefix = self._longest_common_prefix(histories)

        best_confidence = (
            float(ranked_confidence[0][1]) if ranked_confidence else 0.0
        )
        second_confidence = (
            float(ranked_confidence[1][1])
            if len(ranked_confidence) > 1
            else 0.0
        )
        confidence_lead = best_confidence - second_confidence

        ambiguity_observations = max(
            [int(h.get("observations", 0)) for h in ordered] or [0]
        )
        max_pending_moves = max(
            [len(h.get("history", [])) for h in ordered] or [0]
        )

        if common_prefix:
            commit_ready = True
            commit_reason = (
                f"COMMON PREFIX CONFIRMED: {' '.join(common_prefix)}"
            )
        elif ambiguity_observations >= self.SCORE_FALLBACK_OBSERVATIONS:
            commit_ready = True
            commit_reason = (
                f"3-OBSERVATION MAX REACHED; CHOOSE HIGHEST-CONFIDENCE HYPOTHESIS"
            )
        elif len(ordered) == 1 and max_pending_moves > 0:
            commit_ready = True
            commit_reason = "ONLY ONE COMPLETE HYPOTHESIS REMAINS"
        else:
            commit_ready = False
            commit_reason = "WAITING FOR NEXT OBSERVATION TO RESOLVE AMBIGUITY"

        return {
            "top_candidates": top_candidates,
            "hypothesis_count": len(ordered),
            "commit_ready": commit_ready,
            "commit_reason": commit_reason,
            "first_move_groups": first_move_groups,
            "common_prefix": list(common_prefix),
            "best_confidence": round(best_confidence, 4),
            "second_confidence": round(second_confidence, 4),
            "confidence_lead": round(confidence_lead, 4),
            "ambiguity_observations": ambiguity_observations,
            "max_pending_moves": max_pending_moves,
        }

    def _commit_best_hypothesis(self, target_front=None, frame_index=None, observation_time=None):
        """
        Force a bounded ambiguity window to resolve by taking the highest-
        confidence surviving complete-cube hypothesis.
        """
        ranked = self._compute_hypothesis_confidences(self.hypotheses)
        if not ranked:
            return None

        best, confidence = ranked[0]
        local_history = list(best.get("history", []))
        target_front = self._normalize(target_front)

        if not local_history:
            self._reset_hypotheses_from_tracker()
            return {
                "history": [],
                "confidence": float(confidence),
                "hypothesis_count": len(ranked),
            }

        if not self.tracker.history_matches_state():
            print("❌ [CONFIDENCE COMMIT] Authoritative state/history mismatch; refusing commit.")
            return None

        proposed_state = self.tracker.apply_move_sequence(
            self.tracker.state, local_history
        )
        if proposed_state is None:
            print("❌ [CONFIDENCE COMMIT] Selected hypothesis contains invalid moves; refusing commit.")
            return None

        if not np.array_equal(proposed_state, best["state"]):
            print(
                "❌ [CONFIDENCE COMMIT] Selected hypothesis state does not match "
                f"authoritative replay: {' '.join(local_history)}"
            )
            return None

        if target_front and self._normalize(proposed_state[18:27]) != target_front:
            print("❌ [CONFIDENCE COMMIT] Selected hypothesis does not reproduce target front; refusing commit.")
            return None

        old_state = self.tracker.state.copy()
        old_history = list(self.tracker.move_history)
        old_previous_state = self.tracker.previous_state.copy()
        old_last_move = self.tracker.last_move_detected

        proposed_history = self.tracker.compressed_history(
            old_history + local_history
        )
        rebuilt = self.tracker.state_from_history(proposed_history)
        if rebuilt is None or not np.array_equal(rebuilt, proposed_state):
            print("❌ [CONFIDENCE COMMIT] History cannot reproduce selected state; refusing commit.")
            return None

        self.tracker.state = proposed_state.copy()
        self.tracker.move_history = proposed_history
        self.tracker.previous_state = proposed_state.copy()
        self.tracker.last_move_detected = local_history[-1]

        if not self.tracker.history_matches_state():
            self.tracker.state = old_state
            self.tracker.move_history = old_history
            self.tracker.previous_state = old_previous_state
            self.tracker.last_move_detected = old_last_move
            print("❌ [CONFIDENCE COMMIT] Post-commit invariant failed; rolled back.")
            return None

        # Once the bounded decision is made, the losing hypotheses are discarded
        # completely. The next observation starts from the chosen real state.
        self._reset_hypotheses_from_tracker()

        # Journal EVERY accepted commit. This does not delay the move; it only
        # gives the temporal glitch detector a checkpoint it can restore later
        # when a rapid false burst returns to this exact front.
        atomic_local_history = flatten_move_sequence(local_history)
        self.rollback_journal.append({
            "observation_serial": self.observation_serial,
            "frame_index": frame_index,
            "before_state": old_state.copy(),
            "before_history": list(old_history),
            "before_front": self._normalize(old_state[18:27]),
            "after_history": list(self.tracker.move_history),
            "after_front": self._normalize(self.tracker.state[18:27]),
            "transition": list(atomic_local_history),
            "before_move_event_count": int(self.committed_move_event_count),
            "commit_observed_at": (
                float(observation_time)
                if observation_time is not None
                else time.perf_counter()
            ),
            "checkpoint_observed_at": (
                float(self.last_authoritative_front_seen_at)
                if self.last_authoritative_front_seen_at is not None
                else (
                    float(observation_time)
                    if observation_time is not None
                    else time.perf_counter()
                )
            ),
        })
        self.committed_move_event_count += sum(
            2 if str(m).endswith("2") else 1 for m in atomic_local_history
        )
        if len(self.rollback_journal) > self.MAX_ROLLBACK_ENTRIES:
            self.rollback_journal = self.rollback_journal[-self.MAX_ROLLBACK_ENTRIES:]

        return {
            "history": local_history,
            "confidence": float(confidence),
            "hypothesis_count": len(ranked),
        }

    def _maybe_rollback_transient_glitch(
        self, target_front, frame_index=None, observation_time=None
    ):
        """
        Detect a very short A -> several inferred moves -> A episode.

        Normal moves are NOT delayed. Every accepted move is committed immediately
        and journaled. Only when a later camera observation returns to the exact
        visible front that existed before a recent burst do we consider rolling the
        burst back. Three independent signals are required:

          1. exact visible-front return to a prior checkpoint;
          2. at least MIN_ROLLBACK_MOVES committed since that checkpoint;
          3. the return happened very quickly in camera-observation time.

        This specifically targets classifier glitches that manufacture a handful
        of moves and then manufacture their inverse sequence when the colours
        recover. It does NOT make ordinary moves wait for later observations.
        """
        target_front = self._normalize(target_front)
        if not target_front or not self.rollback_journal:
            return None

        current_serial = self.observation_serial
        current_time = (
            float(observation_time)
            if observation_time is not None
            else time.perf_counter()
        )

        # Search oldest -> newest so a whole bad burst is removed as one episode.
        candidate_index = None
        for idx, entry in enumerate(self.rollback_journal):
            serial_delta = current_serial - int(entry.get("observation_serial", 0))
            if serial_delta < 1 or serial_delta > self.MAX_ROLLBACK_OBSERVATIONS:
                continue

            if self._normalize(entry.get("before_front", [])) != target_front:
                continue

            before_events = int(entry.get("before_move_event_count", 0))
            moves_since = self.committed_move_event_count - before_events
            if moves_since < self.MIN_ROLLBACK_MOVES:
                continue

            commit_time = entry.get("commit_observed_at")
            checkpoint_time = entry.get("checkpoint_observed_at")
            if commit_time is None:
                continue

            return_ms = max(0.0, (current_time - float(commit_time)) * 1000.0)
            episode_ms = (
                max(0.0, (current_time - float(checkpoint_time)) * 1000.0)
                if checkpoint_time is not None
                else return_ms
            )

            if return_ms > self.GLITCH_RETURN_MAX_MS:
                continue
            if episode_ms > self.GLITCH_EPISODE_MAX_MS:
                continue

            entry["return_observed_at"] = current_time
            entry["return_ms"] = return_ms
            entry["episode_ms"] = episode_ms
            entry["moves_since_checkpoint"] = moves_since
            candidate_index = idx
            break

        if candidate_index is None:
            # Keep only genuinely recent checkpoints.
            fresh = []
            for entry in self.rollback_journal:
                serial_delta = current_serial - int(entry.get("observation_serial", 0))
                if 0 <= serial_delta <= self.MAX_ROLLBACK_OBSERVATIONS:
                    fresh.append(entry)
            self.rollback_journal = fresh
            return None

        entry = self.rollback_journal[candidate_index]
        old_state = self.tracker.state.copy()
        old_history = list(self.tracker.move_history)

        self.tracker.state = entry["before_state"].copy()
        self.tracker.move_history = list(entry["before_history"])
        self.tracker.previous_state = self.tracker.state.copy()
        self.tracker.last_move_detected = "ROLLED BACK: transient colour read"

        # Remove this checkpoint and all later checkpoints from the bad burst.
        self.rollback_journal = self.rollback_journal[:candidate_index]
        self.committed_move_event_count = int(entry.get("before_move_event_count", 0))
        self._reset_hypotheses_from_tracker()
        self.last_authoritative_front_seen_at = current_time

        if not self.tracker.history_matches_state():
            self.tracker.state = old_state
            self.tracker.move_history = old_history
            self.tracker.previous_state = old_state.copy()
            self.committed_move_event_count = sum(
                2 if str(m).endswith("2") else 1 for m in old_history
            )
            self._reset_hypotheses_from_tracker()
            print(
                "❌ [ROLLBACK GUARD] Saved checkpoint failed state/history "
                "invariant; rollback aborted."
            )
            return None

        info = {
            "frame_index": frame_index,
            "from_observation_serial": entry.get("observation_serial"),
            "before_history": list(entry.get("before_history", [])),
            "after_history": list(entry.get("after_history", [])),
            "transition": list(entry.get("transition", [])),
            "rolled_back_history": list(old_history),
            "restored_history": list(self.tracker.move_history),
            "return_ms": round(float(entry.get("return_ms", 0.0)), 2),
            "episode_ms": round(float(entry.get("episode_ms", 0.0)), 2),
            "moves_since_checkpoint": int(entry.get("moves_since_checkpoint", 0)),
        }
        self.last_rollback_info = info

        with self.metric_lock:
            self.metrics["rollbacks"] = self.metrics.get("rollbacks", 0) + 1
            self.metrics["last_rollback_frame"] = frame_index
            self.metrics["last_process_status"] = "ROLLED BACK TRANSIENT GLITCH"

        print(
            "🔄 [GLITCH ROLLBACK] Exact pre-burst front returned quickly: "
            f"return={info['return_ms']:.1f}ms, "
            f"episode={info['episode_ms']:.1f}ms, "
            f"moves={info['moves_since_checkpoint']}. "
            "Discarded the entire rapid detection burst."
        )
        return info

    def _commit_common_prefix(self, common_prefix, target_front=None, speculative=False, frame_index=None, observation_time=None):
        """
        Commit only the moves proven common to all surviving explanations.

        Hidden/unresolved suffixes remain in the complete-cube hypotheses instead
        of being silently discarded. The authoritative state/history advances only
        by the committed prefix.
        """
        common_prefix = list(common_prefix or [])
        if not common_prefix:
            return False

        if not self.tracker.history_matches_state():
            print(
                "❌ [STATE GUARD] Authoritative state/history mismatch; "
                "refusing to commit."
            )
            return False

        checkpoint_before_state = self.tracker.state.copy()
        checkpoint_before_history = list(self.tracker.move_history)
        checkpoint_before_front = self._normalize(
            checkpoint_before_state[18:27]
        )

        proposed_state = self.tracker.apply_move_sequence(
            self.tracker.state, common_prefix
        )
        if proposed_state is None:
            print(
                "❌ [STATE GUARD] Invalid proposed move sequence: "
                f"{common_prefix}"
            )
            return False

        old_history = list(self.tracker.move_history)
        proposed_history = self.tracker.compressed_history(
            old_history + common_prefix
        )

        rebuilt = self.tracker.state_from_history(proposed_history)
        if rebuilt is None or not np.array_equal(rebuilt, proposed_state):
            print(
                "❌ [STATE GUARD] Proposed history cannot reproduce proposed state; "
                "commit aborted."
            )
            return False

        # Commit only after all consistency checks pass. Keep a rollback copy so
        # an unexpected post-commit invariant failure can never leave a half-
        # committed state behind.
        old_state = self.tracker.state.copy()
        old_history_snapshot = list(self.tracker.move_history)
        old_previous_state = self.tracker.previous_state.copy()
        old_last_move = self.tracker.last_move_detected

        self.tracker.state = proposed_state.copy()
        self.tracker.move_history = proposed_history
        self.tracker.previous_state = proposed_state.copy()
        self.tracker.last_move_detected = common_prefix[-1]

        # Keep the complete-cube hypotheses alive. Strip only the prefix now
        # proven, preserving any hidden suffix such as B/B'.
        prefix_len = len(common_prefix)
        prefix_cost = sum(
            self.MOVE_COSTS.get(move, 1.0) for move in common_prefix
        )

        retained = []
        for hypothesis in self.hypotheses:
            history = list(hypothesis.get("history", []))
            if (
                len(history) >= prefix_len
                and history[:prefix_len] == common_prefix
            ):
                hypothesis["history"] = history[prefix_len:]
                hypothesis["score"] = max(
                    0.0,
                    float(hypothesis.get("score", 0.0)) - prefix_cost,
                )
                retained.append(hypothesis)

        if retained:
            self.hypotheses = self._prune_global_hypotheses(retained)
            if not any(h.get("history", []) for h in self.hypotheses):
                self._reset_hypotheses_from_tracker()
        else:
            self._reset_hypotheses_from_tracker()

        if not self.tracker.history_matches_state():
            print("❌ [STATE GUARD] Post-commit state/history mismatch detected; rolling back!")
            self.tracker.state = old_state
            self.tracker.move_history = old_history_snapshot
            self.tracker.previous_state = old_previous_state
            self.tracker.last_move_detected = old_last_move
            return False

        # Journal EVERY accepted commit, including ordinary one-move commits.
        # The journal is only consulted when a later observation rapidly returns
        # to the exact checkpoint front. There is no waiting here, so normal moves
        # still print immediately.
        atomic_common_prefix = flatten_move_sequence(common_prefix)
        self.rollback_journal.append({
            "observation_serial": self.observation_serial,
            "frame_index": frame_index,
            "before_state": checkpoint_before_state.copy(),
            "before_history": list(checkpoint_before_history),
            "before_front": list(checkpoint_before_front),
            "after_history": list(self.tracker.move_history),
            "after_front": self._normalize(self.tracker.state[18:27]),
            "transition": list(atomic_common_prefix),
            "before_move_event_count": int(self.committed_move_event_count),
            "commit_observed_at": (
                float(observation_time)
                if observation_time is not None
                else time.perf_counter()
            ),
            "checkpoint_observed_at": (
                float(self.last_authoritative_front_seen_at)
                if self.last_authoritative_front_seen_at is not None
                else (
                    float(observation_time)
                    if observation_time is not None
                    else time.perf_counter()
                )
            ),
        })
        self.committed_move_event_count += sum(
            2 if str(m).endswith("2") else 1 for m in atomic_common_prefix
        )
        if len(self.rollback_journal) > self.MAX_ROLLBACK_ENTRIES:
            self.rollback_journal = self.rollback_journal[-self.MAX_ROLLBACK_ENTRIES:]

        return True

    def _sequence_consumer(self):
        while True:
            try:
                queued_item = self.sequence_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                if isinstance(queued_item, dict):
                    target_front = queued_item.get("grid", [])
                    frame_index = queued_item.get("frame_index")
                else:
                    target_front = queued_item
                    frame_index = None

                target_front = self._normalize(target_front)
                self.observation_serial += 1
                process_started = time.perf_counter()
                queued_at = queued_item.get("queued_at") if isinstance(queued_item, dict) else None
                queue_delay_ms = (
                    max(0.0, (process_started - queued_at) * 1000.0)
                    if queued_at is not None
                    else 0.0
                )

                with self.metric_lock:
                    self.metrics["last_queue_delay_ms"] = queue_delay_ms
                    self.metrics["max_queue_delay_ms"] = max(
                        self.metrics["max_queue_delay_ms"], queue_delay_ms
                    )
                    self.metrics["last_queue_depth"] = self.sequence_queue.qsize()
                    self.metrics["max_queue_depth"] = max(
                        self.metrics["max_queue_depth"],
                        self.sequence_queue.qsize(),
                    )

                self.last_deduction_debug = {
                    "top_candidates": [],
                    "chosen": "NONE",
                    "recovery_mode": False,
                    "status": "NO DEDUCTION",
                    "hypothesis_count": len(self.hypotheses),
                    "commit_ready": False,
                    "commit_reason": "PROCESSING",
                    "first_move_groups": {},
                    "common_prefix": [],
                    "state_mutated": False,
                    "queue_delay_ms": round(queue_delay_ms, 2),
                    "queue_depth_at_enqueue": (
                        queued_item.get("queue_depth_at_enqueue")
                        if isinstance(queued_item, dict)
                        else None
                    ),
                }

                if (
                    len(target_front) != 9
                    or "Unknown" in target_front
                ):
                    self.last_deduction_debug.update({
                        "status": "INVALID OBSERVATION / DEFERRED",
                        "commit_reason": "OBSERVATION CONTAINS UNKNOWN OR BAD SIZE",
                        "state_mutated": False,
                    })
                    self.tracker.last_move_detected = "Analyzing..."
                    self.tracker.broadcast_state()
                    continue

                with self._hypothesis_lock:
                    rollback_info = self._maybe_rollback_transient_glitch(
                        target_front,
                        frame_index=frame_index,
                        observation_time=observation_time,
                    )
                    if rollback_info is not None:
                        self.last_deduction_debug.update({
                            "status": "ROLLED BACK TRANSIENT GLITCH",
                            "recovery_mode": True,
                            "chosen": "ROLLBACK",
                            "commit_ready": True,
                            "commit_reason": (
                                "OBSERVATION RETURNED TO PRE-SPECULATIVE KNOWN-GOOD FRONT"
                            ),
                            "common_prefix": [],
                            "state_mutated": True,
                            "rolled_back_transition": rollback_info.get("transition", []),
                            "rollback_restored_history": rollback_info.get("restored_history", []),
                        })
                        self.tracker.broadcast_state()
                        continue

                    # This observation is currently consistent with the engine's
                    # authoritative front. Remember when that known-good checkpoint
                    # was actually seen by the camera; later commits can use this
                    # timestamp to measure a rapid A -> burst -> A episode.
                    if self._normalize(self.tracker.state[18:27]) == target_front:
                        self.last_authoritative_front_seen_at = observation_time

                    # Hard guard: never let an earlier bug contaminate the next
                    # inference cycle.
                    if not self.tracker.history_matches_state():
                        print(
                            "❌ [STATE GUARD] Authoritative state/history mismatch; "
                            "freezing move commits."
                        )
                        self.last_deduction_debug.update({
                            "status": "ENGINE STATE CORRUPTION DETECTED",
                            "commit_reason": "STATE DOES NOT MATCH MOVE HISTORY",
                            "state_mutated": False,
                        })
                        self.tracker.last_move_detected = "STATE ERROR"
                        self.tracker.broadcast_state()
                        continue

                    candidate_hypotheses = []
                    used_recovery = False

                    ordered_inputs = sorted(
                        self.hypotheses,
                        key=lambda h: (
                            h.get("score", 0.0),
                            len(h.get("history", [])),
                            tuple(h.get("history", [])),
                        ),
                    )

                    def append_verified_candidates(hypothesis, matches):
                        for cost, transition, next_state in matches:
                            verified_state = self.tracker.apply_move_sequence(
                                hypothesis["state"],
                                transition,
                            )

                            # Candidate must survive all three checks.
                            if verified_state is None:
                                continue
                            if not np.array_equal(verified_state, next_state):
                                print(
                                    "⚠️ [STATE GUARD] Candidate transition/state "
                                    f"mismatch: {' '.join(transition)}"
                                )
                                continue
                            if self._normalize(verified_state[18:27]) != target_front:
                                continue

                            new_history = (
                                list(hypothesis["history"])
                                + list(transition)
                            )

                            candidate_hypotheses.append({
                                "state": verified_state.copy(),
                                "score": hypothesis["score"] + float(cost),
                                "history": new_history,
                                "observations": hypothesis["observations"] + 1,
                                "last_transition": list(transition),
                            })

                    # PASS 1: run only the cheap normal depth-1/2 search across
                    # every surviving hypothesis. If ANY exact explanation
                    # exists, do not launch expensive depth-4 recovery searches
                    # for the other hypotheses. This is the main latency fix.
                    for hypothesis in ordered_inputs:
                        matches, _ = self._generate_successors_for_hypothesis(
                            hypothesis,
                            target_front,
                            allow_recovery=False,
                        )
                        append_verified_candidates(hypothesis, matches)

                    if candidate_hypotheses:
                        used_recovery = False
                    else:
                        # PASS 2: only when normal search found nothing at all,
                        # run expensive recovery on the strongest few hypotheses.
                        for hypothesis in ordered_inputs[:self.RECOVERY_HYPOTHESES]:
                            matches, recovered = self._generate_successors_for_hypothesis(
                                hypothesis,
                                target_front,
                                allow_recovery=True,
                            )
                            used_recovery = used_recovery or recovered
                            append_verified_candidates(hypothesis, matches)

                    self.hypotheses = self._prune_global_hypotheses(
                        candidate_hypotheses
                    )

                    self.ambiguity_window_observations = max(
                        [int(h.get("observations", 0)) for h in self.hypotheses] or [0]
                    )

                    if not self.hypotheses:
                        # FAIL CLOSED. Never fabricate a cube state from a bad
                        # camera classification.
                        print(
                            "⚠️ [SequenceTracker] Observation rejected: no exact "
                            "physical path. Authoritative state/history unchanged."
                        )

                        self.last_deduction_debug.update({
                            "status": (
                                "OBSERVATION REJECTED — NO PHYSICAL PATH"
                            ),
                            "queue_delay_ms": round(queue_delay_ms, 2),
                            "deduction_ms": round((time.perf_counter() - process_started) * 1000.0, 2),
                            "recovery_mode": True,
                            "chosen": "NONE",
                            "hypothesis_count": 0,
                            "commit_reason": (
                                "NO EXACT PATH; AUTHORITATIVE STATE UNCHANGED"
                            ),
                            "common_prefix": [],
                            "state_mutated": False,
                        })

                        self.tracker.last_move_detected = (
                            "Rejected observation"
                        )

                        # Restart inference from the last known-good state,
                        # NOT from the camera observation.
                        self._reset_hypotheses_from_tracker()
                        self.tracker.broadcast_state()
                        continue

                    summary = self._summarize_hypotheses()
                    self.last_deduction_debug.update(summary)
                    self.last_deduction_debug["queue_delay_ms"] = round(queue_delay_ms, 2)
                    self.last_deduction_debug["deduction_ms"] = round(
                        (time.perf_counter() - process_started) * 1000.0, 2
                    )
                    self.last_deduction_debug["queue_depth_at_process"] = self.sequence_queue.qsize()
                    self.last_deduction_debug["recovery_mode"] = used_recovery
                    self.last_deduction_debug["status"] = (
                        "RECOVERY HYPOTHESES HELD"
                        if used_recovery
                        else "TEMPORAL HYPOTHESES HELD"
                    )

                    ordered = sorted(
                        self.hypotheses,
                        key=lambda h: (
                            h["score"],
                            len(h["history"]),
                            tuple(h["history"]),
                        ),
                    )

                    common_prefix = summary.get("common_prefix", [])
                    ambiguity_reached = (
                        not common_prefix
                        and summary.get("ambiguity_observations", 0)
                        >= self.SCORE_FALLBACK_OBSERVATIONS
                    )

                    if common_prefix:
                        committed = self._commit_common_prefix(
                            common_prefix,
                            target_front=target_front,
                            speculative=(used_recovery or len(common_prefix) > 1),
                            frame_index=frame_index,
                            observation_time=observation_time,
                        )

                        if committed:
                            self.last_deduction_debug["chosen"] = list(
                                common_prefix
                            )
                            self.last_deduction_debug["status"] = (
                                "COMMITTED COMMON PREFIX / RECOVERED"
                                if used_recovery
                                else "COMMITTED COMMON PREFIX"
                            )
                            self.last_deduction_debug["commit_ready"] = True
                            self.last_deduction_debug["commit_reason"] = (
                                summary["commit_reason"]
                            )
                            self.last_deduction_debug["common_prefix"] = list(
                                common_prefix
                            )
                            self.last_deduction_debug["hypothesis_count"] = len(
                                self.hypotheses
                            )
                            self.last_deduction_debug["state_mutated"] = True
                            self.last_authoritative_front_seen_at = observation_time
                        else:
                            self.last_deduction_debug.update({
                                "chosen": "NONE",
                                "status": "COMMIT BLOCKED BY STATE GUARD",
                                "commit_ready": False,
                                "commit_reason": (
                                    "STATE/HISTORY CONSISTENCY CHECK FAILED"
                                ),
                                "state_mutated": False,
                            })

                        self.tracker.broadcast_state()
                    elif ambiguity_reached:
                        confidence_commit = self._commit_best_hypothesis(
                            target_front=target_front,
                            frame_index=frame_index,
                            observation_time=observation_time,
                        )

                        if confidence_commit is not None:
                            chosen_history = list(
                                confidence_commit.get("history", [])
                            )
                            chosen_confidence = float(
                                confidence_commit.get("confidence", 0.0)
                            )

                            self.last_deduction_debug.update({
                                "chosen": chosen_history,
                                "status": "COMMITTED HIGHEST-CONFIDENCE AFTER 3 OBSERVATIONS",
                                "commit_ready": True,
                                "commit_reason": (
                                    "3-OBSERVATION WINDOW EXPIRED; "
                                    f"CHOSE HIGHEST-CONFIDENCE HYPOTHESIS "
                                    f"({chosen_confidence * 100.0:.1f}%)"
                                ),
                                "common_prefix": [],
                                "hypothesis_count": confidence_commit.get(
                                    "hypothesis_count", 1
                                ),
                                "state_mutated": bool(chosen_history),
                                "chosen_confidence": round(chosen_confidence, 4),
                            })
                            if chosen_history:
                                self.last_authoritative_front_seen_at = observation_time
                        else:
                            self.last_deduction_debug.update({
                                "chosen": "NONE",
                                "status": "CONFIDENCE COMMIT BLOCKED BY STATE GUARD",
                                "commit_ready": False,
                                "commit_reason": (
                                    "HIGHEST-CONFIDENCE HYPOTHESIS FAILED "
                                    "STATE VERIFICATION"
                                ),
                                "state_mutated": False,
                            })

                        self.tracker.broadcast_state()
                    else:
                        tentative = ordered[0].get(
                            "last_transition", []
                        )
                        if tentative:
                            self.tracker.last_move_detected = (
                                "TENTATIVE: "
                                + " ".join(tentative)
                            )
                        else:
                            self.tracker.last_move_detected = (
                                "TENTATIVE / HOLD"
                            )

                        self.tracker.broadcast_state()

                        self.tracker.broadcast_state()

            except Exception as e:
                print(f"❌ [Consumer Exception]: {e}")
                traceback.print_exc()

            finally:
                finished_at = time.perf_counter()
                deduction_elapsed_ms = (
                    max(0.0, (finished_at - process_started) * 1000.0)
                    if "process_started" in locals()
                    else 0.0
                )
                with self.metric_lock:
                    self.metrics["processed"] += 1
                    self.metrics["last_deduction_ms"] = deduction_elapsed_ms
                    self.metrics["max_deduction_ms"] = max(
                        self.metrics["max_deduction_ms"],
                        deduction_elapsed_ms,
                    )
                    self.metrics["last_processed_frame"] = locals().get("frame_index")
                    self.metrics["last_process_finished"] = finished_at
                    self.metrics["last_process_status"] = (
                        self.last_deduction_debug.get("status", "UNKNOWN")
                        if isinstance(self.last_deduction_debug, dict)
                        else "UNKNOWN"
                    )

                if queue_delay_ms > 100.0 or deduction_elapsed_ms > 100.0:
                    print(
                        "⏱️ [PIPELINE TIMING] "
                        f"frame={locals().get('frame_index')} "
                        f"queue_delay={queue_delay_ms:.1f}ms "
                        f"deduction={deduction_elapsed_ms:.1f}ms "
                        f"queue_now={self.sequence_queue.qsize()} "
                        f"hypotheses={len(self.hypotheses)} "
                        f"status={self.metrics['last_process_status']}"
                    )

                self._save_debug_for_frame(locals().get("frame_index"))
                self.sequence_queue.task_done()

    def _reconcile_multi_step_bridge(self, start_state, target_front, max_depth=4):
        """Compatibility helper for the legacy recovery path."""
        matches = self._search_front_matches(start_state, target_front, max_depth)
        if not matches:
            return "UNKNOWN"
        return list(matches[0][1])

    def _deduce_probable_move_sequence(self, target_front, max_depth=3):
        """
        Compatibility wrapper used by older callers.

        The live consumer now uses the persistent multi-hypothesis engine above.
        This method remains available for diagnostics/tests and returns the best
        currently explainable path without mutating the tracker state.
        """
        initial_state = self.tracker.state.copy()
        matches = self._search_front_matches(initial_state, target_front, max_depth)

        if not matches:
            self.last_deduction_debug = {
                "top_candidates": [],
                "chosen": "UNKNOWN",
                "recovery_mode": True,
                "status": f"NO PATH WITHIN DEPTH {max_depth}",
                "hypothesis_count": len(self.hypotheses),
                "commit_ready": False,
                "commit_reason": "NO MATCH",
                "first_move_groups": {},
            }
            return "UNKNOWN"

        candidate_debug = [(m[0], m[1]) for m in matches[:6]]
        chosen = list(matches[0][1])
        self.last_deduction_debug = {
            "top_candidates": candidate_debug[:3],
            "chosen": chosen,
            "recovery_mode": len(chosen) > 1,
            "status": "NON-COMMITTING DIAGNOSTIC DEDUCTION",
            "hypothesis_count": len(self.hypotheses),
            "commit_ready": False,
            "commit_reason": "COMPATIBILITY WRAPPER",
            "first_move_groups": {},
        }
        return chosen


# --- CAMERA INITIALIZATION ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# --- FORCE HIGH FPS MODE ON IPHONE / CONTINUITY CAMERA ---
# Lower resolution to unblock high-rate USB transfer mode
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
# Request 120 FPS capture mode
cap.set(cv2.CAP_PROP_FPS, 120.0)
    
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📹 [CAMERA CONFIG]: Initialized at {actual_w}x{actual_h} @ {actual_fps} FPS")

weights_list = glob.glob("runs/segment/rubiks_seg_model/weights/best.pt")
MODEL_PATH = weights_list[0] if weights_list else "yolov8n-seg.pt"
model = YOLO(MODEL_PATH)

COLOR_BGR_MAP = {
    "Red": (0, 0, 255),
    "Orange": (0, 165, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "White": (255, 255, 255),
    "Unknown": (50, 50, 50),
}

DISPLAY_MOVE_MAP = {
    "R": "L",   "R'": "L'",  "R2": "L2",
    "L": "R",   "L'": "R'",  "L2": "R2",
    "F": "F'",  "F'": "F",   "F2": "F2",
    "B": "B",   "B'": "B'",  "B2": "B2",
    "U": "U",   "U'": "U'",  "U2": "U2",
    "D": "D",   "D'": "D'",  "D2": "D2",
}

def remap_kociemba_for_display(raw_solution_str):
    if not raw_solution_str or raw_solution_str == "SOLVED":
        return raw_solution_str
    moves = raw_solution_str.split()
    return " ".join([DISPLAY_MOVE_MAP.get(m, m) for m in moves])


def flatten_move_sequence(deduced_sequence):
    if not deduced_sequence or deduced_sequence == "UNKNOWN":
        return []

    raw_moves = []
    if isinstance(deduced_sequence, str):
        raw_moves = deduced_sequence.strip().split()
    elif isinstance(deduced_sequence, list):
        for item in deduced_sequence:
            if isinstance(item, str):
                raw_moves.extend(item.strip().split())
            else:
                raw_moves.append(str(item))

    if not raw_moves:
        return []

    compressed_moves = []
    for move in raw_moves:
        if compressed_moves:
            prev = compressed_moves[-1]
            if move == prev and not move.endswith("2"):
                base_face = move[0]
                compressed_moves.pop()
                compressed_moves.append(f"{base_face}2")
                continue

        compressed_moves.append(move)

    return compressed_moves


class PermutationCubeTracker:
    def __init__(self):
        self.state = np.empty(54, dtype="<U10")
        self.previous_state = np.empty(54, dtype="<U10")
        self.permutations = {}
        self._init_permutation_matrix()

        self.last_move_detected = "None"
        self.move_history = []
        self.reset_to_solved()

    def reset_to_solved(self):
        self.state = np.array(
            ["White"] * 9
            + ["Yellow"] * 9
            + ["Green"] * 9
            + ["Blue"] * 9
            + ["Orange"] * 9
            + ["Red"] * 9,
            dtype="<U10",
        )
        self.previous_state = self.state.copy()
        self.last_move_detected = "None"
        self.move_history = []
        print("\n[PERMUTATION ENGINE] State reset to SOLVED.\n")
        self.broadcast_state()

    def _init_permutation_matrix(self):
        I = np.arange(54)

        def rot_cw(idx_list):
            return [
                idx_list[6], idx_list[3], idx_list[0],
                idx_list[7], idx_list[4], idx_list[1],
                idx_list[8], idx_list[5], idx_list[2],
            ]

        # R
        pR = I.copy()
        pR[36:45] = rot_cw(I[36:45])
        pR[[18, 21, 24]] = I[[0, 3, 6]]
        pR[[9, 12, 15]]  = I[[18, 21, 24]]
        pR[[35, 32, 29]] = I[[9, 12, 15]]
        pR[[0, 3, 6]]    = I[[35, 32, 29]]
        self.permutations["R"]  = pR
        self.permutations["R'"] = np.argsort(pR)

        # L
        pL = I.copy()
        pL[45:54] = rot_cw(I[45:54])
        pL[[2, 5, 8]]    = I[[20, 23, 26]]
        pL[[33, 30, 27]] = I[[2, 5, 8]]
        pL[[11, 14, 17]] = I[[33, 30, 27]]
        pL[[20, 23, 26]] = I[[11, 14, 17]]
        self.permutations["L"]  = pL
        self.permutations["L'"] = np.argsort(pL)

        # U
        pU = I.copy()
        pU[0:9] = rot_cw(I[0:9])
        pU[[36, 37, 38]] = I[[18, 19, 20]]
        pU[[27, 28, 29]] = I[[36, 37, 38]]
        pU[[45, 46, 47]] = I[[27, 28, 29]]
        pU[[18, 19, 20]] = I[[45, 46, 47]]
        self.permutations["U"]  = pU
        self.permutations["U'"] = np.argsort(pU)

        # D
        pD = I.copy()
        pD[9:18] = rot_cw(I[9:18])
        pD[[51, 52, 53]] = I[[24, 25, 26]]
        pD[[33, 34, 35]] = I[[51, 52, 53]]
        pD[[42, 43, 44]] = I[[33, 34, 35]]
        pD[[24, 25, 26]] = I[[42, 43, 44]]
        self.permutations["D"]  = pD
        self.permutations["D'"] = np.argsort(pD)

        # F
        pF = I.copy()
        pF[18:27] = rot_cw(I[18:27])
        pF[[45, 48, 51]] = I[[6, 7, 8]]
        pF[[11, 10, 9]]  = I[[45, 48, 51]]
        pF[[44, 41, 38]] = I[[11, 10, 9]]
        pF[[6, 7, 8]]    = I[[44, 41, 38]]
        self.permutations["F'"] = pF
        self.permutations["F"]  = np.argsort(pF)

        # B
        pB = I.copy()
        pB[27:36] = rot_cw(I[27:36])
        pB[[36, 39, 42]] = I[[2, 1, 0]]
        pB[[17, 16, 15]] = I[[36, 39, 42]]
        pB[[53, 50, 47]] = I[[17, 16, 15]]
        pB[[0, 1, 2]]    = I[[47, 50, 53]]
        self.permutations["B"]  = pB
        self.permutations["B'"] = np.argsort(pB)

        # Sanity-check the move tables once. A real quarter-turn must return
        # to identity after four repetitions, and each prime move must be the
        # exact inverse of its corresponding move.
        for base_move in ("R", "L", "U", "D", "F", "B"):
            perm = self.permutations[base_move]
            probe = np.arange(54)
            for _ in range(4):
                probe = probe[perm]
            if not np.array_equal(probe, np.arange(54)):
                print(f"⚠️ [PERMUTATION WARNING] {base_move}^4 is not identity!")

            prime = self.permutations[base_move + "'"]
            if not np.array_equal(perm[prime], np.arange(54)):
                print(f"⚠️ [PERMUTATION WARNING] {base_move} / {base_move}' are not inverses!")

    def apply_move_token(self, state, move):
        """Apply one move token to a state without mutating the authoritative state."""
        if state is None or move is None:
            return None

        move = str(move).strip()
        if not move or move == "NONE":
            return state.copy()

        if move.endswith("2"):
            base_move = move[:-1]
            repeat_count = 2
        else:
            base_move = move
            repeat_count = 1

        if base_move not in self.permutations:
            return None

        result = state.copy()
        perm = self.permutations[base_move]
        for _ in range(repeat_count):
            result = result[perm]
        return result

    def apply_move_sequence(self, state, moves):
        """Apply a complete sequence; return None if any move token is invalid."""
        if state is None:
            return None

        result = state.copy()
        if moves is None:
            return result

        if isinstance(moves, str):
            moves = moves.strip().split()

        for move in moves:
            result = self.apply_move_token(result, move)
            if result is None:
                return None

        return result

    def compressed_history(self, moves):
        """
        Combine only two identical consecutive quarter-turns into a 180-degree
        token. Do not cancel inverse moves and do not reduce longer sequences.

        Examples:
            R R   -> R2
            R' R' -> R2
            F F   -> F2

            R R'  -> R R'
            R' R  -> R' R
            R R R -> R2 R
            R R R R -> R2 R R

        This is used for the displayed/history notation. The authoritative cube
        state is still calculated by applying the resulting tokens in order.
        """
        result = []

        for raw_move in (moves or []):
            if raw_move is None:
                continue

            move = str(raw_move).strip()
            if not move or move == "NONE":
                continue

            if result:
                previous = str(result[-1]).strip()

                # Exactly two identical quarter turns become a 180-degree turn.
                if previous in {move, move.replace("2", "")} and move in {
                    "R", "R'", "L", "L'", "U", "U'", "D", "D'",
                    "F", "F'", "B", "B'"
                }:
                    # R+R and R'+R' both represent the same 180-degree turn.
                    if previous == move:
                        result[-1] = move[0] + "2"
                        continue

            result.append(move)

        return result

    def state_from_history(self, history):
        """Rebuild a state from solved using the exact same move application engine."""
        solved = np.array(
            ["White"] * 9
            + ["Yellow"] * 9
            + ["Green"] * 9
            + ["Blue"] * 9
            + ["Orange"] * 9
            + ["Red"] * 9,
            dtype="<U10",
        )
        return self.apply_move_sequence(solved, history)

    def history_matches_state(self):
        """Hard invariant: authoritative state must equal its move history."""
        rebuilt = self.state_from_history(self.move_history)
        return rebuilt is not None and np.array_equal(rebuilt, self.state)

    def get_3x3_face_dict(self):
        return {
            "White":  self.state[0:9].reshape((3, 3)).tolist(),
            "Yellow": self.state[9:18].reshape((3, 3)).tolist(),
            "Green":  self.state[18:27].reshape((3, 3)).tolist(),
            "Blue":   np.fliplr(self.state[27:36].reshape((3, 3))).tolist(),
            "Orange": np.fliplr(self.state[36:45].reshape((3, 3))).tolist(),
            "Red":    np.fliplr(self.state[45:54].reshape((3, 3))).tolist(),
        }

    def broadcast_state(self):
        socketio.emit('cube_update', {
            'active_center': str(self.state[22]),
            'status': 'Active',
            'move_history': self.move_history,
            'last_move': getattr(self, 'last_move_detected', '-'),
            'faces': self.get_3x3_face_dict()
        })

    def state_to_kociemba_string(self, state_array=None):
        if state_array is None:
            state_array = self.state

        flat_state = list(state_array)
        if len(flat_state) != 54:
            return None

        try:
            center_color_map = {
                flat_state[4]: "U",   
                flat_state[13]: "D",  
                flat_state[22]: "F",  
                flat_state[31]: "B",  
                flat_state[40]: "L",  
                flat_state[49]: "R",  
            }

            if len(set(center_color_map.keys())) < 6:
                return None

            u_face = flat_state[0:9]   
            r_face = flat_state[45:54] 
            f_face = flat_state[18:27] 
            d_face = flat_state[9:18]  
            l_face = flat_state[36:45] 
            b_face = flat_state[27:36] 

            kociemba_order = u_face + r_face + f_face + d_face + l_face + b_face
            k_str = "".join([center_color_map[color] for color in kociemba_order])
            
            counts = {char: k_str.count(char) for char in "URFDLB"}
            if any(count != 9 for count in counts.values()):
                return None

            return k_str
        except (KeyError, IndexError):
            return None

    def trigger_kociemba_solve(self):
        k_str = self.state_to_kociemba_string()
        if not k_str:
            print("\n❌ [KOCIEMBA]: Cannot solve. Invalid facelet configuration.\n")
            return None

        try:
            raw_solution = kociemba.solve(k_str)
            if not raw_solution or not raw_solution.strip():
                print("\n🧩 CUBE IS ALREADY SOLVED!\n")
                return "SOLVED"

            display_solution = remap_kociemba_for_display(raw_solution)

            print("\n" + "=" * 50)
            print(f"🧩 RAW KOCIEMBA SOLUTION:     {raw_solution}")
            print(f"👉 DISPLAY REMAPPED SOLUTION:  {display_solution}")
            print("=" * 50 + "\n")

            socketio.emit("kociemba_solution", {
                "raw_solution": raw_solution,
                "solution": display_solution
            })

            return display_solution
        except Exception as e:
            print(f"\n❌ [KOCIEMBA ERROR]: Unsolvable state configuration ({e})\n")
            return None

    def deduce_move_from_grids(self, prev_grid, curr_grid):
        """
        Compares two 3x3 grids and identifies the move or state change.
        """
        if not prev_grid or not curr_grid:
            return None

        prev_center = prev_grid[1][1]
        curr_center = curr_grid[1][1]

        # Center changed -> Cube face shift / rotation
        if prev_center != curr_center:
            return f"Face Shift ({prev_center} -> {curr_center})"

        # Count sticker color differences on current face
        diff_count = sum(
            1 for r in range(3) for c in range(3) 
            if prev_grid[r][c] != curr_grid[r][c]
        )

        if diff_count == 0:
            return None

        # Determine turn type based on facelet changes
        # Face turns alter edge and corner positions while center stays fixed
        if diff_count in [6, 8]:
            return f"{prev_center} Turn (90°)"
        elif diff_count == 4:
            return f"{prev_center} Layer Turn"
        else:
            return f"{prev_center} Face Move ({diff_count} facelets changed)"


# Global Cube Tracker Reference
cube_tracker = PermutationCubeTracker()
if not cube_tracker.history_matches_state():
    raise RuntimeError("Initial cube state/history consistency check failed.")
async_pipeline = SequenceBufferedTracker(cube_tracker)


# --- MULTI-HYPOTHESIS BEAM SEARCH SYSTEM ---
class CubeHypothesis:
    def __init__(self, cube_state, log_likelihood=0.0, move_history=None):
        self.state = copy.deepcopy(cube_state)
        self.log_likelihood = log_likelihood
        self.move_history = move_history if move_history is not None else []


class BeamStateTracker:
    def __init__(self, initial_solved_state, cube_permutations, beam_width=5):
        self.beam_width = beam_width
        self.permutations = cube_permutations
        self.hypotheses = [CubeHypothesis(initial_solved_state, log_likelihood=0.0)]
        self.top_hypothesis_move = "NONE"
        self.hypothesis_confidence = 0.0
        
        # Debounce and state lock mechanism
        self.pending_move = "NONE"
        self.pending_count = 0
        self.STABILITY_THRESHOLD = 5  # Must hold candidate move stable for 5 frames (~40ms at 120fps)
        self.last_applied_move = "NONE"

    def process_observed_grid(self, observed_grid):
        new_candidates = []
        possible_moves = ["F", "F'", "U", "U'", "R", "R'", 
                          "L", "L'", "B", "B'", "D", "D'", "NONE"]

        for hyp in self.hypotheses:
            for move in possible_moves:
                simulated_state = self._apply_virtual_move(hyp.state, move)
                obs_score = self._compute_observation_likelihood(simulated_state, observed_grid)
                move_penalty = 0.0 if move == "NONE" else -0.5
                total_score = hyp.log_likelihood + obs_score + move_penalty
                
                new_history = hyp.move_history + ([move] if move != "NONE" else [])
                new_candidates.append(CubeHypothesis(simulated_state, total_score, new_history))

        new_candidates.sort(key=lambda h: h.log_likelihood, reverse=True)
        self.hypotheses = new_candidates[:self.beam_width]

        # Normalize likelihoods
        max_score = self.hypotheses[0].log_likelihood
        for h in self.hypotheses:
            h.log_likelihood -= max_score

        best_hyp = self.hypotheses[0]
        candidate_move = best_hyp.move_history[-1] if best_hyp.move_history else "NONE"
        
        self.top_hypothesis_move = candidate_move
        self.hypothesis_confidence = round(float(np.exp(best_hyp.log_likelihood)), 2)

        # DEBOUNCE LOGIC: Only confirm a move if it persists across consecutive frames
        if candidate_move != "NONE" and candidate_move == self.pending_move:
            self.pending_count += 1
        else:
            self.pending_move = candidate_move
            self.pending_count = 1

        if self.pending_count >= self.STABILITY_THRESHOLD and self.pending_move != self.last_applied_move:
            confirmed_move = self.pending_move
            self.last_applied_move = confirmed_move
            
            # Commit move permanently into state engine hypotheses
            for h in self.hypotheses:
                h.state = self._apply_virtual_move(h.state, confirmed_move)
                h.move_history = []  # Reset step history after committing
                
            return confirmed_move

        return "HOLD"

    def get_best_state(self):
        return self.hypotheses[0].state

    def _compute_observation_likelihood(self, cube_state, observed_grid):
        center_color = observed_grid[1][1]
        
        center_indices = {
            "White": 4, "Yellow": 13, "Green": 22, 
            "Blue": 31, "Orange": 40, "Red": 49
        }
        
        if center_color not in center_indices:
            return -1.0
            
        center_idx = center_indices[center_color]
        face_start = (center_idx // 9) * 9
        expected_face = cube_state[face_start:face_start + 9].reshape((3, 3))
        
        matches = 0
        for r in range(3):
            for c in range(3):
                if observed_grid[r][c] == expected_face[r][c]:
                    matches += 1
                    
        return (matches / 9.0) * 2.0 - 1.0 

    def _apply_virtual_move(self, state, move):
        if move == "NONE" or move not in self.permutations:
            return state
        perm = self.permutations[move]
        return state[perm]

# --- GRID LOCK AND CALIBRATOR ---
class GridStabilityLock:
    def __init__(self, required_stable_frames=4):
        self.required_frames = required_stable_frames
        self.last_grid = None
        self.stable_count = 0

    def process(self, grid):
        if grid is None:
            self.stable_count = 0
            self.last_grid = None
            return False

        if self.last_grid is not None and grid == self.last_grid:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_grid = grid

        return self.stable_count == self.required_frames

grid_lock = GridStabilityLock(required_stable_frames=4)
SCAN_ORDER = ["White", "Yellow", "Green", "Blue", "Red", "Orange"]

class GlobalKMeansCalibrator:
    def __init__(self):
        self.scanned_faces = {}
        self.current_scan_idx = 0
        self.is_calibrated = False
        self.cluster_centers = {}

    def get_target_color(self):
        if self.current_scan_idx < len(SCAN_ORDER):
            return SCAN_ORDER[self.current_scan_idx]
        return None

    def capture_face(self, grid_lab_patches):
        target = self.get_target_color()
        if target is None:
            return False

        self.scanned_faces[target] = np.array(grid_lab_patches)
        self.current_scan_idx += 1

        if self.current_scan_idx >= len(SCAN_ORDER):
            self._compute_kmeans_centroids()
            self.is_calibrated = True
            return True

        return False

    def _compute_kmeans_centroids(self):
        all_stickers = np.vstack([self.scanned_faces[color] for color in SCAN_ORDER])

        kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
        labels = kmeans.fit_predict(all_stickers)
        centers = kmeans.cluster_centers_

        center_indices = [i * 9 + 4 for i in range(6)]
        
        self.cluster_centers = {}
        for i, color_name in enumerate(SCAN_ORDER):
            center_idx = center_indices[i]
            cluster_id = labels[center_idx]
            self.cluster_centers[color_name] = centers[cluster_id]

        print("\n[SUCCESS] Dynamic K-Means Centroids Calibrated!")

    def classify_patch(self, mean_lab):
        if not self.is_calibrated:
            return "Unknown"

        best_color = "Unknown"
        min_dist = float('inf')

        for color_name, center_lab in self.cluster_centers.items():
            dL = (mean_lab[0] - center_lab[0]) * 0.3
            da = (mean_lab[1] - center_lab[1]) * 1.3
            db = (mean_lab[2] - center_lab[2]) * 1.3
            
            dist = np.sqrt(dL**2 + da**2 + db**2)
            if dist < min_dist:
                min_dist = dist
                best_color = color_name

        return best_color

kmeans_calibrator = GlobalKMeansCalibrator()


def validate_face_grid(grid):
    """Reject structurally invalid classifier output before temporal deduction."""
    if grid is None or not isinstance(grid, (list, tuple)) or len(grid) != 3:
        return False

    if any(
        not isinstance(row, (list, tuple)) or len(row) != 3
        for row in grid
    ):
        return False

    flat = [str(color) for row in grid for color in row]
    valid_colors = {
        "Red", "Orange", "Yellow", "Green", "Blue", "White"
    }

    if any(color not in valid_colors for color in flat):
        return False

    if grid[1][1] not in valid_colors:
        return False

    # A single visible face contains only nine stickers, so a classifier
    # cannot legitimately return more than nine of any one color.
    counts = {color: flat.count(color) for color in valid_colors}
    if any(count > 9 for count in counts.values()):
        return False

    return True


def extract_facelet_grid_with_debug(frame, rect):
    dst_pts = np.array([[0, 0], [149, 0], [149, 149], [0, 149]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    flat_face = cv2.warpPerspective(frame, M, (150, 150))
    lab_face = cv2.cvtColor(flat_face, cv2.COLOR_BGR2LAB)
    
    grid = [["Unknown" for _ in range(3)] for _ in range(3)]
    raw_lab_patches = []
    cell_size = 50

    for r in range(3):
        for c in range(3):
            patch = lab_face[
                r * cell_size + 10 : (r + 1) * cell_size - 10,
                c * cell_size + 10 : (c + 1) * cell_size - 10
            ]
            
            l_chan = patch[:, :, 0]
            valid = (l_chan > 25) & (l_chan < 245)
            if np.sum(valid) > 0:
                cell_mean = np.mean(patch[valid], axis=0)
            else:
                cell_mean = np.mean(patch, axis=(0, 1))

            raw_lab_patches.append(cell_mean)

            if kmeans_calibrator.is_calibrated:
                grid[r][c] = kmeans_calibrator.classify_patch(cell_mean)

    return grid, flat_face, raw_lab_patches

# Place this helper near your UI rendering functions
def draw_debug_inspector_panel(
    frame, 
    grid, 
    raw_lab_patches, 
    is_paused, 
    frame_idx, 
    total_frames, 
    quad_corners=None,
    deduction_info=None
):
    """
    Renders diagnostic panel and overlays live 3x3 color grid on the cube face in the slow-mo window.
    """
    h_f, w_f, _ = frame.shape
    panel_w = 340
    panel_x1 = w_f - panel_w

    # BGR display map for center labels
    COLOR_MAP_BGR = {
        'U': (255, 255, 255), 'D': (0, 255, 255),
        'F': (0, 255, 0),   'B': (0, 0, 255),
        'L': (0, 165, 255), 'R': (255, 0, 0)
    }

    # 1. OVERLAY 3X3 COLOR MESH ON DETECTED CUBE FACE
    if quad_corners is not None and grid is not None:
        pts = quad_corners.astype(np.float32)
        # Perspective transform to project a 3x3 grid over the quad bounding box
        src_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        matrix = cv2.getPerspectiveTransform(src_pts, pts)

        for r in range(3):
            for c in range(3):
                cell_src = np.float32([[[c * 100, r * 100], 
                                        [(c + 1) * 100, r * 100], 
                                        [(c + 1) * 100, (r + 1) * 100], 
                                        [c * 100, (r + 1) * 100]]])
                cell_dst = cv2.perspectiveTransform(cell_src, matrix).astype(np.int32)
                
                facelet_color_char = str(grid[r][c])[0].upper() if grid[r][c] else '?'
                bgr = COLOR_MAP_BGR.get(facelet_color_char, (128, 128, 128))

                # Highlight facelet bounding box
                cv2.polylines(frame, [cell_dst], True, bgr, 2)
                
                # Draw center marker dot for each facelet patch
                center_pt = np.mean(cell_dst[0], axis=0).astype(int)
                cv2.circle(frame, (center_pt[0], center_pt[1]), 4, bgr, -1)
                cv2.circle(frame, (center_pt[0], center_pt[1]), 5, (0, 0, 0), 1)

    # 2. SEMI-TRANSPARENT DEBUG DRAWER
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, 0), (w_f, h_f), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (panel_x1, 0), (panel_x1, h_f), (0, 255, 204), 2)

    # Header
    status_text = "PAUSED" if is_paused else "PLAYING"
    status_color = (0, 165, 255) if is_paused else (0, 255, 0)
    cv2.putText(frame, "REPLAY DEBUG INSPECTOR", (panel_x1 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 204), 2)
    cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames} | {status_text}", (panel_x1 + 15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)
    cv2.putText(frame, "[p] Pause | [,] Prev | [.] Next | [o] Close", (panel_x1 + 15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.line(frame, (panel_x1 + 10, 78), (w_f - 10, 78), (60, 60, 60), 1)

    # 3. 3X3 COLOR GRID MATRIX TELEMETRY
    cv2.putText(frame, "DETECTED FACE COLORS:", (panel_x1 + 15, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if grid is not None:
        for r in range(3):
            for c in range(3):
                color_name = str(grid[r][c])
                x_pos = panel_x1 + 15 + (c * 100)
                y_pos = 110 + (r * 30)

                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 92, y_pos + 24), (30, 30, 30), -1)
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 92, y_pos + 24), (70, 70, 70), 1)
                cv2.putText(frame, f"{r},{c}: {color_name[:5]}", (x_pos + 5, y_pos + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "NO GRID DETECTED IN FRAME", (panel_x1 + 15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    cv2.line(frame, (panel_x1 + 10, 210), (w_f - 10, 210), (60, 60, 60), 1)

    # 4. MOVE DEDUCTION TELEMETRY
    cv2.putText(frame, "DEDUCTION ENGINE TELEMETRY:", (panel_x1 + 15, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 204), 1)
    
    hypothesized_move = deduction_info.get('hypothesized_move', 'NONE') if deduction_info else 'N/A'
    decided_move = deduction_info.get('decided_move', 'HOLD / NO MOVE') if deduction_info else 'N/A'
    confidence = deduction_info.get('confidence', 0.0) if deduction_info else 0.0
    status = deduction_info.get('status', 'NO DATA') if deduction_info else 'NO DATA'
    recovery_mode = deduction_info.get('recovery_mode', False) if deduction_info else False
    top_candidates = deduction_info.get('top_candidates', []) if deduction_info else []

    cv2.putText(frame, f"MODE: {'RECOVERY' if recovery_mode else 'NORMAL'}", (panel_x1 + 15, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 204), 1)
    cv2.putText(frame, f"STATUS: {str(status)[:33]}", (panel_x1 + 15, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    cv2.putText(frame, f"ENGINE CHOICE: {str(hypothesized_move)}", (panel_x1 + 15, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    cv2.putText(frame, f"LIVE COMMIT: {str(decided_move)}", (panel_x1 + 15, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

    cv2.putText(frame, "TOP CANDIDATE PATHS:", (panel_x1 + 15, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    if top_candidates:
        y = 367
        for rank, candidate in enumerate(top_candidates[:3], start=1):
            try:
                score, sequence = candidate
                seq_text = " ".join(sequence) if isinstance(sequence, (list, tuple)) else str(sequence)
                line = f"{rank}. {seq_text}  cost={float(score):.2f}"
            except Exception:
                line = f"{rank}. {str(candidate)}"
            cv2.putText(frame, line[:47], (panel_x1 + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255) if rank == 1 else (220, 220, 220), 1)
            y += 19
    else:
        cv2.putText(frame, "No alternate paths found", (panel_x1 + 15, 367), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)

    cv2.rectangle(frame, (panel_x1 + 15, 427), (w_f - 15, 463), (20, 50, 20), -1)
    cv2.rectangle(frame, (panel_x1 + 15, 427), (w_f - 15, 463), (0, 255, 0), 1)
    cv2.putText(frame, f"DISPLAYED MOVE: {decided_move}", (panel_x1 + 25, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

    hypothesis_count = deduction_info.get('hypothesis_count', 0) if deduction_info else 0
    commit_ready = deduction_info.get('commit_ready', False) if deduction_info else False
    commit_reason = deduction_info.get('commit_reason', '') if deduction_info else ''
    cv2.putText(
        frame,
        f"HYPOTHESES: {hypothesis_count} | {'COMMIT READY' if commit_ready else 'WAITING'}",
        (panel_x1 + 15, 486),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 0) if commit_ready else (0, 200, 255),
        1,
    )
    cv2.putText(
        frame,
        str(commit_reason)[:45],
        (panel_x1 + 15, 505),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (170, 170, 170),
        1,
    )

    best_conf = deduction_info.get('best_confidence', 0.0) if deduction_info else 0.0
    second_conf = deduction_info.get('second_confidence', 0.0) if deduction_info else 0.0
    ambiguity_obs = deduction_info.get('ambiguity_observations', 0) if deduction_info else 0
    pending_moves = deduction_info.get('max_pending_moves', 0) if deduction_info else 0
    cv2.putText(
        frame,
        f"CONF: {best_conf * 100.0:.1f}% vs {second_conf * 100.0:.1f}% | WINDOW: {ambiguity_obs}/3 | PENDING: {pending_moves}",
        (panel_x1 + 15, 523),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (0, 255, 255),
        1,
    )

    # 5. PIPELINE PERFORMANCE TELEMETRY
    q_now = deduction_info.get('queue_now', 0) if deduction_info else 0
    q_drop = deduction_info.get('queue_dropped', 0) if deduction_info else 0
    q_enq = deduction_info.get('queue_enqueued', 0) if deduction_info else 0
    q_delay = deduction_info.get('queue_delay_ms', 0.0) if deduction_info else 0.0
    q_delay_max = deduction_info.get('queue_delay_max_ms', 0.0) if deduction_info else 0.0
    ded_ms = deduction_info.get('deduction_ms', 0.0) if deduction_info else 0.0
    ded_max = deduction_info.get('deduction_max_ms', 0.0) if deduction_info else 0.0
    loop_ms = deduction_info.get('live_loop_ms', 0.0) if deduction_info else 0.0
    loop_max = deduction_info.get('live_loop_max_ms', 0.0) if deduction_info else 0.0

    cv2.putText(
        frame,
        f"QUEUE: {q_now} | DROP: {q_drop} | ENQ: {q_enq}",
        (panel_x1 + 15, 532),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Q DELAY: {q_delay:.1f}ms (MAX {q_delay_max:.1f})",
        (panel_x1 + 15, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        f"DEDUCTION: {ded_ms:.1f}ms (MAX {ded_max:.1f})",
        (panel_x1 + 15, 568),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        f"LIVE LOOP: {loop_ms:.1f}ms (MAX {loop_max:.1f})",
        (panel_x1 + 15, 586),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
    )

def draw_permutation_ui(frame, current_grid, flat_face):
    h, w, _ = frame.shape
    start_x, start_y = 20, 120
    box_s = 18

    cv2.putText(
        frame,
        "PREDICTIVE ENGINE INSPECTOR (54 Facelets)",
        (start_x, start_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

    face_offsets = {
        "White (U)": (start_x + 4 * box_s, start_y, 0),
        "Orange (L)": (start_x, start_y + 3 * box_s, 36),
        "Green (F)": (start_x + 4 * box_s, start_y + 3 * box_s, 18),
        "Red (R)": (start_x + 8 * box_s, start_y + 3 * box_s, 45),
        "Blue (B)": (start_x + 12 * box_s, start_y + 3 * box_s, 27),
        "Yellow (D)": (start_x + 4 * box_s, start_y + 6 * box_s, 9),
    }

    for name, (fx, fy, base_idx) in face_offsets.items():
        for r in range(3):
            for c in range(3):
                idx = base_idx + r * 3 + c
                color_str = cube_tracker.state[idx]
                bgr = COLOR_BGR_MAP.get(color_str, (50, 50, 50))

                x1 = fx + c * box_s
                y1 = fy + r * box_s
                cv2.rectangle(
                    frame, (x1, y1), (x1 + box_s - 2, y1 + box_s - 2), bgr, -1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x1 + box_s - 2, y1 + box_s - 2),
                    (0, 0, 0),
                    1,
                )

    log_x, log_y = 20, start_y + 10 * box_s + 10
    panel_w, panel_h = 290, 130
    cv2.rectangle(
        frame,
        (log_x, log_y),
        (log_x + panel_w, log_y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.rectangle(
        frame,
        (log_x, log_y),
        (log_x + panel_w, log_y + panel_h),
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"ESTIMATED MOVES ({len(cube_tracker.move_history)})",
        (log_x + 10, log_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

    recent_moves = cube_tracker.move_history[-5:]
    if recent_moves:
        history_str = " -> ".join(recent_moves)
        if len(history_str) > 30:
            history_str = "..." + history_str[-27:]
        cv2.putText(
            frame,
            history_str,
            (log_x + 10, log_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )


# --- WEB VISUALIZATION ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>3D Rubik's Cube Viewer</title>
    <style>
        body { margin: 0; background: #111; color: #fff; font-family: sans-serif; overflow: hidden; }
        #hud { 
            position: absolute; 
            top: 10px; 
            left: 10px; 
            background: rgba(0,0,0,0.85); 
            padding: 15px; 
            border-radius: 8px; 
            z-index: 10; 
            width: 260px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }
        #hud h2 { margin-top: 0; font-size: 1.2rem; color: #00ff00; }
        #hud p { margin: 6px 0; font-size: 0.9rem; }
        .history-box { 
            margin-top: 12px; 
            background: #1a1a1a; 
            border: 1px solid #333; 
            padding: 10px; 
            border-radius: 6px; 
            max-height: 180px; 
            overflow-y: auto; 
        }
        .history-title { font-weight: bold; font-size: 0.85rem; color: #00ffff; margin-bottom: 6px; }
        .move-tag { 
            display: inline-block; 
            background: #222; 
            border: 1px solid #444; 
            color: #fff; 
            padding: 3px 6px; 
            margin: 2px; 
            border-radius: 4px; 
            font-family: monospace; 
            font-size: 0.85rem;
        }
        .move-tag.latest { background: #008800; border-color: #00ff00; font-weight: bold; }
        #canvas { width: 100vw; height: 100vh; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <h2>Predictive State Engine</h2>
        <p>Active Center: <b id="active">-</b></p>
        <p>Status: <span id="status">Connecting...</span></p>

        <div class="history-box">
            <div class="history-title">ESTIMATED MOVES (<span id="move-count">0</span>)</div>
            <div id="history-list"><i>No moves recorded</i></div>
        </div>
    </div>
    <div id="canvas"></div>
    <script>
        const socket = io();
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(5, 4, 6); 
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('canvas').appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const COLOR_MAP = {
            'White': 0xffffff, 'Yellow': 0xffff00,
            'Green': 0x00ff00, 'Blue': 0x0000ff,
            'Orange': 0xffa500, 'Red': 0xff0000,
            'Unknown': 0x222222
        };

        let cubies = [];

        function createCube() {
            const size = 0.95;
            const geometry = new THREE.BoxGeometry(size, size, size);

            for (let x = -1; x <= 1; x++) {
                for (let y = -1; y <= 1; y++) {
                    for (let z = -1; z <= 1; z++) {
                        if (x === 0 && y === 0 && z === 0) continue;

                        const materials = Array(6).fill().map(() => new THREE.MeshBasicMaterial({ color: 0x111111 }));
                        const mesh = new THREE.Mesh(geometry, materials);
                        mesh.position.set(x, y, z);
                        scene.add(mesh);
                        cubies.push({ mesh, x, y, z });
                    }
                }
            }
        }
        createCube();

        function update3DCube(faces) {
            cubies.forEach(cubie => {
                const { mesh, x, y, z } = cubie;
                const mats = mesh.material;

                const col = x + 1;
                const rowY = 1 - y;
                const depthZ = 1 - z;

                if (x === 1) mats[0].color.setHex(COLOR_MAP[faces.Red[rowY][depthZ]] || 0x222222);
                if (x === -1) mats[1].color.setHex(COLOR_MAP[faces.Orange[rowY][z + 1]] || 0x222222);
                if (y === 1) mats[2].color.setHex(COLOR_MAP[faces.White[depthZ][col]] || 0x222222);
                if (y === -1) mats[3].color.setHex(COLOR_MAP[faces.Yellow[1 - depthZ][col]] || 0x222222);
                if (z === 1) mats[4].color.setHex(COLOR_MAP[faces.Green[rowY][col]] || 0x222222);
                if (z === -1) mats[5].color.setHex(COLOR_MAP[faces.Blue[rowY][2 - col]] || 0x222222);
            });
        }

        function updateHistory(history) {
            const container = document.getElementById('history-list');
            document.getElementById('move-count').innerText = history ? history.length : 0;
            
            if (!history || history.length === 0) {
                container.innerHTML = '<i>No moves recorded</i>';
                return;
            }
            
            container.innerHTML = history.map((m, idx) => {
                const isLatest = idx === history.length - 1;
                return `<span class="move-tag ${isLatest ? 'latest' : ''}">${idx + 1}. ${m}</span>`;
            }).join(' ');

            container.scrollTop = container.scrollHeight;
        }

        socket.on('cube_update', (data) => {
            document.getElementById('active').innerText = data.active_center || 'None';
            document.getElementById('status').innerText = data.status || 'Active';
            
            if (data.faces) update3DCube(data.faces);
            updateHistory(data.move_history);
        });

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        socket.on('kociemba_solution', (data) => {
            alert("Kociemba Solution: " + data.solution);
            console.log("Kociemba Solution:", data.solution);
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on("connect")
def handle_connect():
    cube_tracker.broadcast_state()


def run_server():
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, use_reloader=False)

import os
import tempfile
import queue

class SlowMoBufferProcessor:
    def __init__(self, cube_tracker, fps=120.0):
        self.cube_tracker = cube_tracker
        self.fps = fps
        self.is_recording = False
        self.recorded_frames = []
        self.lock = threading.Lock()
        self.temp_file_path = None
        self.replay_queue = queue.Queue()

    def start_recording(self):
        with self.lock:
            self.recorded_frames = []
            self.is_recording = True
            print("\n🔴 [SLOW-MO]: Recording started (Press SPACE to stop)...")

    def add_frame(self, frame):
        with self.lock:
            if self.is_recording:
                self.recorded_frames.append(frame.copy())

    def stop_and_process(self, model, extract_grid_fn):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            frames_to_process = list(self.recorded_frames)
            self.recorded_frames = []

        total_frames = len(frames_to_process)
        print(f"\n⏹️ [SLOW-MO]: Processing {total_frames} frames through Permutation Engine...")

        thread = threading.Thread(
            target=self._analyze_frames_async, 
            args=(frames_to_process, model, extract_grid_fn), 
            daemon=True
        )
        thread.start()

    def _analyze_frames_async(self, frames, model, extract_grid_fn):
        if not frames:
            return

        # Temporary video save
        h, w, _ = frames[0].shape
        temp_dir = tempfile.gettempdir()
        self.temp_file_path = os.path.join(temp_dir, "rubiks_slowmo_debug.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.temp_file_path, fourcc, self.fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

        total_frames = len(frames)

        # 1. Snapshot live history length to isolate new moves from the video run
        initial_move_count = len(getattr(self.cube_tracker, 'move_history', []))

        # 2. Process video frame-by-frame through your system's extraction & pipeline
        for idx, frame in enumerate(frames):
            display_replay = frame.copy()
            h_f, w_f, _ = display_replay.shape

            results = model(display_replay, conf=0.40, verbose=False)[0]

            if results.masks is not None and len(results.masks) > 0:
                masks = results.masks.data.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                best_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])

                mask_resized = cv2.resize(masks[best_idx], (w_f, h_f))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                    pts = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(cnt))

                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)
                    rect = np.zeros((4, 2), dtype="float32")
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]

                    cv2.polylines(display_replay, [rect.astype(np.int32)], True, (0, 255, 0), 2)

                    # Coordinate ordering for raw frame extraction
                    rect_raw = rect.copy()
                    rect_raw[:, 0] = w_f - rect_raw[:, 0]
                    s_raw = rect_raw.sum(axis=1)
                    diff_raw = np.diff(rect_raw, axis=1)
                    rect_raw_ordered = np.zeros((4, 2), dtype="float32")
                    rect_raw_ordered[0] = rect_raw[np.argmin(s_raw)]
                    rect_raw_ordered[2] = rect_raw[np.argmax(s_raw)]
                    rect_raw_ordered[1] = rect_raw[np.argmin(diff_raw)]
                    rect_raw_ordered[3] = rect_raw[np.argmax(diff_raw)]

                    current_grid, _, _ = extract_grid_fn(frame, rect_raw_ordered)

                    # PASS GRID DIRECTLY TO YOUR PERMUTATION SYSTEM
                    if current_grid and hasattr(self.cube_tracker, 'update'):
                        self.cube_tracker.update(current_grid)
                    elif current_grid and hasattr(self.cube_tracker, 'process_grid'):
                        self.cube_tracker.process_grid(current_grid)

            # --- HUD OVERLAY ---
            cv2.rectangle(display_replay, (0, 0), (w_f, 75), (0, 0, 0), -1)
            progress_pct = int(((idx + 1) / total_frames) * 100)

            cv2.putText(
                display_replay,
                f"SLOW-MO ENGINE | FRAME {idx + 1}/{total_frames} ({progress_pct}%)",
                (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Retrieve current move list from your tracker
            current_history = getattr(self.cube_tracker, 'move_history', [])
            video_moves = current_history[initial_move_count:]
            moves_str = " ".join(video_moves) if video_moves else "..."

            cv2.putText(
                display_replay,
                f"MOVES: {moves_str}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            # Queue the original capture, without analyzer overlays or transforms.
            # The replay window applies the same mirror transform as the live UI exactly once.
            self.replay_queue.put(frame.copy())

        self.replay_queue.put(None)

        # 3. Print the moves logged by your Permutation Engine
        final_history = getattr(self.cube_tracker, 'move_history', [])
        video_parsed_moves = final_history[initial_move_count:]

        print("\n" + "=" * 50)
        print("     PERMUTATION SYSTEM MOVES (SLOW-MO RUN)     ")
        print("=" * 50)
        if video_parsed_moves:
            print(f"  👉 {' '.join(video_parsed_moves)}")
        else:
            print("  ⚠️ No notation moves registered by tracker during replay.")
        print("=" * 50 + "\n")

        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)


def draw_replay_move_tab(frame, replay_move_events, replay_idx, scroll_offset=0):
    """Draw a vertically scrollable move-history tab for replay playback."""
    h_f, w_f, _ = frame.shape

    tab_x = 15
    tab_y = 105
    tab_w = 220
    tab_bottom = h_f - 45
    tab_h = max(120, tab_bottom - tab_y)

    row_h = 27
    header_h = 42
    footer_h = 24
    list_y = tab_y + header_h
    list_bottom = tab_bottom - footer_h
    list_h = max(row_h, list_bottom - list_y)
    visible_rows = max(1, list_h // row_h)

    events = [e for e in replay_move_events if e.get("frame", -1) <= replay_idx]
    total_events = len(events)
    max_offset = max(0, total_events - visible_rows)
    scroll_offset = max(0, min(int(scroll_offset), max_offset))

    # Semi-transparent panel.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (tab_x, tab_y),
        (tab_x + tab_w, tab_bottom),
        (10, 10, 10),
        -1,
    )
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.rectangle(
        frame,
        (tab_x, tab_y),
        (tab_x + tab_w, tab_bottom),
        (0, 255, 204),
        1,
    )

    cv2.putText(
        frame,
        "MOVE HISTORY",
        (tab_x + 12, tab_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 204),
        1,
    )
    cv2.putText(
        frame,
        f"{total_events} moves",
        (tab_x + tab_w - 78, tab_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (170, 170, 170),
        1,
    )
    cv2.line(
        frame,
        (tab_x + 8, list_y - 5),
        (tab_x + tab_w - 8, list_y - 5),
        (55, 55, 55),
        1,
    )

    # Clip-ish list area by drawing only inside the calculated rows.
    for row in range(visible_rows):
        event_index = scroll_offset + row
        if event_index >= total_events:
            break

        event = events[event_index]
        move = str(event.get("move", "?"))
        frame_no = event.get("frame", "?")
        y1 = list_y + row * row_h
        y2 = y1 + row_h - 3

        is_current = int(frame_no) == int(replay_idx) if str(frame_no).isdigit() else False

        if is_current:
            cv2.rectangle(
                frame,
                (tab_x + 7, y1),
                (tab_x + tab_w - 18, y2),
                (20, 65, 35),
                -1,
            )
            cv2.rectangle(
                frame,
                (tab_x + 7, y1),
                (tab_x + tab_w - 18, y2),
                (0, 255, 0),
                1,
            )

        cv2.putText(
            frame,
            f"{event_index + 1:03d}",
            (tab_x + 13, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (150, 150, 150),
            1,
        )
        cv2.putText(
            frame,
            move,
            (tab_x + 56, y1 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0) if is_current else (255, 255, 255),
            1 if not is_current else 2,
        )
        cv2.putText(
            frame,
            f"F{frame_no}",
            (tab_x + 112, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (180, 180, 180),
            1,
        )

    # Scroll bar.
    scrollbar_x = tab_x + tab_w - 12
    track_top = list_y
    track_bottom = list_bottom
    cv2.rectangle(
        frame,
        (scrollbar_x, track_top),
        (scrollbar_x + 5, track_bottom),
        (45, 45, 45),
        -1,
    )

    if total_events > visible_rows:
        thumb_h = max(22, int((visible_rows / total_events) * (track_bottom - track_top)))
        travel = max(1, (track_bottom - track_top) - thumb_h)
        thumb_y = track_top + int((scroll_offset / max_offset) * travel)
        cv2.rectangle(
            frame,
            (scrollbar_x, thumb_y),
            (scrollbar_x + 5, thumb_y + thumb_h),
            (0, 255, 204),
            -1,
        )

    cv2.line(
        frame,
        (tab_x + 8, list_bottom + 2),
        (tab_x + tab_w - 8, list_bottom + 2),
        (55, 55, 55),
        1,
    )
    cv2.putText(
        frame,
        "Mouse wheel: scroll",
        (tab_x + 10, tab_bottom - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (170, 170, 170),
        1,
    )

    return scroll_offset, max_offset


def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("\n[3D VIEWER READY] Open http://localhost:5001 in your browser.")
    print("⌨️ Controls: [SPACE] = Record/Process Slow-Mo | [c] = Scan Face | [r] = Reset | [q] = Quit\n")
    print("🎬 Replay Controls: [p] = Pause/Play Video | [,] = Prev Frame | [.] = Next Frame | [d] = Toggle Replay Debug | [o] = Close Replay\n")

    slowmo_processor = SlowMoBufferProcessor(cube_tracker, fps=60.0)

    last_valid_grid = None
    last_valid_flat_face = None

    beam_tracker = BeamStateTracker(
        cube_tracker.state, cube_tracker.permutations, beam_width=5
    )

    def extract_fn(raw_frame, rect):
        return extract_facelet_grid_with_debug(raw_frame, rect)

    # Replay Video State Management
    replay_frames = []
    move_log = {}  # Map: frame_index -> move_string (e.g. {14: 'U', 32: "R'"})
    replay_idx = 0
    replay_paused = False
    show_replay_debug = True
    show_debug_ui = True

    # --- Replay move debugging ---
    replay_move_events = []
    recording_frame_idx = 0
    recording_history_snapshot = list(cube_tracker.move_history)
    active_replay_move = ""
    replay_move_timer = 0

    # --- LIVE LOOP TIMING DIAGNOSTICS ---
    live_loop_last_ms = 0.0
    live_loop_max_ms = 0.0
    live_loop_ema_ms = 0.0
    live_loop_counter = 0

    # Replay window explicitly keeps the camera aspect ratio.
    cv2.namedWindow("Slow-Mo Video Analyzer", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Scrollable replay move-tab state. Mouse wheel is active only when the
    # cursor is over the left-side move history tab.
    replay_move_scroll = {"offset": 0}

    def replay_mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEWHEEL:
            return

        # Match the move-tab geometry used by draw_replay_move_tab().
        tab_x = 15
        tab_y = 105
        tab_w = 220
        tab_bottom = replay_display_height[0] - 45

        if tab_x <= x <= tab_x + tab_w and tab_y <= y <= tab_bottom:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 1 if flags > 0 else -1

            step = -3 if delta > 0 else 3
            replay_move_scroll["offset"] = max(
                0, replay_move_scroll["offset"] + step
            )

    # Mutable holder because frame height is known only during playback.
    replay_display_height = [720]
    cv2.setMouseCallback("Slow-Mo Video Analyzer", replay_mouse_callback)

    while True:
        loop_started_at = time.perf_counter()

        # LIVE CAMERA FEED ALWAYS RUNS
        ret, frame = cap.read()
        if not ret:
            break

        current_recording_frame_idx = None
        if slowmo_processor.is_recording:
            current_recording_frame_idx = recording_frame_idx
            recording_frame_idx += 1

        slowmo_processor.add_frame(frame)
        display_frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = display_frame.shape

        # --- LIVE YOLO INFERENCE & TRACKING ---
        results = model(display_frame, conf=0.40, verbose=False)[0]
        quad_corners = None
        current_grid, flat_face, raw_lab_patches = None, None, None

        if results.masks is not None and len(results.masks) > 0:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            best_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])

            mask_resized = cv2.resize(masks[best_idx], (w_frame, h_frame))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                pts = (
                    approx.reshape(4, 2)
                    if len(approx) == 4
                    else cv2.boxPoints(cv2.minAreaRect(cnt))
                )

                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect = np.zeros((4, 2), dtype="float32")
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                quad_corners = rect

                cv2.polylines(
                    display_frame,
                    [quad_corners.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2,
                )

        if quad_corners is not None:
            raw_corners = quad_corners.copy()
            raw_corners[:, 0] = w_frame - raw_corners[:, 0]

            s_raw = raw_corners.sum(axis=1)
            diff_raw = np.diff(raw_corners, axis=1)
            rect_raw = np.zeros((4, 2), dtype="float32")
            rect_raw[0] = raw_corners[np.argmin(s_raw)]
            rect_raw[2] = raw_corners[np.argmax(s_raw)]
            rect_raw[1] = raw_corners[np.argmin(diff_raw)]
            rect_raw[3] = raw_corners[np.argmax(diff_raw)]

            current_grid, flat_face, raw_lab_patches = extract_facelet_grid_with_debug(
                frame, rect_raw
            )

            if validate_face_grid(current_grid):
                async_pipeline.process_camera_frame(
                    current_grid,
                    frame_index=current_recording_frame_idx
                )

                if grid_lock.process(current_grid):
                    beam_tracker.process_observed_grid(current_grid)
                
                last_valid_grid = current_grid
                last_valid_flat_face = flat_face

        if last_valid_grid is not None:
            draw_permutation_ui(display_frame, last_valid_grid, last_valid_flat_face)

        if not kmeans_calibrator.is_calibrated:
            target_color = kmeans_calibrator.get_target_color()
            if target_color:
                cv2.rectangle(display_frame, (10, h_frame - 80), (w_frame - 10, h_frame - 10), (0, 0, 0), -1)
                cv2.putText(
                    display_frame,
                    f"SCANNING ({kmeans_calibrator.current_scan_idx + 1}/6): Hold {target_color.upper()} center face, press 'C'",
                    (20, h_frame - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

        if slowmo_processor.is_recording:
            current_history = list(cube_tracker.move_history)

            # Replay logging must compare the ACTUAL history contents, not just
            # the number of entries. A notation change such as:
            #     ["R"] -> ["R2"]
            # changes the move even though the list length stays at 1.
            # This also correctly handles normal appends, replacements, and
            # transient-glitch rollbacks.
            if current_history != recording_history_snapshot:
                previous_history = list(recording_history_snapshot)
                common_len = min(len(previous_history), len(current_history))
                first_diff = 0

                while (
                    first_diff < common_len
                    and previous_history[first_diff] == current_history[first_diff]
                ):
                    first_diff += 1

                # Remove every replay event from the first changed history index.
                replay_move_events[:] = [
                    e for e in replay_move_events
                    if e.get("move_index", -1) < first_diff
                ]

                # Re-add the current suffix exactly as it exists now.
                for idx in range(first_diff, len(current_history)):
                    replay_move_events.append({
                        "frame": current_recording_frame_idx,
                        "move": current_history[idx],
                        "move_index": idx,
                    })

                recording_history_snapshot = current_history

            cv2.circle(display_frame, (w_frame - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                display_frame,
                "REC (SLOW-MO)",
                (w_frame - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Rubik's Cube Tracker", display_frame)

        # --- COLLECT REPLAY FRAMES ---
        while not slowmo_processor.replay_queue.empty():
            rf = slowmo_processor.replay_queue.get()
            if rf is None:
                break
            replay_frames.append(rf)

        # --- RENDER VIDEO REPLAY ANALYZER WINDOW ---
        if len(replay_frames) > 0:
            if replay_idx >= len(replay_frames):
                replay_idx = len(replay_frames) - 1
            raw_frame_data = replay_frames[replay_idx]
            if not isinstance(raw_frame_data, np.ndarray):
                raw_frame_data = np.array(raw_frame_data, dtype=np.uint8)

            replay_raw_frame = np.ascontiguousarray(raw_frame_data, dtype=np.uint8)
            replay_display = cv2.flip(replay_raw_frame, 1)
            replay_display = np.ascontiguousarray(replay_display, dtype=np.uint8)
            
            h_r, w_r, _ = replay_display.shape
            replay_display_height[0] = h_r

            # YOLO inference on replay frame
            r_results = model(replay_display, conf=0.40, verbose=False)[0]
            r_quad_corners = None
            r_grid, r_flat_face, r_raw_lab = None, None, None

            if r_results.masks is not None and len(r_results.masks) > 0:
                r_masks = r_results.masks.data.cpu().numpy()
                r_boxes = r_results.boxes.xyxy.cpu().numpy().astype(int)
                r_best_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in r_boxes])

                r_mask_resized = cv2.resize(r_masks[r_best_idx], (w_r, h_r))
                r_mask_binary = (r_mask_resized > 0.5).astype(np.uint8) * 255

                r_contours, _ = cv2.findContours(r_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if r_contours:
                    cnt = max(r_contours, key=cv2.contourArea)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                    pts = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(cnt))

                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)
                    rect = np.zeros((4, 2), dtype="float32")
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    r_quad_corners = rect

            if r_quad_corners is not None:
                r_raw_corners = r_quad_corners.copy()
                r_raw_corners[:, 0] = w_r - r_raw_corners[:, 0]

                s_raw = r_raw_corners.sum(axis=1)
                diff_raw = np.diff(r_raw_corners, axis=1)
                rect_raw = np.zeros((4, 2), dtype="float32")
                rect_raw[0] = r_raw_corners[np.argmin(s_raw)]
                rect_raw[2] = r_raw_corners[np.argmax(s_raw)]
                rect_raw[1] = r_raw_corners[np.argmin(diff_raw)]
                rect_raw[3] = r_raw_corners[np.argmax(diff_raw)]

                r_grid, r_flat_face, r_raw_lab = extract_facelet_grid_with_debug(replay_raw_frame, rect_raw)

            # Build move list from moves actually detected during live recording
            active_moves = [
                f"{e['move']}@F{e['frame']}"
                for e in replay_move_events
                if e['frame'] <= replay_idx
            ]

            frame_moves = [
                e['move']
                for e in replay_move_events
                if e['frame'] == replay_idx
            ]

            debug_data = async_pipeline.get_debug_for_frame(replay_idx) or {}

            engine_metrics = async_pipeline.get_diagnostic_metrics()

            deduction_info = {
                'hypothesized_move': debug_data.get('chosen', 'NONE'),
                'decided_move': ', '.join(frame_moves) if frame_moves else 'HOLD',
                'confidence': 1.0,
                'top_candidates': debug_data.get('top_candidates', []),
                'recovery_mode': debug_data.get('recovery_mode', False),
                'status': debug_data.get('status', 'NO DEDUCTION DATA'),
                'hypothesis_count': debug_data.get('hypothesis_count', 0),
                'commit_ready': debug_data.get('commit_ready', False),
                'commit_reason': debug_data.get('commit_reason', ''),
                'best_confidence': debug_data.get('best_confidence', 0.0),
                'second_confidence': debug_data.get('second_confidence', 0.0),
                'confidence_lead': debug_data.get('confidence_lead', 0.0),
                'ambiguity_observations': debug_data.get('ambiguity_observations', 0),
                'max_pending_moves': debug_data.get('max_pending_moves', 0),
                'queue_now': engine_metrics.get('queue_now', 0),
                'queue_dropped': engine_metrics.get('dropped', 0),
                'queue_enqueued': engine_metrics.get('enqueued', 0),
                'queue_delay_max_ms': engine_metrics.get('max_queue_delay_ms', 0.0),
                'deduction_max_ms': engine_metrics.get('max_deduction_ms', 0.0),
                'live_loop_ms': live_loop_last_ms,
                'live_loop_max_ms': live_loop_max_ms,
            }

            # Draw Debug Inspector Panel
            if show_debug_ui:
                draw_debug_inspector_panel(
                    replay_display,
                    grid=r_grid,
                    raw_lab_patches=r_raw_lab,
                    is_paused=replay_paused,
                    frame_idx=replay_idx,
                    total_frames=len(replay_frames),
                    quad_corners=r_quad_corners,
                    deduction_info=deduction_info
                )

            # Vertical, scrollable replay move history.
            replay_move_scroll["offset"], _ = draw_replay_move_tab(
                replay_display,
                replay_move_events,
                replay_idx,
                replay_move_scroll["offset"],
            )

            # Playback bar overlay
            cv2.putText(
                replay_display,
                f"FRAME {replay_idx + 1}/{len(replay_frames)} | {'PAUSED' if replay_paused else 'PLAYING'}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255) if replay_paused else (0, 255, 0),
                2,
            )
            
            if show_replay_debug:
                cv2.putText(
                    replay_display,
                    "[p] Pause/Play | [,] Prev | [.] Next | Mouse Wheel=Moves | [d] UI | [o] Close",
                    (20, h_r - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )

            cv2.imshow("Slow-Mo Video Analyzer", replay_display)

            if not replay_paused:
                replay_idx = (replay_idx + 1) % len(replay_frames)

        # --- LIVE LOOP TIMING UPDATE ---
        live_loop_last_ms = (time.perf_counter() - loop_started_at) * 1000.0
        live_loop_max_ms = max(live_loop_max_ms, live_loop_last_ms)
        live_loop_counter += 1
        if live_loop_ema_ms <= 0.0:
            live_loop_ema_ms = live_loop_last_ms
        else:
            live_loop_ema_ms = (0.95 * live_loop_ema_ms) + (0.05 * live_loop_last_ms)

        if live_loop_counter % 120 == 0:
            m = async_pipeline.get_diagnostic_metrics()
            print(
                "📈 [LIVE PIPELINE] "
                f"loop={live_loop_last_ms:.1f}ms avg={live_loop_ema_ms:.1f}ms max={live_loop_max_ms:.1f}ms | "
                f"queue={m['queue_now']} drops={m['dropped']} | "
                f"qdelay={m['last_queue_delay_ms']:.1f}/{m['max_queue_delay_ms']:.1f}ms | "
                f"deduction={m['last_deduction_ms']:.1f}/{m['max_deduction_ms']:.1f}ms | "
                f"hypotheses={len(async_pipeline.hypotheses)}"
            )

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # SPACEBAR: Start / Stop Recording
            if not slowmo_processor.is_recording:
                replay_frames.clear()
                move_log.clear()
                replay_move_events.clear()
                replay_move_scroll["offset"] = 0
                recording_frame_idx = 0
                recording_history_snapshot = list(cube_tracker.move_history)
                with async_pipeline._deduction_history_lock:
                    async_pipeline.deduction_history.clear()
                async_pipeline.last_deduction_debug = {
                    "top_candidates": [],
                    "chosen": "NONE",
                    "recovery_mode": False,
                    "status": "RECORDING RESET",
                    "hypothesis_count": len(async_pipeline.hypotheses),
                    "commit_ready": False,
                    "commit_reason": "RECORDING RESET",
                    "first_move_groups": {},
                }
                slowmo_processor.start_recording()
            else:
                slowmo_processor.stop_and_process(model, extract_fn)
        elif key == ord("o"):  # CLOSE WINDOW
            if len(replay_frames) > 0 or cv2.getWindowProperty("Slow-Mo Video Analyzer", cv2.WND_PROP_VISIBLE) >= 1:
                replay_frames.clear()
                move_log.clear()
                replay_move_scroll["offset"] = 0
                replay_idx = 0
                replay_paused = False
                try:
                    cv2.destroyWindow("Slow-Mo Video Analyzer")
                except cv2.error:
                    pass
                print("[REPLAY] Video Debug Window closed.")
        elif key == ord("p"):  # PAUSE / PLAY
            replay_paused = not replay_paused
            print(f"[REPLAY] {'Paused' if replay_paused else 'Playing'}")
        elif key == ord("."):  # STEP FORWARD
            if len(replay_frames) > 0:
                replay_paused = True
                replay_idx = (replay_idx + 1) % len(replay_frames)
        elif key == ord(","):  # STEP BACKWARD
            if len(replay_frames) > 0:
                replay_paused = True
                replay_idx = (replay_idx - 1) % len(replay_frames)
        elif key == ord("d"):  # TOGGLE UI
            show_debug_ui = not show_debug_ui
            print(f"[DEBUG UI] {'Enabled' if show_debug_ui else 'Disabled'}")
        elif key == ord("c"):
            if raw_lab_patches is not None and not kmeans_calibrator.is_calibrated:
                target = kmeans_calibrator.get_target_color()
                is_done = kmeans_calibrator.capture_face(raw_lab_patches)
                print(f"[SCANNED] Captured face with center: {target}")
        elif key == ord("r"):
            cube_tracker.reset_to_solved()
            async_pipeline.reset_tracking()
            last_valid_grid = None
            last_valid_flat_face = None
            beam_tracker = BeamStateTracker(
                cube_tracker.state, cube_tracker.permutations, beam_width=5
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()