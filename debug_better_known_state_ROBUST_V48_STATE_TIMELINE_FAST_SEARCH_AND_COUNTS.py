# V42: global timeline reconstruction + full-frame reanalysis + transformed-frame YOLO retries + live replay grid/status
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



# =============================================================================
# PHYSICAL MOTION EVIDENCE GATE
# =============================================================================
class MotionEvidenceGate:
    """
    Geometry-only physical-motion gate adapted directly from the two supplied
    motion prototypes.

    IMPORTANT:
    - This gate does NOT use colour changes to decide whether motion happened.
    - It rejects hand-holding/camera-jitter by removing the cube's rigid motion
      first, then requiring a *large sudden* and *structured* face/slice rotation pattern.
    - Optical flow may report physical movement for diagnostics only.
      No move name is inferred from the motion vectors and it never gates a
      persistent camera-state change.
    - Whole-cube rigid motion is measured separately for diagnostics, but it is
      NOT sufficient by itself to authorize a face move.

    This is intentionally an EVIDENCE gate only. The permutation engine still
    decides which state transition actually occurred, including hidden B/B'
    deductions when later visible observations make them necessary.
    """

    def __init__(
        self,
        vector_deadzone=0.75,
        min_total_energy=4.00,
        min_slice_displacement=1.80,
        isolation_ratio=1.25,
        purity_ratio=1.25,
        motion_recency_ms=80.0,
        required_structured_frames=1,
        strong_energy=9.5,
        min_pattern_cells=3,
        whole_cube_angle_deg=1.5,
        whole_cube_translation_px=5.0,
        whole_cube_scale_delta=0.025,
        # New anti-jitter / burst thresholds.  These deliberately make the
        # motion gate conservative: small continuous hand movement is not
        # considered evidence of a real move.
        min_raw_energy=6.5,
        burst_ratio=2.20,
        min_energy_jump=2.0,
        small_common_mean_mag=1.50,
        small_common_energy=999.0,
        common_direction_coherence=0.90,
        common_magnitude_cv=0.55,
        baseline_window=15,
    ):
        self.vector_deadzone = float(vector_deadzone)
        self.min_total_energy = float(min_total_energy)
        self.min_slice_displacement = float(min_slice_displacement)
        self.isolation_ratio = float(isolation_ratio)
        self.purity_ratio = float(purity_ratio)
        self.motion_recency_ms = float(motion_recency_ms)
        self.required_structured_frames = int(required_structured_frames)
        self.strong_energy = float(strong_energy)
        self.min_pattern_cells = int(min_pattern_cells)
        self.whole_cube_angle_deg = float(whole_cube_angle_deg)
        self.whole_cube_translation_px = float(whole_cube_translation_px)
        self.whole_cube_scale_delta = float(whole_cube_scale_delta)

        # A move must look like a large burst, not ordinary hand jitter.
        self.min_raw_energy = float(min_raw_energy)
        self.burst_ratio = float(burst_ratio)
        self.min_energy_jump = float(min_energy_jump)
        self.small_common_mean_mag = float(small_common_mean_mag)
        self.small_common_energy = float(small_common_energy)
        self.common_direction_coherence = float(common_direction_coherence)
        self.common_magnitude_cv = float(common_magnitude_cv)
        self.energy_history = deque(maxlen=max(5, int(baseline_window)))

        self.prev_gray = None
        self.prev_quad = None
        self.dis_flow = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_FAST
        )

        self.last_motion_time = None
        self.last_motion_move = None  # Deliberately unused: motion never guesses a move.
        self.last_motion_energy = 0.0
        self.last_active_cells = 0
        self.last_category = "NONE"
        self.motion_streak = 0

        self.last_whole_cube_angle_deg = 0.0
        self.last_whole_cube_translation_px = 0.0
        self.last_whole_cube_scale_delta = 0.0

        self.last_raw_energy = 0.0
        self.last_motion_baseline = 0.0
        self.last_energy_jump = 0.0
        self.last_burst = False
        self.last_common_small_motion = False

    @staticmethod
    def _order_points(pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _subdivide_3x3(self, quad_pts):
        quad_pts = self._order_points(quad_pts)
        tl, tr, br, bl = quad_pts

        src = np.array(
            [[0, 0], [3, 0], [3, 3], [0, 3]], dtype=np.float32
        )
        dst = np.array([tl, tr, br, bl], dtype=np.float32)

        H, _ = cv2.findHomography(src, dst)
        if H is None:
            return None

        cell_polys = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                unit_cell = np.array(
                    [
                        [c, r],
                        [c + 1, r],
                        [c + 1, r + 1],
                        [c, r + 1],
                    ],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(unit_cell, H)
                cell_polys[r][c] = projected.reshape(4, 2).astype(np.int32)

        return cell_polys

    def _cell_centers(self, quad_pts):
        cells = self._subdivide_3x3(quad_pts)
        if cells is None:
            return None

        centers = np.zeros((3, 3, 2), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                poly = cells[r][c]
                M = cv2.moments(poly)
                if M["m00"] == 0:
                    return None
                centers[r, c] = [
                    float(M["m10"] / M["m00"]),
                    float(M["m01"] / M["m00"]),
                ]
        return centers

    def _rigid_transform(self, prev_quad, curr_quad):
        """
        Estimate the cube's common rigid image motion from the four YOLO
        corners. This is the key anti-hand-jitter step: if the entire cube
        drifts because the user is holding it imperfectly still, that motion is
        removed before evaluating a face turn.
        """
        self.last_whole_cube_angle_deg = 0.0
        self.last_whole_cube_translation_px = 0.0
        self.last_whole_cube_scale_delta = 0.0

        if prev_quad is None or curr_quad is None:
            return None

        prev_quad = self._order_points(prev_quad)
        curr_quad = self._order_points(curr_quad)

        matrix, inliers = cv2.estimateAffinePartial2D(
            prev_quad,
            curr_quad,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is None:
            return None

        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        c, d = float(matrix[1, 0]), float(matrix[1, 1])

        scale_x = float(np.hypot(a, c))
        scale_y = float(np.hypot(b, d))
        scale = 0.5 * (scale_x + scale_y)
        angle_deg = float(np.degrees(np.arctan2(c, a)))
        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        translation = float(np.hypot(tx, ty))

        self.last_whole_cube_angle_deg = angle_deg
        self.last_whole_cube_translation_px = translation
        self.last_whole_cube_scale_delta = abs(scale - 1.0)

        return matrix

    def _motion_burst_metrics(self, vecs):
        """
        Measure whether the current optical-flow frame is a *large sudden
        burst* rather than normal hand/camera jitter.

        During simple holding, the observed vectors tend to:
          - be small,
          - point in nearly the same direction, and
          - have similar magnitudes.

        That combination is explicitly rejected before the face-turn pattern
        classifier is allowed to fire.
        """
        if vecs is None or vecs.shape != (3, 3, 2):
            return {
                "raw_energy": 0.0,
                "baseline": 0.0,
                "jump": 0.0,
                "burst": False,
                "small_common": False,
                "direction_coherence": 0.0,
                "magnitude_cv": 999.0,
                "mean_mag": 0.0,
                "active_cells": 0,
                "max_cell_mag": 0.0,
            }

        flat = vecs.reshape(9, 2).astype(np.float32)
        mags = np.linalg.norm(flat, axis=1)
        raw_energy = float(np.sum(mags))
        active = mags[mags >= self.vector_deadzone]
        active_cells = int(active.size)
        max_cell_mag = float(np.max(mags)) if mags.size else 0.0

        if active_cells > 0:
            active_vecs = flat[mags >= self.vector_deadzone]
            mean_mag = float(np.mean(active))
            std_mag = float(np.std(active))
            magnitude_cv = std_mag / max(mean_mag, 1e-6)

            unit = active_vecs / np.maximum(active[:, None], 1e-6)
            direction_coherence = float(np.linalg.norm(np.mean(unit, axis=0)))
        else:
            mean_mag = 0.0
            magnitude_cv = 999.0
            direction_coherence = 0.0

        baseline = float(np.median(self.energy_history)) if self.energy_history else 0.0
        jump = raw_energy - baseline

        # "Big and sudden" means the absolute energy is high AND substantially
        # above the recent normal level.  The strong-energy escape hatch allows
        # a genuinely large single-frame movement to trigger immediately.
        burst = bool(
            raw_energy >= self.min_raw_energy
            and (
                raw_energy >= self.strong_energy
                or (
                    jump >= self.min_energy_jump
                    and raw_energy >= max(self.min_raw_energy, baseline * self.burst_ratio)
                )
            )
        )

        # This is the exact holding pattern described by the user: many vectors
        # move together in roughly the same direction, with roughly the same
        # small magnitude.  Do not let imperfect cube/hand stability through.
        small_common = bool(
            active_cells >= 5
            and mean_mag <= self.small_common_mean_mag
            and direction_coherence >= self.common_direction_coherence
            and magnitude_cv <= self.common_magnitude_cv
        )

        self.last_raw_energy = raw_energy
        self.last_motion_baseline = baseline
        self.last_energy_jump = jump
        self.last_burst = burst
        self.last_common_small_motion = small_common

        return {
            "raw_energy": raw_energy,
            "baseline": baseline,
            "jump": jump,
            "burst": burst,
            "small_common": small_common,
            "direction_coherence": direction_coherence,
            "magnitude_cv": magnitude_cv,
            "mean_mag": mean_mag,
            "max_cell_mag": max_cell_mag,
            "active_cells": active_cells,
        }

    def process_frame(self, frame, quad_corners):
        now = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = {
            "recent": False,
            "move_detected": False,
            "move": None,
            "category": "NONE",
            "energy": 0.0,
            "raw_energy": 0.0,
            "motion_baseline": 0.0,
            "energy_jump": 0.0,
            "burst": False,
            "common_small": False,
            "direction_coherence": 0.0,
            "magnitude_cv": 0.0,
            "max_cell_mag": 0.0,
            "active_cells": 0,
            "motion_age_ms": None,
            "observed_at": now,
            "structured": False,
            "whole_cube": False,
            "whole_cube_angle_deg": 0.0,
            "whole_cube_translation_px": 0.0,
            "whole_cube_scale_delta": 0.0,
        }

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_quad = None if quad_corners is None else self._order_points(quad_corners)
            return result

        if quad_corners is None:
            self.prev_gray = gray
            self.prev_quad = None
            self.motion_streak = 0
            return result

        try:
            flow = self.dis_flow.calc(self.prev_gray, gray, None)
        except Exception:
            self.prev_gray = gray
            self.prev_quad = self._order_points(quad_corners)
            self.motion_streak = 0
            return result

        curr_quad = self._order_points(quad_corners)
        rigid_matrix = self._rigid_transform(self.prev_quad, curr_quad)

        cell_polys = self._subdivide_3x3(curr_quad)
        centers = self._cell_centers(curr_quad)
        if cell_polys is None or centers is None:
            self.prev_gray = gray
            self.prev_quad = curr_quad
            self.motion_streak = 0
            return result

        raw_vecs = np.zeros((3, 3, 2), dtype=np.float32)
        h, w = gray.shape[:2]

        for r in range(3):
            for c in range(3):
                poly = cell_polys[r][c]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                c_flow = flow[mask == 255]

                if len(c_flow) == 0:
                    continue

                u = float(np.median(c_flow[:, 0]))
                v = float(np.median(c_flow[:, 1]))
                mag = float(np.hypot(u, v))
                if mag >= self.vector_deadzone:
                    raw_vecs[r, c] = [u, v]

        # IMPORTANT: The motion system is ONLY an event detector.
        # It must NOT guess R/R'/U/U'/... from the optical-flow pattern.
        # The colour/permutation engine remains responsible for deciding which
        # cube move explains the resulting state change.
        #
        # A valid physical-movement event is simply a large, sudden optical-flow
        # burst that is not the small common-direction motion produced while the
        # user is merely holding the cube.
        raw_burst = self._motion_burst_metrics(raw_vecs)

        whole_cube = (
            abs(self.last_whole_cube_angle_deg) >= self.whole_cube_angle_deg
            or self.last_whole_cube_translation_px >= self.whole_cube_translation_px
            or self.last_whole_cube_scale_delta >= self.whole_cube_scale_delta
        )

        physical_motion = bool(
            raw_burst["burst"]
            and not raw_burst["small_common"]
        )

        if physical_motion:
            self.last_motion_time = now
            self.last_motion_move = None
            self.last_motion_energy = float(raw_burst["raw_energy"])
            self.last_active_cells = int(raw_burst["active_cells"])
            self.last_category = "PHYSICAL_MOTION"

        age_ms = (
            (now - self.last_motion_time) * 1000.0
            if self.last_motion_time is not None
            else None
        )
        recent = age_ms is not None and age_ms <= self.motion_recency_ms

        result.update({
            "recent": bool(recent),
            "move_detected": bool(recent),
            "move": None,
            "category": (
                "PHYSICAL_MOTION" if recent
                else ("WHOLE_CUBE" if whole_cube else ("JITTER" if raw_burst["small_common"] else "NONE"))
            ),
            "energy": float(raw_burst["raw_energy"]),
            "raw_energy": float(raw_burst["raw_energy"]),
            "motion_baseline": float(raw_burst["baseline"]),
            "energy_jump": float(raw_burst["jump"]),
            "burst": bool(raw_burst["burst"]),
            "common_small": bool(raw_burst["small_common"]),
            "direction_coherence": float(raw_burst["direction_coherence"]),
            "magnitude_cv": float(raw_burst["magnitude_cv"]),
            "max_cell_mag": float(raw_burst["max_cell_mag"]),
            "active_cells": int(raw_burst["active_cells"]),
            "motion_age_ms": (
                round(max(0.0, age_ms), 2)
                if age_ms is not None
                else None
            ),
            "structured": bool(recent),
            "whole_cube": bool(whole_cube),
            "whole_cube_angle_deg": float(self.last_whole_cube_angle_deg),
            "whole_cube_translation_px": float(self.last_whole_cube_translation_px),
            "whole_cube_scale_delta": float(self.last_whole_cube_scale_delta),
        })

        self.energy_history.append(raw_burst["raw_energy"])

        self.prev_gray = gray
        self.prev_quad = curr_quad
        return result

        # Keep the current raw energy only after the decision has been made, so
        # it becomes part of the baseline for the *next* frame.
        self.energy_history.append(raw_burst["raw_energy"])

        self.prev_gray = gray
        self.prev_quad = curr_quad
        return result


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

    def process_camera_frame(self, scanned_grid, frame_index=None, motion_evidence=None):
        # Timestamp the observation at the camera-ingestion point, not later in
        # the consumer thread. The temporal glitch detector uses this value to
        # distinguish a genuine sequence from a rapid classifier glitch.
        observation_time = time.perf_counter()

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
            # state. A persistent DIFFERENT grid is always allowed through to
            # the state engine; no motion requirement is applied here.
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
                    "observation_time": observation_time,
                    "motion_evidence": copy.deepcopy(motion_evidence) if motion_evidence is not None else None,
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
                    observation_time = queued_item.get("observation_time")
                    motion_evidence = queued_item.get("motion_evidence") or {}
                else:
                    target_front = queued_item
                    frame_index = None
                    observation_time = None
                    motion_evidence = {}

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
                    "motion_gate": bool(motion_evidence.get("structured", False)),
                    "motion_category": motion_evidence.get("category", "NONE"),
                    "motion_move": motion_evidence.get("move"),
                    "motion_energy": round(float(motion_evidence.get("energy", 0.0)), 3),
                    "motion_age_ms": motion_evidence.get("motion_age_ms"),
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
                    authoritative_front_now = self._normalize(
                        self.tracker.state[18:27]
                    )

                    # V46: NO HARD MOTION GATE.
                    # A persistent camera state change is allowed to enter the
                    # normal state-transition engine even when the motion system
                    # reports nothing. Motion is retained only as diagnostic data.
                    # This prevents a missed optical-flow burst from deleting a
                    # real move before the cube-state engine can evaluate it.
                    if target_front != authoritative_front_now:
                        self.last_deduction_debug["state_change_without_motion"] = not bool(
                            motion_evidence.get("structured", False)
                        )

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
    reanalysis_variant = str(deduction_info.get('reanalysis_variant', '')) if deduction_info else ''
    if reanalysis_variant:
        cv2.putText(frame, f"VIDEO PERCEPTION: {reanalysis_variant[:28]}", (panel_x1 + 15, 333), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 230, 255), 1)

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


class FinalSolveAnalyzer:
    """
    Offline/deferred solve reconstruction pass.

    The live V29 tracker remains fast and immediate. This analyzer is different:
    it receives the entire recorded observation timeline after the user stops
    recording and can therefore search substantially deeper, keep more complete
    cube-state hypotheses, and use later observations to correct earlier
    ambiguous deductions.

    It never mutates the live cube tracker while analysing. It returns a fully
    verified state/history pair to the main thread, which then becomes
    authoritative.
    """

    # Final analysis is intentionally deeper than live tracking, but it must
    # also finish promptly.  We use a bounded beam search rather than an
    # exhaustive depth-8 BFS, which could explode into millions of states for
    # a single checkpoint and starve the UI.
    # Fast-first final analysis.  The post-solve pass is allowed to correct
    # live mistakes, but it is explicitly capped so it can never turn into
    # an exhaustive search that keeps the user waiting.
    FINAL_ANALYSIS_MAX_TIME_MS = 50000.0
    FINAL_BEAM_WIDTH = 6
    FINAL_NORMAL_DEPTH = 3
    FINAL_RECOVERY_DEPTH = 5
    FINAL_RECOVERY_HYPOTHESES = 2
    FINAL_NORMAL_MAX_SEARCH_NODES = 8000
    FINAL_RECOVERY_MAX_SEARCH_NODES = 20000
    FINAL_MAX_MATCHES_PER_SEARCH = 12
    FINAL_NORMAL_BEAM = 120
    FINAL_RECOVERY_BEAM = 220
    FINAL_NORMAL_TIME_BUDGET_MS = 80.0
    FINAL_RECOVERY_TIME_BUDGET_MS = 300.0
    # A move is created from persistent observed state changes, not motion.
    MIN_STABLE_FRAMES = 3
    MAX_TRANSIENT_RUN_FRAMES = 4
    MAX_TRANSIENT_RUN_MS = 550.0
    MOTION_CONTEXT_FRAMES = 10
    LOCAL_MOTION_PRE_FRAMES = 4
    LOCAL_MOTION_POST_FRAMES = 8
    MOTION_POST_CHECKPOINT_SAMPLES = 4
    FINAL_TAIL_SAMPLES = 6

    # Final correction layer: use the recorded physical-motion bursts as a
    # constraint on how many cube moves could plausibly have happened.  The
    # previous final analyzer only asked "can some sequence produce this front?"
    # which can turn a 3-move solve into an arbitrary 7-move explanation.
    MOTION_CORRECTION_BEAM = 24
    MOTION_CORRECTION_MAX_BURSTS_PER_GAP = 6
    MOTION_CORRECTION_LENGTH_TOLERANCE = 1
    MOTION_CORRECTION_LIVE_PRIOR_WEIGHT = 1.25
    MOTION_CORRECTION_LENGTH_PENALTY = 3.0
    MOTION_CORRECTION_HOLD_PENALTY = 0.75
    MOTION_CORRECTION_MAX_TOTAL_EXTRA_MOVES = 2

    # ------------------------------------------------------------------
    # V39 TEMPORAL GLOBAL CORRECTION
    #
    # This layer is deliberately different from front-only/deep search.
    # It treats each physical motion burst as an event, builds temporally
    # stable colour observations before/after the event, and reconstructs
    # the complete solve with one move per event whenever possible.
    # ------------------------------------------------------------------
    TEMPORAL_EVENT_MERGE_GAP_MS = 140.0
    TEMPORAL_PRE_SAMPLES = 10
    TEMPORAL_POST_SAMPLES = 10
    TEMPORAL_MIN_GRID_SAMPLES = 3
    TEMPORAL_MIN_CELL_AGREEMENT = 0.55
    TEMPORAL_BEAM_WIDTH = 12
    TEMPORAL_MAX_EXTRA_MOVES = 1
    TEMPORAL_ONE_MOVE_MISMATCH_LIMIT = 2
    TEMPORAL_TWO_MOVE_MISMATCH_LIMIT = 1
    TEMPORAL_MOVE_PRIOR_WEIGHT = 0.35
    TEMPORAL_EXTRA_MOVE_PENALTY = 4.0
    TEMPORAL_HOLD_PENALTY_WITH_CHANGED_FRONT = 8.0
    TEMPORAL_HOLD_PENALTY_WITH_SAME_FRONT = 1.5
    TEMPORAL_FINAL_FRONT_WEIGHT = 16.0
    TEMPORAL_EVENT_MISMATCH_WEIGHT = 7.0
    TEMPORAL_BEFORE_MISMATCH_WEIGHT = 1.5
    TEMPORAL_MIN_FINAL_CONFIDENCE = 0.0

    MOVE_ORDER = (
        "R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"
    )

    INVERSES = {
        "R": "R'", "R'": "R",
        "L": "L'", "L'": "L",
        "U": "U'", "U'": "U",
        "D": "D'", "D'": "D",
        "F": "F'", "F'": "F",
        "B": "B'", "B'": "B",
    }

    def __init__(self, tracker):
        self.tracker = tracker
        self._v47_sequence_levels = None

    @staticmethod
    def _normalize(grid):
        if grid is None:
            return []
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            grid = [v for row in grid for v in row]
        return [str(v) for v in list(grid)]

    @classmethod
    def _valid_grid(cls, grid):
        flat = cls._normalize(grid)
        return len(flat) == 9 and "Unknown" not in flat

    @classmethod
    def _grid_key(cls, grid):
        return tuple(cls._normalize(grid))

    @staticmethod
    def _motion_is_real(evidence):
        return bool((evidence or {}).get("structured", False))

    @classmethod
    def _count_motion_bursts(cls, records, start_idx, end_idx):
        if not records:
            return 0
        start_idx = max(0, int(start_idx))
        end_idx = min(len(records) - 1, int(end_idx))
        if end_idx < start_idx:
            return 0

        bursts = 0
        prev = False
        for i in range(start_idx, end_idx + 1):
            current = cls._motion_is_real(records[i].get("motion_evidence"))
            if current and not prev:
                bursts += 1
            prev = current
        return bursts

    @classmethod
    def _build_runs(cls, observations):
        """Run-length encode stable colour grids while retaining timeline indices."""
        runs = []
        current = None

        for idx, item in enumerate(observations):
            grid = cls._normalize(item.get("grid"))
            if not cls._valid_grid(grid):
                continue

            key = tuple(grid)
            if current is None or current["key"] != key:
                if current is not None:
                    runs.append(current)
                current = {
                    "key": key,
                    "grid": list(grid),
                    "first_idx": idx,
                    "last_idx": idx,
                    "first_frame": item.get("frame_index"),
                    "last_frame": item.get("frame_index"),
                    "first_time": item.get("observation_time"),
                    "last_time": item.get("observation_time"),
                    "count": 1,
                }
            else:
                current["last_idx"] = idx
                current["last_frame"] = item.get("frame_index")
                current["last_time"] = item.get("observation_time")
                current["count"] += 1

        if current is not None:
            runs.append(current)
        return runs

    @classmethod
    def _prepare_checkpoints(cls, observations, base_front):
        """
        Build stable colour checkpoints without letting one bad plateau poison
        the entire offline solve.

        The old final pass had two problems:
          1. It counted motion anywhere in the whole gap, so a glare plateau
             appearing after a real move could be accepted as if it were a real
             checkpoint.
          2. If one checkpoint could not be explained, it stopped the analysis
             entirely. That meant all later moves were never reconsidered.

        This version keeps *all* stable runs as candidate checkpoints and adds
        local evidence about whether a physical-motion burst occurred immediately
        around the transition into the run. The analysis stage can then skip a
        suspicious run and continue to later checkpoints.
        """
        runs = cls._build_runs(observations)
        if not runs:
            return [], {
                "stable_runs": 0,
                "candidate_checkpoints": 0,
                "initial_checkpoint_found": False,
                "ignored_short_runs": 0,
                "motion_bursts_local": 0,
                "motion_bursts_global": 0,
                "transient_runs": 0,
            }

        base_key = tuple(cls._normalize(base_front))
        candidates = []
        ignored_short = 0
        local_bursts_total = 0
        global_bursts_total = 0
        transient_runs = 0

        def burst_count(a, b):
            return cls._count_motion_bursts(observations, a, b)

        # Find the first stable run matching the known starting front.
        initial_run_index = None
        for run_index, run in enumerate(runs):
            if run["count"] >= cls.MIN_STABLE_FRAMES and run["key"] == base_key:
                candidates.append({
                    "grid": list(run["grid"]),
                    "frame": run["last_frame"],
                    "time": run["last_time"],
                    "first_frame": run["first_frame"],
                    "first_time": run["first_time"],
                    "run_index": run_index,
                    "first_idx": run["first_idx"],
                    "last_idx": run["last_idx"],
                    "count": run["count"],
                    "motion_bursts": 0,
                    "local_motion_bursts": 0,
                    "motion_near_transition": False,
                    "reason": "INITIAL KNOWN FRONT",
                    "suspect": False,
                })
                initial_run_index = run_index
                break

        for run_index, run in enumerate(runs):
            if run_index == initial_run_index:
                continue

            if run["count"] < cls.MIN_STABLE_FRAMES:
                ignored_short += 1
                continue

            # Search for a local physical-motion burst around the point where
            # this new colour plateau first appeared. This is deliberately local
            # rather than counting every burst since the last checkpoint.
            local_start = max(0, run["first_idx"] - cls.LOCAL_MOTION_PRE_FRAMES)
            local_end = min(
                len(observations) - 1,
                run["first_idx"] + cls.LOCAL_MOTION_POST_FRAMES,
            )
            local_bursts = burst_count(local_start, local_end)
            global_start = max(0, run["first_idx"] - cls.MOTION_CONTEXT_FRAMES)
            global_end = min(
                len(observations) - 1,
                run["last_idx"] + cls.MOTION_CONTEXT_FRAMES,
            )
            global_bursts = burst_count(global_start, global_end)
            local_bursts_total += local_bursts
            global_bursts_total += global_bursts

            first_time = run.get("first_time")
            last_time = run.get("last_time")
            duration_ms = 0.0
            if first_time is not None and last_time is not None:
                duration_ms = max(0.0, (float(last_time) - float(first_time)) * 1000.0)

            # A short plateau that immediately returns to the previous stable
            # colour and has no local motion is a classic transient colour read.
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            returns_to_previous = (
                next_run is not None
                and next_run["key"] == (
                    candidates[-1]["key"] if candidates and "key" in candidates[-1]
                    else base_key
                )
            )
            # For a transient A -> B -> A colour glitch, the next run should
            # return to the run immediately before B. Do not use the most recent
            # candidate in ``candidates`` here because that candidate may itself
            # be the suspicious B plateau.
            previous_run_key = (
                runs[run_index - 1]["key"]
                if run_index > 0
                else base_key
            )
            returns_to_previous = (
                next_run is not None
                and next_run["count"] >= cls.MIN_STABLE_FRAMES
                and next_run["key"] == previous_run_key
            )
            suspect_transient = (
                local_bursts == 0
                and returns_to_previous
                and run["count"] <= cls.MAX_TRANSIENT_RUN_FRAMES
                and duration_ms <= cls.MAX_TRANSIENT_RUN_MS
            )
            if suspect_transient:
                transient_runs += 1

            candidates.append({
                "grid": list(run["grid"]),
                "frame": run["last_frame"],
                "time": run["last_time"],
                "first_frame": run["first_frame"],
                "first_time": run["first_time"],
                "run_index": run_index,
                "first_idx": run["first_idx"],
                "last_idx": run["last_idx"],
                "count": run["count"],
                "motion_bursts": global_bursts,
                "local_motion_bursts": local_bursts,
                "motion_near_transition": bool(local_bursts > 0),
                "duration_ms": duration_ms,
                "reason": (
                    "STABLE FRONT WITH LOCAL PHYSICAL MOTION"
                    if local_bursts > 0
                    else "STABLE FRONT WITHOUT LOCAL MOTION"
                ),
                "suspect": bool(suspect_transient),
            })

        return candidates, {
            "stable_runs": len([r for r in runs if r["count"] >= cls.MIN_STABLE_FRAMES]),
            "candidate_checkpoints": len(candidates),
            "initial_checkpoint_found": initial_run_index is not None,
            "ignored_short_runs": ignored_short,
            "motion_bursts_local": local_bursts_total,
            "motion_bursts_global": global_bursts_total,
            "transient_runs": transient_runs,
        }

    def _search_matches(
        self,
        start_state,
        target_front,
        max_depth,
        max_nodes=None,
        time_budget_ms=None,
    ):
        """
        Bounded deep search for exact visible-front matches.

        The earlier final analyzer used exhaustive breadth-first search. At
        depth 8 that can become enormous and, because Python is doing the work
        in a worker thread, it can make the replay/UI appear frozen.  The final
        pass only needs strong candidate paths, not every possible path, so we
        use a beam guided by visible-front mismatch while retaining full-cube
        state deduplication.
        """
        target_front = self._normalize(target_front)
        start_state = start_state.copy()
        initial_front = self._normalize(start_state[18:27])

        if target_front == initial_front:
            return [{
                "cost": 0.0,
                "path": [],
                "state": start_state.copy(),
            }]

        node_limit = int(
            self.FINAL_RECOVERY_MAX_SEARCH_NODES
            if max_nodes is None else max_nodes
        )
        if time_budget_ms is None:
            time_budget_ms = (
                self.FINAL_RECOVERY_TIME_BUDGET_MS
                if max_depth > self.FINAL_NORMAL_DEPTH
                else self.FINAL_NORMAL_TIME_BUDGET_MS
            )
        deadline = time.perf_counter() + (float(time_budget_ms) / 1000.0)

        beam_limit = (
            self.FINAL_RECOVERY_BEAM
            if max_depth > self.FINAL_NORMAL_DEPTH
            else self.FINAL_NORMAL_BEAM
        )

        def front_mismatch(state):
            front = state[18:27]
            return sum(1 for a, b in zip(front, target_front) if a != b)

        # Each entry is (state, path).  The ordering of the beam is guided by
        # the number of mismatched visible stickers, then path length.
        frontier = [(start_state.copy(), [])]
        seen_depth = {tuple(start_state.tolist()): 0}
        matches = {}
        nodes = 0

        for depth in range(1, max_depth + 1):
            next_frontier = []
            expansions_since_yield = 0

            for curr_state, path in frontier:
                if nodes >= node_limit or time.perf_counter() >= deadline:
                    break

                for move in self.MOVE_ORDER:
                    if path and self.INVERSES.get(path[-1]) == move:
                        continue

                    nodes += 1
                    expansions_since_yield += 1
                    next_state = self.tracker.apply_move_token(curr_state, move)
                    if next_state is None:
                        continue

                    new_path = path + [move]
                    key = tuple(next_state.tolist())

                    if self._normalize(next_state[18:27]) == target_front:
                        if key not in matches:
                            matches[key] = {
                                "cost": float(depth),
                                "path": list(new_path),
                                "state": next_state.copy(),
                            }
                            if len(matches) >= self.FINAL_MAX_MATCHES_PER_SEARCH:
                                break
                        continue

                    previous_depth = seen_depth.get(key)
                    if previous_depth is not None and previous_depth <= depth:
                        continue
                    seen_depth[key] = depth

                    # A small path-length component keeps ordering stable while
                    # the visible-front mismatch provides the main guidance.
                    heuristic = float(front_mismatch(next_state)) + (0.02 * depth)
                    next_frontier.append((heuristic, next_state, new_path))

                    # Give the main/UI thread regular opportunities to run.
                    if expansions_since_yield >= 1500:
                        time.sleep(0)
                        expansions_since_yield = 0

                if (
                    nodes >= node_limit
                    or time.perf_counter() >= deadline
                    or len(matches) >= self.FINAL_MAX_MATCHES_PER_SEARCH
                ):
                    break

            # If we found exact matches at this depth, they are the shortest
            # paths from this starting state, so there is no reason to search
            # deeper for this transition.
            if matches:
                values = list(matches.values())
                values.sort(
                    key=lambda x: (x["cost"], len(x["path"]), tuple(x["path"]))
                )
                return values

            if (
                nodes >= node_limit
                or time.perf_counter() >= deadline
                or not next_frontier
            ):
                break

            # Keep only the most promising states, while preserving a little
            # diversity across different first moves so hidden B/B' branches
            # are less likely to disappear.
            next_frontier.sort(
                key=lambda item: (
                    item[0],
                    len(item[2]),
                    item[2][0] if item[2] else "",
                    tuple(item[2]),
                )
            )

            selected = []
            first_seen = set()
            for item in next_frontier:
                first = item[2][0] if item[2] else "HOLD"
                if first not in first_seen:
                    selected.append(item)
                    first_seen.add(first)
                if len(selected) >= beam_limit:
                    break

            if len(selected) < beam_limit:
                selected_keys = {tuple(item[1].tolist()) for item in selected}
                for item in next_frontier:
                    key = tuple(item[1].tolist())
                    if key in selected_keys:
                        continue
                    selected.append(item)
                    selected_keys.add(key)
                    if len(selected) >= beam_limit:
                        break

            frontier = [(state, path) for _, state, path in selected]

            if not frontier:
                break

            time.sleep(0)

        return []

    @staticmethod
    def _merge_key(state):
        return tuple(state.tolist())

    def _prune(self, hypotheses):
        """Keep complete-state diversity while retaining cheap alternatives."""
        dedup = {}
        for h in hypotheses:
            key = self._merge_key(h["state"])
            existing = dedup.get(key)
            if existing is None or (
                h["score"], len(h["moves"]), tuple(h["moves"])
            ) < (
                existing["score"], len(existing["moves"]), tuple(existing["moves"])
            ):
                dedup[key] = h

        ordered = sorted(
            dedup.values(),
            key=lambda h: (h["score"], len(h["moves"]), tuple(h["moves"])),
        )

        selected = []
        first_move_seen = set()
        selected_state_keys = set()

        # Preserve direct/hidden branch diversity first.
        for h in ordered:
            first = h["moves"][0] if h["moves"] else "HOLD"
            state_key = self._merge_key(h["state"])
            if (
                first not in first_move_seen
                and state_key not in selected_state_keys
                and len(selected) < self.FINAL_BEAM_WIDTH
            ):
                selected.append(h)
                first_move_seen.add(first)
                selected_state_keys.add(state_key)

        for h in ordered:
            if len(selected) >= self.FINAL_BEAM_WIDTH:
                break
            state_key = self._merge_key(h["state"])
            if state_key not in selected_state_keys:
                selected.append(h)
                selected_state_keys.add(state_key)

        return selected[:self.FINAL_BEAM_WIDTH]

    @staticmethod
    def _interpolate_event_frames(start_frame, end_frame, count):
        if count <= 0:
            return []
        if start_frame is None or end_frame is None:
            return [end_frame for _ in range(count)]
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if count == 1:
            return [end_frame]
        span = max(1, end_frame - start_frame)
        frames = []
        for i in range(count):
            frac = float(i + 1) / float(count)
            frames.append(int(round(start_frame + span * frac)))
        return frames

    @classmethod
    def _extract_motion_bursts(cls, observations):
        """Return rising-edge physical-motion events from the recorded timeline."""
        bursts = []
        previous = False
        for idx, item in enumerate(observations or []):
            evidence = item.get("motion_evidence") or {}
            current = bool(evidence.get("burst", False))
            if current and not previous:
                bursts.append({
                    "obs_idx": idx,
                    "frame": item.get("frame_index"),
                    "time": item.get("observation_time"),
                })
            previous = current
        return bursts

    @classmethod
    def _build_motion_aligned_checkpoints(cls, observations, base_front):
        """
        Build state checkpoints where *physical motion* is the constraint.

        A stable colour plateau that appears without a nearby physical-motion
        burst is treated as a colour/detection excursion and is not allowed to
        create a new cube state.  This is deliberately different from the old
        deep search: the number of candidate moves for a transition comes from
        the number of physical motion bursts between its checkpoints.
        """
        runs = cls._build_runs(observations)
        bursts = cls._extract_motion_bursts(observations)
        base_key = tuple(cls._normalize(base_front))

        if not runs:
            return [], {"bursts": bursts, "stable_runs": 0, "accepted": 0, "ignored_no_motion": 0}

        # Locate the first stable known starting state.
        start_idx = None
        for i, run in enumerate(runs):
            if run["count"] >= cls.MIN_STABLE_FRAMES and run["key"] == base_key:
                start_idx = i
                break

        if start_idx is None:
            start_run = runs[0]
            current = {
                **start_run,
                "burst_count": 0,
                "reason": "INITIAL / FALLBACK START",
            }
            start_idx = 0
        else:
            start_run = runs[start_idx]
            current = {
                **start_run,
                "burst_count": 0,
                "reason": "INITIAL KNOWN FRONT",
            }

        checkpoints = [current]
        ignored_no_motion = 0

        def count_bursts(a, b):
            a = max(0, int(a))
            b = min(len(observations) - 1, int(b)) if observations else -1
            if b < a:
                return 0
            return sum(
                1
                for burst in bursts
                if a <= int(burst["obs_idx"]) <= b
            )

        for run in runs[start_idx + 1:]:
            if run["count"] < cls.MIN_STABLE_FRAMES:
                continue

            gap_bursts = count_bursts(current["last_idx"] + 1, run["first_idx"])

            # No physical event -> do not promote a different colour plateau
            # to an authoritative checkpoint.  This is the second, offline
            # defence against glare.
            if gap_bursts == 0 and run["key"] != current["key"]:
                ignored_no_motion += 1
                continue

            if gap_bursts == 0 and run["key"] == current["key"]:
                # Just more stable frames of the same state.
                current["last_idx"] = run["last_idx"]
                current["last_frame"] = run["last_frame"]
                current["last_time"] = run["last_time"]
                current["count"] += run["count"]
                continue

            cp = {
                **run,
                "burst_count": gap_bursts,
                "start_idx": current["last_idx"],
                "start_frame": current.get("last_frame"),
                "start_time": current.get("last_time"),
                "reason": (
                    "PHYSICAL-MOTION ALIGNED CHECKPOINT"
                    if run["key"] != current["key"]
                    else "PHYSICAL MOTION / SAME FRONT (HIDDEN-MOVE CANDIDATE)"
                ),
            }
            checkpoints.append(cp)
            current = cp

        stats = {
            "bursts": bursts,
            "stable_runs": len([r for r in runs if r["count"] >= cls.MIN_STABLE_FRAMES]),
            "accepted": max(0, len(checkpoints) - 1),
            "ignored_no_motion": ignored_no_motion,
        }
        return checkpoints, stats

    @classmethod
    def _live_moves_between(cls, live_move_events, start_frame, end_frame):
        if start_frame is None and end_frame is None:
            return []
        lo = -10**18 if start_frame is None else int(start_frame)
        hi = 10**18 if end_frame is None else int(end_frame)
        result = []
        for event in live_move_events or []:
            frame = event.get("frame")
            if frame is None:
                continue
            if lo <= int(frame) <= hi:
                move = event.get("move")
                if move:
                    result.append(str(move))
        return result

    @classmethod
    def _sequence_edit_distance(cls, a, b):
        a = list(a or [])
        b = list(b or [])
        prev = list(range(len(b) + 1))
        for i, av in enumerate(a, start=1):
            cur = [i]
            for j, bv in enumerate(b, start=1):
                cur.append(min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if av == bv else 1),
                ))
            prev = cur
        return prev[-1]

    def _motion_aligned_transition_candidates(
        self,
        start_state,
        target_front,
        burst_count,
        live_hint,
        deadline,
    ):
        """
        Find short move sequences constrained by observed physical motion.

        Unlike the old final search, this does NOT search arbitrary depths just
        because a front face can be reproduced that way. The physical-motion
        burst count sets the expected sequence length. The live history is only
        a prior; it cannot overrule the recorded cube observations.
        """
        start_state = start_state.copy()
        target_front = self._normalize(target_front)
        burst_count = int(max(0, burst_count))

        if burst_count == 0:
            if self._normalize(start_state[18:27]) == target_front:
                return [{"state": start_state.copy(), "path": [], "score": 0.0}]
            return []

        # Normally use exactly one move per physical burst.  Only if that has
        # no explanation do we allow one fewer or one extra move. This is the
        # critical protection against a 3-burst transition becoming a 7-move
        # colour-only solution.
        lengths = [burst_count]
        if burst_count <= self.MOTION_CORRECTION_MAX_BURSTS_PER_GAP:
            for delta in (-1, 1):
                candidate = burst_count + delta
                if 1 <= candidate <= burst_count + self.MOTION_CORRECTION_LENGTH_TOLERANCE:
                    if candidate not in lengths:
                        lengths.append(candidate)

        # If the gap is very long, still keep the search shallow enough to be
        # useful. The live event count can tell us the likely length.
        if live_hint:
            hinted_len = len(live_hint)
            if hinted_len > 0 and hinted_len <= 4 and hinted_len not in lengths:
                if abs(hinted_len - burst_count) <= self.MOTION_CORRECTION_LENGTH_TOLERANCE:
                    lengths.append(hinted_len)

        lengths = sorted(set(lengths), key=lambda n: (abs(n - burst_count), n))

        def mismatch(state):
            front = self._normalize(state[18:27])
            return sum(a != b for a, b in zip(front, target_front))

        found = []

        # HOLD is allowed only when the visible front did not change. This lets
        # harmless whole-cube repositioning coexist with hidden B/B' hypotheses.
        if self._normalize(start_state[18:27]) == target_front:
            hold_penalty = self.MOTION_CORRECTION_HOLD_PENALTY * burst_count
            live_penalty = self._sequence_edit_distance([], live_hint) * self.MOTION_CORRECTION_LIVE_PRIOR_WEIGHT
            found.append({
                "state": start_state.copy(),
                "path": [],
                "score": hold_penalty + live_penalty,
                "burst_count": burst_count,
            })

        for length in lengths:
            if time.perf_counter() >= deadline:
                break

            beam = [(start_state.copy(), [], 0.0)]
            for depth in range(length):
                if time.perf_counter() >= deadline:
                    break
                next_beam = []
                for state, path, score in beam:
                    for move in self.MOVE_ORDER:
                        if path and self.INVERSES.get(path[-1]) == move:
                            continue
                        next_state = self.tracker.apply_move_token(state, move)
                        if next_state is None:
                            continue
                        new_path = path + [move]

                        # Keep the beam directed toward the target front, but
                        # retain some diversity so B/B' branches survive.
                        h = float(mismatch(next_state))
                        prior_match = 0
                        if depth < len(live_hint) and live_hint[depth] == move:
                            prior_match = -self.MOTION_CORRECTION_LIVE_PRIOR_WEIGHT
                        elif depth < len(live_hint):
                            prior_match = 0.20

                        new_score = h * 1.0 + score + prior_match
                        next_beam.append((new_state, new_path, new_score))

                next_beam.sort(
                    key=lambda x: (
                        x[2],
                        mismatch(x[0]),
                        tuple(x[1]),
                    )
                )

                # Deduplicate by complete cube state while preserving different
                # first-move branches.
                dedup = {}
                for item in next_beam:
                    key = tuple(item[0].tolist())
                    existing = dedup.get(key)
                    if existing is None or item[2] < existing[2]:
                        dedup[key] = item

                beam = list(dedup.values())[:self.MOTION_CORRECTION_BEAM]

            if time.perf_counter() >= deadline:
                break

            for state, path, score in beam:
                if self._normalize(state[18:27]) != target_front:
                    continue
                length_penalty = abs(length - burst_count) * self.MOTION_CORRECTION_LENGTH_PENALTY
                live_penalty = self._sequence_edit_distance(path, live_hint) * self.MOTION_CORRECTION_LIVE_PRIOR_WEIGHT
                found.append({
                    "state": state.copy(),
                    "path": list(path),
                    "score": float(score + length_penalty + live_penalty),
                    "burst_count": burst_count,
                })

        # Keep only the best complete states, preserving distinct paths.
        found.sort(key=lambda x: (x["score"], len(x["path"]), tuple(x["path"])))
        result = []
        seen = set()
        for item in found:
            key = tuple(item["state"].tolist())
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= 8:
                break
        return result

    def _compress_moves_with_frames(self, moves, frames):
        """Compress only identical adjacent quarter-turns, keeping replay frames aligned."""
        result_moves = []
        result_frames = []
        for move, frame in zip(list(moves or []), list(frames or [])):
            if result_moves and result_moves[-1] == move and not str(move).endswith("2"):
                result_moves[-1] = str(move)[0] + "2"
                result_frames[-1] = frame
            else:
                result_moves.append(move)
                result_frames.append(frame)
        return result_moves, result_frames

    def analyze_motion_aligned(
        self,
        observations,
        base_state,
        base_history,
        live_history_at_stop=None,
        live_move_events=None,
        progress_callback=None,
        max_time_ms=5000.0,
    ):
        """
        Primary post-solve correction layer.

        It reconstructs the solve from the recorded physical-motion timeline,
        using the live log as a prior rather than as truth. This is intentionally
        a different layer from the old front-only deep search.
        """
        started = time.perf_counter()
        deadline = started + (float(max_time_ms) / 1000.0)
        base_state = base_state.copy()
        base_history = list(base_history or [])
        live_history_at_stop = list(live_history_at_stop or base_history)
        live_move_events = list(live_move_events or [])
        base_front = self._normalize(base_state[18:27])

        checkpoints, stats = self._build_motion_aligned_checkpoints(observations, base_front)
        if len(checkpoints) <= 1:
            return {"ok": False, "reason": "NO PHYSICAL-MOTION-ALIGNED TRANSITIONS", "analysis_ms": (time.perf_counter()-started)*1000.0}

        hypotheses = [{
            "state": base_state.copy(),
            "moves": [],
            "events": [],
            "score": 0.0,
        }]

        processed = 0

        for cp_index, checkpoint in enumerate(checkpoints[1:], start=1):
            if time.perf_counter() >= deadline:
                break

            burst_count = int(checkpoint.get("burst_count", 0))
            target_front = list(checkpoint["grid"])
            start_frame = checkpoint.get("start_frame")
            end_frame = checkpoint.get("last_frame")
            live_hint = self._live_moves_between(
                live_move_events, start_frame, end_frame
            )

            if progress_callback is not None:
                try:
                    progress_callback(
                        cp_index,
                        max(1, len(checkpoints) - 1),
                        f"Motion-aligned correction {cp_index}/{len(checkpoints)-1} (bursts={burst_count})",
                    )
                except Exception:
                    pass

            next_hypotheses = []
            for h in hypotheses:
                if time.perf_counter() >= deadline:
                    break
                candidates = self._motion_aligned_transition_candidates(
                    h["state"],
                    target_front,
                    burst_count,
                    live_hint,
                    deadline,
                )
                for cand in candidates:
                    path = list(cand["path"])
                    new_events = list(h["events"])
                    if path:
                        burst_frames = [
                            b.get("frame")
                            for b in stats.get("bursts", [])
                            if start_frame is not None and end_frame is not None
                            and int(start_frame) <= int(b.get("frame") if b.get("frame") is not None else -10**18) <= int(end_frame)
                        ]
                        burst_frames = [f for f in burst_frames if f is not None]
                        event_frames = (
                            burst_frames[:len(path)]
                            if len(burst_frames) >= len(path)
                            else self._interpolate_event_frames(start_frame, end_frame, len(path))
                        )
                        for move, frame in zip(path, event_frames):
                            new_events.append({"frame": frame, "move": move})

                    next_hypotheses.append({
                        "state": cand["state"].copy(),
                        "moves": h["moves"] + path,
                        "events": new_events,
                        "score": float(h["score"] + cand["score"]),
                    })

            if not next_hypotheses:
                # A bad colour checkpoint cannot destroy the whole solve.
                continue

            # Global beam across checkpoints keeps later evidence available to
            # resolve earlier hidden B/B' ambiguity.
            dedup = {}
            for h in next_hypotheses:
                key = tuple(h["state"].tolist())
                existing = dedup.get(key)
                if existing is None or (h["score"], len(h["moves"]), tuple(h["moves"])) < (existing["score"], len(existing["moves"]), tuple(existing["moves"])):
                    dedup[key] = h

            ordered = sorted(
                dedup.values(),
                key=lambda h: (h["score"], len(h["moves"]), tuple(h["moves"])),
            )
            hypotheses = ordered[:self.MOTION_CORRECTION_BEAM]
            processed += 1

        if not hypotheses:
            return {"ok": False, "reason": "NO MOTION-ALIGNED HYPOTHESIS SURVIVED", "analysis_ms": (time.perf_counter()-started)*1000.0}

        # Prefer a hypothesis whose complete state matches the newest recorded
        # valid front. This is a validation step, not a search target.
        final_observed_front = None
        if checkpoints:
            final_observed_front = list(checkpoints[-1]["grid"])
        else:
            for item in reversed(observations or []):
                grid = self._normalize(item.get("grid"))
                if self._valid_grid(grid):
                    final_observed_front = grid
                    break

        ranked = []
        for h in hypotheses:
            final_front = self._normalize(h["state"][18:27])
            final_mismatch = (
                0
                if final_observed_front is None
                else sum(a != b for a, b in zip(final_front, final_observed_front))
            )
            live_suffix = live_history_at_stop[len(base_history):]
            history = self.tracker.compressed_history(base_history + h["moves"])
            edit_penalty = self._sequence_edit_distance(history[len(base_history):], live_suffix)
            ranked.append((
                final_mismatch * 20.0 + h["score"] + 0.5 * edit_penalty,
                h,
                final_mismatch,
            ))

        ranked.sort(key=lambda x: (x[0], len(x[1]["moves"]), tuple(x[1]["moves"])))
        _, best, final_mismatch = ranked[0]

        final_raw_history = base_history + list(best["moves"])
        final_history = self.tracker.compressed_history(final_raw_history)
        final_state = self.tracker.state_from_history(final_history)
        if final_state is None:
            return {"ok": False, "reason": "MOTION-ALIGNED HISTORY REBUILD FAILED", "analysis_ms": (time.perf_counter()-started)*1000.0}

        if final_observed_front is not None:
            actual = self._normalize(final_state[18:27])
            if actual != final_observed_front:
                # Do not invent moves simply to force a front match. The caller
                # will retain the live result when this layer cannot validate it.
                return {
                    "ok": False,
                    "reason": f"MOTION-ALIGNED RESULT MISSES FINAL FRONT ({final_mismatch} stickers)",
                    "analysis_ms": (time.perf_counter()-started)*1000.0,
                }

        new_moves = list(best["moves"])
        raw_frames = [e.get("frame") for e in best["events"]]
        compressed_suffix, compressed_frames = self._compress_moves_with_frames(new_moves, raw_frames)
        events = [
            {"frame": frame, "move": move}
            for move, frame in zip(compressed_suffix, compressed_frames)
            if frame is not None
        ]

        # Hard sanity check: the corrected sequence should be physically
        # compatible with the number of motion events. This prevents exactly the
        # 3-bursts -> 7-moves failure the old front-only search allowed.
        total_bursts = len(stats.get("bursts", []))
        corrected_move_count = len(compressed_suffix)
        if total_bursts > 0 and corrected_move_count > total_bursts + self.MOTION_CORRECTION_MAX_TOTAL_EXTRA_MOVES:
            return {
                "ok": False,
                "reason": f"CORRECTION TOO LONG FOR PHYSICAL MOTION ({corrected_move_count} moves vs {total_bursts} bursts)",
                "analysis_ms": (time.perf_counter()-started)*1000.0,
            }

        full_live = list(live_history_at_stop)
        corrected = compressed_suffix != full_live[len(base_history):]
        return {
            "ok": True,
            "state": final_state.copy(),
            "history": list(base_history) + list(compressed_suffix),
            "new_moves": list(compressed_suffix),
            "events": events,
            "live_history_at_stop": full_live,
            "live_suffix": full_live[len(base_history):],
            "final_suffix": list(compressed_suffix),
            "corrected": bool(corrected),
            "checkpoints": checkpoints,
            "checkpoint_stats": stats,
            "hypothesis_count": len(hypotheses),
            "deep_searches": 0,
            "normal_searches": processed,
            "processed_checkpoints": processed,
            "skipped_checkpoints": max(0, len(checkpoints) - 1 - processed),
            "tail_recovered": False,
            "last_matched_checkpoint": checkpoints[-1] if checkpoints else None,
            "best_score": float(ranked[0][0]),
            "score_margin": (float(ranked[1][0] - ranked[0][0]) if len(ranked) > 1 else None),
            "analysis_ms": (time.perf_counter()-started)*1000.0,
            "time_limit_ms": float(max_time_ms),
            "time_limit_hit": time.perf_counter() >= deadline,
            "correction_method": "MOTION-BURST-ALIGNED MULTI-LAYER RECONSTRUCTION",
        }

    @classmethod
    def _temporal_mode_grid(cls, observations, start_idx, end_idx):
        """
        Build a temporally stable 3x3 grid by majority vote.

        V46 intentionally does NOT inspect motion evidence here.  A state
        observation remains valid even when the motion detector says that no
        movement happened.  Motion is not allowed to remove camera evidence.
        """
        if not observations:
            return None, 0.0, 0

        start_idx = max(0, int(start_idx))
        end_idx = min(len(observations) - 1, int(end_idx))
        if end_idx < start_idx:
            return None, 0.0, 0

        samples = []
        for idx in range(start_idx, end_idx + 1):
            grid = cls._normalize(observations[idx].get("grid"))
            if cls._valid_grid(grid):
                samples.append((idx, grid))

        if len(samples) < cls.TEMPORAL_MIN_GRID_SAMPLES:
            return None, 0.0, len(samples)

        consensus = []
        agreements = []
        for cell in range(9):
            counts = {}
            for _, grid in samples:
                value = str(grid[cell])
                counts[value] = counts.get(value, 0) + 1
            best_value, best_count = max(
                counts.items(), key=lambda item: (item[1], item[0])
            )
            consensus.append(best_value)
            agreements.append(best_count / float(len(samples)))

        return consensus, float(np.mean(agreements)), len(samples)

    @classmethod
    def _cluster_temporal_motion_events(cls, observations):
        """
        Convert motion-burst frames into physical manipulation events.

        Multiple adjacent burst frames represent ONE move. Very close burst
        edges are merged so a hand motion does not become several artificial
        cube moves.
        """
        raw = []
        for idx, item in enumerate(observations or []):
            evidence = item.get("motion_evidence") or {}
            if not bool(evidence.get("burst", False)):
                continue
            raw.append({
                "obs_idx": idx,
                "frame": item.get("frame_index"),
                "time": item.get("observation_time"),
            })

        if not raw:
            return []

        events = []
        current = [raw[0]]
        for item in raw[1:]:
            prev = current[-1]
            prev_time = prev.get("time")
            curr_time = item.get("time")
            gap_ms = 0.0
            if prev_time is not None and curr_time is not None:
                gap_ms = max(0.0, (float(curr_time) - float(prev_time)) * 1000.0)
            elif prev.get("frame") is not None and item.get("frame") is not None:
                gap_ms = max(0.0, float(int(item["frame"]) - int(prev["frame"]))) * (1000.0 / 30.0)

            if gap_ms <= cls.TEMPORAL_EVENT_MERGE_GAP_MS:
                current.append(item)
            else:
                events.append(current)
                current = [item]
        events.append(current)

        out = []
        for group in events:
            first = group[0]
            last = group[-1]
            out.append({
                "obs_indices": [g["obs_idx"] for g in group],
                "start_obs_idx": first["obs_idx"],
                "end_obs_idx": last["obs_idx"],
                "frame": first.get("frame"),
                "end_frame": last.get("frame"),
                "time": first.get("time"),
                "end_time": last.get("time"),
                "count": len(group),
            })
        return out

    @classmethod
    def _temporal_event_windows(cls, observations, event):
        """Return temporally stable before/after grids around one motion event."""
        start = int(event["start_obs_idx"])
        end = int(event["end_obs_idx"])
        before_start = max(0, start - cls.TEMPORAL_PRE_SAMPLES)
        before_end = max(0, start - 1)
        after_start = min(len(observations), end + 1)
        after_end = min(len(observations) - 1, end + cls.TEMPORAL_POST_SAMPLES)

        before_grid, before_conf, before_n = cls._temporal_mode_grid(
            observations, before_start, before_end
        )
        after_grid, after_conf, after_n = cls._temporal_mode_grid(
            observations, after_start, after_end
        )

        return {
            "before_grid": before_grid,
            "after_grid": after_grid,
            "before_confidence": float(before_conf),
            "after_confidence": float(after_conf),
            "before_samples": int(before_n),
            "after_samples": int(after_n),
            "before_start": before_start,
            "before_end": before_end,
            "after_start": after_start,
            "after_end": after_end,
        }

    @classmethod
    def _grid_mismatch(cls, state, target_grid):
        target = cls._normalize(target_grid)
        if len(target) != 9:
            return 9
        front = cls._normalize(state[18:27])
        return sum(a != b for a, b in zip(front, target))

    def _temporal_one_move_candidates(self, state, before_grid, after_grid, live_hint=None):
        """
        Candidate generator for ONE physical event.

        This is intentionally not a deep search. One motion event normally
        corresponds to one face move. Hidden B/B' branches naturally survive
        because both produce the same front when the visible front is unchanged.
        """
        candidates = []
        before_mismatch = self._grid_mismatch(state, before_grid) if before_grid else 0
        same_front = (
            before_grid is not None
            and after_grid is not None
            and tuple(self._normalize(before_grid)) == tuple(self._normalize(after_grid))
        )
        center_changed = (
            before_grid is not None
            and after_grid is not None
            and before_grid[4] != after_grid[4]
        )

        # HOLD is important for whole-cube repositioning. Prefer it when the
        # observed center changes, because a notation face move does not change
        # the face center colour.
        hold_state = state.copy()
        hold_after_mismatch = self._grid_mismatch(hold_state, after_grid) if after_grid else 0
        hold_score = (
            self.TEMPORAL_HOLD_PENALTY_WITH_SAME_FRONT
            if same_front else self.TEMPORAL_HOLD_PENALTY_WITH_CHANGED_FRONT
        )
        if center_changed:
            hold_score -= 3.0
        hold_score += self.TEMPORAL_BEFORE_MISMATCH_WEIGHT * before_mismatch
        hold_score += self.TEMPORAL_EVENT_MISMATCH_WEIGHT * hold_after_mismatch
        candidates.append({
            "state": hold_state,
            "path": [],
            "score": float(hold_score),
            "after_mismatch": int(hold_after_mismatch),
            "before_mismatch": int(before_mismatch),
            "used_extra": 0,
        })

        if center_changed:
            # A changed observed center is strong evidence that the cube itself
            # was reoriented in the camera, not that a face turn happened.
            return candidates

        for move_index, move in enumerate(self.MOVE_ORDER):
            next_state = self.tracker.apply_move_token(state, move)
            if next_state is None:
                continue
            after_mismatch = self._grid_mismatch(next_state, after_grid) if after_grid else 0
            score = (
                self.TEMPORAL_BEFORE_MISMATCH_WEIGHT * before_mismatch
                + self.TEMPORAL_EVENT_MISMATCH_WEIGHT * after_mismatch
            )
            if live_hint and move == live_hint:
                score -= self.TEMPORAL_MOVE_PRIOR_WEIGHT
            candidates.append({
                "state": next_state.copy(),
                "path": [move],
                "score": float(score),
                "after_mismatch": int(after_mismatch),
                "before_mismatch": int(before_mismatch),
                "used_extra": 0,
            })

        candidates.sort(key=lambda c: (c["score"], c["after_mismatch"], tuple(c["path"])))
        return candidates

    def _temporal_two_move_fallback(self, state, after_grid, live_hint=None):
        """Small 2-move fallback for a physically single but visually combined event."""
        candidates = []
        for first in self.MOVE_ORDER:
            s1 = self.tracker.apply_move_token(state, first)
            if s1 is None:
                continue
            for second in self.MOVE_ORDER:
                if self.INVERSES.get(first) == second:
                    continue
                s2 = self.tracker.apply_move_token(s1, second)
                if s2 is None:
                    continue
                mismatch = self._grid_mismatch(s2, after_grid) if after_grid else 9
                score = self.TEMPORAL_EVENT_MISMATCH_WEIGHT * mismatch
                if live_hint:
                    hint_parts = list(live_hint) if isinstance(live_hint, (list, tuple)) else [str(live_hint)]
                    if hint_parts and first == hint_parts[0]:
                        score -= self.TEMPORAL_MOVE_PRIOR_WEIGHT
                    if len(hint_parts) > 1 and second == hint_parts[1]:
                        score -= self.TEMPORAL_MOVE_PRIOR_WEIGHT
                score += self.TEMPORAL_EXTRA_MOVE_PENALTY
                candidates.append({
                    "state": s2.copy(),
                    "path": [first, second],
                    "score": float(score),
                    "after_mismatch": int(mismatch),
                    "before_mismatch": 0,
                    "used_extra": 1,
                })
        candidates.sort(key=lambda c: (c["score"], c["after_mismatch"], tuple(c["path"])))
        return candidates[:18]

    def analyze_temporal_global(
        self,
        observations,
        base_state,
        base_history,
        live_history_at_stop=None,
        progress_callback=None,
        live_move_events=None,
        max_time_ms=5000.0,
    ):
        """
        V39 correction layer.

        This is a temporal reconstruction problem, not a deeper front-face BFS.
        We:
          1. cluster physical motion into manipulation events;
          2. majority-vote the stable colour grids before/after each event;
          3. constrain each event to one move (or one small two-move fallback);
          4. keep a global beam so later observations can resolve earlier B/B'
             ambiguity;
          5. compare the reconstructed sequence against the final stable grid.

        More compute is spent on temporal evidence and global consistency, not
        on inventing longer sequences that merely match an endpoint.
        """
        started = time.perf_counter()
        deadline = started + float(max_time_ms) / 1000.0
        base_state = base_state.copy()
        base_history = list(base_history or [])
        live_history = list(live_history_at_stop or base_history)
        live_events = list(live_move_events or [])

        events = self._cluster_temporal_motion_events(observations)
        if not events:
            return {
                "ok": False,
                "reason": "NO TEMPORAL PHYSICAL MOTION EVENTS",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "events": [],
                "temporal_events": [],
            }

        temporal_events = []
        for event_index, event in enumerate(events):
            if time.perf_counter() >= deadline:
                break
            windows = self._temporal_event_windows(observations, event)
            enriched = dict(event)
            enriched.update(windows)
            temporal_events.append(enriched)

        if not temporal_events:
            return {
                "ok": False,
                "reason": "NO TEMPORAL WINDOWS AVAILABLE",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "events": events,
                "temporal_events": [],
            }

        # Align live move hints to physical events only by index. Extra live
        # moves are deliberately ignored rather than used to create extra events.
        live_suffix = live_history[len(base_history):]
        hypotheses = [{
            "state": base_state.copy(),
            "moves": [],
            "events": [],
            "score": 0.0,
            "extra_used": 0,
        }]

        processed = 0
        skipped = 0

        for event_index, event in enumerate(temporal_events):
            if time.perf_counter() >= deadline:
                break

            before_grid = event.get("before_grid")
            after_grid = event.get("after_grid")
            before_conf = float(event.get("before_confidence", 0.0))
            after_conf = float(event.get("after_confidence", 0.0))
            event_live_hint = live_suffix[event_index] if event_index < len(live_suffix) else None

            if progress_callback is not None:
                try:
                    progress_callback(
                        event_index + 1,
                        len(temporal_events),
                        f"Temporal correction event {event_index + 1}/{len(temporal_events)}",
                    )
                except Exception:
                    pass

            next_hypotheses = []
            for h in hypotheses:
                if time.perf_counter() >= deadline:
                    break

                one_move = self._temporal_one_move_candidates(
                    h["state"], before_grid, after_grid, event_live_hint
                )

                # Keep only plausible one-move candidates first.
                plausible = [
                    c for c in one_move
                    if c["after_mismatch"] <= self.TEMPORAL_ONE_MOVE_MISMATCH_LIMIT
                    or (not after_grid)
                    or not before_grid
                ]

                if not plausible:
                    plausible = one_move[:4]

                # If one move cannot explain the stable after-state at all,
                # allow a *single* two-move fallback for this physical event.
                # It is penalized heavily and the whole solve is allowed only one
                # such fallback event.
                if (
                    plausible
                    and min(c["after_mismatch"] for c in plausible) > self.TEMPORAL_TWO_MOVE_MISMATCH_LIMIT
                    and h["extra_used"] < self.TEMPORAL_MAX_EXTRA_MOVES
                    and time.perf_counter() < deadline
                ):
                    two_move = self._temporal_two_move_fallback(
                        h["state"], after_grid, [event_live_hint] if event_live_hint else []
                    )
                    plausible.extend(two_move[:8])

                # Keep a small amount of ambiguity — particularly B/B' when the
                # visible front does not change — but do not allow combinatorial growth.
                plausible.sort(
                    key=lambda c: (
                        c["score"],
                        c["after_mismatch"],
                        len(c["path"]),
                        tuple(c["path"]),
                    )
                )
                for cand in plausible[:8]:
                    event_quality_penalty = (1.0 - after_conf) * 2.0 + (1.0 - before_conf)
                    new_score = float(h["score"] + cand["score"] + event_quality_penalty)
                    new_moves = h["moves"] + list(cand["path"])
                    new_events = list(h["events"])

                    if cand["path"]:
                        start_frame = event.get("frame")
                        end_frame = event.get("end_frame")
                        event_frame = end_frame if end_frame is not None else start_frame
                        if len(cand["path"]) == 1:
                            new_events.append({"frame": event_frame, "move": cand["path"][0]})
                        else:
                            frames = self._interpolate_event_frames(
                                start_frame, end_frame, len(cand["path"])
                            )
                            for move, frame in zip(cand["path"], frames):
                                new_events.append({"frame": frame, "move": move})

                    next_hypotheses.append({
                        "state": cand["state"].copy(),
                        "moves": new_moves,
                        "events": new_events,
                        "score": new_score,
                        "extra_used": h["extra_used"] + int(cand.get("used_extra", 0)),
                    })

            if not next_hypotheses:
                skipped += 1
                continue

            # Global state beam: later events will determine which earlier
            # hidden-move branch survives.
            dedup = {}
            for h in next_hypotheses:
                key = tuple(h["state"].tolist())
                old = dedup.get(key)
                rank = (h["score"], h["extra_used"], len(h["moves"]), tuple(h["moves"]))
                if old is None or rank < (old["score"], old["extra_used"], len(old["moves"]), tuple(old["moves"])):
                    dedup[key] = h

            ordered = sorted(
                dedup.values(),
                key=lambda h: (h["score"], h["extra_used"], len(h["moves"]), tuple(h["moves"])),
            )
            hypotheses = ordered[:self.TEMPORAL_BEAM_WIDTH]
            processed += 1

        if not hypotheses:
            return {
                "ok": False,
                "reason": "TEMPORAL RECONSTRUCTION PRODUCED NO HYPOTHESES",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "events": events,
                "temporal_events": temporal_events,
            }

        # Build a final stable grid from the tail of the recording.
        final_grid = None
        final_conf = 0.0
        valid_indices = [
            i for i, item in enumerate(observations or [])
            if self._valid_grid(item.get("grid"))
        ]
        if valid_indices:
            tail_anchor = valid_indices[-1]
            tail_start = max(0, tail_anchor - self.TEMPORAL_POST_SAMPLES)
            final_grid, final_conf, _ = self._temporal_mode_grid(
                observations, tail_start, tail_anchor
            )

        ranked = []
        for h in hypotheses:
            final_mismatch = self._grid_mismatch(h["state"], final_grid) if final_grid else 0
            # Prefer fewer invented moves, but do not punish a correction merely
            # because it differs from the live history.
            live_edit = self._sequence_edit_distance(h["moves"], live_suffix)
            score = (
                h["score"]
                + self.TEMPORAL_FINAL_FRONT_WEIGHT * final_mismatch
                + 0.15 * live_edit
                + 0.5 * h["extra_used"]
            )
            ranked.append((score, h, final_mismatch))

        ranked.sort(
            key=lambda item: (
                item[0],
                item[2],
                item[1]["extra_used"],
                len(item[1]["moves"]),
                tuple(item[1]["moves"]),
            )
        )
        best_score, best, final_mismatch = ranked[0]

        # Require a genuinely coherent final temporal observation. If the tail
        # is unusable, don't overwrite a good live result with speculation.
        if final_grid is not None and final_mismatch > 2:
            return {
                "ok": False,
                "reason": f"TEMPORAL RESULT MISSES FINAL STABLE FRONT ({final_mismatch} stickers)",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "events": events,
                "temporal_events": temporal_events,
                "final_grid": final_grid,
                "final_confidence": final_conf,
                "hypothesis_count": len(hypotheses),
            }

        # Validate the reconstructed complete state against its own history.
        final_history = self.tracker.compressed_history(base_history + best["moves"])
        final_state = self.tracker.state_from_history(final_history)
        if final_state is None:
            return {
                "ok": False,
                "reason": "TEMPORAL HISTORY REBUILD FAILED",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "events": events,
                "temporal_events": temporal_events,
            }

        # Corrected event count should track physical event count. A single
        # two-move fallback is allowed, but arbitrary 7-move endpoint solutions are not.
        corrected_suffix = final_history[len(base_history):]
        if len(corrected_suffix) > len(temporal_events) + self.TEMPORAL_MAX_EXTRA_MOVES:
            return {
                "ok": False,
                "reason": f"TEMPORAL HISTORY TOO LONG ({len(corrected_suffix)} moves vs {len(temporal_events)} motion events)",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "events": events,
                "temporal_events": temporal_events,
                "hypothesis_count": len(hypotheses),
            }

        full_live = list(live_history)
        corrected = corrected_suffix != full_live[len(base_history):]
        return {
            "ok": True,
            "state": final_state.copy(),
            "history": list(final_history),
            "new_moves": list(corrected_suffix),
            "events": list(best["events"]),
            "live_history_at_stop": full_live,
            "live_suffix": full_live[len(base_history):],
            "final_suffix": list(corrected_suffix),
            "corrected": bool(corrected),
            "checkpoints": [],
            "checkpoint_stats": {
                "temporal_events": len(temporal_events),
                "motion_events": len(events),
                "skipped_events": skipped,
                "final_confidence": final_conf,
                "final_mismatch": final_mismatch,
            },
            "hypothesis_count": len(hypotheses),
            "deep_searches": 0,
            "normal_searches": 0,
            "processed_checkpoints": processed,
            "skipped_checkpoints": skipped,
            "tail_recovered": False,
            "last_matched_checkpoint": None,
            "best_score": float(best_score),
            "score_margin": (float(ranked[1][0] - ranked[0][0]) if len(ranked) > 1 else None),
            "analysis_ms": (time.perf_counter() - started) * 1000.0,
            "time_limit_ms": float(max_time_ms),
            "time_limit_hit": time.perf_counter() >= deadline,
            "correction_method": "TEMPORAL GLOBAL MOVE-EVENT RECONSTRUCTION",
            "temporal_events": temporal_events,
        }

    def _global_event_anchor_map(self, observations, motion_events):
        """
        Convert the recorded timeline into stable front-face anchors indexed by
        the number of physical manipulation events that have occurred.

        A stable run is attached to event-count k when k motion events have
        finished before that run begins.  This lets the final solver compare the
        predicted state after exactly k candidate moves against what the camera
        actually saw, rather than searching for an arbitrary path to one endpoint.
        """
        runs = self._build_runs(observations)
        anchors = {}
        stable_runs = []

        # Motion events are clustered rising-edge bursts.  We count events whose
        # end observation occurs before a stable colour run starts.
        for run in runs:
            if run["count"] < self.MIN_STABLE_FRAMES:
                continue

            event_count = 0
            for event in motion_events:
                # Only count an event once the physical burst has finished.
                # Using the burst START here can accidentally label a stable
                # pre-move/read-through run as the state AFTER the move.
                end_idx = int(event.get("end_obs_idx", -1))
                if end_idx < int(run["first_idx"]):
                    event_count += 1

            anchor = {
                "event_index": int(event_count),
                "grid": list(run["grid"]),
                "first_idx": int(run["first_idx"]),
                "last_idx": int(run["last_idx"]),
                "first_frame": run.get("first_frame"),
                "last_frame": run.get("last_frame"),
                "count": int(run["count"]),
                "weight": max(1.0, min(8.0, float(run["count"]) / 2.0)),
            }
            anchors.setdefault(event_count, []).append(anchor)
            stable_runs.append(anchor)

        # For each event index, retain a few distinct stable observations rather
        # than blindly taking the last run.  This is useful when a glare plateau
        # survives long enough to form a run: the optimizer can see that it
        # conflicts with the surrounding evidence instead of treating it as truth.
        for key in list(anchors.keys()):
            grouped = {}
            for anchor in anchors[key]:
                grid_key = tuple(anchor["grid"])
                old = grouped.get(grid_key)
                if old is None or anchor["weight"] > old["weight"]:
                    grouped[grid_key] = anchor
            anchors[key] = sorted(
                grouped.values(),
                key=lambda a: (-a["weight"], a["first_idx"]),
            )[:3]

        return anchors, stable_runs

    @staticmethod
    def _nearest_live_move_for_event(event, live_events, max_frame_gap=18):
        """Find the closest live-logged move near one physical event, if any."""
        event_frame = event.get("frame")
        if event_frame is None:
            return None
        best = None
        best_gap = None
        for item in live_events or []:
            frame = item.get("frame")
            move = item.get("move")
            if frame is None or move is None:
                continue
            gap = abs(int(frame) - int(event_frame))
            if gap <= int(max_frame_gap) and (best_gap is None or gap < best_gap):
                best = str(move)
                best_gap = gap
        return best

    def _anchor_mismatch_score(self, state, anchors):
        """Score a predicted state against one or more stable camera anchors."""
        if not anchors:
            return 0.0, 0

        front = self._normalize(state[18:27])
        scores = []
        for anchor in anchors:
            grid = self._normalize(anchor.get("grid"))
            if len(grid) != 9:
                continue
            mismatch = sum(a != b for a, b in zip(front, grid))
            weight = float(anchor.get("weight", 1.0))
            scores.append((mismatch * weight, mismatch))

        if not scores:
            return 0.0, 0

        # Use the best-supported stable observation at this event count.  This
        # prevents one isolated bad run from overpowering several good frames.
        best_score, best_mismatch = min(scores, key=lambda x: (x[0], x[1]))
        return float(best_score), int(best_mismatch)

    def _build_state_first_checkpoints(self, observations, base_state, motion_events=None):
        """
        V46: build checkpoints ONLY from persistent camera state changes.

        This is the core change in this version.  The motion detector is not
        used to create, split, suppress, or count move events.  Every distinct
        high-confidence stable front-face state becomes a chronological
        checkpoint.

        Example:
            A A A A -> B B B B -> C C C C

        becomes:
            BASE/A -> B -> C

        Therefore B and C are move evidence even if the motion detector missed
        both physical movements.  The transition search decides which legal
        move(s) explain A->B and B->C.
        """
        runs = self._build_runs(observations)
        checkpoints = []

        base_front = self._normalize(base_state[18:27])
        checkpoints.append({
            "grid": list(base_front),
            "first_idx": -1,
            "last_idx": -1,
            "first_frame": None,
            "last_frame": None,
            "count": 999999,
            "confidence": 1.0,
            "kind": "BASE",
        })

        # Only persistent runs are considered high-confidence state evidence.
        # No motion field is consulted anywhere in this builder.
        for run in runs:
            count = int(run.get("count", 0))
            if count < self.MIN_STABLE_FRAMES:
                continue

            grid = self._normalize(run.get("grid"))
            if not self._valid_grid(grid):
                continue

            confidence = min(1.0, max(0.35, count / float(self.MIN_STABLE_FRAMES * 2)))
            checkpoints.append({
                "grid": list(grid),
                "first_idx": int(run.get("first_idx", 0)),
                "last_idx": int(run.get("last_idx", run.get("first_idx", 0))),
                "first_frame": run.get("first_frame"),
                "last_frame": run.get("last_frame"),
                "count": count,
                "confidence": float(confidence),
                "kind": "STABLE_STATE_CHANGE",
            })

        # Chronological order is authoritative.  Consecutive identical stable
        # runs are collapsed just in case the raw timeline was fragmented.
        checkpoints.sort(key=lambda c: int(c.get("first_idx", -1)))
        compacted = []
        for cp in checkpoints:
            if not compacted:
                compacted.append(cp)
                continue

            prev = compacted[-1]
            if tuple(prev.get("grid", [])) == tuple(cp.get("grid", [])):
                # Keep the stronger representation while extending the span.
                if float(cp.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    prev.update(cp)
                else:
                    prev["last_idx"] = max(int(prev.get("last_idx", -1)), int(cp.get("last_idx", -1)))
                    prev["last_frame"] = cp.get("last_frame", prev.get("last_frame"))
                    prev["count"] = int(prev.get("count", 0)) + int(cp.get("count", 0))
                    prev["confidence"] = min(1.0, max(float(prev.get("confidence", 0.0)), float(cp.get("confidence", 0.0))))
                continue

            compacted.append(cp)

        # The first stable run often simply confirms the base state. Do not
        # manufacture a zero-move transition for it.
        while len(compacted) > 1:
            if tuple(compacted[1].get("grid", [])) == tuple(compacted[0].get("grid", [])):
                compacted.pop(1)
            else:
                break

        return compacted

    @classmethod
    def _search_state_transition_paths(
        cls,
        tracker,
        start_state,
        target_grid,
        expected_moves=0,
        max_depth=3,
        deadline=None,
        local_beam=72,
        max_results=8,
    ):
        """
        Find short legal move paths between TWO OBSERVED FRONT STATES.

        Crucially, max_depth is not tied to the motion-event count.  A missed
        motion event therefore does not make a real move impossible to recover.

        This searches quarter turns only; identical consecutive turns are later
        compressed to R2/F2/etc.  The search may therefore represent hidden
        B/B' moves whose visible front does not change.
        """
        target = [str(x) for x in (target_grid or [])]
        if len(target) != 9:
            return []

        max_depth = max(0, int(max_depth))
        expected_moves = max(0, int(expected_moves))
        results = []

        def mismatch(state):
            front = state[18:27]
            return int(sum(str(a) != str(b) for a, b in zip(front, target)))

        # Each node: state, path, depth.  Keeping a state key per depth prevents
        # loops from filling the beam while still allowing R/R' to exist as a
        # real two-move path when the surrounding evidence calls for it.
        frontier = [(start_state.copy(), [], 0)]
        seen_by_depth = {0: {tuple(start_state.tolist())}}

        def terminal_score(depth, mm):
            # Matching the observed state dominates.  Move-count agreement is
            # only a soft physical constraint because motion can miss events.
            excess = max(0, depth - expected_moves - 1)
            return float(
                18.0 * mm
                + 1.2 * abs(depth - expected_moves)
                + 1.5 * (excess ** 2)
                + 0.18 * depth
            )

        for depth in range(max_depth + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break

            if depth > 0:
                expanded = []
                for state, path, _ in frontier:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    for move in cls.MOVE_ORDER:
                        next_state = tracker.apply_move_token(state, move)
                        if next_state is None:
                            continue

                        # Do not immediately reverse a move unless we have to.
                        # It remains legal and will still be generated when it is
                        # the only/most supported route; this just keeps the beam
                        # focused on useful historical paths.
                        if path and cls.INVERSES.get(path[-1]) == move:
                            reverse_penalty = 0.55
                        else:
                            reverse_penalty = 0.0

                        key = tuple(next_state.tolist())
                        depth_seen = seen_by_depth.setdefault(depth, set())
                        if key in depth_seen:
                            continue
                        depth_seen.add(key)
                        expanded.append((next_state, path + [move], depth, reverse_penalty))

                if not expanded:
                    break

                # Beam-prune using the target front. Hidden moves that leave the
                # front unchanged naturally score well; visible moves with a bad
                # temporary front can survive because the beam is deliberately
                # wide and depth is only 3-5.
                expanded.sort(
                    key=lambda item: (
                        18.0 * mismatch(item[0])
                        + 1.8 * abs(item[2] - expected_moves)
                        + 0.18 * item[2]
                        + float(item[3]),
                        mismatch(item[0]),
                        tuple(item[1]),
                    )
                )
                frontier = [(a, b, c) for a, b, c, _ in expanded[:local_beam]]

            for state, path, d in frontier:
                mm = mismatch(state)
                results.append({
                    "state": state.copy(),
                    "path": list(path),
                    "depth": int(d),
                    "mismatch": int(mm),
                    "score": terminal_score(d, mm),
                })

        if not results:
            return []

        # Deduplicate identical complete states, preferring the path whose depth
        # fits the physical evidence.  This is state-first: path identity does
        # not matter when the complete predicted cube state is identical.
        dedup = {}
        for item in results:
            key = tuple(item["state"].tolist())
            old = dedup.get(key)
            if old is None or item["score"] < old["score"]:
                dedup[key] = item

        ranked = sorted(
            dedup.values(),
            key=lambda item: (
                item["score"],
                item["mismatch"],
                abs(item["depth"] - expected_moves),
                tuple(item["path"]),
            )
        )
        return ranked[:max_results]

    # ======================================================================
    # V47 STATE-TIMELINE RECONSTRUCTION
    # ======================================================================
    # The previous versions still reduced the video to exact 3-frame colour
    # runs before reconstruction.  That is too brittle: a move can be visible
    # for only a few frames, while a single misclassified sticker can split a
    # real plateau into several runs.  V47 smooths the observed front state in
    # time first, then treats every persistent state change as move evidence.
    # Motion is not consulted by any of the methods below.

    V47_SMOOTH_WINDOW = 5
    V47_MIN_PERSISTENCE = 2
    V47_MERGE_HAMMING = 1
    V47_MIN_STATE_CHANGE = 2
    V47_MAX_CHECKPOINTS = 140
    V47_LOCAL_BEAM = 120
    V47_GLOBAL_BEAM = 30
    V47_MAX_DEPTH_NORMAL = 3
    V47_MAX_DEPTH_RECOVERY = 4

    @classmethod
    def _grid_hamming(cls, a, b):
        aa = cls._normalize(a)
        bb = cls._normalize(b)
        if len(aa) != 9 or len(bb) != 9:
            return 9
        return int(sum(x != y for x, y in zip(aa, bb)))

    @classmethod
    def _temporal_majority_grid(cls, grids, center_grid=None):
        """Per-sticker temporal majority vote for a tiny offline smoothing window."""
        usable = [cls._normalize(g) for g in grids if cls._valid_grid(g)]
        usable = [g for g in usable if len(g) == 9]
        center = cls._normalize(center_grid) if cls._valid_grid(center_grid) else None
        if not usable:
            return center

        out = []
        for cell in range(9):
            counts = {}
            for g in usable:
                value = str(g[cell])
                counts[value] = counts.get(value, 0) + 1
            if not counts:
                out.append(center[cell] if center is not None else "Unknown")
                continue
            # Deterministic tie-break: retain the center observation when it is
            # tied; otherwise choose the most common label.
            best_count = max(counts.values())
            winners = sorted([k for k, v in counts.items() if v == best_count])
            center_value = str(center[cell]) if center is not None and cell < len(center) else None
            if center_value in winners:
                out.append(center_value)
            else:
                out.append(winners[0])
        return out

    @classmethod
    def _build_state_first_checkpoints_v47(cls, observations, base_state):
        """
        Build a dense, state-only front timeline.

        A high-confidence state is not required to be the exact same nine labels
        for three consecutive observations.  Instead we use a centered 5-frame
        majority vote per sticker, then form short persistent plateaus.  This
        lets the offline analyzer recover moves that were missed by the live
        move detector *and* recover moves whose camera state was present only a
        couple of frames.

        The motion detector is deliberately absent from this function.
        """
        valid = []
        for idx, item in enumerate(observations or []):
            grid = cls._normalize(item.get("grid"))
            if not cls._valid_grid(grid):
                continue
            valid.append({
                "obs_idx": int(idx),
                "frame": item.get("frame_index"),
                "time": item.get("observation_time"),
                "grid": list(grid),
            })

        if not valid:
            return [], {
                "raw_valid_observations": 0,
                "smoothed_observations": 0,
                "candidate_runs": 0,
                "accepted_state_changes": 0,
                "short_state_runs": 0,
                "base_match_found": False,
            }

        half = max(1, int(cls.V47_SMOOTH_WINDOW) // 2)
        smoothed = []
        n = len(valid)
        for i, rec in enumerate(valid):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            local = [x["grid"] for x in valid[a:b]]
            sg = cls._temporal_majority_grid(local, rec["grid"])
            if not cls._valid_grid(sg):
                sg = list(rec["grid"])
            smoothed.append({
                **rec,
                "grid": list(sg),
                "support": len(local),
            })

        # Run-length encode with a 1-sticker tolerance. A one-sticker change
        # is overwhelmingly more likely to be perception noise than a real
        # Rubik's Cube face turn.
        runs = []
        current = None
        for rec in smoothed:
            grid = list(rec["grid"])
            if current is None:
                current = {
                    "grid_samples": [grid],
                    "first_obs_idx": rec["obs_idx"],
                    "last_obs_idx": rec["obs_idx"],
                    "first_frame": rec.get("frame"),
                    "last_frame": rec.get("frame"),
                    "first_time": rec.get("time"),
                    "last_time": rec.get("time"),
                    "support": rec.get("support", 0),
                }
                continue

            representative = cls._temporal_majority_grid(
                current["grid_samples"], current["grid_samples"][len(current["grid_samples"]) // 2]
            )
            diff = cls._grid_hamming(representative, grid)
            if diff <= cls.V47_MERGE_HAMMING:
                current["grid_samples"].append(grid)
                current["last_obs_idx"] = rec["obs_idx"]
                current["last_frame"] = rec.get("frame")
                current["last_time"] = rec.get("time")
                current["support"] += rec.get("support", 0)
            else:
                runs.append(current)
                current = {
                    "grid_samples": [grid],
                    "first_obs_idx": rec["obs_idx"],
                    "last_obs_idx": rec["obs_idx"],
                    "first_frame": rec.get("frame"),
                    "last_frame": rec.get("frame"),
                    "first_time": rec.get("time"),
                    "last_time": rec.get("time"),
                    "support": rec.get("support", 0),
                }
        if current is not None:
            runs.append(current)

        # Convert each run to a representative grid and preserve the raw span.
        cooked = []
        for run in runs:
            rep = cls._temporal_majority_grid(run["grid_samples"], run["grid_samples"][len(run["grid_samples"]) // 2])
            count = len(run["grid_samples"])
            cooked.append({
                "grid": list(rep),
                "first_idx": int(run["first_obs_idx"]),
                "last_idx": int(run["last_obs_idx"]),
                "first_frame": run.get("first_frame"),
                "last_frame": run.get("last_frame"),
                "first_time": run.get("first_time"),
                "last_time": run.get("last_time"),
                "count": int(count),
                "support": int(run.get("support", 0)),
            })

        base_front = cls._normalize(base_state[18:27])
        base_match_index = None
        for i, run in enumerate(cooked):
            d = cls._grid_hamming(run["grid"], base_front)
            if d <= cls.V47_MERGE_HAMMING and run["count"] >= cls.V47_MIN_PERSISTENCE:
                base_match_index = i
                break

        # If the first visible state is the known base, use it directly. If it
        # occurs a little later, discard the pre-roll so perception startup does
        # not manufacture moves.
        if base_match_index is not None:
            cooked = cooked[base_match_index:]
        else:
            # No exact base checkpoint. Keep a synthetic BASE followed by the
            # observed runs, but do not call the first mismatch a move with high
            # confidence unless it persists.
            pass

        checkpoints = [{
            "grid": list(base_front),
            "first_idx": -1,
            "last_idx": -1,
            "first_frame": None,
            "last_frame": None,
            "first_time": None,
            "last_time": None,
            "count": 999999,
            "confidence": 1.0,
            "kind": "BASE",
        }]

        previous_grid = list(base_front)
        short_state_runs = 0
        accepted = 0
        for run in cooked:
            grid = list(run["grid"])
            if run["count"] < cls.V47_MIN_PERSISTENCE:
                short_state_runs += 1
                continue

            diff = cls._grid_hamming(previous_grid, grid)
            if diff < cls.V47_MIN_STATE_CHANGE:
                continue

            persistence_conf = min(1.0, 0.50 + 0.15 * max(0, run["count"] - 2))
            change_conf = min(1.0, 0.65 + 0.05 * diff)
            confidence = min(1.0, persistence_conf * change_conf)
            checkpoints.append({
                "grid": grid,
                "first_idx": int(run["first_idx"]),
                "last_idx": int(run["last_idx"]),
                "first_frame": run.get("first_frame"),
                "last_frame": run.get("last_frame"),
                "first_time": run.get("first_time"),
                "last_time": run.get("last_time"),
                "count": int(run["count"]),
                "confidence": float(confidence),
                "kind": "HIGH_CONFIDENCE_FRONT_STATE_CHANGE",
                "hamming_from_previous": int(diff),
            })
            previous_grid = list(grid)
            accepted += 1

        # Never let a pathological perception stream explode the beam. Keep
        # chronology and prefer the stronger persistent changes when capping.
        if len(checkpoints) - 1 > cls.V47_MAX_CHECKPOINTS:
            middle = checkpoints[1:]
            middle = sorted(
                middle,
                key=lambda c: (-float(c.get("confidence", 0.0)), -int(c.get("count", 0)), int(c.get("first_idx", 0))),
            )[:cls.V47_MAX_CHECKPOINTS]
            keep = [checkpoints[0]] + middle
            keep.sort(key=lambda c: int(c.get("first_idx", -1)))
            checkpoints = keep

        return checkpoints, {
            "raw_valid_observations": len(valid),
            "smoothed_observations": len(smoothed),
            "candidate_runs": len(cooked),
            "accepted_state_changes": int(accepted),
            "short_state_runs": int(short_state_runs),
            "base_match_found": base_match_index is not None,
        }

    def _get_v48_sequence_level(self, depth):
        """Return precomputed composed move permutations for one short depth."""
        depth = int(depth)
        if depth <= 0:
            return np.arange(54, dtype=np.uint8)[None, :], [()]
        if self._v47_sequence_levels is None:
            self._v47_sequence_levels = {
                0: (np.arange(54, dtype=np.uint8)[None, :], [()])
            }
        if depth in self._v47_sequence_levels:
            return self._v47_sequence_levels[depth]

        prev_perms, prev_paths = self._get_v48_sequence_level(depth - 1)
        move_perms = [np.asarray(self.tracker.permutations[m], dtype=np.uint8) for m in self.MOVE_ORDER]
        total = prev_perms.shape[0] * len(move_perms)
        perms = np.empty((total, 54), dtype=np.uint8)
        paths = []
        k = 0
        for i in range(prev_perms.shape[0]):
            base_perm = prev_perms[i]
            base_path = prev_paths[i]
            for move, mp in zip(self.MOVE_ORDER, move_perms):
                # Applying move sequence p1 then p2 maps output positions through
                # p1[p2]. This is exactly the same convention as state[perm].
                perms[k] = base_perm[mp]
                paths.append(base_path + (move,))
                k += 1
        self._v47_sequence_levels[depth] = (perms, paths)
        return perms, paths

    def _search_state_transition_paths_v47(
        self,
        start_state,
        target_grid,
        deadline=None,
        max_depth=4,
        local_beam=120,
        max_results=16,
    ):
        """
        V48 fast state-transition search.

        The old version expanded a Python beam separately for every hypothesis.
        That made 20+ checkpoints consume the entire 8 s analysis budget.  V48
        precomputes short move-sequence permutations and compares ALL short
        paths to the observed front in vectorised NumPy operations.

        Exact front matches are collected at every depth so a hidden move is not
        discarded merely because a shorter sequence also looks identical.
        """
        target = self._normalize(target_grid)
        if len(target) != 9 or start_state is None:
            return []
        target_arr = np.asarray(target, dtype=start_state.dtype)
        state = np.asarray(start_state)
        exact_candidates = []
        near_candidates = []
        max_depth = max(0, min(int(max_depth), 4))

        for depth in range(1, max_depth + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            perms, paths = self._get_v48_sequence_level(depth)
            front_indices = perms[:, 18:27]
            produced = state[front_indices]
            mismatches = np.sum(produced != target_arr[None, :], axis=1)

            exact_idx = np.flatnonzero(mismatches == 0)
            # Keep several exact alternatives from this depth. This preserves
            # different hidden-state explanations for later checkpoints.
            for idx in exact_idx[:6]:
                i = int(idx)
                exact_candidates.append({
                    "state": state[perms[i]].copy(),
                    "path": list(paths[i]),
                    "depth": depth,
                    "mismatch": 0,
                    "score": 0.03 * depth,
                })

            if len(exact_idx) == 0:
                take_n = min(4, len(mismatches))
                if take_n:
                    best_idx = np.argpartition(mismatches, take_n - 1)[:take_n]
                    for idx in best_idx:
                        i = int(idx)
                        near_candidates.append({
                            "state": state[perms[i]].copy(),
                            "path": list(paths[i]),
                            "depth": depth,
                            "mismatch": int(mismatches[i]),
                            "score": 16.0 * float(mismatches[i]) + 0.03 * depth,
                        })

        if exact_candidates:
            # Stable order: exact visible match first, then smaller path length.
            exact_candidates.sort(key=lambda x: (x["depth"], tuple(x["path"])))
            return exact_candidates[:int(max_results)]

        near_candidates.sort(key=lambda x: (x["mismatch"], x["depth"], tuple(x["path"])))
        return near_candidates[:int(max_results)]


    def analyze_state_timeline_v47(
        self,
        observations,
        base_state,
        base_history,
        live_history_at_stop=None,
        progress_callback=None,
        max_time_ms=8000.0,
    ):
        """V47: reconstruct from the front-state timeline; motion/live are validation only."""
        started = time.perf_counter()
        deadline = started + float(max_time_ms) / 1000.0
        base_state = base_state.copy()
        base_history = list(base_history or [])
        live_history = list(live_history_at_stop or base_history)

        checkpoints, timeline_stats = self._build_state_first_checkpoints_v47(observations, base_state)
        if len(checkpoints) <= 1:
            return {
                "ok": False,
                "reason": "NO HIGH-CONFIDENCE FRONT STATE CHANGES AVAILABLE",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "live_suffix": list(live_history[len(base_history):]),
                "final_suffix": [],
                "best_candidate_moves": [],
                "checkpoint_stats": {**timeline_stats, "state_changes": 0},
                "live_validation": {},
            }

        current = [{
            "state": base_state.copy(),
            "moves": [],
            "score": 0.0,
            "checkpoint_diagnostics": [],
        }]

        processed = 0
        skipped = 0
        state_transition_count = len(checkpoints) - 1
        recovery_searches = 0
        normal_searches = 0

        for checkpoint_index in range(1, len(checkpoints)):
            if time.perf_counter() >= deadline:
                skipped += len(checkpoints) - 1 - processed
                break

            cp = checkpoints[checkpoint_index]
            target = self._normalize(cp["grid"])
            state_changed = True

            if progress_callback is not None:
                try:
                    progress_callback(
                        checkpoint_index,
                        state_transition_count,
                        f"STATE CHANGE {checkpoint_index}/{state_transition_count} | searching legal transitions",
                    )
                except Exception:
                    pass

            next_global = []
            for hyp in current:
                if time.perf_counter() >= deadline:
                    break

                normal_searches += 1
                paths = self._search_state_transition_paths_v47(
                    hyp["state"],
                    target,
                    deadline=deadline,
                    max_depth=self.V47_MAX_DEPTH_NORMAL,
                    local_beam=self.V47_LOCAL_BEAM,
                    max_results=16,
                )

                exact = [x for x in paths if int(x.get("mismatch", 9)) == 0]
                if not exact and time.perf_counter() < deadline:
                    recovery_searches += 1
                    paths = self._search_state_transition_paths_v47(
                        hyp["state"],
                        target,
                        deadline=deadline,
                        max_depth=self.V47_MAX_DEPTH_RECOVERY,
                        local_beam=min(160, self.V47_LOCAL_BEAM + 40),
                        max_results=16,
                    )
                    exact = [x for x in paths if int(x.get("mismatch", 9)) == 0]

                candidates = exact if exact else paths
                if not candidates:
                    continue

                for path_info in candidates:
                    path = list(path_info.get("path", []))
                    mismatch = int(path_info.get("mismatch", 9))

                    # State match is dominant. Path length is deliberately only
                    # a tiny tie-breaker so a genuine two-move transition can
                    # survive beside an equally visible one-move transition.
                    transition_penalty = 0.08 * len(path)
                    mismatch_penalty = 20.0 * mismatch
                    cumulative = float(hyp["score"]) + mismatch_penalty + transition_penalty

                    next_global.append({
                        "state": path_info["state"].copy(),
                        "moves": hyp["moves"] + path,
                        "score": cumulative,
                        "checkpoint_diagnostics": hyp["checkpoint_diagnostics"] + [{
                            "checkpoint_index": int(checkpoint_index),
                            "frame": cp.get("first_frame"),
                            "target_grid": list(target),
                            "moves": list(path),
                            "depth": int(path_info.get("depth", len(path))),
                            "mismatch": mismatch,
                            "score": float(path_info.get("score", 0.0)),
                            "state_change": True,
                            "confidence": float(cp.get("confidence", 0.0)),
                        }],
                    })

            if not next_global:
                skipped += 1
                continue

            # Deduplicate only identical COMPLETE cube states. They have the
            # same future predictive power even if their historical path differs.
            dedup = {}
            for candidate in next_global:
                key = tuple(candidate["state"].tolist())
                old = dedup.get(key)
                if old is None or candidate["score"] < old["score"]:
                    dedup[key] = candidate

            ordered = sorted(
                dedup.values(),
                key=lambda h: (float(h["score"]), len(h["moves"]), tuple(h["moves"])),
            )
            current = ordered[:self.V47_GLOBAL_BEAM]
            processed += 1

        if not current:
            live_suffix = list(live_history[len(base_history):])
            return {
                "ok": False,
                "reason": "STATE-FIRST TIMELINE PRODUCED NO SURVIVING HYPOTHESES",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "live_suffix": live_suffix,
                "final_suffix": [],
                "best_candidate_moves": [],
                "checkpoint_stats": {
                    **timeline_stats,
                    "state_changes": state_transition_count,
                    "processed_checkpoints": processed,
                    "skipped_checkpoints": skipped,
                    "normal_searches": normal_searches,
                    "recovery_searches": recovery_searches,
                },
                "live_validation": {},
            }

        final_cp = checkpoints[-1]
        final_grid = self._normalize(final_cp["grid"])
        ranked = []
        for hyp in current:
            mm = self._grid_hamming(hyp["state"][18:27], final_grid)
            ranked.append((int(mm), float(hyp["score"]), len(hyp["moves"]), hyp))
        ranked.sort(key=lambda x: (x[0], x[1], x[2], tuple(x[3]["moves"])))

        best_mm, best_score, _, best = ranked[0]
        atomic_moves = list(best.get("moves", []))
        raw_history = base_history + atomic_moves
        displayed_history = self.tracker.compressed_history(raw_history)
        displayed_suffix = list(displayed_history[len(base_history):])
        live_suffix = list(live_history[len(base_history):])

        rebuilt = self.tracker.state_from_history(displayed_history)
        rebuilt_mm = self._grid_hamming(rebuilt[18:27], final_grid) if rebuilt is not None else 9

        # Compare the actual token histories after reconstruction. This is never
        # fed back into scoring.
        validation_rows = []
        matched = 0
        disagreed = 0
        compare_n = max(len(live_suffix), len(displayed_suffix))
        for i in range(compare_n):
            lv = live_suffix[i] if i < len(live_suffix) else None
            cv = displayed_suffix[i] if i < len(displayed_suffix) else None
            if lv is None:
                status = "STATE FOUND MISSED MOVE"
                disagreed += 1
            elif cv is None:
                status = "LIVE MOVE NOT STATE-SUPPORTED"
                disagreed += 1
            elif str(lv) == str(cv):
                status = "LIVE MATCHES STATE"
                matched += 1
            else:
                status = "LIVE DISAGREES WITH STATE"
                disagreed += 1
            validation_rows.append({
                "index": int(i + 1),
                "calculated_move": cv,
                "live_move": lv,
                "status": status,
            })

        live_validation = {
            "matched": int(matched),
            "disagreed": int(disagreed),
            "ambiguous": 0,
            "unvalidated": 0,
            "validated": int(matched + disagreed),
            "total": int(len(live_suffix)),
            "missing_live_moves": int(max(0, len(displayed_suffix) - len(live_suffix))),
            "live_extra_moves": int(max(0, len(live_suffix) - len(displayed_suffix))),
        }

        events = []
        for diag in best.get("checkpoint_diagnostics", []):
            frame = diag.get("frame")
            for move in diag.get("moves", []):
                events.append({"frame": frame if frame is not None else -1, "move": move})

        corrected = displayed_suffix != live_suffix
        result = {
            "ok": bool(rebuilt is not None and rebuilt_mm == 0 and processed == state_transition_count and best_mm == 0),
            "state": rebuilt.copy() if rebuilt is not None else best["state"].copy(),
            "history": list(displayed_history),
            "calculated_history": list(raw_history),
            "calculated_suffix": list(atomic_moves),
            "new_moves": list(displayed_suffix),
            "events": events,
            "live_history_at_stop": list(live_history),
            "live_suffix": list(live_suffix),
            "final_suffix": list(displayed_suffix),
            "corrected": bool(corrected),
            "final_exact": bool(rebuilt is not None and rebuilt_mm == 0 and best_mm == 0),
            "checkpoints": checkpoints,
            "validation_rows": validation_rows,
            "live_validation": live_validation,
            "hypothesis_count": len(current),
            "normal_searches": int(normal_searches),
            "deep_searches": int(recovery_searches),
            "processed_checkpoints": int(processed),
            "skipped_checkpoints": int(skipped),
            "best_score": float(best_score),
            "best_final_front_mismatch": int(best_mm),
            "rebuilt_final_front_mismatch": int(rebuilt_mm),
            "analysis_ms": (time.perf_counter() - started) * 1000.0,
            "time_limit_ms": float(max_time_ms),
            "time_limit_hit": time.perf_counter() >= deadline,
            "correction_method": (
                "V48 STATE TIMELINE: TEMPORALLY SMOOTHED HIGH-CONFIDENCE FRONT STATE CHANGES; "
                "MOTION NOT USED; LIVE LOG VALIDATION ONLY"
            ),
            "checkpoint_stats": {
                **timeline_stats,
                "state_changes": int(state_transition_count),
                "processed_checkpoints": int(processed),
                "skipped_checkpoints": int(skipped),
                "normal_searches": int(normal_searches),
                "recovery_searches": int(recovery_searches),
                "live_picked_count": int(len(live_suffix)),
                "state_calculated_count": int(len(atomic_moves)),
                "analysis_displayed_count": int(len(displayed_suffix)),
                "missed_live_moves_recovered": int(max(0, len(displayed_suffix) - len(live_suffix))),
                "final_front_mismatch": int(rebuilt_mm),
                "best_candidate_front_mismatch": int(best_mm),
            },
            "timeline_stats": timeline_stats,
            "best_candidate_moves": list(atomic_moves),
            "reason": "" if (rebuilt is not None and rebuilt_mm == 0 and best_mm == 0 and processed == state_transition_count) else (
                f"STATE-FIRST RECONSTRUCTION PARTIAL: {processed}/{state_transition_count} state changes processed; "
                f"best final front mismatch={best_mm}, rebuilt mismatch={rebuilt_mm}"
            ),
        }

        return result

    def analyze_global_timeline(
        self,
        observations,
        base_state,
        base_history,
        live_history_at_stop=None,
        live_move_events=None,
        progress_callback=None,
        max_time_ms=8000.0,
    ):
        """
        V46 STATE-CHANGE-FIRST RECONSTRUCTION.

        Primary evidence:
            persistent observed front-face state changes.

        Secondary evidence:
            legal transitions between complete cube states.

        Validation only:
            live move log.

        The motion detector is deliberately excluded from reconstruction. It
        cannot create, suppress, count, or limit moves. This prevents missed or
        false motion bursts from deleting real moves from the final history.
        """
        started = time.perf_counter()
        deadline = started + float(max_time_ms) / 1000.0
        base_state = base_state.copy()
        base_history = list(base_history or [])
        live_history = list(live_history_at_stop or base_history)

        # Motion is diagnostic-only in V46. Do not let it determine checkpoints.
        checkpoints = self._build_state_first_checkpoints(observations, base_state)
        motion_events = []

        if len(checkpoints) <= 1:
            return {
                "ok": False,
                "reason": "NO HIGH-CONFIDENCE FRONT STATE CHANGES AVAILABLE",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "temporal_events": [],
                "stable_anchor_count": 0,
            }

        # Keep the full chronological timeline unless it becomes pathological.
        # Prefer persistent state runs with the strongest evidence if reduction
        # is ever needed, while always preserving the first and last state.
        if len(checkpoints) > 75:
            middle = checkpoints[1:-1]
            middle = sorted(middle, key=lambda c: (float(c.get("confidence", 0.0)), int(c.get("count", 0))), reverse=True)[:73]
            keep_ids = {id(c) for c in middle}
            checkpoints = [c for c in checkpoints if id(c) in keep_ids or c is checkpoints[0] or c is checkpoints[-1]]
            checkpoints.sort(key=lambda c: int(c.get("first_idx", -1)))

        global_beam = 24
        LOCAL_RESULTS = 10
        LOCAL_BEAM = 96
        current = [{
            "state": base_state.copy(),
            "moves": [],
            "score": 0.0,
            "checkpoint_diagnostics": [],
            "move_frames": [],
        }]

        processed_checkpoints = 0
        skipped_checkpoints = 0
        gap_summaries = []

        for checkpoint_index in range(1, len(checkpoints)):
            if time.perf_counter() >= deadline:
                skipped_checkpoints = len(checkpoints) - 1 - processed_checkpoints
                break

            checkpoint = checkpoints[checkpoint_index]
            prev_checkpoint = checkpoints[checkpoint_index - 1]
            target_grid = self._normalize(checkpoint.get("grid"))
            target_idx = int(checkpoint.get("first_idx", 0))

            previous_grid = self._normalize(prev_checkpoint.get("grid"))
            state_changed = tuple(previous_grid) != tuple(target_grid)

            # A persistent state change is itself move evidence. We do not care
            # whether motion was detected. Most gaps should be one turn, while a
            # gap created by missed/invalid camera frames may need a short path.
            max_depth = 4 if state_changed else 0
            expected_moves = 0

            gap_summaries.append({
                "checkpoint_index": int(checkpoint_index),
                "expected_motion_bursts": 0,
                "max_search_depth": int(max_depth),
                "same_visible_front": bool(not state_changed),
                "target_idx": int(target_idx),
                "state_change": bool(state_changed),
            })

            if progress_callback is not None:
                try:
                    progress_callback(
                        checkpoint_index,
                        len(checkpoints) - 1,
                        f"State-change checkpoint {checkpoint_index}/{len(checkpoints)-1} "
                        f"({'MOVE' if state_changed else 'HOLD'}, depth≤{max_depth})",
                    )
                except Exception:
                    pass

            next_global = []
            for hyp in current:
                if time.perf_counter() >= deadline:
                    break

                parent_state = hyp["state"]
                local_paths = self._search_state_transition_paths(
                    self.tracker,
                    parent_state,
                    target_grid,
                    expected_moves=expected_moves,
                    max_depth=max_depth,
                    deadline=deadline,
                    local_beam=LOCAL_BEAM,
                    max_results=LOCAL_RESULTS,
                )

                for local in local_paths:
                    path = list(local["path"])
                    child_state = local["state"].copy()

                    # Camera state is primary. The run confidence scales the
                    # mismatch score, while no motion/live-log score is applied.
                    checkpoint_weight = 1.0 + 2.0 * float(checkpoint.get("confidence", 0.5))
                    cumulative = float(hyp["score"]) + checkpoint_weight * float(local["score"])

                    # A persistent changed state with a zero-length path is not
                    # acceptable. A persistent unchanged state should remain a
                    # legal HOLD.
                    if state_changed and not path:
                        cumulative += 80.0

                    next_global.append({
                        "state": child_state,
                        "moves": hyp["moves"] + path,
                        "score": cumulative,
                        "checkpoint_diagnostics": hyp["checkpoint_diagnostics"] + [{
                            "checkpoint_index": int(checkpoint_index),
                            "frame": checkpoint.get("first_frame"),
                            "target_grid": list(target_grid),
                            "expected_bursts": 0,
                            "moves": list(path),
                            "depth": int(local["depth"]),
                            "mismatch": int(local["mismatch"]),
                            "score": float(local["score"]),
                            "same_visible_front": bool(not state_changed),
                            "state_change": bool(state_changed),
                        }],
                        "move_frames": hyp["move_frames"] + [
                            checkpoint.get("first_frame") for _ in path
                        ],
                    })

            if not next_global:
                skipped_checkpoints += 1
                continue

            # Deduplicate by complete 54-sticker state. Different paths that
            # converge to the same state have identical future predictive power.
            state_dedup = {}
            for candidate in next_global:
                key = tuple(candidate["state"].tolist())
                old = state_dedup.get(key)
                if old is None or candidate["score"] < old["score"]:
                    state_dedup[key] = candidate

            ordered = sorted(
                state_dedup.values(),
                key=lambda h: (h["score"], len(h["moves"]), tuple(h["moves"])),
            )

            # Retain state diversity first, because front-only ambiguity can hide
            # different complete cube states that future checkpoints resolve.
            selected = []
            recent_move = set()
            for candidate in ordered:
                last_move = candidate["moves"][-1] if candidate["moves"] else "HOLD"
                if last_move not in recent_move and len(selected) < global_beam:
                    selected.append(candidate)
                    recent_move.add(last_move)
            for candidate in ordered:
                if len(selected) >= global_beam:
                    break
                if all(candidate is not x for x in selected):
                    selected.append(candidate)

            current = selected[:global_beam]
            processed_checkpoints += 1

        if not current:
            return {
                "ok": False,
                "reason": "STATE-CHANGE RECONSTRUCTION PRODUCED NO HYPOTHESES",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "temporal_events": [],
                "stable_anchor_count": len(checkpoints) - 1,
                "processed_checkpoints": int(processed_checkpoints),
                "skipped_checkpoints": int(skipped_checkpoints),
            }

        # The final checkpoint was already part of the chronological search.
        # Do NOT run a separate endpoint-fitting recovery search; that was one of
        # the mechanisms that could collapse a long real history into a tiny path.
        final_checkpoint = checkpoints[-1]
        final_grid = self._normalize(final_checkpoint.get("grid"))

        ranked_final = []
        for hyp in current:
            mismatch = self._grid_mismatch(hyp["state"], final_grid)
            ranked_final.append((
                int(mismatch),
                float(hyp["score"]),
                len(hyp.get("moves", [])),
                hyp,
            ))

        ranked_final.sort(key=lambda item: (item[0], item[1], item[2], tuple(item[3].get("moves", []))))
        final_mismatch, best_total_score, _, best = ranked_final[0]

        atomic_moves = list(best["moves"])
        final_raw_history = base_history + atomic_moves
        final_history = self.tracker.compressed_history(final_raw_history)
        final_state = self.tracker.state_from_history(final_history)
        if final_state is None:
            return {
                "ok": False,
                "reason": "STATE-CHANGE HISTORY REBUILD FAILED",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "temporal_events": [],
            }

        rebuilt_final_mismatch = self._grid_mismatch(final_state, final_grid)
        if rebuilt_final_mismatch != 0:
            return {
                "ok": False,
                "reason": (
                    f"BEST STATE-CHANGE SEQUENCE MISSES FINAL FRONT "
                    f"({rebuilt_final_mismatch} stickers)"
                ),
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "temporal_events": [],
                "stable_anchor_count": len(checkpoints) - 1,
                "processed_checkpoints": int(processed_checkpoints),
                "skipped_checkpoints": int(skipped_checkpoints),
                "best_score": float(best_total_score),
                "final_grid": list(final_grid),
                "final_confidence": float(final_checkpoint.get("confidence", 0.0)),
                "best_candidate_history": list(final_history),
                "best_candidate_moves": list(atomic_moves),
                "gap_summaries": gap_summaries,
            }

        # --------------------------------------------------------------
        # LIVE LOG VALIDATION — NEVER PART OF RECONSTRUCTION
        # --------------------------------------------------------------
        live_suffix = list(live_history[len(base_history):])
        final_suffix = list(final_history[len(base_history):])

        matched = 0
        disagreed = 0
        unvalidated = 0
        validation_rows = []

        max_compare = max(len(live_suffix), len(final_suffix))
        for i in range(max_compare):
            live_move = live_suffix[i] if i < len(live_suffix) else None
            calc_move = final_suffix[i] if i < len(final_suffix) else None
            if live_move is None:
                status = "STATE FOUND MISSED MOVE"
                disagreed += 1
            elif calc_move is None:
                status = "LIVE MOVE NOT STATE-SUPPORTED"
                disagreed += 1
            elif str(live_move) == str(calc_move):
                status = "LIVE MATCHES STATE"
                matched += 1
            else:
                status = "LIVE DISAGREES WITH STATE"
                disagreed += 1
            validation_rows.append({
                "index": int(i + 1),
                "calculated_move": calc_move,
                "live_move": live_move,
                "status": status,
            })

        live_validation = {
            "matched": int(matched),
            "disagreed": int(disagreed),
            "ambiguous": 0,
            "unvalidated": int(unvalidated),
            "validated": int(matched + disagreed),
            "total": int(len(live_suffix)),
            "missing_live_moves": int(max(0, len(final_suffix) - len(live_suffix))),
            "live_extra_moves": int(max(0, len(live_suffix) - len(final_suffix))),
        }

        result_events = []
        frame_iter = []
        for diag in best.get("checkpoint_diagnostics", []):
            frame_iter.extend([diag.get("frame")] * len(diag.get("moves", [])))
        for move, frame in zip(atomic_moves, frame_iter):
            result_events.append({
                "frame": frame if frame is not None else -1,
                "move": move,
            })

        corrected = final_suffix != live_suffix
        exact_final = rebuilt_final_mismatch == 0

        return {
            "ok": True,
            "state": final_state.copy(),
            "history": list(final_history),
            "new_moves": list(final_suffix),
            "events": result_events,
            "live_history_at_stop": list(live_history),
            "live_suffix": list(live_suffix),
            "final_suffix": list(final_suffix),
            "corrected": bool(corrected),
            "final_exact": bool(exact_final),
            "checkpoints": checkpoints,
            "checkpoint_stats": {
                "physical_events": 0,
                "stable_anchors": max(0, len(checkpoints) - 1),
                "final_front_mismatch": 0,
                "final_front_confidence": float(final_checkpoint.get("confidence", 0.0)),
                "processed_checkpoints": int(processed_checkpoints),
                "skipped_checkpoints": int(skipped_checkpoints),
                "live_validation": live_validation,
                "recovered_moves": int(len(final_suffix)),
                "missed_live_moves_recovered": int(max(0, len(final_suffix) - len(live_suffix))),
                "final_recovery_depth": 0,
            },
            "live_validation": live_validation,
            "validation_rows": validation_rows,
            "hypothesis_count": len(current),
            "deep_searches": 0,
            "normal_searches": int(processed_checkpoints),
            "processed_checkpoints": int(processed_checkpoints),
            "skipped_checkpoints": int(skipped_checkpoints),
            "tail_recovered": False,
            "last_matched_checkpoint": checkpoints[min(len(checkpoints) - 1, processed_checkpoints)] if checkpoints else None,
            "best_score": float(best_total_score),
            "score_margin": (
                float(ranked_final[1][1] - ranked_final[0][1])
                if len(ranked_final) > 1 else None
            ),
            "analysis_ms": (time.perf_counter() - started) * 1000.0,
            "time_limit_ms": float(max_time_ms),
            "time_limit_hit": time.perf_counter() >= deadline,
            "correction_method": (
                "V46 STATE-CHANGE-FIRST: HIGH-CONFIDENCE FRONT STATE CHANGES PRIMARY; "
                "MOTION IGNORED FOR RECONSTRUCTION; LIVE LOG VALIDATION ONLY"
            ),
            "temporal_events": [],
            "gap_summaries": gap_summaries,
            "best_candidate_moves": list(atomic_moves),
        }

    def analyze(self, observations, base_state, base_history, live_history_at_stop=None, progress_callback=None, live_move_events=None):
        """V47 authoritative final analysis: state timeline first, live log only for validation."""
        started = time.perf_counter()
        try:
            result = self.analyze_state_timeline_v47(
                observations,
                base_state,
                base_history,
                live_history_at_stop=live_history_at_stop,
                progress_callback=progress_callback,
                max_time_ms=min(8000.0, self.FINAL_ANALYSIS_MAX_TIME_MS),
            )

            if result.get("ok"):
                print("🧠 [FINAL ANALYSIS] V47 state-timeline reconstruction completed.")
            else:
                print("⚠️ [FINAL ANALYSIS] V47 state-timeline reconstruction was partial / not validated.")
                print("   Reason:", result.get("reason", "unknown"))
            print(
                "   COUNTS | "
                f"live={len(result.get('live_suffix', []))} | "
                f"calculated={len(result.get('calculated_suffix', []))} | "
                f"displayed={len(result.get('final_suffix', []))} | "
                f"state_changes={result.get('checkpoint_stats', {}).get('state_changes', 0)}"
            )
            print(
                "   State-vs-live validation: "
                + str(result.get("live_validation", {}))
            )
            return result
        except Exception as exc:
            print("⚠️ [FINAL ANALYSIS] V47 exception:", exc)
            traceback.print_exc()
            return {
                "ok": False,
                "reason": f"V47 STATE-TIMELINE EXCEPTION: {exc}",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
                "live_suffix": list((live_history_at_stop or base_history)[len(base_history):]),
                "calculated_suffix": [],
                "final_suffix": [],
                "checkpoint_stats": {},
                "live_validation": {},
            }


class SlowMoBufferProcessor:
    def __init__(self, cube_tracker, fps=120.0):
        self.cube_tracker = cube_tracker
        self.fps = fps
        self.is_recording = False
        self.analysis_in_progress = False
        self.recorded_frames = []
        self.recorded_observations = []
        self.lock = threading.Lock()
        self.temp_file_path = None
        self.replay_queue = queue.Queue()
        self.final_result_queue = queue.Queue(maxsize=2)
        self.analysis_status = {
            "active": False,
            "status": "IDLE",
            "progress": 0.0,
            "processed": 0,
            "total": 0,
            "message": "",
            "phase": "IDLE",
            "started_at": None,
            "elapsed_s": 0.0,
            "yolo_ok": 0,
            "grid_valid": 0,
            "yolo_retry": 0,
            "motion_bursts": 0,
        }
        self.base_state_snapshot = None
        self.base_history_snapshot = []
        self.live_history_at_stop = []
        self.live_move_events_at_stop = []
        # Keep the completed solve available for an explicit E-key reanalysis.
        # This is intentionally separate from the live tracker and replay UI.
        self.last_completed_frames = []
        self.last_completed_observations = []
        self.last_completed_source_fps = 30.0
        # Fresh perception artifacts from the explicit E-key pass.  These are
        # small (grid + quad only) and let the replay window display the
        # reanalyzed colour/grid result while the background pass is running.
        self.reanalysis_visuals = {}
        self.reanalysis_visuals_lock = threading.Lock()
        self.final_analyzer = FinalSolveAnalyzer(cube_tracker)

    def start_recording(self, initial_state=None, initial_history=None):
        with self.lock:
            self.recorded_frames = []
            self.recorded_observations = []
            self.base_state_snapshot = (
                initial_state.copy()
                if initial_state is not None
                else self.cube_tracker.state.copy()
            )
            self.base_history_snapshot = list(
                initial_history
                if initial_history is not None
                else self.cube_tracker.move_history
            )
            self.live_history_at_stop = []
            self.live_move_events_at_stop = []
            self.last_completed_frames = []
            self.last_completed_observations = []
            with self.reanalysis_visuals_lock:
                self.reanalysis_visuals.clear()
            self.is_recording = True
            self.analysis_in_progress = False
            self.analysis_status = {
                "active": False,
                "status": "RECORDING",
                "progress": 0.0,
                "processed": 0,
                "total": 0,
                "message": "",
                "phase": "RECORDING",
                "started_at": None,
                "elapsed_s": 0.0,
                "yolo_ok": 0,
                "grid_valid": 0,
                "yolo_retry": 0,
                "motion_bursts": 0,
            }
            print("\n🔴 [SLOW-MO]: Recording started (frames + observation timeline)...")

    def add_frame(self, frame):
        with self.lock:
            if self.is_recording:
                self.recorded_frames.append(frame.copy())

    def add_observation(self, frame_index, grid, motion_evidence):
        """Store lightweight per-frame evidence for the final offline pass."""
        with self.lock:
            if not self.is_recording:
                return
            self.recorded_observations.append({
                "frame_index": frame_index,
                "observation_time": time.perf_counter(),
                "grid": copy.deepcopy(grid) if grid is not None else None,
                "motion_evidence": copy.deepcopy(motion_evidence or {}),
            })

    def stop_and_process(
        self,
        model=None,
        extract_grid_fn=None,
        live_move_events=None,
        auto_analyze=True,
        source_fps=30.0,
    ):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.analysis_in_progress = True
            frames_to_process = list(self.recorded_frames)
            observations_to_process = list(self.recorded_observations)
            self.recorded_frames = []
            self.recorded_observations = []
            # Keep a copy for the explicit E-key video reanalysis pass.
            self.last_completed_frames = [f.copy() for f in frames_to_process]
            self.last_completed_observations = copy.deepcopy(observations_to_process)
            self.last_completed_source_fps = float(source_fps or 30.0)
            base_state = (
                self.base_state_snapshot.copy()
                if self.base_state_snapshot is not None
                else self.cube_tracker.state.copy()
            )
            base_history = list(self.base_history_snapshot)
            self.live_history_at_stop = list(self.cube_tracker.move_history)
            self.live_move_events_at_stop = list(live_move_events or [])
            live_history_at_stop = list(self.live_history_at_stop)
            live_move_events_at_stop = list(self.live_move_events_at_stop)

        total_frames = len(frames_to_process)

        # Make the replay available IMMEDIATELY after SPACE.  The old V31
        # implementation only populated replay_queue after the deep analysis
        # had finished, which made the application look frozen while the search
        # was running.
        for frame in frames_to_process:
            self.replay_queue.put(frame.copy())
        self.replay_queue.put(None)

        self.analysis_status = {
            "active": True,
            "status": "FINAL ANALYSIS RUNNING",
            "progress": 0.0,
            "processed": 0,
            "total": 0,
            "message": "Preparing recorded observations...",
        }

        print(
            f"\n⏹️ [RECORDING STOPPED]: {total_frames} frames, "
            f"{len(observations_to_process)} live observations captured."
        )

        if not auto_analyze:
            # This recording is finished and ready for the explicit E-key pass.
            # Do NOT leave analysis_in_progress=True here, otherwise the E-key
            # handler will believe an analysis is already running and the user
            # will see a stale 100%/VIDEO READY state forever.
            self.analysis_in_progress = False
            print("🎬 [VIDEO READY]: Press 'e' to run FULL video reanalysis from the recording.")
            self.analysis_status = {
                "active": False,
                "status": "VIDEO READY — PRESS E",
                "progress": 0.0,
                "processed": 0,
                "total": max(0, total_frames),
                "message": "Press E to fully re-run YOLO, grid extraction and state-timeline reconstruction; motion is diagnostic only.",
                "phase": "READY",
                "started_at": None,
                "elapsed_s": 0.0,
                "yolo_ok": 0,
                "grid_valid": 0,
                "yolo_retry": 0,
                "motion_bursts": 0,
            }
            return

        print("🧠 [FINAL ANALYSIS]: Running fast bounded solve reconstruction (hard limit 50 s)...")

        thread = threading.Thread(
            target=self._analyze_frames_async,
            args=(
                frames_to_process,
                observations_to_process,
                base_state,
                base_history,
                live_history_at_stop,
                live_move_events_at_stop,
            ),
            daemon=True,
        )
        thread.start()

    def reanalyze_recorded_video(self, model, extract_grid_fn, source_fps=None):
        """
        Explicit second-pass video analysis, triggered by the E key.

        Unlike the live log, this pass re-reads the recorded frames, reruns YOLO
        + colour extraction, and rebuilds the observation timeline with a fresh
        motion detector. It does NOT reuse the live move list as the answer.
        The saved live history is only used afterward as a comparison target.
        """
        with self.lock:
            if self.is_recording:
                print("⚠️ [VIDEO REANALYSIS]: Stop recording first.")
                return False
            if self.analysis_in_progress:
                print("⏳ [VIDEO REANALYSIS]: Analysis already running.")
                return False

            frames = [f.copy() for f in self.last_completed_frames]
            if not frames:
                print("⚠️ [VIDEO REANALYSIS]: No completed recording is available. Press SPACE to record first.")
                return False

            base_state = (
                self.base_state_snapshot.copy()
                if self.base_state_snapshot is not None
                else self.cube_tracker.state.copy()
            )
            base_history = list(self.base_history_snapshot)
            live_history_at_stop = list(self.live_history_at_stop or self.cube_tracker.move_history)
            live_move_events_at_stop = copy.deepcopy(self.live_move_events_at_stop)
            prior_observations = copy.deepcopy(self.last_completed_observations)
            fps = float(source_fps or self.last_completed_source_fps or 30.0)
            self.analysis_in_progress = True
            with self.reanalysis_visuals_lock:
                self.reanalysis_visuals.clear()

        reanalysis_started_at = time.perf_counter()
        self.analysis_status = {
            "active": True,
            "status": "VIDEO REANALYSIS RUNNING",
            "progress": 0.0,
            "processed": 0,
            "total": len(frames),
            "message": "Selecting temporal correction windows...",
            "phase": "PREPARING",
            "started_at": reanalysis_started_at,
            "elapsed_s": 0.0,
            "yolo_ok": 0,
            "grid_valid": 0,
            "yolo_retry": 0,
            "motion_bursts": 0,
        }
        print("\n🔁 [VIDEO REANALYSIS]: Fresh temporal correction pass over recorded frames...")

        thread = threading.Thread(
            target=self._reanalyze_video_async,
            args=(
                frames,
                prior_observations,
                base_state,
                base_history,
                live_history_at_stop,
                live_move_events_at_stop,
                model,
                extract_grid_fn,
                fps,
            ),
            daemon=True,
        )
        thread.start()
        return True

    @staticmethod
    def _contiguous_ranges(indices):
        indices = sorted(set(int(i) for i in indices if i is not None and int(i) >= 0))
        if not indices:
            return []
        ranges = []
        start = prev = indices[0]
        for value in indices[1:]:
            if value == prev + 1:
                prev = value
            else:
                ranges.append((start, prev))
                start = prev = value
        ranges.append((start, prev))
        return ranges

    def _select_video_reanalysis_indices(self, frame_count, prior_observations):
        """
        V41 explicit E-pass: reanalyze EVERY recorded frame.

        The purpose of E is to actually revisit the footage, not to feed the
        same sparse observation list back into the live engine. Full-frame
        re-perception gives us temporal colour evidence and lets transformed
        frame retries recover frames where YOLO/grid extraction failed.
        """
        n = max(0, int(frame_count))
        return list(range(n))

    def _reanalyze_video_async(
        self,
        frames,
        prior_observations,
        base_state,
        base_history,
        live_history_at_stop,
        live_move_events_at_stop,
        model,
        extract_grid_fn,
        source_fps,
    ):
        started = time.perf_counter()
        observations = []
        selected = self._select_video_reanalysis_indices(len(frames), prior_observations)
        total_selected = len(selected)
        # Hard E-pass budget: leave ~8 seconds for the temporal correction layer
        # so the complete post-solve process stays below one minute.
        perception_deadline = started + 42.0
        perception_time_limit_hit = False

        yolo_ok_count = 0
        grid_valid_count = 0
        yolo_retry_count = 0
        motion_burst_count = 0

        print(f"🎥 [VIDEO REANALYSIS]: Reprocessing {total_selected}/{len(frames)} recorded frames.")

        # Fresh detector: no state from the live motion gate is reused.
        motion_gate = MotionEvidenceGate()
        previous_idx = None

        def _set_status(progress, processed, total, message, phase):
            elapsed = time.perf_counter() - started
            self.analysis_status = {
                "active": True,
                "status": "VIDEO REANALYSIS RUNNING",
                "progress": float(progress),
                "processed": int(processed),
                "total": int(total),
                "message": str(message),
                "phase": str(phase),
                "started_at": started,
                "elapsed_s": float(elapsed),
                "yolo_ok": int(yolo_ok_count),
                "grid_valid": int(grid_valid_count),
                "yolo_retry": int(yolo_retry_count),
                "motion_bursts": int(motion_burst_count),
            }

        def _make_frame_variants(frame):
            """
            Return a small, targeted set of image variants.  We only use these
            when the normal perception path fails or looks suspicious, so the
            common case stays fast.

            Variants: original, CLAHE contrast, gray-world white balance,
            white-balance + CLAHE, and mild gamma correction.
            """
            variants = [("ORIGINAL", frame)]

            try:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l2 = clahe.apply(l)
                clahe_frame = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
                variants.append(("CLAHE", clahe_frame))
            except Exception:
                pass

            try:
                f = frame.astype(np.float32)
                means = f.reshape(-1, 3).mean(axis=0)
                target = float(np.mean(means))
                scales = target / np.maximum(means, 1e-6)
                wb = np.clip(f * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
                variants.append(("WHITE_BALANCE", wb))

                wb_lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
                wl, wa, wb2 = cv2.split(wb_lab)
                wl = clahe.apply(wl)
                wb_clahe = cv2.cvtColor(cv2.merge([wl, wa, wb2]), cv2.COLOR_LAB2BGR)
                variants.append(("WHITE_BALANCE_CLAHE", wb_clahe))
            except Exception:
                pass

            try:
                gamma = 0.85
                table = np.array([min(255, max(0, int(((i / 255.0) ** gamma) * 255.0))) for i in range(256)], dtype=np.uint8)
                variants.append(("GAMMA", cv2.LUT(frame, table)))
            except Exception:
                pass

            return variants

        def _grid_from_quad(source_frame, quad_corners):
            if quad_corners is None:
                return None, None, None, None

            h_frame, w_frame = source_frame.shape[:2]
            raw_corners = quad_corners.copy()
            raw_corners[:, 0] = w_frame - raw_corners[:, 0]
            ordered = MotionEvidenceGate._order_points(raw_corners)
            grid, flat_face, raw_lab = extract_grid_fn(source_frame, ordered)
            return grid, flat_face, raw_lab, ordered

        def _quad_perturbations(ordered_quad):
            """Try slightly inset/outset perspective crops when the YOLO quad is usable but the grid is poor."""
            if ordered_quad is None:
                return []
            center = np.mean(ordered_quad, axis=0)
            out = []
            for name, factor in (("INSET", 0.965), ("OUTSET", 1.035)):
                q = center + (ordered_quad - center) * factor
                out.append((name, q.astype(np.float32)))
            return out

        def _grid_diff(g1, g2):
            a = [str(x) for x in (g1 or [])]
            b = [str(x) for x in (g2 or [])]
            if len(a) != 9 or len(b) != 9:
                return 9
            return sum(x != y for x, y in zip(a, b))

        def _candidate_score(grid, previous_grid, motion_evidence, variant_name):
            if not validate_face_grid(grid):
                return 10_000.0
            score = 0.0
            # V48: DO NOT penalise ordinary sticker changes relative to the
            # previous frame. A real move is supposed to change the front. Only
            # the centre sticker is used as a hard temporal sanity check because
            # the camera remains on the same cube face during a recording.
            if previous_grid is not None and len(previous_grid) == 9 and len(grid) == 9:
                if str(grid[4]) != str(previous_grid[4]):
                    score += 30.0
            # Keep the original perception path preferred when both candidates
            # are equally plausible; transformed variants are a repair fallback,
            # never a reason to suppress a true state change.
            if variant_name == "ORIGINAL":
                score += 0.00
            elif variant_name == "WHITE_BALANCE":
                score += 0.02
            elif variant_name == "CLAHE":
                score += 0.03
            else:
                score += 0.04
            return score

        def _detect_quad_with_retry(display_frame):
            """Fresh YOLO pass, with one lower-confidence retry only when needed."""
            nonlocal yolo_retry_count

            def _run(conf):
                results = model(display_frame, conf=conf, verbose=False)[0]
                if results.masks is None or len(results.masks) == 0:
                    return None

                masks = results.masks.data.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                if len(boxes) == 0:
                    return None

                best_idx = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
                h_frame, w_frame = display_frame.shape[:2]
                mask_resized = cv2.resize(masks[best_idx], (w_frame, h_frame))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    return None

                cnt = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(cnt))
                frame_area = float(w_frame * h_frame)
                if area < max(1000.0, 0.03 * frame_area):
                    return None

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                pts = (
                    approx.reshape(4, 2)
                    if len(approx) == 4
                    else cv2.boxPoints(cv2.minAreaRect(cnt))
                )
                if pts.shape != (4, 2):
                    return None

                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect = np.zeros((4, 2), dtype="float32")
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                return rect

            quad = _run(0.40)
            if quad is not None:
                return quad, False

            yolo_retry_count += 1
            quad = _run(0.25)
            return quad, True

        try:
            for pos, frame_idx in enumerate(selected, start=1):
                if not self.analysis_in_progress:
                    break
                if time.perf_counter() >= perception_deadline:
                    perception_time_limit_hit = True
                    print(f"⚠️ [VIDEO REANALYSIS]: Perception budget reached at {pos-1}/{total_selected} frames; moving to temporal correction.")
                    break

                # Preserve consecutive frames for the fresh motion detector.
                if previous_idx is None or int(frame_idx) != int(previous_idx) + 1:
                    motion_gate = MotionEvidenceGate()

                frame = frames[int(frame_idx)]

                # -----------------------------------------------------------------
                # FRESH FULL-FRAME PERCEPTION
                # -----------------------------------------------------------------
                # The explicit E pass is intentionally independent of the live
                # detection result.  It revisits every frame from scratch.
                # When the ordinary image cannot produce a usable grid, or a
                # stationary grid suddenly disagrees with the previous fresh
                # grid, we alter the image and rerun YOLO/grid extraction.
                # -----------------------------------------------------------------
                display_frame = cv2.flip(frame, 1)

                motion_evidence = None
                quad_corners = None
                current_grid = None
                selected_variant = "NONE"
                selected_flat = None
                selected_lab = None
                selected_rect = None

                # Keep the motion detector continuous across every frame now
                # that the E pass processes the complete video sequentially.
                # (Do not reset it for sampling gaps because there are no gaps.)
                if previous_idx is not None and int(frame_idx) != int(previous_idx) + 1:
                    motion_gate = MotionEvidenceGate()

                # First attempt: original frame.
                quad_corners, retried = _detect_quad_with_retry(display_frame)
                if quad_corners is not None:
                    yolo_ok_count += 1

                motion_evidence = motion_gate.process_frame(display_frame, quad_corners)
                if bool((motion_evidence or {}).get("burst", False)):
                    motion_burst_count += 1

                current_grid, selected_flat, selected_lab, selected_rect = _grid_from_quad(
                    frame, quad_corners
                )
                if validate_face_grid(current_grid):
                    selected_variant = "ORIGINAL"

                previous_fresh_grid = None
                with self.reanalysis_visuals_lock:
                    prior_vis = self.reanalysis_visuals.get(int(frame_idx) - 1)
                    if prior_vis is not None:
                        previous_fresh_grid = copy.deepcopy(prior_vis.get("grid"))

                # A valid grid can still be suspicious: if the cube is not
                # physically moving but several stickers abruptly changed,
                # revisit the image rather than accepting the first classification.
                suspicious_stationary_grid = bool(
                    validate_face_grid(current_grid)
                    and previous_fresh_grid is not None
                    and _grid_diff(current_grid, previous_fresh_grid) >= 3
                )

                needs_visual_retry = (not validate_face_grid(current_grid)) or suspicious_stationary_grid
                candidates = []
                if validate_face_grid(current_grid):
                    candidates.append((
                        _candidate_score(current_grid, previous_fresh_grid, motion_evidence, "ORIGINAL"),
                        "ORIGINAL", current_grid, selected_flat, selected_lab, selected_rect, quad_corners
                    ))

                if needs_visual_retry:
                    for variant_name, variant_frame in _make_frame_variants(frame):
                        if variant_name == "ORIGINAL":
                            continue

                        variant_display = cv2.flip(variant_frame, 1)
                        v_quad, v_retried = _detect_quad_with_retry(variant_display)
                        if v_quad is None:
                            continue
                        yolo_ok_count += 1
                        if v_retried:
                            retried = True

                        v_motion = motion_gate.process_frame(variant_display, v_quad)
                        # Do NOT use a second motion result to decide events.
                        # It is only being evaluated for this alternate image;
                        # the original-frame motion evidence remains authoritative.

                        v_grid, v_flat, v_lab, v_rect = _grid_from_quad(variant_frame, v_quad)
                        if not validate_face_grid(v_grid):
                            # Try the same YOLO quad with small perspective
                            # perturbations before abandoning the variant.
                            ordered_v = MotionEvidenceGate._order_points(
                                v_rect if v_rect is not None else np.asarray(v_quad)
                            )
                            for pert_name, pert_rect in _quad_perturbations(ordered_v):
                                try:
                                    p_grid, p_flat, p_lab = extract_grid_fn(variant_frame, pert_rect)
                                except Exception:
                                    continue
                                if validate_face_grid(p_grid):
                                    v_grid, v_flat, v_lab, v_rect = p_grid, p_flat, p_lab, pert_rect
                                    variant_name = variant_name + "+" + pert_name
                                    break

                        if validate_face_grid(v_grid):
                            candidates.append((
                                _candidate_score(v_grid, previous_fresh_grid, motion_evidence, variant_name),
                                variant_name, v_grid, v_flat, v_lab, v_rect, v_quad
                            ))

                    if not candidates:
                        # Final attempt: slight perspective perturbation of the
                        # original YOLO quadrilateral, even if the original grid
                        # was merely suspicious rather than invalid.
                        if selected_rect is not None:
                            for pert_name, pert_rect in _quad_perturbations(selected_rect):
                                try:
                                    p_grid, p_flat, p_lab = extract_grid_fn(frame, pert_rect)
                                except Exception:
                                    continue
                                if validate_face_grid(p_grid):
                                    candidates.append((
                                        _candidate_score(p_grid, previous_fresh_grid, motion_evidence, pert_name),
                                        pert_name, p_grid, p_flat, p_lab, pert_rect, quad_corners
                                    ))

                if candidates:
                    candidates.sort(key=lambda item: (float(item[0]), str(item[1])))
                    _, selected_variant, current_grid, selected_flat, selected_lab, selected_rect, quad_corners = candidates[0]
                else:
                    current_grid = None

                if validate_face_grid(current_grid):
                    grid_valid_count += 1

                # Cache the fresh visual evidence for the replay window so the
                # user can inspect exactly what the E-pass saw on each frame.
                with self.reanalysis_visuals_lock:
                    self.reanalysis_visuals[int(frame_idx)] = {
                        "grid": copy.deepcopy(current_grid) if validate_face_grid(current_grid) else None,
                        "quad": copy.deepcopy(quad_corners) if quad_corners is not None else None,
                        "variant": str(selected_variant),
                        "yolo_ok": bool(quad_corners is not None),
                    }
                    # Bound memory if an unusually long recording is used.
                    if len(self.reanalysis_visuals) > 6000:
                        oldest = min(self.reanalysis_visuals)
                        del self.reanalysis_visuals[oldest]

                # Visual status is updated for EVERY frame, not just selected
                # checkpoints, so progress cannot jump immediately to 100%.
                previous_idx = int(frame_idx)

                observations.append({
                    "frame_index": int(frame_idx),
                    "observation_time": float(frame_idx) / max(1.0, float(source_fps)),
                    "grid": copy.deepcopy(current_grid) if validate_face_grid(current_grid) else None,
                    "motion_evidence": copy.deepcopy(motion_evidence),
                    "source": "RAW_VIDEO_REANALYSIS",
                    "yolo_ok": bool(quad_corners is not None),
                    "yolo_retried": bool(retried),
                })

                previous_idx = int(frame_idx)
                _set_status(
                    (float(pos) / max(1, total_selected)) * 55.0,
                    pos,
                    total_selected,
                    f"YOLO + grid pass {pos}/{total_selected}",
                    "YOLO / GRID REANALYSIS",
                )

            if not observations:
                raise RuntimeError("VIDEO REANALYSIS PRODUCED NO OBSERVATIONS")

            perception_message = (
                f"Fresh YOLO/grid pass complete ({len(observations)}/{total_selected} frames)"
                + (" — perception time budget reached" if perception_time_limit_hit else "")
            )
            _set_status(55.0, len(observations), total_selected, perception_message, "TEMPORAL CORRECTION")

            def _analysis_progress(processed, total, message):
                # Final reconstruction occupies the remaining 45% of the UI bar.
                elapsed = time.perf_counter() - started
                self.analysis_status = {
                    "active": True,
                    "status": "VIDEO REANALYSIS RUNNING",
                    "progress": 55.0 + (float(processed) / max(1, total)) * 45.0,
                    "processed": int(processed),
                    "total": int(total),
                    "message": str(message),
                    "phase": "TEMPORAL CORRECTION",
                    "started_at": started,
                    "elapsed_s": float(elapsed),
                    "yolo_ok": int(yolo_ok_count),
                    "grid_valid": int(grid_valid_count),
                    "yolo_retry": int(yolo_retry_count),
                    "motion_bursts": int(motion_burst_count),
                }

            result = self.final_analyzer.analyze(
                observations,
                base_state,
                base_history,
                live_history_at_stop=live_history_at_stop,
                progress_callback=_analysis_progress,
                live_move_events=live_move_events_at_stop,
            )

            result["reanalysis"] = True
            result["raw_video_observations"] = len(observations)
            result["raw_video_frames_selected"] = len(selected)
            result["yolo_ok_count"] = yolo_ok_count
            result["grid_valid_count"] = grid_valid_count
            result["yolo_retry_count"] = yolo_retry_count
            result["motion_burst_count"] = motion_burst_count
            result["perception_time_limit_hit"] = bool(perception_time_limit_hit)
            result["analysis_ms"] = (time.perf_counter() - started) * 1000.0

            try:
                self.final_result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.final_result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.final_result_queue.put_nowait(result)

            if result.get("ok"):
                self.analysis_status = {
                    "active": True,
                    "status": "VIDEO REANALYSIS COMPLETE",
                    "progress": 100.0,
                    "processed": len(observations),
                    "total": len(observations),
                    "message": f"Temporal correction result ready. Fresh perception cache: {len(self.reanalysis_visuals)} frames.",
                    "phase": "COMPLETE",
                    "started_at": started,
                    "elapsed_s": float(time.perf_counter() - started),
                    "yolo_ok": int(yolo_ok_count),
                    "grid_valid": int(grid_valid_count),
                    "yolo_retry": int(yolo_retry_count),
                    "motion_bursts": int(motion_burst_count),
                }
                print("\n" + "=" * 60)
                print("          VIDEO REANALYSIS COMPLETE          ")
                print("=" * 60)
                print(f"⏱️ Reanalysis time: {result.get('analysis_ms', 0.0):.1f} ms")
                print(f"🎥 Raw frames selected: {len(selected)}/{len(frames)}")
                print(f"🤖 YOLO successful: {yolo_ok_count} | retries: {yolo_retry_count}")
                print(f"🎨 Valid grids: {grid_valid_count}/{len(observations)}")
                print(f"🌀 Motion bursts: {motion_burst_count}")
                print(f"🧪 Fresh observations: {len(observations)}")
                print(f"🧩 Fresh frame-cache entries: {len(self.reanalysis_visuals)}")
                if perception_time_limit_hit:
                    print("⏱️ Perception time budget reached before the final frames; temporal correction used the freshly analyzed portion.")
                print("🧩 TEMPORALLY CORRECTED HISTORY: " + " ".join(result.get("history", [])))
                if result.get("corrected"):
                    print("🔧 Live history was corrected by the video reanalysis.")
                else:
                    print("✅ Reanalysis agrees with the live history.")
                print("=" * 60 + "\n")
            else:
                self.analysis_status = {
                    "active": True,
                    "status": "VIDEO REANALYSIS FAILED",
                    "progress": 100.0,
                    "processed": len(observations),
                    "total": len(observations),
                    "message": str(result.get("reason", "unknown failure")),
                    "phase": "FAILED",
                    "started_at": started,
                    "elapsed_s": float(time.perf_counter() - started),
                    "yolo_ok": int(yolo_ok_count),
                    "grid_valid": int(grid_valid_count),
                    "yolo_retry": int(yolo_retry_count),
                    "motion_bursts": int(motion_burst_count),
                }
                print("\n❌ [VIDEO REANALYSIS FAILED]: " + str(result.get("reason", "unknown failure")) + "\n")

        except Exception as exc:
            self.analysis_status = {
                "active": True,
                "status": "VIDEO REANALYSIS EXCEPTION",
                "progress": 100.0,
                "processed": len(observations),
                "total": max(len(observations), total_selected),
                "message": str(exc),
                "phase": "EXCEPTION",
                "started_at": started,
                "elapsed_s": float(time.perf_counter() - started),
                "yolo_ok": int(yolo_ok_count),
                "grid_valid": int(grid_valid_count),
                "yolo_retry": int(yolo_retry_count),
                "motion_bursts": int(motion_burst_count),
            }
            result = {
                "ok": False,
                "reason": f"VIDEO REANALYSIS EXCEPTION: {exc}",
                "reanalysis": True,
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
            }
            traceback.print_exc()
            try:
                self.final_result_queue.put_nowait(result)
            except queue.Full:
                pass

    def _analyze_frames_async(
        self,
        frames,
        observations,
        base_state,
        base_history,
        live_history_at_stop,
        live_move_events_at_stop,
    ):
        started = time.perf_counter()
        try:
            def _analysis_progress(processed, total, message):
                self.analysis_status = {
                    "active": True,
                    "status": "FINAL ANALYSIS RUNNING",
                    "progress": (float(processed) / float(max(1, total))) * 100.0,
                    "processed": int(processed),
                    "total": int(total),
                    "message": str(message),
                }

            result = self.final_analyzer.analyze(
                observations,
                base_state,
                base_history,
                live_history_at_stop=live_history_at_stop,
                progress_callback=_analysis_progress,
                live_move_events=live_move_events_at_stop,
            )

            result["total_pipeline_ms"] = (time.perf_counter() - started) * 1000.0
            result["frame_count"] = len(frames)
            result["observation_count"] = len(observations)

            # Replay frames were published immediately at STOP.  Do not enqueue
            # them again here, or the replay would be duplicated.

            try:
                self.final_result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.final_result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.final_result_queue.put_nowait(result)

            if result.get("ok"):
                self.analysis_status = {
                    "active": True,
                    "status": (
                        "FINAL CORRECTED HISTORY"
                        if result.get("corrected")
                        else "FINAL VERIFIED HISTORY"
                    ),
                    "progress": 100.0,
                    "processed": result.get("processed_checkpoints", 0),
                    "total": result.get("processed_checkpoints", 0) + result.get("skipped_checkpoints", 0),
                    "message": "Analysis complete",
                }
                print("\n" + "=" * 60)
                print("          FINAL SOLVE ANALYSIS COMPLETE          ")
                print("=" * 60)
                print(
                    f"⏱️ Analysis time: {result.get('analysis_ms', 0.0):.1f} ms"
                )
                if result.get("correction_method"):
                    print("🧠 Correction method:", result.get("correction_method"))
                print(
                    f"📍 Checkpoints: {result.get('processed_checkpoints', 0)} | "
                    f"Skipped: {result.get('skipped_checkpoints', 0)} | "
                    f"Deep searches: {result.get('deep_searches', 0)} | "
                    f"Hypotheses remaining: {result.get('hypothesis_count', 0)}"
                )
                if result.get("tail_recovered"):
                    print("🛠️ [FINAL ANALYSIS]: Trailing checkpoint recovered with bounded search.")
                print(
                    "🧩 FINAL CORRECTED HISTORY: "
                    + " ".join(result.get("history", []))
                )
                if result.get("corrected"):
                    print("🔧 Live history was corrected by the final analysis.")
                else:
                    print("✅ Live history already matched the final analysis.")
                validation = result.get("live_validation", {}) or {}
                if validation:
                    print(
                        "🔎 State-vs-live validation: "
                        f"{validation.get('matched', 0)} matched | "
                        f"{validation.get('disagreed', 0)} disagreed | "
                        f"{validation.get('ambiguous', 0)} ambiguous | "
                        f"{validation.get('unvalidated', 0)} unvalidated"
                    )
                if result.get("score_margin") is not None:
                    print(
                        f"📊 Best-vs-second score margin: "
                        f"{result['score_margin']:.2f}"
                    )
                print("=" * 60 + "\n")
            else:
                self.analysis_status = {
                    "active": True,
                    "status": "FINAL ANALYSIS FAILED",
                    "progress": 100.0,
                    "processed": result.get("processed_checkpoints", 0),
                    "total": result.get("processed_checkpoints", 0) + result.get("skipped_checkpoints", 0),
                    "message": str(result.get("reason", "unknown reason")),
                }
                print(
                    "\n❌ [FINAL ANALYSIS FAILED]: "
                    + str(result.get("reason", "unknown reason"))
                    + "\n"
                )
        except Exception as exc:
            self.analysis_status = {
                "active": True,
                "status": "FINAL ANALYSIS EXCEPTION",
                "progress": 100.0,
                "processed": 0,
                "total": 0,
                "message": str(exc),
            }
            result = {
                "ok": False,
                "reason": f"FINAL ANALYSIS EXCEPTION: {exc}",
                "analysis_ms": (time.perf_counter() - started) * 1000.0,
            }
            traceback.print_exc()
            try:
                self.final_result_queue.put_nowait(result)
            except queue.Full:
                pass

    def consume_final_result(self):
        try:
            result = self.final_result_queue.get_nowait()
        except queue.Empty:
            return None
        # The main thread now owns the authoritative state update.
        self.analysis_in_progress = False
        return result


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
        "Wheel / drag bar / PgUp-PgDn: scroll",
        (tab_x + 10, tab_bottom - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (170, 170, 170),
        1,
    )

    return scroll_offset, max_offset



def draw_final_analysis_log_panel(frame, final_analysis_display, scroll_offset=0):
    """Separate final-analysis panel with explicit LIVE / CALCULATED / DISPLAYED counts."""
    h, w = frame.shape[:2]
    x1 = 245
    x2 = min(w - 15, 730)
    y1 = 115
    y2 = min(h - 45, 335)
    if x2 <= x1 + 150 or y2 <= y1 + 100:
        return int(scroll_offset), 0

    history = list(final_analysis_display.get("history") or [])
    status = str(final_analysis_display.get("status", "NO FINAL ANALYSIS"))
    corrected = bool(final_analysis_display.get("corrected", False))
    reason = str(final_analysis_display.get("reason", ""))
    live_count = int(final_analysis_display.get("live_move_count", len(final_analysis_display.get("live_history") or [])))
    calculated_count = int(final_analysis_display.get("calculated_move_count", 0))
    displayed_count = int(final_analysis_display.get("displayed_move_count", len(history)))
    state_changes = int(final_analysis_display.get("state_change_count", 0))
    applied = bool(final_analysis_display.get("analysis_applied", False))

    moves_per_row = 8
    rows = [history[i:i + moves_per_row] for i in range(0, len(history), moves_per_row)]
    header_h = 32
    summary_h = 92
    footer_h = 22
    row_h = 24
    list_y = y1 + header_h + summary_h
    list_bottom = y2 - footer_h
    visible_rows = max(1, (list_bottom - list_y) // row_h)
    max_offset = max(0, len(rows) - visible_rows)
    scroll_offset = max(0, min(int(scroll_offset), max_offset))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 10, 14), -1)
    cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)

    border = (0, 255, 0) if applied else ((0, 255, 204) if corrected else (0, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
    cv2.putText(frame, "DEEP ANALYSIS — STATE-FIRST", (x1 + 12, y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, border, 2)

    # Three requested counts, plus how many high-confidence state changes existed.
    cv2.putText(frame, f"LIVE PICKED: {live_count}", (x1 + 12, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1)
    cv2.putText(frame, f"STATE CALC: {calculated_count}", (x1 + 145, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 230, 255), 1)
    cv2.putText(frame, f"DISPLAYED: {displayed_count}", (x1 + 278, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 255, 150), 1)
    cv2.putText(frame, f"STATE CHANGES: {state_changes}", (x1 + 12, y1 + 67), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (190, 220, 255), 1)

    if applied:
        apply_text = "APPLIED TO CUBE: YES"
        apply_color = (0, 255, 0)
    elif status.upper().startswith("FINAL ANALYSIS FAILED") or (not history and reason):
        apply_text = "APPLIED TO CUBE: NO"
        apply_color = (0, 165, 255)
    else:
        apply_text = "APPLIED TO CUBE: PENDING"
        apply_color = (0, 255, 255)
    cv2.putText(frame, apply_text, (x1 + 12, y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.31, apply_color, 1)

    validation = final_analysis_display.get("live_validation") or {}
    if validation:
        vm = int(validation.get("matched", 0))
        vd = int(validation.get("disagreed", 0))
        vl = int(validation.get("missing_live_moves", 0))
        cv2.putText(
            frame,
            f"LIVE CHECK: {vm} match | {vd} differ | {vl} live-missed",
            (x1 + 220, y1 + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (190, 220, 255),
            1,
        )

    cv2.line(frame, (x1 + 8, list_y - 6), (x2 - 8, list_y - 6), (55, 55, 55), 1)

    for row_offset in range(visible_rows):
        row_index = scroll_offset + row_offset
        if row_index >= len(rows):
            break
        row = rows[row_index]
        y = list_y + row_offset * row_h
        x = x1 + 12
        for move in row:
            token = str(move)
            token_w = max(34, 18 + cv2.getTextSize(token, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0])
            if x + token_w + 6 > x2 - 18:
                break
            cv2.rectangle(frame, (x, y), (x + token_w, y + row_h - 4), (22, 42, 30) if applied else (25, 30, 35), -1)
            cv2.rectangle(frame, (x, y), (x + token_w, y + row_h - 4), border, 1)
            cv2.putText(frame, token, (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            x += token_w + 6

    if not history:
        empty_text = reason[:62] if reason else "No state-derived final history yet."
        cv2.putText(frame, empty_text, (x1 + 12, list_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 165, 255), 1)

    if max_offset > 0:
        scrollbar_x = x2 - 9
        track_top = list_y
        track_bottom = list_bottom
        cv2.rectangle(frame, (scrollbar_x, track_top), (scrollbar_x + 4, track_bottom), (45, 45, 45), -1)
        thumb_h = max(18, int((visible_rows / max(1, len(rows))) * (track_bottom - track_top)))
        travel = max(1, (track_bottom - track_top) - thumb_h)
        thumb_y = track_top + int((scroll_offset / max_offset) * travel)
        cv2.rectangle(frame, (scrollbar_x, thumb_y), (scrollbar_x + 4, thumb_y + thumb_h), border, -1)

    cv2.putText(frame, status[:50], (x1 + 12, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (170, 170, 170), 1)
    return scroll_offset, max_offset


def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("\n[3D VIEWER READY] Open http://localhost:5001 in your browser.")
    print("⌨️ Controls: [SPACE] = Start/Stop Recording | [e] = Temporal Correct Recorded Solve | [c] = Scan Face | [r] = Reset | [q] = Quit\n")
    print("🎬 Replay Controls: [p] = Pause/Play Video | [,] = Prev Frame | [.] = Next Frame | [d] = Toggle Replay Debug | [o] = Close Replay\n")

    slowmo_processor = SlowMoBufferProcessor(cube_tracker, fps=60.0)

    last_valid_grid = None
    last_valid_flat_face = None

    beam_tracker = BeamStateTracker(
        cube_tracker.state, cube_tracker.permutations, beam_width=5
    )

    # Geometry-based physical motion detector.  It is ONLY a physical-motion
    # gate for move logging; it never guesses the move token.
    motion_gate = MotionEvidenceGate()

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
    final_analysis_display = {
        "active": False,
        "status": "LIVE ANALYSIS",
        "history": [],
        "live_history": [],
        "corrected": False,
        "reason": "",
        "live_validation": {},
        "validation_rows": [],
        "live_move_count": 0,
        "calculated_move_count": 0,
        "displayed_move_count": 0,
        "state_change_count": 0,
        "analysis_applied": False,
    }

    # --- LIVE LOOP TIMING DIAGNOSTICS ---
    live_loop_last_ms = 0.0
    live_loop_max_ms = 0.0
    live_loop_ema_ms = 0.0
    live_loop_counter = 0

    # Replay window explicitly keeps the camera aspect ratio.
    cv2.namedWindow("Slow-Mo Video Analyzer", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Scrollable replay move-tab state.
    #
    # OpenCV's mouse-wheel flags are backend/platform dependent, and on macOS
    # trackpads the value passed in `flags` is not always decoded reliably by
    # cv2.getMouseWheelDelta().  Decode it defensively and also provide drag
    # support on the visible scrollbar.
    replay_move_scroll = {"offset": 0}
    final_log_scroll = {"offset": 0}
    replay_scroll_drag = {
        "active": False,
        "start_y": 0,
        "start_offset": 0,
    }

    def _decode_mouse_wheel_delta(flags):
        """Best-effort wheel delta decoding across OpenCV backends/platforms."""
        try:
            delta = int(cv2.getMouseWheelDelta(flags))
            if delta != 0:
                return delta
        except Exception:
            pass

        # Common OpenCV/GUI convention: signed 16-bit wheel delta in the
        # high word of the flags integer.
        raw = (int(flags) >> 16) & 0xFFFF
        if raw:
            if raw & 0x8000:
                raw -= 0x10000
            if raw != 0:
                return raw

        # Final fallback for backends that encode the sign directly.
        if int(flags) > 0:
            return 1
        if int(flags) < 0:
            return -1
        return 0

    def replay_mouse_callback(event, x, y, flags, param):
        # Deep-analysis final log panel geometry.
        final_x1 = 245
        final_x2 = min(715, max(final_x1 + 140, replay_display_width[0] - 15))
        final_y1 = 115
        final_y2 = min(replay_display_height[0] - 45, 335)

        if event == cv2.EVENT_MOUSEWHEEL:
            if final_x1 <= x <= final_x2 and final_y1 <= y <= final_y2:
                delta = _decode_mouse_wheel_delta(flags)
                if delta != 0:
                    step = -2 if delta > 0 else 2
                    if abs(delta) >= 120:
                        step *= max(1, abs(delta) // 120)
                    final_log_scroll["offset"] = max(
                        0, final_log_scroll["offset"] + step
                    )
                return

        # Match the move-tab geometry used by draw_replay_move_tab().
        tab_x = 15
        tab_y = 105
        tab_w = 220
        tab_bottom = replay_display_height[0] - 45

        # Keep wheel scrolling active anywhere over the history panel, not
        # just the text rows. This is more reliable with trackpads.
        if event == cv2.EVENT_MOUSEWHEEL:
            if tab_x <= x <= tab_x + tab_w and tab_y <= y <= tab_bottom:
                delta = _decode_mouse_wheel_delta(flags)
                if delta != 0:
                    # One wheel notch = 3 moves.  Trackpad deltas naturally
                    # result in smaller/larger changes depending on backend.
                    step = -3 if delta > 0 else 3
                    if abs(delta) >= 120:
                        step *= max(1, abs(delta) // 120)

                    replay_move_scroll["offset"] = max(
                        0, replay_move_scroll["offset"] + step
                    )
            return

        # Click/drag the visible scrollbar as a second, mouse-only way to reach
        # every move when wheel events are poorly supported by the platform.
        scrollbar_x1 = tab_x + tab_w - 18
        scrollbar_x2 = tab_x + tab_w
        track_top = tab_y + 42
        track_bottom = tab_bottom - 24

        if event == cv2.EVENT_LBUTTONDOWN:
            if scrollbar_x1 <= x <= scrollbar_x2 and track_top <= y <= track_bottom:
                replay_scroll_drag["active"] = True
                replay_scroll_drag["start_y"] = y
                replay_scroll_drag["start_offset"] = replay_move_scroll["offset"]
                return

        elif event == cv2.EVENT_MOUSEMOVE and replay_scroll_drag["active"]:
            track_h = max(1, track_bottom - track_top)
            # Approximate thumb travel using current visible event count.
            total_events = len([
                e for e in replay_move_events if e.get("frame", -1) <= replay_idx
            ])
            visible_rows = max(1, (track_bottom - track_top) // 27)
            max_offset = max(0, total_events - visible_rows)

            if max_offset > 0:
                thumb_h = max(22, int((visible_rows / max(1, total_events)) * track_h))
                travel = max(1, track_h - thumb_h)
                offset_per_pixel = max_offset / travel
                dy = y - replay_scroll_drag["start_y"]
                new_offset = int(round(
                    replay_scroll_drag["start_offset"] + dy * offset_per_pixel
                ))
                replay_move_scroll["offset"] = max(0, min(max_offset, new_offset))
            return

        elif event == cv2.EVENT_LBUTTONUP:
            replay_scroll_drag["active"] = False

    # Mutable holder because replay dimensions are known only during playback.
    replay_display_height = [720]
    replay_display_width = [1280]
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

        # FINAL ANALYSIS RESULT HAND-OFF
        # The background final analyzer never mutates the live tracker. Once its
        # result is ready, the main thread atomically installs the corrected
        # state/history and resets the live inference workers from that state.
        final_result = slowmo_processor.consume_final_result()
        if final_result is not None:
            if final_result.get("ok"):
                final_analysis_display.update({
                    "active": True,
                    "status": (
                        "FINAL CORRECTED HISTORY"
                        if final_result.get("corrected")
                        else "FINAL VERIFIED HISTORY"
                    ),
                    "history": list(final_result.get("history", [])),
                    "live_history": list(final_result.get("live_history_at_stop", cube_tracker.move_history)),
                    "corrected": bool(final_result.get("corrected")),
                    "reason": str(final_result.get("reason", "")),
                    "live_validation": dict(final_result.get("live_validation", {})),
                    "validation_rows": list(final_result.get("validation_rows", [])),
                    "live_move_count": int(len(final_result.get("live_suffix", []))),
                    "calculated_move_count": int(len(final_result.get("calculated_suffix", final_result.get("best_candidate_moves", [])))),
                    "displayed_move_count": int(len(final_result.get("final_suffix", []))),
                    "state_change_count": int(final_result.get("checkpoint_stats", {}).get("state_changes", 0)),
                    "analysis_applied": True,
                })
                cube_tracker.state = final_result["state"].copy()
                cube_tracker.previous_state = cube_tracker.state.copy()
                cube_tracker.move_history = list(final_result["history"])
                cube_tracker.last_move_detected = (
                    final_result["new_moves"][-1]
                    if final_result.get("new_moves")
                    else "FINAL VERIFIED"
                )

                # Flush any stale live observations that accumulated while the
                # final pass was running, then restart live inference from the
                # corrected authoritative state.
                async_pipeline.reset_tracking()
                beam_tracker = BeamStateTracker(
                    cube_tracker.state, cube_tracker.permutations, beam_width=5
                )

                # Replace the replay move log with the corrected offline events.
                replay_move_events[:] = list(final_result.get("events", []))
                replay_move_scroll["offset"] = 0
                recording_history_snapshot = list(cube_tracker.move_history)

                cube_tracker.broadcast_state()

                print(
                    "🏁 [FINAL ANALYSIS APPLIED] Corrected history is now "
                    "authoritative."
                )
            else:
                final_analysis_display.update({
                    "active": True,
                    "status": "FINAL ANALYSIS FAILED",
                    "history": list(final_result.get("history", [])),
                    "live_history": list(final_result.get("live_history_at_stop", cube_tracker.move_history)),
                    "corrected": False,
                    "reason": str(final_result.get("reason", "unknown failure")),
                    "live_validation": dict(final_result.get("live_validation", {})),
                    "validation_rows": list(final_result.get("validation_rows", [])),
                    "live_move_count": int(len(final_result.get("live_suffix", []))),
                    "calculated_move_count": int(len(final_result.get("calculated_suffix", final_result.get("best_candidate_moves", [])))),
                    "displayed_move_count": int(len(final_result.get("final_suffix", []))),
                    "state_change_count": int(final_result.get("checkpoint_stats", {}).get("state_changes", 0)),
                    "analysis_applied": False,
                })
                print(
                    "⚠️ [FINAL ANALYSIS] No correction applied: "
                    + str(final_result.get("reason", "unknown failure"))
                )
                cube_tracker.last_move_detected = "FINAL ANALYSIS FAILED"
                cube_tracker.broadcast_state()

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

        # Physical motion is measured on the displayed/mirrored camera frame,
        # matching the supplied pure-geometry tracker.  This uses the same YOLO
        # quad already computed above; no second YOLO pass is introduced.
        motion_evidence = motion_gate.process_frame(
            display_frame,
            quad_corners
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
                if not slowmo_processor.analysis_in_progress:
                    async_pipeline.process_camera_frame(
                        current_grid,
                        frame_index=current_recording_frame_idx,
                        motion_evidence=motion_evidence,
                    )

                    if grid_lock.process(current_grid):
                        beam_tracker.process_observed_grid(current_grid)

                last_valid_grid = current_grid
                last_valid_flat_face = flat_face

        # Capture every valid/invalid per-frame observation while recording.
        # The final analyzer uses this lightweight timeline rather than running
        # YOLO a second time over the completed recording.
        if slowmo_processor.is_recording:
            slowmo_processor.add_observation(
                current_recording_frame_idx,
                current_grid if validate_face_grid(current_grid) else None,
                motion_evidence,
            )

        if last_valid_grid is not None:
            draw_permutation_ui(display_frame, last_valid_grid, last_valid_flat_face)

        # Motion-gate diagnostic: makes it obvious during testing whether a
        # proposed colour transition had real geometric motion behind it.
        mg_recent = bool(motion_evidence.get("structured", False))
        mg_move = "-"
        mg_energy = float(motion_evidence.get("energy", 0.0))
        mg_raw = float(motion_evidence.get("raw_energy", 0.0))
        mg_jump = float(motion_evidence.get("energy_jump", 0.0))
        mg_max = float(motion_evidence.get("max_cell_mag", 0.0))
        mg_burst = bool(motion_evidence.get("burst", False))
        mg_jitter = bool(motion_evidence.get("common_small", False))
        mg_age = motion_evidence.get("motion_age_ms")
        mg_age_txt = f"{mg_age:.0f}ms" if mg_age is not None else "-"
        cv2.putText(
            display_frame,
            f"MOTION: {'YES' if mg_recent else 'NO'} | raw={mg_raw:.1f} max={mg_max:.1f} jump={mg_jump:.1f} burst={'Y' if mg_burst else 'N'} jitter={'Y' if mg_jitter else 'N'}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0) if mg_recent else (0, 165, 255),
            1,
        )

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
            replay_display_width[0] = w_r

            # During the explicit E-pass, the replay window displays the
            # freshly reanalyzed YOLO/grid result cached by the background
            # worker. This restores the colour/grid inspector while avoiding a
            # competing third YOLO loop in the UI thread.
            r_quad_corners = None
            r_grid, r_flat_face, r_raw_lab = None, None, None
            with slowmo_processor.reanalysis_visuals_lock:
                fresh_vis = slowmo_processor.reanalysis_visuals.get(int(replay_idx))
                if fresh_vis is None and slowmo_processor.reanalysis_visuals:
                    # Nearest already-processed frame is better than showing a
                    # permanently empty grid while the pass catches up.
                    keys = [k for k in slowmo_processor.reanalysis_visuals.keys() if k <= int(replay_idx)]
                    if keys:
                        fresh_vis = slowmo_processor.reanalysis_visuals[max(keys)]
                if fresh_vis is not None:
                    r_grid = copy.deepcopy(fresh_vis.get("grid"))
                    r_quad_corners = copy.deepcopy(fresh_vis.get("quad"))

            # Before E starts (or when the replay predates the fresh cache), use
            # the normal single-frame replay perception so the grid remains
            # visible exactly as before.
            if r_grid is None and not slowmo_processor.analysis_in_progress:
                r_results = model(replay_display, conf=0.40, verbose=False)[0]
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

            if final_analysis_display.get("active"):
                debug_status = final_analysis_display.get("status", "FINAL ANALYSIS")
                debug_reason = final_analysis_display.get("reason", "")
            else:
                debug_status = debug_data.get("status", "NO DEDUCTION DATA")
                debug_reason = debug_data.get("commit_reason", "")

            fresh_variant = ''
            with slowmo_processor.reanalysis_visuals_lock:
                if fresh_vis is not None:
                    fresh_variant = str(fresh_vis.get('variant', ''))

            deduction_info = {
                'hypothesized_move': debug_data.get('chosen', 'NONE'),
                'decided_move': ', '.join(frame_moves) if frame_moves else 'HOLD',
                'confidence': 1.0,
                'top_candidates': debug_data.get('top_candidates', []),
                'recovery_mode': debug_data.get('recovery_mode', False),
                'status': debug_status,
                'hypothesis_count': debug_data.get('hypothesis_count', 0),
                'commit_ready': debug_data.get('commit_ready', False),
                'commit_reason': debug_reason if debug_reason else debug_data.get('commit_reason', ''),
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
                'reanalysis_variant': fresh_variant,
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

            # Separate final-analysis log: this is intentionally independent of
            # the replay/live event list so the user can immediately see whether
            # E produced a different move sequence.
            final_log_scroll["offset"], _ = draw_final_analysis_log_panel(
                replay_display,
                final_analysis_display,
                final_log_scroll["offset"],
            )

            if final_analysis_display.get("active"):
                fa_status = str(final_analysis_display.get("status", "FINAL ANALYSIS"))
                if slowmo_processor.analysis_in_progress:
                    analysis_status = slowmo_processor.analysis_status
                    progress = float(analysis_status.get("progress", 0.0))
                    message = str(analysis_status.get("message", "Working..."))
                    processed = int(analysis_status.get("processed", 0))
                    total = int(analysis_status.get("total", 0))
                    bar_x1, bar_y1 = 20, 78
                    bar_x2, bar_y2 = min(w_r - 20, 520), 112
                    cv2.rectangle(replay_display, (bar_x1, bar_y1), (bar_x2, bar_y2), (15, 15, 15), -1)
                    cv2.rectangle(replay_display, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 255, 255), 1)
                    fill_w = int((bar_x2 - bar_x1 - 4) * max(0.0, min(100.0, progress)) / 100.0)
                    if fill_w > 0:
                        cv2.rectangle(replay_display, (bar_x1 + 2, bar_y1 + 2), (bar_x1 + 2 + fill_w, bar_y2 - 2), (0, 180, 255), -1)
                    progress_text = (
                        f"FINAL ANALYSIS: {progress:5.1f}%"
                        + (f"  ({processed}/{total})" if total else "")
                    )
                    cv2.putText(replay_display, progress_text, (bar_x1 + 8, bar_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    cv2.putText(replay_display, message[:62], (bar_x1 + 8, bar_y1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1)

                    # Detailed live progress diagnostics so E-key analysis can be
                    # distinguished from a stalled process.
                    phase = str(analysis_status.get("phase", "WORKING"))
                    elapsed_s = float(analysis_status.get("elapsed_s", 0.0))
                    yolo_ok = int(analysis_status.get("yolo_ok", 0))
                    grid_valid = int(analysis_status.get("grid_valid", 0))
                    yolo_retry = int(analysis_status.get("yolo_retry", 0))
                    motion_bursts = int(analysis_status.get("motion_bursts", 0))
                    cv2.putText(replay_display, f"PHASE: {phase[:34]}", (bar_x1 + 8, bar_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                    cv2.putText(replay_display, f"ELAPSED: {elapsed_s:4.1f}s", (bar_x1 + 160, bar_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                    cv2.putText(replay_display, f"YOLO: {yolo_ok}  GRID: {grid_valid}  RETRY: {yolo_retry}", (bar_x1 + 8, bar_y2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (200, 240, 255), 1)
                    cv2.putText(replay_display, f"MOTION BURSTS (DIAG): {motion_bursts}", (bar_x1 + 8, bar_y2 + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (200, 255, 200), 1)
                    cv2.putText(
                        replay_display,
                        f"COUNTS: LIVE {int(final_analysis_display.get('live_move_count', 0))} | CALC {int(final_analysis_display.get('calculated_move_count', 0))} | SHOW {int(final_analysis_display.get('displayed_move_count', 0))}",
                        (bar_x1 + 8, bar_y2 + 66),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.31,
                        (255, 220, 170),
                        1,
                    )
                    # Animated activity marker: changes while the process is updating.
                    dot = 1 + (int(elapsed_s * 4.0) % 3)
                    cv2.putText(replay_display, "." * dot, (bar_x2 - 42, bar_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            if final_analysis_display.get("active"):
                fa_status = str(final_analysis_display.get("status", "FINAL ANALYSIS"))
                fa_text = fa_status[:42]
                cv2.rectangle(
                    replay_display,
                    (10, 78),
                    (min(w_r - 10, 430), 103),
                    (15, 15, 15),
                    -1,
                )
                cv2.putText(
                    replay_display,
                    fa_text,
                    (18, 96),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 0) if final_analysis_display.get("corrected") else (0, 255, 255),
                    1,
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
            if slowmo_processor.analysis_in_progress:
                print("⏳ [FINAL ANALYSIS]: Still processing previous solve; please wait for the result.")
                continue

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
                slowmo_processor.start_recording(
                    initial_state=cube_tracker.state.copy(),
                    initial_history=list(cube_tracker.move_history),
                )
            else:
                final_analysis_display.update({
                    "active": True,
                    "status": "FINAL ANALYSIS RUNNING",
                    "history": list(cube_tracker.move_history),
                    "live_history": list(cube_tracker.move_history),
                    "corrected": False,
                    "reason": "Starting deep reconstruction...",
                    "live_move_count": max(0, len(cube_tracker.move_history) - len(recording_history_snapshot)),
                    "calculated_move_count": 0,
                    "displayed_move_count": len(cube_tracker.move_history),
                    "state_change_count": 0,
                    "analysis_applied": False,
                })
                slowmo_processor.stop_and_process(
                    model,
                    extract_fn,
                    live_move_events=list(replay_move_events),
                    auto_analyze=False,
                    source_fps=float(actual_fps if actual_fps and actual_fps > 1.0 else 30.0),
                )
                final_analysis_display.update({
                    "active": True,
                    "status": "VIDEO READY — PRESS E TO REANALYZE",
                    "history": list(cube_tracker.move_history),
                    "corrected": False,
                    "reason": "Press E to re-run perception and reconstruction from the recorded video.",
                    "live_move_count": max(0, len(cube_tracker.move_history) - len(recording_history_snapshot)),
                    "calculated_move_count": 0,
                    "displayed_move_count": len(cube_tracker.move_history),
                    "state_change_count": 0,
                    "analysis_applied": False,
                })
        elif key == ord("e"):
            # Explicit second-pass video analysis.  This intentionally uses the
            # original recorded frames and a fresh perception/motion timeline;
            # it is NOT the same as replaying the live move-event list.
            if slowmo_processor.is_recording:
                print("⚠️ [VIDEO REANALYSIS]: Stop recording first (SPACE).")
                continue
            if slowmo_processor.analysis_in_progress:
                running_status = slowmo_processor.analysis_status or {}
                print(
                    "⏳ [VIDEO REANALYSIS]: Already running — "
                    f"{running_status.get('phase', running_status.get('status', 'WORKING'))} "
                    f"{float(running_status.get('progress', 0.0)):.1f}% "
                    f"({int(running_status.get('processed', 0))}/{int(running_status.get('total', 0))}), "
                    f"elapsed={float(running_status.get('elapsed_s', 0.0)):.1f}s"
                )
                final_analysis_display.update({
                    "active": True,
                    "status": str(running_status.get("status", "VIDEO REANALYSIS RUNNING")),
                    "history": list(cube_tracker.move_history),
                    "corrected": False,
                    "reason": str(running_status.get("message", "Reanalysis is still running.")),
                })
                continue

            final_log_scroll["offset"] = 0
            final_analysis_display.update({
                "active": True,
                "status": "VIDEO REANALYSIS STARTING",
                "history": list(cube_tracker.move_history),
                "live_history": list(cube_tracker.move_history),
                "corrected": False,
                "reason": "Re-running YOLO, grid extraction and state-timeline reconstruction on every recorded frame; motion is diagnostic only.",
                "live_move_count": max(0, len(cube_tracker.move_history) - len(recording_history_snapshot)),
                "calculated_move_count": 0,
                "displayed_move_count": len(cube_tracker.move_history),
                "state_change_count": 0,
                "analysis_applied": False,
            })

            started_reanalysis = slowmo_processor.reanalyze_recorded_video(
                model,
                extract_fn,
                source_fps=float(actual_fps if actual_fps and actual_fps > 1.0 else 30.0),
            )
            if not started_reanalysis:
                final_analysis_display.update({
                    "active": True,
                    "status": "VIDEO REANALYSIS NOT STARTED",
                    "history": list(cube_tracker.move_history),
                    "corrected": False,
                    "reason": "No completed recording is available or another analysis is running.",
                })
        elif key == ord("o"):  # CLOSE WINDOW
            if len(replay_frames) > 0 or cv2.getWindowProperty("Slow-Mo Video Analyzer", cv2.WND_PROP_VISIBLE) >= 1:
                replay_frames.clear()
                move_log.clear()
                replay_move_scroll["offset"] = 0
                final_log_scroll["offset"] = 0
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
        elif key == 216:  # PAGE DOWN on some OpenCV/macOS backends
            replay_move_scroll["offset"] += 8
        elif key == 121:  # PAGE UP on some OpenCV/macOS backends
            replay_move_scroll["offset"] = max(0, replay_move_scroll["offset"] - 8)
        elif key == ord(","):  # STEP BACKWARD
            if len(replay_frames) > 0:
                replay_paused = True
                replay_idx = (replay_idx - 1) % len(replay_frames)
        elif key == ord("["):
            replay_move_scroll["offset"] = 0
        elif key == ord("]"):
            replay_move_scroll["offset"] += 8
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