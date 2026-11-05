# V62: orientation-proof solving + scan, SPACE timer, auto video, leaderboard, colour sets, 3D view, faster display
# V61: cleaned up + clickable UI + move arrows | front camera uses the on-screen move names again
#      (F = the face the camera sees), algorithms are translated so they always work physically,
#      "Fix tracking" button, trainer/timer/scan/camera modes from V60 | V59: any orientation |
#      V54: tuned live decoder (the tracking core is unchanged since V54/V59)
import glob
import threading
import time
import copy
import queue
import traceback
import types
from collections import deque
from difflib import SequenceMatcher

import cv2
import numpy as np
import kociemba
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from ultralytics import YOLO

# Moves are shown in the user's notation (set up in main()).
MOVE_NAMING_ACTIVE = None


def user_names(tokens):
    return MOVE_NAMING_ACTIVE.to_user(list(tokens)) if MOVE_NAMING_ACTIVE is not None else list(tokens)


# --- FLASK + SOCKET.IO SETUP (3D viewer at http://localhost:5001) ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "rubiks_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


import json as _json
import os as _os

# =============================================================================
#  EASY-EDIT SETTINGS  -  change the look and behaviour here
# =============================================================================
# Colours are (Blue, Green, Red), 0-255.  Sizes are in pixels.
# Button keys: the keyboard key that does the same thing (" " = SPACE).
# Nothing below this block needs editing to restyle the app.
UI_CONFIG = {
    "layout": {
        "camera_width": 960,        # width of the camera picture in the window
        "sidebar_width": 390,
        "top_bar_height": 40,
        "guide_height": 116,        # the move strip under the camera
        "button_height": 25,
        "button_gap": 5,
        "button_columns": 3,
        "cube_view_height": 140,    # 3D / 2D cube box in the sidebar
        "padding": 14,
    },
    "colours": {
        "background": (27, 24, 22),
        "panel": (40, 36, 33),
        "panel_light": (54, 49, 45),
        "line": (70, 64, 58),
        "text": (238, 238, 238),
        "muted": (160, 155, 150),
        "dim": (105, 100, 96),
        "accent": (235, 170, 45),   # blue
        "good": (110, 205, 90),     # green
        "warn": (40, 185, 255),     # amber
        "bad": (85, 85, 235),       # red
        "arrow": (0, 230, 255),     # yellow move arrow
        "arrow_undo": (70, 70, 255),
        "timer_hold": (80, 80, 240),     # SPACE held, not ready yet
        "timer_ready": (110, 225, 90),   # SPACE held long enough - release to start
    },
    "sticker_colours": {
        "White": (240, 240, 240), "Yellow": (0, 220, 255), "Green": (70, 185, 40),
        "Blue": (200, 90, 20), "Orange": (0, 130, 255), "Red": (50, 45, 215),
    },
    "text_size": {
        "title": 0.72, "body": 0.5, "small": 0.42, "button": 0.42,
        "guide_move": 0.72, "timer": 1.05, "toast": 0.52,
    },
    # (section title, [(key, label), ...])
    "buttons": [
        ("GUIDE", [("g", "Scramble"), ("s", "Solve"), ("n", "Preset"),
                   ("w", "Fix tracking"), ("k", "Close guide")]),
        ("PRACTICE", [("a", "Timer"), ("j", "PLL case"), ("h", "OLL case"),
                      ("l", "Leaderboard")]),
        ("VIDEO", [("x", "Record"), ("u", "Auto-record"), ("o", "Replay"),
                   ("y", "Save video"), ("e", "Check video")]),
        ("SETUP", [("f", "Front cam"), ("t", "Top-down"), ("m", "Learn cam"),
                   ("i", "Scan cube"), ("v", "Colours"), ("z", "Colour sets"),
                   ("r", "Reset")]),
    ],
    "text": {
        "title": "CUBE TRACKER",
        "idle_hint": "Scramble for a scramble, Solve for a solution - or hold SPACE, release, and solve.",
        "timer_hint": "Hold SPACE, release to start",
    },
}

APP_SETTINGS = {
    # Tracking runs in the background at most this often (per second). The
    # move detector was tuned at ~11 per second: raising this can make it
    # less stable. The camera picture itself is shown at full speed.
    "tracking_max_hz": 12.0,
    "threaded_tracking": True,      # False = everything in one loop (V61 behaviour)
    "auto_record": True,            # record every timed solve
    "max_record_seconds": 180,
    "record_jpeg_quality": 92,
    "timer_hold_seconds": 0.5,      # how long SPACE must be held before release starts the timer
    "lighting_check": True,
    "remember_camera_mode": False,  # True = start in the camera mode used last time         # suggest recalibrating when the light changes a lot
    "player": "Me",                 # default name on the leaderboard
    "videos_folder": "recordings",
}

SETTINGS_FILE = "settings.json"


def load_app_settings():
    try:
        if _os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as fh:
                APP_SETTINGS.update(_json.load(fh))
    except Exception as exc:
        print(f"⚠️ [SETTINGS]: could not load ({exc})")


def save_app_settings():
    try:
        with open(SETTINGS_FILE, "w") as fh:
            _json.dump(APP_SETTINGS, fh, indent=1)
    except Exception as exc:
        print(f"⚠️ [SETTINGS]: could not save ({exc})")


# =============================================================================
# PHYSICAL MOTION EVIDENCE (video re-analysis; live tracking uses the rigid part)
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

    def process_frame(self, frame, quad_corners, timestamp=None):
        # V50: offline passes must supply the frame's real capture timestamp.
        # Using wall-clock time while reprocessing a recording made the 80 ms
        # "recent" window meaningless.
        now = float(timestamp) if timestamp is not None else time.perf_counter()
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

        curr_quad = self._order_points(quad_corners)
        rigid_matrix = self._rigid_transform(self.prev_quad, curr_quad)

        cell_polys = self._subdivide_3x3(curr_quad)
        centers = self._cell_centers(curr_quad)
        if cell_polys is None or centers is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.prev_quad = curr_quad
            self.motion_streak = 0
            return result

        # V50: compute optical flow only on the cube's bounding box (plus a
        # margin), not the whole 1280x720 frame, and build the per-cell masks
        # at crop size. The old version allocated nine full-frame masks per
        # frame, which cost ~55 ms/frame and slowed the live loop.
        h, w = gray.shape[:2]
        box_pts = curr_quad if self.prev_quad is None else np.vstack([curr_quad, self.prev_quad])
        margin = 24
        x0 = int(max(0, np.floor(box_pts[:, 0].min()) - margin))
        y0 = int(max(0, np.floor(box_pts[:, 1].min()) - margin))
        x1 = int(min(w, np.ceil(box_pts[:, 0].max()) + margin))
        y1 = int(min(h, np.ceil(box_pts[:, 1].max()) + margin))
        if x1 - x0 < 16 or y1 - y0 < 16:
            self.prev_gray = gray
            self.prev_quad = curr_quad
            self.motion_streak = 0
            return result

        try:
            flow = self.dis_flow.calc(
                np.ascontiguousarray(self.prev_gray[y0:y1, x0:x1]),
                np.ascontiguousarray(gray[y0:y1, x0:x1]),
                None,
            )
        except Exception:
            self.prev_gray = gray
            self.prev_quad = curr_quad
            self.motion_streak = 0
            return result

        cell_vecs = np.zeros((3, 3, 2), dtype=np.float32)
        crop_h, crop_w = y1 - y0, x1 - x0
        offset = np.array([x0, y0], dtype=np.int32)

        for r in range(3):
            for c in range(3):
                poly = cell_polys[r][c] - offset
                mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                c_flow = flow[mask == 255]

                if len(c_flow) == 0:
                    continue

                cell_vecs[r, c] = [
                    float(np.median(c_flow[:, 0])),
                    float(np.median(c_flow[:, 1])),
                ]

        # V50: remove whole-cube / hand / camera motion BEFORE measuring the
        # burst. The docstring always promised this but the old code never
        # applied it. The per-axis median of the nine cell vectors is a robust
        # estimate of the common motion: an R/L/U/D turn moves only 3 of 9
        # cells, and an F turn's vectors cancel around the centre, so the
        # median stays at the shared (rigid) motion.
        common = np.median(cell_vecs.reshape(9, 2), axis=0)
        residual = cell_vecs - common.reshape(1, 1, 2)

        raw_vecs = np.zeros((3, 3, 2), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                u, v = residual[r, c]
                if float(np.hypot(u, v)) >= self.vector_deadzone:
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

# V50: Ultralytics only uses the GPU automatically on NVIDIA/CUDA. On a Mac
# it silently runs on the CPU unless device="mps" is passed to every call.
def _select_yolo_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

YOLO_DEVICE = _select_yolo_device()
print(f"🧠 [YOLO]: model={MODEL_PATH} device={YOLO_DEVICE}")

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
        # V50 FIX: these two lines were reversed ([17,16,15]) in V49, which
        # sent each back corner's three stickers to three DIFFERENT corners.
        # Edges happened to be right, so the bug stayed hidden until a
        # corrupted corner sticker was later turned onto the visible front.
        pB[[15, 16, 17]] = I[[36, 39, 42]]
        pB[[53, 50, 47]] = I[[15, 16, 17]]
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

        # V50: the X^4 / inverse checks above cannot detect a table that
        # scrambles corners (the V49 B bug passed them). On a real cube the
        # commutator X Y X' Y' has order 6 for adjacent faces and 1 for
        # opposite faces. Refuse to start if any pair violates this.
        def _order(p):
            q, n = p.copy(), 1
            while not np.array_equal(q, I):
                q, n = q[p], n + 1
                if n > 2000:
                    return n
            return n

        opposite = {frozenset("RL"), frozenset("UD"), frozenset("FB")}
        faces = "RLUDFB"
        for i, a in enumerate(faces):
            for b in faces[i + 1:]:
                P = self.permutations
                comm = I[P[a]][P[b]][P[a + "'"]][P[b + "'"]]
                expected = 1 if frozenset(a + b) in opposite else 6
                got = _order(comm)
                if got != expected:
                    raise RuntimeError(
                        f"[PERMUTATION ERROR] commutator {a} {b} {a}' {b}' has order "
                        f"{got}, expected {expected}: move tables are not a real cube."
                    )

        self._init_extended_moves()

    # ------------------------------------------------------------------
    # V52: slice moves (M E S), rotations (x y z) and wide moves (r l u d f b)
    # ------------------------------------------------------------------
    FACE_MOVES = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]

    # Every new move follows the turning direction of a reference face move
    # in THIS program's naming (the names you see in the history), so the new
    # moves use exactly the same convention as the face moves you already use:
    #   M follows L, E follows D, S follows F      (middle layer only)
    #   x follows R, y follows U, z follows F      (whole cube)
    #   r=R+middle, l=L+middle, u=U+middle, d=D+middle, f=F+middle, b=B+middle
    EXTENDED_MOVE_REFS = {
        "M": ("L", "slice"), "E": ("D", "slice"), "S": ("F", "slice"),
        "x": ("R", "all"), "y": ("U", "all"), "z": ("F", "all"),
        "r": ("R", "wide"), "l": ("L", "wide"), "u": ("U", "wide"),
        "d": ("D", "wide"), "f": ("F", "wide"), "b": ("B", "wide"),
    }

    # Where each of the 54 state indices sits on a physical 3x3x3 model
    # (index into the model sticker list built below). Derived once from the
    # six face tables and re-verified every startup in _init_extended_moves.
    STICKER_TO_MODEL = [
        18, 21, 24, 19, 22, 25, 20, 23, 26, 29, 32, 35, 28, 31, 34, 27, 30, 33,
        38, 41, 44, 37, 40, 43, 36, 39, 42, 53, 50, 47, 52, 49, 46, 51, 48, 45,
        15, 16, 17, 12, 13, 14, 9, 10, 11, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    ]

    @staticmethod
    def _model_stickers():
        stickers = []
        for n in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            ax = int(np.argmax(np.abs(n)))
            others = [i for i in range(3) if i != ax]
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    p = [0, 0, 0]
                    p[ax] = n[ax]
                    p[others[0]] = a
                    p[others[1]] = b
                    stickers.append((tuple(p), tuple(n)))
        return stickers

    @staticmethod
    def _rotation_matrix(axis, k):
        c = [1, 0, -1, 0][k % 4]
        s_ = [0, 1, 0, -1][k % 4]
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        if axis == 1:
            return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])

    def _init_extended_moves(self):
        stickers = self._model_stickers()
        where = {st: i for i, st in enumerate(stickers)}
        phi = np.array(self.STICKER_TO_MODEL)
        inv_phi = np.argsort(phi)

        def state_perm(axis, k, layers):
            rot = self._rotation_matrix(axis, k)
            model = np.arange(54)
            for i, (p, n) in enumerate(stickers):
                if p[axis] in layers:
                    q = tuple(int(v) for v in rot @ np.array(p))
                    nn = tuple(int(v) for v in rot @ np.array(n))
                    model[where[(q, nn)]] = i
            return inv_phi[model[phi]]

        # 1. Recover (axis, direction, layer) of every existing face move and
        #    prove the physical model reproduces the hand-written tables.
        face_geometry = {}
        for face in ("R", "L", "U", "D", "F", "B"):
            found = None
            for axis in range(3):
                for layer in (-1, 1):
                    for k in (1, -1):
                        if np.array_equal(state_perm(axis, k, [layer]), self.permutations[face]):
                            found = (axis, k, layer)
            if found is None:
                raise RuntimeError(
                    f"[PERMUTATION ERROR] face move {face} does not match the 3D model; "
                    "extended moves cannot be built safely."
                )
            face_geometry[face] = found

        # 2. Build the new moves from their reference face's rotation.
        for name, (ref, kind) in self.EXTENDED_MOVE_REFS.items():
            axis, k, layer = face_geometry[ref]
            layers = {"slice": [0], "all": [-1, 0, 1], "wide": [layer, 0]}[kind]
            perm = state_perm(axis, k, layers)
            self.permutations[name] = perm
            self.permutations[name + "'"] = np.argsort(perm)

        self.EXTENDED_MOVES = [m + suffix for m in self.EXTENDED_MOVE_REFS for suffix in ("", "'")]
        self.ALL_QUARTER_MOVES = list(self.FACE_MOVES) + list(self.EXTENDED_MOVES)

        # 3. Sanity: every new quarter move has order 4, and the relations that
        #    hold on any real cube hold here (a rotation = its three layers).
        I = np.arange(54)
        for name in self.EXTENDED_MOVE_REFS:
            p = self.permutations[name]
            if not np.array_equal(p[p][p][p], I):
                raise RuntimeError(f"[PERMUTATION ERROR] {name}^4 is not identity")
        P = self.permutations
        def seq(*moves):
            out = I.copy()
            for mv in moves:
                out = out[P[mv]]
            return out
        checks = {
            "x = R M' L'": (seq("x"), seq("R", "M'", "L'")),
            "y = U E' D'": (seq("y"), seq("U", "E'", "D'")),
            "r = R M'": (seq("r"), seq("R", "M'")),
            "u = U E'": (seq("u"), seq("U", "E'")),
            "f = F S": (seq("f"), seq("F", "S")),
        }
        for label, (a, b) in checks.items():
            if not np.array_equal(a, b):
                raise RuntimeError(f"[PERMUTATION ERROR] relation {label} fails")

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
                if previous in {move, move.replace("2", "")} and move in set(
                    getattr(self, "ALL_QUARTER_MOVES", [])
                    or ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
                ):
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
            'move_history': user_names(self.move_history),
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

            display_solution = " ".join(user_names(remap_kociemba_for_display(raw_solution).split()))

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


cube_tracker = PermutationCubeTracker()
if not cube_tracker.history_matches_state():
    raise RuntimeError("Initial cube state/history consistency check failed.")

# V59: start (and reset with [r]) from a solved cube held in ANY orientation.
START_ANY_ORIENTATION = True

# V54: V53 features can be switched off individually to compare with V52.
USE_SOFT_COLOURS = True        # glare-aware per-sticker confidence (False = plain colour labels)
USE_FRAME_GATE = True          # ignore frames while the whole cube is being moved
USE_COLOUR_ADAPTATION = True   # slowly follow lighting changes when the tracker is confident


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

    # V53: each scanned face is solved, so its colour is KNOWN. Use the median
    # of its 9 stickers (ignores one glared sticker) instead of guessing with
    # k-means, and measure how spread each colour is to set the confidence scale.
    SIGMA_MIN = 12.0   # V54: was 6; real lighting shifts more than one scan shows
    SIGMA_MAX = 24.0

    def _compute_kmeans_centroids(self):
        self.cluster_centers = {}
        spreads = []
        for color_name in SCAN_ORDER:
            patches = np.asarray(self.scanned_faces[color_name], dtype=float)
            centre = np.median(patches, axis=0)
            self.cluster_centers[color_name] = centre
            spreads.extend(self._weighted_dist(p, centre) for p in patches)
        self.sigma = float(np.clip(1.5 * np.median(spreads), self.SIGMA_MIN, self.SIGMA_MAX))
        white_L = float(self.cluster_centers["White"][0])
        # Glare = brighter than the calibrated white sticker AND colourless.
        self.glare_L = float(np.clip(white_L + 14.0, 232.0, 250.0))
        print(f"\n[SUCCESS] Colours calibrated from the scan | sigma={self.sigma:.1f} "
              f"| white L={white_L:.0f} -> glare above L={self.glare_L:.0f}")

    @staticmethod
    def _weighted_dist(mean_lab, center_lab):
        dL = (mean_lab[0] - center_lab[0]) * 0.3
        da = (mean_lab[1] - center_lab[1]) * 1.3
        db = (mean_lab[2] - center_lab[2]) * 1.3
        return float(np.sqrt(dL ** 2 + da ** 2 + db ** 2))

    def colour_likelihoods(self, mean_lab, glare_frac=0.0, good_frac=1.0):
        """V54: per-sticker evidence (DECODER_COLORS order).
        A clean sticker counts EXACTLY like V52's plain colour label (right
        colour 0.9, each other 0.02), so good conditions behave as before.
        Only glare, too few clean pixels, or a colour far from every calibrated
        sticker (fingers, shadows) fade the reading towards 'unsure'."""
        sigma = float(getattr(self, "sigma", 12.0))
        d = np.array([self._weighted_dist(mean_lab, self.cluster_centers[c]) for c in DECODER_COLORS])
        best = int(np.argmin(d))
        white_idx = DECODER_COLORS.index("White")
        if best == white_idx:
            # glare's signature failure: a sticker washed out to white
            w_glare = float(np.clip((glare_frac - 0.15) / 0.25, 0.0, 1.0))
        else:
            w_glare = float(np.clip((glare_frac - 0.35) / 0.40, 0.0, 1.0))
        if good_frac < 0.25:
            w_glare = 1.0
        w_far = float(np.clip((d[best] / sigma - 3.5) / 2.5, 0.0, 1.0))
        w = max(w_glare, w_far)
        row = np.full(6, 0.02)
        row[best] = 0.90
        row = row / row.sum()
        return (1.0 - w) * row + w / 6.0

    def adapt(self, color_name, mean_lab, rate=0.02):
        """Slowly follow lighting changes using stickers whose true colour the
        tracker is confident about. Refuses outliers so misreads cannot drag it."""
        centre = self.cluster_centers.get(color_name)
        if centre is None:
            return
        if self._weighted_dist(mean_lab, centre) > 2.0 * float(getattr(self, "sigma", 12.0)):
            return
        self.cluster_centers[color_name] = (1.0 - rate) * centre + rate * np.asarray(mean_lab, dtype=float)


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


def _robust_patch_lab(patch, glare_L=240.0):
    """Mean colour of the sticker's UNGLARED pixels + how much of it is glare."""
    px = patch.reshape(-1, 3).astype(float)
    L = px[:, 0]
    chroma = np.hypot(px[:, 1] - 128.0, px[:, 2] - 128.0)
    # V54: real glare is a solid bright patch; normal surface sheen is
    # scattered specks. Blurring first makes isolated specks fall below the
    # threshold while a genuine glare patch stays above it.
    blurred = cv2.blur(patch, (5, 5)).reshape(-1, 3).astype(float)
    L_b = blurred[:, 0]
    chroma_b = np.hypot(blurred[:, 1] - 128.0, blurred[:, 2] - 128.0)
    glare = (L_b >= glare_L) & (chroma_b <= 18.0)
    good = (~glare) & (L > 25)
    glare_frac = float(glare.mean())
    good_frac = float(good.mean())
    if good_frac >= 0.25:
        mean = px[good].mean(axis=0)
    else:
        valid = L > 25
        mean = px[valid].mean(axis=0) if valid.any() else px.mean(axis=0)
    return mean, glare_frac, good_frac


def extract_facelet_grid_soft(frame, rect):
    """V53: grid + per-sticker colour probabilities (9x6, DECODER_COLORS order)
    + per-sticker glare fraction. Grid labels are the most likely colour."""
    dst_pts = np.array([[0, 0], [149, 0], [149, 149], [0, 149]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    flat_face = cv2.warpPerspective(frame, M, (150, 150))
    lab_face = cv2.cvtColor(flat_face, cv2.COLOR_BGR2LAB)
    grid = [["Unknown" for _ in range(3)] for _ in range(3)]
    raw_lab_patches, probs, glare = [], [], []
    glare_L = float(getattr(kmeans_calibrator, "glare_L", 240.0))
    for r in range(3):
        for c in range(3):
            patch = lab_face[r * 50 + 10:(r + 1) * 50 - 10, c * 50 + 10:(c + 1) * 50 - 10]
            mean, g_frac, good_frac = _robust_patch_lab(patch, glare_L)
            raw_lab_patches.append(mean)
            glare.append(g_frac)
            if kmeans_calibrator.is_calibrated:
                p = kmeans_calibrator.colour_likelihoods(mean, g_frac, good_frac)
                probs.append(p)
                grid[r][c] = DECODER_COLORS[int(np.argmax(p))]
    probs = np.array(probs) if probs else None
    return grid, flat_face, raw_lab_patches, probs, glare


def extract_facelet_grid_with_debug(frame, rect):
    grid, flat_face, raw_lab_patches, _probs, _glare = extract_facelet_grid_soft(frame, rect)
    return grid, flat_face, raw_lab_patches


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


# =============================================================================
# V50: FRAME-LEVEL BEAM DECODER (primary E-pass reconstruction)
# =============================================================================
# Every recorded frame is scored against every surviving complete-cube
# hypothesis. A frame is a noisy observation of the front face (each sticker
# right with prob p_sticker, or the whole frame garbage with prob p_garbage: a
# hand, a half-turned layer, glare). Between frames the cube stays still or does
# 1-2 quarter turns, each move costing a prior penalty. Identical cube states are
# merged, so the exact frame of a hidden B never splits the beam.
# Unlike the checkpoint analyser there are no hard constraints: a misread
# plateau costs a few points instead of forcing an impossible search.

DECODER_COLORS = ["White", "Yellow", "Green", "Blue", "Orange", "Red"]
DECODER_CIDX = {c: i for i, c in enumerate(DECODER_COLORS)}
DECODER_INVERSE = {}
for _base in ["R", "L", "U", "D", "F", "B", "M", "E", "S", "x", "y", "z",
              "r", "l", "u", "d", "f", "b"]:
    DECODER_INVERSE[_base] = _base + "'"
    DECODER_INVERSE[_base + "'"] = _base

DECODER_FACE_MOVES = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
# V52: small extra cost for non-face moves, so an ordinary face turn wins
# whenever both explain the camera equally well.
DECODER_EXTENDED_EXTRA_COST = 3.0
# S and b never change the front face (like B), so the camera cannot tell
# them apart from B/B2 until much later. Make them expensive unless the guide
# says they are coming next (the guide prior overrides this cost).
DECODER_FRONT_INVISIBLE = {"S", "b"}
DECODER_HIDDEN_EXTRA_COST = 14.0


def observation_loglik(grid=None, probs=None):
    """V53: 9x6 table of log P(reading | true colour) for one frame, or None.
    With soft probabilities (live camera) they are used directly. With only a
    colour label (E pass, old recordings) a confusion table is used in which a
    WHITE reading is weak evidence (glare makes anything look white) and
    orange/red are easily confused."""
    if probs is not None:
        return np.log(np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0))
    if grid is None:
        return None
    flat = [c for row in grid for c in row] if isinstance(grid[0], (list, tuple)) else list(grid)
    if len(flat) != 9 or not all(str(c) in DECODER_CIDX for c in flat):
        return None
    table = np.empty((9, 6))
    W, O, R = DECODER_CIDX["White"], DECODER_CIDX["Orange"], DECODER_CIDX["Red"]
    for i, c in enumerate(flat):
        k = DECODER_CIDX[str(c)]
        row = np.full(6, 0.024)
        if k == W:
            row[:] = 0.056
            row[W] = 0.72
        elif k in (O, R):
            row[:] = 0.025
            row[k] = 0.80
            row[R if k == O else O] = 0.10
        else:
            row[k] = 0.88
        table[i] = np.log(row / row.sum())
    return table


OBS_UNIFORM_LL = 9 * np.log(1.0 / 6.0)
_OBS_ROWS = np.arange(9)


def decoder_move_set(permutations):
    """(moves, per-move extra cost, double-move pairs) for the decoders.
    Doubles: any two face moves (fast turning) plus X X for every other move
    (M2, x2, r2 ...). Other mixed pairs are rare and would be very expensive."""
    moves = [m for m in DECODER_INVERSE if m in permutations]
    moves = [m for m in DECODER_FACE_MOVES if m in moves] + [m for m in moves if m not in DECODER_FACE_MOVES]
    extra = np.array([
        0.0 if m in DECODER_FACE_MOVES
        else (DECODER_HIDDEN_EXTRA_COST if m.rstrip("'") in DECODER_FRONT_INVISIBLE else DECODER_EXTENDED_EXTRA_COST)
        for m in moves
    ])
    pairs = []
    for i, a in enumerate(moves):
        for j, b in enumerate(moves):
            if DECODER_INVERSE[a] == b:
                continue
            if (a in DECODER_FACE_MOVES and b in DECODER_FACE_MOVES) or a == b:
                pairs.append((i, j))
    return moves, extra, pairs


ROTATION_MOVES = {"x", "x'", "y", "y'", "z", "z'", "x2", "y2", "z2"}


def strip_leading_rotations(moves):
    """V59: rotations before the first real turn only describe how the cube
    was picked up; with an any-orientation start they are not moves."""
    i = 0
    while i < len(moves) and moves[i] in ROTATION_MOVES:
        i += 1
    return list(moves[i:])


def is_solved_state(state):
    return all(len(set(state[i * 9:(i + 1) * 9])) == 1 for i in range(6))


def all_orientations(state, permutations):
    """V59: the 24 ways the same cube can be held (whole-cube rotations)."""
    if not all(k in permutations for k in ("x", "y", "z")):
        return [np.asarray(state).copy()]
    seen = {}
    frontier = [np.asarray(state).copy()]
    while frontier:
        st = frontier.pop()
        key = st.tobytes()
        if key in seen:
            continue
        seen[key] = st
        for r in ("x", "y", "z"):
            frontier.append(st[np.asarray(permutations[r])])
    return list(seen.values())


def decode_front_video(observations, base_state, permutations, beam=400,
           p_sticker=0.95, p_garbage=0.20, move_cost=4.0, second_move_cost=2.0,
           allow_double=True, progress=None):
    moves, extra_cost, pairs = decoder_move_set(permutations)
    P = np.stack([np.asarray(permutations[m]) for m in moves])
    P2 = np.stack([P[i][P[j]] for i, j in pairs]) if allow_double else None
    pair_extra = np.array([extra_cost[i] + extra_cost[j] for i, j in pairs])

    ll_match = np.log(p_sticker)
    ll_miss = np.log((1 - p_sticker) / 5.0)
    ll_garbage = 9 * np.log(1 / 6.0) + np.log(p_garbage)
    ll_clean = np.log(1 - p_garbage)

    # V59: a solved starting cube may be held in any of 24 orientations; all of
    # them start as candidates and the footage decides.
    starts = all_orientations(np.asarray(base_state), permutations) if is_solved_state(base_state) else [base_state]
    states = np.array([[DECODER_CIDX[str(c)] for c in st] for st in starts], dtype=np.uint8)
    scores = np.zeros(len(states))
    back = []   # per frame: (parent_idx array, move-code array)

    n = len(observations)
    for t, ob in enumerate(observations):
        # ---- expand ----
        cand = [states]
        code = [np.full(len(states), -1)]
        cost = [np.zeros(len(states))]
        par = [np.arange(len(states))]
        if t > 0:
            s1 = states[:, P].reshape(-1, 54)                                 # (N*12,54)
            cand.append(s1)
            code.append(np.tile(np.arange(len(moves)), len(states)))
            cost.append(np.tile(move_cost + extra_cost, len(states)))
            par.append(np.repeat(np.arange(len(states)), len(moves)))
            if allow_double:
                s2 = states[:, P2].reshape(-1, 54)
                cand.append(s2)
                code.append(np.tile(np.arange(len(pairs)) + 1000, len(states)))
                cost.append(np.tile(2 * move_cost + second_move_cost + pair_extra, len(states)))
                par.append(np.repeat(np.arange(len(states)), len(pairs)))
        cand = np.concatenate(cand)
        code = np.concatenate(code)
        par = np.concatenate(par)
        sc = np.repeat(scores, 1)  # placeholder
        sc = scores[par] - np.concatenate(cost)

        # ---- observe ----
        table = observation_loglik(ob.get("grid"), ob.get("probs"))
        if table is not None:
            fit = table[_OBS_ROWS, cand[:, 18:27]].sum(axis=1)
            sc = sc + np.logaddexp(ll_clean + fit, np.log(p_garbage) + OBS_UNIFORM_LL)

        # ---- merge identical states, keep best ----
        order = np.argsort(-sc, kind="stable")
        cand, code, par, sc = cand[order], code[order], par[order], sc[order]
        _, first = np.unique(cand.view(np.dtype((np.void, 54))).ravel(), return_index=True)
        keep = np.sort(first)[:beam]
        states, scores = cand[keep], sc[keep]
        back.append((par[keep], code[keep]))
        if progress and t % 25 == 0:
            progress(t, n)

    # ---- backtrack ----
    idx = int(np.argmax(scores))
    seq = []
    for t in range(n - 1, -1, -1):
        p, c = back[t]
        cc = int(c[idx])
        if cc >= 1000:
            i, j = pairs[cc - 1000]
            seq.append((t, [moves[i], moves[j]]))
        elif cc >= 0:
            seq.append((t, [moves[cc]]))
        idx = int(p[idx])
    seq.reverse()
    events = [{"frame": observations[t].get("frame_index", t), "move": mv}
              for t, mvs in seq for mv in mvs]
    final = states[int(np.argmax(scores))]
    return events, final, float(scores.max())


# =============================================================================
# V51: LIVE FRAME DECODER (replaces SequenceBufferedTracker for live tracking)
# =============================================================================
class _MoveNode:
    """One link in a hypothesis' move history (shared between hypotheses)."""
    __slots__ = ("moves", "frame", "parent", "depth")

    def __init__(self, moves, frame, parent):
        self.moves = moves
        self.frame = frame
        self.parent = parent
        self.depth = (parent.depth + 1) if parent is not None else 0


class LiveFrameDecoder:
    """
    V51: streaming version of the frame-level beam decoder, used as the LIVE
    tracking engine.

    Every camera frame updates a beam of complete-cube hypotheses. The
    displayed state/history is always the current best explanation of ALL
    frames seen so far, so an early misread that later proves impossible is
    corrected automatically instead of poisoning the state forever.

    A move is "confirmed" once every hypothesis within CONFIRM_MARGIN of the
    best one agrees on it; the rest of the history is "provisional" and may
    still be revised (typically for a few frames, longer after a hidden B).
    """

    def __init__(self, tracker, beam=200, move_cost=12.0, p_sticker=0.9,
                 p_garbage=0.4, second_move_cost=2.0, allow_double=True,
                 prune_margin=90.0, confirm_margin=20.0, confirm_contenders=30):
        self.tracker = tracker
        self.beam = int(beam)
        self.move_cost = float(move_cost)
        self.second_move_cost = float(second_move_cost)
        self.allow_double = bool(allow_double)
        self.prune_margin = float(prune_margin)
        self.confirm_margin = float(confirm_margin)
        self.confirm_contenders = int(confirm_contenders)

        self.ll_match = np.log(p_sticker)
        self.ll_miss = np.log((1 - p_sticker) / 5.0)
        self.ll_garbage = 9 * np.log(1 / 6.0) + np.log(p_garbage)
        self.ll_garbage_frame = OBS_UNIFORM_LL + np.log(p_garbage)
        self.ll_clean = np.log(1 - p_garbage)

        # V52: all 36 quarter moves (faces, slices, rotations, wide moves).
        self.moves, self.extra_cost, self.pairs = decoder_move_set(tracker.permutations)
        self.P = np.stack([np.asarray(tracker.permutations[m]) for m in self.moves])
        self.move_index = {m: i for i, m in enumerate(self.moves)}
        self.P2 = np.stack([self.P[i][self.P[j]] for i, j in self.pairs])
        self.pair_extra = np.array([self.extra_cost[i] + self.extra_cost[j] for i, j in self.pairs])
        self.lock = threading.Lock()
        self.reset(tracker.state, tracker.move_history)

    # ------------------------------------------------------------------ setup
    def reset(self, base_state, base_history=None, any_orientation=False):
        """V59: with any_orientation (solved cube only), all 24 ways of holding
        it start as candidates; the camera and the first side turns decide."""
        with self.lock:
            self.base_history = list(base_history or [])
            starts = [np.asarray(base_state)]
            if any_orientation:
                # V60: also for a scanned scrambled cube (the camera view picks the orientation)
                starts = all_orientations(np.asarray(base_state), self.tracker.permutations)
            self.states = np.array([[DECODER_CIDX[str(c)] for c in st] for st in starts], dtype=np.uint8)
            self.scores = np.zeros(len(starts))
            self.nodes = [None] * len(starts)
            self.any_orientation_start = len(starts) > 1
            self.center = str(base_state[22])
            self.frames_seen = 0
            self.frames_used = 0
            self.last_step_ms = 0.0
            self.max_step_ms = 0.0
            self.confirmed_moves = 0
            self._best_cache = None

    # --------------------------------------------------------------- stepping
    # V52: guide prior. A move the active guide expects next (from the exact
    # cube state it expects it in) costs less; an expected move that does not
    # change the visible front (B, S, ...) is presumed done, because the camera
    # can never confirm it. Unexpected moves keep their normal cost, so going
    # off-plan is still tracked and a wrong presumption is corrected later.
    GUIDE_VISIBLE_COST_FACTOR = 0.35
    GUIDE_HIDDEN_COST = 0.5          # cheap, but only added when later evidence needs it
    GUIDE_HIDDEN_FINAL_COST = -1.0   # last move(s) of a plan and invisible: presume done
    # V59: while a guide plan is active, rotations/slice/wide moves that are not
    # in the plan are unlikely (the user is following face turns).
    GUIDE_UNPLANNED_EXTENDED_EXTRA = 10.0

    # V53: moves are much less likely while the motion detector sees nothing.
    # Glare/misreads appear on a STILL cube; real turns create a motion burst.
    QUIET_MOVE_COST_FACTOR = 3.0
    MOTION_WINDOW_FRAMES = 3

    def step(self, grid, frame_index=None, expected=None, probs=None, motion_burst=None):
        """Feed ONE camera frame (grid or None). Returns True if the best
        history changed. `expected`: {state_bytes: [(move, hidden), ...]}."""
        t0 = time.perf_counter()
        with self.lock:
            self.frames_seen += 1
            g = None
            if grid is not None:
                flat = [c for row in grid for c in row] if isinstance(grid[0], (list, tuple)) else list(grid)
                # V52: centres can move now (M/E/S, x/y/z, wide moves), so the
                # centre sticker is scored like any other; a whole-cube
                # rotation is tracked instead of ignored.
                if len(flat) == 9 and all(str(c) in DECODER_CIDX for c in flat):
                    g = np.array([DECODER_CIDX[str(c)] for c in flat], dtype=np.uint8)
            table = observation_loglik(grid, probs) if g is not None else None

            if motion_burst:
                self.frames_since_motion = 0
            else:
                self.frames_since_motion = getattr(self, "frames_since_motion", 999) + 1
            quiet = motion_burst is not None and self.frames_since_motion > self.MOTION_WINDOW_FRAMES
            cost_factor = self.QUIET_MOVE_COST_FACTOR if quiet else 1.0

            n = len(self.states)
            cand = [self.states]
            code = [np.full(n, -1)]
            cost = [np.zeros(n)]
            par = [np.arange(n)]
            if self.frames_seen > 1:
                s1 = self.states[:, self.P].reshape(-1, 54)
                cand.append(s1)
                code.append(np.tile(np.arange(len(self.moves)), n))
                per_move = (self.move_cost + self.extra_cost) * cost_factor
                if expected:
                    per_move = per_move + np.array(
                        [0.0 if m in DECODER_FACE_MOVES else self.GUIDE_UNPLANNED_EXTENDED_EXTRA for m in self.moves]
                    )
                single_cost = np.tile(per_move, n)
                if expected:
                    for i in range(n):
                        for mv, mode in expected.get(self.states[i].tobytes(), ()):
                            if mv in self.move_index:
                                single_cost[i * len(self.moves) + self.move_index[mv]] = {
                                    "visible": self.move_cost * self.GUIDE_VISIBLE_COST_FACTOR,
                                    "hidden": self.GUIDE_HIDDEN_COST,
                                    "hidden_final": self.GUIDE_HIDDEN_FINAL_COST,
                                }[mode]
                cost.append(single_cost)
                par.append(np.repeat(np.arange(n), len(self.moves)))
                # Two moves in one frame gap: only worth the ~10x cost when the
                # frame is visible AND no 0/1-move candidate explains it well
                # (fast double turns at ~11 fps).
                if self.allow_double and g is not None:
                    best_single = int(max(
                        (self.states[:, 18:27] == g).sum(axis=1).max(),
                        (s1[:, 18:27] == g).sum(axis=1).max(),
                    ))
                    if best_single < 8:
                        s2 = self.states[:, self.P2].reshape(-1, 54)
                        cand.append(s2)
                        code.append(np.tile(np.arange(len(self.pairs)) + 1000, n))
                        pair_cost = (2 * self.move_cost + self.second_move_cost + self.pair_extra) * cost_factor
                        if expected:
                            pair_cost = pair_cost + np.array([
                                (0.0 if self.moves[i] in DECODER_FACE_MOVES else self.GUIDE_UNPLANNED_EXTENDED_EXTRA)
                                + (0.0 if self.moves[j] in DECODER_FACE_MOVES else self.GUIDE_UNPLANNED_EXTENDED_EXTRA)
                                for i, j in self.pairs
                            ])
                        cost.append(np.tile(pair_cost, n))
                        par.append(np.repeat(np.arange(n), len(self.pairs)))
            cand = np.concatenate(cand)
            code = np.concatenate(code)
            par = np.concatenate(par)
            sc = self.scores[par] - np.concatenate(cost)

            if table is not None:
                self.frames_used += 1
                fit = table[_OBS_ROWS, cand[:, 18:27]].sum(axis=1)
                sc = sc + np.logaddexp(self.ll_clean + fit, self.ll_garbage_frame)

            order = np.argsort(-sc, kind="stable")
            cand, code, par, sc = cand[order], code[order], par[order], sc[order]
            _, first = np.unique(cand.view(np.dtype((np.void, 54))).ravel(), return_index=True)
            keep = np.sort(first)
            keep = keep[sc[keep] >= sc[keep[0]] - self.prune_margin][: self.beam]

            new_nodes = []
            for idx in keep:
                parent = self.nodes[par[idx]]
                c = int(code[idx])
                if c < 0:
                    new_nodes.append(parent)
                elif c >= 1000:
                    i, j = self.pairs[c - 1000]
                    new_nodes.append(_MoveNode((self.moves[i], self.moves[j]), frame_index, parent))
                else:
                    new_nodes.append(_MoveNode((self.moves[c],), frame_index, parent))

            prev_best = self.nodes[0]
            self.states = cand[keep]
            self.scores = sc[keep] - sc[keep[0]]
            self.nodes = new_nodes
            changed = self.nodes[0] is not prev_best
            if changed:
                self._best_cache = None
            self._update_confirmed()

        ms = (time.perf_counter() - t0) * 1000.0
        self.last_step_ms = ms
        self.max_step_ms = max(self.max_step_ms, ms)
        return changed

    # ------------------------------------------------------------ inspection
    @staticmethod
    def _chain(node):
        out = []
        while node is not None:
            out.extend(reversed(node.moves))
            node = node.parent
        out.reverse()
        return out

    def _update_confirmed(self):
        best = self.nodes[0]
        on_best = {}
        node = best
        while node is not None:
            on_best[id(node)] = node.depth
            node = node.parent
        confirmed_depth = best.depth if best is not None else -1
        for i in range(1, min(len(self.nodes), self.confirm_contenders)):
            if self.scores[i] < -self.confirm_margin:
                break
            node = self.nodes[i]
            while node is not None and id(node) not in on_best:
                node = node.parent
            meet = node.depth if node is not None else -1
            confirmed_depth = min(confirmed_depth, meet)
        # Translate node depth into a quarter-turn count.
        count = 0
        node = best
        while node is not None:
            if node.depth <= confirmed_depth:
                count += len(node.moves)
            node = node.parent
        self.confirmed_moves = count

    def best_moves(self):
        with self.lock:
            if self._best_cache is None:
                self._best_cache = self._chain(self.nodes[0])
            return list(self._best_cache)

    def best_state_strings(self):
        with self.lock:
            return np.array([DECODER_COLORS[i] for i in self.states[0]], dtype="<U10")

    def best_history(self):
        moves = self.best_moves()
        if getattr(self, "any_orientation_start", False) and not self.base_history:
            moves = strip_leading_rotations(moves)
        return self.tracker.compressed_history(
            tidy_history(self.base_history + moves)
        )

    def orientation_resolved(self):
        """False while candidates within the confirm margin disagree on which
        colour is where (e.g. top colour not yet revealed by a side turn)."""
        with self.lock:
            close = self.states[self.scores >= -self.confirm_margin]
            centres = {tuple(st[[4, 13, 22, 31, 40, 49]]) for st in close}
            return len(centres) <= 1

    def stats(self):
        with self.lock:
            contenders = int(np.sum(self.scores >= -self.confirm_margin))
            return {
                "hypotheses": int(len(self.states)),
                "contenders": contenders,
                "confirmed_moves": int(self.confirmed_moves),
                "frames_seen": int(self.frames_seen),
                "frames_used": int(self.frames_used),
                "last_step_ms": float(self.last_step_ms),
                "max_step_ms": float(self.max_step_ms),
            }


import os
import tempfile


class FinalSolveAnalyzer:
    """Video re-analysis ([e]): the frame-level beam decoder over every
    recorded frame. Returns a verified state/history or ok=False."""

    DECODER_MIN_EXPLAINED = 0.50

    def __init__(self, tracker):
        self.tracker = tracker

    def _best_alignment_score(self, observations, base_state, history):
        """Frames a move history can explain exactly under its best timing."""
        quarter = []
        for mv in history:
            quarter += [mv[0], mv[0]] if str(mv).endswith("2") else [mv]
        state = base_state.copy()
        fronts = [list(state[18:27])]
        for mv in quarter:
            state = self.tracker.apply_move_token(state, mv)
            if state is None:
                return 0
            fronts.append(list(state[18:27]))
        obs = [
            [str(c) for row in o["grid"] for c in row] if o.get("grid") else None
            for o in observations
        ]
        K = len(fronts)
        NEG = -10 ** 9
        dp = [NEG] * K
        dp[0] = 0
        for g in obs:
            new = [NEG] * K
            for k in range(K):
                best = max(dp[k], dp[k - 1] if k >= 1 else NEG, dp[k - 2] if k >= 2 else NEG)
                if best <= NEG // 2:
                    continue
                new[k] = best + (1 if (g is not None and g == fronts[k]) else 0)
            dp = new
        return int(dp[K - 1])

    def analyze_frame_decoder(self, observations, base_state, base_history,
                              live_history_at_stop=None, live_move_events=None,
                              progress_callback=None):
        started = time.perf_counter()
        base_history = list(base_history or [])
        live_history = list(live_history_at_stop or base_history)
        live_suffix = live_history[len(base_history):]

        def _progress(t, n):
            if progress_callback is None:
                return
            try:
                progress_callback(t, n, f"Frame decoder {t}/{n}", {"phase": "FRAME DECODER"})
            except Exception:
                pass

        events, _final_codes, _score = decode_front_video(
            observations, base_state, self.tracker.permutations,
            beam=300, move_cost=12.0, p_garbage=0.4, p_sticker=0.9,
            allow_double=True, progress=_progress,
        )
        moves = tidy_history([e["move"] for e in events])
        if is_solved_state(base_state) and not base_history:
            moves = strip_leading_rotations(moves)
        history = self.tracker.compressed_history(base_history + moves)
        state = self.tracker.state_from_history(history)
        # base_history may not start from solved (e.g. after a reset mid-way);
        # rebuild from the recording's own base state to be safe.
        state_from_base = self.tracker.apply_move_sequence(base_state.copy(), moves)
        if state_from_base is not None:
            state = state_from_base
        # V59: the decoder may have found the cube was held in another
        # orientation from the start; its own final state is authoritative.
        if _final_codes is not None:
            state = np.array([DECODER_COLORS[i] for i in _final_codes], dtype="<U10")

        valid_frames = sum(1 for o in observations if o.get("grid"))
        start_for_scoring = base_state
        if state is not None and is_solved_state(base_state):
            # undo the decoded moves to find the orientation it started in
            back = state.copy()
            for mv in reversed(moves):
                back = back[np.argsort(np.asarray(self.tracker.permutations[mv]))] if mv in self.tracker.permutations else back
            start_for_scoring = back
        explained = self._best_alignment_score(observations, start_for_scoring, history[len(base_history):])
        live_explained = self._best_alignment_score(observations, base_state, live_suffix)
        share = explained / float(max(1, valid_frames))

        suffix = list(history[len(base_history):])
        matcher = SequenceMatcher(a=live_suffix, b=suffix, autojunk=False)
        matched = sum(size for _, _, size in matcher.get_matching_blocks())

        ok = bool(
            state is not None
            and self.tracker.state_to_kociemba_string(state) is not None
            and share >= self.DECODER_MIN_EXPLAINED
            and explained >= live_explained
        )
        reason = (
            f"FRAME DECODER explains {explained}/{valid_frames} frames "
            f"(live history: {live_explained})"
        )
        print(f"🧠 [FRAME DECODER] {reason} | {len(suffix)} moves | "
              f"{(time.perf_counter() - started):.1f}s | accepted={ok}")
        print("   Live:    " + " ".join(live_suffix))
        print("   Decoded: " + " ".join(suffix))

        return {
            "ok": ok,
            "reason": None if ok else "NOT ACCEPTED: " + reason,
            "decoder_summary": reason,
            "state": state.copy() if state is not None else base_state.copy(),
            "history": history,
            "new_moves": suffix,
            "events": events,
            "live_history_at_stop": live_history,
            "live_suffix": live_suffix,
            "final_suffix": suffix,
            "corrected": suffix != live_suffix,
            "final_exact": True,
            "correction_method": "FRAME-LEVEL BEAM DECODER",
            "explained_frames": explained,
            "live_explained_frames": live_explained,
            "valid_frames": valid_frames,
            "live_validation": {
                "matched": int(matched),
                "disagreed": int(max(0, len(live_suffix) - matched)),
                "ambiguous": 0,
                "unvalidated": 0,
                "total": len(live_suffix),
            },
            "validation_rows": [],
            "processed_checkpoints": 0,
            "skipped_checkpoints": 0,
            "analysis_ms": (time.perf_counter() - started) * 1000.0,
        }

    def analyze(self, observations, base_state, base_history, live_history_at_stop=None,
                progress_callback=None, live_move_events=None):
        try:
            return self.analyze_frame_decoder(
                observations, base_state, base_history,
                live_history_at_stop=live_history_at_stop,
                live_move_events=live_move_events,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            traceback.print_exc()
            live_history = list(live_history_at_stop or base_history)
            return {"ok": False, "reason": f"FRAME DECODER EXCEPTION: {exc}",
                    "live_history_at_stop": live_history,
                    "live_suffix": live_history[len(base_history):]}


class SlowMoBufferProcessor:
    # V50: was a hard-coded 42 s. With the retry/motion fixes each frame is far
    # cheaper, but a long solve still needs more than 42 s to re-read fully.
    # If this budget is hit the result is marked and NOT treated as complete.
    REANALYSIS_PERCEPTION_BUDGET_S = 120.0

    def __init__(self, cube_tracker, fps=120.0):
        self.cube_tracker = cube_tracker
        self.fps = fps
        self.is_recording = False
        self.analysis_in_progress = False
        self.recorded_frames = []
        self.recorded_frame_times = []  # V50: real capture time of each frame
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
        self.last_completed_frame_times = []
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
            self.recorded_frame_times = []
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

    def add_frame(self, frame, timestamp=None):
        with self.lock:
            if self.is_recording:
                self.recorded_frames.append(frame.copy())
                self.recorded_frame_times.append(
                    float(timestamp) if timestamp is not None else time.perf_counter()
                )

    def add_observation(self, frame_index, grid, motion_evidence, extra=None):
        """Store lightweight per-frame evidence for the final offline pass."""
        with self.lock:
            if not self.is_recording:
                return
            self.recorded_observations.append({
                "frame_index": frame_index,
                "observation_time": time.perf_counter(),
                "grid": copy.deepcopy(grid) if grid is not None else None,
                "motion_evidence": copy.deepcopy(motion_evidence or {}),
                "extra": extra or {},
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
            frame_times_to_process = list(self.recorded_frame_times)
            observations_to_process = list(self.recorded_observations)
            self.recorded_frames = []
            self.recorded_frame_times = []
            self.recorded_observations = []
            # Keep a copy for the explicit E-key video reanalysis pass.
            # The frames are never modified after capture, so keep references
            # instead of a second full copy of every 1280x720 frame (V50).
            self.last_completed_frames = frames_to_process
            self.last_completed_frame_times = frame_times_to_process
            self.last_completed_observations = copy.deepcopy(observations_to_process)
            # V50: cv2.CAP_PROP_FPS reports what the driver was asked for (120),
            # not how fast this loop actually stored frames. Measure it.
            measured_fps = None
            if len(frame_times_to_process) >= 2:
                span = frame_times_to_process[-1] - frame_times_to_process[0]
                if span > 0:
                    measured_fps = (len(frame_times_to_process) - 1) / span
            self.last_completed_source_fps = float(measured_fps or source_fps or 30.0)
            print(
                f"📏 [RECORDING]: {len(frame_times_to_process)} frames, measured "
                f"{self.last_completed_source_fps:.1f} fps (driver reported {source_fps})."
            )
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
        self._dump_live_session(observations_to_process, base_state, base_history, live_history_at_stop)

        # Make the replay available IMMEDIATELY after SPACE.  The old V31
        # implementation only populated replay_queue after the deep analysis
        # had finished, which made the application look frozen while the search
        # was running.
        for frame in frames_to_process:
            self.replay_queue.put(frame)
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
                "message": "Press E to fully re-run YOLO, grid extraction, motion and temporal correction.",
                "phase": "READY",
                "started_at": None,
                "elapsed_s": 0.0,
                "yolo_ok": 0,
                "grid_valid": 0,
                "yolo_retry": 0,
                "motion_bursts": 0,
            }
            return

        # (V61: analysis only runs when you ask for it - Analyse / [e])

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

            frames = self.last_completed_frames
            frame_times = list(self.last_completed_frame_times)
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
            # Prefer the measured recording fps over the driver-reported value.
            fps = float(self.last_completed_source_fps or source_fps or 30.0)
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
                frame_times,
            ),
            daemon=True,
        )
        thread.start()
        return True


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
        frame_times=None,
    ):
        started = time.perf_counter()
        frame_times = list(frame_times or [])

        def _frame_time(idx):
            if 0 <= int(idx) < len(frame_times):
                return float(frame_times[int(idx)] - frame_times[0])
            return float(idx) / max(1.0, float(source_fps))
        observations = []
        selected = self._select_video_reanalysis_indices(len(frames), prior_observations)
        total_selected = len(selected)
        # Hard E-pass budget: leave ~8 seconds for the temporal correction layer
        # so the complete post-solve process stays below one minute.
        perception_deadline = started + float(self.REANALYSIS_PERCEPTION_BUDGET_S)
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
            # V46: motion does not decide whether a fresh state is valid. Use
            # temporal agreement only as a perception-quality preference.
            if previous_grid is not None:
                score += 4.0 * _grid_diff(grid, previous_grid)
            if variant_name == "ORIGINAL":
                score += 0.15
            elif variant_name == "WHITE_BALANCE":
                score += 0.10
            return score

        def _detect_quad_with_retry(display_frame):
            """Fresh YOLO pass, with one lower-confidence retry only when needed."""
            nonlocal yolo_retry_count

            def _run(conf):
                results = model(display_frame, conf=conf, verbose=False, device=YOLO_DEVICE)[0]
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

                motion_evidence = motion_gate.process_frame(
                    display_frame, quad_corners, timestamp=_frame_time(frame_idx)
                )
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
                # V50: the old check ignored motion, so every real move (which
                # always changes >=3 stickers) triggered 4 extra YOLO passes.
                suspicious_stationary_grid = bool(
                    validate_face_grid(current_grid)
                    and previous_fresh_grid is not None
                    and _grid_diff(current_grid, previous_fresh_grid) >= 3
                    and not bool((motion_evidence or {}).get("burst", False))
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

                        # V50: do NOT feed enhanced variant images into the
                        # motion gate. Doing so overwrote its previous-frame
                        # buffer, so the NEXT real frame was compared against a
                        # CLAHE/gamma image and produced a fake burst.

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
                    "observation_time": _frame_time(frame_idx),
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

            def _analysis_progress(processed, total, message, telemetry=None):
                # Final reconstruction occupies the remaining 45% of the UI bar.
                elapsed = time.perf_counter() - started
                telemetry = dict(telemetry or {})
                self.analysis_status = {
                    "active": True,
                    "status": "VIDEO REANALYSIS RUNNING",
                    "progress": 55.0 + (float(processed) / max(1, total)) * 45.0,
                    "processed": int(processed),
                    "total": int(total),
                    "message": str(message),
                    "phase": str(telemetry.get("phase", "TEMPORAL CORRECTION")),
                    "started_at": started,
                    "elapsed_s": float(elapsed),
                    "yolo_ok": int(yolo_ok_count),
                    "grid_valid": int(grid_valid_count),
                    "yolo_retry": int(yolo_retry_count),
                    "motion_bursts": int(motion_burst_count),
                    **telemetry,
                }

            result = self.final_analyzer.analyze(
                observations,
                base_state,
                base_history,
                live_history_at_stop=live_history_at_stop,
                progress_callback=_analysis_progress,
                live_move_events=live_move_events_at_stop,
            )

            self._dump_reanalysis(
                observations, base_state, base_history, live_history_at_stop,
                live_move_events_at_stop, result, source_fps, total_selected,
                perception_time_limit_hit,
            )

            result["reanalysis"] = True
            result["raw_video_observations"] = len(observations)
            result["raw_video_frames_selected"] = len(selected)
            result["yolo_ok_count"] = yolo_ok_count
            result["grid_valid_count"] = grid_valid_count
            result["yolo_retry_count"] = yolo_retry_count
            result["motion_burst_count"] = motion_burst_count
            result["perception_time_limit_hit"] = bool(perception_time_limit_hit)
            if perception_time_limit_hit and result.get("ok"):
                # V50: a reconstruction of only the first part of the video must
                # not replace the live cube state (the main loop installs any
                # ok=True result as authoritative).
                result["ok"] = False
                result["reason"] = (
                    f"PARTIAL VIDEO: perception budget hit after {len(observations)}/"
                    f"{total_selected} frames; result not applied"
                )
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
                    "total": int(total_selected),
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
                    "total": int(total_selected),  # V50: was len(observations), which hid budget cut-offs
                    "message": (
                        str(result.get("reason") or result.get("fallback_reason") or "unknown failure")
                        + (
                            f" | PERCEPTION BUDGET HIT: only {len(observations)}/{total_selected} frames analysed"
                            if perception_time_limit_hit else ""
                        )
                    ),
                    "phase": "FAILED",
                    "started_at": started,
                    "elapsed_s": float(time.perf_counter() - started),
                    "yolo_ok": int(yolo_ok_count),
                    "grid_valid": int(grid_valid_count),
                    "yolo_retry": int(yolo_retry_count),
                    "motion_bursts": int(motion_burst_count),
                }
                print(
                    "\n❌ [VIDEO REANALYSIS FAILED]: "
                    + str(result.get("reason") or result.get("fallback_reason") or "unknown failure")
                    + (f" (perception budget hit at {len(observations)}/{total_selected} frames)" if perception_time_limit_hit else "")
                    + "\n"
                )

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


    def _dump_live_session(self, observations, base_state, base_history, live_history):
        """V54: save what the LIVE tracker saw during a recording (per-frame
        grid, colour confidence, glare, view status, motion) for offline replay."""
        try:
            import json, os
            keep = ("burst", "raw_energy", "whole_cube_translation_px",
                    "whole_cube_scale_delta", "whole_cube_angle_deg")
            payload = {
                "version": "V54-live",
                "base_state": [str(x) for x in base_state],
                "base_history": list(base_history),
                "live_history_at_stop": list(live_history or []),
                "observations": [
                    {
                        "frame_index": o.get("frame_index"),
                        "observation_time": o.get("observation_time"),
                        "grid": o.get("grid"),
                        "motion_evidence": {k: (o.get("motion_evidence") or {}).get(k) for k in keep},
                        "extra": o.get("extra", {}),
                    }
                    for o in observations
                ],
            }
            os.makedirs("reanalysis_dumps", exist_ok=True)
            path = os.path.join("reanalysis_dumps", time.strftime("live_%Y%m%d_%H%M%S.json"))
            with open(path, "w") as fh:
                json.dump(payload, fh, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
            print(f"💾 [LIVE SESSION]: Live tracking data saved to {os.path.abspath(path)}")
        except Exception as exc:
            print(f"⚠️ [LIVE SESSION]: Could not save dump: {exc}")

    def _dump_reanalysis(self, observations, base_state, base_history, live_history,
                         live_events, result, source_fps, total_selected, budget_hit):
        """V50: save everything the E-pass analyser saw, so a failed run can be
        replayed exactly offline (no camera/YOLO needed). Never raises."""
        try:
            import json, os
            def slim_evidence(ev):
                ev = ev or {}
                keep = ("burst", "structured", "raw_energy", "energy_jump",
                        "motion_baseline", "max_cell_mag", "active_cells",
                        "common_small", "whole_cube", "observed_at")
                return {k: ev.get(k) for k in keep if k in ev}
            payload = {
                "version": "V50",
                "source_fps": float(source_fps),
                "total_frames": int(total_selected),
                "perception_budget_hit": bool(budget_hit),
                "base_state": [str(x) for x in base_state],
                "base_history": list(base_history),
                "live_history_at_stop": list(live_history or []),
                "live_move_events": list(live_events or []),
                "observations": [
                    {
                        "frame_index": o.get("frame_index"),
                        "observation_time": o.get("observation_time"),
                        "grid": o.get("grid"),
                        "yolo_ok": o.get("yolo_ok"),
                        "yolo_retried": o.get("yolo_retried"),
                        "motion_evidence": slim_evidence(o.get("motion_evidence")),
                    }
                    for o in observations
                ],
                "result": {
                    k: result.get(k)
                    for k in ("ok", "reason", "fallback_reason", "history",
                              "checkpoints", "rejected_final_candidates",
                              "checkpoint_stats", "processed_checkpoints",
                              "skipped_checkpoints")
                },
            }
            os.makedirs("reanalysis_dumps", exist_ok=True)
            path = os.path.join(
                "reanalysis_dumps", time.strftime("reanalysis_%Y%m%d_%H%M%S.json")
            )
            with open(path, "w") as fh:
                json.dump(payload, fh, default=str)
            print(f"💾 [VIDEO REANALYSIS]: Full analyser input saved to {os.path.abspath(path)}")
        except Exception as exc:
            print(f"⚠️ [VIDEO REANALYSIS]: Could not save dump: {exc}")

    def consume_final_result(self):
        try:
            result = self.final_result_queue.get_nowait()
        except queue.Empty:
            return None
        # The main thread now owns the authoritative state update.
        self.analysis_in_progress = False
        return result


def tidy_history(history):
    """Drop adjacent cancelling pairs of NON-face moves (x x', y' y, M M' ...).
    These are almost always the cube being tilted and put back; the state is
    identical with or without them. Face-move pairs are left alone."""
    out = []
    for mv in history:
        if (
            out
            and mv not in DECODER_FACE_MOVES
            and DECODER_INVERSE.get(out[-1]) == mv
        ):
            out.pop()
            continue
        out.append(mv)
    return out


def is_state_solved(state):
    return all(len(set(state[i * 9:(i + 1) * 9])) == 1 for i in range(6))


def solve_state_with_kociemba(tracker, state):
    """Kociemba solution in THIS program's move names, verified by applying it.
    Returns (tokens, error_message)."""
    if is_state_solved(state):
        return [], None
    k_str = tracker.state_to_kociemba_string(state)
    if not k_str:
        return None, "Invalid colour configuration (check the tracked state)"
    try:
        raw = kociemba.solve(k_str)
    except Exception as exc:
        return None, f"Kociemba could not solve this state ({exc})"
    tokens = remap_kociemba_for_display(raw).split()
    final = tracker.apply_move_sequence(state, tokens)
    if final is None or not is_state_solved(final):
        return None, "Solution failed verification; not shown"
    return tokens, None


SCRAMBLE_FACES = ["R", "L", "U", "D", "F", "B"]
SCRAMBLE_AXIS = {"R": 0, "L": 0, "U": 1, "D": 1, "F": 2, "B": 2}


def random_scramble(length=20, rng=None):
    """WCA-style random-move scramble: never the same face twice in a row and
    never three turns on the same axis in a row."""
    import random as _random
    rng = rng or _random
    out = []
    while len(out) < length:
        face = rng.choice(SCRAMBLE_FACES)
        if out and out[-1][0] == face:
            continue
        if len(out) >= 2 and SCRAMBLE_AXIS[out[-1][0]] == SCRAMBLE_AXIS[face] == SCRAMBLE_AXIS[out[-2][0]]:
            continue
        out.append(face + rng.choice(["", "'", "2"]))
    return out


PRESET_SCRAMBLES = [
    ("Checkerboard", "M2 E2 S2"),
    ("Superflip", "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"),
    ("Cube in a cube", "F L F U' R U F2 L2 U' L' B D' B' L2 U"),
    ("Slice practice", "M E S M' E' S' x y z"),
    ("Wide practice", "r U r' u R u' f R f'"),
    ("Practice scramble A", "R2 D' B2 U F2 R2 D2 B2 U' R' B' L F' R2 U' F L2 D' F'"),
    ("Practice scramble B", "D2 F2 U' B2 L2 D R2 D' F2 U2 R' B' U2 L' D' F' L2 U' R2 B"),
    ("Practice scramble C", "U' L2 F2 D B2 U2 R2 U' B2 D' L B U2 F' R D' B' L' U2 F2"),
]


class FrameQualityGate:
    """V53: decide whether a camera frame is a fair look at a still cube.
    Frames taken while the WHOLE cube is being moved (picked up, repositioned,
    shaken) or seen at a steep angle are ignored, plus a short settle period,
    so they can never be explained as moves. Turning a layer keeps the cube's
    outline roughly still, so normal turning is unaffected."""
    MOVE_FRACTION = 0.15      # V54: was 0.07; whole-cube shift per frame vs face size
    MOVE_SCALE = 0.12         # V54: was 0.06
    SETTLE_FRAMES = 1         # V54: was 2
    MAX_SIDE_RATIO = 2.2      # V54: was 1.7; steeper views than this are ignored
    MIN_AREA_FRACTION = 0.01

    def __init__(self):
        self.settle = 0
        self.last_reason = "OK"

    def check(self, quad, motion_evidence, frame_shape):
        reason = "OK"
        if quad is None:
            reason = "NO CUBE"
        else:
            q = np.asarray(quad, dtype=float).reshape(4, 2)
            sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
            side = float(np.mean(sides))
            area = float(cv2.contourArea(q.astype(np.float32)))
            ev = motion_evidence or {}
            if max(sides) / max(min(sides), 1e-6) > self.MAX_SIDE_RATIO:
                reason = "STEEP ANGLE"
            elif area < self.MIN_AREA_FRACTION * frame_shape[0] * frame_shape[1]:
                reason = "TOO SMALL"
            elif (
                # V54: rotation of the outline is NOT used: an F turn rotates it.
                float(ev.get("whole_cube_translation_px", 0.0)) > self.MOVE_FRACTION * side
                or float(ev.get("whole_cube_scale_delta", 0.0)) > self.MOVE_SCALE
            ):
                reason = "CUBE MOVING"
        if reason == "CUBE MOVING":
            self.settle = self.SETTLE_FRAMES
        elif reason == "OK" and self.settle > 0:
            self.settle -= 1
            reason = "SETTLING"
        self.last_reason = reason
        return reason == "OK", reason


class MoveGuide:
    """Shows a move sequence (scramble or solution) and follows the tracked
    cube through it. Progress is STATE-based: after each quarter turn the
    guide knows exactly which cube state to expect, so it advances when the
    cube reaches it (R2 done as R' R' also counts) and flags wrong moves."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.clear()

    def clear(self):
        self.active = False
        self.kind = ""
        self.title = ""
        self.tokens = []
        self.quarter_owner = []   # quarter step -> token index
        self.targets = []         # state after 0..n quarter steps
        self.progress = 0         # quarter steps completed
        self.on_track = True
        self.fix = []
        self.anchor_history_len = 0
        self.message = ""
        self.completed_at = None
        self.expected = {}

    def _build_targets(self, start_state):
        state = start_state.copy()
        targets = [state.copy()]
        for tok in self.tokens:
            for q in ([tok[:-1]] * 2 if tok.endswith("2") else [tok]):
                state = self.tracker.apply_move_token(state, q)
                targets.append(state.copy())
        return targets

    def load(self, kind, title, tokens, start_state, history):
        self.clear()
        self.active = True
        self.kind = kind
        self.title = title
        self.tokens = list(tokens)
        # V59: on a solved cube the tracker may not yet know how it is held,
        # so prepare the plan for every orientation; update() follows the one
        # the cube turns out to be in.
        starts = [start_state.copy()]
        if is_solved_state(start_state):
            starts = all_orientations(start_state.copy(), self.tracker.permutations)
        self.variants = [self._build_targets(st) for st in starts]
        self.start_state = start_state.copy()
        self.quarter_moves = []
        for tok in self.tokens:
            self.quarter_moves += [tok[:-1]] * 2 if tok.endswith("2") else [tok]
        self._reanchor_key = None
        state = start_state.copy()
        self.targets = [state.copy()]
        for ti, tok in enumerate(self.tokens):
            quarters = [tok[:-1]] * 2 if tok.endswith("2") else [tok]
            for q in quarters:
                state = self.tracker.apply_move_token(state, q)
                self.targets.append(state.copy())
                self.quarter_owner.append(ti)
        self.anchor_history_len = len(history)
        if not self.tokens:
            self.completed_at = time.perf_counter()
        self._build_expected()

    def _build_expected(self):
        # Decoder prior: state-before (colour codes) -> expected quarter move.
        self.expected = {}
        quarter_moves = []
        for tok in self.tokens:
            quarter_moves += [tok[:-1]] * 2 if tok.endswith("2") else [tok]
        hidden_flags = [
            bool(np.array_equal(self.targets[k][18:27], self.targets[k + 1][18:27]))
            for k in range(len(quarter_moves))
        ]
        for variant in getattr(self, "variants", [self.targets]):
            hidden_flags = [
                bool(np.array_equal(variant[k][18:27], variant[k + 1][18:27]))
                for k in range(len(quarter_moves))
            ]
            for k, mv in enumerate(quarter_moves):
                before = variant[k]
                key = np.array([DECODER_CIDX[str(c)] for c in before], dtype=np.uint8).tobytes()
                if not hidden_flags[k]:
                    mode = "visible"
                elif all(hidden_flags[k:]):
                    mode = "hidden_final"   # nothing visible follows: presume it
                else:
                    mode = "hidden"         # filled in once the next visible move needs it
                if (mv, mode) not in self.expected.get(key, []):
                    self.expected.setdefault(key, []).append((mv, mode))


    def decoder_prior(self):
        if not self.active or self.complete:
            return None
        return self.expected

    @property
    def complete(self):
        return self.active and self.progress >= len(self.targets) - 1

    def token_progress(self):
        """(tokens fully done, current token index, current token half done)."""
        if not self.quarter_owner:
            return 0, 0, False
        if self.progress >= len(self.quarter_owner):
            return len(self.tokens), len(self.tokens), False
        cur = self.quarter_owner[self.progress]
        half = self.progress > 0 and self.quarter_owner[self.progress - 1] == cur
        return cur, cur, half

    def update(self, state, history):
        if not self.active:
            return
        matches = [k for k, t in enumerate(self.targets) if np.array_equal(state, t)]
        if not matches:
            matches = self._try_reanchor(state)
        if not matches and len(getattr(self, "variants", [])) > 1:
            # V59: the cube may be held in another orientation than assumed;
            # switch to the plan variant the tracked cube actually matches.
            for variant in self.variants:
                vm = [k for k, t in enumerate(variant) if np.array_equal(state, t)]
                if vm:
                    self.targets = variant
                    matches = vm
                    break
        if matches:
            # Prefer the match nearest the current progress (sequences can
            # revisit a state, e.g. R R R R), ties go forward.
            best = min(matches, key=lambda k: (abs(k - self.progress), -k))
            self.progress = best
            self.on_track = True
            self.fix = []
            self.anchor_history_len = len(history)
            if self.complete and self.completed_at is None:
                self.completed_at = time.perf_counter()
        else:
            self.on_track = False
            key = state.tobytes()
            if key != getattr(self, "_fix_key", None):
                self._fix_key = key
                self.fix = self._search_fix(state) or self._history_fix(history)

    def _try_reanchor(self, state):
        """V59: the tracker may revise what it believed when the plan was
        loaded (e.g. drop a phantom move from glare). If the tracked cube fits
        the plan from a corrected start - a solved cube in any orientation, or
        one move away from the assumed start - rebuild the plan from there.
        Returns [step] when re-anchored, else []."""
        key = state.tobytes()
        if key == self._reanchor_key:
            return []
        self._reanchor_key = key
        P = self.tracker.permutations
        if any(m not in P for m in self.quarter_moves):
            return []
        inverse = {m: np.argsort(np.asarray(P[m])) for m in set(self.quarter_moves)}
        near_start = {self.start_state.tobytes()}
        for m in getattr(self.tracker, "ALL_QUARTER_MOVES", list(P.keys())):
            if m in P:
                near_start.add(self.start_state[np.asarray(P[m])].tobytes())
        # Undo the plan's quarter turns from the end; after undoing all turns
        # after step k we have the start the plan would need if the cube is
        # at step k right now. Prefer the furthest step that fits.
        n = len(self.quarter_moves)
        for k in range(n, -1, -1):
            s0 = state.copy()
            for mv in reversed(self.quarter_moves[:k]):
                s0 = s0[inverse[mv]]
            if is_solved_state(s0) or s0.tobytes() in near_start:
                self.start_state = s0.copy()
                self.targets = self._build_targets(s0)
                self.variants = [self.targets]
                self._build_expected()   # the tracker must be told the corrected plan
                return [k]
        return []

    def _history_fix(self, history):
        extra = list(history[self.anchor_history_len:])
        return [mv if mv.endswith("2") else DECODER_INVERSE.get(mv, mv) for mv in reversed(extra)]

    def _search_fix(self, state):
        """Shortest (<=2 quarter turns) way from the current cube state back to
        the plan: to the last on-track step or any later step. Face moves are
        tried first so the hint is the simplest thing to do."""
        goals = {}
        for k in range(self.progress, len(self.targets)):
            goals.setdefault(self.targets[k].tobytes(), k)
        P = self.tracker.permutations
        face = [m for m in DECODER_FACE_MOVES if m in P]
        other = [m for m in getattr(self.tracker, "ALL_QUARTER_MOVES", face) if m not in face]
        for pool in (face, face + other):
            best = None
            for a in pool:
                s1 = state[P[a]]
                k = goals.get(s1.tobytes())
                if k is not None and (best is None or len(best[0]) > 1 or k > best[1]):
                    best = ([a], k)
            if best:
                return best[0]
            for a in pool:
                s1 = state[P[a]]
                for b in pool:
                    if DECODER_INVERSE.get(a) == b:
                        continue
                    k = goals.get(s1[P[b]].tobytes())
                    if k is not None and (best is None or k > best[1]):
                        best = ([a, b], k)
            if best:
                seq = best[0]
                return [seq[0] + "2"] if seq[0] == seq[1] else seq
        return None


# =============================================================================
# FEATURES: move names for your camera, colour memory, scan-to-start, timer,
# trainer. None of this changes colour reading, move detection or the guide.
# =============================================================================
import itertools as _itertools
import json as _json
import os as _os
import random as _random

CAMERA_SETUP_FILE = "camera_setup.json"
COLOUR_CALIBRATION_FILE = "colour_calibration.json"
SOLVES_FILE = "solves.json"

# Standard notation in the user's frame (x = right, y = up, z = towards the
# user). (axis, layers, direction): direction -1 = clockwise seen from the
# +axis side, i.e. a -90 degree turn about +axis.
USER_MOVE_SPECS = {
    "R": (0, [1], -1), "L": (0, [-1], 1), "U": (1, [1], -1), "D": (1, [-1], 1),
    "F": (2, [1], -1), "B": (2, [-1], 1),
    "M": (0, [0], 1), "E": (1, [0], 1), "S": (2, [0], -1),
    "x": (0, [-1, 0, 1], -1), "y": (1, [-1, 0, 1], -1), "z": (2, [-1, 0, 1], -1),
    "r": (0, [1, 0], -1), "l": (0, [-1, 0], 1), "u": (1, [1, 0], -1),
    "d": (1, [-1, 0], 1), "f": (2, [1, 0], -1), "b": (2, [-1, 0], 1),
}


def _all_frames():
    frames = []
    for perm in _itertools.permutations(range(3)):
        for signs in _itertools.product((1, -1), repeat=3):
            Q = np.zeros((3, 3), dtype=int)
            for row, (col, sg) in enumerate(zip(perm, signs)):
                Q[row, col] = sg
            frames.append(Q)
    return frames


def _build_user_table(tracker, Q):
    """user quarter move name -> the tracker's internal move name, for a user
    frame given by Q (model_vector = Q @ user_vector)."""
    stickers = tracker._model_stickers()
    where = {st: i for i, st in enumerate(stickers)}
    phi = np.array(tracker.STICKER_TO_MODEL)
    inv_phi = np.argsort(phi)
    lookup = {}
    for name, perm in tracker.permutations.items():
        lookup.setdefault(np.asarray(perm).tobytes(), name)
    table = {}
    for base, (axis, layers, sign) in USER_MOVE_SPECS.items():
        for prime in (False, True):
            k = sign * (-1 if prime else 1)
            Ru = tracker._rotation_matrix(axis, k)
            model = np.arange(54)
            for i, (p, n) in enumerate(stickers):
                pu = Q.T @ np.array(p)
                if pu[axis] in layers:
                    p2 = tuple(int(v) for v in Q @ (Ru @ pu))
                    n2 = tuple(int(v) for v in Q @ (Ru @ (Q.T @ np.array(n))))
                    model[where[(p2, n2)]] = i
            name = lookup.get(inv_phi[model[phi]].tobytes())
            if name is None:
                return None
            table[base + ("'" if prime else "")] = name
    return table


# The camera looks at the model's +z face.
#   "physical": the real geometry, used for STANDARD-notation algorithms (trainer,
#               alg.cubing.net links). Front camera: you look at the cube from the
#               other side (a y2 turn): your front is the face towards you.
#   "display":  the names you see and follow. Front camera: the names used since
#               V52 - the face the camera sees is F and every face turn is named
#               as it looks on the (mirrored) screen. None = the tracker's names.
CAMERA_PRESETS = {
    "camera facing you": {
        "physical": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        "display": None,
        "mirror": True,
        "label": "Front camera",
        "hint": "F = face the camera sees",
    },
    "top-down": {
        # camera above, looking down at your U face, TOP of the camera image
        # towards you (V62.2: checked on the real setup - R, L, F, B all right)
        "physical": [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        "display": [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        "mirror": False,
        "label": "Top-down camera",
        "hint": "U = camera face, F = towards you",
    },
}


class MoveNaming:
    """Translates between the names you see (display), standard notation
    (physical, for algorithms) and the tracker's internal move names."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.frame_name = "camera facing you"
        self._apply(CAMERA_PRESETS[self.frame_name])
        self.load()

    # ---------------------------------------------------------------- tables
    def _table(self, Q):
        if Q is None:
            ref = _build_user_table(self.tracker, np.eye(3, dtype=int)) or {}
            return {k: k for k in ref if k in self.tracker.permutations}
        table = _build_user_table(self.tracker, np.array(Q))
        if table is None:
            raise RuntimeError("camera frame does not map onto the cube moves")
        return table

    def _apply(self, preset):
        self.preset = preset
        self.Q = None if preset["display"] is None else np.array(preset["display"])
        self.to_int = self._table(preset["display"])
        self.to_usr = {v: k for k, v in self.to_int.items()}
        self.std_int = self._table(preset["physical"])
        self.std_usr = {v: k for k, v in self.std_int.items()}
        self.mirror_view = bool(preset.get("mirror", True))
        self.label = preset.get("label", self.frame_name)
        self.hint = preset.get("hint", "")

    @staticmethod
    def _map_token(tok, table):
        tok = str(tok)
        if tok.endswith("2"):
            base = table.get(tok[:-1])
            return None if base is None else base.rstrip("'") + "2"
        return table.get(tok)

    def _map(self, tokens, table):
        out = []
        for t in tokens:
            m = self._map_token(t, table)
            out.append(m if m is not None else t)
        return out

    def to_internal(self, tokens):
        """names you see -> tracker names"""
        return self._map(tokens, self.to_int)

    def to_user(self, tokens):
        """tracker names -> names you see"""
        return self._map(tokens, self.to_usr)

    def std_to_internal(self, tokens):
        """standard notation (algorithms) -> tracker names"""
        return self._map(tokens, self.std_int)

    def to_standard(self, tokens):
        """tracker names -> standard notation (for alg.cubing.net)"""
        return self._map(tokens, self.std_usr)

    @property
    def frame_description(self):
        return f"{self.label}: {self.hint}" if self.hint else self.label

    # --------------------------------------------------------------- presets
    def use_preset(self, name):
        self.frame_name = name
        self._apply(CAMERA_PRESETS[name])
        self.save()
        print(f"📷 [CAMERA]: {self.frame_description}")

    def learn(self, internal_first, internal_second):
        """The user just did R then U. Find the camera frame in which that is
        what the tracker saw. Only WHICH layer turned is used (directions can be
        ambiguous on a solved cube). Proper rotations only."""
        base = lambda m: str(m).rstrip("'2")
        if base(internal_first) == "R" and base(internal_second) == "U":
            self.use_preset("camera facing you")
            return True
        for Q in _all_frames():
            if round(np.linalg.det(Q)) != 1:
                continue
            table = _build_user_table(self.tracker, Q)
            if table and base(table.get("R")) == base(internal_first) and base(table.get("U")) == base(internal_second):
                for name, preset in CAMERA_PRESETS.items():
                    if preset["display"] is not None and np.array_equal(np.array(preset["display"]), Q):
                        self.use_preset(name)
                        return True
                self.frame_name = "custom (learned)"
                self._apply({"physical": Q.tolist(), "display": Q.tolist(), "mirror": True,
                             "label": "Custom camera", "hint": "learned from your R and U"})
                self.save()
                return True
        return False

    def save(self):
        try:
            data = {"frame": self.frame_name}
            if self.frame_name not in CAMERA_PRESETS:
                data["Q"] = self.Q.tolist()
            with open(CAMERA_SETUP_FILE, "w") as fh:
                _json.dump(data, fh)
        except Exception as exc:
            print(f"⚠️ [CAMERA]: could not save: {exc}")

    def load(self):
        # V62.1: start in Front camera mode every time unless asked otherwise:
        # a top-down / learned mode left over from an earlier session silently
        # renamed every move (a top-row turn shown as "B").
        if not APP_SETTINGS.get("remember_camera_mode", False):
            return
        try:
            if not _os.path.exists(CAMERA_SETUP_FILE):
                return
            with open(CAMERA_SETUP_FILE) as fh:
                data = _json.load(fh)
            name = data.get("frame")
            if name in CAMERA_PRESETS:
                self.frame_name = name
                self._apply(CAMERA_PRESETS[name])
            elif data.get("Q") is not None:
                Q = data["Q"]
                self.frame_name = "custom (learned)"
                self._apply({"physical": Q, "display": Q, "mirror": True,
                             "label": "Custom camera", "hint": "learned from your R and U"})
            print(f"📷 [CAMERA]: {self.frame_description}")
        except Exception as exc:
            print(f"⚠️ [CAMERA]: could not load ({exc}); using {self.label}")


class CameraSetupSession:
    """[m]: the user turns R then U; the tracker's view of those two turns
    tells us where the camera is relative to the user. Only which layers
    turned is needed, so it does not wait for the turn directions to be
    certain (they can stay ambiguous on a solved cube)."""

    SETTLE_S = 0.8
    TIMEOUT_S = 12.0

    def __init__(self):
        self.active = False
        self.message = ""

    def start(self, decoder):
        self.active = True
        self.start_len = len(decoder.best_moves())
        self.started_at = time.perf_counter()
        self.last_moves = None
        self.last_change = time.perf_counter()
        self.message = "CAMERA SETUP: turn R (your right face, clockwise), then U"

    def update(self, decoder, naming):
        if not self.active:
            return
        now = time.perf_counter()
        moves = [m for m in decoder.best_moves()[self.start_len:] if m.rstrip("'2") not in ("x", "y", "z")]
        if moves != self.last_moves:
            self.last_moves = list(moves)
            self.last_change = now
        if len(moves) == 1:
            self.message = "CAMERA SETUP: good - now turn U (top face, clockwise)"
        if len(moves) >= 2 and now - self.last_change >= self.SETTLE_S:
            self.active = False
            if naming.learn(moves[0], moves[1]):
                self.message = f"Camera setup saved ({naming.frame_name}). Undo with U' R'."
            else:
                self.message = "Could not recognise R then U - press [m] and try again"
            print(f"📷 [CAMERA SETUP]: {self.message}")
        elif now - self.started_at > self.TIMEOUT_S and len(moves) < 2:
            self.active = False
            self.message = "Camera setup: I couldn't see two turns - press [m] and try again"
            print(f"📷 [CAMERA SETUP]: {self.message}")


def save_colour_calibration(cal):
    try:
        data = {
            "centres": {k: [float(x) for x in v] for k, v in cal.cluster_centers.items()},
            "sigma": float(getattr(cal, "sigma", 12.0)),
            "glare_L": float(getattr(cal, "glare_L", 240.0)),
        }
        with open(COLOUR_CALIBRATION_FILE, "w") as fh:
            _json.dump(data, fh)
        print(f"💾 [COLOURS]: calibration saved to {COLOUR_CALIBRATION_FILE}")
    except Exception as exc:
        print(f"⚠️ [COLOURS]: could not save calibration: {exc}")


def load_colour_calibration(cal):
    try:
        if not _os.path.exists(COLOUR_CALIBRATION_FILE):
            return False
        with open(COLOUR_CALIBRATION_FILE) as fh:
            data = _json.load(fh)
        cal.cluster_centers = {k: np.array(v, dtype=float) for k, v in data["centres"].items()}
        cal.sigma = float(data.get("sigma", 12.0))
        cal.glare_L = float(data.get("glare_L", 240.0))
        cal.is_calibrated = True
        cal.current_scan_idx = len(SCAN_ORDER)
        print(f"🎨 [COLOURS]: loaded saved calibration ({COLOUR_CALIBRATION_FILE}); press [v] to redo it")
        return True
    except Exception as exc:
        print(f"⚠️ [COLOURS]: could not load saved calibration: {exc}")
        return False


def restart_colour_calibration(cal):
    cal.scanned_faces = {}
    cal.current_scan_idx = 0
    cal.is_calibrated = False
    cal.cluster_centers = {}


class CubeGeometry:
    """Where each of the 54 state indices sits on the physical cube."""

    def __init__(self, tracker):
        stickers = tracker._model_stickers()
        phi = tracker.STICKER_TO_MODEL
        self.pos = [tuple(stickers[phi[i]][0]) for i in range(54)]
        self.nrm = [tuple(stickers[phi[i]][1]) for i in range(54)]
        self.centre_index = {self.nrm[i]: i for i in range(54) if sum(map(abs, self.pos[i])) == 1}
        groups = {}
        for i in range(54):
            if sum(map(abs, self.pos[i])) >= 2:
                groups.setdefault(self.pos[i], []).append(i)
        self.cubies = list(groups.values())
        # rotation permutations (24) as state permutations
        rots = [np.arange(54)]
        seen = {rots[0].tobytes()}
        frontier = [rots[0]]
        while frontier:
            p = frontier.pop()
            for r in ("x", "y", "z"):
                q = p[np.asarray(tracker.permutations[r])]
                if q.tobytes() not in seen:
                    seen.add(q.tobytes())
                    rots.append(q)
                    frontier.append(q)
        self.rotations = rots

    def centre(self, state, normal):
        return state[self.centre_index[normal]]

    def cross_on(self, state, n):
        for cubie in self.cubies:
            if len(cubie) != 2:
                continue
            if not any(self.nrm[i] == n for i in cubie):
                continue
            for i in cubie:
                if state[i] != self.centre(state, self.nrm[i]):
                    return False
        return True

    def f2l_on(self, state, n):
        ax = int(np.argmax(np.abs(n)))
        side = n[ax]
        for i in range(54):
            p = self.pos[i]
            if p[ax] in (side, 0) and state[i] != self.centre(state, self.nrm[i]):
                return False
        return True

    def oll_on(self, state, n):
        if not self.f2l_on(state, n):
            return False
        opp = tuple(-v for v in n)
        return all(state[i] == self.centre(state, opp) for i in range(54) if self.nrm[i] == opp)

    def stages(self, state):
        """(cross face or None, f2l face or None, oll done, solved)."""
        normals = list(self.centre_index.keys())
        cross = [n for n in normals if self.cross_on(state, n)]
        f2l = [n for n in cross if self.f2l_on(state, n)]
        oll = any(self.oll_on(state, n) for n in f2l)
        return (cross[0] if cross else None), (f2l[0] if f2l else None), oll, is_solved_state(state)


class ScrambleScanSession:
    """[i] then [b] six times: show each face of a scrambled cube to the
    camera, in any order and any way up. The only valid cube consistent with
    the six photos is found by trying every way up for each face."""

    def __init__(self, tracker, geometry):
        self.tracker = tracker
        self.geo = geometry
        self.active = False
        self.faces = {}
        self.message = ""

    def start(self):
        self.active = True
        self.faces = {}
        self.message = "SCAN: show any face to the camera and press [b] (0/6)"

    @staticmethod
    def stable_grid(recent):
        """Per-sticker majority vote over recent readings that show the same
        centre as the newest one (a single frame can have a misread)."""
        if not recent:
            return None
        newest = recent[-1]
        same = [g for g in recent if g[1][1] == newest[1][1]]
        flat = []
        for k in range(9):
            votes = {}
            for g in same:
                c = g[k // 3][k % 3]
                votes[c] = votes.get(c, 0) + 1
            flat.append(max(votes.items(), key=lambda kv: kv[1])[0])
        return [flat[0:3], flat[3:6], flat[6:9]]

    def capture(self, grid):
        if not self.active:
            return None
        flat = [str(c) for row in grid for c in row] if grid else []
        if len(flat) != 9 or not all(c in DECODER_CIDX for c in flat):
            self.message = "SCAN: no clear face in view - hold it still and press [b] again"
            return None
        centre = flat[4]
        self.faces[centre] = flat
        self.message = f"SCAN: got the {centre.upper()}-centre face ({len(self.faces)}/6) - show another and press [b]"
        if len(self.faces) < 6:
            return None
        state, n_valid = self.assemble()
        self.active = False
        if state is None:
            self.message = "SCAN FAILED: those 6 faces don't make a real cube (a sticker was misread?) - press [i] to rescan"
            return None
        self.message = "SCAN OK - tracking starts from the scanned cube"
        return state

    def assemble(self):
        ref = self.tracker.state_from_history([])
        options = []
        for colour, flat in self.faces.items():
            opts = []
            for rot in self.geo.rotations:
                if ref[rot[22]] == colour:
                    opts.append([int(rot[18 + k]) for k in range(9)])
            if len(opts) != 4:
                return None, 0
            options.append((flat, opts))
        valid_sets = sorted(tuple(sorted(ref[i] for i in cubie)) for cubie in self.geo.cubies)
        found = []
        for combo in _itertools.product(*[o[1] for o in options]):
            st = ref.copy()
            for (flat, _), idx in zip(options, combo):
                for k in range(9):
                    st[idx[k]] = flat[k]
            sets = sorted(tuple(sorted(st[i] for i in cubie)) for cubie in self.geo.cubies)
            if sets != valid_sets:
                continue
            k_str = self.tracker.state_to_kociemba_string(st)
            if not k_str:
                continue
            try:
                if is_solved_state(st):
                    found.append(st.copy())
                    continue
                kociemba.solve(k_str)
                found.append(st.copy())
            except Exception:
                continue
        if not found:
            return None, 0
        return found[0], len(found)


def scan_candidates(session, limit=24):
    """Every real cube that fits the six scanned photos (a small scramble can
    fit more than one way: a face that looks alike turned round). Simplest
    first (shortest solution). Tracking keeps all of them in play and the
    first turns decide - instead of silently guessing one."""
    ref = session.tracker.state_from_history([])
    options = []
    for colour, flat in session.faces.items():
        opts = []
        for rot in session.geo.rotations:
            if ref[rot[22]] == colour:
                opts.append([int(rot[18 + k]) for k in range(9)])
        if len(opts) != 4:
            return []
        options.append((flat, opts))
    valid_sets = sorted(tuple(sorted(ref[i] for i in cubie)) for cubie in session.geo.cubies)
    found = {}
    for combo in _itertools.product(*[o[1] for o in options]):
        st = ref.copy()
        for (flat, _), idx in zip(options, combo):
            for k in range(9):
                st[idx[k]] = flat[k]
        key = st.tobytes()
        if key in found:
            continue
        sets = sorted(tuple(sorted(st[i] for i in cubie)) for cubie in session.geo.cubies)
        if sets != valid_sets:
            continue
        if is_solved_state(st):
            found[key] = (0, st.copy())
            continue
        k_str = session.tracker.state_to_kociemba_string(st)
        if not k_str:
            continue
        try:
            found[key] = (len(kociemba.solve(k_str).split()), st.copy())
        except Exception:
            continue
    ranked = sorted(found.values(), key=lambda x: x[0])
    return [st for _, st in ranked[:limit]]


def decoder_reset_candidates(decoder, candidates, history=None):
    """Like LiveFrameDecoder.reset(any_orientation=True), but for several
    possible cubes at once (every way of holding each)."""
    starts, seen = [], set()
    for rank, cand in enumerate(candidates):
        for st in all_orientations(np.asarray(cand), decoder.tracker.permutations):
            key = st.tobytes()
            if key not in seen:
                seen.add(key)
                starts.append((rank, st))
    decoder.reset(candidates[0], history, any_orientation=True)
    with decoder.lock:
        decoder.states = np.array([[DECODER_CIDX[str(c)] for c in st] for _, st in starts], dtype=np.uint8)
        # a whisker of preference for the simpler explanation, never decisive
        decoder.scores = np.array([-0.5 * rank for rank, _ in starts], dtype=float)
        decoder.nodes = [None] * len(starts)
        decoder.any_orientation_start = True
        decoder._best_cache = None


def decoder_unsure(decoder):
    """True while the tracker still has more than one cube (or way of holding
    it) that fits everything seen so far."""
    with decoder.lock:
        close = decoder.states[decoder.scores >= -decoder.confirm_margin]
        return len({st.tobytes() for st in close}) > 1


def _html_token(t):
    return t.replace("'", "-")


def alg_cubing_link(scramble_tokens, solve_tokens, title="Solve"):
    setup = "_".join(_html_token(t) for t in scramble_tokens)
    alg = "_".join(_html_token(t) for t in solve_tokens)
    return f"https://alg.cubing.net/?setup={setup}&alg={alg}&title={title.replace(' ', '_')}"


def append_solve_record(record):
    try:
        data = []
        if _os.path.exists(SOLVES_FILE):
            with open(SOLVES_FILE) as fh:
                data = _json.load(fh)
        data.append(record)
        with open(SOLVES_FILE, "w") as fh:
            _json.dump(data, fh, indent=1)
    except Exception as exc:
        print(f"⚠️ [SOLVES]: could not save: {exc}")


PLL_ALGS = {
    "Aa": "x R' U R' D2 R U' R' D2 R2 x'", "Ab": "x R2 D2 R U R' D2 R U' R x'",
    "E": "x' R U' R' D R U R' D' R U R' D R U' R' D' x",
    "F": "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "Ga": "R2 U R' U R' U' R U' R2 U' D R' U R D'", "Gb": "R' U' R U D' R2 U R' U R U' R U' R2 D",
    "Gc": "R2 U' R U' R U R' U R2 U D' R U' R' D", "Gd": "R U R' U' D R2 U' R U' R' U R' U R2 D'",
    "H": "M2 U M2 U2 M2 U M2", "Ja": "x R2 F R F' R U2 r' U r U2 x'",
    "Jb": "R U R' F' R U R' U' R' F R2 U' R'",
    "Na": "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "Nb": "R' U R U' R' F' U' F R U R' F R' F' R U' R",
    "Ra": "R U' R' U' R U R D R' U' R D' R' U2 R'", "Rb": "R2 F R U R U' R' F' R U2 R' U2 R",
    "T": "R U R' U' R' F R2 U' R' U' R U R' F'", "Ua": "M2 U M U2 M' U M2", "Ub": "M2 U' M U2 M' U' M2",
    "V": "R' U R' U' y R' F' R2 U' R' U R' F R F", "Y": "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "Z": "M' U M2 U M2 U M' U2 M2",
}
OLL_2LOOK_ALGS = {
    "OLL edges: line": "F R U R' U' F'", "OLL edges: L-shape": "f R U R' U' f'",
    "OLL edges: dot": "F R U R' U' F' f R U R' U' f'",
    "OLL Sune": "R U R' U R U2 R'", "OLL Antisune": "R U2 R' U' R U' R'",
    "OLL H": "R U R' U R U' R' U R U2 R'", "OLL Pi": "R U2 R2 U' R2 U' R2 U2 R",
    "OLL Headlights": "R2 D R' U2 R D' R' U2 R'", "OLL T": "r U R' U' r' F R F'",
    "OLL Bowtie": "F' r U R' U' r' F R",
}


def invert_tokens(tokens):
    out = []
    for t in reversed(tokens):
        if t.endswith("2"):
            out.append(t)
        elif t.endswith("'"):
            out.append(t[:-1])
        else:
            out.append(t + "'")
    return out


def load_solves():
    try:
        if _os.path.exists(SOLVES_FILE):
            with open(SOLVES_FILE) as fh:
                return _json.load(fh)
    except Exception:
        pass
    return []


def average_of(times, n):
    """WCA-style average of the last n: drop the best and worst, mean the rest."""
    if len(times) < n:
        return None
    last = sorted(times[-n:])
    trim = max(1, int(round(n * 0.05))) if n >= 12 else 1
    return sum(last[trim:-trim]) / float(n - 2 * trim)


def best_average(times, n):
    vals = [average_of(times[:k], n) for k in range(n, len(times) + 1)]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def solve_stats(records, mode="SOLVE", case=None, player=None):
    """PB, averages and counts for one kind of solve (and one player)."""
    times = [r["time_s"] for r in records
             if r.get("mode") == mode and (case is None or r.get("case") == case) and r.get("time_s")
             and not r.get("dnf") and (player is None or r.get("player", "Me") == player)]
    if not times:
        return None
    return {"best": min(times), "ao5": average_of(times, 5), "ao12": average_of(times, 12),
            "best_ao5": best_average(times, 5), "mean": sum(times) / len(times),
            "count": len(times), "last": times[-1]}


def save_solves(records):
    try:
        with open(SOLVES_FILE, "w") as fh:
            _json.dump(records, fh, indent=1)
    except Exception as exc:
        print(f"⚠️ [SOLVES]: could not save: {exc}")


class SolveTimer:
    """Armed after a scramble (or with the Timer button); starts on the first
    turn and stops when the cube is solved (or, for an OLL case, oriented).
    Records CFOP splits by watching the cube state."""

    ROTATIONS = ("x", "y", "z")

    def __init__(self, geometry, naming):
        self.geo = geometry
        self.naming = naming
        self.state = "IDLE"
        self.last_result = None
        self.message = ""
        self.splits = {}
        self.mode, self.case, self.case_alg = "SOLVE", None, None
        self.scramble_internal = []
        self.records = load_solves()

    def arm(self, history, now, scramble_internal=None, mode="SOLVE", case=None, case_alg=None):
        self.state = "READY"
        self.ready_len = len(history)
        self.scramble_internal = list(scramble_internal or [])
        self.mode = mode
        self.case = case
        self.case_alg = case_alg
        self.splits = {}
        self.message = "Ready - the timer starts on your first turn" if mode == "SOLVE" else "Case ready - solve it!"

    def cancel(self):
        self.state = "IDLE"
        self.message = "Timer cancelled"

    def elapsed(self, now):
        if self.state == "RUNNING":
            return now - self.start_time
        if self.state == "DONE" and self.last_result:
            return self.last_result["time_s"]
        return 0.0

    def stats(self):
        player = APP_SETTINGS.get("player", "Me")
        if self.state in ("READY", "RUNNING", "DONE") and getattr(self, "mode", "SOLVE") == "CASE":
            return solve_stats(self.records, "CASE", self.case, player)
        return solve_stats(self.records, "SOLVE", None, player)

    def start_now(self, history, now):
        """SPACE released: the clock starts immediately."""
        if self.state != "READY":
            self.mode, self.case, self.case_alg = "SOLVE", None, None
            self.scramble_internal = []
        self.state = "RUNNING"
        self.start_time = now
        self.start_len = len(history)
        self.splits = {}
        self.message = ""

    def stop_manual(self, cube_state, history, now):
        """SPACE pressed while running: stop now (marked if not solved)."""
        if self.state != "RUNNING":
            return None
        return self._finish(now - self.start_time, history, solved=is_solved_state(cube_state), manual=True)

    def update(self, cube_state, history, now):
        if self.state == "READY":
            new = [m for m in history[self.ready_len:] if m.rstrip("'2") not in self.ROTATIONS]
            if new:
                self.state = "RUNNING"
                self.start_time = now
                self.start_len = self.ready_len
                self.message = ""
            return None
        if self.state != "RUNNING":
            return None
        cross, f2l, oll, solved = self.geo.stages(cube_state)
        t = now - self.start_time
        if cross is not None and "cross" not in self.splits:
            self.splits["cross"] = t
        if f2l is not None and "F2L" not in self.splits:
            self.splits.setdefault("cross", t)
            self.splits["F2L"] = t
        if oll and "OLL" not in self.splits:
            self.splits.setdefault("cross", t)
            self.splits.setdefault("F2L", t)
            self.splits["OLL"] = t
        finished = solved or (self.mode == "CASE" and self.case and self.case.startswith("OLL") and oll)
        if not finished:
            return None
        if solved:
            self.splits.setdefault("cross", t)
            self.splits.setdefault("F2L", t)
            self.splits.setdefault("OLL", t)
            self.splits["PLL"] = t
        return self._finish(t, history, solved=True, manual=False)

    def _finish(self, t, history, solved, manual):
        internal = list(history[self.start_len:])
        moves_user = self.naming.to_user(internal)
        turns = [m for m in moves_user if m.rstrip("'2") not in self.ROTATIONS]
        record = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            "case": self.case,
            "time_s": round(t, 3),
            "moves": moves_user,
            "move_count": len(turns),
            "tps": round(len(turns) / t, 2) if t > 0 else 0.0,
            "splits": {k: round(v, 3) for k, v in self.splits.items()},
            "scramble": self.naming.to_user(self.scramble_internal),
            # alg.cubing.net uses standard notation from the solver's view
            "link": alg_cubing_link(self.naming.to_standard(self.scramble_internal),
                                    self.naming.to_standard(internal), self.case or "Solve"),
            "case_alg": self.case_alg,
            "camera": self.naming.frame_name,
            "player": APP_SETTINGS.get("player", "Me"),
            "stopped": "space" if manual else "solved",
            "tracked_solved": bool(solved),
        }
        self.state = "DONE"
        self.last_result = record
        self.records.append(record)
        append_solve_record(record)
        label = self.case or "SOLVE"
        print(f"⏱️ [{label}] {t:.2f}s | {len(turns)} moves | {record['tps']} TPS | splits {record['splits']}")
        print(f"   replay: {record['link']}")
        return record


class CaseTrainer:
    """Random PLL / 2-look OLL case. The guide shows the setup (which scrambles
    only the last layer); when it is done the timer runs until the case is
    solved. Algorithms are standard notation and are translated for the
    camera, so the moves shown always do the right thing physically."""

    def __init__(self, tracker, geometry, naming):
        self.tracker = tracker
        self.geo = geometry
        self.naming = naming
        self.valid = {}
        self.pending = None
        self._validate()

    def _validate(self):
        solved = self.tracker.state_from_history([])
        self.valid = {}
        for name, alg in list(PLL_ALGS.items()) + list(OLL_2LOOK_ALGS.items()):
            internal = self.naming.std_to_internal(alg.split())
            st = self.tracker.apply_move_sequence(solved.copy(), internal)
            ok = st is not None and any(self.geo.f2l_on(st, n) for n in self.geo.centre_index)
            if ok:
                self.valid[name if name.startswith("OLL") else f"PLL {name}"] = alg
            else:
                print(f"⚠️ [TRAINER]: skipping '{name}' - it does not keep the first two layers solved")
        print(f"🎯 [TRAINER]: {sum(1 for k in self.valid if k.startswith('PLL'))} PLL and "
              f"{sum(1 for k in self.valid if k.startswith('OLL'))} OLL cases ready")

    def display_alg(self, alg_std):
        """the algorithm in the names you see"""
        return self.naming.to_user(self.naming.std_to_internal(alg_std.split()))

    def next_case(self, kind, cube_state):
        """Returns (setup in the names you see, message)."""
        if not is_solved_state(cube_state):
            return None, "Solve the cube first (Solve button), then try again"
        names = [k for k in self.valid if k.startswith(kind)]
        if not names:
            return None, f"No {kind} cases available"
        name = _random.choice(names)
        alg = self.valid[name].split()
        auf = _random.choice([[], ["U"], ["U'"], ["U2"]])
        setup_internal = self.naming.std_to_internal(invert_tokens(alg) + auf)
        self.pending = (name, self.valid[name])
        return self.naming.to_user(setup_internal), f"{name}: set it up, then solve it"


def build_case_prior(tracker, naming, start_state, alg_std_tokens):
    """During a trainer case the algorithm is known. Tell the tracker to expect
    it (with any U turn to line the case up before/after), exactly like the
    guide does for a scramble - essential for turns the camera cannot see."""
    aufs = [[], ["U"], ["U'"], ["U2"]]
    expected = {}
    for pre in aufs:
        for post in aufs:
            quarters = []
            for tok in naming.std_to_internal(pre + list(alg_std_tokens) + post):
                quarters += [tok[:-1]] * 2 if tok.endswith("2") else [tok]
            states = [start_state.copy()]
            for q in quarters:
                states.append(tracker.apply_move_token(states[-1], q))
            hidden = [bool(np.array_equal(states[k][18:27], states[k + 1][18:27])) for k in range(len(quarters))]
            for k, mv in enumerate(quarters):
                key = np.array([DECODER_CIDX[str(c)] for c in states[k]], dtype=np.uint8).tobytes()
                mode = "visible" if not hidden[k] else ("hidden_final" if all(hidden[k:]) else "hidden")
                if (mv, mode) not in expected.get(key, []):
                    expected.setdefault(key, []).append((mv, mode))
    # The same hidden turn can be the last move (no U after it) or be followed
    # by a U: presume it done either way - a following U confirms it.
    for key, entries in expected.items():
        finals = {mv for mv, mode in entries if mode == "hidden_final"}
        if finals:
            expected[key] = [(mv, mode) for mv, mode in entries if not (mode == "hidden" and mv in finals)]
    return expected


class SpaceHold:
    """Speedcube-timer SPACE: hold, wait for green, release to start.
    OpenCV only reports key presses, so holding is recognised from the key
    repeating; a quick tap (or no key repeat) arms the timer instead, and it
    then starts on your first turn."""

    def __init__(self):
        self.first = None

    def press(self, now):
        if self.first is None:
            self.first, self.last, self.count = now, now, 1
        else:
            self.last, self.count = now, self.count + 1

    def poll(self, now):
        """None / 'holding' / 'ready' / 'released' / 'tap' / 'cancel'"""
        if self.first is None:
            return None
        hold_s = float(APP_SETTINGS.get("timer_hold_seconds", 0.5))
        held = self.last - self.first
        gap = now - self.last
        if self.count >= 2:
            if gap > 0.15:
                self.first = None
                return "released" if held >= hold_s else "cancel"
            return "ready" if now - self.first >= hold_s else "holding"
        if gap > 0.7:
            self.first = None
            return "tap"
        return "holding"


class Leaderboard:
    """Solve list you can sort through, rename, delete; PB and averages."""

    def __init__(self, timer):
        self.timer = timer
        self.view = "SOLVE"
        self.player_filter = None

    def players(self):
        return sorted({r.get("player", "Me") for r in self.timer.records})

    def fill_panel(self, panel):
        recs = [(i, r) for i, r in enumerate(self.timer.records)
                if r.get("mode") == self.view and (self.player_filter is None or r.get("player", "Me") == self.player_filter)]
        recs.sort(key=lambda ir: ir[1].get("time_s") or 1e9)
        panel.title = "Leaderboard - " + ("solves" if self.view == "SOLVE" else "PLL / OLL cases")
        if self.view == "SOLVE":
            panel.columns = [("#", 40), ("Time", 90), ("Player", 150), ("Moves", 70), ("TPS", 60), ("Date", 170)]
            panel.rows = [[str(k + 1), f"{r['time_s']:.2f}" + ("*" if not r.get("tracked_solved", True) else ""),
                           r.get("player", "Me"), str(r.get("move_count", "")), str(r.get("tps", "")), r.get("date", "")]
                          for k, (i, r) in enumerate(recs)]
        else:
            panel.columns = [("#", 40), ("Time", 90), ("Case", 170), ("Player", 150), ("Date", 170)]
            panel.rows = [[str(k + 1), f"{r['time_s']:.2f}", r.get("case", ""), r.get("player", "Me"), r.get("date", "")]
                          for k, (i, r) in enumerate(recs)]
        panel.row_ids = [i for i, _ in recs]
        me = APP_SETTINGS.get("player", "Me")
        panel.row_colours = [UI_GOOD if r.get("player", "Me") == me else UI_TEXT for _, r in recs]
        st = solve_stats(self.timer.records, self.view, None, self.player_filter)
        fmt = lambda v: f"{v:.2f}" if v else "-"
        who = self.player_filter or "everyone"
        panel.info = [(f"Showing {who}.  PB {fmt(st and st['best'])}   ao5 {fmt(st and st['ao5'])}   "
                       f"best ao5 {fmt(st and st['best_ao5'])}   ao12 {fmt(st and st['ao12'])}   "
                       f"mean {fmt(st and st['mean'])}   solves {st['count'] if st else 0}", UI_TEXT),
                      (f"New solves are saved as: {me}      (* = timer stopped with SPACE before the cube looked solved)", UI_MUTED)]
        panel.buttons = [("rename", "Rename"), ("delete", "Delete"), ("player", f"You: {me}"),
                         ("filter", f"Show: {self.player_filter or 'all'}"),
                         ("view", "Cases" if self.view == "SOLVE" else "Solves"),
                         ("link", "Open replay"), ("close", "Close")]

    def handle(self, panel, action, value, say):
        recs = self.timer.records
        sel = recs[panel.selected] if isinstance(panel.selected, int) and 0 <= panel.selected < len(recs) else None
        if action == "rename" and sel:
            panel.start_edit("Player name for this solve", sel.get("player", "Me"), "rename_done")
        elif action == "rename_done" and sel and value:
            sel["player"] = value
            save_solves(recs)
        elif action == "delete" and sel:
            recs.remove(sel)
            panel.selected = None
            save_solves(recs)
            say("Solve deleted", UI_WARN)
        elif action == "player":
            panel.start_edit("Your name (used for new solves)", APP_SETTINGS.get("player", "Me"), "player_done")
        elif action == "player_done" and value:
            APP_SETTINGS["player"] = value
            save_app_settings()
        elif action == "filter":
            options = [None] + self.players()
            k = options.index(self.player_filter) if self.player_filter in options else 0
            self.player_filter = options[(k + 1) % len(options)]
        elif action == "view":
            self.view = "CASE" if self.view == "SOLVE" else "SOLVE"
            panel.selected = None
        elif action == "link" and sel and sel.get("link"):
            try:
                import webbrowser
                webbrowser.open(sel["link"])
            except Exception:
                pass
        elif action == "close":
            return "close"
        self.fill_panel(panel)
        return None


COLOUR_SETS_FILE = "colour_sets.json"


def _time_of_day(hour=None):
    h = time.localtime().tm_hour if hour is None else hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "night"


def _light_word(white_L):
    if white_L >= 205:
        return "bright"
    if white_L >= 160:
        return "normal light"
    return "dim"


class ColourSets:
    """Saved colour calibrations with labels (cube, place, time of day), so a
    change of cube or room is one click - or automatic."""

    def __init__(self):
        self.sets = []
        self.active = None
        self.auto_pick = True
        self.load()

    # ------------------------------------------------------------ storage
    def load(self):
        try:
            if _os.path.exists(COLOUR_SETS_FILE):
                with open(COLOUR_SETS_FILE) as fh:
                    data = _json.load(fh)
                self.sets = data.get("sets", [])
                self.active = data.get("active")
                self.auto_pick = bool(data.get("auto_pick", True))
        except Exception as exc:
            print(f"⚠️ [COLOUR SETS]: could not load ({exc})")

    def save(self):
        try:
            with open(COLOUR_SETS_FILE, "w") as fh:
                _json.dump({"sets": self.sets, "active": self.active, "auto_pick": self.auto_pick}, fh, indent=1)
        except Exception as exc:
            print(f"⚠️ [COLOUR SETS]: could not save ({exc})")

    def get(self, name):
        return next((s for s in self.sets if s["name"] == name), None)

    # -------------------------------------------------------- calibrations
    def add_from(self, cal):
        """A calibration just finished: store it with an automatic label."""
        white_L = float(cal.cluster_centers["White"][0])
        cube = APP_SETTINGS.get("cube_name", "My cube")
        place = APP_SETTINGS.get("place", "Home")
        when = _time_of_day()
        base = f"{cube} - {place} - {when}"
        name, k = base, 2
        while self.get(name) is not None:
            name, k = f"{base} ({k})", k + 1
        entry = {
            "name": name, "cube": cube, "place": place, "when": when,
            "light": _light_word(white_L), "date": time.strftime("%Y-%m-%d %H:%M"),
            "centres": {c: [float(x) for x in v] for c, v in cal.cluster_centers.items()},
            "sigma": float(getattr(cal, "sigma", 12.0)), "glare_L": float(getattr(cal, "glare_L", 240.0)),
        }
        self.sets.append(entry)
        self.active = name
        self.save()
        return name

    def apply(self, name, cal):
        s = self.get(name)
        if s is None:
            return False
        cal.cluster_centers = {c: np.array(v, dtype=float) for c, v in s["centres"].items()}
        cal.sigma = float(s.get("sigma", 12.0))
        cal.glare_L = float(s.get("glare_L", 240.0))
        cal.is_calibrated = True
        cal.current_scan_idx = len(SCAN_ORDER)
        self.active = name
        self.save()
        save_colour_calibration(cal)
        return True

    def import_current(self, cal):
        """First run with colour sets: keep the calibration you already had."""
        if self.sets or not cal.is_calibrated or set(SCAN_ORDER) - set(cal.cluster_centers):
            return
        name = self.add_from(cal)
        self.get(name)["name"] = self.active = "Saved before colour sets"
        self.save()

    # ------------------------------------------------------------- panel
    def fill_panel(self, panel):
        panel.title = "Colour sets"
        panel.columns = [("Name", 330), ("Cube", 130), ("Place", 110), ("Time", 100), ("Light", 110)]
        panel.rows = [[s["name"], s.get("cube", ""), s.get("place", ""), s.get("when", ""), s.get("light", "")]
                      for s in self.sets]
        panel.row_ids = [s["name"] for s in self.sets]
        panel.row_colours = [UI_GOOD if s["name"] == self.active else UI_TEXT for s in self.sets]
        if panel.selected not in panel.row_ids:
            panel.selected = self.active
        panel.info = [
            (f"In use: {self.active or '-'}", UI_GOOD),
            (f"New calibrations are labelled: {APP_SETTINGS.get('cube_name', 'My cube')} - "
             f"{APP_SETTINGS.get('place', 'Home')} - time of day", UI_MUTED),
        ]
        panel.buttons = [("use", "Use"), ("rename", "Rename"), ("delete", "Delete"),
                         ("cube", "Cube name"), ("place", "Place"),
                         ("auto", f"Auto-pick: {'on' if self.auto_pick else 'off'}"),
                         ("new", "New calibration"), ("close", "Close")]

    def handle(self, panel, action, value, cal, say):
        """returns a key to run (e.g. 'v' for a new calibration) or None"""
        sel = self.get(panel.selected) if panel.selected else None
        if action == "use" and sel:
            self.apply(sel["name"], cal)
            say(f"Colours: {sel['name']}", UI_GOOD)
        elif action == "rename" and sel:
            panel.start_edit("New name for this colour set", sel["name"], "rename_done")
        elif action == "rename_done" and sel and value:
            if self.active == sel["name"]:
                self.active = value
            sel["name"] = value
            panel.selected = value
            self.save()
        elif action == "delete" and sel:
            self.sets.remove(sel)
            if self.active == sel["name"]:
                self.active = None
            self.save()
        elif action == "cube":
            panel.start_edit("Which cube is this? (used to label new calibrations)",
                             APP_SETTINGS.get("cube_name", "My cube"), "cube_done")
        elif action == "cube_done" and value:
            APP_SETTINGS["cube_name"] = value
            save_app_settings()
        elif action == "place":
            panel.start_edit("Where are you? (used to label new calibrations)",
                             APP_SETTINGS.get("place", "Home"), "place_done")
        elif action == "place_done" and value:
            APP_SETTINGS["place"] = value
            save_app_settings()
        elif action == "auto":
            self.auto_pick = not self.auto_pick
            self.save()
        elif action == "new":
            return "v"
        elif action == "close":
            return "close"
        self.fill_panel(panel)
        return None


class LightingMonitor:
    """Watches how well the stickers the camera sees fit the calibrated
    colours. A big, lasting drop = the light changed: suggest recalibrating,
    or switch to a saved colour set that fits better."""

    BAD_FIT = 3.0          # typical distance (in calibration spreads) that means trouble
    BAD_SECONDS = 3.0
    PROMPT_EVERY = 60.0

    BAD_MISREAD = 0.2     # share of stickers read as the wrong colour

    def __init__(self):
        self.ema = None
        self.misread = 0.0
        self.bad_since = None
        self.last_prompt = -1e9
        self.samples = deque(maxlen=60)
        self.last_pick = 0.0

    @staticmethod
    def fit(centres, sigma, labs):
        if not labs:
            return None
        d = []
        for lab in labs:
            d.append(min(GlobalKMeansCalibrator._weighted_dist(lab, np.asarray(c)) for c in centres.values()) / sigma)
        return float(np.median(d))

    def update(self, cal, labs, glare, now, sets, labels=None, truth=None):
        """returns ('prompt', None) / ('switch', name) / None.
        labels/truth: the colours read and the colours the tracker knows are
        there (when it is sure) - a lasting mismatch means the colours are off."""
        if not cal.is_calibrated or labs is None or glare is None:
            return None
        clean = [np.asarray(l, dtype=float) for l, g in zip(labs, glare) if g < 0.15]
        if len(clean) < 5:
            return None
        if labels is not None and truth is not None:
            pairs = [(a, b) for a, b, g in zip(labels, truth, glare) if g < 0.15]
            if pairs:
                wrong = sum(1 for a, b in pairs if str(a) != str(b)) / float(len(pairs))
                self.misread = 0.93 * self.misread + 0.07 * wrong
        self.samples.extend(clean)
        f = self.fit(cal.cluster_centers, float(getattr(cal, "sigma", 12.0)), clean)
        self.ema = f if self.ema is None else 0.9 * self.ema + 0.1 * f
        if sets.auto_pick and len(sets.sets) > 1 and now - self.last_pick > 1.0 and len(self.samples) >= 30:
            self.last_pick = now
            scores = [(self.fit(s["centres"], float(s.get("sigma", 12.0)), list(self.samples)), s["name"]) for s in sets.sets]
            best_fit, best = min(scores)
            cur = next((sc for sc, n in scores if n == sets.active), None)
            if best != sets.active and best_fit < 2.5 and (cur is None or best_fit < 0.75 * cur):
                self.ema = None
                self.bad_since = None
                return ("switch", best)
        if self.ema > self.BAD_FIT or self.misread > self.BAD_MISREAD:
            if self.bad_since is None:
                self.bad_since = now
            elif now - self.bad_since > self.BAD_SECONDS and now - self.last_prompt > self.PROMPT_EVERY:
                self.last_prompt = now
                return ("prompt", None)
        else:
            self.bad_since = None
        return None


class JpegFrames:
    """A recording kept as JPEG bytes (about 20x smaller than raw frames);
    frames are decoded only when looked at."""

    def __init__(self, blobs):
        self.blobs = list(blobs)

    def __len__(self):
        return len(self.blobs)

    def __bool__(self):
        return bool(self.blobs)

    def __getitem__(self, i):
        buf = self.blobs[i]
        return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


class SolveRecorder(SlowMoBufferProcessor):
    """Records the camera for a solve. Every camera frame goes into the video
    (for replay / saving); the frames the tracker actually used are what the
    video check ([e]) re-reads, so it sees exactly what live tracking saw."""

    def __init__(self, tracker):
        super().__init__(tracker, fps=60.0)
        self.all_blobs, self.all_times, self.tracked_idx = [], [], []
        self.last_all_frames = JpegFrames([])
        self.last_all_times = []
        self.auto = False
        self.save_status = ""
        self._q = None
        self._encoder = None

    # ------------------------------------------------------------- recording
    def start_recording(self, initial_state=None, initial_history=None, auto=False):
        super().start_recording(initial_state, initial_history)
        self.all_blobs, self.all_times, self.tracked_idx = [], [], []
        self.auto = bool(auto)
        self.started_at = time.perf_counter()
        self._q = queue.Queue()
        self._encoder = threading.Thread(target=self._encode_loop, args=(self._q,), daemon=True)
        self._encoder.start()

    def _encode_loop(self, q):
        quality = int(APP_SETTINGS.get("record_jpeg_quality", 92))
        while True:
            item = q.get()
            if item is None:
                return
            idx, frame = item
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                self.all_blobs[idx] = buf.tobytes()

    def add_frame(self, frame, timestamp=None):
        """Store one camera frame; returns its index in the recording."""
        if not self.is_recording:
            return None
        if time.perf_counter() - self.started_at > float(APP_SETTINGS.get("max_record_seconds", 180)):
            return None
        idx = len(self.all_times)
        self.all_times.append(float(timestamp) if timestamp is not None else time.perf_counter())
        self.all_blobs.append(None)
        self._q.put((idx, frame))
        return idx

    def mark_tracked(self, idx):
        if idx is not None and self.is_recording:
            self.tracked_idx.append(idx)

    def stop(self, live_move_events=None):
        """Finish the recording; returns True if there is something to keep."""
        with self.lock:
            if not self.is_recording:
                return False
            self.is_recording = False
            observations = list(self.recorded_observations)
            self.recorded_observations = []
        if self._q is not None:
            self._q.put(None)
            self._encoder.join(timeout=30)
        tracked = sorted({i for i in self.tracked_idx if 0 <= i < len(self.all_blobs) and self.all_blobs[i] is not None})
        blobs = [b for b in self.all_blobs if b is not None]
        times = [t for b, t in zip(self.all_blobs, self.all_times) if b is not None]
        self.last_all_frames = JpegFrames(blobs)
        self.last_all_times = times
        self.last_completed_frames = JpegFrames([self.all_blobs[i] for i in tracked])
        self.last_completed_frame_times = [self.all_times[i] for i in tracked]
        self.last_completed_observations = observations
        span = (self.last_completed_frame_times[-1] - self.last_completed_frame_times[0]) if len(tracked) > 1 else 0
        self.last_completed_source_fps = (len(tracked) - 1) / span if span > 0 else 12.0
        base_state = self.base_state_snapshot.copy() if self.base_state_snapshot is not None else self.cube_tracker.state.copy()
        base_history = list(self.base_history_snapshot)
        self.live_history_at_stop = list(self.cube_tracker.move_history)
        self.live_move_events_at_stop = list(live_move_events or [])
        self.all_blobs, self.all_times, self.tracked_idx = [], [], []
        print(f"⏹️ [RECORDING]: {len(blobs)} frames of video, {len(tracked)} tracked frames "
              f"({self.last_completed_source_fps:.1f} per second).")
        self._dump_live_session(observations, base_state, base_history, self.live_history_at_stop)
        self.analysis_status = {"active": False, "status": "VIDEO READY", "progress": 0.0}
        return bool(blobs)

    def discard(self):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
        if self._q is not None:
            self._q.put(None)
        self.all_blobs, self.all_times, self.tracked_idx = [], [], []

    def video_fps(self):
        t = self.last_all_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 30.0

    # ----------------------------------------------------------------- saving
    def save_video(self, mirror=True):
        """Write the last recording to an .mp4 in the background."""
        frames = self.last_all_frames
        if not frames:
            self.save_status = "Nothing recorded yet"
            return None
        folder = APP_SETTINGS.get("videos_folder", "recordings")
        _os.makedirs(folder, exist_ok=True)
        path = _os.path.join(folder, time.strftime("solve_%Y%m%d_%H%M%S.mp4"))
        fps = max(1.0, min(60.0, self.video_fps()))

        def work():
            try:
                first = frames[0]
                h, w = first.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for i in range(len(frames)):
                    f = frames[i]
                    writer.write(cv2.flip(f, 1) if mirror else f)
                    self.save_status = f"Saving video {100 * (i + 1) // len(frames)}%"
                writer.release()
                self.save_status = f"Video saved: {_os.path.abspath(path)}"
                print(f"💾 [VIDEO]: {self.save_status}")
            except Exception as exc:
                self.save_status = f"Could not save the video: {exc}"
                print(f"⚠️ [VIDEO]: {self.save_status}")

        self.save_status = "Saving video..."
        threading.Thread(target=work, daemon=True).start()
        return path


# =============================================================================
# USER INTERFACE: one window, clickable buttons, move arrows on the cube,
# 3D/2D cube view, pop-up panels (leaderboard, colour sets).
# Look and sizes come from UI_CONFIG at the top of the file.
# =============================================================================
WINDOW_MAIN = "Rubik's Cube Tracker"
WINDOW_REPLAY = "Replay"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
_AA = cv2.LINE_AA


def C(name):
    """a colour from UI_CONFIG"""
    return tuple(UI_CONFIG["colours"][name])


def TS(name):
    """a text size from UI_CONFIG"""
    return float(UI_CONFIG["text_size"][name])


# short names used across the code (kept in sync with UI_CONFIG)
UI_BG, UI_PANEL, UI_PANEL_2, UI_LINE = C("background"), C("panel"), C("panel_light"), C("line")
UI_TEXT, UI_MUTED, UI_DIM = C("text"), C("muted"), C("dim")
UI_ACCENT, UI_GOOD, UI_WARN, UI_BAD = C("accent"), C("good"), C("warn"), C("bad")
UI_ARROW, UI_ARROW_FIX = C("arrow"), C("arrow_undo")
STICKER_BGR = {k: tuple(v) for k, v in UI_CONFIG["sticker_colours"].items()}


def ui_text(img, s, x, y, scale=0.5, color=UI_TEXT, thick=1, font=_FONT):
    cv2.putText(img, str(s), (int(x), int(y)), font, scale, color, thick, _AA)


def ui_text_w(s, scale=0.5, thick=1, font=_FONT):
    return cv2.getTextSize(str(s), font, scale, thick)[0][0]


def ui_fit(s, scale, width, thick=1, font=_FONT):
    """shorten a text with '..' so it fits a width"""
    s = str(s)
    if ui_text_w(s, scale, thick, font) <= width:
        return s
    while s and ui_text_w(s + "..", scale, thick, font) > width:
        s = s[:-1]
    return s + ".."


def ui_rrect(img, x1, y1, x2, y2, color, r=6, thick=-1):
    x1, y1, x2, y2, r = int(x1), int(y1), int(x2), int(y2), int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thick < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, _AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, _AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, _AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, _AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, _AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thick, _AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thick, _AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thick, _AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thick, _AA)


def ui_blend_box(img, x1, y1, x2, y2, color, alpha=0.72):
    """Translucent box without copying the whole frame."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0, dst=roi)


def _in(rect, x, y):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


class MoveArrows:
    """Where a move sends the stickers the camera sees, as arrows on the cube.
    Worked out from the tracker's own permutations and 3D model, so the arrow
    always matches what the tracker expects to see."""

    def __init__(self, tracker, geometry):
        self.tracker = tracker
        self.geo = geometry
        face = list(range(18, 27))
        X = np.array([[geometry.pos[i][0], geometry.pos[i][1], 1.0] for i in face])
        UV = np.array([[(i - 18) % 3 + 0.5, (i - 18) // 3 + 0.5] for i in face])
        sol, *_ = np.linalg.lstsq(X, UV, rcond=None)
        self.A = sol.T            # 2x3: model (x, y, 1) -> grid (u, v)
        self.cache = {}

    def _uv(self, i):
        p = self.geo.pos[i]
        return self.A @ np.array([p[0], p[1], 1.0])

    def plan(self, move):
        """{'kind': 'arc'|'lines', ...} in grid units (0..3), or None."""
        if move in self.cache:
            return self.cache[move]
        self.cache[move] = out = self._plan(move)
        return out

    def _plan(self, move):
        P = self.tracker.permutations.get(move)
        if P is None:
            return None
        dest = np.argsort(np.asarray(P))          # sticker at a ends up at dest[a]
        pos = [np.array(self.geo.pos[i], dtype=float) + 0.5 * np.array(self.geo.nrm[i], dtype=float)
               for i in range(54)]
        moving = [a for a in range(54) if dest[a] != a]
        if not moving:
            return None
        axis = next((k for k in range(3)
                     if all(abs(pos[a][k] - pos[int(dest[a])][k]) < 1e-6 for a in moving)), None)
        if axis is None:
            return None
        sense = float(np.sign(sum(np.cross(pos[a], pos[int(dest[a])])[axis] for a in moving)))
        if sense == 0:
            return None
        L = self.A[:, :2]                          # model xy -> grid uv (linear part)
        moving_face = [i for i in range(18, 27) if dest[i] != i]
        if axis == 2:
            # turning parallel to the camera face: a curved arrow
            return {"kind": "arc", "sense": sense * float(np.sign(np.linalg.det(L))),
                    "hidden": not moving_face}
        e = np.zeros(3)
        e[axis] = 1.0
        lines = {}
        for i in moving_face:
            vel = sense * np.cross(e, pos[i])      # which way this sticker starts moving
            d = L @ vel[:2]
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            d = np.round(d / n).astype(int)
            u, v = (i - 18) % 3, (i - 18) // 3
            key = (int(d[0]), int(d[1]), u if d[0] == 0 else v)
            lines.setdefault(key, []).append((u + 0.5, v + 0.5))
        segs = []
        for (dx, dy, _), pts in lines.items():
            d = np.array([dx, dy], dtype=float)
            proj = [np.dot(p, d) for p in pts]
            segs.append((np.array(pts[int(np.argmin(proj))]) - 0.42 * d,
                         np.array(pts[int(np.argmax(proj))]) + 0.42 * d))
        return {"kind": "lines", "segs": segs} if segs else None


def naming_basis(tracker, geo, naming):
    """3x3 matrix whose rows point to the faces you call R, U and F, so the
    3D view is drawn the way you name the faces."""
    rows = []
    for letter in ("R", "U", "F"):
        tok = naming.to_internal([letter])[0].rstrip("'")
        P = tracker.permutations.get(tok)
        vec = np.zeros(3)
        if P is not None:
            dest = np.argsort(np.asarray(P))
            moving = [a for a in range(54) if dest[a] != a]
            for k in range(3):
                vals = {geo.pos[a][k] for a in moving}
                if len(vals) == 1:
                    vec[k] = float(np.sign(list(vals)[0]))
        rows.append(vec)
    return np.array(rows)


class Cube3DView:
    """The tracked cube drawn in 3D (drag to turn it, double-click to reset)."""
    DEFAULT = (-32.0, 24.0)

    def __init__(self, geometry):
        self.geo = geometry
        self.yaw, self.pitch = self.DEFAULT
        self.pos = np.array([np.array(p, dtype=float) + 0.5 * np.array(n, dtype=float)
                             for p, n in zip(geometry.pos, geometry.nrm)])
        self.nrm = np.array([np.array(n, dtype=float) for n in geometry.nrm])

    def rotation(self):
        a, b = np.radians(self.yaw), np.radians(self.pitch)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        return Rx @ Ry

    def draw(self, img, rect, state, basis):
        x1, y1, x2, y2 = rect
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        scale = min(x2 - x1, y2 - y1) / 5.6
        M = self.rotation() @ basis
        P = self.pos @ M.T
        N = self.nrm @ M.T
        faces = {}
        for i in range(54):
            if N[i][2] > 0.05:
                faces.setdefault(int(np.argmax(np.abs(self.nrm[i]))) * 2 + int(self.nrm[i].max() > 0), []).append(i)
        order = sorted(faces.values(), key=lambda idxs: float(np.mean([P[i][2] for i in idxs])))

        def to_px(v):
            return (int(cx + scale * v[0]), int(cy - scale * v[1]))

        for idxs in order:
            n = N[idxs[0]]
            k = int(np.argmax(np.abs(self.nrm[idxs[0]])))
            t1, t2 = [M[:, j] for j in range(3) if j != k]    # the face's own grid directions
            centre = np.mean([P[i] for i in idxs], axis=0)
            body = np.array([to_px(centre + 1.55 * (a * t1 + b * t2)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.int32)
            cv2.fillPoly(img, [body], (12, 12, 12), _AA)
            for i in idxs:
                c = P[i]
                quad = np.array([to_px(c + 0.43 * (a * t1 + b * t2)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.int32)
                col = STICKER_BGR.get(str(state[i]), (70, 70, 70))
                shade = 0.72 + 0.28 * float(max(0.0, n[2]))
                cv2.fillPoly(img, [quad], tuple(int(v * shade) for v in col), _AA)


class UIButton:
    def __init__(self, key, label):
        self.key = key
        self.label = label
        self.hint = "SPACE" if key == " " else key.upper()
        self.rect = (0, 0, 0, 0)
        self.visible = True
        self.enabled = True
        self.active = False


class ListPanel:
    """A pop-up table over the camera picture (leaderboard, colour sets):
    click a row to select it, click a button to act, mouse wheel to scroll."""

    def __init__(self, name, title):
        self.name = name
        self.title = title
        self.columns = []           # [(header, width)]
        self.rows = []              # [[cell, ...]]
        self.row_ids = []
        self.row_colours = []
        self.info = []              # lines of text under the title
        self.buttons = []           # [(action, label)]
        self.selected = None
        self.scroll = 0
        self.edit = None            # {"prompt":..., "text":..., "action":...}
        self.rects = {}             # hit areas, set while drawing
        self.actions = deque()

    def start_edit(self, prompt, text, action):
        self.edit = {"prompt": prompt, "text": str(text or ""), "action": action}

    def key(self, key):
        """keyboard while a text box is open; returns True if used"""
        if self.edit is None:
            return False
        if key in (13, 10):
            self.actions.append((self.edit["action"], self.edit["text"].strip()))
            self.edit = None
        elif key == 27:
            self.edit = None
        elif key in (8, 127):
            self.edit["text"] = self.edit["text"][:-1]
        elif 32 <= key < 127 and len(self.edit["text"]) < 28:
            self.edit["text"] += chr(key)
        return True

    def mouse(self, event, x, y, flags):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            self.scroll = max(0, self.scroll + (-1 if delta > 0 else 1))
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        for action, rect in self.rects.items():
            if _in(rect, x, y):
                if action.startswith("row:"):
                    self.selected = self.row_ids[int(action[4:])]
                else:
                    self.actions.append((action, None))
                return

    def draw(self, img, area):
        x1, y1, x2, y2 = area
        ui_blend_box(img, x1, y1, x2, y2, (16, 16, 16), 0.9)
        ui_rrect(img, x1, y1, x2, y2, UI_LINE, 10, 1)
        self.rects = {}
        ui_text(img, self.title, x1 + 22, y1 + 36, TS("title"), UI_TEXT, 2, _FONT_BOLD)
        y = y1 + 62
        for line, col in self.info:
            ui_text(img, line, x1 + 22, y, TS("body"), col, 1)
            y += 24
        # table
        y += 8
        x = x1 + 22
        for header, w in self.columns:
            ui_text(img, header, x, y, TS("small"), UI_MUTED, 1)
            x += w
        y += 10
        cv2.line(img, (x1 + 16, y), (x2 - 16, y), UI_LINE, 1)
        row_h = 28
        bottom = y2 - 64 - (48 if self.edit else 0)
        visible = max(1, (bottom - y - 6) // row_h)
        self.scroll = min(self.scroll, max(0, len(self.rows) - visible))
        for k in range(self.scroll, min(len(self.rows), self.scroll + visible)):
            ry = y + 6 + (k - self.scroll) * row_h
            sel = self.row_ids[k] == self.selected
            if sel:
                ui_rrect(img, x1 + 14, ry, x2 - 14, ry + row_h - 3, UI_ACCENT, 6)
            x = x1 + 22
            colour = (255, 255, 255) if sel else (self.row_colours[k] if k < len(self.row_colours) else UI_TEXT)
            for (header, w), cell in zip(self.columns, self.rows[k]):
                ui_text(img, ui_fit(cell, TS("body"), w - 10), x, ry + 19, TS("body"), colour, 1)
                x += w
            self.rects[f"row:{k}"] = (x1 + 14, ry, x2 - 14, ry + row_h - 3)
        if not self.rows:
            ui_text(img, "(nothing here yet)", x1 + 22, y + 30, TS("body"), UI_DIM, 1)
        if len(self.rows) > visible:
            ui_text(img, f"{self.scroll + 1}-{min(len(self.rows), self.scroll + visible)} of {len(self.rows)}  (scroll)",
                    x2 - 200, y1 + 36, TS("small"), UI_MUTED, 1)
        # text box
        if self.edit:
            ey = y2 - 108
            ui_text(img, self.edit["prompt"], x1 + 22, ey, TS("small"), UI_MUTED, 1)
            ui_rrect(img, x1 + 20, ey + 8, x2 - 20, ey + 40, UI_PANEL_2, 6)
            caret = "|" if int(time.perf_counter() * 2) % 2 == 0 else ""
            ui_text(img, self.edit["text"] + caret, x1 + 30, ey + 31, TS("body"), UI_TEXT, 1)
            ui_text(img, "Enter = save   Esc = cancel", x2 - 230, ey, TS("small"), UI_MUTED, 1)
        # buttons
        bx = x1 + 20
        by = y2 - 50
        for action, label in self.buttons:
            w = ui_text_w(label, TS("button")) + 28
            hov = False
            ui_rrect(img, bx, by, bx + w, by + 32, UI_PANEL_2 if not hov else UI_LINE, 7)
            ui_text(img, label, bx + 14, by + 21, TS("button"), UI_TEXT, 1)
            self.rects[action] = (bx, by, bx + w, by + 32)
            bx += w + 8


class TrackerUI:
    def __init__(self):
        L = UI_CONFIG["layout"]
        self.VIEW_W = int(L["camera_width"])
        self.SIDE_W = int(L["sidebar_width"])
        self.TOP_H = int(L["top_bar_height"])
        self.BOTTOM_H = int(L["guide_height"])
        self.canvas = None
        self.view_h = None
        self.hover = (-1, -1)
        self.clicks = deque()
        self.sections = [(title, [UIButton(k, lab) for k, lab in items]) for title, items in UI_CONFIG["buttons"]]
        self.capture_button = UIButton("c", "Capture colours")
        self.all_buttons = [b for _, bs in self.sections for b in bs] + [self.capture_button]
        self.panel = None            # the open ListPanel, if any
        self.cube3d = None           # Cube3DView, set by main()
        self.show_3d = True
        self.drag = None

    def button(self, key):
        for b in self.all_buttons:
            if b.key == key:
                return b
        return None

    # ----------------------------------------------------------------- input
    def on_mouse(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y)
            if self.drag and self.cube3d is not None:
                dx, dy = x - self.drag[0], y - self.drag[1]
                self.cube3d.yaw = self.drag[2] + dx * 0.8
                self.cube3d.pitch = float(np.clip(self.drag[3] + dy * 0.8, -89, 89))
            return
        if self.panel is not None and self.canvas is not None and _in(self.panel_area, x, y):
            self.panel.mouse(event, x, y, flags)
            return
        if self.canvas is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN and _in(self.cube_rect, x, y) and not _in(self.toggle_rect, x, y):
            if self.show_3d and self.cube3d is not None:
                self.drag = (x, y, self.cube3d.yaw, self.cube3d.pitch)
            return
        if event == cv2.EVENT_LBUTTONDBLCLK and _in(self.cube_rect, x, y) and self.cube3d is not None:
            self.cube3d.yaw, self.cube3d.pitch = Cube3DView.DEFAULT
            return
        if event == cv2.EVENT_LBUTTONUP:
            if self.drag is not None:
                moved = abs(x - self.drag[0]) + abs(y - self.drag[1]) > 4
                self.drag = None
                if moved:
                    return
            if _in(self.toggle_rect, x, y):
                self.clicks.append(ord("d"))
                return
            for b in self.all_buttons:
                if b.visible and b.enabled and _in(b.rect, x, y):
                    self.clicks.append(ord(b.key))
                    break

    def pop_key(self):
        return self.clicks.popleft() if self.clicks else None

    # ---------------------------------------------------------------- layout
    def _layout(self, frame_shape):
        L = UI_CONFIG["layout"]
        fh, fw = frame_shape[:2]
        self.scale = self.VIEW_W / float(fw)
        self.view_h = int(round(fh * self.scale))
        H = self.TOP_H + self.view_h + self.BOTTOM_H
        W = self.VIEW_W + self.SIDE_W
        self.canvas = np.empty((H, W, 3), dtype=np.uint8)
        self.side_x = self.VIEW_W
        pad, gap, bh = int(L["padding"]), int(L["button_gap"]), int(L["button_height"])
        cols = max(1, int(L["button_columns"]))
        x0 = self.side_x + pad
        col_w = (self.SIDE_W - 2 * pad - (cols - 1) * gap) // cols
        rows = sum((len(bs) + cols - 1) // cols for _, bs in self.sections)
        y = H - pad - rows * (bh + gap) - len(self.sections) * 18
        self.section_titles = []
        for title, bs in self.sections:
            self.section_titles.append((title, y + 12))
            y += 18
            for k, b in enumerate(bs):
                cx = x0 + (k % cols) * (col_w + gap)
                b.rect = (cx, y, cx + col_w, y + bh)
                if k % cols == cols - 1 or k == len(bs) - 1:
                    y += bh + gap
        self.buttons_top = self.section_titles[0][1] - 16
        # sidebar content, top to bottom
        self.cube_rect = (x0, 64, self.side_x + self.SIDE_W - pad, 64 + int(L["cube_view_height"]))
        self.toggle_rect = (self.cube_rect[2] - 44, self.cube_rect[1] + 4, self.cube_rect[2] - 4, self.cube_rect[1] + 24)
        self.timer_y = self.cube_rect[3] + 12
        self.moves_y = self.timer_y + 80
        cb = self.capture_button
        cb.rect = (x0, self.buttons_top - 40, self.side_x + self.SIDE_W - pad, self.buttons_top - 8)
        self.panel_area = (40, self.TOP_H + 20, self.VIEW_W - 40, self.TOP_H + self.view_h + self.BOTTOM_H - 20)

    # -------------------------------------------------------------- drawing
    def _button(self, img, b, hovered):
        if not b.visible:
            return
        x1, y1, x2, y2 = b.rect
        if not b.enabled:
            fill, tc = UI_PANEL, UI_DIM
        elif b.active:
            fill, tc = UI_ACCENT, (255, 255, 255)
        else:
            fill, tc = (UI_LINE if hovered else UI_PANEL_2), UI_TEXT
        ui_rrect(img, x1, y1, x2, y2, fill, 6)
        hw = ui_text_w(b.hint, 0.3)
        ui_text(img, ui_fit(b.label, TS("button"), x2 - x1 - hw - 14), x1 + 7, y1 + (y2 - y1) // 2 + 5, TS("button"), tc, 1)
        ui_text(img, b.hint, x2 - hw - 5, y1 + (y2 - y1) // 2 + 4, 0.3, UI_MUTED if not b.active else (240, 240, 240), 1)

    def _net(self, img, rect, state, basis):
        """Unfolded cube, laid out with the faces you call U, L, F, R, B, D."""
        x1, y1, x2, y2 = rect
        s = int(min((x2 - x1) / 12.5, (y2 - y1) / 9.5))
        x = x1 + ((x2 - x1) - 12 * s) // 2
        y = y1 + ((y2 - y1) - 9 * s) // 2
        geo = self.cube3d.geo
        # face (normal in your frame) -> (net column, net row, how its stickers unfold)
        layout = {
            (0, 0, 1): (3, 3, lambda p: (p[0], -p[1])),
            (1, 0, 0): (6, 3, lambda p: (-p[2], -p[1])),
            (0, 0, -1): (9, 3, lambda p: (-p[0], -p[1])),
            (-1, 0, 0): (0, 3, lambda p: (p[2], -p[1])),
            (0, 1, 0): (3, 0, lambda p: (p[0], p[2])),
            (0, -1, 0): (3, 6, lambda p: (p[0], -p[2])),
        }
        for i in range(54):
            n = tuple(int(round(v)) for v in basis @ np.array(geo.nrm[i], dtype=float))
            if n not in layout:
                continue
            fx, fy, unfold = layout[n]
            c, r = unfold(basis @ np.array(geo.pos[i], dtype=float))
            px, py = x + (fx + int(round(c)) + 1) * s, y + (fy + int(round(r)) + 1) * s
            col = STICKER_BGR.get(str(state[i]), (60, 60, 60))
            cv2.rectangle(img, (px, py), (px + s - 2, py + s - 2), col, -1)
        cx, cy = x + 3 * s, y + 3 * s
        cv2.rectangle(img, (cx - 2, cy - 2), (cx + 3 * s, cy + 3 * s), UI_ACCENT, 1)

    def _chips(self, img, tokens, x, y, w, max_lines, dim_from=None):
        """Move tokens as chips: the newest ones that fit, oldest first."""
        lh, gap = 22, 4
        widths = [ui_text_w(t, 0.47, 1) + 12 for t in tokens]

        def flow(first):
            lines, cur, cur_w = [], [], 0
            for i in range(first, len(tokens)):
                if cur and cur_w + widths[i] > w:
                    lines.append(cur)
                    cur, cur_w = [], 0
                cur.append(i)
                cur_w += widths[i] + gap
            if cur:
                lines.append(cur)
            return lines

        first = 0
        while first < len(tokens) and len(flow(first)) > max_lines:
            first += 1
        for li, idxs in enumerate(flow(first)):
            cx = x
            for i in idxs:
                unsure = dim_from is not None and i >= dim_from
                newest = i == len(tokens) - 1
                fill = UI_PANEL_2 if not newest else UI_ACCENT
                ui_rrect(img, cx, y + li * (lh + 4), cx + widths[i], y + li * (lh + 4) + lh, fill, 5)
                ui_text(img, tokens[i], cx + 6, y + li * (lh + 4) + 16, 0.47,
                        UI_MUTED if unsure else (255, 255, 255), 1)
                cx += widths[i] + gap

    def _grid_to_view(self, geom, pts):
        """grid units (u, v in 0..3) -> canvas pixels."""
        H_inv, frame_w, mirror = geom
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2) * 50.0
        raw = cv2.perspectiveTransform(p, H_inv).reshape(-1, 2)
        if mirror:
            raw[:, 0] = frame_w - raw[:, 0]
        return raw * self.scale + np.array([0.0, self.TOP_H])

    def _draw_move_arrow(self, img, geom, plan, colour, label=None, double=False):
        if plan is None or geom is None:
            return
        th = 5
        if plan["kind"] == "lines":
            for a, b in plan["segs"]:
                pa, pb = self._grid_to_view(geom, [a, b])
                pa, pb = tuple(int(v) for v in pa), tuple(int(v) for v in pb)
                cv2.arrowedLine(img, pa, pb, (0, 0, 0), th + 4, _AA, tipLength=0.28)
                cv2.arrowedLine(img, pa, pb, colour, th, _AA, tipLength=0.28)
            end = pb
        else:
            sense = plan["sense"]
            angles = np.radians(np.linspace(200.0, 200.0 + sense * 280.0, 40))
            radius = 1.05 if not plan.get("hidden") else 1.3
            pts = np.stack([1.5 + radius * np.cos(angles), 1.5 + radius * np.sin(angles)], axis=1)
            vp = self._grid_to_view(geom, pts).astype(np.int32)
            if plan.get("hidden"):
                for k in range(0, len(vp) - 2, 4):
                    cv2.line(img, tuple(vp[k]), tuple(vp[k + 2]), (0, 0, 0), th + 4, _AA)
                    cv2.line(img, tuple(vp[k]), tuple(vp[k + 2]), colour, th, _AA)
            else:
                cv2.polylines(img, [vp[:-3]], False, (0, 0, 0), th + 4, _AA)
                cv2.polylines(img, [vp[:-3]], False, colour, th, _AA)
            tip = vp[-1].astype(float)
            d = tip - vp[-6].astype(float)
            d /= max(1e-6, np.linalg.norm(d))
            nrm = np.array([-d[1], d[0]])
            size = max(14.0, 0.09 * float(np.linalg.norm(vp[0] - vp[len(vp) // 2])))
            tri = np.array([tip + d * size * 0.6, tip - d * size + nrm * size * 0.8,
                            tip - d * size - nrm * size * 0.8], dtype=np.int32)
            cv2.polylines(img, [tri], True, (0, 0, 0), 4, _AA)
            cv2.fillPoly(img, [tri], colour, _AA)
            end = tuple(int(v) for v in tip)
        if double:
            ui_rrect(img, end[0] + 6, end[1] - 14, end[0] + 40, end[1] + 10, (0, 0, 0), 6)
            ui_text(img, "x2", end[0] + 11, end[1] + 4, 0.55, colour, 2)
        if label:
            centre = self._grid_to_view(geom, [[1.5, 3.55]])[0]
            tw = ui_text_w(label, 0.55, 1)
            x1 = int(centre[0] - tw / 2 - 10)
            y1 = int(centre[1])
            ui_blend_box(img, x1, y1, x1 + tw + 20, y1 + 28, (0, 0, 0), 0.7)
            ui_text(img, label, x1 + 10, y1 + 20, 0.55, colour, 1)

    def render(self, view_frame, ctx):
        if self.canvas is None:
            self._layout(view_frame.shape)
        img = self.canvas
        H, W = img.shape[:2]
        np.copyto(img, self._static_layer(H, W))
        top, vw, vh = self.TOP_H, self.VIEW_W, self.view_h
        img[top:top + vh, 0:vw] = cv2.resize(view_frame, (vw, vh), interpolation=cv2.INTER_LINEAR)

        # ---------------------------------------------------- cube outline + arrow
        geom = ctx.get("geom")
        if geom is not None:
            corners = self._grid_to_view(geom, [[0, 0], [3, 0], [3, 3], [0, 3]]).astype(np.int32)
            cv2.polylines(img, [corners], True, UI_GOOD if ctx.get("view_ok") else UI_WARN, 2, _AA)
        arrow = ctx.get("arrow")
        if arrow is not None:
            self._draw_move_arrow(img, geom, arrow["plan"], arrow["colour"], arrow.get("label"), arrow.get("double"))

        # ---------------------------------------------------- top bar
        chip_text, chip_col = ctx.get("status", ("", UI_MUTED))
        cw = ui_text_w(chip_text, TS("body"), 1) + 30
        ui_rrect(img, 10, 8, 10 + cw, top - 8, UI_PANEL_2, 11)
        cv2.circle(img, (24, top // 2), 5, chip_col, -1, _AA)
        ui_text(img, chip_text, 34, top // 2 + 6, TS("body"), UI_TEXT, 1)
        x = 10 + cw + 14
        for t in ctx.get("top_info", []):
            ui_text(img, t, x, top // 2 + 6, 0.47, UI_MUTED, 1)
            x += ui_text_w(t, 0.47) + 18
        right = ctx.get("fps_text", "")
        ui_text(img, right, vw - ui_text_w(right, 0.47) - 12, top // 2 + 6, 0.47, UI_MUTED, 1)
        if ctx.get("recording"):
            rx = vw - ui_text_w(right, 0.47) - 90
            cv2.circle(img, (rx, top // 2), 7, UI_BAD, -1, _AA)
            ui_text(img, "REC", rx + 12, top // 2 + 6, TS("body"), UI_BAD, 1)

        badge = ctx.get("mode_badge")
        if badge:
            txt, col = badge
            bw_ = ui_text_w(txt, 0.5, 1) + 24
            ui_blend_box(img, vw - bw_ - 10, top + vh - 38, vw - 10, top + vh - 10, (15, 15, 15), 0.75)
            ui_text(img, txt, vw - bw_ + 2, top + vh - 18, 0.5, col, 1)

        # ---------------------------------------------------- messages over the camera
        y = top + 12
        for msg, col in ctx.get("toasts", []):
            if not msg:
                continue
            tw = ui_text_w(msg, TS("toast"), 1)
            x1 = (vw - tw) // 2 - 14
            ui_blend_box(img, x1, y, x1 + tw + 28, y + 30, (20, 20, 20), 0.78)
            ui_text(img, msg, x1 + 14, y + 21, TS("toast"), col, 1)
            y += 36
        card = ctx.get("card")
        if card:
            title, lines, col = card
            cw_ = max([ui_text_w(title, 0.7, 2)] + [ui_text_w(l, 0.52) for l in lines]) + 40
            ch_ = 46 + 26 * len(lines)
            x1, y1 = (vw - cw_) // 2, top + vh - ch_ - 16
            ui_blend_box(img, x1, y1, x1 + cw_, y1 + ch_, (18, 18, 18), 0.82)
            cv2.rectangle(img, (x1, y1), (x1 + 4, y1 + ch_), col, -1)
            ui_text(img, title, x1 + 20, y1 + 32, 0.7, col, 2, _FONT_BOLD)
            for k, l in enumerate(lines):
                ui_text(img, l, x1 + 20, y1 + 60 + 26 * k, 0.52, UI_TEXT, 1)
        big = ctx.get("big_timer")
        if big:
            txt, col = big
            tw = ui_text_w(txt, 2.6, 5, _FONT_BOLD)
            ui_blend_box(img, (vw - tw) // 2 - 24, top + vh // 2 - 70, (vw + tw) // 2 + 24, top + vh // 2 + 30, (10, 10, 10), 0.55)
            ui_text(img, txt, (vw - tw) // 2, top + vh // 2 + 10, 2.6, col, 5, _FONT_BOLD)

        # ---------------------------------------------------- bottom: guide
        self._draw_guide(img, 0, top + vh, vw, H, ctx)

        # ---------------------------------------------------- sidebar
        sx = self.side_x
        pad = int(UI_CONFIG["layout"]["padding"])
        ui_text(img, ui_fit(ctx.get("mode", ""), TS("small"), self.SIDE_W - 2 * pad), sx + pad, 52, TS("small"), UI_ACCENT, 1)
        state = ctx.get("state")
        cr = self.cube_rect
        ui_rrect(img, cr[0], cr[1], cr[2], cr[3], UI_BG, 8)
        if state is not None:
            if self.show_3d and self.cube3d is not None and ctx.get("basis") is not None:
                self.cube3d.draw(img, cr, state, ctx["basis"])
            else:
                self._net(img, (cr[0] + 6, cr[1] + 6, cr[2] - 6, cr[3] - 6), state,
                          ctx["basis"] if ctx.get("basis") is not None else np.eye(3))
        tr_ = self.toggle_rect
        ui_rrect(img, tr_[0], tr_[1], tr_[2], tr_[3], UI_PANEL_2, 6)
        lab = "2D" if self.show_3d else "3D"
        ui_text(img, lab, tr_[0] + (tr_[2] - tr_[0] - ui_text_w(lab, 0.4)) // 2, tr_[1] + 15, 0.4, UI_TEXT, 1)
        if self.show_3d:
            ui_text(img, "drag to turn", cr[0] + 8, cr[3] - 8, 0.34, UI_DIM, 1)
        # timer block
        timer = ctx.get("timer")
        ty = self.timer_y
        if timer:
            t_txt, t_col, sub = timer
            ui_text(img, t_txt, sx + pad, ty + 30, TS("timer"), t_col, 2, _FONT_BOLD)
            tx = sx + pad + ui_text_w(t_txt, TS("timer"), 2, _FONT_BOLD) + 16
            for k, l in enumerate(sub[:2]):
                ui_text(img, ui_fit(l, TS("small"), W - pad - tx), tx, ty + 14 + 18 * k, TS("small"), UI_MUTED, 1)
        splits = ctx.get("splits")
        if splits:
            x = sx + pad
            for name, val in splits:
                s_ = f"{name} {val}"
                ui_text(img, s_, x, ty + 54, TS("small"), UI_TEXT, 1)
                x += ui_text_w(s_, TS("small")) + 12
        # moves
        hist = ctx.get("history", [])
        my = self.moves_y
        ui_text(img, f"MOVES  {ctx.get('move_count', len(hist))}", sx + pad, my, TS("small"), UI_MUTED, 1)
        if ctx.get("unsure"):
            s_ = f"{ctx['unsure']} unsure"
            ui_text(img, s_, W - pad - ui_text_w(s_, TS("small")), my, TS("small"), UI_WARN, 1)
        lines = max(1, (self.buttons_top - 48 - my) // 26) if not self.capture_button.visible else 1
        self._chips(img, hist, sx + pad, my + 8, self.SIDE_W - 2 * pad, min(3, lines),
                    dim_from=(len(hist) - ctx["unsure"]) if ctx.get("unsure") else None)

        # ---------------------------------------------------- pop-up panel
        if self.panel is not None:
            self.panel.draw(img, self.panel_area)
        return img

    def _static_layer(self, H, W):
        """Panels, titles and buttons: redrawn only when a button changes."""
        hov = None
        for b in self.all_buttons:
            if b.visible and _in(b.rect, *self.hover):
                hov = id(b)
        key = (hov,) + tuple((b.key, b.label, b.visible, b.enabled, b.active) for b in self.all_buttons)
        if getattr(self, "_static_key", None) == key:
            return self._static
        img = np.empty((H, W, 3), dtype=np.uint8)
        img[:] = UI_BG
        cv2.rectangle(img, (0, 0), (self.VIEW_W, self.TOP_H - 1), UI_PANEL, -1)
        cv2.rectangle(img, (0, self.TOP_H + self.view_h), (self.VIEW_W, H), UI_PANEL, -1)
        cv2.line(img, (0, self.TOP_H + self.view_h), (self.VIEW_W, self.TOP_H + self.view_h), UI_LINE, 1)
        sx = self.side_x
        pad = int(UI_CONFIG["layout"]["padding"])
        cv2.rectangle(img, (sx, 0), (W, H), UI_PANEL, -1)
        cv2.line(img, (sx, 0), (sx, H), UI_LINE, 1)
        ui_text(img, UI_CONFIG["text"]["title"], sx + pad, 30, TS("title"), UI_TEXT, 2, _FONT_BOLD)
        for title, ty in self.section_titles:
            ui_text(img, title, sx + pad, ty, 0.38, UI_MUTED, 1)
        for b in self.all_buttons:
            self._button(img, b, id(b) == hov)
        self._static, self._static_key = img, key
        return img

    def _draw_guide(self, img, x0, y0, x1, y1, ctx):
        g = ctx.get("guide")
        if not g:
            ui_text(img, ui_fit(ctx.get("idle_hint", ""), 0.52, x1 - x0 - 32), x0 + 16, y0 + 34, 0.52, UI_MUTED, 1)
            for k, l in enumerate(ctx.get("idle_lines", [])[:2]):
                ui_text(img, ui_fit(l, 0.47, x1 - x0 - 32), x0 + 16, y0 + 64 + 24 * k, 0.47, UI_DIM, 1)
            return
        head, head_col = g["head"]
        ui_text(img, head, x0 + 16, y0 + 26, 0.58, head_col, 1, _FONT_BOLD)
        if g.get("right"):
            ui_text(img, g["right"], x1 - ui_text_w(g["right"], 0.45) - 16, y0 + 25, 0.45, UI_MUTED, 1)
        tokens, done, current, half, on_track, complete = g["tokens"], g["done"], g["current"], g["half"], g["on_track"], g["complete"]
        bw, bh, gap = 56, 42, 6
        ry = y0 + 38
        visible = max(1, (x1 - x0 - 32) // (bw + gap))
        start = max(0, min(current - visible // 3, max(0, len(tokens) - visible)))
        for slot, ti in enumerate(range(start, min(len(tokens), start + visible))):
            bx = x0 + 16 + slot * (bw + gap)
            if ti < done:
                fill, tc, border = UI_BG, UI_DIM, None
            elif ti == current and not complete:
                fill, tc, border = (UI_ACCENT if on_track else UI_BAD), (255, 255, 255), (255, 255, 255)
            else:
                fill, tc, border = UI_PANEL_2, UI_TEXT, None
            ui_rrect(img, bx, ry, bx + bw, ry + bh, fill, 7)
            if border:
                ui_rrect(img, bx, ry, bx + bw, ry + bh, border, 7, 2)
            if ti == current and half:
                cv2.rectangle(img, (bx + 6, ry + bh - 7), (bx + bw // 2, ry + bh - 4), UI_GOOD, -1)
            tw = ui_text_w(tokens[ti], TS("guide_move"), 2)
            ui_text(img, tokens[ti], bx + (bw - tw) // 2, ry + 29, TS("guide_move"), tc, 2)
        if start > 0:
            ui_text(img, "<", x0 + 3, ry + 28, 0.6, UI_MUTED, 2)
        if start + visible < len(tokens):
            ui_text(img, ">", x1 - 14, ry + 28, 0.6, UI_MUTED, 2)
        foot, foot_col = g.get("foot", ("", UI_MUTED))
        ui_text(img, ui_fit(foot, 0.5, x1 - x0 - 32), x0 + 16, y1 - 12, 0.5, foot_col, 1)


def render_replay(frame, info, ui_state):
    """Replay window: the recorded video, playback and analysis status, with
    clickable controls along the bottom."""
    h, w = frame.shape[:2]
    bar = 54
    img = np.empty((h + bar, w, 3), dtype=np.uint8)
    img[:h] = frame
    img[h:] = UI_PANEL
    ui_blend_box(img, 0, 0, w, 44, (15, 15, 15), 0.7)
    ui_text(img, info.get("title", ""), 14, 29, 0.62, UI_TEXT, 1, _FONT_BOLD)
    status, col = info.get("status", ("", UI_MUTED))
    ui_text(img, status, w - ui_text_w(status, 0.52) - 14, 29, 0.52, col, 1)
    progress = info.get("progress")
    if progress is not None:
        cv2.rectangle(img, (14, 50), (w - 14, 58), UI_PANEL_2, -1)
        cv2.rectangle(img, (14, 50), (14 + int((w - 28) * max(0.0, min(1.0, progress))), 58), UI_ACCENT, -1)
    y = 84
    for line, lcol in info.get("lines", []):
        tw = ui_text_w(line, 0.5)
        ui_blend_box(img, 10, y - 20, 24 + tw, y + 8, (15, 15, 15), 0.7)
        ui_text(img, line, 17, y, 0.5, lcol, 1)
        y += 32
    buttons = [(",", "< Prev"), ("p", "Pause" if not info.get("paused") else "Play"), (".", "Next >"),
               ("e", "Check video"), ("y", "Save video"), ("o", "Close")]
    bw = min(150, (w - 28 - 8 * (len(buttons) - 1)) // len(buttons))
    ui_state["rects"] = []
    for k, (key, label) in enumerate(buttons):
        x1 = 14 + k * (bw + 8)
        y1 = h + 10
        hov = _in((x1, y1, x1 + bw, y1 + 34), *ui_state.get("hover", (-1, -1)))
        ui_rrect(img, x1, y1, x1 + bw, y1 + 34, UI_LINE if hov else UI_PANEL_2, 7)
        ui_text(img, label, x1 + (bw - ui_text_w(label, 0.52)) // 2, y1 + 23, 0.52, UI_TEXT, 1)
        ui_state["rects"].append(((x1, y1, x1 + bw, y1 + 34), key))
    return img


def replay_mouse(event, x, y, flags, ui_state):
    if event == cv2.EVENT_MOUSEMOVE:
        ui_state["hover"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        for rect, key in ui_state.get("rects", []):
            if _in(rect, x, y):
                ui_state.setdefault("clicks", deque()).append(ord(key))
                break


class LiveCubeMotion:
    """Whole-cube movement from the YOLO outline (what the frame quality gate
    needs). Same numbers as MotionEvidenceGate's rigid-motion step, without
    the optical flow, which live tracking does not use (V54: motion prior off)."""

    def __init__(self):
        self.gate = MotionEvidenceGate()
        self.prev = None

    def process(self, quad):
        ev = {"whole_cube_translation_px": 0.0, "whole_cube_scale_delta": 0.0,
              "whole_cube_angle_deg": 0.0, "burst": False}
        if quad is None:
            self.prev = None
            return ev
        cur = MotionEvidenceGate._order_points(quad)
        if self.prev is not None:
            self.gate._rigid_transform(self.prev, cur)
            ev["whole_cube_translation_px"] = float(self.gate.last_whole_cube_translation_px)
            ev["whole_cube_scale_delta"] = float(self.gate.last_whole_cube_scale_delta)
            ev["whole_cube_angle_deg"] = float(self.gate.last_whole_cube_angle_deg)
        self.prev = cur
        return ev


def _order_quad(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_cube_quad(display_frame):
    """YOLO segmentation -> the cube face's four corners (display coords)."""
    h_frame, w_frame = display_frame.shape[:2]
    results = model(display_frame, conf=0.40, verbose=False, device=YOLO_DEVICE)[0]
    if results.masks is None or len(results.masks) == 0:
        return None
    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    best_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
    mask_resized = cv2.resize(masks[best_idx], (w_frame, h_frame))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    pts = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(cnt))
    return _order_quad(pts)


class RateMeter:
    def __init__(self):
        self.t = deque(maxlen=30)

    def tick(self, now):
        self.t.append(now)

    def rate(self):
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 and self.t[-1] > self.t[0] else 0.0


def main():
    load_app_settings()
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("\n[3D VIEWER READY] Open http://localhost:5001 in your browser.")
    print("🖱️ Click the buttons in the window (keyboard shortcuts are shown on them). [q] quits.\n")

    global MOVE_NAMING_ACTIVE
    A = types.SimpleNamespace()        # everything the loop and the tracking thread share
    A.lock = threading.RLock()          # held while the cube state / guide / timer change
    A.recorder = SolveRecorder(cube_tracker)
    A.decoder = LiveFrameDecoder(cube_tracker)
    A.decoder.reset(cube_tracker.state, cube_tracker.move_history, any_orientation=START_ANY_ORIENTATION)
    A.gate = FrameQualityGate()
    A.motion = LiveCubeMotion()
    A.guide = MoveGuide(cube_tracker)
    A.naming = MoveNaming(cube_tracker)
    MOVE_NAMING_ACTIVE = A.naming
    A.geo = CubeGeometry(cube_tracker)
    A.arrows = MoveArrows(cube_tracker, A.geo)
    A.camera_setup = CameraSetupSession()
    A.scan = ScrambleScanSession(cube_tracker, A.geo)
    A.timer = SolveTimer(A.geo, A.naming)
    A.trainer = CaseTrainer(cube_tracker, A.geo, A.naming)
    A.board = Leaderboard(A.timer)
    load_colour_calibration(kmeans_calibrator)
    A.colour_sets = ColourSets()
    A.colour_sets.import_current(kmeans_calibrator)
    A.lighting = LightingMonitor()
    A.space = SpaceHold()
    A.note = {"text": "", "until": 0.0, "colour": UI_TEXT}
    A.recent_grids = deque(maxlen=10)
    A.case_prior = None
    A.post_scan = {"state": None, "since": None}
    A.preset_index = 0
    A.seen = {"geom": None, "quad": False, "view_ok": True, "view_reason": "OK", "labs": None, "grid": None}
    A.replay = {"open": False, "idx": 0, "paused": False, "status": ("", UI_MUTED), "lines": []}
    A.replay_events = []
    A.history_snapshot = list(cube_tracker.move_history)
    A.camera_rate, A.track_rate = RateMeter(), RateMeter()
    A.colours_hint_until = 0.0

    ui = TrackerUI()
    ui.cube3d = Cube3DView(A.geo)
    basis = {"key": None, "M": None}
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WINDOW_MAIN, ui.on_mouse)
    replay_ui = {"hover": (-1, -1), "clicks": deque()}

    def say(text, colour=UI_TEXT, seconds=4.0):
        A.note.update(text=text, until=time.perf_counter() + seconds, colour=colour)
        print(f"💬 {text}")

    def set_tracker(state, history, last=None):
        cube_tracker.state = state.copy()
        cube_tracker.previous_state = state.copy()
        cube_tracker.move_history = list(history)
        cube_tracker.last_move_detected = last or (history[-1] if history else "None")
        cube_tracker.broadcast_state()

    def load_guide(kind, title, user_tokens, message=""):
        """tokens are in the names you see; the tracker gets them translated."""
        internal = A.naming.to_internal(list(user_tokens))
        A.guide.load(kind, title, internal, cube_tracker.state.copy(), cube_tracker.move_history)
        A.guide.display_tokens = list(user_tokens)
        A.guide.base_history = list(cube_tracker.move_history)
        A.guide.timer_armed = False
        A.guide.message = message
        print(f"🧭 [GUIDE] {title}: {' '.join(user_tokens) if user_tokens else '(nothing to do)'}")

    def fix_tracking():
        """The tracker disagrees with the guide (e.g. a misread): trust the
        guide at the step that matches what the camera sees right now."""
        g = A.guide
        if not g.active or not g.targets:
            say("Fix tracking needs a scramble or solution on screen. If the cube is solved, use Reset.", UI_WARN)
            return
        grid = ScrambleScanSession.stable_grid(list(A.recent_grids))
        if grid is None:
            say("Show the cube to the camera, then press Fix tracking again", UI_WARN)
            return
        flat = [str(c) for row in grid for c in row]
        lo, hi = max(0, g.progress - 2), min(len(g.targets) - 1, g.progress + 4)
        best = None
        for k in range(lo, hi + 1):
            score = sum(1 for i in range(9) if str(g.targets[k][18 + i]) == flat[i])
            rank = (score, -abs(k - (g.progress + 1)))
            if best is None or rank > best[0]:
                best = (rank, k)
        (score, _), k = best
        if score < 8:
            say("The camera view doesn't match the plan near this step - check the cube", UI_BAD)
            return
        history = cube_tracker.compressed_history(list(getattr(g, "base_history", [])) + list(getattr(g, "quarter_moves", []))[:k])
        target = g.targets[k].copy()
        A.decoder.reset(target, history)
        set_tracker(target, history)
        g.update(cube_tracker.state, cube_tracker.move_history)
        say(f"Tracking fixed - continuing from move {g.token_progress()[1] + 1}", UI_GOOD)

    # ------------------------------------------------------------ recording
    def start_recording(auto):
        if A.recorder.is_recording or A.recorder.analysis_in_progress:
            return
        A.replay_events = []
        A.history_snapshot = list(cube_tracker.move_history)
        A.replay.update(status=("", UI_MUTED), lines=[])
        A.recorder.start_recording(cube_tracker.state.copy(), list(cube_tracker.move_history), auto=auto)

    def stop_recording(show):
        if A.recorder.stop(list(A.replay_events)):
            A.replay.update(status=("Recorded - Check video re-reads it for mistakes", UI_TEXT), lines=[], idx=0)
            if show:
                A.replay["open"] = True

    def timer_started():
        if APP_SETTINGS.get("auto_record") and not A.recorder.is_recording:
            start_recording(auto=True)

    def timer_finished(record):
        A.guide.clear()
        A.case_prior = None
        if A.recorder.is_recording and A.recorder.auto:
            stop_recording(show=False)
        extra = "  (video kept - Replay / Save video)" if A.recorder.last_all_frames else ""
        say(f"{record['time_s']:.2f}s saved" + extra, UI_GOOD, 5.0)

    def timer_cancelled():
        A.case_prior = None
        if A.recorder.is_recording and A.recorder.auto:
            A.recorder.discard()

    # ------------------------------------------------------------ tracking
    def process_frame(frame, rec_idx):
        """One camera frame through perception + the move tracker (runs in
        the tracking thread, or inline when threaded tracking is off)."""
        display_frame = cv2.flip(frame, 1)
        h_frame, w_frame = display_frame.shape[:2]
        if A.recorder.analysis_in_progress:
            return      # the video check is using the cube detector; tracking waits for it
        quad_corners = detect_cube_quad(display_frame)
        current_grid = raw_lab_patches = current_probs = current_glare = None
        geom = None
        motion_evidence = A.motion.process(quad_corners)
        if quad_corners is not None:
            raw_corners = quad_corners.copy()
            raw_corners[:, 0] = w_frame - raw_corners[:, 0]
            rect_raw = _order_quad(raw_corners)
            current_grid, _flat, raw_lab_patches, current_probs, current_glare = extract_facelet_grid_soft(frame, rect_raw)
            dst_pts = np.array([[0, 0], [149, 0], [149, 149], [0, 149]], dtype="float32")
            geom = (cv2.getPerspectiveTransform(dst_pts * (150.0 / 149.0), rect_raw), w_frame, A.naming.mirror_view)
        with A.lock:
            A.recorder.mark_tracked(rec_idx)
            valid = validate_face_grid(current_grid)
            if valid:
                A.recent_grids.append([list(row) for row in current_grid])
            view_ok, view_reason = True, "OK"
            paused = A.scan.active
            if kmeans_calibrator.is_calibrated and not A.recorder.analysis_in_progress and not paused:
                if USE_FRAME_GATE:
                    view_ok, view_reason = A.gate.check(quad_corners, motion_evidence, display_frame.shape)
                else:
                    A.gate.last_reason = "GATE OFF"
                live_grid = current_grid if (view_ok and valid) else None
                live_probs = current_probs if (live_grid is not None and USE_SOFT_COLOURS) else None
                changed = A.decoder.step(live_grid, rec_idx, expected=A.guide.decoder_prior() or A.case_prior,
                                         probs=live_probs, motion_burst=None)
                # V53: follow slow lighting changes, only when the tracker is sure.
                if USE_COLOUR_ADAPTATION and live_probs is not None and raw_lab_patches is not None:
                    ds_now = A.decoder.stats()
                    if ds_now["contenders"] <= 1 and ds_now["confirmed_moves"] == len(A.decoder.best_moves()):
                        true_front = A.decoder.best_state_strings()[18:27]
                        labels = [c for row in live_grid for c in row]
                        if sum(a == b for a, b in zip(true_front, labels)) >= 8:
                            for i in range(9):
                                if current_glare[i] < 0.1 and labels[i] == true_front[i]:
                                    kmeans_calibrator.adapt(str(true_front[i]), raw_lab_patches[i])
                # V62: also follow the decoder when it only works out how the
                # cube is HELD (no new move).
                if A.post_scan["state"] is not None:
                    post_scan_reorient(quad_corners is not None)
                best_state = A.decoder.best_state_strings()
                if changed or not np.array_equal(best_state, cube_tracker.state):
                    set_tracker(best_state, A.decoder.best_history())
                if APP_SETTINGS.get("lighting_check") and view_ok and valid:
                    ds_l = A.decoder.stats()
                    sure = ds_l["contenders"] <= 2
                    verdict = A.lighting.update(
                        kmeans_calibrator, raw_lab_patches, current_glare, time.perf_counter(), A.colour_sets,
                        labels=[c for row in current_grid for c in row] if sure else None,
                        truth=list(A.decoder.best_state_strings()[18:27]) if sure else None,
                    ) if kmeans_calibrator.cluster_centers else None
                    if verdict and verdict[0] == "switch" and A.colour_sets.apply(verdict[1], kmeans_calibrator):
                        say(f"The light changed - switched colours to '{verdict[1]}'", UI_ACCENT, 6.0)
                    elif verdict and verdict[0] == "prompt":
                        A.colours_hint_until = time.perf_counter() + 12.0
                        say("The light has changed a lot - click Colours to recalibrate (or Colour sets)", UI_WARN, 10.0)
            if A.recorder.is_recording:
                A.recorder.add_observation(
                    rec_idx, current_grid if valid else None, motion_evidence,
                    extra={
                        "probs": np.round(current_probs, 4).tolist() if current_probs is not None else None,
                        "glare": [round(float(g), 3) for g in current_glare] if current_glare is not None else None,
                        "lab": [[round(float(v), 1) for v in lab] for lab in raw_lab_patches] if raw_lab_patches is not None else None,
                        "view": A.gate.last_reason,
                        "decoder_history_len": len(cube_tracker.move_history),
                    },
                )
                hist = list(cube_tracker.move_history)
                if hist != A.history_snapshot:
                    first_diff = 0
                    while first_diff < min(len(A.history_snapshot), len(hist)) and A.history_snapshot[first_diff] == hist[first_diff]:
                        first_diff += 1
                    A.replay_events = [e for e in A.replay_events if e.get("move_index", -1) < first_diff]
                    A.replay_events += [{"frame": rec_idx, "move": hist[i], "move_index": i} for i in range(first_diff, len(hist))]
                    A.history_snapshot = hist
            A.seen = {"geom": geom, "quad": quad_corners is not None, "view_ok": view_ok,
                      "view_reason": view_reason, "labs": raw_lab_patches, "grid": current_grid}
            A.track_rate.tick(time.perf_counter())

    def post_scan_reorient(cube_seen):
        """After a scan and before the first real turn, turning the whole cube
        round is not a move: whenever the cube is held still and the camera
        shows a face of the scanned cube exactly as scanned, start again from
        there. The first real turn ends this."""
        ps = A.post_scan
        sg = ScrambleScanSession.stable_grid(list(A.recent_grids)[-5:]) if cube_seen else None
        now_ = time.perf_counter()
        if sg is None or sg != ps.get("grid"):
            ps["grid"], ps["since"] = sg, now_
            return
        if now_ - ps["since"] < 0.25 or ps.get("checked") is sg:
            return
        ps["checked"] = sg
        flat = np.array([DECODER_CIDX[str(c)] for row in sg for c in row], dtype=np.uint8)
        views = ps["views"]
        match = (views[:, 18:27] == flat).sum(axis=1) >= 8
        if not match.any():
            ps["state"] = None             # a real turn has happened: normal tracking from here
            return
        best = A.decoder.states[0] if len(A.decoder.states) else None
        # keep the tracker as it is if its best guess already is one of these
        # (e.g. a turn and back): its evidence tells the scanned candidates apart
        if best is None or not any((views[match] == best).all(axis=1)):
            decoder_reset_candidates(A.decoder, ps["candidates"], [])

    threaded = bool(APP_SETTINGS.get("threaded_tracking", True))
    slot = {"frame": None, "idx": None, "seq": 0, "stop": False}
    cond = threading.Condition()

    def tracking_worker():
        done_seq = 0
        min_gap = 1.0 / max(1.0, float(APP_SETTINGS.get("tracking_max_hz", 12.0)))
        last = 0.0
        while True:
            with cond:
                while slot["seq"] == done_seq and not slot["stop"]:
                    cond.wait(0.5)
                if slot["stop"]:
                    return
                frame, idx, done_seq = slot["frame"], slot["idx"], slot["seq"]
            wait = min_gap - (time.perf_counter() - last)
            if wait > 0:
                time.sleep(wait)
                with cond:                      # a newer frame may have arrived meanwhile
                    frame, idx, done_seq = slot["frame"], slot["idx"], slot["seq"]
            last = time.perf_counter()
            try:
                process_frame(frame, idx)
            except Exception:
                traceback.print_exc()

    if threaded:
        threading.Thread(target=tracking_worker, daemon=True).start()

    # ------------------------------------------------------------ actions
    def do_key(key):
        now = time.perf_counter()
        if key == 32:  # SPACE: speedcube timer
            if A.timer.state == "RUNNING":
                rec = A.timer.stop_manual(cube_tracker.state, cube_tracker.move_history, now)
                if rec:
                    timer_finished(rec)
            else:
                A.space.press(now)
        elif key == ord("x"):
            if A.recorder.analysis_in_progress:
                say("Still checking the previous recording - please wait", UI_WARN)
            elif A.recorder.is_recording:
                stop_recording(show=True)
            else:
                start_recording(auto=False)
                say("Recording... click Record again to stop", UI_BAD)
        elif key == ord("u"):
            APP_SETTINGS["auto_record"] = not APP_SETTINGS.get("auto_record")
            save_app_settings()
            say("Auto-record " + ("on: every timed solve is recorded" if APP_SETTINGS["auto_record"] else "off"), UI_ACCENT)
        elif key == ord("y"):
            if A.recorder.save_video(mirror=A.naming.mirror_view) is None:
                say("Nothing recorded yet", UI_WARN)
        elif key == ord("e"):
            if A.recorder.is_recording:
                say("Stop recording first", UI_WARN)
            elif A.recorder.analysis_in_progress:
                say("Video check already running", UI_WARN)
            elif A.recorder.reanalyze_recorded_video(model, extract_facelet_grid_with_debug,
                                                     source_fps=A.recorder.last_completed_source_fps):
                A.replay.update(status=("Checking the video...", UI_WARN), lines=[], open=True)
            else:
                say("Nothing to check yet - record a solve first", UI_WARN)
        elif key == ord("o"):
            if A.replay["open"]:
                A.replay["open"] = False
            elif A.recorder.last_all_frames:
                A.replay.update(open=True, idx=0)
            else:
                say("No recording yet - use Record, or time a solve with auto-record on", UI_WARN)
        elif key == ord("p"):
            A.replay["paused"] = not A.replay["paused"]
        elif key in (ord("."), ord(",")):
            n = len(A.recorder.last_all_frames)
            if n:
                A.replay["paused"] = True
                A.replay["idx"] = (A.replay["idx"] + (1 if key == ord(".") else -1)) % n
        elif key == ord("c"):
            labs = A.seen["labs"]
            if labs is not None and not kmeans_calibrator.is_calibrated:
                target = kmeans_calibrator.get_target_color()
                if kmeans_calibrator.capture_face(labs):
                    save_colour_calibration(kmeans_calibrator)
                    name = A.colour_sets.add_from(kmeans_calibrator)
                    say(f"Colours saved as '{name}'", UI_GOOD, 6.0)
                else:
                    say(f"Got the {target} face", UI_GOOD, 2.0)
            elif not kmeans_calibrator.is_calibrated:
                say("I can't see the cube - hold the face up to the camera", UI_WARN)
        elif key == ord("v"):
            A.scan.active = False
            A.camera_setup.active = False
            A.post_scan["state"] = None
            restart_colour_calibration(kmeans_calibrator)
            say("Colour setup: show each face of a SOLVED cube as asked", UI_WARN)
        elif key == ord("z"):
            if ui.panel is not None and ui.panel.name == "colours":
                ui.panel = None
            else:
                ui.panel = ListPanel("colours", "Colour sets")
                A.colour_sets.fill_panel(ui.panel)
        elif key == ord("l"):
            if ui.panel is not None and ui.panel.name == "board":
                ui.panel = None
            else:
                ui.panel = ListPanel("board", "Leaderboard")
                A.board.fill_panel(ui.panel)
        elif key == ord("d"):
            ui.show_3d = not ui.show_3d
        elif key == ord("f"):
            A.naming.use_preset("camera facing you")
            A.trainer._validate()
            say(A.naming.frame_description, UI_ACCENT)
        elif key == ord("t"):
            A.naming.use_preset("top-down")
            A.trainer._validate()
            say(A.naming.frame_description + " (Learn cam if moves look rotated)", UI_ACCENT)
        elif key == ord("m"):
            if A.camera_setup.active:
                A.camera_setup.active = False
                A.camera_setup.message = ""
                say("Camera setup cancelled")
            else:
                A.scan.active = False
                A.post_scan["state"] = None
                A.camera_setup.start(A.decoder)
        elif key == ord("i"):
            if A.scan.active or A.post_scan["state"] is not None:
                A.scan.active = False
                A.scan.message = ""
                A.post_scan["state"] = None
                A.decoder.reset(cube_tracker.state, cube_tracker.move_history)
                say("Scan cancelled")
            else:
                A.camera_setup.active = False
                A.camera_setup.message = ""
                A.guide.clear()
                set_tracker(cube_tracker.state, [])      # a new cube: start a fresh move list
                A.scan.start()
        elif key == ord("b"):
            if A.scan.active:
                scanned = A.scan.capture(ScrambleScanSession.stable_grid(list(A.recent_grids)))
                if scanned is not None:
                    A.post_scan["candidates"] = scan_candidates(A.scan) or [scanned]
                    scanned = A.post_scan["candidates"][0]
                    A.guide.clear()
                    set_tracker(scanned, [])
                    # Turning the cube round to the way you want to start is
                    # not a move: see post_scan_reorient().
                    views = []
                    for cand in A.post_scan["candidates"]:
                        views += [np.array([DECODER_CIDX[str(c)] for c in st], dtype=np.uint8)
                                  for st in all_orientations(np.asarray(cand), cube_tracker.permutations)]
                    A.post_scan.update(state=scanned.copy(), since=None, grid=None, checked=None, views=np.array(views))
                    decoder_reset_candidates(A.decoder, A.post_scan["candidates"], [])
                    say("Scan done - hold the cube however you like and start", UI_GOOD, 5.0)
                elif A.scan.message:
                    say(A.scan.message.replace("[i]", "Scan cube"), UI_BAD, 6.0)
        elif key == ord("a"):
            if A.timer.state in ("READY", "RUNNING"):
                A.timer.cancel()
                timer_cancelled()
            else:
                A.timer.arm(cube_tracker.move_history, now)
                timer_started()
        elif key in (ord("j"), ord("h")):
            setup, msg = A.trainer.next_case("PLL" if key == ord("j") else "OLL", cube_tracker.state)
            say(msg, UI_ACCENT if setup else UI_WARN)
            if setup:
                load_guide("SCRAMBLE", "CASE SETUP", setup, msg)
        elif key == ord("w"):
            fix_tracking()
        elif key == ord("s") and decoder_unsure(A.decoder) and len(A.decoder.best_moves()) < 4:
            say("I can't tell yet exactly how the cube is - turn a side (R, L, U or D) and back, then Solve again", UI_WARN, 6.0)
        elif key == ord("s"):
            tokens, err = solve_state_with_kociemba(cube_tracker, cube_tracker.state)
            if err:
                say(f"Can't solve: {err}", UI_BAD)
                A.guide.clear()
            elif not tokens:
                say("The cube is already solved", UI_GOOD)
            else:
                load_guide("SOLUTION", f"SOLUTION ({len(tokens)} moves)", A.naming.to_user(tokens),
                           "Follow the highlighted move")
                cube_tracker.trigger_kociemba_solve()  # also sends it to the 3D viewer
        elif key == ord("g"):
            load_guide("SCRAMBLE", "RANDOM SCRAMBLE", random_scramble(20), "Then hold SPACE and release to time your solve")
        elif key == ord("n"):
            name, seq = PRESET_SCRAMBLES[A.preset_index % len(PRESET_SCRAMBLES)]
            A.preset_index += 1
            # presets are standard notation: translate so the pattern comes out right
            load_guide("SCRAMBLE", name.upper(), A.naming.to_user(A.naming.std_to_internal(seq.split())),
                       "Click Preset again for the next one")
        elif key == ord("k"):
            A.guide.clear()
        elif key == ord("r"):
            cube_tracker.reset_to_solved()
            A.decoder.reset(cube_tracker.state, cube_tracker.move_history, any_orientation=START_ANY_ORIENTATION)
            A.guide.clear()
            A.scan.active = False
            A.post_scan["state"] = None
            A.timer.cancel()
            timer_cancelled()
            say("Reset: hold a SOLVED cube any way you like", UI_ACCENT)

    # ------------------------------------------------------------ main loop
    loop_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.perf_counter()
        A.camera_rate.tick(now)
        rec_idx = A.recorder.add_frame(frame, now)
        if threaded:
            with cond:
                slot.update(frame=frame, idx=rec_idx, seq=slot["seq"] + 1)
                cond.notify()
        else:
            process_frame(frame, rec_idx)

        with A.lock:
            # ---- video check result
            final_result = A.recorder.consume_final_result()
            if final_result is not None:
                if final_result.get("ok"):
                    set_tracker(final_result["state"], final_result["history"],
                                (final_result["new_moves"][-1] if final_result.get("new_moves") else "VERIFIED"))
                    A.decoder.reset(cube_tracker.state, cube_tracker.move_history)
                    fixed = bool(final_result.get("corrected"))
                    A.replay["status"] = (("Corrected the moves" if fixed else "Moves confirmed"), UI_GOOD)
                    A.replay["lines"] = [
                        ("Live:     " + " ".join(user_names(final_result.get("live_suffix", [])))[-110:], UI_MUTED),
                        ("Checked:  " + " ".join(user_names(final_result.get("new_moves", [])))[-110:], UI_TEXT),
                    ]
                    say("Video check: " + ("the moves were corrected" if fixed else "moves confirmed"), UI_GOOD)
                else:
                    reason = str(final_result.get("reason", ""))
                    A.replay["status"] = ("No better explanation found - live moves kept", UI_WARN)
                    A.replay["lines"] = [(reason[:110], UI_MUTED)]
                    say("Video check found nothing better than the live tracking - kept the live moves", UI_WARN)

            # ---- guide, camera setup, timer
            if A.guide.active:
                A.guide.update(cube_tracker.state, cube_tracker.move_history)
            A.camera_setup.update(A.decoder, A.naming)
            if (A.guide.active and A.guide.kind == "SCRAMBLE" and A.guide.complete
                    and not getattr(A.guide, "timer_armed", True)):
                A.guide.timer_armed = True
                if A.trainer.pending:
                    case_name, case_alg = A.trainer.pending
                    A.trainer.pending = None
                    A.timer.arm(cube_tracker.move_history, now, A.guide.tokens, mode="CASE", case=case_name,
                                case_alg=" ".join(A.trainer.display_alg(case_alg)))
                    A.case_prior = build_case_prior(cube_tracker, A.naming, cube_tracker.state.copy(), case_alg.split())
                else:
                    A.timer.arm(cube_tracker.move_history, now, A.guide.tokens)
                timer_started()
                # The scramble is done: close its guide so it stops steering
                # the tracker back towards the scrambled state while you solve.
                A.guide.clear()
            space = A.space.poll(now)
            if space == "released":
                A.timer.start_now(cube_tracker.move_history, now)
                timer_started()
            elif space == "tap" and A.timer.state not in ("READY", "RUNNING"):
                A.timer.arm(cube_tracker.move_history, now)
                timer_started()
                say("Timer ready - starts on your first turn (or hold SPACE and release)", UI_WARN)
            rec = A.timer.update(cube_tracker.state, cube_tracker.move_history, now)
            if rec:
                timer_finished(rec)
            if A.timer.state not in ("READY", "RUNNING"):
                A.case_prior = None

            # ---- panels
            if ui.panel is not None:
                while ui.panel is not None and ui.panel.actions:
                    action, value = ui.panel.actions.popleft()
                    if ui.panel.name == "board":
                        res = A.board.handle(ui.panel, action, value, say)
                    else:
                        res = A.colour_sets.handle(ui.panel, action, value, kmeans_calibrator, say)
                    if res == "close":
                        ui.panel = None
                    elif res:
                        ui.panel = None
                        do_key(ord(res))

            # ---- screen
            key_now = (A.naming.frame_name, tuple(A.naming.to_int.get(k, "") for k in ("R", "U", "F")))
            if basis["key"] != key_now:
                basis.update(key=key_now, M=naming_basis(cube_tracker, A.geo, A.naming))
            ctx = build_screen_context(A, now, space)
            ctx["basis"] = basis["M"]
            for b in ui.all_buttons:
                b.active = ((b.key == "x" and A.recorder.is_recording and not A.recorder.auto)
                            or (b.key == "u" and APP_SETTINGS.get("auto_record"))
                            or (b.key == "a" and A.timer.state in ("READY", "RUNNING"))
                            or (b.key == "f" and A.naming.frame_name == "camera facing you")
                            or (b.key == "t" and A.naming.frame_name == "top-down")
                            or (b.key == "i" and (A.scan.active or A.post_scan["state"] is not None))
                            or (b.key == "m" and A.camera_setup.active)
                            or (b.key == "v" and (not kmeans_calibrator.is_calibrated or now < A.colours_hint_until))
                            or (b.key == "l" and ui.panel is not None and ui.panel.name == "board")
                            or (b.key == "z" and ui.panel is not None and ui.panel.name == "colours")
                            or (b.key == "o" and A.replay["open"]))
                b.enabled = not ((b.key in ("e", "y") and (A.recorder.is_recording or not A.recorder.last_all_frames))
                                 or (b.key == "o" and not A.recorder.last_all_frames))
            ui.button("x").label = "Stop" if (A.recorder.is_recording and not A.recorder.auto) else "Record"
            cb = ui.capture_button
            cb.visible = (not kmeans_calibrator.is_calibrated) or A.scan.active
            cb.key, cb.label, cb.hint = ("b", "Capture this face", "B") if A.scan.active else ("c", "Capture colours", "C")
        view = cv2.flip(frame, 1) if A.naming.mirror_view else frame
        cv2.imshow(WINDOW_MAIN, ui.render(view, ctx))

        # ---- replay window
        frames = A.recorder.last_all_frames
        if A.replay["open"] and frames:
            A.replay["idx"] = min(A.replay["idx"], len(frames) - 1)
            rframe = frames[A.replay["idx"]]
            if A.naming.mirror_view:
                rframe = cv2.flip(rframe, 1)
            status = A.replay["status"]
            progress = None
            if A.recorder.analysis_in_progress:
                st = A.recorder.analysis_status or {}
                progress = float(st.get("progress", 0.0)) / 100.0
                status = (f"Checking the video... {st.get('progress', 0.0):.0f}%", UI_WARN)
            elif A.recorder.save_status:
                status = (A.recorder.save_status[-70:], UI_TEXT)
            info = {"title": f"Replay  {A.replay['idx'] + 1}/{len(frames)}", "status": status,
                    "progress": progress, "paused": A.replay["paused"], "lines": A.replay["lines"]}
            if not A.replay.get("window"):
                cv2.namedWindow(WINDOW_REPLAY, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.setMouseCallback(WINDOW_REPLAY, replay_mouse, replay_ui)
                A.replay["window"] = True
            cv2.imshow(WINDOW_REPLAY, render_replay(rframe, info, replay_ui))
            if not A.replay["paused"]:
                A.replay["idx"] = (A.replay["idx"] + 1) % len(frames)
        elif A.replay.get("window"):
            try:
                cv2.destroyWindow(WINDOW_REPLAY)
            except cv2.error:
                pass
            A.replay["window"] = False

        loop_counter += 1
        if loop_counter % 300 == 0:
            ds = A.decoder.stats()
            print(f"📈 [LIVE] camera {A.camera_rate.rate():.0f} fps | tracking {A.track_rate.rate():.1f}/s | "
                  f"decoder step {ds['last_step_ms']:.1f}ms | hypotheses={ds['hypotheses']} contenders={ds['contenders']}")

        # ---- keyboard + mouse
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and ui.panel is not None and (ui.panel.edit is not None or key == 27):
            if ui.panel.edit is not None:
                ui.panel.key(key)
            else:
                ui.panel = None
            continue
        if key == 255:
            key = ui.pop_key()
            if key is None and replay_ui["clicks"]:
                key = replay_ui["clicks"].popleft()
        if key is None or key == 255:
            continue
        if key == ord("q"):
            break
        with A.lock:
            do_key(key)

    with cond:
        slot["stop"] = True
        cond.notify()
    cap.release()
    cv2.destroyAllWindows()


def build_screen_context(A, now, space):
    tracker, decoder, guide, naming, timer = cube_tracker, A.decoder, A.guide, A.naming, A.timer
    seen = A.seen
    ctx = {"geom": seen["geom"], "view_ok": seen["view_ok"], "state": tracker.state.copy(),
           "mode_badge": (naming.label, UI_ACCENT if naming.frame_name == "camera facing you" else UI_WARN),
           "fps_text": f"camera {A.camera_rate.rate():.0f} fps   tracking {A.track_rate.rate():.0f}/s",
           "recording": A.recorder.is_recording, "mode": naming.label}

    # ---- status chip
    ds = decoder.stats()
    total_q = len(decoder.best_moves())
    unsure = max(0, total_q - ds["confirmed_moves"])
    if not kmeans_calibrator.is_calibrated:
        ctx["status"] = ("Colour setup", UI_WARN)
    elif A.scan.active:
        ctx["status"] = ("Scanning", UI_ACCENT)
    elif not seen["quad"]:
        ctx["status"] = ("No cube in view", UI_DIM)
    elif seen["view_reason"] in ("CUBE MOVING", "SETTLING"):
        ctx["status"] = ("Hold still", UI_WARN)
    elif seen["view_reason"] == "STEEP ANGLE":
        ctx["status"] = ("Face the camera", UI_WARN)
    elif not decoder.orientation_resolved():
        ctx["status"] = ("Turn R, L, U or D once", UI_WARN)
    else:
        ctx["status"] = ("Tracking", UI_GOOD)
    hist_user = user_names(tracker.move_history)
    ctx["history"] = hist_user[-40:]
    ctx["move_count"] = len(hist_user)
    ctx["unsure"] = min(unsure, len(ctx["history"]))
    ctx["top_info"] = []
    if A.recorder.analysis_in_progress:
        ctx["top_info"].append(f"checking video {float((A.recorder.analysis_status or {}).get('progress', 0)):.0f}%")

    # ---- toasts
    toasts = []
    if A.note["text"] and now < A.note["until"]:
        toasts.append((A.note["text"], A.note["colour"]))
    if A.camera_setup.active:
        toasts.append((A.camera_setup.message, UI_ACCENT))
    if A.scan.message and A.scan.active:
        toasts.append((A.scan.message.replace("[b]", "Capture"), UI_ACCENT))
    ctx["toasts"] = toasts

    # ---- colour setup card
    if not kmeans_calibrator.is_calibrated:
        target = kmeans_calibrator.get_target_color()
        if target:
            ctx["card"] = (f"Colour setup {kmeans_calibrator.current_scan_idx + 1}/6",
                           [f"Hold the SOLVED cube with the {target.upper()} centre facing the camera",
                            "then click Capture colours (or press C). Saved as a colour set."], UI_WARN)

    # ---- timer
    st = timer.stats()
    sub = []
    if st:
        sub.append(f"PB {st['best']:.2f}" + (f"   ao5 {st['ao5']:.2f}" if st.get("ao5") else ""))
        sub.append((f"ao12 {st['ao12']:.2f}   " if st.get("ao12") else "") + f"{st['count']} solves")
    if space in ("holding", "ready"):
        col = C("timer_ready") if space == "ready" else C("timer_hold")
        ctx["timer"] = ("0.00", col, sub)
        ctx["big_timer"] = ("0.00", col)
    elif timer.state == "READY":
        ctx["timer"] = ("0.00", UI_WARN, sub)
        toasts.append(("Timer ready - starts on your first turn" if timer.mode == "SOLVE"
                       else f"{timer.case}: solve it! ({timer.case_alg})", UI_WARN))
    elif timer.state == "RUNNING":
        ctx["timer"] = (f"{timer.elapsed(now):.2f}", UI_ACCENT, sub)
    elif timer.state == "DONE" and timer.last_result:
        r = timer.last_result
        ctx["timer"] = (f"{r['time_s']:.2f}", UI_GOOD, [f"{r['move_count']} moves  {r['tps']} tps"] + sub)
    else:
        ctx["timer"] = ("0.00", UI_DIM, sub or [UI_CONFIG["text"]["timer_hint"]])
    if timer.state in ("RUNNING", "DONE") and timer.splits:
        ctx["splits"] = [(k, f"{v:.1f}") for k, v in timer.splits.items()]

    # ---- guide + arrow
    arrow_move, arrow_colour, arrow_label, arrow_double = None, UI_ARROW, None, False
    if guide.active:
        disp = getattr(guide, "display_tokens", None) or naming.to_user(guide.tokens)
        done, current, half = guide.token_progress()
        total = len(guide.tokens)
        if guide.complete:
            head = (f"{guide.title}: done" + ("  -  cube solved!" if guide.kind == "SOLUTION" else ""), UI_GOOD)
        else:
            head = (f"{guide.title}   {done}/{total}", UI_TEXT if guide.on_track else UI_BAD)
        g = {"head": head, "tokens": disp, "done": done, "current": current, "half": half,
             "on_track": guide.on_track, "complete": guide.complete, "right": naming.frame_description}
        if not guide.on_track and not guide.complete:
            fix = naming.to_user(guide.fix[:8]) if guide.fix else []
            g["foot"] = ((f"Off track - undo with  {' '.join(fix)}" if fix else "Off track")
                         + "     (wrong? click Fix tracking)", UI_BAD)
            if guide.fix:
                arrow_move, arrow_colour = guide.fix[0], UI_ARROW_FIX
                arrow_double = guide.fix[0].endswith("2")
                arrow_label = f"undo: {naming.to_user([guide.fix[0]])[0]}"
        elif not guide.complete and current < total:
            tok = guide.tokens[current]
            arrow_move = tok
            arrow_double = tok.endswith("2") and not half
            arrow_label = disp[current]
            g["foot"] = (guide.message or "", UI_MUTED)
        else:
            g["foot"] = (guide.message or "", UI_MUTED)
        ctx["guide"] = g
    else:
        ctx["idle_hint"] = UI_CONFIG["text"]["idle_hint"]
        last = timer.last_result
        lines = []
        if A.case_prior is not None and timer.case_alg:
            lines.append(f"{timer.case}:  {timer.case_alg}")
        elif last:
            lines.append(f"Last solve {last['time_s']:.2f}s, {last['move_count']} moves  (Leaderboard for all your times)")
        lines.append(naming.frame_description)
        ctx["idle_lines"] = lines
    if arrow_move is not None and seen["geom"] is not None:
        base = arrow_move[:-1] if arrow_move.endswith("2") else arrow_move
        plan = A.arrows.plan(base)
        if plan is not None:
            if plan.get("hidden"):
                arrow_label = (arrow_label or "") + "  (face the camera can't see)"
            if naming.frame_name != "camera facing you":
                arrow_label = (arrow_label or "") + f"  [{naming.label}]"
            ctx["arrow"] = {"plan": plan, "colour": arrow_colour, "label": arrow_label, "double": arrow_double}
    return ctx


if __name__ == "__main__":
    main()
