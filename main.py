import glob
import threading
import time
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO

# --- FLASK & SOCKET.IO SETUP ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- CAMERA INITIALIZATION ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

weights_list = glob.glob("runs/segment/rubiks_seg_model/weights/best.pt")
MODEL_PATH = weights_list[0] if weights_list else "yolov8n-seg.pt"
model = YOLO(MODEL_PATH)

# --- HSV COLOR RANGES & CONSTRAINTS ---
COLOR_RANGES_HSV = {
    "Red": [
        ((0, 150, 60), (6, 255, 255)),
        ((168, 150, 60), (180, 255, 255)),
    ],
    "Orange": [((7, 140, 60), (22, 255, 255))],
    "Yellow": [((23, 80, 100), (38, 255, 255))],
    "Green": [((39, 80, 60), (85, 255, 255))],
    "Blue": [((86, 80, 60), (125, 255, 255))],
    "White": [((0, 0, 150), (180, 60, 255))],
}

VALID_COLORS = ["White", "Yellow", "Red", "Orange", "Green", "Blue"]

OPPOSITES = {
    "White": "Yellow",
    "Yellow": "White",
    "Green": "Blue",
    "Blue": "Green",
    "Orange": "Red",
    "Red": "Orange",
}

MOTION_TO_ENTRY_EDGE = {
    "RIGHT": "LEFT",
    "LEFT": "RIGHT",
    "TOP": "BOTTOM",
    "BOTTOM": "TOP",
}

MIN_MOTION_MAGNITUDE = 2.0
MOTION_STILL_THRESHOLD = 1.0
RESCAN_COOLDOWN = 1.0  # Cooldown time in seconds per face

# --- 3D PIECE PHYSICAL CONSTRAINTS REGISTRY ---
VALID_EDGES = [
    {"White", "Green"},  {"White", "Blue"},   {"White", "Red"},    {"White", "Orange"},
    {"Yellow", "Green"}, {"Yellow", "Blue"},  {"Yellow", "Red"},   {"Yellow", "Orange"},
    {"Green", "Red"},    {"Green", "Orange"}, {"Blue", "Red"},     {"Blue", "Orange"}
]

VALID_CORNERS = [
    {"White", "Green", "Red"},    {"White", "Green", "Orange"},
    {"White", "Blue", "Red"},     {"White", "Blue", "Orange"},
    {"Yellow", "Green", "Red"},   {"Yellow", "Green", "Orange"},
    {"Yellow", "Blue", "Red"},    {"Yellow", "Blue", "Orange"}
]


class CubeModel3D:

    # 3D Adjacency Map for all 24 non-center facelets
    PIECE_ADJACENCIES = {
        # WHITE FACE
        ("White", 0, 1): [("Red", 0, 1)],
        ("White", 2, 1): [("Orange", 0, 1)],
        ("White", 1, 0): [("Green", 0, 1)],
        ("White", 1, 2): [("Blue", 0, 1)],
        ("White", 0, 0): [("Red", 0, 0), ("Green", 0, 2)],
        ("White", 0, 2): [("Red", 0, 2), ("Blue", 0, 0)],
        ("White", 2, 0): [("Orange", 0, 2), ("Green", 0, 0)],
        ("White", 2, 2): [("Orange", 0, 0), ("Blue", 0, 2)],

        # YELLOW FACE
        ("Yellow", 0, 1): [("Red", 2, 1)],
        ("Yellow", 2, 1): [("Orange", 2, 1)],
        ("Yellow", 1, 0): [("Green", 2, 1)],
        ("Yellow", 1, 2): [("Blue", 2, 1)],
        ("Yellow", 0, 0): [("Red", 2, 0), ("Green", 2, 2)],
        ("Yellow", 0, 2): [("Red", 2, 2), ("Blue", 2, 0)],
        ("Yellow", 2, 0): [("Orange", 2, 2), ("Green", 2, 0)],
        ("Yellow", 2, 2): [("Orange", 2, 0), ("Blue", 2, 2)],

        # RED FACE
        ("Red", 0, 1): [("White", 2, 1)],
        ("Red", 2, 1): [("Yellow", 0, 1)],
        ("Red", 1, 0): [("Green", 1, 2)],
        ("Red", 1, 2): [("Blue", 1, 0)],
        ("Red", 0, 0): [("White", 2, 0), ("Green", 0, 2)],
        ("Red", 0, 2): [("White", 2, 2), ("Blue", 0, 0)],
        ("Red", 2, 0): [("Yellow", 0, 0), ("Green", 2, 2)],
        ("Red", 2, 2): [("Yellow", 0, 2), ("Blue", 2, 0)],

        # ORANGE FACE
        ("Orange", 0, 1): [("White", 2, 1)],
        ("Orange", 2, 1): [("Yellow", 2, 1)],
        ("Orange", 1, 0): [("Blue", 1, 2)],
        ("Orange", 1, 2): [("Green", 1, 0)],
        ("Orange", 0, 0): [("White", 2, 2), ("Blue", 0, 2)],
        ("Orange", 0, 2): [("White", 2, 0), ("Green", 0, 0)],
        ("Orange", 2, 0): [("Yellow", 2, 2), ("Blue", 2, 2)],
        ("Orange", 2, 2): [("Yellow", 2, 0), ("Green", 2, 0)],

        # GREEN FACE
        ("Green", 0, 1): [("White", 1, 0)],
        ("Green", 2, 1): [("Yellow", 1, 0)],
        ("Green", 1, 0): [("Orange", 1, 2)],
        ("Green", 1, 2): [("Red", 1, 0)],
        ("Green", 0, 0): [("White", 2, 0), ("Orange", 0, 2)],
        ("Green", 0, 2): [("White", 0, 0), ("Red", 0, 0)],
        ("Green", 2, 0): [("Yellow", 2, 0), ("Orange", 2, 2)],
        ("Green", 2, 2): [("Yellow", 0, 0), ("Red", 2, 0)],

        # BLUE FACE
        ("Blue", 0, 1): [("White", 1, 2)],
        ("Blue", 2, 1): [("Yellow", 1, 2)],
        ("Blue", 1, 0): [("Red", 1, 2)],
        ("Blue", 1, 2): [("Orange", 1, 0)],
        ("Blue", 0, 0): [("White", 0, 2), ("Red", 0, 2)],
        ("Blue", 0, 2): [("White", 2, 2), ("Orange", 0, 0)],
        ("Blue", 2, 0): [("Yellow", 0, 2), ("Red", 2, 2)],
        ("Blue", 2, 2): [("Yellow", 2, 2), ("Orange", 2, 0)],
    }

    def __init__(self):
        self.faces = {
            col: [
                ["Unknown", "Unknown", "Unknown"],
                ["Unknown", col, "Unknown"],
                ["Unknown", "Unknown", "Unknown"],
            ]
            for col in VALID_COLORS
        }

    def reset(self):
        self.__init__()

    def get_color_counts(self):
        counts = {c: 0 for c in VALID_COLORS}
        for face, grid in self.faces.items():
            for r in range(3):
                for c in range(3):
                    val = grid[r][c]
                    if val in counts:
                        counts[val] += 1
        return counts

    def get_unknown_locations(self):
        unknowns = []
        for face, grid in self.faces.items():
            for r in range(3):
                for c in range(3):
                    if grid[r][c] not in VALID_COLORS:
                        unknowns.append((face, r, c))
        return unknowns

    def is_valid_piece(self, colors):
        valid_cols = [c for c in colors if c in VALID_COLORS]
        if len(valid_cols) != len(set(valid_cols)):
            return False
        for c in valid_cols:
            if OPPOSITES.get(c) in valid_cols:
                return False
        return True

    def get_shared_edge_indices(self, entry_edge):
        if entry_edge == "LEFT":
            return [(0, 2), (1, 2), (2, 2)], [(0, 0), (1, 0), (2, 0)]
        elif entry_edge == "RIGHT":
            return [(0, 0), (1, 0), (2, 0)], [(0, 2), (1, 2), (2, 2)]
        elif entry_edge == "TOP":
            return [(2, 0), (2, 1), (2, 2)], [(0, 0), (0, 1), (0, 2)]
        elif entry_edge == "BOTTOM":
            return [(0, 0), (0, 1), (0, 2)], [(2, 0), (2, 1), (2, 2)]
        return [], []

    def score_orientation(self, prev_color, scanned_grid, entry_edge):
        prev_indices, curr_indices = self.get_shared_edge_indices(entry_edge)
        if not prev_indices:
            return 0
        score = 0
        prev_grid = self.faces[prev_color]
        for (pr, pc), (cr, cc) in zip(prev_indices, curr_indices):
            prev_val = prev_grid[pr][pc]
            curr_val = scanned_grid[cr][cc]
            if prev_val in VALID_COLORS and curr_val in VALID_COLORS:
                if self.is_valid_piece([prev_val, curr_val]):
                    score += 2
                    if prev_val == curr_val:
                        score += 1
                else:
                    score -= 5
        return score

    def get_piece_colors(self, face_color, r, c, exclude_self=False):
        if (r, c) == (1, 1):
            return [face_color]

        piece_colors = []
        if not exclude_self:
            val = self.faces[face_color][r][c]
            if val in VALID_COLORS:
                piece_colors.append(val)

        if (face_color, r, c) in self.PIECE_ADJACENCIES:
            for adj_face, adj_r, adj_c in self.PIECE_ADJACENCIES[(face_color, r, c)]:
                adj_val = self.faces[adj_face][adj_r][adj_c]
                if adj_val in VALID_COLORS:
                    piece_colors.append(adj_val)

        return piece_colors

    def get_existing_logged_edges(self):
        logged_edges = []
        edge_coords = [
            (("White", 1, 0), ("Green", 0, 1)), (("White", 1, 2), ("Blue", 0, 1)),
            (("White", 0, 1), ("Red", 0, 1)),   (("White", 2, 1), ("Orange", 0, 1)),
            (("Yellow", 1, 0), ("Green", 2, 1)), (("Yellow", 1, 2), ("Blue", 2, 1)),
            (("Yellow", 0, 1), ("Red", 2, 1)),   (("Yellow", 2, 1), ("Orange", 2, 1)),
            (("Green", 1, 2), ("Red", 1, 0)),   (("Green", 1, 0), ("Orange", 1, 2)),
            (("Blue", 1, 0), ("Red", 1, 2)),    (("Blue", 1, 2), ("Orange", 1, 0))
        ]

        for (f1, r1, c1), (f2, r2, c2) in edge_coords:
            v1, v2 = self.faces[f1][r1][c1], self.faces[f2][r2][c2]
            if v1 in VALID_COLORS and v2 in VALID_COLORS:
                logged_edges.append({v1, v2})

        return logged_edges

    def is_physically_allowed(self, face_color, r, c, candidate_color):
        if candidate_color not in VALID_COLORS:
            return True

        neighbors = self.get_piece_colors(face_color, r, c, exclude_self=True)

        if candidate_color in neighbors:
            return False

        for n in neighbors:
            if OPPOSITES.get(n) == candidate_color:
                return False

        proposed_piece = set(neighbors + [candidate_color])

        if (r, c) in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if len(proposed_piece) == 2:
                if proposed_piece not in VALID_EDGES:
                    return False
                existing_edges = self.get_existing_logged_edges()
                if proposed_piece in existing_edges:
                    return False

        return True

    def auto_fill_deductions(self):
        """Iteratively fills remaining Unknown slots using strict physical and quota elimination."""
        changed = True
        
        # Keep resolving deductions as long as new slots get filled
        while changed:
            changed = False
            counts = self.get_color_counts()
            unknowns = self.get_unknown_locations()

            if not unknowns:
                break

            # Quota elimination: colors that already have 9 logged cannot be used anywhere else
            available_colors = [c for c, count in counts.items() if count < 9]

            for face, r, c in unknowns:
                # Find all colors that can physically fit this exact position
                valid_candidates = []
                for candidate in available_colors:
                    if self.is_physically_allowed(face, r, c, candidate):
                        valid_candidates.append(candidate)

                # If exactly ONE color fits, auto-fill it!
                if len(valid_candidates) == 1:
                    self.faces[face][r][c] = valid_candidates[0]
                    changed = True
                    break  # Restart loop to refresh counts and update dependencies


# --- STATE TRACKING VARIABLES ---
CUBE_3D = CubeModel3D()
INITIAL_FACE_LOGGED = False
LAST_STABLE_COLOR = None
CURRENT_CENTER = None
HUD_LAST_MOTION_MSG = "SEARCHING FOR FIRST FACE..."
face_last_logged_time = {col: 0.0 for col in VALID_COLORS}

PREV_GRAY_SMALL = None
TRACKED_POINTS = None
OPTICAL_FLOW_VECTOR = (0.0, 0.0)
CENTROID_VECTOR = (0.0, 0.0)
PREV_CENTROID = None
CURRENT_CENTROID = None
DETECTED_MOTION_DIR = "NONE"
CURRENT_MOTION_MAGNITUDE = 0.0
SMOOTH_FLOW = np.array([0.0, 0.0])

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
FEATURE_PARAMS = dict(
    maxCorners=40, qualityLevel=0.3, minDistance=7, blockSize=7
)

AUTO_PROCESS_INTERVAL = 0.15
last_auto_process_time = 0.0


def reset_cube_tracker():
    global CUBE_3D, INITIAL_FACE_LOGGED, LAST_STABLE_COLOR, CURRENT_CENTER
    global HUD_LAST_MOTION_MSG, PREV_GRAY_SMALL, TRACKED_POINTS
    global OPTICAL_FLOW_VECTOR, CENTROID_VECTOR, PREV_CENTROID, CURRENT_CENTROID
    global DETECTED_MOTION_DIR, SMOOTH_FLOW, CURRENT_MOTION_MAGNITUDE
    global face_last_logged_time

    CUBE_3D.reset()
    INITIAL_FACE_LOGGED = False
    LAST_STABLE_COLOR = None
    CURRENT_CENTER = None
    HUD_LAST_MOTION_MSG = "SYSTEM RESET - SHOW FIRST FACE"
    face_last_logged_time = {col: 0.0 for col in VALID_COLORS}

    PREV_GRAY_SMALL = None
    TRACKED_POINTS = None
    OPTICAL_FLOW_VECTOR = (0.0, 0.0)
    CENTROID_VECTOR = (0.0, 0.0)
    PREV_CENTROID = None
    CURRENT_CENTROID = None
    DETECTED_MOTION_DIR = "NONE"
    CURRENT_MOTION_MAGNITUDE = 0.0
    SMOOTH_FLOW = np.array([0.0, 0.0])

    broadcast_cube_state()
    print("\n--- CUBE TRACKER RESET COMPLETED ---\n")


def rotate_matrix_90_cw(matrix):
    return [list(row) for row in zip(*matrix[::-1])]

def rotate_matrix_90_ccw(matrix):
    return [list(row) for row in zip(*matrix)][::-1]

def flip_matrix_horizontal(matrix):
    return [row[::-1] for row in matrix]

def flip_matrix_vertical(matrix):
    return matrix[::-1]


def calculate_centroid_and_flow(frame, quad_mask):
    global PREV_GRAY_SMALL, TRACKED_POINTS, OPTICAL_FLOW_VECTOR, DETECTED_MOTION_DIR
    global SMOOTH_FLOW, PREV_CENTROID, CURRENT_CENTROID, CENTROID_VECTOR, CURRENT_MOTION_MAGNITUDE

    h, w = frame.shape[:2]
    scale = 0.25
    small_w, small_h = int(w * scale), int(h * scale)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (small_w, small_h))

    if quad_mask is not None and np.any(quad_mask > 0):
        M = cv2.moments(quad_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            CURRENT_CENTROID = (cx, cy)

            if PREV_CENTROID is not None:
                dx = cx - PREV_CENTROID[0]
                dy = cy - PREV_CENTROID[1]
                CENTROID_VECTOR = (float(dx), float(dy))
            PREV_CENTROID = CURRENT_CENTROID
    else:
        CURRENT_CENTROID = None
        PREV_CENTROID = None
        CENTROID_VECTOR = (0.0, 0.0)

    if PREV_GRAY_SMALL is None:
        PREV_GRAY_SMALL = gray_small
        return

    small_mask = None
    if quad_mask is not None:
        small_mask = cv2.resize(
            quad_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        small_mask = cv2.erode(small_mask, kernel)

    if TRACKED_POINTS is None or len(TRACKED_POINTS) < 8:
        if small_mask is not None and np.any(small_mask > 0):
            pts = cv2.goodFeaturesToTrack(
                gray_small, mask=small_mask, **FEATURE_PARAMS
            )
            if pts is not None:
                TRACKED_POINTS = pts

    raw_dx, raw_dy = 0.0, 0.0
    if TRACKED_POINTS is not None and len(TRACKED_POINTS) > 0:
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            PREV_GRAY_SMALL, gray_small, TRACKED_POINTS, None, **LK_PARAMS
        )
        if new_pts is not None and status is not None:
            good_new = new_pts[status == 1]
            good_old = TRACKED_POINTS[status == 1]
            if len(good_new) > 0:
                movements = (good_new - good_old) / scale
                raw_dx = np.median(movements[:, 0])
                raw_dy = np.median(movements[:, 1])
                TRACKED_POINTS = good_new.reshape(-1, 1, 2)

    PREV_GRAY_SMALL = gray_small
    raw_vec = np.array([raw_dx, raw_dy])
    SMOOTH_FLOW = 0.3 * raw_vec + 0.7 * SMOOTH_FLOW
    OPTICAL_FLOW_VECTOR = (float(SMOOTH_FLOW[0]), float(SMOOTH_FLOW[1]))

    comb_dx = OPTICAL_FLOW_VECTOR[0] * 0.6 + CENTROID_VECTOR[0] * 0.4
    comb_dy = OPTICAL_FLOW_VECTOR[1] * 0.6 + CENTROID_VECTOR[1] * 0.4
    CURRENT_MOTION_MAGNITUDE = float(np.sqrt(comb_dx**2 + comb_dy**2))

    if CURRENT_MOTION_MAGNITUDE > MOTION_STILL_THRESHOLD:
        if abs(comb_dx) > abs(comb_dy):
            DETECTED_MOTION_DIR = "RIGHT" if comb_dx > 0 else "LEFT"
        else:
            DETECTED_MOTION_DIR = "BOTTOM" if comb_dy > 0 else "TOP"
    else:
        DETECTED_MOTION_DIR = "NONE"


def draw_motion_overlay(display_frame):
    if CURRENT_CENTROID is not None:
        cv2.circle(display_frame, CURRENT_CENTROID, 6, (0, 0, 255), -1)

        v_dx = int(OPTICAL_FLOW_VECTOR[0] * 3 + CENTROID_VECTOR[0] * 0.5)
        v_dy = int(OPTICAL_FLOW_VECTOR[1] * 3 + CENTROID_VECTOR[1] * 0.5)
        end_pt = (CURRENT_CENTROID[0] + v_dx, CURRENT_CENTROID[1] + v_dy)

        if abs(v_dx) > 2 or abs(v_dy) > 2:
            color = (0, 255, 0) if CURRENT_MOTION_MAGNITUDE >= MIN_MOTION_MAGNITUDE else (0, 255, 255)
            cv2.arrowedLine(
                display_frame,
                CURRENT_CENTROID,
                end_pt,
                color,
                3,
                tipLength=0.3,
            )

        cv2.putText(
            display_frame,
            f"Motion: {DETECTED_MOTION_DIR} ({CURRENT_MOTION_MAGNITUDE:.1f})",
            (CURRENT_CENTROID[0] + 10, CURRENT_CENTROID[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )


def orient_grid_3d_spatial(scanned_grid, prev_color, curr_color, motion_dir):
    global HUD_LAST_MOTION_MSG

    if motion_dir == "NONE" or motion_dir not in MOTION_TO_ENTRY_EDGE:
        return scanned_grid

    entry_edge = MOTION_TO_ENTRY_EDGE[motion_dir]

    best_grid = scanned_grid
    best_score = -999
    best_steps = 0

    test_grid = scanned_grid
    for steps in range(4):
        # Flatten top choice candidates for score testing
        flat_test = [[test_grid[r][c][0] for c in range(3)] for r in range(3)]
        score = CUBE_3D.score_orientation(prev_color, flat_test, entry_edge)
        if score > best_score:
            best_score = score
            best_grid = test_grid
            best_steps = steps
        test_grid = rotate_matrix_90_cw(test_grid)

    HUD_LAST_MOTION_MSG = f"ROTATED {best_steps * 90}deg | Entry: {entry_edge} ({prev_color}->{curr_color})"
    return best_grid


def broadcast_cube_state():
    socketio.emit(
        "cube_update",
        {
            "faces": CUBE_3D.faces,
            "active_center": CURRENT_CENTER,
            "status": HUD_LAST_MOTION_MSG,
        },
    )


def log_face_to_3d_model(scanned_candidates_grid):
    """Iterates ranked color candidates and logs highest-confidence physically valid choices."""
    global HUD_LAST_MOTION_MSG
    center_color = scanned_candidates_grid[1][1][0]
    if center_color not in VALID_COLORS:
        return

    for r in range(3):
        for c in range(3):
            if (r, c) == (1, 1):
                continue

            options = scanned_candidates_grid[r][c]
            selected_color = None

            # Fallback loop: find highest-ranked candidate allowed by 3D geometry
            for candidate in options:
                if CUBE_3D.is_physically_allowed(center_color, r, c, candidate):
                    selected_color = candidate
                    break

            if selected_color:
                CUBE_3D.faces[center_color][r][c] = selected_color

    CUBE_3D.auto_fill_deductions()
    broadcast_cube_state()


def get_ranked_color_candidates(hsv_patch):
    """Returns a list of candidate colors sorted by pixel match coverage percentage."""
    total_pixels = hsv_patch.shape[0] * hsv_patch.shape[1]
    if total_pixels == 0:
        return ["Unknown"]

    candidates = []
    for col_name, ranges in COLOR_RANGES_HSV.items():
        mask = np.zeros(hsv_patch.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_patch, np.array(lower), np.array(upper)))

        match_pct = cv2.countNonZero(mask) / total_pixels
        if match_pct > 0.05:  # Ignore weak noise
            candidates.append((col_name, match_pct))

    candidates.sort(key=lambda x: x[1], reverse=True)
    ranked_colors = [c[0] for c in candidates]
    return ranked_colors if ranked_colors else ["Unknown"]


def sample_colors_inside_quad(photo_img, quad_pts, warp_size=240):
    global LAST_STABLE_COLOR, CURRENT_CENTER, INITIAL_FACE_LOGGED, HUD_LAST_MOTION_MSG
    global face_last_logged_time

    dst_pts = np.float32([[0, 0], [warp_size, 0], [warp_size, warp_size], [0, warp_size]])
    M = cv2.getPerspectiveTransform(quad_pts.astype(np.float32), dst_pts)
    flat_face = cv2.warpPerspective(photo_img, M, (warp_size, warp_size))
    hsv_face = cv2.cvtColor(flat_face, cv2.COLOR_BGR2HSV)

    cell_size = warp_size // 3
    margin = int(cell_size * 0.35)

    candidates_grid = [[[] for _ in range(3)] for _ in range(3)]

    for r in range(3):
        for c in range(3):
            x_start, y_start = c * cell_size, r * cell_size
            hsv_patch = hsv_face[
                y_start + margin : y_start + cell_size - margin,
                x_start + margin : x_start + cell_size - margin,
            ]
            candidates_grid[r][c] = get_ranked_color_candidates(hsv_patch)

    candidates_grid = flip_matrix_horizontal(candidates_grid)
    detected_center = candidates_grid[1][1][0] if candidates_grid[1][1] else "Unknown"

    if detected_center not in VALID_COLORS:
        return

    # Spatial orientation corrections
    if detected_center in ["Red", "Orange"]:
        candidates_grid = flip_matrix_horizontal(candidates_grid)
    elif detected_center == "Yellow":
        candidates_grid = rotate_matrix_90_cw(candidates_grid)

    now = time.time()

    if not INITIAL_FACE_LOGGED:
        INITIAL_FACE_LOGGED = True
        LAST_STABLE_COLOR = detected_center
        CURRENT_CENTER = detected_center
        HUD_LAST_MOTION_MSG = f"START FACE LOCKED: {detected_center}"
        face_last_logged_time[detected_center] = now
        log_face_to_3d_model(candidates_grid)
        return

    if detected_center == LAST_STABLE_COLOR:
        if (now - face_last_logged_time[detected_center]) < RESCAN_COOLDOWN:
            return

    if LAST_STABLE_COLOR is not None and detected_center != LAST_STABLE_COLOR:
        if CURRENT_MOTION_MAGNITUDE < MIN_MOTION_MAGNITUDE:
            HUD_LAST_MOTION_MSG = f"MOTION TOO SMALL ({CURRENT_MOTION_MAGNITUDE:.1f}/{MIN_MOTION_MAGNITUDE}) - SWIPE DEEPER"
            return

        candidates_grid = orient_grid_3d_spatial(
            candidates_grid,
            LAST_STABLE_COLOR,
            detected_center,
            DETECTED_MOTION_DIR,
        )
        LAST_STABLE_COLOR = detected_center

    CURRENT_CENTER = detected_center
    face_last_logged_time[detected_center] = now
    log_face_to_3d_model(candidates_grid)


def tracking_loop():
    global last_auto_process_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        display_frame = frame.copy()

        results = model(frame, conf=0.40, verbose=False)[0]
        quad_corners = None
        mask_binary = None

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

        calculate_centroid_and_flow(frame, mask_binary)
        draw_motion_overlay(display_frame)

        if quad_corners is not None:
            curr_t = time.time()
            if (curr_t - last_auto_process_time) > AUTO_PROCESS_INTERVAL:
                last_auto_process_time = curr_t
                sample_colors_inside_quad(frame, quad_corners)

        cv2.putText(
            display_frame,
            f"Active Face: {CURRENT_CENTER}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Status: {HUD_LAST_MOTION_MSG}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display_frame,
            "Press 'r' to Reset | Press 'q' to Quit",
            (20, h_frame - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imshow("OpenCV Tracker", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            reset_cube_tracker()

    cap.release()
    cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("request_initial_state")
def handle_initial_state():
    broadcast_cube_state()


if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=lambda: socketio.run(
            app, host="0.0.0.0", port=5001, debug=False, use_reloader=False
        )
    )
    flask_thread.daemon = True
    flask_thread.start()

    print("\n" + "=" * 50)
    print(" Open Web Browser to: http://localhost:5001")
    print(" Press 'r' on the OpenCV window to reset tracking state")
    print("=" * 50 + "\n")

    tracking_loop()