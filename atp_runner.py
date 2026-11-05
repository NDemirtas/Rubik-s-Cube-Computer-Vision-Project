#!/usr/bin/env python3
"""
atp_runner.py  -  Acceptance Test Procedures (ATP-01 ... ATP-09) with results
===========================================================================

Runs your ATPs as guided protocols, measures what can be measured, applies
pass/fail thresholds (edit CRITERIA below) and writes:

  atp_results/<session>/results.csv     one row per test run (every number)
  atp_results/<session>/results.json    the same, with all details
  atp_results/<session>/atp_table.tex   LaTeX rows: ATP | condition | measured | result

The camera, face finder, colour reading and move decoder are the V63 code
(through speed_lab.py, which must sit in the same folder). Default settings
are the optimised speed-lab ones ("recommended"); --preset "V63 now" tests the
old V63 settings, and --set name=value changes single switches.

  python atp_runner.py --camera 0                 # top-down camera, like V63
  python atp_runner.py --camera 0 --cam-setup front
  python atp_runner.py --camera 1 --session webcam_tests
  python atp_runner.py --batch-replay atp_results/<session>/capture_<cube>   # re-run the analysis

BATCH (record each cube ONCE, every simulated condition replayed automatically)
  N  type the cube name   ->   B   ->  follow the 4 prompts (~1 minute):
  calibration scan, 4 s without the cube, the 6 solved faces, one algorithm.
  Conditions (lighting / camera / distance & angle / background clutter) are in
  atp_conditions.json (created on first run) - edit, add or switch rows off.

Keys
----
  1-9      choose ATP-01 ... ATP-09           N   name the condition (brand / light / camera ...)
  SPACE    start / stop the current step      S   skip this step
  K        colour calibration scan (6 faces)  A   cycle the test algorithm
  B        BATCH capture for the named cube   (records once, replays every condition)
  E        type a scramble (ATP-07)           I   +1 developer intervention (ATP-08/09)
  Y / X    tick / cross a checklist item      R   reset the tracker to solved
  Q        quit (results are saved after every step)
"""

import argparse
import collections
import csv
import datetime
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import speed_lab as SL
except ImportError:
    raise SystemExit("atp_runner.py needs speed_lab.py in the same folder")

L = SL.L

# =============================================================================
# Pass criteria - edit to match your requirements
# =============================================================================
CRITERIA = {
    "min_width": 1280, "min_height": 720,       # ATP-01 / 05: at least 720p
    # SP-02: frame rate of the tracked feed (what V63 shows), not the raw camera
    "min_fps_nocube": 27.0,                     # no cube in view: 90 % of the 30 fps camera limit
    "min_fps_tracking": 15.0,                   # cube in view: >= 6 frames per move at 2.5 TPS
    "min_detect": 0.90,                         # cube found in >= 90 % of frames with the cube in view
    "max_false_detect": 0.10,                   # face "found" in <= 10 % of frames with NO cube (ATP-03)
    "min_colour": 0.90,                         # facelets read correctly on solved faces
    "min_moves": 0.90,                          # moves of the test algorithm tracked correctly
    "max_solve_s": 5.0,                         # ATP-07: solution computed within this many seconds
    "max_interventions": 2,                     # ATP-08 / 09: developer interventions allowed
}

FEED_SECONDS = 20
NOCUBE_SECONDS = 10
FACE_SECONDS = 2.0

# =============================================================================
# The ATPs as protocols (steps run in order for every condition you name)
# =============================================================================
ATPS = collections.OrderedDict([
    ("ATP-01", {"title": "Live video feed and performance", "cond": "setup",
                "steps": ["feed:nocube", "feed:cube", "check:feed"]}),
    ("ATP-02", {"title": "Different cube brands / colour shades", "cond": "cube brand",
                "steps": ["calibrate", "faces", "alg"]}),
    ("ATP-03", {"title": "Cluttered backgrounds", "cond": "background",
                "steps": ["nocube", "faces", "alg"]}),
    ("ATP-04", {"title": "Lighting conditions", "cond": "lighting",
                "steps": ["faces", "alg"]}),
    ("ATP-05", {"title": "Different cameras", "cond": "camera",
                "steps": ["feed:cube", "faces", "alg"]}),
    ("ATP-06", {"title": "Distances and angles", "cond": "distance / angle",
                "steps": ["faces", "alg"]}),
    ("ATP-07", {"title": "Predetermined scrambles", "cond": "scramble",
                "steps": ["scramble"]}),
    ("ATP-08", {"title": "New user: interface and notation", "cond": "participant",
                "steps": ["check:atp08"]}),
    ("ATP-09", {"title": "New user: learn and solve with the system", "cond": "participant",
                "steps": ["check:atp09"]}),
])

CHECKLISTS = {
    "feed": ["Live feed window opens without errors",
             "Cube outline drawn on the face",
             "Read colours / cube state shown",
             "Detected moves listed as they happen"],
    "atp08": ["Starts the program and reaches the main screen unaided",
              "Finds and uses calibration / scan unaided",
              "Reads the on-screen move notation correctly (5 moves in a row)",
              "Performs the shown moves on the cube correctly",
              "Navigates between modes / menus unaided"],
    "atp09": ["Follows the guide prompts step by step",
              "Learns at least one technique (e.g. a trigger such as R U R' U')",
              "Recovers from a mistake with the system's help",
              "Solves the cube with the system's assistance"],
}

SCRAMBLES = [
    "R U F' L2 D B' R2 U' F",
    "F2 L' U2 R D' B L U2 R'",
    "U' R2 F B' L D2 R' U F2",
]


# =============================================================================
# Camera + tracking in a background thread (like V63: the feed never waits)
# =============================================================================
class Tracker:
    def __init__(self, cam, pipe):
        self.cam, self.pipe = cam, pipe
        self.lock = threading.Lock()
        self.listener = None          # fn(frame, pipe, now) called after every tracked frame
        self.alive = True
        self.paused = False
        self.last_n = 0
        self.proc_times = collections.deque(maxlen=90)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            frame, t, n = self.cam.latest()
            if self.paused or frame is None or n == self.last_n:
                time.sleep(0.002)
                continue
            self.last_n = n
            now = time.perf_counter()
            with self.lock:
                ran = self.pipe.process(frame, now)
                if ran:
                    self.proc_times.append(now)
                    if self.listener is not None:
                        try:
                            self.listener(frame, self.pipe, now)
                        except Exception as exc:  # never kill the tracker over a recording bug
                            print("listener error:", exc)

    def tracked_fps(self):
        p = self.proc_times
        return (len(p) - 1) / (p[-1] - p[0]) if len(p) > 2 else 0.0


# =============================================================================
# Kociemba (optional) for ATP-07
# =============================================================================
def solve_state(state_internal, cam):
    try:
        import kociemba
    except ImportError:
        return None, 0.0, "kociemba not installed (pip install kociemba)"
    s = "".join("URFDLB"[int(v)] for v in state_internal)   # CubeModel uses kociemba's facelet order
    t0 = time.perf_counter()
    try:
        sol = kociemba.solve(s)
    except Exception as exc:
        return None, (time.perf_counter() - t0), f"solver error: {exc}"
    dt = time.perf_counter() - t0
    moves = cam.to_user(sol.split()) if sol.strip() else []
    return moves, dt, ""


# =============================================================================
# Session / results
# =============================================================================
class Session:
    def __init__(self, name, folder=None):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.dir = folder or os.path.join(HERE, "atp_results", name or stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.path_json = os.path.join(self.dir, "results.json")
        self.rows = []
        if os.path.exists(self.path_json):
            try:
                self.rows = json.load(open(self.path_json))
            except Exception:
                self.rows = []

    def add(self, row):
        row["time"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.rows.append(row)
        self.save()

    def save(self):
        with open(self.path_json, "w") as fh:
            json.dump(self.rows, fh, indent=1, default=float)
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys and not isinstance(r[k], (list, dict)):
                    keys.append(k)
        with open(os.path.join(self.dir, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
        self.write_latex()

    def write_latex(self):
        """One row per ATP + condition: what was measured and PASS / FAIL."""
        groups = collections.OrderedDict()
        for r in self.rows:
            groups.setdefault((r["atp"], r["condition"]), []).append(r)
        lines = ["% ATP | condition | measured | result  (generated by atp_runner.py)"]
        for (atp, cond), rs in groups.items():
            measured = "; ".join(r["summary"] for r in rs if r.get("summary"))
            ok = all(r.get("pass", False) for r in rs)
            lines.append(f"\\textbf{{{atp}}} & {tex(cond)} & {tex(measured)} & "
                         f"\\textbf{{{'PASS' if ok else 'FAIL'}}} \\\\ \\hline")
        with open(os.path.join(self.dir, "atp_table.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def tex(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("'", "\\textquotesingle{}"), (">=", "$\\geq$"), ("<=", "$\\leq$")):
        s = s.replace(a, b)
    return s


def pct(x):
    return f"{100 * x:.0f}%" if x == x else "n/a"


# =============================================================================
# BATCH MODE: record each cube once, replay it under every simulated condition
# =============================================================================
CONDITIONS_FILE = os.path.join(HERE, "atp_conditions.json")
DEFAULT_CONDITIONS = {
    "_help": ("Edit freely. atp = which ATP the row counts for. Settings: temperature (+warm/-cool), "
              "brightness (+-), contrast (x), gain (x, <1 darker), noise (sigma), glare (0-1), "
              "resolution (image height), jpeg (quality 5-95), fps (camera rate), scale (<1 further, >1 closer), "
              "tilt (degrees), clutter (number of everyday objects: shaded, textured, irregular, muted colours), "
              "harsh (true = flat shapes in exact cube colours instead), decoys (true = also cube-like 3x3 squares). "
              "on=false skips a row."),
    "recalibrate_lighting": True,
    "conditions": [
        {"atp": "ATP-02", "name": "as recorded", "settings": {}},
        {"atp": "ATP-04", "name": "warm light", "settings": {"temperature": 45}},
        {"atp": "ATP-04", "name": "cool light", "settings": {"temperature": -45}},
        {"atp": "ATP-04", "name": "bright / sunlit", "settings": {"brightness": 50, "contrast": 1.25, "temperature": 12}},
        {"atp": "ATP-04", "name": "dim", "settings": {"gain": 0.5, "noise": 5}},
        {"atp": "ATP-04", "name": "dark + noisy", "settings": {"gain": 0.3, "noise": 12}},
        {"atp": "ATP-04", "name": "glare", "settings": {"glare": 0.7}},
        {"atp": "ATP-05", "name": "480p webcam", "settings": {"resolution": 480}},
        {"atp": "ATP-05", "name": "360p compressed", "settings": {"resolution": 360, "jpeg": 30}},
        {"atp": "ATP-05", "name": "noisy sensor", "settings": {"noise": 10}},
        {"atp": "ATP-05", "name": "15 fps camera", "settings": {"fps": 15}},
        {"atp": "ATP-06", "name": "far (60%)", "settings": {"scale": 0.6}},
        {"atp": "ATP-06", "name": "very far (45%)", "settings": {"scale": 0.45}},
        {"atp": "ATP-06", "name": "close (140%)", "settings": {"scale": 1.4}},
        {"atp": "ATP-06", "name": "tilted 20 deg", "settings": {"tilt": 20}},
        {"atp": "ATP-06", "name": "tilted 35 deg", "settings": {"tilt": 35}},
        {"atp": "ATP-03", "name": "low clutter", "settings": {"clutter": 8}},
        {"atp": "ATP-03", "name": "medium clutter", "settings": {"clutter": 20}},
        {"atp": "ATP-03", "name": "high clutter", "settings": {"clutter": 45}},
        {"atp": "ATP-03", "name": "cube-coloured shapes (worst case)", "settings": {"clutter": 20, "harsh": True}, "on": False},
        {"atp": "ATP-03", "name": "cube-like decoys (worst case)", "settings": {"clutter": 20, "decoys": True}, "on": False},
    ],
}
LIGHTING_KEYS = ("temperature", "brightness", "contrast", "gain")
REPLAYS = 3          # algorithm replays per condition (median reported)
TAIL_S = 1.0         # seconds of the last frame added after the recording ends
NOCUBE_REC_S, FACE_REC_S = 4.0, 1.5


def load_conditions():
    if not os.path.exists(CONDITIONS_FILE):
        with open(CONDITIONS_FILE, "w") as fh:
            json.dump(DEFAULT_CONDITIONS, fh, indent=1)
    try:
        cfg = json.load(open(CONDITIONS_FILE))
        cfg.setdefault("recalibrate_lighting", True)
        return cfg
    except Exception as exc:
        print("atp_conditions.json unreadable, using defaults:", exc)
        return DEFAULT_CONDITIONS


class RecordingCamera(SL.LiveCamera):
    """LiveCamera that can also keep every frame (JPEG) with its time and a label."""

    def __init__(self, *a, **k):
        self.rec_buf, self.rec_label = None, ""
        super().__init__(*a, **k)

    def _run(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            with self.lock:
                self.frame, self.t = f, now
                self.n += 1
                self.times.append(now)
                buf, label = self.rec_buf, self.rec_label
            if buf is not None:
                ok2, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok2:
                    buf.append((enc.tobytes(), now, label))


# ---------------------------------------------------------------- conditions
class Condition:
    """Applies one condition's settings to frames (deterministic per frame)."""

    def __init__(self, settings, seed=5):
        self.s = dict(settings)
        self.seed = seed
        self._clutter = None

    # Everyday-object colours (BGR): wood, paper, fabric, plastic, metal - muted, not cube colours
    EVERYDAY = [(60, 90, 130), (90, 120, 160), (170, 190, 205), (200, 205, 210), (70, 70, 75), (110, 110, 105),
                (40, 45, 50), (120, 95, 70), (95, 110, 80), (150, 140, 120), (85, 70, 110), (60, 80, 60),
                (180, 170, 150), (100, 130, 140)]
    CUBE_COLS = [(245, 245, 245), (30, 215, 240), (60, 170, 40), (190, 90, 20), (40, 40, 210), (20, 130, 250)]

    def _make_clutter(self, w, h):
        """Build the clutter once as a colour layer + soft alpha. Default: realistic
        objects (irregular, shaded, textured, soft-edged, everyday colours, with
        shadows). harsh=true: flat shapes in exact cube colours; decoys=true also
        adds cube-like 3x3 squares (both are worst cases, not real rooms)."""
        rng = np.random.default_rng(self.seed)
        n = int(self.s.get("clutter", 0))
        harsh = bool(self.s.get("harsh") or self.s.get("decoys"))
        col = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros((h, w), np.float32)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]

        def put(mask_u8, colour, shade=True, texture=True, soft=2.5, shadow=True):
            m = mask_u8.astype(np.float32) / 255.0
            if shadow:      # soft drop shadow first
                sh = cv2.GaussianBlur(np.roll(np.roll(m, int(0.012 * w), 1), int(0.012 * w), 0), (0, 0), 0.01 * w)
                col[:] = col * (1 - 0.35 * sh[..., None]) + 0.0
                np.maximum(alpha, 0.35 * sh, out=alpha)
            c = np.array(colour, np.float32)[None, None, :] * np.ones((h, w, 1), np.float32)
            if shade:       # light from above: brighter top, darker bottom
                c = c * (1.15 - 0.35 * yy[..., None])
            if texture:
                c = c + rng.normal(0, 7, (h, w, 1)).astype(np.float32)
            if soft:
                m = cv2.GaussianBlur(m, (0, 0), soft)
            col[:] = col * (1 - m[..., None]) + c * m[..., None]
            np.maximum(alpha, m, out=alpha)

        for _ in range(n):
            x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
            sz = float(rng.uniform(0.05, 0.18) * w)
            mask = np.zeros((h, w), np.uint8)
            if harsh:
                kind = rng.choice(["rect", "circle", "grid", "stripes"] if self.s.get("decoys") else
                                  ["rect", "circle", "stripes"])
                c = self.CUBE_COLS[int(rng.integers(6))]
                if kind == "grid":
                    cell = int(sz / 3)
                    for i in range(9):
                        r_, k_ = divmod(i, 3)
                        m2 = np.zeros((h, w), np.uint8)
                        cv2.rectangle(m2, (int(x) + k_ * cell + 3, int(y) + r_ * cell + 3),
                                      (int(x) + (k_ + 1) * cell - 3, int(y) + (r_ + 1) * cell - 3), 255, -1)
                        put(m2, self.CUBE_COLS[int(rng.integers(6))], False, False, 0, False)
                    continue
                if kind == "rect":
                    cv2.fillPoly(mask, [cv2.boxPoints(((x, y), (sz, sz * 0.6), float(rng.uniform(0, 90)))).astype(np.int32)], 255)
                elif kind == "circle":
                    cv2.circle(mask, (int(x), int(y)), int(sz / 2), 255, -1)
                else:
                    for i in range(5):
                        cv2.rectangle(mask, (int(x), int(y + i * sz / 5)), (int(x + 2 * sz), int(y + i * sz / 5 + sz / 10)), 255, -1)
                put(mask, c, False, False, 0, False)
                continue
            kind = rng.choice(["blob", "ellipse", "book", "cable"], p=[0.4, 0.25, 0.2, 0.15])
            colour = self.EVERYDAY[int(rng.integers(len(self.EVERYDAY)))]
            if kind == "blob":          # irregular object outline (bag, cloth, toy...)
                k = int(rng.integers(7, 13))
                ang = np.sort(rng.uniform(0, 2 * np.pi, k))
                rad = sz * rng.uniform(0.45, 1.0, k) / 2
                pts = np.stack([x + rad * np.cos(ang), y + rad * np.sin(ang)], 1).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
            elif kind == "ellipse":     # mug, plate, bottle from above
                cv2.ellipse(mask, (int(x), int(y)), (int(sz / 2), int(sz * rng.uniform(0.25, 0.5))),
                            float(rng.uniform(0, 180)), 0, 360, 255, -1)
            elif kind == "book":        # long thin object - never square
                box = cv2.boxPoints(((x, y), (sz * 1.8, sz * rng.uniform(0.25, 0.45)), float(rng.uniform(0, 180))))
                cv2.fillPoly(mask, [box.astype(np.int32)], 255)
            else:                       # cable / strap
                pts = [(x, y)]
                a = rng.uniform(0, 2 * np.pi)
                for _ in range(6):
                    a += rng.normal(0, 0.6)
                    pts.append((pts[-1][0] + 0.4 * sz * np.cos(a), pts[-1][1] + 0.4 * sz * np.sin(a)))
                cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, max(2, int(0.008 * w)))
            put(mask, colour)
        self._clutter = (w, h, np.clip(col, 0, 255), np.clip(alpha, 0, 1))

    def _draw_clutter(self, img, quad):
        h, w = img.shape[:2]
        if self._clutter is None or self._clutter[:2] != (w, h):
            self._make_clutter(w, h)
        col, a = self._clutter[2], self._clutter[3][..., None]
        layer = (img.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)
        if quad is not None:   # keep the cube (and a margin for the hand) untouched
            keep = np.zeros((h, w), np.uint8)
            c = quad.mean(0)
            cv2.fillPoly(keep, [((quad - c) * 1.6 + c).astype(np.int32)], 255)
            layer[keep > 0] = img[keep > 0]
        return layer

    def apply(self, frame, quad=None, idx=0):
        s = self.s
        img = frame
        h, w = img.shape[:2]
        if s.get("clutter"):
            img = self._draw_clutter(img, quad)
        if s.get("glare"):
            img = img.copy()
            g = float(s["glare"])
            c = quad.mean(0) if quad is not None else np.array([w / 2, h / 2])
            size = (np.linalg.norm(quad[0] - quad[1]) if quad is not None else 0.25 * w)
            overlay = np.zeros((h, w), np.float32)
            for k, (ox, oy) in enumerate(((-0.18, -0.15), (0.22, 0.12))):
                px = int(c[0] + (ox + 0.08 * math.sin(0.07 * idx + k)) * size)
                py = int(c[1] + (oy + 0.08 * math.cos(0.05 * idx + k)) * size)
                cv2.ellipse(overlay, (px, py), (int(0.22 * size), int(0.13 * size)), 25, 0, 360, 1.0, -1)
            overlay = cv2.GaussianBlur(overlay, (0, 0), max(1.0, 0.06 * size)) * g
            img = np.clip(img.astype(np.float32) + overlay[..., None] * 255, 0, 255).astype(np.uint8)
        if s.get("scale") and float(s["scale"]) != 1.0:
            sc = float(s["scale"])
            if sc < 1:
                small = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                bg = cv2.GaussianBlur(img, (0, 0), 12)
                y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
                bg[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
                img = bg
            else:
                c = quad.mean(0) if quad is not None else np.array([w / 2, h / 2])
                cw, ch = int(w / sc), int(h / sc)
                x0 = int(np.clip(c[0] - cw / 2, 0, w - cw))
                y0 = int(np.clip(c[1] - ch / 2, 0, h - ch))
                img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        if s.get("tilt"):
            t = math.radians(float(s["tilt"]))
            squeeze = 0.5 * w * (1 - math.cos(t)) * 0.9
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([[squeeze, h * (1 - math.cos(t)) * 0.5], [w - squeeze, h * (1 - math.cos(t)) * 0.5],
                              [w, h], [0, h]])
            img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)
        if any(k in s for k in ("temperature", "brightness", "contrast", "gain")):
            f = img.astype(np.float32)
            t = float(s.get("temperature", 0))
            f[..., 2] *= 1 + t / 200.0
            f[..., 0] *= 1 - t / 200.0
            f = (f - 128.0) * float(s.get("contrast", 1.0)) + 128.0
            f = f * float(s.get("gain", 1.0)) + float(s.get("brightness", 0))
            img = np.clip(f, 0, 255).astype(np.uint8)
        if s.get("noise"):
            rng = np.random.default_rng(idx + 17)
            img = np.clip(img.astype(np.float32) + rng.normal(0, float(s["noise"]), img.shape), 0, 255).astype(np.uint8)
        if s.get("resolution"):
            rh = int(s["resolution"])
            if rh < h:
                img = cv2.resize(img, (int(w * rh / h), rh), interpolation=cv2.INTER_AREA)
        if s.get("jpeg"):
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(s["jpeg"])])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return img

    def scale_quad(self, quad, frame_shape, out_shape):
        """The baseline cube outline moved into this condition's image (for calibration)."""
        if quad is None:
            return None
        s = self.s
        h, w = frame_shape[:2]
        q = np.asarray(quad, np.float32).copy()
        if s.get("scale") and float(s["scale"]) != 1.0:
            sc = float(s["scale"])
            if sc < 1:
                q = q * sc + np.float32([(w - int(w * sc)) // 2, (h - int(h * sc)) // 2])
            else:
                c = quad.mean(0)
                cw, ch = int(w / sc), int(h / sc)
                x0 = int(np.clip(c[0] - cw / 2, 0, w - cw))
                y0 = int(np.clip(c[1] - ch / 2, 0, h - ch))
                q = (q - np.float32([x0, y0])) * np.float32([w / cw, h / ch])
        if s.get("tilt"):
            t = math.radians(float(s["tilt"]))
            squeeze = 0.5 * w * (1 - math.cos(t)) * 0.9
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([[squeeze, h * (1 - math.cos(t)) * 0.5], [w - squeeze, h * (1 - math.cos(t)) * 0.5],
                              [w, h], [0, h]])
            q = cv2.perspectiveTransform(q.reshape(-1, 1, 2), cv2.getPerspectiveTransform(src, dst)).reshape(4, 2)
        oh, ow = out_shape[:2]
        return q * np.float32([ow / w, oh / h])


# ---------------------------------------------------------------- capture
class Capture:
    """A recorded cube: segments 'nocube', 'face:<Colour>', 'alg' with JPEG frames."""

    def __init__(self, folder):
        self.folder = folder
        self.meta = json.load(open(os.path.join(folder, "capture.json")))
        self.frames = self.meta["frames"]          # [{file, t, label}]

    @staticmethod
    def save(folder, buf, meta):
        os.makedirs(folder, exist_ok=True)
        t0 = buf[0][1] if buf else 0.0
        frames = []
        for i, (jpg, t, label) in enumerate(buf):
            name = f"f{i:05d}.jpg"
            with open(os.path.join(folder, name), "wb") as fh:
                fh.write(jpg)
            frames.append({"file": name, "t": round(t - t0, 4), "label": label})
        meta = dict(meta, frames=frames)
        with open(os.path.join(folder, "capture.json"), "w") as fh:
            json.dump(meta, fh, indent=1)

    def image(self, i):
        return cv2.imread(os.path.join(self.folder, self.frames[i]["file"]))

    def indices(self, label):
        return [i for i, f in enumerate(self.frames) if f["label"] == label]

    def face_labels(self):
        out = []
        for f in self.frames:
            if f["label"].startswith("face:") and f["label"] not in out:
                out.append(f["label"])
        return out


def _c5_from_json(d):
    c5 = L.CalibratedSoft()
    c5.centres = {k: np.array(v, float) for k, v in d["centres"].items()}
    c5.sigma, c5.glare_L = float(d.get("sigma", 12.0)), float(d.get("glare_L", 240.0))
    c5.reason = ""
    return c5


def _patch_means(img, quad):
    flat = SL.warp_face(img, quad)
    out = []
    for r_ in range(3):
        for c_ in range(3):
            p = flat[r_ * L.CELL + L.MARGIN:(r_ + 1) * L.CELL - L.MARGIN, c_ * L.CELL + L.MARGIN:(c_ + 1) * L.CELL - L.MARGIN]
            out.append(L.CalibratedSoft.robust_mean(cv2.cvtColor(p, cv2.COLOR_BGR2LAB), 240.0)[0])
    return out


def analyse_capture(folder, seg, opt, session, criteria=CRITERIA, progress=None):
    """Replay one recorded cube under every enabled condition; add rows to the session."""
    cap = Capture(folder)
    cfg = load_conditions()
    conds = [c for c in cfg["conditions"] if c.get("on", True)]
    cube = cap.meta.get("cube", os.path.basename(folder))
    alg = cap.meta.get("alg", "sexy6")
    expected = SL.ALGS[alg][1].split()
    base_c5 = _c5_from_json(cap.meta["calibration"])
    opt = dict(opt, camera=cap.meta.get("camera_setup", opt.get("camera", "top-down")))
    nocube_i, alg_i = cap.indices("nocube"), cap.indices("alg")
    faces = cap.face_labels()
    total = len(conds) * (len(nocube_i) + sum(len(cap.indices(f)) for f in faces) + REPLAYS * len(alg_i))
    done = [0]

    def tick():
        done[0] += 1
        if progress and done[0] % 10 == 0:
            progress(done[0] / max(1, total))

    # baseline outlines (as recorded) for clutter masks / glare / zoom centre / recalibration
    base_quads = {}
    pipe = SL.Pipeline(opt, seg, base_c5)
    for i in range(len(cap.frames)):
        img = cap.image(i)
        pipe.opt["decoder_hz"] = 0          # outline on every frame
        pipe.process(img, cap.frames[i]["t"])
        base_quads[i] = None if pipe.quad is None else pipe.quad.copy()
    last = None
    for i in range(len(cap.frames)):       # fill gaps with the last known outline (not for 'nocube')
        if cap.frames[i]["label"] == "nocube":
            continue
        if base_quads[i] is None:
            base_quads[i] = last
        else:
            last = base_quads[i]

    # a re-analysis replaces this cube's previous batch rows (so old results do not linger)
    session.rows = [r for r in session.rows if not (r.get("test") == "batch" and r.get("cube") == cube)]
    # a re-analysis replaces this cube's previous batch rows (so old results do not linger)
    session.rows = [r for r in session.rows if not (r.get("test") == "batch" and r.get("cube") == cube)]
    results = []
    for cond in conds:
        C = Condition(cond.get("settings", {}))
        s = C.s
        # colour calibration: the user re-scans when the lighting changes
        c5 = base_c5
        if cfg.get("recalibrate_lighting", True) and any(k in s for k in LIGHTING_KEYS):
            scans = {}
            for fl in faces:
                idx = cap.indices(fl)
                mid = idx[len(idx) // 2]
                src = cap.image(mid)
                img = C.apply(src, base_quads.get(mid), mid)
                q = C.scale_quad(base_quads.get(mid), src.shape, img.shape)
                if q is not None:
                    scans[fl.split(":")[1]] = _patch_means(img, q)
            if len(scans) == 6:
                c5 = L.CalibratedSoft()
                c5.calibrate(scans)
        row_base = {"atp": cond["atp"], "condition": f"{cube}: {cond['name']}" if cond["atp"] != "ATP-02" else cube,
                    "test": "batch", "simulated": cond["atp"] != "ATP-02", "cube": cube, "setting": cond["name"],
                    "preset": opt.get("_preset", ""), "camera_setup": opt["camera"]}
        # 1) no cube: false detections
        pipe = SL.Pipeline(dict(opt, decoder_hz=0), seg, c5)
        n = fd = 0
        for i in nocube_i:
            img = C.apply(cap.image(i), None, i)
            pipe.process(img, cap.frames[i]["t"])
            n += 1
            fd += int(pipe.quad is not None and pipe.last_probs is not None)
            tick()
        false_det = fd / max(1, n)
        # 2) solved faces: detection + colours
        pipe = SL.Pipeline(dict(opt, decoder_hz=0), seg, c5)
        fn = ff = right = 0
        per_face = {}
        for fl in faces:
            colour = fl.split(":")[1]
            a = b = r_ = 0
            for i in cap.indices(fl):
                img = C.apply(cap.image(i), base_quads.get(i), i)
                pipe.process(img, cap.frames[i]["t"])
                a += 1
                if pipe.quad is not None and pipe.last_probs is not None:
                    b += 1
                    r_ += sum(1 for k in pipe.last_probs.argmax(1) if L.COLORS[int(k)] == colour)
                tick()
            fn, ff, right = fn + a, ff + b, right + r_
            per_face[colour] = round(r_ / max(1, 9 * b), 3)
        detect = ff / max(1, fn)
        colour_acc = right / max(1, 9 * ff)
        # 3) algorithm: real-time replay (frames dropped while busy, like live). Replayed
        #    REPLAYS times with the camera clock shifted by a fraction of a frame (which frames
        #    the tracker happens to get differs each time); the median is reported. A 1 s
        #    tail holding the last frame lets the tracker confirm the final move.
        ts = [cap.frames[i]["t"] for i in alg_i]
        t0 = ts[0] if ts else 0.0
        rel = np.array(ts) - t0
        cam_fps = float(s.get("fps") or cap.meta.get("camera_fps") or 30.0)
        runs = []
        for rep in range(REPLAYS):
            off = rep / (REPLAYS * cam_fps)
            pipe = SL.Pipeline(opt, seg, c5)
            found_alg = [0, 0]

            def source_at(t, off=off):
                k = int(np.clip(np.searchsorted(rel, t + off), 0, len(alg_i) - 1))
                i = alg_i[k]
                tick()
                return C.apply(cap.image(i), base_quads.get(i), i)

            orig = pipe.process

            def proc(frame, now, truth_quad=None, orig=orig, pipe=pipe, found_alg=found_alg):
                ran = orig(frame, now)
                if ran:
                    found_alg[0] += 1
                    found_alg[1] += int(pipe.quad is not None)
                return ran
            pipe.process = proc
            if len(alg_i) > 1:
                SL.replay(source_at, float(rel[-1]) + TAIL_S, cam_fps, pipe)
            got_r = pipe.user_moves()
            runs.append((SL.score_moves(got_r, expected), got_r, found_alg[1] / max(1, found_alg[0]),
                         found_alg[0] / max(1e-3, float(rel[-1]) + TAIL_S if len(rel) else 1.0)))
        runs.sort(key=lambda r: r[0])
        moves, got, alg_det, trk_fps = runs[len(runs) // 2]
        alg_tps = len(expected) / max(1e-3, float(rel[-1])) if len(rel) > 1 else 0.0
        row = dict(row_base, false_detect=round(false_det, 3), detect=round(detect, 3), colour=round(colour_acc, 3),
                   per_face=per_face, moves_correct=round(moves, 3), moves_min=round(runs[0][0], 3),
                   moves_max=round(runs[-1][0], 3), alg=alg, alg_tps=round(alg_tps, 2), alg_detect=round(alg_det, 3),
                   tracked=" ".join(got), tracked_fps=round(trk_fps, 1))
        ok = (detect >= criteria["min_detect"] and colour_acc >= criteria["min_colour"]
              and moves >= criteria["min_moves"])
        if cond["atp"] == "ATP-03" or cond["atp"] == "ATP-02":
            ok = ok and false_det <= criteria["max_false_detect"]
        row["pass"] = bool(ok)
        sim = " (simulated)" if row["simulated"] else ""
        row["summary"] = (f"detected {pct(detect)}, colours {pct(colour_acc)}, moves {pct(moves)}"
                          + (f" (range {pct(runs[0][0])}-{pct(runs[-1][0])})" if runs[0][0] != runs[-1][0] else "")
                          + (f", false detections {pct(false_det)}" if cond["atp"] in ("ATP-02", "ATP-03") else "")
                          + sim)
        session.add(row)
        results.append(row)
        print(f"[batch] {cube:18s} {cond['atp']} {cond['name']:18s} det {detect:5.1%} col {colour_acc:5.1%} "
              f"moves {moves:5.1%} [{runs[0][0]:.0%}-{runs[-1][0]:.0%}] false {false_det:5.1%} "
              f"(alg at {alg_tps:.1f} TPS)  {'PASS' if ok else 'FAIL'}", flush=True)
    chart_capture(results, cube, os.path.join(session.dir, f"batch_{safe(cube)}.png"))
    return results


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def chart_capture(rows, cube, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [f"{r['atp'][-2:]} {r['setting']}" for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6), dpi=150)
    for off, key, col, lab in ((-0.3, "detect", "#9AA3B5", "cube found"), (-0.1, "colour", "#2F6DB5", "colours correct"),
                               (0.1, "moves_correct", "#2E8B3E", "moves correct"),
                               (0.3, "false_detect", "#C0392B", "false detections (no cube)")):
        ax.barh(y + off, [100 * r[key] for r in rows], 0.19, color=col, label=lab)
    ax.axvline(100 * CRITERIA["min_colour"], color="#B53A3A", lw=1, ls="--")
    for yi, r in zip(y, rows):
        ax.text(101.5, yi, "PASS" if r["pass"] else "FAIL", va="center", fontsize=8,
                color="#2E8B3E" if r["pass"] else "#B53A3A", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("%")
    ax.set_title(f"{cube}: recorded once, replayed under every condition", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8, ncol=4, bbox_to_anchor=(0, -0.02 - 1.2 / (len(rows) + 4)))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# The runner
# =============================================================================
class Runner:
    def __init__(self, args):
        self.args = args
        opt = dict(SL.DEFAULTS, **SL.PIPE_PRESETS[args.preset])
        opt.update(SL.parse_set(args.set))
        opt["camera"] = args.cam_setup
        seg_path = args.seg or L.first_existing(L.CONFIG["seg_weights"])
        self.seg = SL.Seg(seg_path)
        if not self.seg.ok:
            print(self.seg.reason)
        self.c5 = SL.load_c5(args.calib)
        self.cam = RecordingCamera(args.camera, args.width, args.height)
        if not self.cam.alive:
            raise SystemExit("camera not available")
        self.pipe = SL.Pipeline(opt, self.seg, self.c5, live=True)
        self.tracker = Tracker(self.cam, self.pipe)
        self.session = Session(args.session)
        self.atp = "ATP-01"
        self.cond = ""
        self.step_i = 0
        self.rec = None              # active recording state
        self.text_mode = None        # None / "cond" / "scramble"
        self.text = ""
        self.alg = "sexy6"
        self.scramble = SCRAMBLES[0]
        self.banner = "Choose an ATP (1-9), name the condition (N), then SPACE to start"
        self.scan_left, self.scans = [], {}
        self.display_times = collections.deque(maxlen=120)
        self.check = None            # checklist state
        self.batch = None            # batch capture state (key B)

    # ------------------------------------------------------------- helpers
    @property
    def steps(self):
        return ATPS[self.atp]["steps"]

    @property
    def step(self):
        return self.steps[self.step_i] if self.step_i < len(self.steps) else None

    def next_step(self):
        self.step_i += 1
        self.rec = None
        self.check = None
        if self.step is None:
            self.banner = f"{self.atp} '{self.cond}' done - N for the next condition, or another ATP"

    def base_row(self, kind):
        o = self.pipe.opt
        return {"atp": self.atp, "condition": self.cond or "-", "test": kind,
                "preset": self.args.preset, "camera_setup": o["camera"]}

    def listener(self, fn):
        with self.tracker.lock:
            self.tracker.listener = fn

    # ------------------------------------------------------------- steps
    def start_step(self):
        st = self.step
        if st is None:
            return
        if not self.cond and ATPS[self.atp]["cond"]:
            self.banner = f"Name the {ATPS[self.atp]['cond']} first (press N)"
            return
        if st.startswith("feed"):
            self.start_feed(cube=(st != "feed:nocube"))
        elif st == "nocube":
            self.start_nocube()
        elif st == "faces":
            self.start_faces()
        elif st == "alg":
            self.start_alg()
        elif st == "scramble":
            self.start_scramble()
        elif st == "calibrate":
            self.scan_left, self.scans = list(L.COLORS), {}
            self.banner = f"Calibration: show the {self.scan_left[0].upper()} face, press K"
        elif st.startswith("check:"):
            items = CHECKLISTS[st.split(":")[1]]
            self.check = {"items": items, "marks": [None] * len(items), "i": 0, "t0": time.time(), "interventions": 0}
            self.banner = "Checklist: Y = yes, X = no for each item, I = intervention, SPACE = finish"

    # FEED: resolution, camera fps, display fps, tracked fps
    def start_feed(self, cube=True):
        self.rec = {"kind": "feed", "cube": cube, "t0": time.time(), "disp": [], "cam": [], "trk": []}
        self.banner = (f"Recording the live feed for {FEED_SECONDS} s - " +
                       ("turn the cube normally (tracking)" if cube else "keep the cube OUT of view"))

    def tick_feed(self):
        r = self.rec
        r["cam"].append(self.cam.fps())
        r["trk"].append(self.tracker.tracked_fps())
        if time.time() - r["t0"] >= FEED_SECONDS:
            d = np.diff(np.array(r["disp"])) if len(r["disp"]) > 2 else np.array([1.0])
            inst = 1.0 / np.clip(d, 1e-3, None)
            h, w = self.frame.shape[:2]
            c = CRITERIA
            need = c["min_fps_tracking"] if r["cube"] else c["min_fps_nocube"]
            row = self.base_row("live feed, tracking cube" if r["cube"] else "live feed, no cube")
            row.update(width=w, height=h, camera_fps=round(float(np.median(r["cam"][5:] or [0])), 1),
                       display_fps=round(float(len(r["disp"]) / FEED_SECONDS), 1),
                       display_fps_p5=round(float(np.percentile(inst, 5)), 1),
                       tracked_fps=round(float(np.median(r["trk"][5:] or [0])), 1),
                       required_fps=need)
            row["pass"] = bool(w >= c["min_width"] and h >= c["min_height"] and row["tracked_fps"] >= need)
            row["summary"] = (f"{w}x{h}, {'tracking' if r['cube'] else 'no cube'} "
                              f"{row['tracked_fps']:.0f} fps (need {need:.0f}), camera {row['camera_fps']:.0f} fps")
            self.session.add(row)
            self.banner = ("PASS: " if row["pass"] else "FAIL: ") + row["summary"]
            self.next_step()

    # NO CUBE: false detections
    def start_nocube(self):
        self.rec = {"kind": "nocube", "t0": time.time(), "n": 0, "found": 0}
        self.banner = f"Keep the cube OUT of view for {NOCUBE_SECONDS} s (background only)"

        def on(frame, pipe, now):
            self.rec["n"] += 1
            self.rec["found"] += int(pipe.quad is not None and pipe.last_probs is not None)
        self.listener(on)

    def tick_nocube(self):
        r = self.rec
        if time.time() - r["t0"] >= NOCUBE_SECONDS:
            self.listener(None)
            fd = r["found"] / max(1, r["n"])
            row = self.base_row("no cube in view")
            row.update(frames=r["n"], false_detect=round(fd, 3))
            row["pass"] = fd <= CRITERIA["max_false_detect"]
            row["summary"] = f"false detections {pct(fd)}"
            self.session.add(row)
            self.banner = ("PASS: " if row["pass"] else "FAIL: ") + row["summary"]
            self.next_step()

    # FACES: detection + colour accuracy on the 6 solved faces
    def start_faces(self):
        if not self.c5.ready:
            self.banner = "Calibrate colours first (K)"
            return
        self.rec = {"kind": "faces", "faces": list(L.COLORS), "cur": None, "n": 0, "found": 0, "right": 0,
                    "stickers": 0, "per_face": {}}
        self.banner = f"Show the solved {self.rec['faces'][0].upper()} face, press SPACE (records {FACE_SECONDS:.0f} s)"

    def faces_space(self):
        r = self.rec
        if r["cur"] is not None or not r["faces"]:
            return
        colour = r["faces"][0]
        r["cur"] = {"colour": colour, "t0": time.time(), "n": 0, "found": 0, "right": 0}

        def on(frame, pipe, now):
            cur = r["cur"]
            if cur is None:
                return
            cur["n"] += 1
            if pipe.quad is not None and pipe.last_probs is not None:
                cur["found"] += 1
                labels = [L.COLORS[int(i)] for i in pipe.last_probs.argmax(1)]
                cur["right"] += sum(1 for c in labels if c == cur["colour"])
        self.listener(on)
        self.banner = f"Recording the {colour.upper()} face..."

    def tick_faces(self):
        r = self.rec
        cur = r.get("cur")
        if cur is None or time.time() - cur["t0"] < FACE_SECONDS:
            return
        self.listener(None)
        r["n"] += cur["n"]
        r["found"] += cur["found"]
        r["right"] += cur["right"]
        r["stickers"] += 9 * cur["found"]
        r["per_face"][cur["colour"]] = (cur["right"] / max(1, 9 * cur["found"]), cur["found"] / max(1, cur["n"]))
        r["faces"].pop(0)
        r["cur"] = None
        if r["faces"]:
            self.banner = f"Show the solved {r['faces'][0].upper()} face, press SPACE"
            return
        det = r["found"] / max(1, r["n"])
        col = r["right"] / max(1, r["stickers"])
        row = self.base_row("solved faces")
        row.update(frames=r["n"], detect=round(det, 3), colour=round(col, 3),
                   per_face={k: [round(a, 3), round(b, 3)] for k, (a, b) in r["per_face"].items()},
                   worst_face=min(r["per_face"], key=lambda k: r["per_face"][k][0]))
        row["pass"] = det >= CRITERIA["min_detect"] and col >= CRITERIA["min_colour"]
        row["summary"] = f"detected {pct(det)}, colours {pct(col)} (worst: {row['worst_face']})"
        self.session.add(row)
        self.banner = ("PASS: " if row["pass"] else "FAIL: ") + row["summary"]
        self.next_step()

    # ALG: track a known algorithm from solved
    def start_alg(self):
        with self.tracker.lock:
            self.pipe.dec.reset()
        self.rec = {"kind": "alg", "t0": time.time(), "n": 0, "found": 0}

        def on(frame, pipe, now):
            self.rec["n"] += 1
            self.rec["found"] += int(pipe.quad is not None)
        self.listener(on)
        self.banner = (f"From SOLVED ({self.pipe.cam.hold_hint()}): do {SL.ALGS[self.alg][0]}, "
                       f"then SPACE")

    def stop_alg(self):
        self.listener(None)
        r = self.rec
        dur = time.time() - r["t0"]
        exp = SL.ALGS[self.alg][1].split()
        with self.tracker.lock:
            got = self.pipe.user_moves()
            solved = bool(np.array_equal(self.pipe.dec.best_state(), SL.MODEL.solved))
        acc = SL.score_moves(got, exp)
        det = r["found"] / max(1, r["n"])
        row = self.base_row("algorithm")
        row.update(alg=self.alg, seconds=round(dur, 1), tps=round(len(exp) / max(dur, 1e-3), 2),
                   moves_correct=round(acc, 3), ends_solved=solved, detect=round(det, 3), moves=" ".join(got))
        row["pass"] = acc >= CRITERIA["min_moves"] and det >= CRITERIA["min_detect"]
        row["summary"] = f"{SL.ALGS[self.alg][0]}: moves {pct(acc)} at {row['tps']:.1f} TPS, detected {pct(det)}"
        self.session.add(row)
        self.banner = ("PASS: " if row["pass"] else "FAIL: ") + row["summary"]
        self.next_step()

    # SCRAMBLE (ATP-07): perform a scramble from solved, check the tracked state, solve it
    def start_scramble(self):
        with self.tracker.lock:
            self.pipe.dec.reset()
        self.rec = {"kind": "scramble", "t0": time.time()}
        self.listener(None)
        self.banner = f"From SOLVED, do the scramble:  {self.scramble}   then SPACE   (E = type another)"

    def stop_scramble(self):
        exp = self.scramble.split()
        dur = time.time() - self.rec["t0"]
        t0 = time.perf_counter()
        with self.tracker.lock:
            st = self.pipe.dec.best_state().copy()
            got = self.pipe.user_moves()
            cam = self.pipe.cam
        target = SL.MODEL.apply_seq(SL.MODEL.solved.copy(), cam.to_internal(exp))
        match = bool(np.array_equal(st, target))
        construct = time.perf_counter() - t0
        sol, solve_s, err = solve_state(st, cam)
        row = self.base_row("scramble")
        row.update(scramble=self.scramble, seconds=round(dur, 1), state_correct=match,
                   moves_correct=round(SL.score_moves(got, exp), 3), tracked=" ".join(got),
                   solve_seconds=round(solve_s + construct, 3), solution=" ".join(sol or []), solver_note=err)
        row["pass"] = match and sol is not None and (solve_s + construct) <= CRITERIA["max_solve_s"]
        row["summary"] = (f"state {'correct' if match else 'WRONG'}, solution "
                          f"{len(sol) if sol is not None else '-'} moves in {solve_s + construct:.2f} s")
        self.session.add(row)
        self.banner = ("PASS: " if row["pass"] else "FAIL: ") + row["summary"] + (f"  ({err})" if err else "")
        self.next_step()

    def finish_check(self):
        c = self.check
        marks = c["marks"]
        row = self.base_row("checklist")
        done = sum(1 for m in marks if m)
        row.update(items_passed=done, items=len(marks), minutes=round((time.time() - c["t0"]) / 60, 1),
                   interventions=c["interventions"],
                   checklist={it: m for it, m in zip(c["items"], marks)})
        limit = CRITERIA["max_interventions"] if self.atp in ("ATP-08", "ATP-09") else 99
        row["pass"] = done == len(marks) and c["interventions"] <= limit
        row["summary"] = (f"{done}/{len(marks)} items, {c['interventions']} interventions, "
                          f"{row['minutes']:.0f} min")
        self.session.add(row)
        self.banner = ("PASS: " if row["pass"] else "FAIL: ") + row["summary"]
        self.next_step()

    # ------------------------------------------------------------- batch
    def start_batch(self):
        if self.batch is not None:
            self.cam.rec_buf = None
            self.batch = None
            self.banner = "batch capture cancelled"
            return
        if not self.cond:
            self.banner = "Batch: first press N and type the CUBE name (e.g. MoYu MeiLong), then B"
            return
        self.batch = {"stage": "calib", "buf": [], "faces": list(L.COLORS), "cube": self.cond}
        self.scan_left, self.scans = list(L.COLORS), {}
        self.banner = (f"BATCH '{self.cond}' 1/4 calibration: show the {self.scan_left[0].upper()} face, press K "
                       f"(S = keep the current calibration)")

    def batch_after_calib(self):
        if self.batch and self.batch["stage"] == "calib":
            self.scan_left = []
            self.batch.update(stage="nocube_wait", until=time.time() + 4.0)

    def _rec(self, label):
        b = self.batch
        self.cam.rec_label = label
        self.cam.rec_buf = b["buf"]

    def _stop_rec(self):
        self.cam.rec_buf = None

    def tick_batch(self):
        b = self.batch
        now = time.time()
        st = b["stage"]
        if st == "nocube_wait":
            left = b["until"] - now
            self.banner = f"BATCH 2/4: take the cube OUT of view - recording in {max(0, left):.0f} s"
            if left <= 0:
                self._rec("nocube")
                b.update(stage="nocube", until=now + NOCUBE_REC_S)
        elif st == "nocube":
            self.banner = f"BATCH 2/4: recording the background ({max(0, b['until'] - now):.0f} s)"
            if now >= b["until"]:
                self._stop_rec()
                b["stage"] = "faces"
                self.banner = f"BATCH 3/4: show the solved {b['faces'][0].upper()} face, press SPACE"
        elif st == "face_rec" and now >= b["until"]:
            self._stop_rec()
            b["faces"].pop(0)
            if b["faces"]:
                b["stage"] = "faces"
                self.banner = f"BATCH 3/4: show the solved {b['faces'][0].upper()} face, press SPACE"
            else:
                b["stage"] = "alg_ready"
                self.banner = (f"BATCH 4/4: hold SOLVED ({self.pipe.cam.hold_hint()}). SPACE, do "
                               f"{SL.ALGS[self.alg][0]} at your normal speed, SPACE when done")

    def batch_space(self):
        b = self.batch
        if b["stage"] == "faces":
            self._rec("face:" + b["faces"][0])
            b.update(stage="face_rec", until=time.time() + FACE_REC_S)
            self.banner = f"BATCH 3/4: recording the {b['faces'][0].upper()} face..."
        elif b["stage"] == "alg_ready":
            self._rec("alg")
            b["stage"] = "alg"
            self.banner = f"BATCH 4/4: recording - do {SL.ALGS[self.alg][0]}, SPACE when done"
        elif b["stage"] == "alg":
            self._stop_rec()
            self.finish_batch()

    def finish_batch(self):
        b, self.batch = self.batch, None
        folder = os.path.join(self.session.dir, "capture_" + safe(b["cube"]))
        h, w = self.frame.shape[:2]
        meta = {"cube": b["cube"], "alg": self.alg, "camera_setup": self.pipe.opt["camera"],
                "camera_fps": round(self.cam.fps(), 1), "resolution": [w, h], "preset": self.args.preset,
                "calibration": self.c5.to_json(),
                "recorded": datetime.datetime.now().isoformat(timespec="seconds")}
        self.banner = "saving the recording..."
        self.show_progress("saving the recording", 0.0)
        Capture.save(folder, b["buf"], meta)
        self.analyse(folder)

    def analyse(self, folder):
        self.tracker.paused = True
        time.sleep(0.1)
        try:
            opt = dict(self.pipe.opt, _preset=self.args.preset)
            with self.tracker.lock:
                analyse_capture(folder, self.seg, opt, self.session,
                                progress=lambda f: self.show_progress("replaying every condition", f))
            self.banner = (f"batch done: {os.path.relpath(self.session.dir, HERE)}  (atp_table.tex, "
                           f"batch_*.png) - N for the next cube, then B")
        finally:
            self.tracker.paused = False

    def show_progress(self, text, frac):
        img = np.full((160, 700, 3), 30, np.uint8)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.rectangle(img, (20, 85), (680, 115), (80, 80, 90), 1)
        cv2.rectangle(img, (20, 85), (20 + int(660 * frac), 115), (90, 200, 120), -1)
        cv2.putText(img, f"{100 * frac:.0f}%", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.imshow("ATP runner", img)
        cv2.waitKey(1)

    # ------------------------------------------------------------- UI
    def draw(self):
        frame = self.frame
        view = cv2.flip(frame, 1) if self.pipe.opt["mirror"] else frame.copy()
        h, w = view.shape[:2]
        sc = 720 / h
        view = cv2.resize(view, (int(w * sc), 720))
        q = self.pipe.quad
        if q is not None:
            cv2.polylines(view, [(q * sc).astype(np.int32)], True, (0, 255, 255), 2)
        pr = self.pipe.last_probs
        if pr is not None:
            for i, row in enumerate(pr):
                r_, c_ = divmod(i, 3)
                name = L.COLORS[int(np.argmax(row))]
                cv2.rectangle(view, (16 + c_ * 30, 16 + r_ * 30), (42 + c_ * 30, 42 + r_ * 30), L.DRAW_BGR[name], -1)
        panel = np.full((720, 520, 3), 28, np.uint8)
        y = [30]

        def line(t, col=(225, 225, 225), dy=24, s=0.52, th=1):
            for part in wrap(t, 54 if s < 0.6 else 40):
                cv2.putText(panel, part, (14, y[0]), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)
                y[0] += dy

        line("ACCEPTANCE TESTS", (80, 200, 255), 30, 0.75, 2)
        line(f"{self.atp}: {ATPS[self.atp]['title']}", (255, 255, 255), 26, 0.58, 2)
        line(f"{ATPS[self.atp]['cond']}: {self.cond or '(press N)'}", (200, 200, 200))
        steps = "  ".join(("[" + s + "]") if i == self.step_i else s for i, s in enumerate(self.steps))
        line("steps: " + steps, (170, 170, 170), 24, 0.45)
        line(f"camera {self.cam.fps():4.1f} fps   feed {self.display_fps():4.1f}   tracking "
             f"{self.tracker.tracked_fps():4.1f} fps   {w}x{h}", (170, 170, 170), 26, 0.45)
        line(f"preset {self.args.preset}   camera setup {self.pipe.opt['camera']}   colours "
             f"{'calibrated' if self.c5.ready else 'NOT calibrated (K)'}", (170, 170, 170), 26, 0.45)
        y[0] += 6
        if self.text_mode:
            line(f"type {self.text_mode}: {self.text}_   (Enter = ok)", (0, 220, 255), 28, 0.55, 2)
        if self.scan_left:
            line(f"SCAN: show the {self.scan_left[0].upper()} face, press K", (0, 220, 255), 28, 0.55, 2)
        line(self.banner, (120, 230, 140), 24, 0.52, 2)
        if self.check:
            y[0] += 6
            for i, it in enumerate(self.check["items"]):
                m = self.check["marks"][i]
                mark = "[Y]" if m else ("[X]" if m is False else ("[>]" if i == self.check["i"] else "[ ]"))
                line(f"{mark} {it}", (220, 220, 220) if i == self.check["i"] else (170, 170, 170), 22, 0.45)
            line(f"interventions: {self.check['interventions']}", (220, 180, 120), 24, 0.48)
        y[0] = max(y[0], 560)
        line("moves: " + " ".join(self.pipe.user_moves()[-12:]), (210, 210, 210), 24, 0.45)
        line(f"test alg (A): {SL.ALGS[self.alg][0]}", (170, 170, 170), 22, 0.45)
        line("1-9 ATP  N name  SPACE start/stop  S skip  K scan  B batch (per cube)  E scramble  I intervene  Y/X  Q",
             (140, 140, 150), 20, 0.4)
        line(f"results: {os.path.relpath(self.session.dir, HERE)}", (140, 140, 150), 20, 0.4)
        return np.hstack([view, panel])

    def display_fps(self):
        d = self.display_times
        return (len(d) - 1) / (d[-1] - d[0]) if len(d) > 2 else 0.0

    # ------------------------------------------------------------- keys
    def key(self, k):
        if self.text_mode:
            if k in (13, 10):
                if self.text_mode == "condition":
                    self.cond = self.text.strip()
                    self.step_i = 0
                    self.rec = None
                    self.banner = f"{self.atp} - {self.cond}: SPACE to start '{self.step}'"
                else:
                    if self.text.strip():
                        self.scramble = self.text.strip()
                    self.banner = f"scramble: {self.scramble}"
                self.text_mode = None
            elif k in (8, 127):
                self.text = self.text[:-1]
            elif k == 27:
                self.text_mode = None
            elif 32 <= k < 127:
                self.text += chr(k)
            return True
        c = chr(k).lower() if 0 <= k < 128 else ""
        if c == "q" or k == 27:
            return False
        if c and c in "123456789" and self.rec is None:
            self.atp = list(ATPS)[int(c) - 1]
            self.cond, self.step_i, self.check = "", 0, None
            self.banner = f"{self.atp}: name the {ATPS[self.atp]['cond']} (N), then SPACE"
        elif c == "b":
            self.start_batch()
        elif c == "s" and self.batch is not None and self.batch["stage"] == "calib":
            if self.c5.ready:
                self.batch_after_calib()
            else:
                self.banner = "no calibration yet - scan the 6 faces with K"
        elif c == "n":
            self.text_mode, self.text = "condition", ""
        elif c == "e":
            self.text_mode, self.text = "scramble", ""
        elif c == "a":
            names = list(SL.ALGS)
            self.alg = names[(names.index(self.alg) + 1) % len(names)]
        elif c == "r":
            with self.tracker.lock:
                self.pipe.dec.reset()
            self.banner = "tracker reset to solved"
        elif c == "s":
            self.listener(None)
            self.next_step()
        elif c == "k":
            self.scan_key()
        elif c == "i" and self.check:
            self.check["interventions"] += 1
        elif c in ("y", "x") and self.check:
            i = self.check["i"]
            if i < len(self.check["items"]):
                self.check["marks"][i] = (c == "y")
                self.check["i"] += 1
            if self.check["i"] >= len(self.check["items"]):
                self.finish_check()
        elif k == 32:
            self.space()
        return True

    def space(self):
        if self.batch is not None:
            self.batch_space()
            return
        st = self.step
        if self.rec is None and self.check is None:
            if st == "calibrate":
                self.banner = "Calibration: press K on each face"
                if not self.scan_left:
                    self.scan_left, self.scans = list(L.COLORS), {}
                return
            self.start_step()
            return
        if self.check is not None:
            self.check["marks"] = [m if m is not None else False for m in self.check["marks"]]
            self.finish_check()
        elif self.rec["kind"] == "faces":
            self.faces_space()
        elif self.rec["kind"] == "alg":
            self.stop_alg()
        elif self.rec["kind"] == "scramble":
            self.stop_scramble()

    def scan_key(self):
        if not self.scan_left:
            self.scan_left, self.scans = list(L.COLORS), {}
            self.banner = ""
            return
        q = self.pipe.quad
        if q is None:
            self.banner = "no face found - hold the face steady in view"
            return
        frame = self.frame if not self.pipe.opt["mirror"] else cv2.flip(self.frame, 1)
        flat = SL.warp_face(frame, q)
        means = []
        for r_ in range(3):
            for c_ in range(3):
                p = flat[r_ * L.CELL + L.MARGIN:(r_ + 1) * L.CELL - L.MARGIN,
                         c_ * L.CELL + L.MARGIN:(c_ + 1) * L.CELL - L.MARGIN]
                means.append(L.CalibratedSoft.robust_mean(cv2.cvtColor(p, cv2.COLOR_BGR2LAB), 240.0)[0])
        self.scans[self.scan_left.pop(0)] = means
        if not self.scan_left:
            self.c5.calibrate(self.scans)
            with self.tracker.lock:
                self.pipe.c5, self.pipe.fast = self.c5, SL.FastColour(self.c5)
                self.pipe.dec.reset()
            if self.batch is not None:
                self.batch_after_calib()
                return
            if self.step == "calibrate":
                row = self.base_row("calibration")
                row.update(sigma=round(self.c5.sigma, 1), glare_L=round(self.c5.glare_L, 1))
                row["pass"] = True
                row["summary"] = "calibrated"
                self.session.add(row)
                self.next_step()
            self.banner = "colours calibrated"

    # ------------------------------------------------------------- loop
    def run(self):
        win = "ATP runner"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last = 0
        while True:
            frame, t, n = self.cam.latest()
            if frame is None or n == last:
                if not self.key_poll(1):
                    break
                continue
            last = n
            self.frame = frame
            now = time.perf_counter()
            self.display_times.append(now)
            if self.batch is not None and self.batch["stage"] != "calib":
                self.tick_batch()
            if self.rec is not None:
                if self.rec["kind"] == "feed":
                    self.rec["disp"].append(now)
                    self.tick_feed()
                elif self.rec["kind"] == "nocube":
                    self.tick_nocube()
                elif self.rec["kind"] == "faces":
                    self.tick_faces()
            cv2.imshow(win, self.draw())
            if not self.key_poll(1):
                break
        self.tracker.alive = False
        self.cam.close()
        cv2.destroyAllWindows()
        print("results in", self.session.dir)

    def key_poll(self, ms):
        k = cv2.waitKey(ms)
        if k < 0:
            return True
        return self.key(k & 0xFF)


def wrap(text, width):
    words, out, cur = str(text).split(), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 > width and cur:
            out.append(cur)
            cur = w_
        else:
            cur = (cur + " " + w_).strip()
    if cur:
        out.append(cur)
    return out or [""]


def main():
    ap = argparse.ArgumentParser(description="Acceptance test runner (ATP-01..09)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam-setup", default="top-down", choices=list(SL.CAMERAS))
    ap.add_argument("--preset", default="recommended", choices=list(SL.PIPE_PRESETS),
                    help="tracking settings under test (default: the optimised speed-lab settings; "
                         "'V63 now' = the old V63 settings)")
    ap.add_argument("--set", action="append", metavar="NAME=VALUE",
                    help="override one switch, e.g. --set yolo_every=6 (repeatable)")
    ap.add_argument("--session", default=None, help="results folder name (re-use it to add to a session)")
    ap.add_argument("--seg", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--batch-replay", default=None, metavar="CAPTURE_DIR",
                    help="re-analyse a recorded cube (atp_results/<session>/capture_<cube>) without a camera")
    args = ap.parse_args()
    if args.batch_replay:
        folder = os.path.abspath(args.batch_replay)
        seg = SL.Seg(args.seg or L.first_existing(L.CONFIG["seg_weights"]))
        if not seg.ok:
            print(seg.reason)
        opt = dict(SL.DEFAULTS, **SL.PIPE_PRESETS[args.preset], _preset=args.preset)
        opt.update(SL.parse_set(args.set))
        analyse_capture(folder, seg, opt, Session(None, folder=os.path.dirname(folder)),
                        progress=lambda f: print(f"\r{100 * f:5.1f}%", end="", flush=True))
        print()
        return
    Runner(args).run()


if __name__ == "__main__":
    main()
