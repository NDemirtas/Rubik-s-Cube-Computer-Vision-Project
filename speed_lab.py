#!/usr/bin/env python3
"""
speed_lab.py  -  How fast can the tracker go?  (fast-solver test bench)
======================================================================

A stripped-down copy of the tracking pipeline where every speed idea is a
switch, so each one can be measured on its own. No guide, no solver, no UI
polish - just: camera -> find face -> read 9 colours -> beam decoder -> moves.

Three ways to use it
--------------------
1. SIMULATION (no camera, perfect ground truth, runs anywhere)
     python speed_lab.py --sim                 # full sweep -> speed_runs/sim_<time>/
     python speed_lab.py --sim --quick         # small sweep (~1 min)
   Fake fast solves at 3-14 turns per second, seen by a camera at 30/60/120 fps,
   processed at different rates. Answers: "which switches let us follow which TPS?"

2. LIVE CAMERA (your real pipeline speed + known-algorithm trials)
     python speed_lab.py --camera 0              # starts on the "recommended" preset (P cycles presets)
   Start from a SOLVED cube, white on top, green in front. Pick a test algorithm
   with 1-6, press SPACE, do it as fast as you can, press SPACE again.
   Every test algorithm returns the cube to solved, so the score is exact.

3. VIDEO REPLAY IN "REAL TIME" (best for very fast solving)
     python speed_lab.py --video slowmo.mov --alg sexy6 [--camera-fps 30]
   Record the algorithm on the iPhone in SLOW-MO (120/240 fps). The lab replays it
   as if it were a live camera at --camera-fps, dropping frames whenever the
   pipeline is still busy - exactly what would happen live. Try 30, 60, 120.
     python speed_lab.py --video slowmo.mov --alg sexy6 --sweep   # every config x fps

Switches (keys in live mode, --set name=value on the command line)
-------------------------------------------------------------------
  yolo_every      1  run YOLO on every Nth processed frame (1 = every frame)        [Y]
  track           flow  between YOLO runs: "flow" (optical flow + homography) / "hold"   [T]
  yolo_async      0  run YOLO in a background thread, never wait for it            [A]
  yolo_imgsz      640  YOLO input size (320 is ~3x cheaper)                         [I]
  roi             0  run YOLO only on a crop around the last face                   [O]
  colour          c5  "c5" (V54/V63 code) or "fast" (same maths, vectorised)        [C]
  decoder_hz      11  max decoder rate (11 = V54/V63 cadence, 0 = every frame)      [H]
  obs_weight      frame  "frame" (V63) or "time" (evidence scaled by time between frames) [W]
  doubles         1  allow 2 moves between decoder frames                           [2]
  triples         0  allow 3 moves between decoder frames (fast solving)            [3]
  beam            200  candidates kept                                              [B]
  mirror          0  flip the camera image (front cameras)                          [M]

Everything is written to speed_runs/.
"""

import argparse
import collections
import csv
import datetime
import itertools
import json
import math
import os
import random
import sys
import threading
import time

import cv2
import numpy as np

import glob

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Pieces copied from cube_lab.py so the speed lab runs on its own
# (any cube_lab version, or none at all)
# =============================================================================


COLORS = ["White", "Yellow", "Green", "Blue", "Red", "Orange"]


DRAW_BGR = {"White": (245, 245, 245), "Yellow": (0, 230, 255), "Green": (40, 190, 40),
            "Blue": (220, 90, 20), "Red": (30, 30, 220), "Orange": (0, 140, 255),
            "Unknown": (110, 110, 110)}


WARP = 150


CELL = 50


MARGIN = 10


FACE_ORDER = "URFDLB"


_FACE_VEC = {   # normal, right, down (reading direction seen from outside)
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}


MOVE_SPECS = {
    "R": (0, [1], -1), "L": (0, [-1], 1), "U": (1, [1], -1), "D": (1, [-1], 1),
    "F": (2, [1], -1), "B": (2, [-1], 1),
    "M": (0, [0], 1), "E": (1, [0], 1), "S": (2, [0], -1),
    "x": (0, [-1, 0, 1], -1), "y": (1, [-1, 0, 1], -1), "z": (2, [-1, 0, 1], -1),
}


def _rot90(axis, direction):
    c, s = 0, direction  # cos(±90)=0, sin(±90)=±1
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def first_existing(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def order_quad(pts):
    """Corners in a fixed cyclic order: top-left first, then clockwise on screen.
    Sorting by angle round the centre never puts the same corner in two slots
    (the old x+y / y-x rule did, for faces tilted near 45 degrees)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    cyc = pts[np.argsort(ang)]                 # clockwise on screen (y points down)
    start = int(np.argmin(cyc.sum(axis=1)))
    return np.roll(cyc, -start, axis=0).astype(np.float32)


def select_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def weighted_dist(a, b):
    return float(math.sqrt(((a[0] - b[0]) * 0.3) ** 2 + ((a[1] - b[1]) * 1.3) ** 2 + ((a[2] - b[2]) * 1.3) ** 2))


def _quad_shape_ok(q):
    q = np.asarray(q, np.float32)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(sides) > 8 and max(sides) / max(min(sides), 1e-6) < 3.0


class Classifier:
    key = "C?"
    name = "classifier"
    short = "clf"
    desc = ""
    needs_calibration = False
    available = True
    reason = ""

    @property
    def ready(self):
        return self.available

    def classify(self, patches):
        """-> (labels[9], confidence[9] in 0..1)"""
        raise NotImplementedError


class CalibratedSoft(Classifier):
    """Exact port of V54/V62: robust glare-free patch mean, weighted LAB distance
    to calibrated median centres, soft probabilities faded by glare/far-ness."""
    key, short = "C5", "Calibrated soft"
    name = "C5 Calibrated soft (CURRENT)"
    desc = "median centres + sigma, weighted LAB, glare-aware probabilities"
    needs_calibration = True
    SIGMA_MIN, SIGMA_MAX = 12.0, 24.0

    def __init__(self):
        self.centres = None
        self.sigma = 12.0
        self.glare_L = 240.0
        self.source = ""
        self.reason = "press K to scan, or have colour_calibration.json"

    @property
    def ready(self):
        return self.centres is not None

    def load(self, path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.centres = {k: np.array(v, float) for k, v in data["centres"].items()}
            self.sigma = float(data.get("sigma", 12.0))
            self.glare_L = float(data.get("glare_L", 240.0))
            self.source = os.path.basename(path)
            self.reason = ""
            return True
        except Exception:
            return False

    def calibrate(self, scans):
        self.centres, spreads = {}, []
        for c in COLORS:
            P = np.asarray(scans[c], float)
            centre = np.median(P, axis=0)
            self.centres[c] = centre
            spreads.extend(weighted_dist(p, centre) for p in P)
        self.sigma = float(np.clip(1.5 * np.median(spreads), self.SIGMA_MIN, self.SIGMA_MAX))
        self.glare_L = float(np.clip(self.centres["White"][0] + 14.0, 232.0, 250.0))
        self.source = "lab scan"
        self.reason = ""

    def to_json(self):
        return {"centres": {k: [float(x) for x in v] for k, v in self.centres.items()},
                "sigma": self.sigma, "glare_L": self.glare_L}

    @staticmethod
    def robust_mean(patch, glare_L):
        px = patch.reshape(-1, 3).astype(float)
        L = px[:, 0]
        blurred = cv2.blur(patch, (5, 5)).reshape(-1, 3).astype(float)
        chroma_b = np.hypot(blurred[:, 1] - 128.0, blurred[:, 2] - 128.0)
        glare = (blurred[:, 0] >= glare_L) & (chroma_b <= 18.0)
        good = (~glare) & (L > 25)
        gf, ok = float(glare.mean()), float(good.mean())
        if ok >= 0.25:
            mean = px[good].mean(axis=0)
        else:
            valid = L > 25
            mean = px[valid].mean(axis=0) if valid.any() else px.mean(axis=0)
        return mean, gf, ok

    def probs(self, mean, gf, ok):
        d = np.array([weighted_dist(mean, self.centres[c]) for c in COLORS])
        best = int(np.argmin(d))
        if COLORS[best] == "White":
            wg = float(np.clip((gf - 0.15) / 0.25, 0, 1))
        else:
            wg = float(np.clip((gf - 0.35) / 0.40, 0, 1))
        if ok < 0.25:
            wg = 1.0
        wf = float(np.clip((d[best] / self.sigma - 3.5) / 2.5, 0, 1))
        w = max(wg, wf)
        row = np.full(6, 0.02)
        row[best] = 0.90
        row /= row.sum()
        return (1 - w) * row + w / 6.0

    def classify(self, patches):
        labels, conf = [], []
        for p in patches:
            lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB)
            mean, gf, ok = self.robust_mean(lab, self.glare_L)
            pr = self.probs(mean, gf, ok)
            i = int(np.argmax(pr))
            labels.append(COLORS[i]); conf.append(float(pr[i]))
        return labels, conf


class CubeModel:
    def __init__(self):
        pos = []
        for f in FACE_ORDER:
            n, r, d = (np.array(v) for v in _FACE_VEC[f])
            for i in range(3):
                for j in range(3):
                    pos.append(3 * n + 2 * ((j - 1) * r + (i - 1) * d))
        self.pos = np.array(pos)
        self.index = {tuple(p): k for k, p in enumerate(self.pos)}
        self.perms = {}
        for name, (axis, layers, direction) in MOVE_SPECS.items():
            base = self._perm(axis, layers, direction)
            inv = np.argsort(base)
            self.perms[name] = base
            self.perms[name + "'"] = inv
            self.perms[name + "2"] = base[base]
        self.solved = np.repeat(np.arange(6, dtype=np.uint8), 9)

    def _perm(self, axis, layers, direction):
        R = _rot90(axis, direction)
        perm = np.arange(54)
        for i, p in enumerate(self.pos):
            c = p[axis]
            layer = 0 if c == 0 else (1 if c > 0 else -1)
            if layer in layers:
                perm[self.index[tuple(R @ p)]] = i
        return perm

    def apply(self, state, move):
        return state[self.perms[move]]

    def apply_seq(self, state, moves):
        for m in moves:
            state = state[self.perms[m]]
        return state

    def moveset(self, kind):
        faces = [f + s for f in "URFDLB" for s in ("", "'", "2")]
        if kind == "faces":
            return faces
        return faces + [f + s for f in "MESxyz" for s in ("", "'", "2")]

    def composites(self, moves, depth):
        """All move sequences up to `depth`, deduplicated by resulting permutation."""
        ident = np.arange(54)
        seen = {ident.tobytes(): ((), ident)}
        frontier = [(ident, ())]
        for _ in range(depth):
            nxt = []
            for p, seq in frontier:
                for m in moves:
                    q = p[self.perms[m]]
                    k = q.tobytes()
                    if k not in seen:
                        seen[k] = (seq + (m,), q)
                        nxt.append((q, seq + (m,)))
            frontier = nxt
        vals = list(seen.values())
        P = np.stack([v[1] for v in vals])
        seqs = [v[0] for v in vals]
        lens = np.array([len(s) for s in seqs])
        return P, seqs, lens

    def self_test(self, trials=30):
        rng = random.Random(3)
        for name, p in self.perms.items():
            q = np.arange(54)
            for _ in range(4):
                q = q[self.perms[name]]
            assert np.array_equal(q, np.arange(54)), name
        try:
            import kociemba
        except ImportError:
            return "kociemba not installed - permutation order checks passed"
        faces = self.moveset("faces")
        for _ in range(trials):
            seq = [rng.choice(faces) for _ in range(20)]
            st = self.apply_seq(self.solved.copy(), seq)
            s = "".join(FACE_ORDER[v] for v in st)
            sol = kociemba.solve(s).split()
            assert np.array_equal(self.apply_seq(st, sol), self.solved), "kociemba check failed"
        return f"self-test passed: {len(self.perms)} moves, {trials} kociemba scrambles solved"


_LAB_CONFIG = {
    "seg_weights": ["runs/segment/rubiks_seg_model/weights/best.pt",
                    "runs/segment/*/weights/best.pt", "yolov8n-seg.pt"],
    "calibration_file": "colour_calibration.json",
}


class _LabNamespace:
    pass


L = _LabNamespace()
for _name in ["COLORS", "DRAW_BGR", "WARP", "CELL", "MARGIN", "first_existing", "order_quad",
              "select_device", "weighted_dist", "_quad_shape_ok", "CalibratedSoft", "CubeModel"]:
    setattr(L, _name, globals()[_name])
L.CONFIG = _LAB_CONFIG

OUT_DIR = os.path.join(HERE, "speed_runs")

# =============================================================================
# V63 face finder (copied unchanged from cube_tracker_V63.py): YOLO region +
# facelet contours + 3x3 lattice fit + memory + state lock (D9 / D10 in the lab)
# =============================================================================
def _face_size(q):
    q = np.asarray(q, np.float32)
    return float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]))


_ROLL_IDX = np.array([[(i - k) % 4 for i in range(4)] for k in range(4)])   # np.roll(q, k) == q[_ROLL_IDX[k]]


def _corner_error(q, ref):
    # speed lab: same value as V63's loop over np.roll, computed in one numpy call
    q, ref = np.asarray(q, np.float32), np.asarray(ref, np.float32)
    return float(np.sqrt(((q[_ROLL_IDX] - ref) ** 2).sum(-1)).mean(1).min())


def _corner_errors_many(q, refs):
    """_corner_error(q, r) for every r in refs (m, 4, 2) at once."""
    R = q[_ROLL_IDX]                                                  # (4 rolls, 4, 2)
    d = np.sqrt(((R[None] - refs[:, None]) ** 2).sum(-1)).mean(-1)   # (m, 4)
    return d.min(1)


def _cell_centres(q):
    src = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
    M = cv2.getPerspectiveTransform(src, np.asarray(q, np.float32))
    pts = np.float32([[c + 0.5, r + 0.5] for r in range(3) for c in range(3)]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def _sticker_regions(gray, area_lo, area_hi, edge_lo=25, edge_hi=75, erode=1):
    edges = cv2.Canny(gray, edge_lo, edge_hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    regions = cv2.bitwise_not(edges)
    if erode:
        regions = cv2.erode(regions, np.ones((3, 3), np.uint8), iterations=erode)
    cnts, _ = cv2.findContours(regions, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_lo or a > area_hi:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if not (4 <= len(approx) <= 6):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.6 <= w / float(h) <= 1.4):
            continue
        ha = cv2.contourArea(cv2.convexHull(c))
        if ha <= 0 or a / ha < 0.85:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        out.append((np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], np.float32), math.sqrt(a)))
    out.sort(key=lambda t: -t[1])
    kept = []
    for c in out:     # the same sticker can come back twice (inner/outer edge)
        if all(np.linalg.norm(c[0] - k[0]) > 0.4 * min(c[1], k[1]) for k in kept):
            kept.append(c)
    return kept


def _quad_shape_ok(q):
    q = np.asarray(q, np.float32)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(sides) > 8 and max(sides) / max(min(sides), 1e-6) < 3.0


def _spacing_jump(cells, pts):
    gaps = []
    for (c, r), k in cells.items():
        for dc, dr in ((1, 0), (0, 1)):
            j = cells.get((c + dc, r + dr))
            if j is not None:
                gaps.append(float(np.linalg.norm(pts[j] - pts[k])))
    return max(gaps) / max(min(gaps), 1e-6) if len(gaps) >= 2 else 1.0


def _squareness(q):
    q = np.asarray(q, np.float32)
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    ratio = min(sides) / max(max(sides), 1e-6)
    sines = []
    for i in range(4):
        a, b = q[(i - 1) % 4] - q[i], q[(i + 1) % 4] - q[i]
        sines.append(abs(float(a[0] * b[1] - a[1] * b[0])) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    return ratio * min(sines)


def fit_sticker_lattices(pts, sizes, max_out=8, all_windows=False):
    """Every 3x3 sticker lattice the candidates support (see the lab's D9)."""
    n = len(pts)
    if n < 4:
        return []
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    med = float(np.median(sizes))
    hyps = []
    for i in range(n):
        s = math.sqrt(float(sizes[i]) * med)
        order = [j for j in np.argsort(D[i]) if j != i and 0.85 * s < D[i, j] < 2.2 * s][:6]
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                u, v = pts[order[a]] - pts[i], pts[order[b]] - pts[i]
                lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                if not (0.6 < lu / lv < 1.67) or abs(float(u @ v)) / (lu * lv) > 0.7:
                    continue
                hyps.append((i, u, v))
    found, done = [], []
    corners_g = np.float32([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]])
    for i, u, v in hyps:
        lu = np.linalg.norm(u)
        dup = False
        for ins, du, dv in done:
            if i in ins:
                close = lambda a, b: min(np.linalg.norm(a - b), np.linalg.norm(a + b)) < 0.2 * lu
                if (close(u, du) and close(v, dv)) or (close(u, dv) and close(v, du)):
                    dup = True
                    break
        if dup:
            continue
        A = np.column_stack([u, v])
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        g = (pts - pts[i]) @ Ainv.T
        r = np.round(g).astype(int)
        res = np.linalg.norm(g - r, axis=1)
        ok = (res < 0.28) & (np.abs(r).max(axis=1) <= 2)
        if ok.sum() < 4:
            continue
        # a lattice of DIAGONAL neighbours has real stickers in the middle of its cells
        frac = np.abs(g - np.floor(g) - 0.5)
        mid = int(np.sum((frac.max(axis=1) < 0.2) & (np.abs(g).max(axis=1) <= 2.5)))
        if mid >= max(2, 0.3 * int(ok.sum())):
            continue
        done.append((set(np.nonzero(ok)[0].tolist()), u, v))
        idx = np.nonzero(ok)[0]
        windows = []
        for a0 in (-2, -1, 0):
            for b0 in (-2, -1, 0):
                cells = {}
                for k in idx:
                    c, rr = r[k, 0] - a0, r[k, 1] - b0
                    if 0 <= c <= 2 and 0 <= rr <= 2:
                        key = (int(c), int(rr))
                        if key not in cells or res[k] < res[cells[key]]:
                            cells[key] = int(k)
                if len(cells) >= 4 and len({c for c, _ in cells}) >= 2 and len({rr for _, rr in cells}) >= 2:
                    windows.append((len(cells), cells))
        if not windows:
            continue
        best = max(w[0] for w in windows)
        for cnt, cells in windows:
            if cnt != best and not (all_windows and cnt >= max(4, best - 3)):
                continue
            src = np.float32([[c, rr] for (c, rr) in cells])
            dst = np.float32([pts[k] for k in cells.values()])
            full = len({c for c, _ in cells}) == 3 and len({rr for _, rr in cells}) == 3
            if cnt >= 6 and full:
                H, _ = cv2.findHomography(src, dst, 0)
                if H is None:
                    continue
                quad = cv2.perspectiveTransform(corners_g.reshape(-1, 1, 2), H).reshape(4, 2)
                pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            else:
                X = np.column_stack([src, np.ones(len(src))])
                M, *_ = np.linalg.lstsq(X, dst, rcond=None)
                quad = np.column_stack([corners_g, np.ones(4)]) @ M
                pred = X @ M
            pitch = 0.5 * (np.linalg.norm(u) + np.linalg.norm(v))
            rms = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1)))) / pitch
            quad = order_quad_robust(np.float32(quad))
            if _quad_shape_ok(quad):
                found.append({"quad": quad, "count": cnt, "rms": rms, "jump": _spacing_jump(cells, pts),
                              "extra": cnt != best})
    found.sort(key=lambda f: (-f["count"], f["rms"]))
    kept, kq = [], []
    for f in found:
        sz = _face_size(f["quad"])
        if not kq or bool((_corner_errors_many(f["quad"], np.asarray(kq)) > 0.2 * sz).all()):
            kept.append(f)
            kq.append(f["quad"])
        if len(kept) >= max_out:
            break
    return kept


def order_quad_robust(pts):
    """V63: corners in a fixed cyclic order - top-left first, then clockwise on
    screen. Sorting by angle round the centre never puts the same corner in
    two slots (the old x+y / y-x rule did for faces tilted ~45 degrees, which
    collapsed the grid and read the stickers in the wrong order)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    cyc = pts[np.argsort(ang)]
    start = int(np.argmin(cyc.sum(axis=1)))
    return np.roll(cyc, -start, axis=0).astype(np.float32)


def _rot_grid(g):
    return [g[6], g[3], g[0], g[7], g[4], g[1], g[8], g[5], g[2]]


class HybridFaceFinder:
    W_REGION, W_MEMORY, W_FRONT, W_FIT, W_JUMP = 2.0, 3.0, 2.0, 30.0, 6.0
    W_STATE_SOFT, W_LOCK, W_NEAR = 1.0, 5.0, 1.5
    LOCK_FRACTION = 0.78         # >= 7 of 9 sure stickers match the believed face
    HOLD_FRAMES = 3              # keep the last face this long if YOLO still sees the cube there
    MAX_STATE_READS = 6          # colour reads per frame for the lock (cost cap)

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.reset()

    def reset(self):
        self.prev = None
        self.prev_age = 99
        self.last_id = None
        self.counts = {"lattice": 0, "lock": 0, "held": 0, "yolo": 0, "none": 0}
        self.last_note = ""

    def find(self, frame, mask, yolo_quad, belief, frame_id=None, read_fn=None, area_ref=None):
        """frame: the picture YOLO saw (display coords). mask: YOLO mask (uint8)
        or None. yolo_quad: YOLO's outline or None. belief: None or
        (current_grids, near_grids) as colour-name lists for the lock;
        read_fn(quad) -> (labels, sure) reads a candidate's colours.
        Returns (quad or None, info)."""
        if frame_id is not None and self.last_id is not None and frame_id not in (self.last_id, self.last_id + 1):
            self.prev, self.prev_age = None, 99
        self.last_id = frame_id
        h, w = frame.shape[:2]
        L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
        g = cv2.GaussianBlur(self.clahe.apply(L), (5, 5), 0)
        A_ = float(area_ref) if area_ref else float(w * h)   # speed lab: crops keep full-frame size limits
        cands = _sticker_regions(g, 0.0004 * A_, 0.02 * A_)
        region = None
        if mask is not None:
            k = max(3, int(0.15 * math.sqrt(max(1, cv2.countNonZero(mask))))) | 1
            region = cv2.dilate(mask, np.ones((k, k), np.uint8))
        elif self.prev is not None and self.prev_age <= 5:
            region = np.zeros((h, w), np.uint8)
            c = self.prev.mean(axis=0)
            cv2.fillPoly(region, [((self.prev - c) * 1.8 + c).astype(np.int32)], 255)
        if region is not None:
            cands = [c for c in cands if region[min(h - 1, max(0, int(c[0][1]))), min(w - 1, max(0, int(c[0][0])))] > 0]
        use_state = belief is not None and read_fn is not None
        lattices = fit_sticker_lattices(np.array([c[0] for c in cands], np.float32).reshape(-1, 2),
                                        np.array([c[1] for c in cands], np.float32),
                                        max_out=12 if use_state else 8,
                                        all_windows=use_state) if len(cands) >= 4 else []
        # geometric score first; colour reads only for the best few (cost cap)
        scored = []
        for Lt in lattices:
            q = Lt["quad"]
            sz = _face_size(q)
            cnt = Lt["count"]
            s = (min(cnt, 6) + 0.3 * max(0, cnt - 6) - self.W_FIT * Lt["rms"]
                 - self.W_JUMP * max(0.0, Lt["jump"] - 1.15) + self.W_FRONT * _squareness(q))
            if mask is not None:
                inside = sum(1 for (x, y) in _cell_centres(q)
                             if 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)] > 0)
                s += self.W_REGION * inside / 9.0
            mem = 0.0
            if self.prev is not None and self.prev_age <= 5:
                mem = math.exp(-(_corner_error(q, self.prev) / max(sz, 1)) / 0.15)
                s += self.W_MEMORY * mem
            scored.append([s, Lt, mem, False, 0])
        if use_state and scored:
            current, near = belief
            reads = 0
            for item in sorted(scored, key=lambda t: -t[0]):
                if reads >= self.MAX_STATE_READS:
                    break
                labels, sure = read_fn(item[1]["quad"])
                reads += 1
                n_sure = int(sum(sure))
                if n_sure < 5:
                    continue
                hit_cur = max((sum(1 for a, b, ok in zip(labels, gg, sure) if ok and a == b) for gg in current), default=0)
                hit_near = max((sum(1 for a, b, ok in zip(labels, gg, sure) if ok and a == b) for gg in near), default=0)
                f_cur, f_near = hit_cur / float(n_sure), hit_near / float(n_sure)
                lock = f_cur >= self.LOCK_FRACTION
                item[0] += self.W_STATE_SOFT * max(f_cur, f_near) + (self.W_LOCK if lock else 0.0) \
                    + (self.W_NEAR if (not lock and f_near >= self.LOCK_FRACTION) else 0.0)
                item[3], item[4] = lock, hit_cur
        best, best_s = None, -1e9
        for s, Lt, mem, lock, hits in scored:
            if Lt.get("extra") and not lock:
                continue          # shifted blocks only count when the colours prove they are the face
            enough = Lt["count"] >= 5 or (Lt["count"] == 4 and (mem > 0.5 or lock))
            if enough and s > best_s:
                best, best_s = (Lt, lock, hits), s
        info = {"source": "", "stickers": len(cands), "grids": len(lattices), "lock": False}
        if best is not None:
            q = best[0]["quad"]
            info.update(source="lattice", count=best[0]["count"], lock=best[1])
            self.counts["lattice"] += 1
            if best[1]:
                self.counts["lock"] += 1
        elif self.prev is not None and self.prev_age < self.HOLD_FRAMES and yolo_quad is not None \
                and cv2.pointPolygonTest(np.asarray(yolo_quad, np.float32).reshape(-1, 1, 2),
                                         tuple(float(v) for v in self.prev.mean(axis=0)), False) >= 0:
            # the stickers vanished for a moment (blur, a finger) but the cube is
            # still where it was: keep the face instead of jumping to YOLO's outline
            self.prev_age += 1
            self.counts["held"] += 1
            info.update(source="held")
            self.last_note = "held"
            return self.prev.copy(), info
        elif yolo_quad is not None:
            q = np.asarray(yolo_quad, np.float32)
            info.update(source="yolo")
            self.counts["yolo"] += 1
        else:
            self.prev_age += 1
            self.counts["none"] += 1
            info.update(source="none")
            return None, info
        q = order_quad_robust(np.float32(q))
        if self.prev is not None and self.prev_age <= 1 and info["source"] == "lattice":
            if _corner_error(q, self.prev) < 0.05 * _face_size(q):     # nearly still: steady the grid
                p = min((np.roll(self.prev, k, axis=0) for k in range(4)),
                        key=lambda r_: float(np.linalg.norm(r_ - q, axis=1).sum()))
                q = order_quad_robust(0.5 * q + 0.5 * p)
        if info["source"] == "lattice":
            self.prev, self.prev_age = q.copy(), 0
        else:
            self.prev_age += 1
        self.last_note = info["source"] + (" lock" if info.get("lock") else "")
        return q, info


# =============================================================================
# Switches
# =============================================================================
DEFAULTS = {
    "finder": "hybrid+lock",  # V63 face finder ("hybrid+lock" = V63 default, "hybrid", "yolo" = V62 outline)
    "finder_on": "every",     # run the V63 finder on "every" frame, or only on "yolo" frames (flow between)
    "finder_crop": 1,         # run the finder only on a crop around the cube (same result, much cheaper)
    "lock_reads": 6,          # colour reads per frame for the state lock (V63: 6)
    "yolo_every": 1,
    "track": "flow",
    "yolo_async": 0,
    "yolo_imgsz": 640,
    "roi": 0,
    "colour": "c5",
    "decoder_hz": 11.0,
    "obs_weight": "frame",
    "ref_hz": 11.0,          # evidence of one frame at this rate counts as 1 (obs_weight=time)
    "doubles": 1,
    "triples": 0,
    "beam": 200,
    "extended": 0,           # slices / rotations in the move set (off for speed tests)
    "mirror": 0,
    "camera": "front",      # live/video: "top-down", "top-down (top away)", "front"
    "move_cost": 12.0,
    "prune": 90.0,
    "motion_gate": 0,        # moves cost x3 while the frame-difference detector sees no motion (V53 idea)
    "quiet_factor": 3.0,
}

# Named configurations for the sweeps (each is DEFAULTS + these changes)
PRESETS = collections.OrderedDict([
    ("V63 now",             {}),
    ("every frame",         {"decoder_hz": 0}),
    ("every frame + gate",  {"decoder_hz": 0, "motion_gate": 1}),
    ("every frame + trip",  {"decoder_hz": 0, "triples": 1}),
    ("time-weighted",       {"decoder_hz": 0, "obs_weight": "time", "ref_hz": 20}),
])

# Test algorithms. All of them return a solved cube to solved, so the expected
# move list is known exactly. (Standard notation, white top / green front.)
ALGS = collections.OrderedDict([
    ("sexy6",  ("(R U R' U') x6", "R U R' U' " * 6)),
    ("sune6",  ("Sune x6", "R U R' U R U2 R' " * 6)),
    ("tperm2", ("T-perm x2", "R U R' U' R' F R2 U' R' U' R U R' F' " * 2)),
    ("uperm3", ("U-perm x3", "R U' R U R U R U' R' U' R2 " * 3)),
    ("r2u2",   ("(R2 U2) x6", "R2 U2 " * 6)),
    ("ruf",    ("(R U F) x... mixed 3-face", "R U F R' U' F' " * 4)),
])

# colour name <-> CubeModel face index (solved: U white, R red, F green, D yellow, L orange, B blue)
FACE_OF_COLOUR = {"White": 0, "Red": 1, "Green": 2, "Yellow": 3, "Orange": 4, "Blue": 5}
COLOUR_OF_FACE = {v: k for k, v in FACE_OF_COLOUR.items()}
FRONT = slice(18, 27)

MODEL = L.CubeModel()

# -----------------------------------------------------------------------------
# Camera set-ups. Inside the tracker the camera always looks at "F" with the top
# of the image as "U" and the image's right as "R". A camera set-up says which
# of YOUR faces those really are (same matrices as V63's CAMERA_PRESETS):
# your_vector = M @ tracker_vector.
# -----------------------------------------------------------------------------
CAMERAS = collections.OrderedDict([
    # camera above, looking down at your U face, TOP of the image towards you (V63 "top-down")
    ("top-down",          [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    # camera above, TOP of the image pointing away from you
    ("top-down (top away)", [[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    # camera in front, image not mirrored (use mirror=1 for a selfie-style view)
    ("front",             [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
])
_NORMAL = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1), "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}
_FACE_OF_NORMAL = {v: k for k, v in _NORMAL.items()}
USER_COLOUR = {"U": "White", "F": "Green", "R": "Red", "D": "Yellow", "L": "Orange", "B": "Blue"}


class CameraFrame:
    """Translates between your move names / colours and the tracker's."""

    def __init__(self, name):
        self.name = name
        M = np.array(CAMERAS[name])
        self.int2usr = {f: _FACE_OF_NORMAL[tuple(int(v) for v in M @ np.array(n))] for f, n in _NORMAL.items()}
        self.usr2int = {v: k for k, v in self.int2usr.items()}
        # colour -> tracker face index (CubeModel order U R F D L B)
        self.face_of_colour = {USER_COLOUR[self.int2usr[f]]: "URFDLB".index(f) for f in "URFDLB"}
        self.colour_index = np.array([self.face_of_colour[c] for c in L.COLORS])

    def _map(self, moves, table):
        return [table[m[0]] + m[1:] if m and m[0] in table else m for m in moves]

    def to_internal(self, moves):
        return self._map(moves, self.usr2int)

    def to_user(self, moves):
        return self._map(moves, self.int2usr)

    def hold_hint(self):
        cam_face = self.int2usr["F"]
        top_face = self.int2usr["U"]
        return (f"hold solved: {USER_COLOUR[cam_face].lower()} facing the camera, "
                f"{USER_COLOUR[top_face].lower()} at the top of the image")


def canonical(moves):
    """Merge consecutive turns of the same face into face+quarter-count tokens,
    so R R == R2 == R' R' and the comparison does not care how a double was seen."""
    out = []
    for m in moves:
        f = m[0]
        q = 2 if m.endswith("2") else (3 if m.endswith("'") else 1)
        if out and out[-1][0] == f:
            out[-1] = (f, (out[-1][1] + q) % 4)
            if out[-1][1] == 0:
                out.pop()
        else:
            out.append((f, q))
    return out


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def score_moves(decoded, expected):
    a, b = canonical(decoded), canonical(expected)
    if not b:
        return 1.0
    return max(0.0, 1.0 - edit_distance(a, b) / len(b))


# =============================================================================
# Beam decoder (same scoring as V63 LiveFrameDecoder) with speed switches
# =============================================================================
FACE_Q = ["U", "U'", "R", "R'", "F", "F'", "D", "D'", "L", "L'", "B", "B'"]
EXT_Q = ["M", "M'", "E", "E'", "S", "S'", "x", "x'", "y", "y'", "z", "z'"]


def inverse(m):
    return m[:-1] if m.endswith("'") else m + "'"


class _Node:
    __slots__ = ("moves", "parent")

    def __init__(self, moves, parent):
        self.moves, self.parent = moves, parent


class FastDecoder:
    P_STICKER, P_GARBAGE = 0.9, 0.4

    def __init__(self, opt):
        self.opt = dict(opt)
        moves = list(FACE_Q) + (list(EXT_Q) if self.opt["extended"] else [])
        extra = [0.0 if m in FACE_Q else (14.0 if m[0] == "S" else 3.0) for m in moves]
        self.moves = moves
        self.P = np.stack([MODEL.perms[m] for m in moves])
        mc = float(self.opt["move_cost"])
        self.cost1 = mc + np.array(extra)
        pairs = [(i, j) for i, a in enumerate(moves) for j, b in enumerate(moves)
                 if inverse(a) != b and ((a in FACE_Q and b in FACE_Q) or a == b)]
        self.pairs = pairs
        self.P2 = np.stack([self.P[i][self.P[j]] for i, j in pairs])
        self.cost2 = np.array([2 * mc + 2.0 + extra[i] + extra[j] for i, j in pairs])
        fi = [moves.index(m) for m in FACE_Q]
        trip = [(i, j, k) for i in fi for j in fi for k in fi
                if inverse(moves[i]) != moves[j] and inverse(moves[j]) != moves[k]]
        self.trips = trip
        self.P3 = np.stack([self.P[i][self.P[j]][self.P[k]] for i, j, k in trip])
        self.cost3 = np.full(len(trip), 3 * mc + 5.0)
        self.ll_clean = np.log(1 - self.P_GARBAGE)
        self.ll_garbage = 9 * np.log(1 / 6.0) + np.log(self.P_GARBAGE)
        self.reset()

    def reset(self, state=None):
        st = MODEL.solved.copy() if state is None else np.asarray(state, np.uint8)
        self.states = st[None].copy()
        self.scores = np.zeros(1)
        self.nodes = [None]
        self.first = True
        self.step_ms = collections.deque(maxlen=120)
        self.steps = 0

    def step(self, obs, dt=None, quiet=False):
        """obs: (9, 6) probabilities in CubeModel face order, or None (no reading).
        quiet: the motion detector saw nothing recently (only used with motion_gate)."""
        t0 = time.perf_counter()
        o = self.opt
        cf = float(o["quiet_factor"]) if (o["motion_gate"] and quiet) else 1.0
        n = len(self.states)
        cand = [self.states]
        cost = [np.zeros(n)]
        par = [np.arange(n)]
        code = [np.full(n, -1)]
        if not self.first:
            s1 = self.states[:, self.P].reshape(-1, 54)
            cand.append(s1)
            cost.append(np.tile(self.cost1 * cf, n))
            par.append(np.repeat(np.arange(n), len(self.moves)))
            code.append(np.tile(np.arange(len(self.moves)), n))
            if obs is not None and (o["doubles"] or o["triples"]):
                g = np.argmax(obs, axis=1)
                best = max(int((self.states[:, FRONT] == g).sum(1).max()), int((s1[:, FRONT] == g).sum(1).max()))
                if o["doubles"] and best < 8:
                    s2 = self.states[:, self.P2].reshape(-1, 54)
                    cand.append(s2)
                    cost.append(np.tile(self.cost2 * cf, n))
                    par.append(np.repeat(np.arange(n), len(self.pairs)))
                    code.append(np.tile(np.arange(len(self.pairs)) + 10000, n))
                    best = max(best, int((s2[:, FRONT] == g).sum(1).max()))
                if o["triples"] and best < 8:
                    top = min(n, 12)          # only from the strongest candidates (cost!)
                    s3 = self.states[:top][:, self.P3].reshape(-1, 54)
                    cand.append(s3)
                    cost.append(np.tile(self.cost3 * cf, top))
                    par.append(np.repeat(np.arange(top), len(self.trips)))
                    code.append(np.tile(np.arange(len(self.trips)) + 20000, top))
        self.first = False
        cand = np.concatenate(cand)
        par = np.concatenate(par)
        code = np.concatenate(code)
        sc = self.scores[par] - np.concatenate(cost)
        if obs is not None:
            w = 1.0
            if o["obs_weight"] == "time" and dt is not None:
                w = float(np.clip(dt * o["ref_hz"], 0.05, 1.0))
            lp = np.log(np.clip(obs, 1e-6, 1.0))
            fit = lp[np.arange(9), cand[:, FRONT]].sum(1)
            sc = sc + w * np.logaddexp(self.ll_clean + fit, self.ll_garbage)
        order = np.argsort(-sc, kind="stable")
        cs = np.ascontiguousarray(cand[order])
        _, first = np.unique(cs.view(np.dtype((np.void, 54))).ravel(), return_index=True)
        keep = order[np.sort(first)]
        keep = keep[sc[keep] >= sc[keep[0]] - o["prune"]][: int(o["beam"])]
        nodes = []
        for k in keep:
            parent = self.nodes[par[k]]
            c = int(code[k])
            if c < 0:
                nodes.append(parent)
            elif c >= 20000:
                i, j, l = self.trips[c - 20000]
                nodes.append(_Node((self.moves[i], self.moves[j], self.moves[l]), parent))
            elif c >= 10000:
                i, j = self.pairs[c - 10000]
                nodes.append(_Node((self.moves[i], self.moves[j]), parent))
            else:
                nodes.append(_Node((self.moves[c],), parent))
        self.states = cand[keep]
        self.scores = sc[keep] - sc[keep[0]]
        self.nodes = nodes
        self.steps += 1
        self.step_ms.append((time.perf_counter() - t0) * 1000)

    def best_moves(self):
        out, node = [], self.nodes[0]
        while node is not None:
            out.extend(reversed(node.moves))
            node = node.parent
        return out[::-1]

    def best_state(self):
        return self.states[0]


# =============================================================================
# Fast-solve simulator (no images): what does the camera see at time t?
# =============================================================================
def layer_motion(move):
    """How the turning layer looks from the front camera.
    ('rows'|'cols', indices, +1/-1 direction) / ('rot', None, +1/-1) / ('none', None, 0)."""
    base = move.rstrip("'2")
    inv = -1 if move.endswith("'") else 1
    table = {
        "U": ("rows", [0], -1), "D": ("rows", [2], 1), "E": ("rows", [1], 1), "y": ("rows", [0, 1, 2], -1),
        "R": ("cols", [2], -1), "L": ("cols", [0], 1), "M": ("cols", [1], 1), "x": ("cols", [0, 1, 2], -1),
        "F": ("rot", None, 1), "z": ("rot", None, 1), "B": ("none", None, 0), "S": ("none", None, 0),
    }
    kind, idx, d = table[base]
    return kind, idx, d * inv


def onehot_probs(front, p=0.9):
    pr = np.full((9, 6), (1 - p) / 5)
    pr[np.arange(9), front] = p
    return pr


class SolveTimeline:
    """A solve: still, then moves at `tps` turns/second, then still."""

    def __init__(self, moves, tps, turn_frac=0.65, lead=0.4, tail=0.6):
        self.moves = moves
        self.slot = 1.0 / tps
        self.turn = self.slot * turn_frac
        self.lead, self.tail = lead, tail
        self.states = [MODEL.solved.copy()]
        for m in moves:
            self.states.append(MODEL.apply(self.states[-1], m))
        self.duration = lead + len(moves) * self.slot + tail

    def at(self, t):
        """(state_before, move or None, phase 0..1)"""
        if t < self.lead:
            return self.states[0], None, 0.0
        k = int((t - self.lead) // self.slot)
        if k >= len(self.moves):
            return self.states[-1], None, 0.0
        tin = t - self.lead - k * self.slot
        if tin >= self.turn:
            return self.states[k + 1], None, 0.0
        return self.states[k], self.moves[k], tin / self.turn


def front_mixture(state, move, phase):
    """(9,6) colour mixture seen on the front at one instant."""
    if move is None or phase <= 0:
        return onehot_probs(state[FRONT], 1.0)
    after = MODEL.apply(state, move)
    kind, idx, d = layer_motion(move)
    half = move.endswith("2")
    old = state[FRONT].reshape(3, 3)
    new = after[FRONT].reshape(3, 3)
    mix = np.zeros((3, 3, 6))
    for r in range(3):
        for c in range(3):
            mix[r, c, old[r, c]] = 1.0
    if kind == "none":
        return mix.reshape(9, 6)
    if kind == "rot":
        # whole face turning: stickers are between grid positions
        a = phase if not half else phase
        m = np.zeros((3, 3, 6))
        for r in range(3):
            for c in range(3):
                m[r, c, old[r, c]] += 1 - a
                m[r, c, new[r, c]] += a
        return m.reshape(9, 6)
    mid = MODEL.apply(state, move.rstrip("2")) if half else None
    for line in idx:
        if kind == "rows":
            seq_old, seq_new = old[line, :], new[line, :]
            seq_mid = mid[FRONT].reshape(3, 3)[line, :] if half else None
        else:
            seq_old, seq_new = old[:, line], new[:, line]
            seq_mid = mid[FRONT].reshape(3, 3)[:, line] if half else None
        # tape of cells sliding past the camera in direction d
        if d < 0:
            tape = list(seq_old) + (list(seq_mid) if half else []) + list(seq_new)
        else:
            tape = list(seq_new) + (list(seq_mid) if half else []) + list(seq_old)
        span = (6 if half else 3)
        shift = phase * span
        for p in range(3):
            if d < 0:
                x = p + shift
            else:
                x = p + span - shift
            i0 = int(math.floor(x))
            f = x - i0
            v = np.zeros(6)
            v[tape[min(i0, len(tape) - 1)]] += 1 - f
            v[tape[min(i0 + 1, len(tape) - 1)]] += f
            if kind == "rows":
                mix[line, p] = v
            else:
                mix[p, line] = v
    return mix.reshape(9, 6)


class MotionSignal:
    """What a frame-difference motion detector reports: true while a layer turns
    (and ~0.1 s after), plus false alarms from hands regripping."""

    def __init__(self, tl, rng, false_rate=0.8):
        self.tl = tl
        self.fa = []
        t = 0.0
        while true_rate_ok(false_rate):
            t += rng.exponential(1.0 / false_rate)
            if t >= tl.duration:
                break
            self.fa.append((t, t + rng.uniform(0.1, 0.3)))

    def moving(self, t, window=0.12):
        for k in range(4):
            s = t - window * k / 3
            if s >= 0 and self.tl.at(s)[1] is not None:
                return True
        return any(a <= t < b for a, b in self.fa)


def true_rate_ok(rate):
    return rate > 0


class Episodes:
    """Misreads that LAST (glare on a sticker, a finger over it): the same wrong
    reading for 0.1-0.5 s. These are what make a fast decoder count the same
    error several times - independent per-frame noise is far too kind."""

    def __init__(self, duration, rng, rate=0.6):
        self.ev = []
        t = 0.0
        while True:
            t += rng.exponential(1.0 / rate) if rate > 0 else duration + 1
            if t >= duration:
                break
            length = rng.uniform(0.1, 0.5)
            cells = rng.choice(9, size=int(rng.integers(1, 3)), replace=False)
            kind = rng.choice(["glare", "wrong"], p=[0.5, 0.5])
            colour = int(rng.integers(6))
            self.ev.append((t, t + length, cells, kind, colour))

    def apply(self, t, pr):
        for a, b, cells, kind, colour in self.ev:
            if a <= t < b:
                for i in cells:
                    if kind == "glare":   # washed out: reads white-ish, unsure
                        row = np.full(6, 0.1)
                        row[0] = 0.5
                    else:                 # finger / reflection: a confident wrong colour
                        row = np.full(6, 0.06)
                        row[colour] = 0.7
                    pr[i] = row / row.sum()
        return pr


def simulate_observation(tl, t, shutter, rng, misread=0.02, glare=0.03, episodes=None):
    """Blurred, noisy (9,6) probabilities for a frame exposed over [t, t+shutter], or None
    when the face cannot be found (mid F/z turn: the outline is a diamond)."""
    subs = [t + shutter * (k + 0.5) / 4 for k in range(4)]
    acc = np.zeros((9, 6))
    unreadable = 0
    for s in subs:
        st, mv, ph = tl.at(s)
        if mv is not None and layer_motion(mv)[0] == "rot" and 0.2 < ph < 0.8:
            unreadable += 1
        acc += front_mixture(st, mv, ph)
    if unreadable >= 2 and rng.random() < 0.8:
        return None
    acc /= len(subs)
    # what a calibrated classifier reports for a mixed / blurred sticker
    pr = np.empty((9, 6))
    for i in range(9):
        top = acc[i].max()
        if top >= 0.85:
            row = np.full(6, 0.02)
            row[acc[i].argmax()] = 0.9
        else:  # mixed colours: the reading is the dominant one, with low confidence
            fade = float(np.clip((0.85 - top) / 0.35, 0, 1))
            row = np.full(6, 0.02)
            row[acc[i].argmax()] = 0.9
            row = (1 - fade) * row + fade * (0.5 * acc[i] + 0.5 / 6)
        u = rng.random()
        if u < misread:
            row = np.full(6, 0.02)
            row[rng.integers(6)] = 0.9
        elif u < misread + glare:
            row = np.full(6, 1 / 6.0)
        pr[i] = row / row.sum()
    if episodes is not None:
        pr = episodes.apply(t, pr)
    return pr


SOLVE_WEIGHTS = {"R": 6, "U": 6, "F": 2, "L": 2, "D": 1.5, "B": 0.4}


def random_solve(n, rng):
    faces = list(SOLVE_WEIGHTS)
    w = np.array([SOLVE_WEIGHTS[f] for f in faces], float)
    w /= w.sum()
    out, last = [], None
    while len(out) < n:
        f = faces[rng.choice(len(faces), p=w)]
        if f == last:
            continue
        out.append(f + rng.choice(["", "'", "2"], p=[0.45, 0.4, 0.15]))
        last = f
    return out


def run_sim_trial(moves, tps, cam_fps, opt, rng, proc_ms=None, shutter_max=1 / 60.0, episode_rate=0.6):
    """One simulated solve. proc_ms: pipeline time per frame (face+colour), the
    decoder itself is timed for real. Returns accuracy, state_ok, decoder frames."""
    tl = SolveTimeline(moves, tps)
    eps = Episodes(tl.duration, rng, episode_rate)
    motion = MotionSignal(tl, rng)
    dec = FastDecoder(opt)
    shutter = min(1.0 / cam_fps, shutter_max)
    t_free, last_t = 0.0, None
    period = 1.0 / opt["decoder_hz"] if opt["decoder_hz"] else 0.0
    next_dec = 0.0
    frames = 0
    t = 0.0
    while t < tl.duration:
        if t >= t_free and t >= next_dec:
            obs = simulate_observation(tl, t, shutter, rng, episodes=eps)
            dt = None if last_t is None else t - last_t
            dec.step(None if obs is None else obs, dt, quiet=not motion.moving(t))
            last_t = t
            frames += 1
            busy = (proc_ms or 0.0) / 1000.0 + dec.step_ms[-1] / 1000.0
            t_free = t + busy
            next_dec = t + period
        t += 1.0 / cam_fps
    acc = score_moves(dec.best_moves(), moves)
    ok = bool(np.array_equal(dec.best_state(), tl.states[-1]))
    return acc, ok, frames, float(np.mean(dec.step_ms)) if dec.step_ms else 0.0


def sim_sweep(quick=False, seed=11):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"sim_{stamp}")
    os.makedirs(out, exist_ok=True)
    tps_list = [3, 5, 7, 9, 11, 13] if not quick else [4, 8, 12]
    trials = 8 if not quick else 3
    n_moves = 24 if not quick else 16
    cams = [30, 60, 120] if not quick else [30, 60]
    rows = []
    # pipeline cost outside the decoder (face finding + colours), ms per frame
    proc = {"V63 now": 45.0}  # YOLO every frame
    configs = [(name, cam) for name in PRESETS for cam in cams]
    configs = [c for c in configs if not (c[0] == "V63 now" and c[1] != 30)]
    for name, cam in configs:
        opt = dict(DEFAULTS, **PRESETS[name])
        pm = proc.get(name, 10.0)  # YOLO every few frames + flow + fast colour (measure live!)
        for tps in tps_list:
            rng = np.random.default_rng(seed)
            accs, oks, frs, ms = [], [], [], []
            for _ in range(trials):
                mv = random_solve(n_moves, rng)
                a, ok, fr, dm = run_sim_trial(mv, tps, cam, opt, rng, proc_ms=pm)
                accs.append(a); oks.append(ok); frs.append(fr); ms.append(dm)
            row = {"config": name, "camera_fps": cam, "tps": tps,
                   "move_accuracy": round(100 * float(np.mean(accs)), 1),
                   "state_correct": round(100 * float(np.mean(oks)), 1),
                   "decoder_fps": round(float(np.mean(frs)) / (0.4 + n_moves / tps + 0.6), 1),
                   "decoder_ms": round(float(np.mean(ms)), 2)}
            rows.append(row)
            print(f"{name:20s} cam {cam:3d}  {tps:4.1f} TPS  moves {row['move_accuracy']:5.1f}%  "
                  f"state {row['state_correct']:5.1f}%  decoder {row['decoder_fps']:5.1f} Hz  "
                  f"{row['decoder_ms']:.1f} ms/step", flush=True)
    with open(os.path.join(out, "sim_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    chart_sim(rows, out)
    print("saved", out)
    return rows, out


def chart_sim(rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    series = collections.OrderedDict()
    for r in rows:
        series.setdefault(f"{r['config']} @ {r['camera_fps']} fps", []).append(r)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
    cmap = plt.get_cmap("tab10")
    for i, (k, rs) in enumerate(series.items()):
        ls = "-" if "@ 30" in k else ("--" if "@ 60" in k else ":")
        ax.plot([r["tps"] for r in rs], [r["move_accuracy"] for r in rs], ls, marker="o",
                color=cmap(list(PRESETS).index(k.split(" @")[0])), lw=2, label=k)
    ax.set_xlabel("turns per second")
    ax.set_ylabel("moves tracked correctly (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.axvspan(8, 14, color="#f3e6d6", alpha=0.5, zorder=0)
    ax.text(8.2, 4, "fast solvers", color="#8a5a20")
    ax.legend(fontsize=8, loc="lower left", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sim_accuracy_vs_tps.png"))
    plt.close(fig)


# =============================================================================
# Live / video pipeline
# =============================================================================
STEPS = collections.OrderedDict([      # tweakable values, in order, for the live UI
    ("finder", ["hybrid+lock", "hybrid", "yolo"]),
    ("finder_on", ["every", "yolo"]),
    ("finder_crop", [0, 1]),
    ("lock_reads", [1, 2, 3, 4, 6]),
    ("yolo_every", [1, 2, 3, 4, 5, 6, 8, 10, 15]),
    ("track", ["flow", "hold"]),
    ("yolo_async", [0, 1]),
    ("yolo_imgsz", [224, 256, 320, 416, 480, 640]),
    ("roi", [0, 1]),
    ("colour", ["c5", "fast"]),
    ("decoder_hz", [8, 11, 15, 20, 30, 45, 60, 0]),
    ("obs_weight", ["frame", "time"]),
    ("ref_hz", [11, 15, 20, 30]),
    ("doubles", [0, 1]),
    ("triples", [0, 1]),
    ("motion_gate", [0, 1]),
    ("beam", [50, 100, 150, 200, 300, 400]),
    ("move_cost", [6, 8, 10, 12, 14, 16, 20]),
    ("mirror", [0, 1]),
    ("camera", list(CAMERAS)),
])

# Whole-pipeline configurations for the video sweep
PIPE_PRESETS = collections.OrderedDict([
    ("V63 now",          {}),
    ("V62 outline (old)", {"finder": "yolo"}),
    ("decoder every frame", {"decoder_hz": 0}),
    ("YOLO/3 + flow",    {"decoder_hz": 0, "yolo_every": 3, "colour": "fast"}),
    ("YOLO/3 async",     {"decoder_hz": 0, "yolo_every": 3, "colour": "fast", "yolo_async": 1}),
    ("YOLO/6 320 roi",   {"decoder_hz": 0, "yolo_every": 6, "colour": "fast", "yolo_imgsz": 320, "roi": 1}),
    ("YOLO/3 + gate",    {"decoder_hz": 0, "yolo_every": 3, "colour": "fast", "motion_gate": 1}),
    # best combination from the simulation + synthetic replays (start here live)
    # finder "hybrid" = D9 (no state lock): faster, and better in the live trials
    ("recommended",      {"decoder_hz": 0, "yolo_every": 4, "track": "flow", "yolo_async": 1, "colour": "fast",
                          "finder": "hybrid", "finder_on": "every"}),
])

COLOUR_INDEX = np.array([FACE_OF_COLOUR[c] for c in L.COLORS])   # lab colour order -> face index


def to_face_order(pr_lab_order):
    """(9,6) probabilities in L.COLORS order -> CubeModel face order."""
    out = np.empty_like(pr_lab_order)
    out[:, COLOUR_INDEX] = pr_lab_order
    return out


class Seg:
    """YOLO segmentation with a selectable input size and optional crop."""

    def __init__(self, path):
        self.path = path
        self.model = None
        self.device = "cpu"
        self.reason = ""
        if not path or not os.path.exists(path):
            self.reason = "no YOLO seg weights (use --seg)"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.device = L.select_device()
            for s in (320, 640):
                self.model(np.zeros((480, 640, 3), np.uint8), imgsz=s, verbose=False, device=self.device)
        except Exception as exc:
            self.model = None
            self.reason = f"YOLO: {exc}"[:90]

    @property
    def ok(self):
        return self.model is not None

    def detect(self, frame, imgsz=640, roi_quad=None):
        """-> ((mask or None, quad or None), ms). The mask (frame size) is the
        region the V63 face finder searches for facelets."""
        q, ms, poly = self._run(frame, imgsz, roi_quad)
        mask = None
        if poly is not None:
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        return (mask, q), ms

    def quad(self, frame, imgsz=640, roi_quad=None):
        q, ms, _ = self._run(frame, imgsz, roi_quad)
        return q, ms

    def _run(self, frame, imgsz=640, roi_quad=None):
        """-> (quad or None, ms, polygon or None)"""
        if not self.ok:
            return None, 0.0, None
        t0 = time.perf_counter()
        ox = oy = 0
        img = frame
        if roi_quad is not None:
            h, w = frame.shape[:2]
            x0, y0 = roi_quad.min(0)
            x1, y1 = roi_quad.max(0)
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.9
            ox, oy = int(max(0, cx - half)), int(max(0, cy - half))
            ex, ey = int(min(w, cx + half)), int(min(h, cy + half))
            if ex - ox > 40 and ey - oy > 40:
                img = frame[oy:ey, ox:ex]
            else:
                ox = oy = 0
        try:
            res = self.model(img, imgsz=int(imgsz), conf=0.4, verbose=False, device=self.device)[0]
        except Exception:
            res = self.model(img, imgsz=int(imgsz), conf=0.4, verbose=False)[0]
        q, poly_out = None, None
        if res.masks is not None and len(res.masks) and len(res.masks.xy):
            boxes = res.boxes.xyxy.cpu().numpy()
            i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            poly = np.asarray(res.masks.xy[i], np.float32)
            if len(poly) >= 4:
                poly = poly + np.float32([ox, oy])
                poly_out = poly
                c = poly.reshape(-1, 1, 2)
                peri = cv2.arcLength(c, True)
                ap = cv2.approxPolyDP(c, 0.03 * peri, True)
                q = ap.reshape(-1, 2) if len(ap) == 4 else cv2.boxPoints(cv2.minAreaRect(c))
                q = L.order_quad(q)
        return q, (time.perf_counter() - t0) * 1000, poly_out


class OracleSeg:
    """Stands in for YOLO on synthetic video: returns the true outline, and the
    time YOLO would have taken on your machine (charged to the virtual clock)."""
    fake = True

    def __init__(self, full_ms=45.0):
        self.full_ms = full_ms
        self.truth = None
        self.ok = True

    def quad(self, frame, imgsz=640, roi_quad=None):
        scale = (imgsz / 640.0) ** 2 * (0.5 if roi_quad is not None else 1.0)
        q = None if self.truth is None else self.truth + np.random.normal(0, 1.0, (4, 2)).astype(np.float32)
        return q, self.full_ms * scale

    def detect(self, frame, imgsz=640, roi_quad=None):
        q, ms = self.quad(frame, imgsz, roi_quad)
        mask = None
        if q is not None:
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, (q + (q - q.mean(0)) * 0.04).astype(np.int32), 255)
        return (mask, q), ms


class FlowTracker:
    """Moves the face outline from frame to frame: feature points inside the face,
    Lucas-Kanade optical flow, a homography fitted with RANSAC (so the turning
    layer, which moves differently, is ignored as outliers)."""

    def __init__(self):
        self.pts = None
        self.quad = None
        self.gray = None

    def reset(self, gray, quad):
        self.gray, self.quad = gray, np.asarray(quad, np.float32)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, self.quad.astype(np.int32), 255)
        self.pts = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=6, mask=mask)

    def anchor(self, gray, quad):
        """Adopt a better outline (from the face finder) without paying for new
        feature points every frame; reseed only when the points run low."""
        if self.pts is None or len(self.pts) < 25 or self.gray is None or self.gray.shape != gray.shape:
            self.reset(gray, quad)
        else:
            self.quad = np.asarray(quad, np.float32)

    def update(self, gray):
        if self.quad is None or self.pts is None or len(self.pts) < 8:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.pts, None, winSize=(21, 21), maxLevel=2)
        ok = st.reshape(-1) == 1
        if ok.sum() < 8:
            self.quad = None
            return None
        a, b = self.pts[ok], p1[ok]
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        if H is None or inl.sum() < 8:
            self.quad = None
            return None
        q = cv2.perspectiveTransform(self.quad.reshape(-1, 1, 2), H).reshape(4, 2)
        if not L._quad_shape_ok(q):
            self.quad = None
            return None
        self.quad = q
        keep = b[inl.reshape(-1) == 1]
        self.gray = gray
        if len(keep) < 25:
            self.reset(gray, q)
        else:
            self.pts = keep.reshape(-1, 1, 2)
        return q


class VirtualAsync:
    """Background YOLO on a virtual clock (video replay): result usable once the
    time it really took has passed."""

    def __init__(self):
        self.pending = None   # (ready_time, quad)
        self.busy_until = -1.0

    def submit(self, now, fn):
        if now < self.busy_until:
            return
        q, ms = fn()
        self.busy_until = now + ms / 1000.0
        self.pending = (self.busy_until, q)

    def poll(self, now):
        if self.pending and now >= self.pending[0]:
            q = self.pending[1]
            self.pending = None
            return q, True
        return None, False


class ThreadAsync:
    """Background YOLO in a real thread (live camera)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.result = None
        self.running = False

    def submit(self, now, fn):
        with self.lock:
            if self.running:
                return
            self.running = True

        def work():
            q, _ = fn()
            with self.lock:
                self.result = q
                self.running = False
        threading.Thread(target=work, daemon=True).start()

    def poll(self, now):
        with self.lock:
            if self.result is not None:
                q, self.result = self.result, None
                return q, True
        return None, False


class FastColour:
    """Same maths as C5 (robust glare-free mean, weighted LAB distance, faded
    probabilities) but one colour conversion and numpy for all 9 stickers."""

    def __init__(self, c5):
        self.c5 = c5
        self.C = np.stack([c5.centres[c] for c in L.COLORS])
        self.wt = np.array([0.3, 1.3, 1.3])
        self.white = L.COLORS.index("White")

    def probs(self, flat):
        lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
        blur = cv2.blur(lab, (5, 5))
        cells, cb = [], []
        for r in range(3):
            for c in range(3):
                sl = (slice(r * L.CELL + L.MARGIN, (r + 1) * L.CELL - L.MARGIN),
                      slice(c * L.CELL + L.MARGIN, (c + 1) * L.CELL - L.MARGIN))
                cells.append(lab[sl].reshape(-1, 3))
                cb.append(blur[sl].reshape(-1, 3))
        px = np.stack(cells).astype(np.float32)          # (9, n, 3)
        bl = np.stack(cb).astype(np.float32)
        chroma = np.hypot(bl[..., 1] - 128, bl[..., 2] - 128)
        glare = (bl[..., 0] >= self.c5.glare_L) & (chroma <= 18)
        good = (~glare) & (px[..., 0] > 25)
        gf, okf = glare.mean(1), good.mean(1)
        wgood = good.astype(np.float32)[..., None]
        mean_good = (px * wgood).sum(1) / np.maximum(wgood.sum(1), 1)
        valid = (px[..., 0] > 25).astype(np.float32)[..., None]
        mean_valid = (px * valid).sum(1) / np.maximum(valid.sum(1), 1)
        mean = np.where((okf >= 0.25)[:, None], mean_good, mean_valid)
        d = np.sqrt((((mean[:, None, :] - self.C[None]) * self.wt) ** 2).sum(-1))   # (9,6)
        best = d.argmin(1)
        dbest = d[np.arange(9), best]
        wg = np.where(best == self.white, np.clip((gf - 0.15) / 0.25, 0, 1), np.clip((gf - 0.35) / 0.40, 0, 1))
        wg = np.where(okf < 0.25, 1.0, wg)
        wf = np.clip((dbest / self.c5.sigma - 3.5) / 2.5, 0, 1)
        w = np.maximum(wg, wf)[:, None]
        row = np.full((9, 6), 0.02)
        row[np.arange(9), best] = 0.9
        row /= row.sum(1, keepdims=True)
        return (1 - w) * row + w / 6.0


def c5_probs(c5, flat):
    out = []
    for r in range(3):
        for c in range(3):
            p = flat[r * L.CELL + L.MARGIN:(r + 1) * L.CELL - L.MARGIN,
                     c * L.CELL + L.MARGIN:(c + 1) * L.CELL - L.MARGIN]
            lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB)
            mean, gf, ok = c5.robust_mean(lab, c5.glare_L)
            out.append(c5.probs(mean, gf, ok))
    return np.array(out)


def warp_face(frame, quad):
    dst = np.float32([[0, 0], [L.WARP - 1, 0], [L.WARP - 1, L.WARP - 1], [0, L.WARP - 1]])
    M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
    return cv2.warpPerspective(frame, M, (L.WARP, L.WARP))


class Pipeline:
    """camera frame -> face outline -> 9 colour probabilities -> decoder."""

    def __init__(self, opt, seg, c5, oracle=None, live=False):
        self.opt = dict(opt)
        self.seg, self.c5, self.oracle = seg, c5, oracle
        self.fast = FastColour(c5) if c5 is not None and c5.ready else None
        self.live = live
        self.reset()

    def reset(self):
        self.cam = CameraFrame(self.opt.get("camera", "front"))
        self.dec = FastDecoder(self.opt)
        self.flow = FlowTracker()
        self.finder = HybridFaceFinder()
        self.frame_no = 0
        self.asyncer = ThreadAsync() if self.live else VirtualAsync()
        self.quad = None
        self.since_yolo = 10 ** 6
        self.last_dec_t = None
        self.last_flat = None
        self.last_motion_t = -1.0
        self.ms = collections.defaultdict(lambda: collections.deque(maxlen=90))
        self.counts = collections.Counter()
        self.last_probs = None

    def set(self, key, value):
        self.opt[key] = value
        if key == "camera":
            self.cam = CameraFrame(value)
            self.dec.reset()
        if key in ("beam", "move_cost", "doubles", "triples", "obs_weight", "ref_hz", "motion_gate", "extended",
                   "decoder_hz", "prune", "quiet_factor"):
            self.dec.opt[key] = value
            if key in ("move_cost", "extended"):
                self.dec = FastDecoder(self.opt)

    def _yolo(self, frame):
        roi = self.quad if (self.opt["roi"] and self.quad is not None) else None
        return self.seg.detect(frame, self.opt["yolo_imgsz"], roi)

    def _belief(self):
        """V63 state_lock_belief for this decoder: fronts the tracker expects."""
        d = self.dec
        if len(d.states) == 0:
            return None
        name = {v: k for k, v in self.cam.face_of_colour.items()}
        close = d.states[d.scores >= -20.0][:6]
        children = d.states[0][d.P][:, FRONT]
        current, seen = [], set()
        for st in close:
            g = [name[int(i)] for i in st[FRONT]]
            for _ in range(4):
                if tuple(g) not in seen:
                    seen.add(tuple(g))
                    current.append(list(g))
                g = _rot_grid(g)
        near = []
        for ch in children:
            g = [name[int(i)] for i in ch]
            if tuple(g) not in seen:
                seen.add(tuple(g))
                near.append(g)
        return current, near

    def _read_fn(self, frame):
        def read(quad):
            flat = warp_face(frame, order_quad_robust(np.float32(quad)))
            pr = self.fast.probs(flat) if self.fast is not None else c5_probs(self.c5, flat)
            return [L.COLORS[int(i)] for i in pr.argmax(1)], [bool(v >= 0.5) for v in pr.max(1)]
        return read

    def _find_cropped(self, frame, mask, fallback, belief, read_fn):
        """Run the V63 finder on a crop around the cube (mask / fallback / last
        face, expanded) - identical logic, a fraction of the pixels."""
        h, w = frame.shape[:2]
        box = None
        if self.opt["finder_crop"]:
            pts = []
            if mask is not None:
                ys, xs = np.nonzero(mask[::4, ::4])
                if len(xs):
                    pts.append(np.stack([xs * 4, ys * 4], 1).astype(np.float32))
            for q_ in (fallback, self.finder.prev):
                if q_ is not None:
                    pts.append(np.asarray(q_, np.float32).reshape(-1, 2))
            if pts:
                P_ = np.concatenate(pts)
                (x0, y0), (x1, y1) = P_.min(0), P_.max(0)
                pad = 0.45 * max(x1 - x0, y1 - y0)
                x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
                x1, y1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
                if x1 - x0 > 60 and y1 - y0 > 60 and (x1 - x0) * (y1 - y0) < 0.9 * w * h:
                    box = (x0, y0, x1, y1)
        if box is None:
            self.finder.MAX_STATE_READS = int(self.opt["lock_reads"])
            return self.finder.find(frame, mask, fallback, belief, frame_id=self.frame_no, read_fn=read_fn)
        x0, y0, x1, y1 = box
        off = np.float32([x0, y0])
        sub = frame[y0:y1, x0:x1]
        sub_mask = mask[y0:y1, x0:x1] if mask is not None else None
        sub_fb = None if fallback is None else np.asarray(fallback, np.float32) - off
        if self.finder.prev is not None:
            self.finder.prev = self.finder.prev - off
        sub_read = None if read_fn is None else (lambda q_: read_fn(np.asarray(q_, np.float32) + off))
        self.finder.MAX_STATE_READS = int(self.opt["lock_reads"])
        q, info = self.finder.find(sub, sub_mask, sub_fb, belief, frame_id=self.frame_no,
                                   read_fn=sub_read, area_ref=w * h)
        if self.finder.prev is not None:
            self.finder.prev = self.finder.prev + off
        return (None if q is None else q + off), info

    def find_face(self, frame, gray, now, truth_quad=None):
        """YOLO every `yolo_every` frames (optionally in the background); the
        V63 finder (facelet lattice + memory + state lock) on EVERY frame, using
        the last YOLO mask / the previous face as the search region; optical
        flow only as a fallback outline when the finder sees no grid."""
        o = self.opt
        t0 = time.perf_counter()
        self.frame_no += 1
        if getattr(self.seg, "fake", False):
            self.seg.truth = truth_quad
        if self.seg is None or not self.seg.ok:
            self.quad = truth_quad
            self.ms["face"].append((time.perf_counter() - t0) * 1000)
            return self.quad
        mask = yq = None
        due = self.quad is None or self.since_yolo >= int(o["yolo_every"])
        if o["yolo_async"]:
            res, fresh = self.asyncer.poll(now)
            if fresh and res is not None:
                mask, yq = res
                self.since_yolo = 0
                self.counts["yolo"] += 1
            if due:
                self.asyncer.submit(now, lambda: self._yolo(frame))
            self.since_yolo += 1
        elif due:
            (mask, yq), ms = self._yolo(frame)
            if getattr(self.seg, "fake", False):
                self.fake_ms += ms
            self.ms["yolo"].append(ms)
            self.counts["yolo"] += 1
            self.since_yolo = 1
        else:
            self.since_yolo += 1
        flow_q = None
        if o["track"] == "flow" and self.flow.quad is not None:
            tq = time.perf_counter()
            flow_q = self.flow.update(gray)
            self.ms["flow"].append((time.perf_counter() - tq) * 1000)
        fallback = yq if yq is not None else (flow_q if flow_q is not None else
                                              (self.quad if o["track"] == "hold" else None))
        run_finder = o["finder"] != "yolo" and (o["finder_on"] == "every" or yq is not None or fallback is None)
        if not run_finder:
            q = fallback
            self.finder.last_note = "yolo" if yq is not None else ("flow" if flow_q is not None else "hold")
            if o["finder"] != "yolo":
                self.finder.prev_age += 1
        else:
            th = time.perf_counter()
            lock = o["finder"] == "hybrid+lock" and self.c5 is not None and self.c5.ready
            q, info = self._find_cropped(frame, mask, fallback, self._belief() if lock else None,
                                         self._read_fn(frame) if lock else None)
            self.counts["finder " + (info.get("source") or "none")] += 1
            self.ms["finder"].append((time.perf_counter() - th) * 1000)
        if q is not None:
            if o["track"] == "flow":
                self.flow.anchor(gray, q)
        else:
            self.since_yolo = 10 ** 6          # lost the cube: ask YOLO on the next frame
        self.quad = q
        self.ms["face"].append((time.perf_counter() - t0) * 1000)
        return self.quad

    def process(self, frame, now, truth_quad=None):
        """One camera frame at time `now` (s). Returns True if the decoder ran."""
        o = self.opt
        if o["mirror"]:
            frame = cv2.flip(frame, 1)
        if o["decoder_hz"] and self.last_dec_t is not None and now - self.last_dec_t < 1.0 / o["decoder_hz"] - 1e-4:
            self.counts["skipped (rate cap)"] += 1
            return False
        t_all = time.perf_counter()
        self.fake_ms = 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad = self.find_face(frame, gray, now, truth_quad)
        probs = None
        if quad is not None and self.c5 is not None and self.c5.ready:
            tc = time.perf_counter()
            flat = warp_face(frame, quad)
            if o["colour"] == "fast" and self.fast is not None:
                probs = self.fast.probs(flat)
            else:
                probs = c5_probs(self.c5, flat)
            g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
            if self.last_flat is not None and float(cv2.absdiff(g, self.last_flat).mean()) > 9.0:
                self.last_motion_t = now
            self.last_flat = g
            self.ms["colour"].append((time.perf_counter() - tc) * 1000)
        self.last_probs = probs
        dt = None if self.last_dec_t is None else now - self.last_dec_t
        quiet = (now - self.last_motion_t) > 0.12
        obs = None
        if probs is not None:
            obs = np.empty_like(probs)
            obs[:, self.cam.colour_index] = probs
        self.dec.step(obs, dt, quiet=quiet)
        self.ms["decoder"].append(self.dec.step_ms[-1])
        self.ms["total"].append((time.perf_counter() - t_all) * 1000)
        self.last_dec_t = now
        self.counts["decoded"] += 1
        return True

    def user_moves(self):
        return self.cam.to_user(self.dec.best_moves())

    def avg(self, key):
        v = self.ms.get(key)
        return float(np.mean(v)) if v else 0.0


# =============================================================================
# Sources
# =============================================================================
class LiveCamera:
    """Grabs frames in a thread so the camera never waits for us; we always take
    the newest frame and count how many we could not process."""

    def __init__(self, index, width=1280, height=720, fps=60):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.lock = threading.Lock()
        self.frame, self.t, self.n = None, 0.0, 0
        self.times = collections.deque(maxlen=120)
        self.alive = self.cap.isOpened()
        threading.Thread(target=self._run, daemon=True).start()

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

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.n

    def fps(self):
        with self.lock:
            if len(self.times) < 2:
                return 0.0
            return (len(self.times) - 1) / (self.times[-1] - self.times[0])

    def close(self):
        self.alive = False
        self.cap.release()


class SyntheticVideo:
    """Rendered fast solve with the real move animation, motion blur and hand shake.
    Used to test the pipeline without a camera (outline is known exactly)."""

    PAL = {0: (245, 245, 245), 1: (40, 40, 210), 2: (60, 170, 40), 3: (30, 215, 240),
           4: (20, 130, 250), 5: (190, 90, 20)}   # BGR for faces U R F D L B (W R G Y O B)

    def __init__(self, moves, tps, fps=120, size=(640, 480), shutter=1 / 60.0, seed=3):
        self.tl = SolveTimeline(moves, tps)
        self.fps, self.size, self.shutter = fps, size, shutter
        self.n = int(self.tl.duration * fps)
        rng = np.random.default_rng(seed)
        self.bg = cv2.GaussianBlur(rng.integers(30, 90, (size[1], size[0], 3)).astype(np.uint8), (0, 0), 8)

    def quad_at(self, t):
        w, h = self.size
        cx, cy = w / 2 + 12 * math.sin(1.3 * t), h / 2 + 8 * math.sin(1.7 * t + 1)
        a, s = 0.08 * math.sin(0.9 * t), 230
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        base = np.array([[-1, -1], [1, -1], [1.05, 1], [-1.05, 1]]) * s / 2
        return (base @ R.T + [cx, cy]).astype(np.float32)

    def _texture(self, st, mv, ph):
        T = 300
        tex = np.zeros((T, T, 3), np.uint8)
        cell = T // 3

        def put(r, c, colour_idx, dx=0.0, dy=0.0):
            x0, y0 = int((c + dx) * cell), int((r + dy) * cell)
            cv2.rectangle(tex, (x0 + 6, y0 + 6), (x0 + cell - 6, y0 + cell - 6), self.PAL[int(colour_idx)], -1)

        old = st[FRONT].reshape(3, 3)
        if mv is None:
            for r in range(3):
                for c in range(3):
                    put(r, c, old[r, c])
            return tex
        kind, idx, d = layer_motion(mv)
        after = MODEL.apply(st, mv)[FRONT].reshape(3, 3)
        half = mv.endswith("2")
        if kind == "rot":
            for r in range(3):
                for c in range(3):
                    put(r, c, old[r, c])
            ang = -d * (180 if half else 90) * ph
            M = cv2.getRotationMatrix2D((T / 2, T / 2), ang, 1.0)
            return cv2.warpAffine(tex, M, (T, T))
        mid = MODEL.apply(st, mv.rstrip("2"))[FRONT].reshape(3, 3) if half else None
        for r in range(3):
            for c in range(3):
                moving = (kind == "rows" and r in idx) or (kind == "cols" and c in idx)
                if not moving:
                    put(r, c, old[r, c])
        span = 6 if half else 3
        shift = ph * span
        for line in idx:
            seq_o = old[line, :] if kind == "rows" else old[:, line]
            seq_n = after[line, :] if kind == "rows" else after[:, line]
            seq_m = (mid[line, :] if kind == "rows" else mid[:, line]) if half else []
            tape = (list(seq_o) + list(seq_m) + list(seq_n)) if d < 0 else (list(seq_n) + list(seq_m) + list(seq_o))
            off = -shift if d < 0 else -(span - shift)
            sub = np.zeros((cell, T, 3), np.uint8) if kind == "rows" else np.zeros((T, cell, 3), np.uint8)
            for k, col in enumerate(tape):
                p = (k + off) * cell
                if p > T or p + cell < 0:
                    continue
                if kind == "rows":
                    cv2.rectangle(sub, (int(p) + 6, 6), (int(p) + cell - 6, cell - 6), self.PAL[int(col)], -1)
                else:
                    cv2.rectangle(sub, (6, int(p) + 6), (cell - 6, int(p) + cell - 6), self.PAL[int(col)], -1)
            if kind == "rows":
                tex[line * cell:(line + 1) * cell, :] = sub
            else:
                tex[:, line * cell:(line + 1) * cell] = sub
        return tex

    def render(self, t):
        acc = np.zeros((self.size[1], self.size[0], 3), np.float32)
        subs = 3
        for k in range(subs):
            s = t + self.shutter * (k + 0.5) / subs
            st, mv, ph = self.tl.at(s)
            tex = self._texture(st, mv, ph)
            q = self.quad_at(s)
            src = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
            M = cv2.getPerspectiveTransform(src, q)
            img = self.bg.copy()
            face = cv2.warpPerspective(tex, M, self.size)
            mask = cv2.warpPerspective(np.full((300, 300), 255, np.uint8), M, self.size)
            img[mask > 0] = face[mask > 0]
            acc += img
        return (acc / subs).astype(np.uint8), self.quad_at(t)


def synthetic_c5():
    c5 = L.CalibratedSoft()
    scans = {}
    for name in L.COLORS:
        bgr = SyntheticVideo.PAL[FACE_OF_COLOUR[name]]
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        scans[name] = [lab + np.random.default_rng(i).normal(0, 1.5, 3) for i in range(9)]
    c5.calibrate(scans)
    return c5


class VideoFrames:
    def __init__(self, path):
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if f.shape[1] > 1280:
                f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])))
            self.frames.append(f)
        cap.release()
        self.duration = len(self.frames) / self.fps

    def at(self, t):
        i = min(len(self.frames) - 1, int(round(t * self.fps)))
        return self.frames[i]


# =============================================================================
# Real-time replay: a camera at `cam_fps`, frames dropped while we are busy
# =============================================================================
def replay(source_at, duration, cam_fps, pipe, truth_quad_at=None, yolo_ms_model=None):
    """source_at(t) -> frame. Processing time is measured for real; the next frame
    we get is the newest one the camera has produced by the time we are free."""
    t_free = 0.0
    k = 0
    seen = 0
    while True:
        t = k / cam_fps
        if t >= duration:
            break
        if t + 1e-9 >= t_free:
            frame = source_at(t)
            tq = truth_quad_at(t) if truth_quad_at else None
            t0 = time.perf_counter()
            ran = pipe.process(frame, t, truth_quad=tq)
            busy = time.perf_counter() - t0
            if ran:
                busy += getattr(pipe, "fake_ms", 0.0) / 1000.0
                if truth_quad_at is not None and pipe.quad is not None and tq is not None:
                    pipe.ms["outline_err_px"].append(float(np.abs(order_like(pipe.quad, tq) - tq).max()))
            t_free = t + busy
            seen += int(bool(ran))
            # jump to the newest camera frame available when we are free
            k = max(k + 1, int(math.ceil(t_free * cam_fps - 1e-9)))
        else:
            k += 1
    return seen


def order_like(q, ref):
    """Rotate q's corner order to best match ref (for error measurement)."""
    q = np.asarray(q, np.float32)
    best = min(range(4), key=lambda r: np.abs(np.roll(q, r, 0) - ref).sum())
    return np.roll(q, best, 0)


def yolo_cost_model(full_ms=45.0):
    """For synthetic replays (no YOLO here): charge what YOLO would cost on the
    frames where the pipeline would have run it."""
    def cost(pipe):
        o = pipe.opt
        if o["yolo_async"]:
            return 0.0
        every = max(1, int(o["yolo_every"]))
        scale = (o["yolo_imgsz"] / 640.0) ** 2 * (0.5 if o["roi"] else 1.0)
        flow = 2.5 if (every > 1 and o["track"] == "flow") else 0.0
        return full_ms * scale / every + flow * (every - 1) / every
    return cost


def replay_trial(source, expected, cam_fps, opt, seg, c5, synthetic=False):
    if synthetic:
        pipe = Pipeline(opt, OracleSeg(), c5)
        n = replay(lambda t: source.render(t)[0], source.tl.duration, cam_fps, pipe,
                   truth_quad_at=source.quad_at)
        dur = source.tl.duration
    else:
        pipe = Pipeline(opt, seg, c5)
        n = replay(source.at, source.duration, cam_fps, pipe)
        dur = source.duration
    moves = pipe.user_moves()
    return {"accuracy": round(100 * score_moves(moves, expected), 1),
            "state_ok": bool(np.array_equal(pipe.dec.best_state(),
                                            MODEL.apply_seq(MODEL.solved.copy(), pipe.cam.to_internal(expected)))),
            "decoded_fps": round(n / dur, 1),
            "face_ms": round(pipe.avg("face"), 1), "colour_ms": round(pipe.avg("colour"), 1),
            "decoder_ms": round(pipe.avg("decoder"), 1), "yolo_runs": pipe.counts["yolo"],
            "outline_err_px": round(float(np.mean(pipe.ms["outline_err_px"])), 1) if pipe.ms["outline_err_px"] else "",
            "moves": " ".join(moves)}


def video_sweep(source, expected, cams, seg, c5, synthetic, label):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"video_{stamp}")
    os.makedirs(out, exist_ok=True)
    rows = []
    for name, ch in PIPE_PRESETS.items():
        for cam in cams:
            opt = dict(DEFAULTS, **ch)
            r = replay_trial(source, expected, cam, opt, seg, c5, synthetic)
            r = {"config": name, "camera_fps": cam, **r}
            rows.append(r)
            print(f"{name:22s} cam {cam:3.0f}  moves {r['accuracy']:5.1f}%  state {'ok' if r['state_ok'] else '--'}  "
                  f"decoded {r['decoded_fps']:5.1f} fps  face {r['face_ms']:.1f} ms  colour {r['colour_ms']:.1f} ms  "
                  f"decoder {r['decoder_ms']:.1f} ms  outline err {r['outline_err_px']} px", flush=True)
    with open(os.path.join(out, "video_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "info.json"), "w") as fh:
        json.dump({"source": label, "expected": " ".join(expected)}, fh, indent=1)
    print("saved", out)
    return rows


# =============================================================================
# Live camera UI
# =============================================================================
def load_c5(path=None):
    c5 = L.CalibratedSoft()
    for p in [path, os.path.join(OUT_DIR, "calibration.json"), L.CONFIG["calibration_file"],
              os.path.join(HERE, L.CONFIG["calibration_file"])]:
        if p and os.path.exists(p) and c5.load(p):
            print("colour calibration:", p)
            return c5
    return c5


def fmt_val(v):
    return "every" if v == 0 else str(v)


def draw_live(frame, pipe, cam_fps, proc_fps, dropped, sel, alg, banner, scan_left):
    o = pipe.opt
    view = frame.copy()
    if o["mirror"]:
        view = cv2.flip(view, 1)
    h, w = view.shape[:2]
    scale = 760 / h
    view = cv2.resize(view, (int(w * scale), 760))
    if pipe.quad is not None:
        q = (pipe.quad * scale).astype(np.int32)
        cv2.polylines(view, [q], True, (0, 255, 255), 2)
    if pipe.last_probs is not None:
        for i, row in enumerate(pipe.last_probs):
            r, c = divmod(i, 3)
            name = L.COLORS[int(np.argmax(row))]
            x0, y0 = 20 + c * 34, 20 + r * 34
            cv2.rectangle(view, (x0, y0), (x0 + 30, y0 + 30), L.DRAW_BGR[name], -1)
            if row.max() < 0.6:
                cv2.rectangle(view, (x0, y0), (x0 + 30, y0 + 30), (0, 0, 255), 2)
    panel = np.full((760, 430, 3), 30, np.uint8)
    y = 30

    def line(txt, col=(225, 225, 225), dy=24, sc=0.55, th=1):
        nonlocal y
        cv2.putText(panel, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
        y += dy

    line("SPEED LAB", (80, 200, 255), 30, 0.75, 2)
    line(f"camera     {cam_fps:5.1f} fps")
    line(f"processed  {proc_fps:5.1f} fps   dropped {dropped:4.0f}%")
    line(f"face {pipe.avg('face'):5.1f} (yolo {pipe.avg('yolo'):4.1f} finder {pipe.avg('finder'):4.1f})  "
         f"colour {pipe.avg('colour'):4.1f}  dec {pipe.avg('decoder'):4.1f} ms", (225, 225, 225), 24, 0.48)
    line(f"face from: {pipe.finder.last_note or '-'}", (180, 180, 180), 22, 0.48)
    line(f"total per frame {pipe.avg('total'):5.1f} ms", (180, 180, 180), 30)
    line("SWITCHES   , .  select    - =  change", (150, 150, 160), 24, 0.48)
    for i, k in enumerate(STEPS):
        v = o[k]
        col = (80, 220, 120) if i == sel else (200, 200, 200)
        mark = ">" if i == sel else " "
        line(f"{mark} {k:12s} {fmt_val(v) if k == 'decoder_hz' else v}", col, 21, 0.5, 2 if i == sel else 1)
    y += 6
    line(f"TEST  [1-6]: {ALGS[alg][0]}", (80, 200, 255), 24, 0.52)
    line(pipe.cam.hold_hint(), (150, 150, 160), 22, 0.42)
    line("SPACE start/stop  K scan  R reset  P preset", (150, 150, 160), 22, 0.45)
    moves = pipe.user_moves()[-14:]
    line("moves: " + " ".join(moves), (230, 230, 230), 24, 0.45)
    if scan_left:
        line(f"SCAN: show the {scan_left[0].upper()} face, press K", (0, 200, 255), 24, 0.55, 2)
    if banner:
        for b in banner.split("\n"):
            line(b, (90, 230, 120), 22, 0.5, 2)
    return np.hstack([view, panel])


def live(args, opt, seg, c5):
    cam = LiveCamera(args.camera)
    if not cam.alive:
        print("camera not available")
        return
    pipe = Pipeline(opt, seg, c5, live=True)
    keys = list(STEPS)
    sel, alg, preset_i = 0, "sexy6", -1
    trial, banner = None, ""
    scans, scan_left = {}, []
    last_n, proc_times, drop_hist = 0, collections.deque(maxlen=60), collections.deque(maxlen=120)
    os.makedirs(OUT_DIR, exist_ok=True)
    log = os.path.join(OUT_DIR, "live_trials.csv")
    cv2.namedWindow("speed lab", cv2.WINDOW_NORMAL)
    while True:
        frame, t, n = cam.latest()
        if frame is None or n == last_n:
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            continue
        if last_n:
            drop_hist.append(max(0, n - last_n - 1))
        last_n = n
        now = time.perf_counter()
        if pipe.process(frame, now):
            proc_times.append(now)
        pf = (len(proc_times) - 1) / (proc_times[-1] - proc_times[0]) if len(proc_times) > 2 else 0.0
        dropped = 100.0 * sum(drop_hist) / max(1, sum(drop_hist) + len(drop_hist))
        cv2.imshow("speed lab", draw_live(frame, pipe, cam.fps(), pf, dropped, sel, alg, banner, scan_left))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("."):
            sel = (sel + 1) % len(keys)
        elif k == ord(","):
            sel = (sel - 1) % len(keys)
        elif k in (ord("="), ord("+"), ord("-"), ord("_")):
            key = keys[sel]
            vals = STEPS[key]
            cur = pipe.opt[key]
            i = vals.index(cur) if cur in vals else 0
            i = min(len(vals) - 1, i + 1) if k in (ord("="), ord("+")) else max(0, i - 1)
            pipe.set(key, vals[i])
            if key in ("track", "yolo_every", "roi", "yolo_imgsz", "yolo_async"):
                pipe.since_yolo = 10 ** 6
        elif k == ord("p"):
            names = list(PIPE_PRESETS)
            preset_i = (preset_i + 1) % len(names)
            for key, val in dict(DEFAULTS, **PIPE_PRESETS[names[preset_i]]).items():
                if key not in ("mirror", "camera"):
                    pipe.set(key, val)
            pipe.since_yolo = 10 ** 6
            banner = f"preset: {names[preset_i]}"
        elif ord("1") <= k <= ord("6"):
            alg = list(ALGS)[k - ord("1")]
        elif k == ord("r"):
            pipe.dec.reset()
            banner = "decoder reset to solved"
        elif k == ord("k"):
            if not scan_left:
                scan_left, scans = list(L.COLORS), {}
                banner = ""
            elif pipe.quad is not None:
                flat = warp_face(frame if not pipe.opt["mirror"] else cv2.flip(frame, 1), pipe.quad)
                means = []
                for r in range(3):
                    for c in range(3):
                        p = flat[r * L.CELL + L.MARGIN:(r + 1) * L.CELL - L.MARGIN,
                                 c * L.CELL + L.MARGIN:(c + 1) * L.CELL - L.MARGIN]
                        means.append(L.CalibratedSoft.robust_mean(cv2.cvtColor(p, cv2.COLOR_BGR2LAB), 240.0)[0])
                scans[scan_left.pop(0)] = means
                if not scan_left:
                    c5.calibrate(scans)
                    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as fh:
                        json.dump(c5.to_json(), fh)
                    pipe.c5, pipe.fast = c5, FastColour(c5)
                    pipe.dec.reset()
                    banner = "calibrated - hold solved, white top, green front"
        elif k == ord(" "):
            if trial is None:
                pipe.dec.reset()
                trial = now
                banner = f"GO: {ALGS[alg][0]} ... SPACE when done"
            else:
                dur = now - trial
                exp = ALGS[alg][1].split()
                got = pipe.user_moves()
                acc = 100 * score_moves(got, exp)
                solved = bool(np.array_equal(pipe.dec.best_state(), MODEL.solved))
                tps = len(exp) / dur
                row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "alg": alg,
                       "tps": round(tps, 2), "accuracy": round(acc, 1), "ends_solved": solved,
                       "camera_fps": round(cam.fps(), 1), "processed_fps": round(pf, 1),
                       "face_ms": round(pipe.avg("face"), 1), "colour_ms": round(pipe.avg("colour"), 1),
                       "decoder_ms": round(pipe.avg("decoder"), 1), "moves": " ".join(got),
                       **{k2: pipe.opt[k2] for k2 in STEPS}}
                new = not os.path.exists(log)
                with open(log, "a", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        wr.writeheader()
                    wr.writerow(row)
                banner = f"{tps:.1f} TPS  moves {acc:.0f}%  {'ends solved' if solved else 'NOT solved'}\nsaved to live_trials.csv"
                trial = None
    cam.close()
    cv2.destroyAllWindows()


# =============================================================================
# main
# =============================================================================
def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in DEFAULTS:
            raise SystemExit(f"unknown switch {k}")
        d = DEFAULTS[k]
        out[k] = type(d)(float(v)) if isinstance(d, (int, float)) and not isinstance(d, bool) else v
        if isinstance(d, int):
            out[k] = int(float(v))
    return out


def main():
    ap = argparse.ArgumentParser(description="Speed lab for fast solving")
    ap.add_argument("--sim", action="store_true", help="simulation sweep (no camera)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--video", default=None, help="recorded solve (slow-mo recommended)")
    ap.add_argument("--synthetic", action="store_true", help="rendered solve instead of a video")
    ap.add_argument("--tps", type=float, default=10.0, help="turns/second for --synthetic")
    ap.add_argument("--alg", default="sexy6", choices=list(ALGS))
    ap.add_argument("--camera-fps", default="30", help="e.g. 30 or 30,60,120")
    ap.add_argument("--sweep", action="store_true", help="every pipeline preset x camera fps")
    ap.add_argument("--set", nargs="*", help="switch=value ...")
    ap.add_argument("--preset", default=None, choices=list(PIPE_PRESETS), help="start from a pipeline preset")
    ap.add_argument("--seg", default=None, help="YOLO seg weights")
    ap.add_argument("--cam-setup", default="top-down", choices=list(CAMERAS),
                    help="where the camera is (live / video). Default: top-down like your V63 setup")
    ap.add_argument("--calib", default=None, help="colour calibration json")
    args = ap.parse_args()
    if args.camera is None and not (args.video or args.synthetic or args.sim):
        args.camera = 0                  # plain "python speed_lab.py" = live camera 0
    preset = args.preset or ("recommended" if (args.camera is not None and not args.video) else None)
    opt = dict(DEFAULTS, **(PIPE_PRESETS[preset] if preset else {}))
    if not args.synthetic and not args.sim:
        opt["camera"] = args.cam_setup
    opt.update(parse_set(args.set))
    if args.sim:
        sim_sweep(quick=args.quick)
        return
    cams = [float(x) for x in args.camera_fps.split(",")]
    expected = ALGS[args.alg][1].split()
    if args.synthetic:
        src = SyntheticVideo(expected, args.tps)
        c5 = synthetic_c5()
        if args.sweep:
            video_sweep(src, expected, cams, None, c5, True, f"synthetic {args.alg} @ {args.tps} TPS")
        else:
            for cam in cams:
                print(cam, replay_trial(src, expected, cam, opt, None, c5, True))
        return
    seg_path = args.seg or L.first_existing(L.CONFIG["seg_weights"])
    seg = Seg(seg_path)
    if not seg.ok:
        print(seg.reason)
    c5 = load_c5(args.calib)
    if args.video:
        if not c5.ready:
            raise SystemExit("need a colour calibration: run live once and press K, or pass --calib")
        src = VideoFrames(args.video)
        print(f"video: {len(src.frames)} frames at {src.fps:.0f} fps ({src.duration:.1f} s)")
        if args.sweep:
            video_sweep(src, expected, cams, seg, c5, False, args.video)
        else:
            for cam in cams:
                print(cam, replay_trial(src, expected, cam, opt, seg, c5, False))
        return
    live(args, opt, seg, c5)


if __name__ == "__main__":
    main()
