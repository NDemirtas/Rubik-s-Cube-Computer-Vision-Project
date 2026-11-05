#!/usr/bin/env python3
"""
cube_lab.py  -  Developer test bench for the Rubik's cube tracker
=================================================================

Runs the face-isolation systems and colour classifiers from the whole
project SIDE BY SIDE on the same frames, measures them, stress-tests them
and exports numbers + charts for the report.

  python cube_lab.py                         # live camera (index 0)
  python cube_lab.py --camera 1              # other camera
  python cube_lab.py --source synthetic      # rendered cube with perfect ground truth
  python cube_lab.py --source video.mp4      # replay a recording (same frames for every system)
  python cube_lab.py --source frames_dir/    # folder of images
  python cube_lab.py --seg path/to/best.pt   # your trained YOLO-seg weights
  python cube_lab.py --headless --source synthetic --frames 300 --export
  python cube_lab.py --tracker-sweep [--quick]

Press H inside the app for every control. Everything is also clickable.

Systems compared
----------------
Face isolation (where are the 9 stickers?):
  D1 Sticker contours (Jay Hack)     adaptive threshold -> square sticker contours -> 3x3
  D2 Fixed centre box                no detection, user holds the cube in a box
  D3 Dark-background minRect         Jokinen thesis: mask dark pixels, contour filters, minAreaRect
  D4 Sticker grid rebuild            CLAHE stickers -> adjacency groups -> rebuild 3x3 from pitch
  D5 YOLO bounding box               YOLO box -> axis-aligned square
  D6 YOLO seg + minAreaRect          YOLO mask -> always the smallest rotated rectangle
  D7 YOLO seg quad (CURRENT V62)     YOLO mask -> approxPolyDP 4 corners (minAreaRect fallback)
  D8 YOLO pose keypoints             7 keypoint model -> face from keypoints (needs pose weights)
  D9 Hybrid lattice                  YOLO region + every 3x3 sticker lattice, chosen by fit/region/memory,
                                     YOLO quad as fallback (proposed V63 detector)
  D10 Hybrid + state hint            D9 + soft preference for the grid whose colours fit a face the tracker
                                     expects (j: true / wrong / off). Colours are still read from pixels.
  D11 V63 (old settings)             the real V63 tracker from speed_lab.py (finder + colours + decoder),
                                     YOLO on every frame. Needs speed_lab.py in this folder.
  D12 V63 optimised                  the same code with the speed-lab settings: YOLO every 4th frame in
                                     the background, finder on a crop, 3 lock reads, decoder every frame

Colour classification (which colour is each sticker?):
  C1 HSV fixed ranges                fixed hue/sat/value limits, majority of pixels
  C2 HSV + skin mask                 C1 + finger/skin rejection + red/orange hue refinement
  C3 LAB fixed references            nearest hand-set LAB reference colour
  C4 K-means LAB (calibrated)        6-face scan, k-means clusters (pre-V53 method)
  C5 Calibrated soft (CURRENT V54)   median centres, weighted LAB distance, glare-aware probabilities
  C6 CNN (Jay Hack)                  HackRubiksNet on 32x32 crops (needs rubiks_model.pth + torch)
"""

import argparse
import collections
import copy
import csv
import datetime
import glob
import itertools
import json
import math
import os
import random
import sys
import time
import warnings

import cv2
import numpy as np

# =============================================================================
# CONFIG  (edit freely - everything the lab uses lives here)
# =============================================================================
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

CONFIG = {
    # Model weights. The first existing path in each list is used.
    "seg_weights": ["runs/segment/rubiks_seg_model/weights/best.pt",
                    "runs/segment/*/weights/best.pt", "yolov8n-seg.pt"],
    "det_weights": ["runs/detect/*/weights/best.pt"],          # optional separate box model
    "pose_weights": ["runs/pose/train-3/weights/best.pt", "runs/pose/*/weights/best.pt"],
    "cnn_weights": ["rubiks_model.pth"],
    "calibration_file": "colour_calibration.json",            # the main tracker's file
    "yolo_conf": 0.40,
    "output_dir": "lab_runs",
    # Window
    "window_w": 1600,
    "window_h": 940,
    "sidebar_w": 300,
    # Benchmark
    "benchmark_frames": 300,
    # Robustness sweep: frames per level (synthetic) / repeats (frozen frame)
    "sweep_samples_synthetic": 12,
    "sweep_repeats_frozen": 4,
}

COLORS = ["White", "Yellow", "Green", "Blue", "Red", "Orange"]
LETTER = {"White": "W", "Yellow": "Y", "Green": "G", "Blue": "B", "Red": "R",
          "Orange": "O", "Unknown": "?"}
FROM_LETTER = {v: k for k, v in LETTER.items() if k != "Unknown"}
DRAW_BGR = {"White": (245, 245, 245), "Yellow": (0, 230, 255), "Green": (40, 190, 40),
            "Blue": (220, 90, 20), "Red": (30, 30, 220), "Orange": (0, 140, 255),
            "Unknown": (110, 110, 110)}

UI = {
    "bg": (24, 24, 28), "panel": (36, 36, 42), "panel2": (48, 48, 56),
    "text": (230, 230, 230), "dim": (150, 150, 160), "accent": (255, 190, 60),
    "on": (90, 200, 120), "off": (90, 90, 100), "bad": (70, 70, 230), "good": (90, 210, 110),
    "warn": (40, 180, 240), "gt": (0, 255, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# Small utilities
# =============================================================================
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


def order_quad_legacy(pts):
    """V54/V62 rule, kept to measure how often it breaks."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def quad_area(q):
    return float(abs(cv2.contourArea(np.asarray(q, np.float32))))


def quad_iou(a, b):
    """IoU of two quads (rasterised in their joint bounding box)."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    allp = np.vstack([a, b])
    x0, y0 = np.floor(allp.min(0)) - 2
    x1, y1 = np.ceil(allp.max(0)) + 2
    w, h = int(x1 - x0), int(y1 - y0)
    if w <= 0 or h <= 0 or w * h > 9e6:
        return 0.0
    s = 1.0
    if max(w, h) > 600:
        s = 600.0 / max(w, h)
    W, H = max(1, int(w * s)), max(1, int(h * s))
    ma = np.zeros((H, W), np.uint8)
    mb = np.zeros((H, W), np.uint8)
    cv2.fillPoly(ma, [((a - [x0, y0]) * s).astype(np.int32)], 1)
    cv2.fillPoly(mb, [((b - [x0, y0]) * s).astype(np.int32)], 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter) / union if union else 0.0


def corner_error(q, ref):
    """Mean corner distance (px), best over the 4 cyclic orderings."""
    if q is None or ref is None:
        return None
    q = np.asarray(q, np.float32)
    ref = np.asarray(ref, np.float32)
    best = 1e9
    for k in range(4):
        e = float(np.linalg.norm(np.roll(q, k, axis=0) - ref, axis=1).mean())
        best = min(best, e)
    return best


def face_size(q):
    if q is None:
        return 1.0
    q = np.asarray(q, np.float32)
    return max(1.0, float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])))


def put(img, text, org, scale=0.45, color=None, thick=1):
    cv2.putText(img, str(text), (int(org[0]), int(org[1])), FONT, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(text), (int(org[0]), int(org[1])), FONT, scale,
                color or UI["text"], thick, cv2.LINE_AA)


def put_plain(img, text, org, scale=0.45, color=None, thick=1):
    cv2.putText(img, str(text), (int(org[0]), int(org[1])), FONT, scale,
                color or UI["text"], thick, cv2.LINE_AA)


def text_w(text, scale=0.45, thick=1):
    return cv2.getTextSize(str(text), FONT, scale, thick)[0][0]


class LatStats:
    """Rolling statistics of a timing series (ms)."""

    def __init__(self, maxlen=5000):
        self.v = collections.deque(maxlen=maxlen)

    def add(self, x):
        if x is not None:
            self.v.append(float(x))

    def _a(self):
        return np.asarray(self.v, dtype=float)

    @property
    def n(self):
        return len(self.v)

    def mean(self):
        return float(self._a().mean()) if self.v else float("nan")

    def median(self):
        return float(np.median(self._a())) if self.v else float("nan")

    def p95(self):
        return float(np.percentile(self._a(), 95)) if self.v else float("nan")

    def max(self):
        return float(self._a().max()) if self.v else float("nan")

    def summary(self):
        return {"n": self.n, "mean": self.mean(), "median": self.median(),
                "p95": self.p95(), "max": self.max()}


def fmt(x, nd=1, unit=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{nd}f}{unit}"


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


# =============================================================================
# Synthetic cube scenes (perfect ground truth for quad AND sticker colours)
# =============================================================================
STICKER_BGR = {  # what a real camera roughly sees (BGR)
    "White": (225, 228, 222), "Yellow": (40, 210, 228), "Green": (70, 165, 35),
    "Blue": (165, 85, 25), "Red": (45, 35, 195), "Orange": (25, 115, 240),
}


class SyntheticScene:
    """Renders a cube face (plus a side face and background clutter) with known
    corners and sticker colours. The cube drifts slowly; every few seconds the
    face pattern changes so every colour gets tested."""

    def __init__(self, w=1280, h=720, seed=None, clutter=True, solved=False, vary_light=True, dark_bg=False,
                 angled=False):
        self.w, self.h = w, h
        self.angled = angled
        self.dark_bg = dark_bg
        self.vary_light = vary_light
        self.rng = np.random.default_rng(seed)
        self.clutter = clutter
        self.solved = solved
        self.frame_idx = 0
        self._bg = None
        self._new_background()
        self.cx, self.cy = w * 0.5, h * 0.5
        self.size = 260.0
        self.angle = 0.0
        self.vx = self.vy = self.va = 0.0
        self.persp = (0.0, 0.0)
        self._new_pattern()

    def _new_background(self):
        w, h = self.w, self.h
        # every "scene" has its own lighting: brightness and warm/cool tint drift
        # away from the light the colours were calibrated in
        self.gain = float(self.rng.uniform(0.72, 1.12)) if self.vary_light else 1.0
        self.tint = float(self.rng.uniform(-22, 22)) if self.vary_light else 0.0
        g = np.linspace(0, 1, w)[None, :, None]
        base = self.rng.uniform(40, 120, 3)
        top = self.rng.uniform(60, 170, 3)
        if self.dark_bg:   # Jokinen's requirement: black shirt behind the cube
            base, top = self.rng.uniform(5, 25, 3), self.rng.uniform(10, 35, 3)
        bg = (base * (1 - g) + top * g) * np.ones((h, w, 3))
        bg = bg.astype(np.uint8)
        if self.clutter and not self.dark_bg:
            for _ in range(14):
                x0 = int(self.rng.uniform(0, w))
                y0 = int(self.rng.uniform(0, h))
                ww = int(self.rng.uniform(40, 260))
                hh = int(self.rng.uniform(30, 200))
                col = tuple(int(c) for c in self.rng.uniform(20, 240, 3))
                if self.rng.random() < 0.35:   # some clutter in cube colours
                    col = STICKER_BGR[COLORS[int(self.rng.integers(0, 6))]]
                cv2.rectangle(bg, (x0, y0), (x0 + ww, y0 + hh), col, -1)
        self._bg = cv2.GaussianBlur(bg, (5, 5), 0)

    def _new_pattern(self):
        if self.solved:
            c = COLORS[int(self.rng.integers(0, 6))]
            self.grid = [c] * 9
        else:
            self.grid = [COLORS[int(self.rng.integers(0, 6))] for _ in range(9)]
            self.grid[4] = COLORS[int(self.rng.integers(0, 6))]
        self.side_grid = [COLORS[int(self.rng.integers(0, 6))] for _ in range(9)]
        self.right_grid = [COLORS[int(self.rng.integers(0, 6))] for _ in range(9)]

    def randomize_pose(self):
        self.cx = self.rng.uniform(self.w * 0.3, self.w * 0.7)
        self.cy = self.rng.uniform(self.h * 0.3, self.h * 0.7)
        self.size = self.rng.uniform(180, 330)
        self.angle = self.rng.uniform(-35, 35)
        self.persp = tuple(self.rng.uniform(-0.12, 0.12, 2))

    def quad(self):
        s = self.size / 2.0
        a = math.radians(self.angle)
        ca, sa = math.cos(a), math.sin(a)
        px, py = self.persp
        base = np.array([[-s, -s], [s, -s], [s, s], [-s, s]], np.float32)
        # simple keystone: shrink one side
        base[0, 0] *= (1 - px); base[3, 0] *= (1 - px)
        base[1, 0] *= (1 + px); base[2, 0] *= (1 + px)
        base[0, 1] *= (1 - py); base[1, 1] *= (1 - py)
        base[2, 1] *= (1 + py); base[3, 1] *= (1 + py)
        R = np.array([[ca, -sa], [sa, ca]], np.float32)
        q = base @ R.T + np.array([self.cx, self.cy], np.float32)
        return order_quad(q)

    def step(self):
        self.frame_idx += 1
        # slow random drift
        self.vx = 0.9 * self.vx + self.rng.normal(0, 0.6)
        self.vy = 0.9 * self.vy + self.rng.normal(0, 0.6)
        self.va = 0.9 * self.va + self.rng.normal(0, 0.15)
        self.cx = float(np.clip(self.cx + self.vx, self.w * 0.25, self.w * 0.75))
        self.cy = float(np.clip(self.cy + self.vy, self.h * 0.28, self.h * 0.72))
        self.angle = float(np.clip(self.angle + self.va, -35, 35))
        if self.frame_idx % 45 == 0:
            self._new_pattern()
        if self.frame_idx % 120 == 0:
            self._new_background()
        return self.render()

    def render(self, grid=None):
        grid = grid or self.grid
        img = self._bg.copy()
        q = self.quad()
        # side face (above the front face, darker) to look 3D
        up = q[0] - q[3]
        k_up = 0.75 if self.angled else 0.35
        if self.angled:   # held at an angle: top AND side faces nearly as big as the front
            right = q[1] - q[0]
            skew = up * 0.25
            rside = np.array([q[1], q[1] + right * 0.6 + skew, q[2] + right * 0.6 + skew, q[2]], np.float32)
            side = np.array([q[0] + up * k_up + right * 0.25, q[1] + up * k_up + right * 0.25,
                             q[1], q[0]], np.float32)
            self._draw_face(img, rside, self.right_grid, shade=0.7)
            self._draw_face(img, side, self.side_grid, shade=0.85)
        elif not self.dark_bg:   # Jokinen's setup shows the face straight on
            side = np.array([q[0] + up * k_up, q[1] + up * k_up, q[1], q[0]], np.float32)
            self._draw_face(img, side, self.side_grid, shade=0.55, squash=True)
        self._draw_face(img, q, grid, shade=1.0)
        return img, q.copy(), list(grid)

    def _draw_face(self, img, quad, grid, shade=1.0, squash=False):
        cv2.fillPoly(img, [quad.astype(np.int32)], (18, 18, 18), cv2.LINE_AA)
        src = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
        M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        gap = 0.09
        light = np.array([1.0, 1.0, 1.0])
        for r in range(3):
            for c in range(3):
                cell = np.float32([[c + gap, r + gap], [c + 1 - gap, r + gap],
                                   [c + 1 - gap, r + 1 - gap], [c + gap, r + 1 - gap]])
                pts = cv2.perspectiveTransform(cell.reshape(-1, 1, 2), M).reshape(-1, 2)
                base = np.array(STICKER_BGR[grid[r * 3 + c]], float)
                # small lighting gradient across the face + per sticker jitter
                k = shade * (0.92 + 0.08 * (c / 2.0)) * (1.0 + self.rng.normal(0, 0.025))
                col = base * k * self.gain + self.rng.normal(0, 4, 3)
                col = np.clip(col + np.array([-self.tint, 0.0, self.tint]), 0, 255)
                cv2.fillPoly(img, [pts.astype(np.int32)], tuple(int(x) for x in col), cv2.LINE_AA)


# =============================================================================
# Frame sources
# =============================================================================
class Source:
    name = "source"
    gt_available = False

    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class CameraSource(Source):
    def __init__(self, index=0, width=1280, height=720):
        self.name = f"camera {index}"
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened() and index != 0:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, f = self.cap.read()
        return (f if ok else None), None, None

    def release(self):
        self.cap.release()


class VideoSource(Source):
    def __init__(self, path, loop=True):
        self.name = os.path.basename(path)
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)

    def read(self):
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            ok, f = self.cap.read()
        return (f if ok else None), None, None

    def release(self):
        self.cap.release()


class FolderSource(Source):
    def __init__(self, path):
        self.name = os.path.basename(os.path.normpath(path))
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.files = sorted(f for e in exts for f in glob.glob(os.path.join(path, e)))
        self.i = 0

    def read(self):
        if not self.files:
            return None, None, None
        f = cv2.imread(self.files[self.i % len(self.files)])
        self.i += 1
        return f, None, None


class SyntheticSource(Source):
    gt_available = True

    def __init__(self, seed=0, solved=False, dark_bg=False, angled=False):
        self.name = "synthetic" + (" (dark background)" if dark_bg else "") + (" (angled)" if angled else "")
        self.scene = SyntheticScene(seed=seed, solved=solved, dark_bg=dark_bg, angled=angled)

    def read(self):
        img, q, grid = self.scene.step()
        return img, q, grid


# =============================================================================
# Perturbations (stress tests applied to the frame before every system)
# =============================================================================
class Perturb:
    """name, default, min, max, step, description"""
    SPECS = [
        ("brightness", 0, -150, 150, 15, "add to every pixel"),
        ("contrast", 1.0, 0.3, 2.0, 0.1, "multiply around mid-grey"),
        ("gamma", 1.0, 0.3, 3.0, 0.1, ">1 darker mid-tones"),
        ("temperature", 0, -80, 80, 10, "+ warm (orange light) / - cool"),
        ("desaturate", 0.0, 0.0, 1.0, 0.1, "wash colours towards grey"),
        ("noise", 0, 0, 50, 5, "gaussian sensor noise (std)"),
        ("motion_blur", 0, 0, 31, 2, "horizontal blur length (px)"),
        ("glare", 0.0, 0.0, 1.0, 0.1, "bright specular spot on the cube"),
        ("fingers", 0, 0, 4, 1, "skin-coloured fingers over the face"),
        ("jpeg", 100, 5, 100, 10, "JPEG quality"),
        ("downscale", 1.0, 0.2, 1.0, 0.1, "resolution factor"),
    ]

    def __init__(self):
        self.values = {s[0]: s[1] for s in self.SPECS}
        self.rng = np.random.default_rng(1)

    def spec(self, name):
        return next(s for s in self.SPECS if s[0] == name)

    def nudge(self, name, direction):
        _, d, lo, hi, step, _ = self.spec(name)
        v = self.values[name] + direction * step
        v = min(hi, max(lo, v))
        self.values[name] = round(v, 3) if isinstance(step, float) else int(v)

    def reset(self):
        self.values = {s[0]: s[1] for s in self.SPECS}

    def active(self):
        return {k: v for k, v in self.values.items() if v != self.spec(k)[1]}

    def describe(self):
        a = self.active()
        return ", ".join(f"{k}={v}" for k, v in a.items()) if a else "none"

    def apply(self, img, focus_quad=None, values=None, rng=None):
        v = values if values is not None else self.values
        rng = rng or self.rng
        out = img
        if v.get("downscale", 1.0) < 0.999:
            f = float(v["downscale"])
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        out = out.astype(np.float32)
        if v.get("contrast", 1.0) != 1.0:
            out = (out - 128.0) * float(v["contrast"]) + 128.0
        if v.get("brightness", 0):
            out = out + float(v["brightness"])
        if v.get("temperature", 0):
            t = float(v["temperature"])
            out[..., 2] += t * 0.8
            out[..., 0] -= t * 0.8
        if v.get("desaturate", 0.0) > 0:
            g = out.mean(axis=2, keepdims=True)
            a = float(v["desaturate"])
            out = out * (1 - a) + g * a
        out = np.clip(out, 0, 255)
        if v.get("gamma", 1.0) != 1.0:
            out = 255.0 * (out / 255.0) ** float(v["gamma"])
        out = out.astype(np.uint8)
        h, w = out.shape[:2]
        centre = np.array([w / 2, h / 2], np.float32)
        size = min(w, h) * 0.35
        if focus_quad is not None:
            centre = np.asarray(focus_quad, np.float32).mean(axis=0)
            size = face_size(focus_quad)
        if v.get("glare", 0.0) > 0:
            g = float(v["glare"])
            layer = np.zeros((h, w), np.float32)
            gx = centre[0] + rng.uniform(-0.25, 0.25) * size
            gy = centre[1] + rng.uniform(-0.25, 0.25) * size
            cv2.ellipse(layer, (int(gx), int(gy)), (int(size * 0.22), int(size * 0.13)),
                        float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
            layer = cv2.GaussianBlur(layer, (0, 0), size * 0.04 + 1)
            out = np.clip(out.astype(np.float32) + (layer * g * 255.0)[..., None], 0, 255).astype(np.uint8)
        if v.get("fingers", 0) > 0:
            for _ in range(int(v["fingers"])):
                side = rng.integers(0, 4)
                ang = rng.uniform(-0.4, 0.4) + side * math.pi / 2
                d = np.array([math.cos(ang), math.sin(ang)])
                start = centre + d * size * 0.95
                end = centre + d * size * rng.uniform(0.15, 0.45)
                skin = (int(rng.uniform(110, 160)), int(rng.uniform(140, 180)), int(rng.uniform(190, 230)))
                cv2.line(out, tuple(int(x) for x in start), tuple(int(x) for x in end), skin,
                         int(size * 0.17), cv2.LINE_AA)
                cv2.circle(out, tuple(int(x) for x in end), int(size * 0.085), skin, -1, cv2.LINE_AA)
        if v.get("motion_blur", 0) >= 3:
            k = int(v["motion_blur"]) | 1
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0 / k
            out = cv2.filter2D(out, -1, kern)
        if v.get("noise", 0) > 0:
            n = rng.normal(0, float(v["noise"]), out.shape)
            out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
        if v.get("jpeg", 100) < 100:
            ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(v["jpeg"])])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out


# =============================================================================
# YOLO runner (one inference per frame, shared by every YOLO-based system)
# =============================================================================
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


class YoloRunner:
    def __init__(self, path, label):
        self.label = label
        self.path = path
        self.model = None
        self.reason = ""
        self.device = "cpu"
        self._cache_id = None
        self._cache = (None, 0.0)
        self.generic = bool(path) and os.path.basename(path).startswith("yolov8n")
        if not path or not os.path.exists(path):
            self.reason = f"no {label} weights found"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.device = select_device()
            # warm-up so the first timed frame is not a one-off outlier
            self.model(np.zeros((480, 640, 3), np.uint8), verbose=False, device=self.device)
        except Exception as exc:  # ultralytics missing or bad file
            self.model = None
            self.reason = f"{label}: {exc.__class__.__name__}: {exc}"[:90]

    @property
    def ok(self):
        return self.model is not None

    def run(self, frame, frame_id):
        if not self.ok:
            return None, 0.0, False
        if frame_id == self._cache_id:
            return self._cache[0], self._cache[1], False
        t0 = time.perf_counter()
        try:
            res = self.model(frame, conf=CONFIG["yolo_conf"], verbose=False, device=self.device)[0]
        except Exception:
            res = self.model(frame, conf=CONFIG["yolo_conf"], verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        self._cache_id = frame_id
        self._cache = (res, dt)
        return res, dt, True


# =============================================================================
# Face isolation systems
# =============================================================================
class Detection:
    __slots__ = ("found", "quad", "t_own", "t_shared", "t_extra", "mask", "cands", "points",
                 "box", "note", "keypoints", "stages", "fresh")

    def __init__(self):
        self.found = False
        self.quad = None
        self.t_own = 0.0
        self.t_shared = 0.0
        self.t_extra = 0.0      # work shared with another system on this frame (cached), still charged
        self.mask = None        # full-frame uint8 mask (what the system "sees")
        self.cands = []         # candidate contours (sticker systems)
        self.points = None      # detected sticker centres
        self.box = None
        self.note = ""
        self.keypoints = None
        self.stages = {}        # name -> ms
        self.fresh = False      # True if this system ran the shared YOLO model itself

    @property
    def t_total(self):
        return self.t_own + self.t_shared + self.t_extra


class Detector:
    key = "D?"
    name = "detector"
    short = "det"
    desc = ""
    era = ""
    available = True
    reason = ""

    def detect(self, frame, ctx):
        raise NotImplementedError

    def __call__(self, frame, ctx):
        t0 = time.perf_counter()
        d = self.detect(frame, ctx)
        d.t_own = max(0.0, (time.perf_counter() - t0) * 1000.0 - (d.t_shared if d.fresh else 0.0))
        if d.quad is not None:
            d.quad = order_quad(d.quad)
        d.found = d.quad is not None
        return d


def _sticker_regions(gray, area_lo, area_hi, edge_lo=30, edge_hi=90, erode=0):
    """Regions enclosed by edges that look like square stickers."""
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
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if not (4 <= len(approx) <= 6):
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 0.6 or ar > 1.4:
            continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        if ha <= 0 or a / ha < 0.85:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        out.append((c, np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], np.float32), math.sqrt(a)))
    # the same sticker can come back twice (inner/outer boundary) - keep one
    out.sort(key=lambda t: -t[2])
    kept = []
    for c in out:
        if all(np.linalg.norm(c[1] - k[1]) > 0.4 * min(c[2], k[2]) for k in kept):
            kept.append(c)
    return kept


def _largest_group(cands, link=1.75):
    if not cands:
        return []
    cen = np.array([c[1] for c in cands])
    size = np.median([c[2] for c in cands])
    n = len(cands)
    d = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    adj = d < link * size
    seen = [False] * n
    best = []
    for i in range(n):
        if seen[i]:
            continue
        comp, stack = [], [i]
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in np.nonzero(adj[j])[0]:
                if not seen[k]:
                    seen[k] = True
                    stack.append(int(k))
        if len(comp) > len(best):
            best = comp
    if len(best) > 9:
        g = cen[best]
        med = np.median(g, axis=0)
        order = np.argsort(np.linalg.norm(g - med, axis=1))[:9]
        best = [best[i] for i in order]
    return [cands[i] for i in best]


def _quad_from_centres(centres, pitch=None, rebuild=False):
    pts = np.asarray(centres, np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if pitch is None:
        nn = []
        for i in range(len(pts)):
            d = np.linalg.norm(pts - pts[i], axis=1)
            d = d[d > 1e-3]
            if len(d):
                nn.append(d.min())
        pitch = float(np.median(nn)) if nn else max(w, h) / 2.0
    if rebuild:   # fingers hid some stickers: centres must span 2 pitches each way
        w = max(w, 2 * pitch)
        h = max(h, 2 * pitch)
    w, h = w + pitch, h + pitch  # centre span (2 pitches) -> whole face (3 pitches)
    return cv2.boxPoints(((cx, cy), (w, h), ang)), pitch


class StickerContours(Detector):
    key, short = "D1", "Sticker contours"
    name = "D1 Sticker contours (Jay Hack)"
    era = "first prototype"
    desc = "edges -> square sticker contours -> needs 8+ stickers -> 3x3"

    def detect(self, frame, ctx):
        d = Detection()
        t = time.perf_counter()
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        A = frame.shape[0] * frame.shape[1]
        cands = _sticker_regions(gray, 0.0004 * A, 0.02 * A)
        d.stages["contours"] = (time.perf_counter() - t) * 1000
        d.cands = [c[0] for c in cands]
        grp = _largest_group(cands)
        d.points = np.array([g[1] for g in grp], np.float32) if grp else None
        if len(grp) >= 8:
            d.quad, _ = _quad_from_centres([g[1] for g in grp])
        else:
            d.note = f"{len(grp)} stickers"
        return d


class FixedBox(Detector):
    key, short = "D2", "Fixed box"
    name = "D2 Fixed centre box"
    era = "early HSV grid"
    desc = "no detection: hold the cube inside a fixed box"

    def detect(self, frame, ctx):
        d = Detection()
        h, w = frame.shape[:2]
        s = 0.45 * min(h, w) / 2
        cx, cy = w / 2, h / 2
        d.quad = np.float32([[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]])
        return d


class DarkBackgroundRect(Detector):
    key, short = "D3", "Dark-bg minRect"
    name = "D3 Dark-background minRect (Jokinen)"
    era = "Jokinen 2024 thesis"
    desc = "mask dark pixels, size/centre/angle/square filters, minAreaRect"
    V_DARK = 70

    def detect(self, frame, ctx):
        d = Detection()
        h, w = frame.shape[:2]
        k = w / 640.0                      # thesis numbers were for a 640x480 webcam
        kA = (w * h) / (640.0 * 480.0)
        t = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 2] > self.V_DARK).astype(np.uint8) * 255
        ks = int(9 * k) | 1   # join stickers across the black plastic gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ks, ks), np.uint8))
        d.mask = mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        d.stages["mask+contours"] = (time.perf_counter() - t) * 1000
        tol = 20 * k
        reasons = collections.Counter()
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 10000 * kA or a > 40000 * kA:
                reasons["size"] += 1
                continue
            M = cv2.moments(c)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            if math.hypot(cx - w / 2, cy - h / 2) > 50 * k:
                reasons["off-centre"] += 1
                continue
            rect = cv2.minAreaRect(c)
            (rw, rh), ang = rect[1], rect[2]
            if 30 < ang < 60:
                reasons["rotated"] += 1
                continue
            box = cv2.boxPoints(rect)
            dists = [abs(cv2.pointPolygonTest(box.astype(np.float32), (float(p[0][0]), float(p[0][1])), True))
                     for p in c[:: max(1, len(c) // 120)]]
            if max(dists) > tol or abs(rw - rh) > tol:
                reasons["not square"] += 1
                continue
            d.quad = box
            d.cands = [c]
            return d
        d.note = ("rejected: " + ", ".join(f"{k} {v}" for k, v in reasons.items())) if reasons else "no blob"
        return d


class GridRebuild(Detector):
    key, short = "D4", "Grid rebuild"
    name = "D4 Sticker grid rebuild"
    era = "AI_grid / homography prototypes"
    desc = "CLAHE stickers -> adjacency groups -> rebuild 3x3 from pitch (4+ stickers)"

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def detect(self, frame, ctx):
        d = Detection()
        t = time.perf_counter()
        L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
        g = cv2.GaussianBlur(self.clahe.apply(L), (5, 5), 0)
        A = frame.shape[0] * frame.shape[1]
        cands = _sticker_regions(g, 0.0004 * A, 0.02 * A, 25, 75, erode=1)
        d.stages["clahe+contours"] = (time.perf_counter() - t) * 1000
        d.cands = [c[0] for c in cands]
        grp = _largest_group(cands, link=1.9)
        d.points = np.array([x[1] for x in grp], np.float32) if grp else None
        if len(grp) >= 4:
            d.quad, _ = _quad_from_centres([x[1] for x in grp], rebuild=True)
        else:
            d.note = f"{len(grp)} stickers"
        return d


def _seg_mask(res, frame_shape):
    if res is None or res.masks is None or len(res.masks) == 0:
        return None, None
    h, w = frame_shape[:2]
    boxes = res.boxes.xyxy.cpu().numpy()
    idx = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
    m = res.masks.data[idx].cpu().numpy()
    m = cv2.resize(m, (w, h))
    return (m > 0.5).astype(np.uint8) * 255, boxes[idx]


class YoloBox(Detector):
    key, short = "D5", "YOLO box"
    name = "D5 YOLO bounding box"
    era = "first YOLO detector"
    desc = "largest YOLO box -> axis-aligned square (no shape)"

    def __init__(self, seg, det):
        self.seg, self.det = seg, det
        self.available = (det and det.ok) or (seg and seg.ok)
        self.reason = "" if self.available else "no YOLO weights"

    def detect(self, frame, ctx):
        d = Detection()
        runner = self.det if (self.det and self.det.ok) else self.seg
        res, dt, d.fresh = runner.run(frame, ctx["frame_id"])
        d.t_shared = dt
        d.stages["yolo inference" if d.fresh else "yolo (shared)"] = dt
        if res is None or res.boxes is None or len(res.boxes) == 0:
            d.note = "no box"
            return d
        boxes = res.boxes.xyxy.cpu().numpy()
        b = boxes[int(np.argmax([(x[2] - x[0]) * (x[3] - x[1]) for x in boxes]))]
        d.box = b
        d.quad = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        return d


class YoloSegRect(Detector):
    key, short = "D6", "YOLO seg minRect"
    name = "D6 YOLO seg + minAreaRect"
    era = "seg variant"
    desc = "YOLO mask -> always the smallest rotated rectangle"

    def __init__(self, seg):
        self.seg = seg
        self.available = bool(seg and seg.ok)
        self.reason = "" if self.available else (seg.reason if seg else "no seg weights")

    def detect(self, frame, ctx):
        d = Detection()
        res, dt, d.fresh = self.seg.run(frame, ctx["frame_id"])
        d.t_shared = dt
        d.stages["yolo inference" if d.fresh else "yolo (shared)"] = dt
        t = time.perf_counter()
        mask, _ = _seg_mask(res, frame.shape)
        d.mask = mask
        if mask is None:
            d.note = "no mask"
            return d
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            d.cands = [c]
            d.quad = cv2.boxPoints(cv2.minAreaRect(c))
        d.stages["mask->quad"] = (time.perf_counter() - t) * 1000
        return d


class YoloSegQuad(Detector):
    key, short = "D7", "YOLO seg quad"
    name = "D7 YOLO seg quad (CURRENT)"
    era = "V54 / V62"
    desc = "YOLO mask -> approxPolyDP 4 corners, minAreaRect fallback"

    def __init__(self, seg):
        self.seg = seg
        self.available = bool(seg and seg.ok)
        self.reason = "" if self.available else (seg.reason if seg else "no seg weights")
        self.fallbacks = 0

    def detect(self, frame, ctx):
        d = Detection()
        res, dt, d.fresh = self.seg.run(frame, ctx["frame_id"])
        d.t_shared = dt
        d.stages["yolo inference" if d.fresh else "yolo (shared)"] = dt
        t = time.perf_counter()
        mask, _ = _seg_mask(res, frame.shape)
        d.mask = mask
        if mask is None:
            d.note = "no mask"
            return d
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            d.cands = [c]
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                d.quad = approx.reshape(4, 2).astype(np.float32)
                d.note = "4-corner fit"
            else:
                d.quad = cv2.boxPoints(cv2.minAreaRect(c))
                d.note = f"fallback ({len(approx)} pts)"
                self.fallbacks += 1
        d.stages["mask->quad"] = (time.perf_counter() - t) * 1000
        return d


class YoloPose(Detector):
    key, short = "D8", "YOLO pose"
    name = "D8 YOLO pose keypoints"
    era = "3-face keypoint attempt"
    desc = "7 keypoints -> face polygon [k0,k1,k2,k3]"

    def __init__(self, pose):
        self.pose = pose
        self.available = bool(pose and pose.ok)
        self.reason = "" if self.available else (pose.reason if pose else "no pose weights")

    def detect(self, frame, ctx):
        d = Detection()
        res, dt, d.fresh = self.pose.run(frame, ctx["frame_id"])
        d.t_shared = dt
        d.stages["yolo inference" if d.fresh else "yolo (shared)"] = dt
        if res is None or res.keypoints is None or len(res.keypoints) == 0:
            d.note = "no keypoints"
            return d
        k = res.keypoints.xy[0].cpu().numpy()
        d.keypoints = k
        if len(k) >= 4 and not np.any(np.all(k[:4] == 0, axis=1)):
            d.quad = k[:4].astype(np.float32)
        else:
            d.note = "keypoints missing"
        return d


# =============================================================================
# D9 / D10: hybrid lattice detector
#   YOLO says roughly WHERE the cube is, the stickers say EXACTLY where the
#   face grid is, memory keeps it steady, and D10 adds a soft hint from the
#   face the tracker expects. The hint only picks between grids that really
#   exist in the image; colours are always read from pixels afterwards.
# =============================================================================
_HYB_CACHE = {"id": None, "val": None, "ms": 0.0, "hit": False}


def _hybrid_stickers(frame, frame_id, clahe):
    """Sticker candidates (same pipeline as D4), shared by D9 and D10."""
    if frame_id is not None and _HYB_CACHE["id"] == frame_id:
        _HYB_CACHE["hit"] = True
        return _HYB_CACHE["val"]
    t0 = time.perf_counter()
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
    g = cv2.GaussianBlur(clahe.apply(L), (5, 5), 0)
    A = frame.shape[0] * frame.shape[1]
    cands = _sticker_regions(g, 0.0004 * A, 0.02 * A, 25, 75, erode=1)
    _HYB_CACHE["id"], _HYB_CACHE["val"] = frame_id, cands
    _HYB_CACHE["ms"], _HYB_CACHE["hit"] = (time.perf_counter() - t0) * 1000.0, False
    return cands


def _quad_shape_ok(q):
    q = np.asarray(q, np.float32)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(sides) > 8 and max(sides) / max(min(sides), 1e-6) < 3.0


def _spacing_jump(cells, pts):
    """Largest / smallest gap between neighbouring stickers on the lattice.
    One real face changes spacing smoothly (perspective); a grid that spans
    two faces has a sudden jump at the fold."""
    gaps = []
    for (c, r), k in cells.items():
        for dc, dr in ((1, 0), (0, 1)):
            j = cells.get((c + dc, r + dr))
            if j is not None:
                gaps.append(float(np.linalg.norm(pts[j] - pts[k])))
    return max(gaps) / max(min(gaps), 1e-6) if len(gaps) >= 2 else 1.0


def _fit_lattices(pts, sizes, max_out=8, all_windows=False):
    """Every 3x3 sticker lattice the candidates support.
    Returns dicts: quad, count (stickers on the lattice), rms (fit error in
    pitches), jump (spacing consistency), cells {(col,row): candidate index}."""
    n = len(pts)
    if n < 4:
        return []
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    med = float(np.median(sizes))
    hyps = []
    for i in range(n):
        # neighbours: a wide range, because sticker size vs spacing differs a lot
        # between cubes (stickerless ~1.05x, black-bordered ~1.3-1.6x). Diagonal
        # grids are rejected below by a geometric test instead of by size.
        s = math.sqrt(float(sizes[i]) * med)
        order = [j for j in np.argsort(D[i]) if j != i and 0.85 * s < D[i, j] < 2.2 * s][:6]
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                u = pts[order[a]] - pts[i]
                v = pts[order[b]] - pts[i]
                lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                if not (0.6 < lu / lv < 1.67):
                    continue
                if abs(float(u @ v)) / (lu * lv) > 0.7:
                    continue
                hyps.append((i, u, v))
    found = []
    done = []          # (inlier set, u, v) of lattices already explored
    corners_g = np.float32([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]])
    for i, u, v in hyps:
        # the same lattice is found from every sticker on it - skip repeats
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
        # A grid built from DIAGONAL neighbours has real stickers sitting in the
        # middle of its cells (half-way between grid points). A real face never does.
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
            # D9: only the window with the most stickers. D10 also keeps the
            # neighbouring windows, so colours can say which 3x3 block is the face.
            if cnt != best and not (all_windows and cnt >= max(4, best - 3)):
                continue
            src = np.float32([[c, rr] for (c, rr) in cells])
            dst = np.float32([pts[k] for k in cells.values()])
            full_span = len({c for c, _ in cells}) == 3 and len({rr for _, rr in cells}) == 3
            if cnt >= 6 and full_span:
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
            quad = order_quad(np.float32(quad))
            if _quad_shape_ok(quad):
                found.append({"quad": quad, "count": cnt, "rms": rms, "cells": cells,
                              "jump": _spacing_jump(cells, pts), "pitch": pitch, "extra": cnt != best})
    found.sort(key=lambda f: (-f["count"], f["rms"]))
    kept = []
    for f in found:
        sz = face_size(f["quad"])
        if all(corner_error(f["quad"], k["quad"]) > 0.2 * sz for k in kept):
            kept.append(f)
        if len(kept) >= max_out:
            break
    return kept


def _squareness(q):
    q = np.asarray(q, np.float32)
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    ratio = min(sides) / max(max(sides), 1e-6)
    sines = []
    for i in range(4):
        a, b = q[(i - 1) % 4] - q[i], q[(i + 1) % 4] - q[i]
        sines.append(abs(float(np.cross(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    return ratio * min(sines)


OPPOSITE = {"White": "Yellow", "Yellow": "White", "Green": "Blue", "Blue": "Green",
            "Red": "Orange", "Orange": "Red"}


def belief_grids(grid, mode="true"):
    """(see below)"""
    if mode in ("auto",):
        mode = "true"
    return _belief_grids(grid, mode)


def _belief_grids(grid, mode="true"):
    """Faces the tracker could expect to see (D10 hint): the believed face in
    all 4 rotations, plus - for a solved face - every face one turn away.
    mode 'wrong' simulates a tracker that has NOT noticed a change."""
    if not grid or mode == "off" or any(g not in OPPOSITE for g in grid):
        return None
    grid = list(grid)
    if mode == "wrong":
        c = grid[4]
        wrong = [x for x in COLORS if x not in (c, OPPOSITE[c])][0]
        grid = [wrong] * 9
    out = []
    g = grid
    for _ in range(4):
        out.append(list(g))
        g = [g[6], g[3], g[0], g[7], g[4], g[1], g[8], g[5], g[2]]
    if len(set(grid)) == 1:
        c = grid[0]
        for nb in [x for x in COLORS if x not in (c, OPPOSITE[c])]:
            for strip in ([0, 1, 2], [6, 7, 8], [0, 3, 6], [2, 5, 8]):
                g = [c] * 9
                for k in strip:
                    g[k] = nb
                out.append(g)
    return out


class HybridLattice(Detector):
    key, short = "D9", "Hybrid lattice"
    name = "D9 Hybrid: YOLO region + sticker lattice"
    era = "proposed (V63 candidate)"
    desc = "YOLO region -> every 3x3 sticker lattice -> pick by fit/region/memory -> YOLO fallback"
    use_state = False
    # score = stickers (saturating: 6 is already convincing) + fit + region + memory + face-on
    #         - spacing jump (grid crossing a fold)   [+ state hint for D10]
    W_REGION, W_MEMORY, W_FRONT, W_FIT, W_JUMP = 2.0, 3.0, 2.0, 30.0, 6.0
    HOLD_FRAMES = 2

    def __init__(self, seg):
        self.seg = seg
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.available = True
        self.reason = "" if (seg and seg.ok) else "runs without YOLO (no seg weights)"
        self.reset()

    def reset(self):
        self.prev = None
        self.prev_age = 99
        self.last_id = None
        self.counts = collections.Counter()
        self.mem = None          # D10: colours of the face being tracked
        self.new_grid = None
        self.new_n = 0

    def _yolo(self, frame, ctx, d):
        if not (self.seg and self.seg.ok):
            return None, None
        res, dt, d.fresh = self.seg.run(frame, ctx["frame_id"])
        d.t_shared = dt
        d.stages["yolo inference" if d.fresh else "yolo (shared)"] = dt
        mask, _ = _seg_mask(res, frame.shape)
        if mask is None:
            return None, None
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask, None
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        yq = approx.reshape(4, 2).astype(np.float32) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(c))
        return mask, yq

    # ------------------------------------------------------------ D10 state
    # How a grid's colours are compared with the expected face:
    #   only stickers the classifier is sure about count (a finger is not a
    #   mismatch), at least 5 of them, best over the expected faces/rotations.
    LOCK_FRACTION = 0.78     # >= 7 of 9 matching: "this is the face we are tracking"
    W_STATE_SOFT, W_LOCK, W_NEAR = 1.0, 5.0, 1.5

    def _read(self, frame, quad, clf):
        _, patches = sample_cells(frame, quad)
        labels, conf = clf.classify(patches)
        sure = [lab != "Unknown" and cf >= 0.5 for lab, cf in zip(labels, conf)]
        return labels, sure

    @staticmethod
    def _match(labels, sure, belief):
        """(fraction, hits, is_current). belief[:4] = the face believed to be
        in front (4 rotations); the rest = faces one turn away. Only the
        current face can earn the full lock: a grid straddling the fold looks
        exactly like 'the face after one turn', so that only gets a smaller bonus."""
        n = sum(sure)
        if n < 5 or not belief:
            return None, 0, False
        best, cur = (-1, False), 0
        for i, g in enumerate(belief):
            h = sum(1 for a, b, ok in zip(labels, g, sure) if ok and a == b)
            if (h, i < 4) > best:
                best = (h, i < 4)
        return best[0] / float(n), best[0], best[1]

    def _belief(self, ctx):
        mode = ctx.get("belief_mode", "auto")
        if mode == "off":
            return None, "off"
        if mode in ("auto", "wrong") and ctx.get("belief"):
            return ctx["belief"], ("ground truth" if mode == "auto" else "WRONG face")
        if self.mem is None:
            return None, "learning"
        if mode == "wrong":
            c = self.mem[4]
            bad = [x for x in COLORS if x not in (c, OPPOSITE.get(c))][0]
            return belief_grids([bad] * 9, "true"), "WRONG face"
        return belief_grids(self.mem, "true"), "own memory"

    def _update_memory(self, labels, sure, matched_lock):
        """Stand-in for the move tracker: remember the face being tracked.
        A different face must be read clearly for several frames before the
        memory switches - so a real change is accepted, a glitch is not."""
        if sum(sure) < 8:
            return
        flat = list(labels)
        if self.mem is None or matched_lock:
            self.mem, self.new_grid, self.new_n = flat, None, 0
            return
        if self.new_grid is not None and sum(a == b for a, b in zip(flat, self.new_grid)) >= 8:
            self.new_n += 1
        else:
            self.new_grid, self.new_n = flat, 1
        if self.new_n >= 4:
            self.mem, self.new_grid, self.new_n = flat, None, 0
            self.counts["memory changed"] += 1

    def detect(self, frame, ctx):
        d = Detection()
        fid = ctx.get("frame_id")
        if self.last_id is None or fid is None or fid != self.last_id + 1:
            self.prev, self.prev_age = None, 99
        self.last_id = fid
        mask, yq = self._yolo(frame, ctx, d)
        d.mask = mask
        t = time.perf_counter()
        cands = _hybrid_stickers(frame, fid, self.clahe)
        d.stages["stickers"] = (time.perf_counter() - t) * 1000
        if _HYB_CACHE["hit"]:
            # another system already found the stickers on this frame: charge
            # their cost here too, so D9 and D10 are timed as if each ran alone
            d.t_extra = _HYB_CACHE["ms"]
            d.stages["stickers"] = _HYB_CACHE["ms"]
        t = time.perf_counter()
        # ---- region: YOLO mask (grown), else around the last face, else everything
        region = None
        if mask is not None:
            k = max(3, int(0.15 * math.sqrt(max(1, cv2.countNonZero(mask))))) | 1
            region = cv2.dilate(mask, np.ones((k, k), np.uint8))
        elif self.prev is not None and self.prev_age <= 5:
            region = np.zeros(frame.shape[:2], np.uint8)
            c = self.prev.mean(axis=0)
            big = (self.prev - c) * 1.8 + c
            cv2.fillPoly(region, [big.astype(np.int32)], 255)
        if region is not None:
            h, w = region.shape
            cands = [c for c in cands
                     if region[min(h - 1, max(0, int(c[1][1]))), min(w - 1, max(0, int(c[1][0])))] > 0]
        d.cands = [c[0] for c in cands]
        d.points = np.array([c[1] for c in cands], np.float32) if cands else None
        lattices = _fit_lattices(np.array([c[1] for c in cands], np.float32).reshape(-1, 2),
                                 np.array([c[2] for c in cands], np.float32),
                                 max_out=12 if self.use_state else 8,
                                 all_windows=self.use_state) if len(cands) >= 4 else []
        clf = ctx.get("clf")
        belief, bsrc = self._belief(ctx) if (self.use_state and clf is not None) else (None, "")
        d.stages["lattice fit"] = (time.perf_counter() - t) * 1000
        # ---- score every lattice
        t = time.perf_counter()
        best, best_s = None, -1e9
        for L in lattices:
            q = L["quad"]
            sz = face_size(q)
            cnt = L["count"]
            s = (min(cnt, 6) + 0.3 * max(0, cnt - 6) - self.W_FIT * L["rms"]
                 - self.W_JUMP * max(0.0, L["jump"] - 1.15) + self.W_FRONT * _squareness(q))
            if mask is not None:
                cen = cell_centres(q)
                h, w = mask.shape
                inside = sum(1 for (x, y) in cen if 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)] > 0)
                s += self.W_REGION * inside / 9.0
            mem = 0.0
            if self.prev is not None and self.prev_age <= 5:
                mem = math.exp(-(corner_error(q, self.prev) / max(sz, 1)) / 0.15)
                s += self.W_MEMORY * mem
            lock = False
            if self.use_state and clf is not None:
                L["read"] = self._read(frame, q, clf)
                frac, hits, current = self._match(*L["read"], belief)
                if frac is not None:
                    lock = frac >= self.LOCK_FRACTION and current
                    near = frac >= self.LOCK_FRACTION and not current
                    s += (self.W_STATE_SOFT * frac + (self.W_LOCK if lock else 0.0)
                          + (self.W_NEAR if near else 0.0))
                L["lock"] = lock
                L["hits"] = hits
            if L.get("extra") and not lock:
                continue     # shifted blocks only count when the colours prove they are the face
            enough = L["count"] >= 5 or (L["count"] == 4 and (mem > 0.5 or lock))
            if enough and s > best_s:
                best, best_s = L, s
        d.stages["choose"] = (time.perf_counter() - t) * 1000
        if best is not None:
            q = best["quad"]
            d.note = f"lattice {best['count']}/9 ({len(lattices)} grids)"
            self.counts["lattice"] += 1
            if self.use_state and "read" in best:
                d.note += f" | state: {bsrc}" + (f", match {best['hits']}" if belief else "")
                if best.get("lock"):
                    self.counts["state lock"] += 1
                if ctx.get("belief_mode", "auto") != "off":
                    self._update_memory(*best["read"], matched_lock=best.get("lock", False))
        elif yq is not None:
            q = yq
            d.note = "YOLO fallback"
            self.counts["yolo"] += 1
        elif self.prev is not None and self.prev_age < self.HOLD_FRAMES:
            q = self.prev
            d.note = f"held ({self.prev_age + 1})"
            self.counts["held"] += 1
            self.prev_age += 1
            d.quad = q.copy()
            return d
        else:
            d.note = f"no grid ({len(cands)} stickers)"
            self.counts["none"] += 1
            self.prev_age += 1
            return d
        # ---- light smoothing only while (almost) still, so turns are not lagged
        q = order_quad(np.float32(q))
        if self.prev is not None and self.prev_age <= 1:
            if corner_error(q, self.prev) < 0.05 * face_size(q):
                # line the corners up before averaging (the start corner can change)
                p = min((np.roll(self.prev, k, axis=0) for k in range(4)),
                        key=lambda r: float(np.linalg.norm(r - q, axis=1).sum()))
                q = 0.5 * q + 0.5 * p
        self.prev = order_quad(np.float32(q))
        self.prev_age = 0
        d.quad = q
        return d


class HybridLatticeState(HybridLattice):
    key, short = "D10", "Hybrid + state"
    name = "D10 Hybrid + state lock"
    desc = ("D9 + the colours of the face being tracked: the 3x3 block whose colours match it wins "
            "(>=7/9 = lock). Picks WHERE the face is; colours are still read from pixels")
    use_state = True


# =============================================================================
# D11 / D12: the real V63 pipeline (speed_lab.py) with its old and optimised
# settings. These run the WHOLE tracker per frame (face finder + colours +
# decoder), so their time is the true cost of one frame. With the optimised
# settings YOLO runs in a background thread every 4th frame, so its time is
# not on the frame path - exactly as in the live tracker.
# =============================================================================
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import speed_lab as _SL
except Exception as _exc:          # speed_lab.py not in this folder
    _SL, _SL_REASON = None, f"needs speed_lab.py next to cube_lab.py ({_exc.__class__.__name__})"
else:
    _SL_REASON = ""


class V63Pipeline(Detector):
    includes_colour = True         # the FPS test must not add C5 again

    def __init__(self, key, short, preset, seg_path):
        self.key, self.short, self.preset = key, short, preset
        self.name = f"{key} V63 pipeline ({preset})"
        self.era = "V63 code via speed_lab.py"
        self.desc = ("real V63 finder + colours + decoder, settings '" + preset + "'"
                     + (": YOLO every 4th frame in the background, finder on a crop, 3 lock reads, "
                        "fast colour, decoder every frame" if preset == "recommended" else ""))
        self.available = _SL is not None
        self.reason = _SL_REASON
        self.pipe = None
        self._c5_token = None
        if not self.available:
            return
        self.seg = _SL.Seg(seg_path)           # own model copy: the background thread must not share it
        if not self.seg.ok:
            self.available, self.reason = False, self.seg.reason
        opt = dict(_SL.DEFAULTS, **_SL.PIPE_PRESETS[preset])
        opt["camera"], opt["mirror"] = "top-down", 0
        if preset == "V63 now":
            opt["decoder_hz"] = 0      # time EVERY frame (V63 skips frames above 11 Hz, which would hide the cost)
        self.opt = opt
        self.reset()

    def reset(self):
        if self.available:
            self.pipe = _SL.Pipeline(self.opt, self.seg if self.seg.ok else None, None, live=True)
            self._c5_token = None

    def _sync_colours(self, clf):
        """Use the lab's current C5 calibration (rebuilt when it changes)."""
        if clf is None or getattr(clf, "key", "") != "C5" or not clf.ready:
            return
        token = id(clf.centres)
        if token != self._c5_token:
            self.pipe.c5 = clf
            self.pipe.fast = _SL.FastColour(clf)
            self._c5_token = token

    def detect(self, frame, ctx):
        d = Detection()
        if self.pipe is None:
            return d
        self._sync_colours(ctx.get("clf"))
        self.pipe.process(frame, time.perf_counter())
        q = self.pipe.quad
        d.quad = None if q is None else np.float32(q).reshape(4, 2)
        n = self.pipe.finder.last_note if hasattr(self.pipe.finder, "last_note") else ""
        d.note = f"{self.preset}: {n}" if n else self.preset
        for k in ("finder", "colour", "decoder", "flow", "yolo"):
            v = self.pipe.ms.get(k)
            if v:
                d.stages[k] = float(v[-1])
        return d


# =============================================================================
# Sticker sampling (identical for every system so only the QUAD differs)
# =============================================================================
WARP = 150
CELL = 50
MARGIN = 10


def sample_cells(frame, quad):
    dst = np.float32([[0, 0], [WARP - 1, 0], [WARP - 1, WARP - 1], [0, WARP - 1]])
    M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
    flat = cv2.warpPerspective(frame, M, (WARP, WARP))
    patches = []
    for r in range(3):
        for c in range(3):
            patches.append(flat[r * CELL + MARGIN:(r + 1) * CELL - MARGIN,
                                c * CELL + MARGIN:(c + 1) * CELL - MARGIN].copy())
    return flat, patches


def full_cells(flat):
    """Whole 50x50 cells (sticker + black border) - what the CNN was trained on."""
    return [flat[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] for r in range(3) for c in range(3)]


def cell_centres(quad):
    src = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    pts = np.float32([[c + 0.5, r + 0.5] for r in range(3) for c in range(3)]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def grid_lines(quad):
    src = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
    M = cv2.getPerspectiveTransform(src, np.asarray(quad, np.float32))
    lines = []
    for i in range(1, 3):
        for a, b in (((i, 0), (i, 3)), ((0, i), (3, i))):
            p = cv2.perspectiveTransform(np.float32([[a], [b]]), M).reshape(-1, 2)
            lines.append(p)
    return lines


# =============================================================================
# Colour classifiers
# =============================================================================
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


def lab_mean(patch):
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    valid = (lab[:, 0] > 25) & (lab[:, 0] < 245)
    return lab[valid].mean(0) if valid.sum() > 5 else lab.mean(0)


class HSVFixed(Classifier):
    key, short = "C1", "HSV fixed"
    name = "C1 HSV fixed ranges"
    desc = "hand-set HSV limits, colour covering most pixels (>15%)"
    RANGES = {
        "Red": [((0, 120, 70), (7, 255, 255)), ((165, 120, 70), (180, 255, 255))],
        "Orange": [((8, 120, 70), (20, 255, 255))],
        "Yellow": [((21, 70, 110), (36, 255, 255))],
        "Green": [((37, 70, 70), (85, 255, 255))],
        "Blue": [((86, 70, 70), (130, 255, 255))],
        "White": [((0, 0, 120), (180, 50, 255))],
    }

    def _counts(self, hsv, ranges, exclude=None):
        out = {}
        for name, rs in ranges.items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in rs:
                m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            if exclude is not None:
                m &= cv2.bitwise_not(exclude)
            out[name] = int(cv2.countNonZero(m))
        return out

    def classify(self, patches):
        labels, conf = [], []
        for p in patches:
            hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
            n = p.shape[0] * p.shape[1]
            c = self._counts(hsv, self.RANGES)
            best = max(c, key=c.get)
            frac = c[best] / float(n)
            if frac < 0.15:
                labels.append("Unknown"); conf.append(0.0)
            else:
                labels.append(best); conf.append(1.0)
        return labels, conf


class HSVSkin(HSVFixed):
    key, short = "C2", "HSV + skin"
    name = "C2 HSV + skin mask"
    desc = "C1 + skin/texture rejection + red/orange hue refinement"
    RANGES = {
        "Red": [((0, 150, 60), (6, 255, 255)), ((168, 150, 60), (180, 255, 255))],
        "Orange": [((7, 140, 60), (22, 255, 255))],
        "Yellow": [((23, 80, 100), (38, 255, 255))],
        "Green": [((39, 80, 60), (85, 255, 255))],
        "Blue": [((86, 80, 60), (125, 255, 255))],
        "White": [((0, 0, 150), (180, 60, 255))],
    }

    @staticmethod
    def skin(p):
        ycc = cv2.cvtColor(p, cv2.COLOR_BGR2YCrCb)
        return cv2.inRange(ycc, np.array([0, 130, 75]), np.array([255, 175, 130]))

    def classify(self, patches):
        labels, conf = [], []
        for p in patches:
            hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
            n = float(p.shape[0] * p.shape[1])
            white = cv2.countNonZero(cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))) / n
            if white > 0.5:
                labels.append("White"); conf.append(1.0); continue
            sk = self.skin(p)
            std = float(np.mean(np.std(p, axis=(0, 1))))
            lap = float(cv2.Laplacian(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if cv2.countNonZero(sk) / n > 0.35 or (std > 45 and lap > 220):
                labels.append("Unknown"); conf.append(0.0); continue
            c = self._counts(hsv, self.RANGES, exclude=sk)
            best = max(c, key=c.get)
            if c[best] / n < 0.15:
                labels.append("Unknown"); conf.append(0.0); continue
            if best in ("Red", "Orange"):
                hue = hsv[..., 0][(sk == 0) & (hsv[..., 0] > 0)]
                if hue.size:
                    mh = float(np.mean(hue))
                    if 7.5 <= mh <= 22:
                        best = "Orange"
                    elif mh < 7.5 or mh > 165:
                        best = "Red"
            labels.append(best); conf.append(1.0)
        return labels, conf


class LabFixed(Classifier):
    key, short = "C3", "LAB fixed refs"
    name = "C3 LAB fixed references"
    desc = "nearest hand-set LAB reference colour"
    REFS = {"White": (240, 128, 128), "Yellow": (210, 110, 180), "Red": (120, 180, 150),
            "Orange": (150, 170, 170), "Blue": (100, 150, 70), "Green": (130, 80, 150)}

    def classify(self, patches):
        labels, conf = [], []
        for p in patches:
            m = lab_mean(p)
            d = {k: float(np.linalg.norm(m - np.array(v))) for k, v in self.REFS.items()}
            best = min(d, key=d.get)
            labels.append(best); conf.append(1.0)
        return labels, conf


def weighted_dist(a, b):
    return float(math.sqrt(((a[0] - b[0]) * 0.3) ** 2 + ((a[1] - b[1]) * 1.3) ** 2 + ((a[2] - b[2]) * 1.3) ** 2))


class KMeansLab(Classifier):
    key, short = "C4", "K-means LAB"
    name = "C4 K-means LAB (calibrated)"
    desc = "6-face scan -> k-means clusters labelled by centre sticker"
    needs_calibration = True

    def __init__(self):
        self.centres = None
        self.reason = "press K to scan 6 solved faces"

    @property
    def ready(self):
        return self.centres is not None

    def calibrate(self, scans):
        """scans: {colour: [9 LAB means]} (each face solved, centre known)."""
        X = np.vstack([np.asarray(scans[c], float) for c in COLORS])
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X)
            labels, centres = km.labels_, km.cluster_centers_
        except Exception:
            labels, centres = self._kmeans(X)
        self.centres = {}
        for i, c in enumerate(COLORS):
            self.centres[c] = centres[labels[i * 9 + 4]]   # cluster of that face's centre sticker
        self.reason = ""

    @staticmethod
    def _kmeans(X, k=6, it=30):
        rng = np.random.default_rng(42)
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(it):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
            C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        return lab, C

    def classify(self, patches):
        labels, conf = [], []
        for p in patches:
            m = lab_mean(p)
            d = {k: weighted_dist(m, v) for k, v in self.centres.items()}
            labels.append(min(d, key=d.get)); conf.append(1.0)
        return labels, conf


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


class CNNClassifier(Classifier):
    key, short = "C6", "CNN"
    name = "C6 CNN (Jay Hack)"
    desc = "HackRubiksNet 6-layer CNN on 32x32 sticker crops"
    CLASSES = ["background", "blue", "green", "orange", "red", "white", "yellow"]
    uses_full_cells = True

    def __init__(self, path):
        self.net = None
        self.available = False
        self.reason = "no rubiks_model.pth"
        if not path or not os.path.exists(path):
            return
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class HackRubiksNet(nn.Module):
                def __init__(self, num_classes=7):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.pool1 = nn.MaxPool2d(2, 2)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool2 = nn.MaxPool2d(2, 2)
                    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                    self.pool3 = nn.MaxPool2d(2, 2)
                    self.flatten = nn.Flatten()
                    self.fc1 = nn.Linear(128 * 4 * 4, 256)
                    self.dropout = nn.Dropout(0.5)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, num_classes)

                def forward(self, x):
                    x = self.pool1(F.relu(self.conv1(x)))
                    x = self.pool2(F.relu(self.conv2(x)))
                    x = self.pool3(F.relu(self.conv3(x)))
                    x = self.flatten(x)
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = F.relu(self.fc2(x))
                    return self.fc3(x)

            self.torch = torch
            self.net = HackRubiksNet()
            self.net.load_state_dict(torch.load(path, map_location="cpu"))
            self.net.eval()
            self.mean = np.array([0.485, 0.456, 0.406], np.float32)
            self.std = np.array([0.229, 0.224, 0.225], np.float32)
            self.available = True
            self.reason = ""
        except Exception as exc:
            self.reason = f"CNN: {exc.__class__.__name__}"

    def classify(self, patches):
        batch = []
        for p in patches:
            rgb = cv2.cvtColor(cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
            batch.append(x.transpose(2, 0, 1))
        with self.torch.no_grad():
            out = self.net(self.torch.from_numpy(np.stack(batch)))
            pr = self.torch.softmax(out, 1).numpy()
        labels, conf = [], []
        for row in pr:
            i = int(np.argmax(row))
            name = self.CLASSES[i]
            labels.append("Unknown" if name == "background" else name.capitalize())
            conf.append(float(row[i]))
        return labels, conf


# =============================================================================
# Metrics
# =============================================================================
class DetMetrics:
    def __init__(self):
        self.frames = 0
        self.found = 0
        self.lat = LatStats()
        self.lat_own = LatStats()
        self.iou = LatStats()
        self.cerr = LatStats()       # corner error in % of face size
        self.jitter = LatStats()     # frame-to-frame corner motion, % of face size
        self.prev = None
        self.ref_kind = ""

    def update(self, det, ref_quad, ref_kind, still):
        self.frames += 1
        self.lat.add(det.t_total)
        self.lat_own.add(det.t_own)
        if det.found:
            self.found += 1
            if ref_quad is not None:
                self.ref_kind = ref_kind
                self.iou.add(quad_iou(det.quad, ref_quad))
                ce = corner_error(det.quad, ref_quad)
                self.cerr.add(100.0 * ce / face_size(ref_quad))
            if self.prev is not None and still:
                ce = corner_error(det.quad, self.prev)
                self.jitter.add(100.0 * ce / face_size(det.quad))
            self.prev = det.quad
        else:
            if ref_quad is not None:
                self.iou.add(0.0)
            self.prev = None

    def summary(self):
        return {"frames": self.frames, "found": self.found,
                "detect_rate_%": pct(self.found, self.frames),
                "latency_ms": self.lat.summary(), "own_ms": self.lat_own.summary(),
                "iou_mean": self.iou.mean(), "iou_vs": self.ref_kind,
                "corner_err_%": self.cerr.median(), "jitter_%": self.jitter.median()}


class ColMetrics:
    def __init__(self):
        self.frames = 0          # frames with ground truth
        self.not_found = 0       # frames with GT where the detector failed
        self.stickers = 0
        self.correct = 0
        self.unsure = 0          # Unknown, or confidence < 0.5
        self.conf_wrong = 0      # wrong AND confident (these corrupt a tracker)
        self.conf = np.zeros((6, 7), np.int64)    # gt x pred (+Unknown)
        self.flicker = 0
        self.flicker_n = 0
        self.lat = LatStats()
        self.prev = None

    def update(self, labels, confs, gt, t_ms, same_scene):
        self.lat.add(t_ms)
        if same_scene and self.prev is not None and labels is not None:
            self.flicker += sum(1 for a, b in zip(labels, self.prev) if a != b)
            self.flicker_n += 9
        self.prev = labels
        if gt is None:
            return
        self.frames += 1
        if labels is None:
            self.not_found += 1
            return
        for lab, cf, g in zip(labels, confs, gt):
            self.stickers += 1
            gi = COLORS.index(g)
            pi = COLORS.index(lab) if lab in COLORS else 6
            self.conf[gi, pi] += 1
            if lab == "Unknown" or cf < 0.5:
                self.unsure += 1
                if lab == g:
                    self.correct += 1
                continue
            if lab == g:
                self.correct += 1
            else:
                self.conf_wrong += 1

    def accuracy(self):
        return pct(self.correct, self.stickers)

    def end_to_end(self):
        """% of stickers right counting frames where the face was not found as all wrong."""
        return pct(self.correct, self.stickers + 9 * self.not_found)

    def summary(self):
        ro = self.conf[COLORS.index("Red"), COLORS.index("Orange")] + self.conf[COLORS.index("Orange"), COLORS.index("Red")]
        rr = self.conf[COLORS.index("Red")].sum() + self.conf[COLORS.index("Orange")].sum()
        return {"frames_gt": self.frames, "stickers": self.stickers,
                "accuracy_%": self.accuracy(), "end_to_end_%": self.end_to_end(),
                "unsure_%": pct(self.unsure, self.stickers),
                "confident_wrong_%": pct(self.conf_wrong, self.stickers),
                "red_orange_confusion_%": pct(ro, rr),
                "flicker_%": pct(self.flicker, self.flicker_n),
                "latency_ms": self.lat.summary(),
                "per_colour_acc_%": {c: pct(self.conf[i, i], self.conf[i].sum()) for i, c in enumerate(COLORS)}}


class Book:
    """All metrics for one test condition (e.g. 'bright', 'dim', 'glare')."""

    def __init__(self, condition):
        self.condition = condition
        self.created = time.time()
        self.det = collections.defaultdict(DetMetrics)
        self.col = collections.defaultdict(ColMetrics)      # (det_key, clf_key)
        self.frames = 0
        self.frame_ms = LatStats()
        self.perturb = set()

    def summary(self):
        return {"condition": self.condition, "frames": self.frames,
                "frame_ms": self.frame_ms.summary(),
                "perturbations": sorted(self.perturb),
                "detectors": {k: v.summary() for k, v in self.det.items()},
                "colour": {f"{d}|{c}": v.summary() for (d, c), v in self.col.items()}}


# =============================================================================
# TRACKER LAB - compares move-tracking strategies on simulated camera input
# =============================================================================
# The cube is a 54-sticker array in Kociemba order (U R F D L B, 9 each, reading
# order). Moves are permutations built from 3D sticker coordinates, so they are
# exact by construction (verified against the kociemba solver in self_test()).
FACE_ORDER = "URFDLB"
_FACE_VEC = {   # normal, right, down (reading direction seen from outside)
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "R": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "F": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}
# (axis, layers, direction): direction -1 = clockwise seen from the +axis side
MOVE_SPECS = {
    "R": (0, [1], -1), "L": (0, [-1], 1), "U": (1, [1], -1), "D": (1, [-1], 1),
    "F": (2, [1], -1), "B": (2, [-1], 1),
    "M": (0, [0], 1), "E": (1, [0], 1), "S": (2, [0], -1),
    "x": (0, [-1, 0, 1], -1), "y": (1, [-1, 0, 1], -1), "z": (2, [-1, 0, 1], -1),
}
FRONT = slice(18, 27)


def _rot90(axis, direction):
    c, s = 0, direction  # cos(±90)=0, sin(±90)=±1
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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


# ---------------------------------------------------------------------------
class TrackerBase:
    name = "tracker"
    short = "?"
    desc = ""

    def __init__(self, model, moves):
        self.model = model
        self.moves = moves

    def reset(self, state):
        self.state = state.copy()

    @staticmethod
    def mismatches(front_states, obs):
        known = obs >= 0
        return ((front_states != obs) & known).sum(axis=-1)

    def result(self):
        return self.state


class RuleTracker(TrackerBase):
    name = "T1 Rule-based row/column change"
    short = "Rule-based"
    desc = "only U/D/L/R single turns, exact match (first prototype)"

    def __init__(self, model, moves):
        super().__init__(model, [m for m in moves if m[0] in "UDLR" and not m.endswith("2")])
        self.P = np.stack([model.perms[m] for m in self.moves])

    def observe(self, obs):
        if self.mismatches(self.state[FRONT], obs) == 0:
            return
        cand = self.state[self.P][:, FRONT]
        mm = self.mismatches(cand, obs)
        i = int(np.argmin(mm))
        if mm[i] == 0:
            self.state = self.state[self.P[i]]


class SingleMatchTracker(TrackerBase):
    name = "T2 Predict & match, single move"
    short = "Single match"
    desc = "try every legal move, commit the best if <=2 stickers disagree"

    def __init__(self, model, moves):
        super().__init__(model, moves)
        self.P = np.stack([model.perms[m] for m in moves])

    def observe(self, obs):
        if self.mismatches(self.state[FRONT], obs) <= 1:
            return
        mm = self.mismatches(self.state[self.P][:, FRONT], obs)
        i = int(np.argmin(mm))
        if mm[i] <= 2:
            self.state = self.state[self.P[i]]


class GreedyBFSTracker(TrackerBase):
    name = "T3 Multi-move BFS, commit immediately"
    short = "Greedy BFS"
    desc = "search up to 3 moves, commit the best explanation at once (D4 era)"

    def __init__(self, model, moves, depth=3):
        super().__init__(model, moves)
        self.P, self.seqs, self.lens = model.composites(moves, depth)

    def observe(self, obs):
        if self.mismatches(self.state[FRONT], obs) <= 1:
            return
        mm = self.mismatches(self.state[self.P][:, FRONT], obs)
        score = mm * 100 + self.lens
        i = int(np.argmin(score))
        if mm[i] <= 1:
            self.state = self.state[self.P[i]]


class BeamTracker(TrackerBase):
    name = "T4 Beam of hypotheses, delayed decision"
    short = "Beam (current)"
    desc = "keep the best K whole-cube states, score = moves + mismatches (V51+)"

    def __init__(self, model, moves, width=48, depth=2, mismatch_cost=2.5, move_cost=1.0):
        super().__init__(model, moves)
        self.P, self.seqs, self.lens = model.composites(moves, depth)
        self.width, self.mc, self.mv = width, mismatch_cost, move_cost

    def reset(self, state):
        self.beam = [(0.0, state.copy())]

    def observe(self, obs):
        states, scores = [], []
        for sc, st in self.beam:
            S = st[self.P]
            mm = self.mismatches(S[:, FRONT], obs)
            states.append(S)
            scores.append(sc + self.mv * self.lens + self.mc * mm)
        S = np.concatenate(states)
        sc = np.concatenate(scores)
        k = min(len(sc), self.width * 4)
        idx = np.argpartition(sc, k - 1)[:k]
        idx = idx[np.argsort(sc[idx])]
        beam, seen = [], set()
        for i in idx:
            key = S[i].tobytes()
            if key in seen:
                continue
            seen.add(key)
            beam.append((float(sc[i]), S[i]))
            if len(beam) >= self.width:
                break
        base = beam[0][0]
        self.beam = [(s - base, st) for s, st in beam]

    def result(self):
        return self.beam[0][1]


TRACKERS = [RuleTracker, SingleMatchTracker, GreedyBFSTracker, BeamTracker]


def move_weights(moves, kind):
    """'solve-like': R/U turns dominate real solves, B and slices are rare."""
    if kind != "solve-like":
        return None
    w = {"R": 3.0, "U": 3.0, "L": 1.5, "F": 1.5, "D": 1.2, "B": 0.5,
         "M": 0.4, "E": 0.15, "S": 0.15, "x": 0.3, "y": 0.6, "z": 0.15}
    return [w.get(m[0], 1.0) for m in moves]


def simulate(model, moves, n_moves, p_skip, noise, unknown, rng, weights=None):
    st = model.solved.copy()
    truths, obs = [], []
    for k in range(n_moves):
        mv = rng.choices(moves, weights)[0] if weights else rng.choice(moves)
        st = model.apply(st, mv)
        if rng.random() >= p_skip or k == n_moves - 1:
            o = st[FRONT].astype(np.int16).copy()
            for i in range(9):
                r = rng.random()
                if r < noise:
                    o[i] = rng.choice([c for c in range(6) if c != o[i]])
                elif r < noise + unknown:
                    o[i] = -1
            truths.append(st.copy())
            obs.append(o)
    return truths, obs


def tracker_sweep(quick=True, seed=7):
    """Generator: yields (fraction_done, rows). rows are dicts for charts/CSV."""
    model = CubeModel()
    rng = random.Random(seed)
    trials = 10 if quick else 40
    n_moves = 20 if quick else 30
    noises = [0.0, 0.03, 0.08, 0.15] if quick else [0.0, 0.02, 0.05, 0.1, 0.15, 0.25]
    skips = [0.0, 0.4] if quick else [0.0, 0.3, 0.6]
    movesets = ["faces", "solve-like", "full"]
    built = {}
    for ms in movesets:
        mv = model.moveset("faces" if ms == "faces" else "full")
        built[ms] = [T(model, mv) for T in TRACKERS]
    configs = [(ms, sk, nz) for ms in movesets for sk in skips for nz in noises]
    rows = []
    for ci, (ms, sk, nz) in enumerate(configs):
        mv = model.moveset("faces" if ms == "faces" else "full")
        wts = move_weights(mv, ms)
        acc = {t.short: [0, 0, 0, 0, 0, 0.0] for t in built[ms]}  # final, track, front, n_obs, trials, ms
        for _ in range(trials):
            truths, obs = simulate(model, mv, n_moves, sk, nz, 0.02, rng, wts)
            for tr in built[ms]:
                tr.reset(model.solved)
                t0 = time.perf_counter()
                good = front = 0
                for truth, o in zip(truths, obs):
                    tr.observe(o)
                    r = tr.result()
                    good += int(np.array_equal(r, truth))
                    front += int(np.array_equal(r[FRONT], truth[FRONT]))
                a = acc[tr.short]
                a[0] += int(np.array_equal(tr.result(), truths[-1]))
                a[1] += good
                a[2] += front
                a[3] += len(obs)
                a[4] += 1
                a[5] += (time.perf_counter() - t0) * 1000.0
        for tname, a in acc.items():
            rows.append({"moveset": ms, "skip": sk, "noise": nz, "tracker": tname,
                         "final_state_correct_%": 100.0 * a[0] / a[4],
                         "tracking_accuracy_%": 100.0 * a[1] / a[3],
                         "front_face_correct_%": 100.0 * a[2] / a[3],
                         "ms_per_observation": a[5] / a[3]})
        yield (ci + 1) / len(configs), rows


# =============================================================================
# Charts (matplotlib, also used for the in-app chart views)
# =============================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.max_open_warning"] = 0
    return plt


def fig_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    _plt().close(fig)
    return img


def cond_colours(n):
    """A distinct colour per condition (10 -> tab10, up to 20 -> tab20, more -> spread hsv)."""
    plt = _plt()
    if n <= 10:
        cm = plt.get_cmap("tab10")
        return [cm(i) for i in range(n)]
    if n <= 20:
        cm = plt.get_cmap("tab20")
        order = list(range(0, 20, 2)) + list(range(1, 20, 2))   # strong colours first
        return [cm(order[i]) for i in range(n)]
    cm = plt.get_cmap("hsv")
    return [cm(i / float(n)) for i in range(n)]


def legend_below(fig, ax, n):
    """Condition legend under the plots, so it never covers a bar."""
    for a in fig.axes:
        lo, hi = a.get_ylim()
        a.set_ylim(0, hi if hi > 0 else 1)
    handles, labels = ax.get_legend_handles_labels()
    ncol = min(n, 6)
    rows = int(math.ceil(n / float(ncol)))
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.035 + 0.055 * rows, 1, 1))


def chart_fps(rows, path, cam_fps):
    """Frames per second you would actually get from each face system (+ C5
    colours): its own speed, but never more than the camera delivers."""
    plt = _plt()
    names = [f"{r['system']} {r['name']}" for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(rows) + 1.8), dpi=150)
    top = cam_fps * 1.35
    for off, lab, col, tcol in ((-0.2, "no cube", "#B9C0CC", "#5B6478"), (0.2, "cube", "#2F6DB5", "#1F2430")):
        vals = [r[f"{lab} fps"] for r in rows]
        could = [r[f"{lab} could fps"] for r in rows]
        ax.barh(y + off, vals, 0.38, color=col, label=("no cube in view" if lab == "no cube" else "cube in view"))
        for yi, v, c in zip(y, vals, could):
            txt = f"{v:.0f}" if c <= cam_fps + 0.5 else f"{v:.0f}  (could do {c:.0f})"
            ax.text(v + top * 0.01, yi + off, txt, va="center", fontsize=8, color=tcol)
    ax.axvline(cam_fps, color="#B53A3A", lw=1.2, ls="--")
    ax.text(cam_fps, -0.75, f"camera {cam_fps:.0f} fps", color="#B53A3A", fontsize=8.5, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(len(rows) - 0.4, -1.0)
    ax.set_xlabel("frames per second you get (face system + C5 colours, capped by the camera)")
    ax.set_xlim(0, top * 1.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_tracker(rows, moveset):
    plt = _plt()
    sel = [r for r in rows if r["moveset"] == moveset]
    skips = sorted(set(r["skip"] for r in sel))
    trackers = list(dict.fromkeys(r["tracker"] for r in sel))
    fig, axes = plt.subplots(1, len(skips) + 1, figsize=(4.2 * (len(skips) + 1), 3.6), dpi=100)
    for ax, sk in zip(axes, skips):
        for t in trackers:
            pts = sorted((r["noise"], r["tracking_accuracy_%"]) for r in sel if r["skip"] == sk and r["tracker"] == t)
            ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], marker="o", label=t)
        ax.set_title(f"{moveset}: {int(sk * 100)}% of moves unseen", fontsize=10)
        ax.set_xlabel("sticker misread rate (%)")
        ax.set_ylabel("whole-cube state correct (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
    ax = axes[-1]
    ms = [np.mean([r["ms_per_observation"] for r in sel if r["tracker"] == t]) for t in trackers]
    ax.bar(range(len(trackers)), ms, color="#5b8def")
    ax.set_xticks(range(len(trackers)))
    ax.set_xticklabels(trackers, rotation=20, fontsize=8)
    ax.set_ylabel("ms per observation")
    ax.set_title("cost", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def chart_sweep(sweep, param):
    plt = _plt()
    d = sweep["data"][param]
    lv = d["levels"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), dpi=100)
    for k, v in d["clf_cw"].items():
        axes[2].plot(lv, v, marker="o", label=k)
    axes[2].set_title(f"confident AND wrong vs {param} (bad)", fontsize=10)
    axes[2].set_ylabel("stickers (%)")
    for k, v in d["clf_acc"].items():
        axes[0].plot(lv, v, marker="o", label=k)
    axes[0].set_title(f"colour accuracy vs {param}", fontsize=10)
    axes[0].set_ylabel("stickers correct (%)")
    axes[0].set_ylim(0, 102)
    for k, v in d["det_found"].items():
        axes[1].plot(lv, v, marker="o", label=k)
    axes[1].set_title(f"face found (IoU>0.5) vs {param}", fontsize=10)
    axes[1].set_ylabel("frames (%)")
    axes[1].set_ylim(0, 102)
    for a in axes:
        a.set_xlabel(param)
        a.grid(alpha=0.3)
        a.legend(fontsize=6)
    fig.tight_layout()
    return fig


# =============================================================================
# The app
# =============================================================================
TABS = ["Side by side", "Focus", "Colour lab", "Stats", "Tracker lab", "Help", "Batch"]

# Batch test: every condition below is applied to the SAME captured frames.
BATCH_FILE = "batch_conditions.json"
BATCH_DEFAULT = {
    "frames_per_face": 50,         # frames kept per solved face
    "capture_every": 3,            # keep every Nth camera frame (spreads them over ~5 s per face)
    "calibrate_per_condition": True,
    "conditions": [
        {"name": "normal", "on": True, "settings": {}},
        {"name": "bright", "on": True, "settings": {"brightness": 60}},
        {"name": "dim", "on": True, "settings": {"brightness": -60}},
        {"name": "warm light", "on": True, "settings": {"temperature": 60}},
        {"name": "cool light", "on": True, "settings": {"temperature": -60}},
        {"name": "glare", "on": True, "settings": {"glare": 0.6}},
        {"name": "fingers", "on": True, "settings": {"fingers": 4}},
        {"name": "fast turns", "on": True, "settings": {"motion_blur": 20}},
        {"name": "washed out", "on": False, "settings": {"desaturate": 0.4, "contrast": 0.7}},
        {"name": "dark + noisy", "on": False, "settings": {"brightness": -50, "noise": 15}},
    ],
}
# Only these change what the colours look like, so only these are applied to the
# calibration scan ("calibrated in that light"). Fingers, blur, glare spots etc.
# are not things you would calibrate under.
LIGHTING_KEYS = ("brightness", "contrast", "gamma", "temperature", "desaturate")


def load_batch_config(path=BATCH_FILE):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                cfg = json.load(fh)
            out = copy.deepcopy(BATCH_DEFAULT)
            out.update(cfg)
            for c in out.get("conditions", []):       # older files used a second name for the same test
                if c.get("name") == "fast turns (blur)":
                    c["name"] = "fast turns"
            return out
        except Exception as exc:
            print(f"[lab] could not read {path}: {exc} - using defaults")
    cfg = copy.deepcopy(BATCH_DEFAULT)
    try:
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"[lab] wrote default batch conditions to {path} (edit it to change conditions)")
    except Exception:
        pass
    return cfg


def condition_values(settings):
    v = {sp[0]: sp[1] for sp in Perturb.SPECS}
    v.update(settings or {})
    return v
OVERLAYS = [("grid", "Grid lines", True), ("outline", "Face outline", True),
            ("mask", "Mask / what it sees", False), ("cands", "Candidates / contours", True),
            ("labels", "Colour labels", True), ("inset", "Warped face inset", True),
            ("gt", "Ground-truth outline", True), ("timing", "Timing text", True)]
# Manual conditions (c cycles them): the ones that need different camera shots.
# Lighting, glare, fingers and fast turns are done by the Batch test instead.
CONDITION_PRESETS = ["normal", "far away", "close up", "cluttered background"]
SCAN_ORDER = ["White", "Yellow", "Green", "Blue", "Red", "Orange"]
GT_MODES = ["off", "solved face", "manual grid", "synthetic"]


class App:
    def __init__(self, args):
        self.args = args
        self.headless = args.headless
        # ---- source
        self.source = self._make_source(args)
        self.mirror = isinstance(self.source, CameraSource)
        # ---- models
        seg_path = args.seg or first_existing(CONFIG["seg_weights"])
        det_path = args.det or first_existing(CONFIG["det_weights"])
        pose_path = args.pose or first_existing(CONFIG["pose_weights"])
        cnn_path = args.cnn or first_existing(CONFIG["cnn_weights"])
        self.seg = YoloRunner(seg_path, "segmentation")
        self.det_model = YoloRunner(det_path, "box") if det_path else None
        self.pose = YoloRunner(pose_path, "pose") if pose_path else None
        self.detectors = [StickerContours(), FixedBox(), DarkBackgroundRect(), GridRebuild(),
                          YoloBox(self.seg, self.det_model), YoloSegRect(self.seg),
                          YoloSegQuad(self.seg), YoloPose(self.pose),
                          HybridLattice(self.seg), HybridLatticeState(self.seg),
                          V63Pipeline("D11", "V63 (old settings)", "V63 now", seg_path),
                          V63Pipeline("D12", "V63 optimised", "recommended", seg_path)]
        self.dmap = {d.key: d for d in self.detectors}
        self.classifiers = [HSVFixed(), HSVSkin(), LabFixed(), KMeansLab(), CalibratedSoft(),
                            CNNClassifier(cnn_path)]
        self.cmap = {c.key: c for c in self.classifiers}
        calib = args.calib or CONFIG["calibration_file"]
        if os.path.exists(calib):
            self.cmap["C5"].load(calib)
        self.en_det = {d.key: d.available for d in self.detectors}
        self.en_clf = {c.key: c.available for c in self.classifiers}
        self.overlay = {k: v for k, _, v in OVERLAYS}
        self.perturb = Perturb()
        # ---- state
        self.tab = 0
        self.focus = "D7" if self.dmap["D7"].available else "D1"
        self.tile_clf = "C5"
        self.col_src = "GT" if self.source.gt_available else ("D7" if self.dmap["D7"].available else "D1")
        self.sel_clf = "C5"
        self.frozen = False
        self.step_once = False
        self.frame_id = 0
        self.raw = None
        self.frame = None
        self.gt_quad = None
        self.src_grid = None
        self.results = {}
        self.gt_mode = "synthetic" if self.source.gt_available else "off"
        self.gt_face = "White"
        self.manual_grid = None
        self.d10_belief = "auto"     # D10's knowledge of the tracked face: auto / self / wrong / off
        # ---- batch test
        self.batch_cfg = load_batch_config()
        self.batch = None            # capture state machine
        self.batch_gen = None
        self.batch_progress = 0.0
        self.batch_frames = []       # [(raw frame, solved colour, face start?)]
        self.scan_frames = {}        # colour -> (raw frame, quad) from the calibration scan
        self.prev_gt = None
        self.prev_ref = None
        self.condition = "normal"
        self.books = {}
        self.book = self._book(self.condition)
        self.view_book = None
        self.bench_left = 0
        self.bench_results = []
        self.fpstest = None          # FPS test state (key a)
        self.logging = False
        self.log_fh = None
        self.log_w = None
        self.recorder = None
        self.popouts = False
        self.popout_names = set()
        self.text_mode = None
        self.text_buf = ""
        self.scan = None
        self.scan_data = {}
        self.sweep = None
        self.sweep_gen = None
        self.sweep_progress = 0.0
        self.sweep_param = None
        self.tracker_rows = None
        self.tracker_gen = None
        self.tracker_progress = 0.0
        self.tracker_view = "solve-like"
        self.chart_cache = {}
        self.msg = ""
        self.msg_t = 0
        self.fps = LatStats(60)
        self.last_t = time.perf_counter()
        self.buttons = []
        self.W, self.H = CONFIG["window_w"], CONFIG["window_h"]
        self.session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(args.out or CONFIG["output_dir"], self.session)
        if isinstance(self.source, SyntheticSource):
            self.auto_calibrate_synthetic()

    # ------------------------------------------------------------------ setup
    def _make_source(self, args):
        s = args.source
        if s == "synthetic":
            return SyntheticSource(seed=args.seed, solved=args.solved, dark_bg=getattr(args, "dark_bg", False),
                                   angled=getattr(args, "angled", False))
        if s and os.path.isdir(s):
            return FolderSource(s)
        if s and os.path.isfile(s):
            return VideoSource(s)
        return CameraSource(args.camera)

    def _book(self, cond):
        if cond not in self.books:
            self.books[cond] = Book(cond)
        return self.books[cond]

    def say(self, text, secs=4.0):
        self.msg, self.msg_t = text, time.time() + secs
        print("[lab]", text)

    def ensure_out(self, sub=""):
        p = os.path.join(self.out_dir, sub)
        os.makedirs(p, exist_ok=True)
        return p

    # ------------------------------------------------------------ ground truth
    def gt_grid(self):
        if self.gt_mode == "synthetic":
            return self.src_grid
        if self.gt_mode == "solved face":
            return [self.gt_face] * 9
        if self.gt_mode == "manual grid":
            return self.manual_grid
        return None

    # ------------------------------------------------------------ processing
    def run_systems(self, frame, frame_id, gt_quad=None, dets=None, clfs=None, belief_grid="current"):
        if belief_grid == "current":
            belief_grid = self.gt_grid()
        clf = self.cmap["C5"] if self.cmap["C5"].ready else self.cmap["C1"]
        ctx = {"frame_id": frame_id, "clf": clf, "belief_mode": self.d10_belief,
               "belief": belief_grids(belief_grid, self.d10_belief) if self.d10_belief in ("auto", "wrong") else None}
        out = {}
        dets = dets if dets is not None else [d for d in self.detectors if self.en_det[d.key] and d.available]
        clfs = clfs if clfs is not None else [c for c in self.classifiers if self.en_clf[c.key] and c.ready]
        for d in dets:
            det = d(frame, ctx)
            r = {"det": det, "flat": None, "patches": None, "cls": {}}
            if det.found:
                t = time.perf_counter()
                r["flat"], r["patches"] = sample_cells(frame, det.quad)
                det.stages["warp+sample"] = (time.perf_counter() - t) * 1000
            out[d.key] = r
        if gt_quad is not None:
            g = Detection()
            g.quad = order_quad(gt_quad)
            g.found = True
            flat, patches = sample_cells(frame, g.quad)
            out["GT"] = {"det": g, "flat": flat, "patches": patches, "cls": {}}
        for c in clfs:
            for k, r in out.items():
                if r["patches"] is None:
                    continue
                t = time.perf_counter()
                inp = full_cells(r["flat"]) if getattr(c, "uses_full_cells", False) else r["patches"]
                labels, conf = c.classify(inp)
                r["cls"][c.key] = (labels, conf, (time.perf_counter() - t) * 1000)
        return out

    def _account(self, book, res, gt, gt_quad, st):
        """Add one processed frame to a condition's numbers. st keeps the
        previous reference outline / ground truth (for jitter and flicker)."""
        book.frames += 1
        ref, kind = None, ""
        if gt_quad is not None:
            ref, kind = order_quad(gt_quad), "ground truth"
        elif "D7" in res and res["D7"]["det"].found:
            ref, kind = res["D7"]["det"].quad, "D7 current"
        still = True
        if ref is not None and st.get("prev_ref") is not None:
            still = corner_error(ref, st["prev_ref"]) < 0.03 * face_size(ref)
        same_scene = still and (gt == st.get("prev_gt"))
        for k, r in res.items():
            if k == "GT":
                continue
            rq = None if (kind == "D7 current" and k == "D7") else ref
            book.det[k].update(r["det"], rq, kind, still)
        for k, r in res.items():
            for c in self.classifiers:
                if not (self.en_clf[c.key] and c.ready):
                    continue
                if c.key in r["cls"]:
                    labels, conf, ms = r["cls"][c.key]
                    book.col[(k, c.key)].update(labels, conf, gt, ms, same_scene)
                else:
                    book.col[(k, c.key)].update(None, None, gt, None, same_scene)
        st["prev_ref"] = ref
        st["prev_gt"] = list(gt) if gt else None

    def process(self):
        t_frame = time.perf_counter()
        frame = self.perturb.apply(self.raw, focus_quad=self.gt_quad if self.gt_quad is not None else self.prev_ref)
        self.frame = frame
        self.frame_id += 1
        if self.batch is not None and self.batch.get("state") in ("wait", "rec"):
            # batch capture: only a light live view (YOLO + D9) and no stats
            light = [self.dmap[k] for k in ("D9", "D7") if self.dmap[k].available]
            self.results = self.run_systems(frame, self.frame_id, self.gt_quad, dets=light[:1],
                                            clfs=[self.cmap["C5"]] if self.cmap["C5"].ready else [])
            self.batch_tick()
            return
        res = self.run_systems(frame, self.frame_id, self.gt_quad)
        self.results = res
        if self.fpstest is not None:
            self.fps_tick(res)
        if self.tab == 6:
            return          # Batch tab: live view only, so no stray frames land in a condition
        gt = self.gt_grid()
        book = self.book
        if self.perturb.active():
            book.perturb.add(self.perturb.describe())
        st = {"prev_ref": self.prev_ref, "prev_gt": self.prev_gt}
        self._account(book, res, gt, self.gt_quad, st)
        self.prev_ref, self.prev_gt = st["prev_ref"], st["prev_gt"]
        book.frame_ms.add((time.perf_counter() - t_frame) * 1000)
        if self.logging:
            self.log_frame(res, gt)
        if self.bench_left > 0:
            self.bench_left -= 1
            if self.bench_left == 0:
                self.finish_benchmark()

    # ------------------------------------------------------------ logging
    def log_frame(self, res, gt):
        if self.log_fh is None:
            path = os.path.join(self.ensure_out(), "frames.csv")
            self.log_fh = open(path, "w", newline="")
            self.log_w = csv.writer(self.log_fh)
            self.log_w.writerow(["frame", "time", "condition", "perturbation", "system", "kind",
                                 "found", "latency_ms", "own_ms", "iou_ref", "classifier",
                                 "stickers_correct", "unsure", "labels", "ground_truth"])
        now = time.time()
        ref = order_quad(self.gt_quad) if self.gt_quad is not None else (
            res["D7"]["det"].quad if "D7" in res and res["D7"]["det"].found else None)
        for k, r in res.items():
            d = r["det"]
            iou = quad_iou(d.quad, ref) if (d.found and ref is not None) else ""
            self.log_w.writerow([self.frame_id, f"{now:.3f}", self.condition, self.perturb.describe(), k,
                                 "detector", int(d.found), f"{d.t_total:.2f}", f"{d.t_own:.2f}",
                                 f"{iou:.3f}" if iou != "" else "", "", "", "", "", ""])
            for ck, (labels, conf, ms) in r["cls"].items():
                corr = sum(1 for a, b in zip(labels, gt) if a == b) if gt else ""
                uns = sum(1 for a, cf in zip(labels, conf) if a == "Unknown" or cf < 0.5)
                self.log_w.writerow([self.frame_id, f"{now:.3f}", self.condition, self.perturb.describe(), k,
                                     "colour", 1, f"{ms:.2f}", "", "", ck, corr, uns,
                                     "".join(LETTER[x] for x in labels),
                                     "".join(LETTER[x] for x in gt) if gt else ""])

    def toggle_logging(self):
        self.logging = not self.logging
        if not self.logging and self.log_fh:
            self.log_fh.close()
            self.log_fh = None
            self.say("frame log closed: " + os.path.join(self.out_dir, "frames.csv"))
        elif self.logging:
            self.say("logging every frame to frames.csv")

    # ------------------------------------------------------------ actions
    def set_condition(self, cond):
        self.condition = cond
        self.book = self._book(cond)
        self.view_book = None
        self.say(f"condition: {cond} (stats are kept separately per condition)")

    def cycle_condition(self):
        names = CONDITION_PRESETS + [c for c in self.books if c not in CONDITION_PRESETS]
        i = names.index(self.condition) if self.condition in names else -1
        self.set_condition(names[(i + 1) % len(names)])

    def cycle_gt(self):
        modes = [m for m in GT_MODES if m != "synthetic" or self.source.gt_available]
        i = modes.index(self.gt_mode) if self.gt_mode in modes else 0
        self.gt_mode = modes[(i + 1) % len(modes)]
        if self.gt_mode == "manual grid" and not self.manual_grid:
            self.start_text("manual_gt")
        self.say(f"ground truth: {self.gt_mode}" + (f" ({self.gt_face})" if self.gt_mode == "solved face" else ""))

    def reset_stats(self):
        self.books[self.condition] = Book(self.condition)
        self.book = self.books[self.condition]
        self.say(f"stats reset for '{self.condition}'")

    def cycle_d10_belief(self):
        modes = ["auto", "self", "wrong", "off"]
        self.d10_belief = modes[(modes.index(self.d10_belief) + 1) % len(modes)] if self.d10_belief in modes else "auto"
        for d in self.detectors:
            if d.key == "D10":
                d.mem = None
        self.say({"auto": "D10 state: ground-truth face if GT is set, otherwise D10 learns the face it is tracking",
                  "self": "D10 state: learns the tracked face itself (like the real tracker would supply)",
                  "wrong": "D10 state: deliberately WRONG face (tracker missed a change) - must not be worse than D9",
                  "off": "D10 state off (D10 = D9 with extra grid windows)"}[self.d10_belief])

    # ------------------------------------------------------------ FPS test
    FPS_FRAMES = 90          # frames recorded per phase
    FPS_COUNTDOWN = 4.0      # seconds to move the cube out of / into view

    def start_fps_test(self):
        """Key a: average frames per second of every enabled face system (with
        C5 colours), first with NO cube in view, then with the cube in view."""
        if self.fpstest is not None:
            self.fpstest = None
            self.say("FPS test cancelled")
            return
        self.frozen = False
        self.fpstest = {"phase": "wait", "label": "no cube", "until": time.time() + self.FPS_COUNTDOWN,
                        "data": {"no cube": collections.defaultdict(list), "cube": collections.defaultdict(list)},
                        "frame": collections.defaultdict(list), "left": 0}
        self.say(f"FPS test: take the cube OUT of view - recording in {self.FPS_COUNTDOWN:.0f} s", self.FPS_COUNTDOWN)

    def fps_tick(self, res):
        ft = self.fpstest
        now = time.time()
        if ft["phase"] == "wait":
            left = ft["until"] - now
            if left > 0:
                verb = "OUT of" if ft["label"] == "no cube" else "INTO"
                self.msg, self.msg_t = f"FPS test: move the cube {verb} view - recording in {left:.0f} s", now + 1
                return
            ft["phase"], ft["left"] = "rec", self.FPS_FRAMES
        if ft["phase"] != "rec":
            return
        frame_ms = 0.0
        for key, r in res.items():
            if key == "GT":
                continue
            det = r["det"]
            ms = det.t_total + det.stages.get("warp+sample", 0.0)
            if "C5" in r["cls"] and not getattr(self.dmap.get(key), "includes_colour", False):
                ms += r["cls"]["C5"][2]
            ft["data"][ft["label"]][key].append(ms)
            frame_ms += det.t_own + (det.t_shared if det.fresh else 0.0)
        ft["frame"][ft["label"]].append(frame_ms)
        ft["left"] -= 1
        self.msg, self.msg_t = f"FPS test: recording '{ft['label']}' ({ft['left']} frames left)", now + 1
        if ft["left"] <= 0:
            if ft["label"] == "no cube":
                ft.update(phase="wait", label="cube", until=now + self.FPS_COUNTDOWN)
            else:
                self.finish_fps_test()

    def camera_fps(self):
        """Frames per second the source delivers (camera / video), 30 if unknown."""
        fps = 0.0
        cap = getattr(self.source, "cap", None)
        if cap is not None:
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS))
            except Exception:
                fps = 0.0
        if not fps:
            fps = float(getattr(self.source, "fps", 0) or 0)
        return fps if 1.0 < fps <= 240.0 else 30.0

    def finish_fps_test(self):
        ft, self.fpstest = self.fpstest, None
        keys = [d.key for d in self.detectors if ft["data"]["cube"].get(d.key) or ft["data"]["no cube"].get(d.key)]
        cam = self.camera_fps()
        rows = []
        for k in keys:
            row = {"system": k, "name": self.dmap[k].short}
            for lab in ("no cube", "cube"):
                v = ft["data"][lab].get(k, [])
                ms = float(np.median(v)) if v else float("nan")      # median: one-off hiccups do not count
                could = 1000.0 / ms if v and ms > 0 else 0.0
                row[f"{lab} ms"] = round(ms, 2)
                row[f"{lab} could fps"] = round(could, 1)            # if the camera were infinitely fast
                row[f"{lab} fps"] = round(min(could, cam), 1)        # what you actually get
            rows.append(row)
        stamp = datetime.datetime.now().strftime("%H%M%S")
        out = self.ensure_out()
        with open(os.path.join(out, f"fps_test_{stamp}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        path = os.path.join(out, f"fps_test_{stamp}.png")
        chart_fps(rows, path, cam)
        self.fps_chart = path
        self.say(f"FPS test done -> {path}", 8)
        if not self.headless:
            img = cv2.imread(path)
            if img is not None:
                cv2.imshow("FPS test", img)

    def start_benchmark(self):
        self.reset_stats()
        self.bench_left = int(self.args.bench_frames or CONFIG["benchmark_frames"])
        self.frozen = False
        self.say(f"benchmark: {self.bench_left} frames under '{self.condition}'")

    def finish_benchmark(self):
        s = self.book.summary()
        s["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.bench_results.append(s)
        p = os.path.join(self.ensure_out(), f"benchmark_{len(self.bench_results)}_{self.condition.replace(' ', '_')}.json")
        with open(p, "w") as fh:
            json.dump(s, fh, indent=1, default=float)
        self.say(f"benchmark done -> {p}")

    def screenshot(self, canvas):
        p = os.path.join(self.ensure_out("shots"), f"{TABS[self.tab].replace(' ', '_')}_{self.frame_id}.png")
        cv2.imwrite(p, canvas)
        self.say("saved " + p)

    def save_tiles(self):
        d = self.ensure_out(f"shots/frame_{self.frame_id}")
        if self.raw is not None:
            cv2.imwrite(os.path.join(d, "raw.png"), self.raw)
            cv2.imwrite(os.path.join(d, "input_after_perturbation.png"), self.frame)
        h, w = self.frame.shape[:2]
        for k, r in self.results.items():
            if k == "GT":
                continue
            tile = self.render_tile(k, w, h)
            cv2.imwrite(os.path.join(d, f"{k}_{self.dmap[k].short.replace(' ', '_')}.png"), tile)
            if r["flat"] is not None:
                cv2.imwrite(os.path.join(d, f"{k}_warped_face.png"), cv2.resize(r["flat"], (300, 300), interpolation=cv2.INTER_NEAREST))
            if r["det"].mask is not None:
                cv2.imwrite(os.path.join(d, f"{k}_mask.png"), r["det"].mask)
        self.say("saved full-resolution tiles to " + d)

    def toggle_record(self, canvas):
        if self.recorder is None:
            p = os.path.join(self.ensure_out("video"), f"lab_{self.frame_id}.mp4")
            self.recorder = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 15, (canvas.shape[1], canvas.shape[0]))
            self.say("recording -> " + p)
        else:
            self.recorder.release()
            self.recorder = None
            self.say("recording stopped")

    def toggle_popouts(self):
        self.popouts = not self.popouts
        if not self.popouts:
            for n in list(self.popout_names):
                try:
                    cv2.destroyWindow(n)
                except Exception:
                    pass
            self.popout_names.clear()
        self.say("separate camera windows " + ("on" if self.popouts else "off"))

    # ------------------------------------------------------------ calibration
    def colour_source_patches(self):
        order = ["GT", self.col_src, "D7", "D6", "D4", "D1", "D3", "D2"]
        for k in order:
            r = self.results.get(k)
            if r and r["patches"] is not None:
                return k, r["patches"]
        return None, None

    def start_scan(self):
        if isinstance(self.source, SyntheticSource):
            self.auto_calibrate_synthetic()
            return
        self.scan = 0
        self.scan_data = {}
        self.say(f"calibration: show the SOLVED {SCAN_ORDER[0]} face, press K")

    def scan_capture(self):
        k, patches = self.colour_source_patches()
        if patches is None:
            self.say("no face found - cannot capture")
            return
        colour = SCAN_ORDER[self.scan]
        self.scan_data[colour] = [lab_mean(p) for p in patches]
        quad = self.gt_quad if k == "GT" else self.results[k]["det"].quad
        if self.raw is not None and quad is not None:
            self.scan_frames[colour] = (self.raw.copy(), np.float32(quad))   # for per-condition calibration
        self.scan += 1
        if self.scan >= 6:
            self.finish_scan()
        else:
            self.say(f"captured {colour} (from {k}). Now show the SOLVED {SCAN_ORDER[self.scan]} face, press K")

    def finish_scan(self):
        self.cmap["C4"].calibrate(self.scan_data)
        self.cmap["C5"].calibrate(self.scan_data)
        p = os.path.join(self.ensure_out(), "lab_colour_calibration.json")
        with open(p, "w") as fh:
            json.dump(self.cmap["C5"].to_json(), fh)
        self.scan = None
        self.en_clf["C4"] = self.en_clf["C5"] = True
        if len(self.scan_frames) == 6:
            d = self.ensure_out("capture")
            meta = {}
            for col, (img, q) in self.scan_frames.items():
                cv2.imwrite(os.path.join(d, f"scan_{col}.png"), img)
                meta[col] = q.tolist()
            with open(os.path.join(d, "scan.json"), "w") as fh:
                json.dump(meta, fh)
        self.say("calibrated C4 + C5 from the scan (saved lab_colour_calibration.json)")

    def auto_calibrate_synthetic(self):
        scene = SyntheticScene(seed=99, clutter=False, vary_light=False)
        data = {}
        for col in COLORS:
            scene.randomize_pose()
            img, q, _ = scene.render([col] * 9)
            _, patches = sample_cells(img, q)
            # same LAB means as the tracker's scan (C5 uses robust means internally)
            data[col] = [lab_mean(p) for p in patches]
        self.cmap["C4"].calibrate(data)
        self.cmap["C5"].calibrate(data)
        self.en_clf["C4"] = self.en_clf["C5"] = True
        self.say("synthetic source: C4/C5 auto-calibrated from 6 rendered solved faces")

    # ------------------------------------------------------------ robustness sweep
    def start_sweep(self):
        if not isinstance(self.source, SyntheticSource):
            if self.raw is None or self.gt_grid() is None:
                self.say("sweep needs ground truth: use --source synthetic, or freeze a frame with GT mode 'solved face'/'manual grid'")
                return
        self.sweep_gen = self.sweep_iter()
        self.sweep_progress = 0.0
        self.say("robustness sweep running...")

    def sweep_iter(self):
        synthetic = isinstance(self.source, SyntheticSource)
        params = list(Perturb.SPECS)
        dets = [d for d in self.detectors if self.en_det[d.key] and d.available]
        clfs = [c for c in self.classifiers if self.en_clf[c.key] and c.ready]
        rng = np.random.default_rng(5)
        scene = SyntheticScene(seed=123) if synthetic else None
        base_raw = None if synthetic else self.raw.copy()
        base_gt = None if synthetic else list(self.gt_grid())
        base_quad = None
        if not synthetic:
            r0 = self.run_systems(base_raw, -1, None, dets=[self.dmap["D7"]] if self.dmap["D7"].available else dets[:1], clfs=[])
            q = next((r["det"].quad for r in r0.values() if r["det"].found), None)
            base_quad = q
        n = CONFIG["sweep_samples_synthetic"] if synthetic else CONFIG["sweep_repeats_frozen"]
        data = {}
        total = sum(len(set([float(sp[1])] + [float(v) for v in np.linspace(sp[2], sp[3], 6)])) for sp in params)
        done = 0
        fid = 10_000_000
        for name, default, lo, hi, step, _ in params:
            levels = sorted(set([float(default)] + [float(x) for x in np.linspace(lo, hi, 6)]))
            D = {"levels": levels, "det_found": {d.short: [] for d in dets},
                 "clf_acc": {c.short: [] for c in clfs}, "clf_cw": {c.short: []for c in clfs}}
            for lv in levels:
                found = collections.Counter()
                corr = collections.Counter()
                cw = collections.Counter()
                tot = 0
                for s in range(n):
                    if synthetic:
                        scene.randomize_pose()
                        scene._new_pattern()
                        if s % 4 == 0:
                            scene._new_background()
                        raw, q, gt = scene.render()
                    else:
                        raw, q, gt = base_raw, base_quad, base_gt
                    vals = {sp[0]: sp[1] for sp in Perturb.SPECS}
                    vals[name] = lv
                    img = self.perturb.apply(raw, focus_quad=q, values=vals, rng=rng)
                    fid += 1
                    for d in dets:   # every sweep sample is an unrelated picture
                        if hasattr(d, "reset"):
                            d.reset()
                    res = self.run_systems(img, fid, q, dets=dets, clfs=clfs, belief_grid=gt)
                    for d in dets:
                        dd = res[d.key]["det"]
                        if dd.found and (q is None or quad_iou(dd.quad, q) > 0.5):
                            found[d.short] += 1
                    if "GT" in res:
                        for c in clfs:
                            labels, conf, _ = res["GT"]["cls"][c.key]
                            for a, cf, g in zip(labels, conf, gt):
                                corr[c.short] += int(a == g)
                                cw[c.short] += int(a != g and a != "Unknown" and cf >= 0.5)
                        tot += 9
                for d in dets:
                    D["det_found"][d.short].append(100.0 * found[d.short] / n)
                for c in clfs:
                    D["clf_acc"][c.short].append(100.0 * corr[c.short] / tot if tot else float("nan"))
                    D["clf_cw"][c.short].append(100.0 * cw[c.short] / tot if tot else float("nan"))
                done += 1
                yield min(0.999, done / float(total))
            data[name] = D
            self.sweep = {"source": "synthetic" if synthetic else "frozen frame", "samples": n, "data": data}
        self.sweep_param = self.sweep_param or "brightness"
        self.chart_cache = {k: v for k, v in self.chart_cache.items() if not k.startswith("sweep")}
        yield 1.0

    def start_tracker(self, quick=True):
        self.tracker_gen = tracker_sweep(quick=quick)
        self.tracker_progress = 0.0
        self.say("tracker simulation running (" + ("quick" if quick else "full") + ")...")

    def background_work(self, budget_ms=60):
        t0 = time.perf_counter()
        while self.batch_gen and (time.perf_counter() - t0) * 1000 < budget_ms:
            try:
                self.batch_progress = next(self.batch_gen)
            except StopIteration:
                self.batch_gen = None
                self.say("batch analysis finished - report exported", 10)
        while self.sweep_gen and (time.perf_counter() - t0) * 1000 < budget_ms:
            try:
                self.sweep_progress = next(self.sweep_gen)
            except StopIteration:
                self.sweep_gen = None
                self.sweep_progress = 1.0
                self.say("robustness sweep finished - see Colour lab / export")
        while self.tracker_gen and (time.perf_counter() - t0) * 1000 < budget_ms:
            try:
                self.tracker_progress, self.tracker_rows = next(self.tracker_gen)
                self.chart_cache.pop("tracker", None)
            except StopIteration:
                self.tracker_gen = None
                self.say("tracker simulation finished")

    # ------------------------------------------------------------ batch test
    def batch_conditions(self):
        return [c for c in self.batch_cfg["conditions"] if c.get("on", True)]

    def reload_batch_config(self):
        self.batch_cfg = load_batch_config()
        self.say(f"batch conditions reloaded from {BATCH_FILE}: " +
                 ", ".join(c["name"] for c in self.batch_conditions()))

    def calibration_for(self, settings):
        """C4/C5 calibrated on the scan frames seen under this condition's
        lighting. None -> keep the normal calibration."""
        if not self.batch_cfg.get("calibrate_per_condition", True) or len(self.scan_frames) < 6:
            return None
        light = {k: v for k, v in (settings or {}).items() if k in LIGHTING_KEYS}
        if not light:
            return None
        vals = condition_values(light)
        data = {}
        for col, (img, q) in self.scan_frames.items():
            frame = self.perturb.apply(img, focus_quad=q, values=vals, rng=np.random.default_rng(0))
            _, patches = sample_cells(frame, q)
            data[col] = [lab_mean(p) for p in patches]
        c4, c5 = copy.deepcopy(self.cmap["C4"]), copy.deepcopy(self.cmap["C5"])
        c4.calibrate(data)
        c5.calibrate(data)
        return {"C4": c4, "C5": c5}

    def _swap_classifiers(self, calib):
        """Temporarily use per-condition C4/C5. Returns what to restore."""
        saved = (list(self.classifiers), dict(self.cmap))
        if calib:
            self.classifiers = [calib.get(c.key, c) for c in self.classifiers]
            self.cmap = {c.key: c for c in self.classifiers}
        return saved

    def _restore_classifiers(self, saved):
        self.classifiers, self.cmap = list(saved[0]), dict(saved[1])

    # ---- step 2: capture the six solved faces once
    def start_batch_capture(self):
        if isinstance(self.source, SyntheticSource):
            self.say("batch capture is for the real camera (synthetic already has perfect ground truth)")
            return
        if not (self.cmap["C5"].ready):
            self.say("calibrate first (k, six solved faces) - the batch needs C5")
            return
        self.batch = {"state": "wait", "face": 0, "n": 0, "tick": 0}
        self.batch_frames = []
        self.frozen = False
        self.tab = 6
        self.say(f"BATCH: show the SOLVED {SCAN_ORDER[0]} face, then press U (or click Capture)", 30)

    def batch_go(self):
        if self.batch is None:
            self.start_batch_capture()
            return
        if self.batch["state"] == "wait":
            self.batch.update(state="rec", n=0, tick=0)
            self.say(f"recording {SCAN_ORDER[self.batch['face']]} - tilt the face slowly, keep it in view", 30)

    def batch_tick(self):
        b = self.batch
        if b["state"] != "rec" or self.raw is None:
            return
        b["tick"] += 1
        if b["tick"] % max(1, int(self.batch_cfg["capture_every"])) != 1 and int(self.batch_cfg["capture_every"]) > 1:
            return
        colour = SCAN_ORDER[b["face"]]
        self.batch_frames.append((self.raw.copy(), colour, b["n"] == 0))
        b["n"] += 1
        if b["n"] >= int(self.batch_cfg["frames_per_face"]):
            b["face"] += 1
            if b["face"] >= 6:
                b["state"] = "saving"
                self.finish_batch_capture()
            else:
                b["state"] = "wait"
                self.say(f"done. Now show the SOLVED {SCAN_ORDER[b['face']]} face, then press U", 30)

    def finish_batch_capture(self):
        d = self.ensure_out("capture")
        rows = []
        for i, (img, col, first) in enumerate(self.batch_frames):
            name = f"frame_{i:04d}_{col}.jpg"
            cv2.imwrite(os.path.join(d, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"file": name, "solved_colour": col, "face_start": bool(first)})
        with open(os.path.join(d, "capture.json"), "w") as fh:
            json.dump({"frames": rows, "mirrored": bool(self.mirror)}, fh, indent=1)
        self.say(f"captured {len(rows)} frames (saved to capture/). Analysing every condition...", 10)
        self.start_batch_analysis()

    def load_batch_capture(self, folder):
        """Re-analyse a saved capture (--batch-replay)."""
        with open(os.path.join(folder, "capture.json")) as fh:
            meta = json.load(fh)
        self.batch_frames = [(cv2.imread(os.path.join(folder, r["file"])), r["solved_colour"], r["face_start"])
                             for r in meta["frames"]]
        sp = os.path.join(folder, "scan.json")
        if os.path.exists(sp):
            with open(sp) as fh:
                quads = json.load(fh)
            scans = {}
            for col, q in quads.items():
                img = cv2.imread(os.path.join(folder, f"scan_{col}.png"))
                if img is not None:
                    self.scan_frames[col] = (img, np.float32(q))
                    _, patches = sample_cells(img, np.float32(q))
                    scans[col] = [lab_mean(p) for p in patches]
            if len(scans) == 6:
                self.cmap["C4"].calibrate(scans)
                self.cmap["C5"].calibrate(scans)
                self.en_clf["C4"] = self.en_clf["C5"] = True
        print(f"[lab] loaded {len(self.batch_frames)} captured frames from {folder}")

    # ---- step 3: analyse every condition on the same frames
    def start_batch_analysis(self):
        if not self.batch_frames:
            self.say("nothing captured yet - press U in the Batch tab")
            return
        self.batch = {"state": "analysing"}
        self.frozen = True
        self.batch_gen = self.batch_iter()
        self.batch_progress = 0.0

    def batch_iter(self):
        conds = self.batch_conditions()
        frames = self.batch_frames
        total = len(conds) * len(frames) + len(frames)
        done = 0
        fid = 50_000_000
        # pre-pass: where is the cube in each (clean) frame - glare and fingers are drawn there
        focus = []
        ref_det = next((self.dmap[k] for k in ("D9", "D7", "D4") if self.dmap[k].available), None)
        for raw, col, first in frames:
            fid += 1
            if first and hasattr(ref_det, "reset"):
                ref_det.reset()
            dd = ref_det(raw, {"frame_id": fid, "clf": None, "belief_mode": "off", "belief": None})
            focus.append(dd.quad if dd.found else (focus[-1] if focus else None))
            done += 1
            yield done / total
        for cond in conds:
            name, settings = cond["name"], cond.get("settings", {})
            vals = condition_values(settings)
            calib = self.calibration_for(settings)
            saved = self._swap_classifiers(calib)
            try:
                book = Book(name)
                self.books[name] = book
                desc = ", ".join(f"{k}={v}" for k, v in settings.items()) or "none"
                book.perturb.add(desc + (" | calibrated in this lighting" if calib else ""))
                st = {}
                for i, (raw, col, first) in enumerate(frames):
                    if first:
                        for d in self.detectors:
                            if hasattr(d, "reset"):
                                d.reset()
                        st = {}
                    t0 = time.perf_counter()
                    rng = np.random.default_rng(i * 7919 + len(name))
                    img = self.perturb.apply(raw, focus_quad=focus[i], values=vals, rng=rng)
                    fid += 1
                    gt = [col] * 9
                    res = self.run_systems(img, fid, None, belief_grid=gt)
                    self._account(book, res, gt, None, st)
                    book.frame_ms.add((time.perf_counter() - t0) * 1000)
                    if i == len(frames) // 2:
                        self.frame, self.results = img, res      # something to look at meanwhile
                    done += 1
                    yield done / total
                s = book.summary()
                s["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
                s["batch"] = True
                self.bench_results.append(s)
                with open(os.path.join(self.ensure_out(), f"benchmark_batch_{name.replace(' ', '_')}.json"), "w") as fh:
                    json.dump(s, fh, indent=1, default=float)
            finally:
                self._restore_classifiers(saved)
        for d in self.detectors:
            if hasattr(d, "reset"):
                d.reset()
        self.batch = {"state": "done"}
        self.frozen = False
        self.export()
        yield 1.0

    # ---- one screenshot per condition for the current frame
    def screenshot_conditions(self):
        if self.raw is None:
            return
        conds = self.batch_conditions()
        keep = (self.frame, self.results, dict(self.perturb.values))
        d = self.ensure_out(f"shots/frame_{self.frame_id}_conditions")
        focus = self.gt_quad if self.gt_quad is not None else self.prev_ref
        for j, cond in enumerate(conds):
            vals = condition_values(cond.get("settings", {}))
            calib = self.calibration_for(cond.get("settings", {}))
            saved = self._swap_classifiers(calib)
            try:
                img = self.perturb.apply(self.raw, focus_quad=focus, values=vals, rng=np.random.default_rng(j))
                self.frame = img
                self.results = self.run_systems(img, 60_000_000 + self.frame_id * 100 + j, self.gt_quad)
                self.perturb.values = vals
                cond_save = self.condition
                self.condition = cond["name"]
                canvas = self.render()
                self.condition = cond_save
                cv2.imwrite(os.path.join(d, f"{TABS[self.tab].replace(' ', '_')}_{cond['name'].replace(' ', '_')}.png"), canvas)
            finally:
                self._restore_classifiers(saved)
        self.frame, self.results, self.perturb.values = keep[0], keep[1], keep[2]
        for det in self.detectors:          # their memory saw unrelated pictures
            if hasattr(det, "reset"):
                det.reset()
        self.say(f"saved {len(conds)} condition screenshots -> {d}")

    # ------------------------------------------------------------ export
    def export(self):
        out = self.ensure_out()
        plt = _plt()
        # a condition with only a few frames is almost always an accident (cycling
        # past it with c while the camera runs) - leave it out of the report
        MIN_FRAMES = 20
        skipped = [b.condition for b in self.books.values() if 0 < b.frames < MIN_FRAMES]
        books = [b for b in self.books.values() if b.frames >= MIN_FRAMES]
        if skipped:
            self.say("left out of the report (under 20 frames): " + ", ".join(skipped), 8)
        summ = {"session": self.session, "source": self.source.name,
                "seg_weights": self.seg.path, "calibration": self.cmap["C5"].source,
                "conditions": [b.summary() for b in books],
                "benchmarks": self.bench_results,
                "sweep": self.sweep, "tracker": self.tracker_rows}
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        # ---- detectors.csv
        with open(os.path.join(out, "detectors.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["condition", "system", "name", "frames", "detect_rate_%", "mean_ms", "median_ms", "p95_ms",
                        "own_ms_mean", "iou_mean", "iou_vs", "corner_err_%", "jitter_%"])
            for b in books:
                for k, m in sorted(b.det.items()):
                    s = m.summary()
                    w.writerow([b.condition, k, self.dmap[k].name, s["frames"], fmt(s["detect_rate_%"], 2),
                                fmt(s["latency_ms"]["mean"], 2), fmt(s["latency_ms"]["median"], 2),
                                fmt(s["latency_ms"]["p95"], 2), fmt(s["own_ms"]["mean"], 2),
                                fmt(s["iou_mean"], 3), s["iou_vs"], fmt(s["corner_err_%"], 2), fmt(s["jitter_%"], 2)])
        # ---- colour.csv (every detector x classifier pair)
        with open(os.path.join(out, "colour.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["condition", "face_source", "classifier", "stickers", "accuracy_%", "end_to_end_%",
                        "unsure_%", "confident_wrong_%", "red_orange_confusion_%", "flicker_%", "mean_ms"]
                       + [f"acc_{c}_%" for c in COLORS])
            for b in books:
                for (dk, ck), m in sorted(b.col.items()):
                    s = m.summary()
                    w.writerow([b.condition, dk, self.cmap[ck].name, s["stickers"], fmt(s["accuracy_%"], 2),
                                fmt(s["end_to_end_%"], 2), fmt(s["unsure_%"], 2), fmt(s["confident_wrong_%"], 2),
                                fmt(s["red_orange_confusion_%"], 2), fmt(s["flicker_%"], 2),
                                fmt(s["latency_ms"]["mean"], 3)] + [fmt(s["per_colour_acc_%"][c], 1) for c in COLORS])
        charts = []
        dk_all = [d.key for d in self.detectors if any(d.key in b.det for b in books)]
        names = {d.key: d.short for d in self.detectors}
        if books and dk_all:
            # latency stacked (shared model time + own post-processing), p95 marker
            fig, ax = plt.subplots(figsize=(9, 3.8), dpi=110)
            b0 = max(books, key=lambda b: b.frames)
            own = [b0.det[k].lat_own.mean() if k in b0.det else 0 for k in dk_all]
            tot = [b0.det[k].lat.mean() if k in b0.det else 0 for k in dk_all]
            p95 = [b0.det[k].lat.p95() if k in b0.det else 0 for k in dk_all]
            x = np.arange(len(dk_all))
            ax.bar(x, np.array(tot) - np.array(own), color="#9db4e8", label="shared YOLO inference")
            ax.bar(x, own, bottom=np.array(tot) - np.array(own), color="#3d6bd6", label="system's own processing")
            ax.scatter(x, p95, color="black", marker="_", s=300, label="95th percentile", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([names[k] for k in dk_all], rotation=20, fontsize=8)
            ax.set_ylabel("ms per frame")
            ax.set_title(f"Face isolation time per frame ({b0.condition}, {b0.frames} frames)")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            charts.append(("latency.png", "Detection time per system", fig))
            for metric, title, ylabel, fn in [
                ("detect", "Face found (% of frames)", "%", lambda m: pct(m.found, m.frames)),
                ("iou", "Overlap with reference outline (IoU)", "IoU", lambda m: m.iou.mean()),
                ("jitter", "Corner jitter while still (% of face size, lower is better)", "%", lambda m: m.jitter.median()),
            ]:
                cols = cond_colours(len(books))
                width = max(9.0, 1.5 + 0.11 * len(dk_all) * len(books))
                fig, ax = plt.subplots(figsize=(width, 4.6), dpi=110)
                wbar = 0.8 / max(1, len(books))
                for i, b in enumerate(books):
                    vals = [fn(b.det[k]) if k in b.det else np.nan for k in dk_all]
                    ax.bar(np.arange(len(dk_all)) + i * wbar, vals, wbar, label=b.condition, color=cols[i])
                ax.set_xticks(np.arange(len(dk_all)) + wbar * (len(books) - 1) / 2)
                ax.set_xticklabels([names[k] for k in dk_all], rotation=20, fontsize=8)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(axis="y", alpha=0.3)
                legend_below(fig, ax, len(books))
                charts.append((f"{metric}.png", title, fig))
            # colour accuracy on the colour source
            src = self.col_src
            ck_all = [c.key for c in self.classifiers if any((src, c.key) in b.col and b.col[(src, c.key)].stickers for b in books)]
            if ck_all:
                cols = cond_colours(len(books))
                width = max(13.0, 2.0 + 3 * 0.075 * len(ck_all) * len(books))
                fig, axes = plt.subplots(1, 3, figsize=(width, 4.8), dpi=110)
                wbar = 0.8 / max(1, len(books))
                for ax, (title, fn) in zip(axes, [("sticker accuracy %", lambda m: m.accuracy()),
                                                  ("confident & wrong % (bad)", lambda m: pct(m.conf_wrong, m.stickers)),
                                                  ("unsure % (handed to decoder)", lambda m: pct(m.unsure, m.stickers))]):
                    for i, b in enumerate(books):
                        vals = [fn(b.col[(src, k)]) if (src, k) in b.col else np.nan for k in ck_all]
                        ax.bar(np.arange(len(ck_all)) + i * wbar, vals, wbar, label=b.condition, color=cols[i])
                    ax.set_xticks(np.arange(len(ck_all)) + wbar * (len(books) - 1) / 2)
                    ax.set_xticklabels([self.cmap[k].short for k in ck_all], rotation=25, fontsize=8)
                    ax.set_title(title, fontsize=10)
                    ax.grid(axis="y", alpha=0.3)
                fig.suptitle(f"Colour classification on faces from '{src}'")
                legend_below(fig, axes[0], len(books))
                charts.append(("colour_accuracy.png", "Colour classifier accuracy", fig))
                # confusion matrices
                fig, axes = plt.subplots(1, len(ck_all), figsize=(3.2 * len(ck_all), 3.4), dpi=110)
                axes = np.atleast_1d(axes)
                for ax, k in zip(axes, ck_all):
                    M = sum((b.col[(src, k)].conf for b in books if (src, k) in b.col), np.zeros((6, 7), np.int64))
                    Mn = M / np.maximum(1, M.sum(1, keepdims=True))
                    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
                    ax.set_xticks(range(7))
                    ax.set_xticklabels([LETTER[c] for c in COLORS] + ["?"], fontsize=7)
                    ax.set_yticks(range(6))
                    ax.set_yticklabels([LETTER[c] for c in COLORS], fontsize=7)
                    ax.set_title(self.cmap[k].short, fontsize=9)
                    for i in range(6):
                        for j in range(7):
                            if M[i, j]:
                                ax.text(j, i, f"{100 * Mn[i, j]:.0f}", ha="center", va="center", fontsize=6,
                                        color="white" if Mn[i, j] > 0.5 else "black")
                fig.suptitle("Confusion matrices (rows = true colour, columns = predicted, %)", fontsize=10)
                fig.tight_layout()
                charts.append(("confusion.png", "Confusion matrices", fig))
            # end-to-end heat map: detector x classifier
            ck_any = [c.key for c in self.classifiers if any(k[1] == c.key for b in books for k in b.col)]
            b0 = max(books, key=lambda b: b.frames)
            dks = [k for k in dk_all if k in b0.det]
            if ck_any and dks and any(b0.col[(d, c)].frames for d in dks for c in ck_any if (d, c) in b0.col):
                M = np.array([[b0.col[(d, c)].end_to_end() if (d, c) in b0.col else np.nan for c in ck_any] for d in dks])
                fig, ax = plt.subplots(figsize=(1.4 * len(ck_any) + 3, 0.5 * len(dks) + 2), dpi=110)
                ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100)
                ax.set_xticks(range(len(ck_any)))
                ax.set_xticklabels([self.cmap[c].short for c in ck_any], rotation=25, fontsize=8)
                ax.set_yticks(range(len(dks)))
                ax.set_yticklabels([names[d] for d in dks], fontsize=8)
                for i in range(len(dks)):
                    for j in range(len(ck_any)):
                        if not np.isnan(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
                ax.set_title(f"End-to-end: % of stickers right (missed faces count as wrong) - {b0.condition}", fontsize=9)
                fig.tight_layout()
                charts.append(("end_to_end.png", "End-to-end detector x classifier", fig))
        if self.sweep:
            for p in self.sweep["data"]:
                charts.append((f"sweep_{p}.png", f"Robustness: {p}", chart_sweep(self.sweep, p)))
        if self.tracker_rows:
            for ms in dict.fromkeys(r["moveset"] for r in self.tracker_rows):
                charts.append((f"tracker_{ms}.png", f"Tracker comparison ({ms})", chart_tracker(self.tracker_rows, ms)))
            with open(os.path.join(out, "tracker.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.tracker_rows[0].keys()))
                w.writeheader()
                w.writerows(self.tracker_rows)
        cdir = self.ensure_out("charts")
        for fn, _, fig in charts:
            fig.savefig(os.path.join(cdir, fn))
            plt.close(fig)
        self.write_markdown(out, books, charts)
        self.say(f"report exported -> {out} ({len(charts)} charts)")
        return out

    def write_markdown(self, out, books, charts):
        L = [f"# Cube lab report - {self.session}", "",
             f"- Source: **{self.source.name}**",
             f"- YOLO segmentation weights: `{self.seg.path}`" + (" (generic COCO weights - not the trained cube model)" if self.seg.generic else ""),
             f"- Colour calibration: {self.cmap['C5'].source or 'none'}", ""]
        for b in books:
            L += [f"## Condition: {b.condition}  ({b.frames} frames, {fmt(b.frame_ms.mean(), 1)} ms per frame for all systems)", ""]
            if b.perturb:
                L += ["Stress settings used: " + "; ".join(sorted(b.perturb)), ""]
            L += ["### Face isolation", "",
                  "| System | Found % | Mean ms | p95 ms | Own ms | IoU | vs | Corner err % | Jitter % |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for k, m in sorted(b.det.items()):
                s = m.summary()
                L.append(f"| {self.dmap[k].name} | {fmt(s['detect_rate_%'])} | {fmt(s['latency_ms']['mean'])} | "
                         f"{fmt(s['latency_ms']['p95'])} | {fmt(s['own_ms']['mean'])} | {fmt(s['iou_mean'], 3)} | "
                         f"{s['iou_vs'] or '-'} | {fmt(s['corner_err_%'])} | {fmt(s['jitter_%'], 2)} |")
            src = self.col_src
            rows = [(k, m) for (d, k), m in sorted(b.col.items()) if d == src and m.stickers]
            if rows:
                L += ["", f"### Colour classification (faces from `{src}`)", "",
                      "| Classifier | Stickers | Accuracy % | Confident & wrong % | Unsure % | Red/Orange mix % | Flicker % | ms / face |",
                      "|---|---|---|---|---|---|---|---|"]
                for k, m in rows:
                    s = m.summary()
                    L.append(f"| {self.cmap[k].name} | {s['stickers']} | {fmt(s['accuracy_%'])} | {fmt(s['confident_wrong_%'])} | "
                             f"{fmt(s['unsure_%'])} | {fmt(s['red_orange_confusion_%'])} | {fmt(s['flicker_%'])} | {fmt(s['latency_ms']['mean'], 2)} |")
            L.append("")
        if self.tracker_rows:
            L += ["## Move-tracking strategies (simulation)", "",
                  "Whole-cube state correct after each camera observation, averaged over all noise levels:", "",
                  "| Move mix | Tracker | State correct % | Front face correct % | ms / observation |", "|---|---|---|---|---|"]
            for ms in dict.fromkeys(r["moveset"] for r in self.tracker_rows):
                for t in dict.fromkeys(r["tracker"] for r in self.tracker_rows):
                    rs = [r for r in self.tracker_rows if r["moveset"] == ms and r["tracker"] == t]
                    L.append(f"| {ms} | {t} | {np.mean([r['tracking_accuracy_%'] for r in rs]):.1f} | "
                             f"{np.mean([r['front_face_correct_%'] for r in rs]):.1f} | {np.mean([r['ms_per_observation'] for r in rs]):.2f} |")
            L.append("")
        L += ["## Charts", ""]
        for fn, title, _ in charts:
            L += [f"### {title}", f"![{title}](charts/{fn})", ""]
        L += ["## What the numbers mean", "",
              "- **Found %**: frames where the system produced a face outline.",
              "- **IoU**: overlap between the system's outline and the reference (ground truth on synthetic input, otherwise the current D7 system). 1.0 = identical.",
              "- **Corner err %**: average corner distance from the reference, as % of the face size.",
              "- **Jitter %**: how far the corners move between frames while the cube is still (lower = steadier grid).",
              "- **Accuracy %**: stickers whose top label matches ground truth.",
              "- **Confident & wrong %**: wrong label given with confidence - these are the readings that create false moves.",
              "- **Unsure %**: readings the classifier marks as uncertain (Unknown or < 50% confidence); the current decoder can absorb these.",
              "- **Flicker %**: stickers whose label changed between consecutive frames although nothing moved.",
              "- **End-to-end %**: stickers right for detector + classifier together, counting missed faces as wrong."]
        with open(os.path.join(out, "report.md"), "w") as fh:
            fh.write("\n".join(L))

    # ------------------------------------------------------------ text input
    def start_text(self, mode):
        self.text_mode = mode
        self.text_buf = ""
        prompts = {"condition": "type a condition name, Enter to accept",
                   "manual_gt": "type the 9 colours you see, reading order (w y g b r o), Enter",
                   "face": "press w / y / g / b / r / o for the solved face shown"}
        self.say(prompts[mode], 30)

    def text_key(self, key):
        if key in (27,):
            self.text_mode = None
            self.say("cancelled")
            return
        if self.text_mode == "face":
            ch = chr(key).upper() if 0 < key < 256 else ""
            if ch in FROM_LETTER:
                self.gt_face = FROM_LETTER[ch]
                self.gt_mode = "solved face"
                self.text_mode = None
                self.say(f"ground truth: solved {self.gt_face} face")
            return
        if key in (13, 10):
            if self.text_mode == "condition" and self.text_buf.strip():
                self.set_condition(self.text_buf.strip())
            elif self.text_mode == "manual_gt":
                s = [c for c in self.text_buf.upper() if c in FROM_LETTER]
                if len(s) == 9:
                    self.manual_grid = [FROM_LETTER[c] for c in s]
                    self.gt_mode = "manual grid"
                    self.say("manual ground truth set: " + "".join(s))
                else:
                    self.say("need exactly 9 letters (w y g b r o)")
            self.text_mode = None
            return
        if key in (8, 127):
            self.text_buf = self.text_buf[:-1]
        elif 32 <= key < 127:
            self.text_buf += chr(key)

    # =====================================================================
    # Rendering
    # =====================================================================
    def button(self, img, x, y, w, h, label, cb, active=False, color=None, scale=0.42, align="center", enabled=True):
        bg = color if color is not None else (UI["on"] if active else UI["panel2"])
        if not enabled:
            bg = (40, 40, 44)
        cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 80), 1)
        tw = text_w(label, scale)
        tx = x + (w - tw) // 2 if align == "center" else x + 6
        put_plain(img, label, (tx, y + h // 2 + 5), scale, (20, 20, 20) if active and color is None else
                  (UI["text"] if enabled else UI["dim"]))
        if enabled and cb is not None:
            self.buttons.append((x, y, x + w, y + h, cb))

    def render(self):
        self.buttons = []
        W, H = self.W, self.H
        canvas = np.full((H, W, 3), UI["bg"], np.uint8)
        self.draw_topbar(canvas)
        self.draw_sidebar(canvas)
        x0, y0 = CONFIG["sidebar_w"] + 8, 78
        area = (x0, y0, W - x0 - 8, H - y0 - 30)
        [self.draw_side_by_side, self.draw_focus, self.draw_colour_lab, self.draw_stats,
         self.draw_tracker, self.draw_help, self.draw_batch][self.tab](canvas, area)
        self.draw_footer(canvas)
        return canvas

    def draw_topbar(self, img):
        W = self.W
        cv2.rectangle(img, (0, 0), (W, 36), UI["panel"], -1)
        x = 8
        for i, t in enumerate(TABS):
            w = text_w(f"{i + 1} {t}", 0.5) + 22
            self.button(img, x, 4, w, 28, f"{i + 1} {t}", lambda i=i: setattr(self, "tab", i), active=self.tab == i, scale=0.5)
            x += w + 4
        info = f"{self.source.name} | {fmt(1000.0 / self.fps.mean() if self.fps.n else float('nan'), 1)} fps | condition: {self.condition} | GT: {self.gt_mode} | D10 state: {self.d10_belief}"
        if self.gt_mode == "solved face":
            info += f" ({self.gt_face})"
        put(img, info, (W - text_w(info, 0.45) - 10, 23), 0.45, UI["accent"])
        # action row
        cv2.rectangle(img, (0, 38), (W, 72), (30, 30, 36), -1)
        acts = [
            ("Freeze [space]", lambda: setattr(self, "frozen", not self.frozen), self.frozen),
            ("Step [.]", lambda: setattr(self, "step_once", True), False),
            ("Screenshot [s]", lambda: setattr(self, "_shot", True), False),
            ("Save tiles [t]", self.save_tiles, False),
            ("Record [v]", lambda: setattr(self, "_rec", True), self.recorder is not None),
            (f"Benchmark [b]" + (f" {self.bench_left}" if self.bench_left else ""), self.start_benchmark, self.bench_left > 0),
            ("Export report [e]", self.export, False),
            ("Reset stats [0]", self.reset_stats, False),
            ("Calibrate [k]", self.start_scan if self.scan is None else self.scan_capture, self.scan is not None),
            ("GT mode [g]", self.cycle_gt, self.gt_mode != "off"),
            ("GT face [f]", lambda: self.start_text("face"), False),
            ("Condition [c]", self.cycle_condition, False),
            ("Name cond [n]", lambda: self.start_text("condition"), False),
            ("Windows [p]", self.toggle_popouts, self.popouts),
            ("Mirror [i]", lambda: setattr(self, "mirror", not self.mirror), self.mirror),
            ("Log CSV [l]", self.toggle_logging, self.logging),
        ]
        x = 8
        for label, cb, act in acts:
            w = text_w(label, 0.4) + 14
            if x + w > W - 8:
                break
            self.button(img, x, 42, w, 26, label, cb, active=act, scale=0.4)
            x += w + 4

    def draw_sidebar(self, img):
        sw = CONFIG["sidebar_w"]
        cv2.rectangle(img, (0, 74), (sw, self.H - 26), UI["panel"], -1)
        y = 92
        put_plain(img, "FACE ISOLATION SYSTEMS", (10, y), 0.45, UI["accent"])
        y += 8
        for d in self.detectors:
            on = self.en_det[d.key]
            lbl = d.name if d.available else f"{d.key} {d.short} - {d.reason}"[:44]
            self.button(img, 8, y, sw - 16, 19, lbl[:42], lambda d=d: self.en_det.__setitem__(d.key, not self.en_det[d.key]),
                        active=on and d.available, scale=0.37, align="left", enabled=d.available)
            y += 21
        y += 16
        put_plain(img, "COLOUR CLASSIFIERS", (10, y), 0.45, UI["accent"])
        y += 8
        for c in self.classifiers:
            on = self.en_clf[c.key] and c.ready
            lbl = c.name if c.ready else f"{c.key} {c.short} - {c.reason}"[:44]
            self.button(img, 8, y, sw - 50, 19, lbl[:38], lambda c=c: self.en_clf.__setitem__(c.key, not self.en_clf[c.key]),
                        active=on, scale=0.37, align="left", enabled=c.available)
            self.button(img, sw - 38, y, 30, 19, "T", lambda c=c: setattr(self, "tile_clf", c.key),
                        active=self.tile_clf == c.key, scale=0.37)
            y += 21
        put_plain(img, "T = classifier shown on the camera tiles", (10, y + 10), 0.35, UI["dim"])
        y += 34
        put_plain(img, "OVERLAYS", (10, y), 0.45, UI["accent"])
        y += 6
        col = 0
        for k, label, _ in OVERLAYS:
            bx = 8 + col * ((sw - 16) // 2)
            self.button(img, bx, y, (sw - 20) // 2, 19, label[:19], lambda k=k: self.overlay.__setitem__(k, not self.overlay[k]),
                        active=self.overlay[k], scale=0.36)
            col ^= 1
            if col == 0:
                y += 22
        y += 16
        put_plain(img, "STRESS TESTS (applied to every system)", (10, y), 0.42, UI["accent"])
        y += 6
        for name, default, lo, hi, step, desc in Perturb.SPECS:
            v = self.perturb.values[name]
            changed = v != default
            cv2.rectangle(img, (8, y), (sw - 8, y + 18), (55, 55, 64) if changed else (42, 42, 48), -1)
            put_plain(img, f"{name}", (14, y + 14), 0.38, UI["warn"] if changed else UI["text"])
            put_plain(img, f"{v}", (150, y + 14), 0.38, UI["warn"] if changed else UI["dim"])
            self.button(img, sw - 66, y + 1, 26, 16, "-", lambda n=name: self.perturb.nudge(n, -1), scale=0.45)
            self.button(img, sw - 36, y + 1, 26, 16, "+", lambda n=name: self.perturb.nudge(n, +1), scale=0.45)
            y += 20
        self.button(img, 8, y + 4, sw - 16, 22, "Reset stress tests", self.perturb.reset, scale=0.4)

    def draw_footer(self, img):
        H, W = self.H, self.W
        cv2.rectangle(img, (0, H - 26), (W, H), UI["panel"], -1)
        txt = self.msg if time.time() < self.msg_t else "H = help | click anything | stats are per condition"
        if self.text_mode:
            txt = f"> {self.text_buf}_   ({self.msg})"
        if self.scan is not None:
            txt = f"CALIBRATION {self.scan + 1}/6: show the SOLVED {SCAN_ORDER[self.scan]} face and press K   " + txt
        put_plain(img, txt[:190], (10, H - 8), 0.45, UI["accent"] if self.text_mode or self.scan is not None else UI["text"])
        if self.recorder is not None:
            cv2.circle(img, (W - 20, H - 13), 7, (0, 0, 255), -1)
        if self.frozen:
            put_plain(img, "FROZEN", (W - 100, H - 8), 0.45, UI["warn"])

    # ------------------------------------------------------------ tiles
    def render_tile(self, key, tw, th, big=False):
        tile = np.full((th, tw, 3), (16, 16, 18), np.uint8)
        det_obj = self.dmap.get(key)
        if self.frame is None:
            return tile
        fh, fw = self.frame.shape[:2]
        s = min(tw / fw, (th - 34) / fh)
        iw, ih = int(fw * s), int(fh * s)
        ox, oy = (tw - iw) // 2, 34 + (th - 34 - ih) // 2
        img = cv2.resize(self.frame, (iw, ih), interpolation=cv2.INTER_AREA)
        r = self.results.get(key)
        if det_obj is not None and not det_obj.available:
            tile[oy:oy + ih, ox:ox + iw] = (img * 0.25).astype(np.uint8)
            put(tile, "UNAVAILABLE", (ox + 12, oy + 30), 0.6, UI["bad"], 2)
            put(tile, det_obj.reason[:70], (ox + 12, oy + 56), 0.42)
        elif r is None:
            tile[oy:oy + ih, ox:ox + iw] = (img * 0.35).astype(np.uint8)
            put(tile, "OFF", (ox + 12, oy + 30), 0.6, UI["dim"], 2)
        else:
            d = r["det"]
            if self.overlay["mask"] and d.mask is not None:
                m = cv2.resize(d.mask, (iw, ih), interpolation=cv2.INTER_NEAREST) > 0
                tint = img.copy()
                tint[m] = (0.45 * tint[m] + 0.55 * np.array([255, 120, 0])).astype(np.uint8)
                img = tint
            P = lambda q: (np.asarray(q, np.float32) * s).astype(np.int32)
            if self.overlay["cands"]:
                if d.cands:
                    cv2.drawContours(img, [P(c.reshape(-1, 2)) for c in d.cands], -1, (255, 220, 0), 1, cv2.LINE_AA)
                if d.points is not None:
                    for p in d.points:
                        cv2.circle(img, tuple(P(p)), 3, (255, 0, 255), -1)
                if d.keypoints is not None:
                    for i, p in enumerate(d.keypoints):
                        cv2.circle(img, tuple(P(p)), 4, (0, 255, 255), -1)
                        put(img, f"k{i}", tuple(P(p) + 4), 0.35)
                if d.box is not None:
                    b = d.box
                    cv2.rectangle(img, tuple(P(b[:2])), tuple(P(b[2:])), (200, 200, 255), 1)
            if self.overlay["gt"] and self.gt_quad is not None:
                cv2.polylines(img, [P(self.gt_quad)], True, UI["gt"], 1, cv2.LINE_AA)
            if d.found:
                if self.overlay["outline"]:
                    cv2.polylines(img, [P(d.quad)], True, (0, 255, 0), 2, cv2.LINE_AA)
                if self.overlay["grid"]:
                    for ln in grid_lines(d.quad):
                        cv2.line(img, tuple(P(ln[0])), tuple(P(ln[1])), (0, 255, 0), 1, cv2.LINE_AA)
                cls = r["cls"].get(self.tile_clf)
                if self.overlay["labels"] and cls:
                    gt = self.gt_grid()
                    for i, (p, lab) in enumerate(zip(cell_centres(d.quad), cls[0])):
                        c = tuple(P(p))
                        rad = max(6, int(face_size(d.quad) * s * 0.07))
                        cv2.circle(img, c, rad, DRAW_BGR[lab], -1, cv2.LINE_AA)
                        ring = (0, 0, 0)
                        if gt:
                            ring = UI["good"] if lab == gt[i] else (0, 0, 255)
                        elif cls[1][i] < 0.5:
                            ring = UI["warn"]
                        cv2.circle(img, c, rad, ring, 2, cv2.LINE_AA)
                        put_plain(img, LETTER[lab], (c[0] - 4, c[1] + 4), 0.35, (0, 0, 0) if lab in ("White", "Yellow") else (255, 255, 255))
                if self.overlay["inset"] and r["flat"] is not None:
                    n = 120 if big else max(60, min(100, iw // 5))
                    fl = cv2.resize(r["flat"], (n, n), interpolation=cv2.INTER_NEAREST)
                    for k in (1, 2):
                        cv2.line(fl, (k * n // 3, 0), (k * n // 3, n), (0, 255, 0), 1)
                        cv2.line(fl, (0, k * n // 3), (n, k * n // 3), (0, 255, 0), 1)
                    img[4:4 + n, iw - n - 4:iw - 4] = fl
            else:
                put(img, "NOT FOUND " + (f"({d.note})" if d.note else ""), (10, ih - 12), 0.5, UI["bad"], 1)
            tile[oy:oy + ih, ox:ox + iw] = img
        # header
        cv2.rectangle(tile, (0, 0), (tw, 33), UI["panel2"], -1)
        name = det_obj.name if det_obj else key
        put_plain(tile, name[:60], (6, 14), 0.42, UI["accent"] if key == "D7" else UI["text"])
        m = self.book.det.get(key)
        if m and m.frames and self.overlay["timing"]:
            last = r["det"].t_total if r else float("nan")
            t = (f"{fmt(last)} ms (avg {fmt(m.lat.mean())}, p95 {fmt(m.lat.p95())}) | found {fmt(pct(m.found, m.frames), 0)}%"
                 f" | IoU {fmt(m.iou.mean(), 2)} | jitter {fmt(m.jitter.median(), 2)}%")
            put_plain(tile, t[:110], (6, 29), 0.36, UI["dim"])
        return tile

    def tile_grid(self, n, area):
        x, y, w, h = area
        cols = 1 if n == 1 else (2 if n <= 4 else (3 if n <= 9 else 4))
        rows = int(math.ceil(n / cols))
        tw, th = (w - (cols - 1) * 6) // cols, (h - (rows - 1) * 6) // rows
        return [(x + (i % cols) * (tw + 6), y + (i // cols) * (th + 6), tw, th) for i in range(n)]

    def draw_side_by_side(self, img, area):
        keys = [d.key for d in self.detectors if self.en_det[d.key] or not d.available]
        keys = [k for k in keys if self.en_det[k]] or [self.focus]
        for k, (x, y, w, h) in zip(keys, self.tile_grid(len(keys), area)):
            img[y:y + h, x:x + w] = self.render_tile(k, w, h)
            self.buttons.append((x, y, x + w, y + 33, lambda k=k: (setattr(self, "focus", k), setattr(self, "tab", 1))))

    def draw_focus(self, img, area):
        x, y, w, h = area
        bw = int(w * 0.68)
        img[y:y + h, x:x + bw] = self.render_tile(self.focus, bw, h, big=True)
        px = x + bw + 8
        pw = w - bw - 8
        # system chooser
        yy = y
        for d in self.detectors:
            self.button(img, px, yy, pw, 20, d.name[:40], lambda d=d: setattr(self, "focus", d.key),
                        active=self.focus == d.key, scale=0.38, align="left", enabled=d.available)
            yy += 22
        yy += 6
        r = self.results.get(self.focus)
        d_obj = self.dmap[self.focus]
        put_plain(img, d_obj.desc[:56], (px, yy + 12), 0.36, UI["dim"])
        put_plain(img, "era: " + d_obj.era, (px, yy + 30), 0.36, UI["dim"])
        yy += 56
        if r:
            d = r["det"]
            put_plain(img, "TIME THIS FRAME", (px, yy), 0.42, UI["accent"]); yy += 18
            for k, v in list(d.stages.items()) + [("total", d.t_total)]:
                put_plain(img, f"{k:<16} {v:7.2f} ms", (px + 6, yy), 0.4); yy += 17
            yy += 6
            if d.mask is not None and self.overlay["mask"]:
                n = min(pw, 160)
                mh = int(n * d.mask.shape[0] / d.mask.shape[1])
                img[yy:yy + mh, px:px + n] = cv2.cvtColor(cv2.resize(d.mask, (n, mh)), cv2.COLOR_GRAY2BGR)
                put(img, "what the system sees", (px + 4, yy + 14), 0.36)
                yy += mh + 6
            if r["flat"] is not None:
                n = min(150, pw // 2 - 4)
                img[yy:yy + n, px:px + n] = cv2.resize(r["flat"], (n, n))
                put(img, "warped face", (px + 4, yy + 14), 0.36)
                # every classifier's answer
                cy = yy
                gt = self.gt_grid()
                for ck, (labels, conf, ms) in r["cls"].items():
                    gx = px + n + 8
                    put_plain(img, self.cmap[ck].short[:16], (gx, cy + 10), 0.34, UI["dim"])
                    for i, lab in enumerate(labels):
                        cx, cyy = gx + 100 + (i % 3) * 11, cy + (i // 3) * 11
                        cv2.rectangle(img, (cx, cyy), (cx + 10, cyy + 10), DRAW_BGR[lab], -1)
                        if gt and lab != gt[i]:
                            cv2.rectangle(img, (cx, cyy), (cx + 10, cyy + 10), (0, 0, 255), 1)
                    cy += 36
                yy += max(n, cy - yy) + 8
        m = self.book.det.get(self.focus)
        if m and m.frames:
            s = m.summary()
            put_plain(img, f"STATS ({self.condition})", (px, yy), 0.42, UI["accent"]); yy += 18
            for line in [f"frames {s['frames']}   found {fmt(s['detect_rate_%'])}%",
                         f"mean {fmt(s['latency_ms']['mean'])} ms  p95 {fmt(s['latency_ms']['p95'])} ms",
                         f"own processing {fmt(s['own_ms']['mean'])} ms",
                         f"IoU {fmt(s['iou_mean'], 3)} vs {s['iou_vs'] or '-'}",
                         f"corner error {fmt(s['corner_err_%'])}% of face",
                         f"jitter {fmt(s['jitter_%'], 2)}% of face"]:
                if yy < y + h - 10:
                    put_plain(img, line, (px + 6, yy), 0.4); yy += 17
            if self.focus == "D7":
                put_plain(img, f"minAreaRect fallbacks: {self.dmap['D7'].fallbacks}", (px + 6, yy), 0.4)

    def draw_colour_lab(self, img, area):
        x, y, w, h = area
        # source chooser
        srcs = (["GT"] if self.gt_quad is not None else []) + [d.key for d in self.detectors if self.en_det[d.key] and d.available]
        put_plain(img, "Faces taken from:", (x, y + 15), 0.42, UI["accent"])
        bx = x + 140
        for k in srcs:
            lbl = "Ground truth" if k == "GT" else self.dmap[k].short
            wv = text_w(lbl, 0.38) + 14
            self.button(img, bx, y, wv, 22, lbl, lambda k=k: setattr(self, "col_src", k), active=self.col_src == k, scale=0.38)
            bx += wv + 4
        self.button(img, x + w - 300, y, 140, 22, "Robustness sweep [x]", self.start_sweep, active=self.sweep_gen is not None, scale=0.38)
        self.button(img, x + w - 152, y, 150, 22, "Export report [e]", self.export, scale=0.38)
        y += 30
        r = self.results.get(self.col_src)
        n = 270
        if r and r["flat"] is not None:
            fl = cv2.resize(r["flat"], (n, n), interpolation=cv2.INTER_NEAREST)
            for k in (1, 2):
                cv2.line(fl, (k * n // 3, 0), (k * n // 3, n), (0, 255, 0), 1)
                cv2.line(fl, (0, k * n // 3), (n, k * n // 3), (0, 255, 0), 1)
            c, m = CELL * n / WARP, MARGIN * n / WARP
            for i in range(9):
                rr, cc = divmod(i, 3)
                cv2.rectangle(fl, (int(cc * c + m), int(rr * c + m)), (int((cc + 1) * c - m), int((rr + 1) * c - m)), (255, 0, 255), 1)
            img[y:y + n, x:x + n] = fl
            put(img, "warped face (pink = sampled area)", (x + 4, y + n + 16), 0.38)
        else:
            cv2.rectangle(img, (x, y), (x + n, y + n), UI["panel"], -1)
            put(img, "no face from this source", (x + 10, y + 30), 0.45, UI["bad"])
        gt = self.gt_grid()
        if gt:
            put_plain(img, "ground truth:", (x, y + n + 44), 0.4, UI["dim"])
            for i, g in enumerate(gt):
                cx, cy = x + 110 + (i % 3) * 14, y + n + 26 + (i // 3) * 14
                cv2.rectangle(img, (cx, cy), (cx + 16, cy + 16), DRAW_BGR[g], -1)
        # table of classifiers
        tx = x + n + 20
        put_plain(img, "Classifier                       result   acc%  conf-wrong%  unsure%  R/O mix%  flicker%   ms", (tx, y + 12), 0.4, UI["accent"])
        yy = y + 22
        for c in self.classifiers:
            sel = self.sel_clf == c.key
            cv2.rectangle(img, (tx - 4, yy), (x + w, yy + 44), (52, 52, 62) if sel else UI["panel"], -1)
            self.buttons.append((tx - 4, yy, x + w, yy + 44, lambda c=c: setattr(self, "sel_clf", c.key)))
            put_plain(img, c.name[:32], (tx, yy + 18), 0.4, UI["text"] if c.ready else UI["dim"])
            put_plain(img, (c.desc if c.ready else c.reason)[:48], (tx, yy + 36), 0.33, UI["dim"])
            if r and c.key in r["cls"]:
                labels, conf, ms = r["cls"][c.key]
                for i, lab in enumerate(labels):
                    cx, cy = tx + 260 + (i % 3) * 13, yy + 3 + (i // 3) * 13
                    cv2.rectangle(img, (cx, cy), (cx + 12, cy + 12), DRAW_BGR[lab], -1)
                    if gt and lab != gt[i]:
                        cv2.rectangle(img, (cx - 1, cy - 1), (cx + 13, cy + 13), (0, 0, 255), 2)
                    elif conf[i] < 0.5:
                        cv2.rectangle(img, (cx - 1, cy - 1), (cx + 13, cy + 13), UI["warn"], 1)
            m = self.book.col.get((self.col_src, c.key))
            if m and (m.stickers or m.lat.n):
                s = m.summary()
                vals = [fmt(s["accuracy_%"]), fmt(s["confident_wrong_%"]), fmt(s["unsure_%"]),
                        fmt(s["red_orange_confusion_%"]), fmt(s["flicker_%"]), fmt(s["latency_ms"]["mean"], 2)]
                for j, v in enumerate(vals):
                    put_plain(img, v, (tx + 318 + j * 78, yy + 26), 0.42,
                              UI["good"] if j == 0 and s["accuracy_%"] > 95 else UI["text"])
            yy += 48
        # confusion matrix + per-colour accuracy for the selected classifier
        m = self.book.col.get((self.col_src, self.sel_clf))
        cy0 = max(yy + 14, y + n + 84)
        if m is not None and m.stickers:
            put_plain(img, f"Confusion matrix - {self.cmap[self.sel_clf].name} (rows true, cols predicted)", (x, cy0), 0.42, UI["accent"])
            M = m.conf
            Mn = M / np.maximum(1, M.sum(1, keepdims=True))
            cs = 34
            for j, lab in enumerate(COLORS + ["?"]):
                put_plain(img, LETTER.get(lab, "?"), (x + 30 + j * cs + 12, cy0 + 20), 0.42)
            for i, g in enumerate(COLORS):
                put_plain(img, LETTER[g], (x + 10, cy0 + 26 + i * cs + 22), 0.42)
                for j in range(7):
                    v = Mn[i, j]
                    col = (int(255 - 200 * v), int(255 - 60 * v), int(255 - 200 * v)) if i == j else (int(255 - 60 * v), int(255 - 200 * v), int(255 - 200 * v))
                    cv2.rectangle(img, (x + 30 + j * cs, cy0 + 26 + i * cs), (x + 30 + (j + 1) * cs - 2, cy0 + 26 + (i + 1) * cs - 2), col, -1)
                    if M[i, j]:
                        put_plain(img, f"{100 * v:.0f}", (x + 34 + j * cs, cy0 + 26 + i * cs + 21), 0.36, (0, 0, 0))
            by = cy0 + 26 + 6 * cs + 18
            put_plain(img, "per colour accuracy", (x, by), 0.4, UI["dim"])
            for i, c in enumerate(COLORS):
                a = pct(M[i, i], M[i].sum())
                bw = int(1.4 * (a if not math.isnan(a) else 0))
                yb = by + 8 + i * 19
                cv2.rectangle(img, (x + 70, yb), (x + 70 + bw, yb + 14), DRAW_BGR[c], -1)
                put_plain(img, c, (x, yb + 12), 0.38)
                put_plain(img, f"{fmt(a)}%", (x + 216, yb + 12), 0.38)
        # sweep view
        if self.sweep_gen is not None:
            self.progress_bar(img, x + w - 300, y + h - 24, 290, self.sweep_progress, "sweep")
        elif self.sweep:
            params = list(self.sweep["data"].keys())
            sx = x + 290 if m is not None and m.stickers else x
            sy = cy0
            bx = sx
            for p in params:
                wv = text_w(p, 0.34) + 10
                if bx + wv > x + w:
                    bx, sy = sx, sy + 22
                self.button(img, bx, sy, wv, 20, p, lambda p=p: setattr(self, "sweep_param", p), active=self.sweep_param == p, scale=0.34)
                bx += wv + 3
            sp = self.sweep_param or params[0]
            key = f"sweep_{sp}"
            if key not in self.chart_cache:
                self.chart_cache[key] = fig_to_bgr(chart_sweep(self.sweep, sp))
            ch = self.chart_cache[key]
            aw, ah = x + w - sx, y + h - (sy + 26)
            if aw > 50 and ah > 50:
                sc = min(aw / ch.shape[1], ah / ch.shape[0])
                cimg = cv2.resize(ch, (int(ch.shape[1] * sc), int(ch.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                img[sy + 26:sy + 26 + cimg.shape[0], sx:sx + cimg.shape[1]] = cimg

    def progress_bar(self, img, x, y, w, frac, label):
        cv2.rectangle(img, (x, y), (x + w, y + 18), UI["panel2"], -1)
        cv2.rectangle(img, (x, y), (x + int(w * min(1, frac)), y + 18), UI["on"], -1)
        put_plain(img, f"{label} {100 * frac:.0f}%", (x + 6, y + 14), 0.4, (20, 20, 20))

    def draw_stats(self, img, area):
        x, y, w, h = area
        book = self.view_book or self.book
        put_plain(img, "Conditions:", (x, y + 15), 0.42, UI["accent"])
        bx = x + 100
        for c, b in self.books.items():
            lbl = f"{c} ({b.frames})"
            wv = text_w(lbl, 0.38) + 14
            self.button(img, bx, y, wv, 22, lbl, lambda b=b: setattr(self, "view_book", b), active=book is b, scale=0.38)
            bx += wv + 4
        y += 34
        put_plain(img, f"FACE ISOLATION  -  {book.condition}, {book.frames} frames, all systems together {fmt(book.frame_ms.mean())} ms/frame",
                  (x, y), 0.45, UI["accent"])
        y += 20
        cols = [("System", 300), ("found %", 70), ("mean ms", 75), ("p95 ms", 70), ("own ms", 70), ("IoU", 60),
                ("vs", 110), ("corner err %", 100), ("jitter %", 70)]
        self.table_header(img, x, y, cols)
        y += 20
        for d in self.detectors:
            m = book.det.get(d.key)
            if not m:
                continue
            s = m.summary()
            vals = [d.name, fmt(s["detect_rate_%"]), fmt(s["latency_ms"]["mean"]), fmt(s["latency_ms"]["p95"]),
                    fmt(s["own_ms"]["mean"]), fmt(s["iou_mean"], 3), s["iou_vs"] or "-", fmt(s["corner_err_%"]), fmt(s["jitter_%"], 2)]
            self.table_row(img, x, y, cols, vals, highlight=d.key == "D7")
            y += 19
        y += 14
        src = self.col_src
        put_plain(img, f"COLOUR CLASSIFIERS  -  faces from '{src}'", (x, y), 0.45, UI["accent"])
        y += 20
        cols2 = [("Classifier", 300), ("stickers", 70), ("acc %", 60), ("conf-wrong %", 100), ("unsure %", 75),
                 ("R/O mix %", 80), ("flicker %", 75), ("ms", 60), ("W/Y/G/B/R/O acc %", 260)]
        self.table_header(img, x, y, cols2)
        y += 20
        for c in self.classifiers:
            m = book.col.get((src, c.key))
            if not m:
                continue
            s = m.summary()
            pc = "/".join(fmt(s["per_colour_acc_%"][k], 0) for k in COLORS)
            vals = [c.name, s["stickers"], fmt(s["accuracy_%"]), fmt(s["confident_wrong_%"]), fmt(s["unsure_%"]),
                    fmt(s["red_orange_confusion_%"]), fmt(s["flicker_%"]), fmt(s["latency_ms"]["mean"], 2), pc]
            self.table_row(img, x, y, cols2, vals, highlight=c.key == "C5")
            y += 19
        y += 14
        clfs = [c for c in self.classifiers if any(k[1] == c.key for k in book.col)]
        dets = [d for d in self.detectors if d.key in book.det]
        if clfs and dets:
            put_plain(img, "END-TO-END: % of stickers right (detector x classifier, missed faces count as wrong)", (x, y), 0.45, UI["accent"])
            y += 20
            cols3 = [("Detector \\ classifier", 300)] + [(c.short[:14], 110) for c in clfs]
            self.table_header(img, x, y, cols3)
            y += 20
            for d in dets:
                vals = [d.name]
                for c in clfs:
                    m = book.col.get((d.key, c.key))
                    vals.append(fmt(m.end_to_end()) if m and m.frames else "-")
                self.table_row(img, x, y, cols3, vals, highlight=d.key == "D7")
                y += 19
            if not self.gt_grid():
                put_plain(img, "(needs ground truth: synthetic source, or GT mode 'solved face' / 'manual grid')", (x, y + 14), 0.38, UI["dim"])
        if self.bench_results and y < self.H - 120:
            y += 30
            put_plain(img, "Finished benchmarks: " + ", ".join(f"{b['condition']} ({b['frames']} fr)" for b in self.bench_results), (x, y), 0.4, UI["dim"])

    def table_header(self, img, x, y, cols):
        cx = x
        for name, wv in cols:
            put_plain(img, name, (cx, y + 12), 0.38, UI["dim"])
            cx += wv

    def table_row(self, img, x, y, cols, vals, highlight=False):
        if highlight:
            cv2.rectangle(img, (x - 4, y - 1), (x + sum(c[1] for c in cols), y + 17), (50, 50, 30), -1)
        cx = x
        for (name, wv), v in zip(cols, vals):
            put_plain(img, str(v)[: max(4, wv // 8)], (cx, y + 13), 0.4)
            cx += wv

    def draw_tracker(self, img, area):
        x, y, w, h = area
        put_plain(img, "MOVE-TRACKING STRATEGIES (simulation - no camera needed)", (x, y + 15), 0.5, UI["accent"])
        self.button(img, x, y + 26, 160, 26, "Quick run (~15 s)", lambda: self.start_tracker(True), active=self.tracker_gen is not None, scale=0.42)
        self.button(img, x + 168, y + 26, 160, 26, "Full run (minutes)", lambda: self.start_tracker(False), scale=0.42)
        bx = x + 350
        for ms in ["faces", "solve-like", "full"]:
            self.button(img, bx, y + 26, 110, 26, ms, lambda ms=ms: (setattr(self, "tracker_view", ms), self.chart_cache.pop("tracker", None)),
                        active=self.tracker_view == ms, scale=0.42)
            bx += 116
        if self.tracker_gen is not None:
            self.progress_bar(img, x + 720, y + 30, 300, self.tracker_progress, "simulating")
        yy = y + 74
        lines = ["A random sequence of moves is applied to a virtual cube. After each move (or after several, if 'unseen') the",
                 "tracker only sees the FRONT 9 stickers, with misread and covered stickers mixed in - like the camera.",
                 "It must keep the whole 54-sticker state right. Hidden turns (e.g. B) can only be worked out later.",
                 ""]
        for T in TRACKERS:
            lines.append(f"{T.name}: {T.desc}")
        for ln in lines:
            put_plain(img, ln, (x, yy), 0.4, UI["text"] if ln.startswith("T") else UI["dim"])
            yy += 18
        if self.tracker_rows:
            rows = [r for r in self.tracker_rows if r["moveset"] == self.tracker_view]
            if rows:
                if "tracker" not in self.chart_cache:
                    self.chart_cache["tracker"] = fig_to_bgr(chart_tracker(self.tracker_rows, self.tracker_view))
                ch = self.chart_cache["tracker"]
                aw, ah = w, y + h - yy - 10
                sc = min(aw / ch.shape[1], ah / ch.shape[0], 1.4)
                cimg = cv2.resize(ch, (int(ch.shape[1] * sc), int(ch.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                img[yy + 6:yy + 6 + cimg.shape[0], x:x + cimg.shape[1]] = cimg
        else:
            put_plain(img, "Press 'Quick run' to compare the four strategies.", (x, yy + 20), 0.45, UI["warn"])

    def draw_batch(self, img, area):
        x, y, w, h = area
        cfg = self.batch_cfg
        put_plain(img, "BATCH TEST - capture once, every condition is applied to the same frames afterwards", (x, y + 15), 0.5, UI["accent"])
        yy = y + 40
        # ---- step 1
        n_scan = len(self.scan_frames)
        ok1 = n_scan == 6 or (self.cmap["C5"].ready and not cfg.get("calibrate_per_condition", True))
        put_plain(img, "1  CALIBRATE", (x, yy + 15), 0.48, UI["good"] if ok1 else UI["text"])
        lbl = "Calibrate [k]" if self.scan is None else f"Capture {SCAN_ORDER[self.scan]} [k]"
        self.button(img, x + 150, yy, 190, 24, lbl,
                    lambda: self.start_scan() if self.scan is None else self.scan_capture(), active=self.scan is not None, scale=0.42)
        status = (f"scan frames {n_scan}/6 - each condition gets its own calibration in its lighting"
                  if cfg.get("calibrate_per_condition", True) else
                  "one calibration for every condition (tests robustness to lighting changes)")
        put_plain(img, status, (x + 350, yy + 16), 0.4, UI["dim"])
        yy += 34
        # ---- step 2
        b = self.batch or {}
        st = b.get("state")
        put_plain(img, "2  CAPTURE", (x, yy + 15), 0.48, UI["good"] if self.batch_frames and st in ("analysing", "done") else UI["text"])
        if st == "wait":
            lbl = f"Record {SCAN_ORDER[b['face']]} face [u]"
        elif st == "rec":
            lbl = f"recording {SCAN_ORDER[b['face']]}..."
        else:
            lbl = "Start capture [u]"
        self.button(img, x + 150, yy, 190, 24, lbl, self.batch_go, active=st in ("wait", "rec"), scale=0.42)
        if st in ("wait", "rec"):
            n = b.get("n", 0) if st == "rec" else 0
            self.progress_bar(img, x + 350, yy + 3, 260, (b["face"] + n / float(cfg["frames_per_face"])) / 6.0,
                              f"face {b['face'] + 1}/6")
            tip = ("hold the SOLVED " + SCAN_ORDER[b["face"]] + " face up, then press U") if st == "wait" else \
                  "tilt it slowly, move it around a little, keep it in view"
            put_plain(img, tip, (x + 620, yy + 16), 0.42, UI["warn"])
        else:
            put_plain(img, f"{len(self.batch_frames)} frames captured" if self.batch_frames else
                      "you show each solved face once (~5 s each); ground truth is set automatically",
                      (x + 350, yy + 16), 0.4, UI["dim"])
        yy += 34
        # ---- step 3
        put_plain(img, "3  ANALYSE", (x, yy + 15), 0.48, UI["good"] if st == "done" else UI["text"])
        self.button(img, x + 150, yy, 190, 24, "Analyse + export", self.start_batch_analysis,
                    active=st == "analysing", scale=0.42, enabled=bool(self.batch_frames))
        if st == "analysing":
            self.progress_bar(img, x + 350, yy + 3, 260, self.batch_progress, "analysing")
            put_plain(img, "you can put the cube down", (x + 620, yy + 16), 0.42, UI["dim"])
        else:
            put_plain(img, "runs automatically after capture: every system x every condition, benchmark files + report",
                      (x + 350, yy + 16), 0.4, UI["dim"])
        yy += 44
        # ---- settings
        put_plain(img, "SETTINGS", (x, yy + 12), 0.45, UI["accent"])
        yy += 20

        def nud(key, d, lo, hi):
            cfg[key] = int(min(hi, max(lo, cfg[key] + d)))

        for key, label, step, lo, hi in (("frames_per_face", "frames per face", 10, 10, 300),
                                         ("capture_every", "keep every Nth frame", 1, 1, 10)):
            put_plain(img, f"{label}: {cfg[key]}", (x, yy + 15), 0.42)
            self.button(img, x + 220, yy, 26, 20, "-", lambda k=key, s_=step, a=lo, b_=hi: nud(k, -s_, a, b_), scale=0.45)
            self.button(img, x + 250, yy, 26, 20, "+", lambda k=key, s_=step, a=lo, b_=hi: nud(k, s_, a, b_), scale=0.45)
            yy += 24
        self.button(img, x, yy, 276, 22, "Calibrate per condition: " + ("ON" if cfg.get("calibrate_per_condition", True) else "OFF"),
                    lambda: cfg.__setitem__("calibrate_per_condition", not cfg.get("calibrate_per_condition", True)),
                    active=cfg.get("calibrate_per_condition", True), scale=0.4)
        yy += 26
        self.button(img, x, yy, 276, 22, f"Reload {BATCH_FILE}", self.reload_batch_config, scale=0.4)
        yy += 34
        # ---- conditions
        put_plain(img, "CONDITIONS (click to switch on/off; edit names/values in " + BATCH_FILE + ")", (x, yy + 12), 0.45, UI["accent"])
        yy += 20
        for cond in cfg["conditions"]:
            on = cond.get("on", True)
            self.button(img, x, yy, 180, 21, cond["name"][:22], lambda c=cond: c.__setitem__("on", not c.get("on", True)),
                        active=on, scale=0.4, align="left")
            sets = ", ".join(f"{k}={v}" for k, v in cond.get("settings", {}).items()) or "no change"
            light = any(k in LIGHTING_KEYS for k in cond.get("settings", {}))
            extra = "  (own calibration)" if light and cfg.get("calibrate_per_condition", True) else ""
            put_plain(img, sets + extra, (x + 190, yy + 15), 0.4, UI["text"] if on else UI["dim"])
            yy += 24
        put_plain(img, "Close up / far away / cluttered background: test those by hand (they need different camera shots).",
                  (x, yy + 16), 0.4, UI["dim"])
        # ---- live view
        tw = min(560, w - 620)
        if tw > 200:
            key = next((k for k in ("D9", "D7", "D4") if self.dmap[k].available), "D4")
            tile = self.render_tile(key, tw, int(tw * 0.62) + 34)
            img[y + 160:y + 160 + tile.shape[0], x + w - tw:x + w] = tile

    def draw_help(self, img, area):
        x, y, w, h = area
        lines = [
            ("TABS", "1 side by side   2 focus (click a tile header)   3 colour lab   4 stats   5 tracker lab   6 help   7 batch"),
            ("BATCH", "tab 7: k calibrate once, u capture the 6 solved faces once -> every condition analysed + exported"),
            ("FRAMES", "space freeze   . step   i mirror   s screenshot (+ one per batch condition)   t save every system at full size"),
            ("RECORD", "v start/stop video of the lab window   l log every frame to frames.csv"),
            ("TESTS", "b benchmark N frames (resets the condition first)   0 reset stats   x robustness sweep   e export report"),
            ("FPS", "a FPS test: average fps of every enabled system with NO cube in view, then WITH the cube -> chart"),
            ("CONDITION", "c cycle manual conditions (normal, far, close, cluttered)   n type a name - lighting/fingers/fast turns: Batch tab"),
            ("TRUTH", "g cycle ground-truth mode   f set solved face colour (then w/y/g/b/r/o)   m type the 9 visible colours"),
            ("D10 STATE", "j cycles D10's face knowledge: auto (GT, else it learns the face) / self (learns) / WRONG face / off"),
            ("COLOUR", "k start the 6-face calibration scan (then k on each solved face) -> calibrates C4 and C5"),
            ("SYSTEMS", "click any system in the sidebar to switch it on/off; 'T' picks the classifier drawn on the tiles"),
            ("WINDOWS", "p opens every enabled system in its own window (camera tabs) for side-by-side screen captures"),
            ("STRESS", "sidebar -/+ change brightness, glare, fingers, blur, noise, ... for EVERY system at once"),
            ("QUIT", "q or Esc"),
            ("", ""),
            ("GOOD TESTS", "1. Synthetic source: every number has perfect ground truth (IoU, colour accuracy, end-to-end)."),
            ("", "2. Real cube, solved face held still: GT mode 'solved face' + f + colour, then benchmark per condition."),
            ("", "3. Record a solve with the main tracker, replay it here with --source video.mp4: identical frames for all."),
            ("", "4. Robustness sweep (x) on synthetic: accuracy vs brightness / glare / fingers / blur for every system."),
            ("", "5. Tracker lab: why the beam decoder beat single-move matching and greedy search."),
            ("", ""),
            ("OUTPUT", f"everything goes to {self.out_dir}"),
        ]
        yy = y + 20
        for a, b in lines:
            put_plain(img, a, (x, yy), 0.45, UI["accent"])
            put_plain(img, b, (x + 120, yy), 0.45)
            yy += 26

    # =====================================================================
    # Input
    # =====================================================================
    def on_mouse(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, cb in reversed(self.buttons):
                if x0 <= mx <= x1 and y0 <= my <= y1:
                    cb()
                    return

    def on_key(self, key, canvas):
        if key < 0:
            return True
        key &= 0xFFFF
        if self.text_mode:
            self.text_key(key & 0xFF)
            return True
        k = chr(key & 0xFF) if (key & 0xFF) < 128 else ""
        if k in ("q",) or key == 27:
            return False
        actions = {
            " ": lambda: setattr(self, "frozen", not self.frozen),
            ".": lambda: setattr(self, "step_once", True),
            "s": lambda: self.screenshot(canvas), "t": self.save_tiles,
            "v": lambda: self.toggle_record(canvas), "b": self.start_benchmark, "e": self.export,
            "0": self.reset_stats, "g": self.cycle_gt, "f": lambda: self.start_text("face"),
            "m": lambda: self.start_text("manual_gt"), "c": self.cycle_condition,
            "n": lambda: self.start_text("condition"), "p": self.toggle_popouts,
            "i": lambda: setattr(self, "mirror", not self.mirror), "l": self.toggle_logging,
            "x": self.start_sweep, "h": lambda: setattr(self, "tab", 5),
            "k": lambda: self.start_scan() if self.scan is None else self.scan_capture(),
            "j": self.cycle_d10_belief,
            "u": self.batch_go,
            "a": self.start_fps_test,
            "[": lambda: self.cycle_focus(-1), "]": lambda: self.cycle_focus(1),
        }
        if k in "1234567" and k:
            self.tab = int(k) - 1
        elif k in actions:
            actions[k]()
        return True

    def cycle_focus(self, d):
        keys = [x.key for x in self.detectors if x.available] or [x.key for x in self.detectors]
        i = keys.index(self.focus) if self.focus in keys else 0
        self.focus = keys[(i + d) % len(keys)]

    # =====================================================================
    # Main loop
    # =====================================================================
    def grab(self):
        f, q, g = self.source.read()
        if f is None:
            return False
        if self.mirror:
            f = cv2.flip(f, 1)
            if q is not None:
                q = q.copy()
                q[:, 0] = f.shape[1] - 1 - q[:, 0]
                g = [g[r * 3 + (2 - c)] for r in range(3) for c in range(3)] if g else g
        self.raw, self.gt_quad, self.src_grid = f, q, g
        return True

    def run(self):
        win = "Cube Lab - developer test bench"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.W, self.H)
        cv2.setMouseCallback(win, self.on_mouse)
        self._shot = self._rec = False
        while True:
            if not self.frozen or self.step_once or self.raw is None:
                if not self.grab():
                    self.say("source ended")
                    self.frozen = True
                    if self.raw is None:
                        break
                self.step_once = False
                self.process()
            elif self.perturb.active() and self.raw is not None:
                self.process()   # frozen frame, but stress settings may change live
            self.background_work()
            canvas = self.render()
            if self._shot:
                self._shot = False
                self.screenshot(canvas)
                self.screenshot_conditions()
            if self._rec:
                self._rec = False
                self.toggle_record(canvas)
            if self.recorder is not None:
                self.recorder.write(canvas)
            cv2.imshow(win, canvas)
            if self.popouts:
                for d in self.detectors:
                    n = f"LAB {d.key} {d.short}"
                    if self.en_det[d.key] and d.available:
                        cv2.imshow(n, self.render_tile(d.key, 720, 450))
                        self.popout_names.add(n)
                    elif n in self.popout_names:
                        cv2.destroyWindow(n)
                        self.popout_names.discard(n)
            now = time.perf_counter()
            self.fps.add((now - self.last_t) * 1000)
            self.last_t = now
            if not self.on_key(cv2.waitKeyEx(1), canvas):
                break
        if self.recorder is not None:
            self.recorder.release()
        if self.log_fh:
            self.log_fh.close()
        self.source.release()
        cv2.destroyAllWindows()

    def run_headless(self, frames, do_export, do_sweep, do_tracker, quick):
        n = 0
        t0 = time.time()
        while n < frames and self.grab():
            self.process()
            n += 1
            if n % 50 == 0:
                print(f"[lab] {n}/{frames} frames, {self.book.frame_ms.mean():.1f} ms/frame")
        print(f"[lab] processed {n} frames in {time.time() - t0:.1f}s")
        if do_sweep:
            self.start_sweep()
            while self.sweep_gen:
                self.background_work(10_000)
        if do_tracker:
            self.start_tracker(quick)
            while self.tracker_gen:
                self.background_work(10_000)
        if do_export:
            return self.export()
        return None


def main():
    ap = argparse.ArgumentParser(description="Cube tracker developer test bench")
    ap.add_argument("--source", default=None, help="'synthetic', a video file or an image folder (default: camera)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seg", help="YOLO segmentation weights (your trained best.pt)")
    ap.add_argument("--det", help="YOLO box-detection weights (optional)")
    ap.add_argument("--pose", help="YOLO pose/keypoint weights (optional)")
    ap.add_argument("--cnn", help="HackRubiksNet weights rubiks_model.pth (optional)")
    ap.add_argument("--calib", help="colour_calibration.json from the main tracker")
    ap.add_argument("--out", help="output folder (default lab_runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solved", action="store_true", help="synthetic: show solved faces only")
    ap.add_argument("--dark-bg", action="store_true", help="synthetic: black background (Jokinen's setup)")
    ap.add_argument("--d10-belief", default="auto", choices=["auto", "self", "wrong", "off"],
                    help="what D10 is told the tracker expects (key j in the window)")
    ap.add_argument("--batch-replay", metavar="CAPTURE_DIR",
                    help="re-analyse a saved batch capture (the capture/ folder of an earlier session)")
    ap.add_argument("--angled", action="store_true", help="synthetic: cube seen at an angle (top + side faces large)")
    ap.add_argument("--bench-frames", type=int, default=None)
    ap.add_argument("--headless", action="store_true", help="no window: process frames then export")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="headless: run the robustness sweep")
    ap.add_argument("--tracker-sweep", action="store_true", help="run the move-tracker simulation")
    ap.add_argument("--quick", action="store_true", help="smaller tracker simulation")
    ap.add_argument("--condition", default="normal")
    ap.add_argument("--perturb", default="", help="e.g. brightness=-60,glare=0.5 (headless)")
    args = ap.parse_args()

    if args.tracker_sweep and not args.headless and args.source is None:
        args.headless = True
        args.source = "synthetic"
        args.frames = 0
    if args.batch_replay:
        args.source = args.batch_replay
    app = App(args)
    app.set_condition(args.condition)
    app.d10_belief = args.d10_belief
    for kv in filter(None, args.perturb.split(",")):
        k, v = kv.split("=")
        app.perturb.values[k.strip()] = type(app.perturb.spec(k.strip())[1])(float(v))
    print("[lab] systems:", ", ".join(f"{d.key}{'' if d.available else ' (off: ' + d.reason + ')'}" for d in app.detectors))
    print("[lab] classifiers:", ", ".join(f"{c.key}{'' if c.ready else ' (needs: ' + c.reason + ')'}" for c in app.classifiers))
    if args.batch_replay:
        app.load_batch_capture(args.batch_replay)
        app.start_batch_analysis()
        if args.headless:
            last = -1
            while app.batch_gen:
                app.background_work(10_000)
                pct = int(100 * app.batch_progress)
                if pct // 10 != last // 10:
                    print(f"[lab] batch analysis {pct}%")
                    last = pct
            return
        app.tab = 6
        app.run()
        return
    if args.headless:
        out = app.run_headless(args.frames, args.export or args.tracker_sweep or args.sweep,
                               args.sweep, args.tracker_sweep, args.quick)
        if out:
            print("[lab] report:", os.path.join(out, "report.md"))
    else:
        app.run()


if __name__ == "__main__":
    main()
