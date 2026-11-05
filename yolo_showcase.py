"""
yolo_showcase.py - the four YOLO cube detectors side by side, for screenshots.

    python yolo_showcase.py                 # camera 0
    python yolo_showcase.py --camera 1      # iPhone / Continuity camera
    python yolo_showcase.py --source photo.jpg   (or a video / a folder of images)

Models are found automatically: every best.pt under runs/ (and any .pt in this
folder) is opened and sorted by its task (detect / obb / segment / pose); the
newest of each kind is used. Override with --det --obb --seg --pose.

Keys
    space  freeze / unfreeze        s  save screenshots: side by side + each model on its own
    1-4    show / hide a panel      p  each model in its own window too (for screen captures)
    i      mirror
    q/Esc  quit
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

KINDS = [("detect", "Standard detection (box)"), ("obb", "Oriented box (OBB)"),
         ("segment", "Segmentation (mask)"), ("pose", "Keypoints (pose)")]
COL = {"detect": (60, 180, 255), "obb": (255, 170, 40), "segment": (90, 220, 120), "pose": (230, 90, 230)}


def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def is_stock(model):
    """True for an off-the-shelf model (COCO/DOTA classes), not one trained on the cube."""
    names = [str(n).lower() for n in getattr(model, "names", {}).values()]
    return len(names) >= 15 or any(n in ("person", "plane", "ship", "car") for n in names)


def find_models(args):
    from ultralytics import YOLO
    chosen = {"detect": args.det, "obb": args.obb, "segment": args.seg, "pose": args.pose}
    # your trained runs first (newest run of each kind wins); loose .pt files only as a last resort
    runs = sorted(glob.glob("runs/**/weights/best.pt", recursive=True), key=os.path.getmtime, reverse=True)
    loose = sorted(glob.glob("*.pt"), key=os.path.getmtime, reverse=True)
    for p in runs + loose:
        if all(chosen.values()):
            break
        try:
            m = YOLO(p)
        except Exception:
            continue
        if m.task in chosen and not chosen[m.task] and not is_stock(m):
            chosen[m.task] = p
    models = {}
    for k, path in chosen.items():
        if path and os.path.exists(path):
            try:
                m = YOLO(path)
                models[k] = (m, path)
                if is_stock(m):
                    print(f"[showcase] WARNING: {path} is a stock model ({len(m.names)} everyday classes), not your cube model")
            except Exception as exc:
                print(f"[showcase] could not load {path}: {exc}")
    for k, name in KINDS:
        if k in models:
            m = models[k][0]
            print(f"[showcase] {name:28s}: {models[k][1]}   classes: {list(m.names.values())[:5]}")
        else:
            print(f"[showcase] {name:28s}: not found - give it with --{ {'detect': 'det', 'obb': 'obb', 'segment': 'seg', 'pose': 'pose'}[k] }")
    return models


def put(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def biggest(res, n):
    """index of the largest detection (the cube nearest the camera)."""
    if res.boxes is None or len(res.boxes) == 0:
        return None
    b = res.boxes.xyxy.cpu().numpy()
    return int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))


def draw(kind, img, res):
    c = COL[kind]
    out = img.copy()
    h, w = img.shape[:2]
    t = max(2, w // 400)
    if kind == "obb":
        if res.obb is None or len(res.obb) == 0:
            return out, False
        polys = res.obb.xyxyxyxy.cpu().numpy()
        confs = res.obb.conf.cpu().numpy()
        i = int(np.argmax([cv2.contourArea(p.astype(np.float32)) for p in polys]))
        p = polys[i].astype(np.int32)
        cv2.polylines(out, [p], True, c, t + 1, cv2.LINE_AA)
        for q in p:
            cv2.circle(out, tuple(int(v) for v in q), t + 3, c, -1, cv2.LINE_AA)
        put(out, f"{confs[i]:.2f}", (int(p[:, 0].min()), int(p[:, 1].min()) - 8), 0.7, c, 2)
        return out, True
    i = biggest(res, len(img))
    if i is None:
        return out, False
    conf = float(res.boxes.conf.cpu().numpy()[i])
    x0, y0, x1, y1 = res.boxes.xyxy.cpu().numpy()[i].astype(int)
    if kind == "detect":
        cv2.rectangle(out, (x0, y0), (x1, y1), c, t + 1, cv2.LINE_AA)
    elif kind == "segment":
        if res.masks is None or len(res.masks) <= i:
            return out, False
        m = cv2.resize(res.masks.data[i].cpu().numpy(), (w, h)) > 0.5
        layer = out.copy()
        layer[m] = c
        out = cv2.addWeighted(layer, 0.4, out, 0.6, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            cv2.drawContours(out, [cnt], -1, c, t, cv2.LINE_AA)
    elif kind == "pose":
        if res.keypoints is None or len(res.keypoints) <= i:
            return out, False
        k = res.keypoints.xy[i].cpu().numpy()
        kc = res.keypoints.conf[i].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(k))
        ok = [(j, p) for j, p in enumerate(k) if not np.all(p == 0) and kc[j] > 0.3]
        pts = {j: tuple(int(v) for v in p) for j, p in ok}
        if all(j in pts for j in range(4)):          # first four points = the front face
            cv2.polylines(out, [np.array([pts[j] for j in range(4)], np.int32)], True, c, t, cv2.LINE_AA)
        for j, p in pts.items():
            cv2.circle(out, p, t + 5, c, -1, cv2.LINE_AA)
            put(out, str(j), (p[0] + 8, p[1] - 8), 0.7, (255, 255, 255), 2)
    put(out, f"{conf:.2f}", (x0, max(20, y0 - 8)), 0.7, c, 2)
    return out, True


def panel(kind, name, img, models, device, times, conf):
    h, w = img.shape[:2]
    if kind not in models:
        out = (img * 0.35).astype(np.uint8)
        put(out, "model not found", (20, h // 2), 1.0, (120, 120, 255), 2)
        found = False
    else:
        t0 = time.perf_counter()
        res = models[kind][0](img, conf=conf, verbose=False, device=device)[0]
        times[kind] = 0.85 * times.get(kind, 0) + 0.15 * (time.perf_counter() - t0) * 1000 if kind in times \
            else (time.perf_counter() - t0) * 1000
        out, found = draw(kind, img, res)
        if not found:
            put(out, "no cube found", (20, h - 30), 1.0, (120, 120, 255), 2)
    bar = np.full((44, w, 3), 24, np.uint8)
    put(bar, name, (12, 30), 0.85, COL[kind], 2)
    if kind in times:
        put(bar, f"{times[kind]:.0f} ms", (w - 120, 30), 0.75, (220, 220, 220), 2)
    return np.vstack([bar, out]), found


def frames(args):
    s = args.source
    if s and os.path.isdir(s):
        files = sorted(f for f in glob.glob(os.path.join(s, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        while True:
            for f in files:
                yield cv2.imread(f), os.path.basename(f)
    if s and s.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imread(s)
        while True:
            yield img, os.path.basename(s)
    cap = cv2.VideoCapture(s if s else args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ok, f = cap.read()
        if not ok:
            if s:                                      # a video: loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        yield f, "camera"


def main():
    ap = argparse.ArgumentParser(description="YOLO cube detectors side by side")
    ap.add_argument("--source", help="image, video or folder of images (default: camera)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--det", help="standard detection weights")
    ap.add_argument("--obb", help="oriented box (OBB) weights")
    ap.add_argument("--seg", help="segmentation weights")
    ap.add_argument("--pose", help="keypoint (pose) weights")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--out", default="yolo_shots")
    args = ap.parse_args()

    models = find_models(args)
    device = pick_device()
    for m, _ in models.values():          # warm up, so the first timings are real
        try:
            m(np.zeros((480, 640, 3), np.uint8), verbose=False, device=device)
        except Exception:
            pass
    print(f"[showcase] device: {device}   keys: space freeze, s save, 1-4 panels, p own windows, i mirror, q quit")
    show = {k: True for k, _ in KINDS}
    mirror = args.source is None
    frozen, last, times = False, None, {}
    popouts, open_windows = False, set()
    win = "YOLO detectors"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    src = frames(args)
    while True:
        if not frozen or last is None:
            try:
                img, _name = next(src)
            except StopIteration:
                break
            if img is None:
                continue
            if mirror:
                img = cv2.flip(img, 1)
            last = img
        img = last
        tiles = {}
        for k, name in KINDS:
            if show[k]:
                tiles[k] = panel(k, name, img, models, device, times, args.conf)[0]
        if not tiles:
            grid = np.zeros((360, 640, 3), np.uint8)
            put(grid, "all panels hidden - press 1-4", (20, 180), 0.8)
        else:
            th, tw = next(iter(tiles.values())).shape[:2]
            blank = np.full((th, tw, 3), 16, np.uint8)
            cells = [tiles.get(k, blank) for k, _ in KINDS if k in tiles]
            if len(cells) == 1:
                grid = cells[0]
            elif len(cells) == 2:
                grid = np.hstack(cells)
            else:
                cells += [blank] * (4 - len(cells))
                gap = np.full((th, 8, 3), 40, np.uint8)
                row1 = np.hstack([cells[0], gap, cells[1]])
                row2 = np.hstack([cells[2], gap, cells[3]])
                grid = np.vstack([row1, np.full((8, row1.shape[1], 3), 40, np.uint8), row2])
        if frozen:
            put(grid, "FROZEN", (grid.shape[1] - 140, grid.shape[0] - 20), 0.9, (80, 200, 255), 2)
        cv2.imshow(win, grid)
        for k, name in KINDS:
            wn = f"YOLO - {name}"
            if popouts and k in tiles:
                cv2.imshow(wn, tiles[k])
                open_windows.add(wn)
            elif wn in open_windows:
                cv2.destroyWindow(wn)
                open_windows.discard(wn)
        key = cv2.waitKey(1 if not frozen else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            frozen = not frozen
        elif key == ord("i"):
            mirror = not mirror
        elif key == ord("p"):
            popouts = not popouts
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            k = KINDS[key - ord("1")][0]
            show[k] = not show[k]
        elif key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            d = os.path.join(args.out, stamp)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "all_systems.png"), grid)
            cv2.imwrite(os.path.join(d, "input.png"), img)
            for k, t in tiles.items():
                cv2.imwrite(os.path.join(d, f"{k}.png"), t)
            print(f"[showcase] saved {d}/")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
