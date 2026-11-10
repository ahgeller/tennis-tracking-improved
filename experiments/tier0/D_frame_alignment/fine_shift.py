"""Step 1c of test D: the integer shift at the model's full 768x432 luma, by exhaustive search within
``--radius`` px, for the transition (t against t - stride) and for a longer span (t against
t - ``--span`` source frames), so a slow pan below one 768 px per transition still shows up as drift.
Prints per clip the share of transitions whose best shift is nonzero and the drift distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "system"))

from tennis_tracker.frame_rate import require_phase_stride  # noqa: E402

MODEL = (768, 432)


def luma(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, MODEL, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def residual(frame: np.ndarray, previous: np.ndarray, dx: int, dy: int) -> float:
    h, w = frame.shape
    xs, ys = slice(max(0, dx), min(w, w + dx)), slice(max(0, dy), min(h, h + dy))
    xp, yp = slice(max(0, -dx), min(w, w - dx)), slice(max(0, -dy), min(h, h - dy))
    return float(np.abs(frame[ys, xs] - previous[yp, xp]).mean())


def best_shift(frame: np.ndarray, previous: np.ndarray, radius: int):
    # Crop a margin so every shift compares the same number of pixels (no border bias).
    h, w = frame.shape
    inner = frame[radius:h - radius, radius:w - radius]
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            moved = previous[radius - dy:h - radius - dy, radius - dx:w - radius - dx]
            value = float(np.abs(inner - moved).mean())
            if best is None or value < best[0]:
                best = (value, dx, dy)
    zero = residual(frame, previous, 0, 0)
    return best, zero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=["grid_match8", "grid_match16", "grid_match22", "video1", "video6"])
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--span", type=int, default=30)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--videos", default=str(REPOSITORY / "finetune" / "videos"))
    args = parser.parse_args()
    for clip in args.clips:
        capture = cv2.VideoCapture(str(Path(args.videos) / f"{clip}.mp4"))
        stride = require_phase_stride(capture.get(cv2.CAP_PROP_FPS))
        history, index = [], 0
        transition, drift = [], []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pic = luma(frame)
            history.append(pic)
            if len(history) > args.span:
                history.pop(0)
            if index % args.every == 0 and len(history) > stride:
                (value, dx, dy), zero = best_shift(pic, history[-stride - 1], args.radius)
                transition.append((index, dx, dy, value, zero))
                if len(history) == args.span:
                    (value, dx, dy), zero = best_shift(pic, history[0], args.radius)
                    drift.append((index, dx, dy, value, zero))
            index += 1
        capture.release()
        t = np.array(transition, dtype=float)
        d = np.array(drift, dtype=float)
        tmag = np.hypot(t[:, 1], t[:, 2])
        dmag = np.hypot(d[:, 1], d[:, 2])
        print(f"{clip:14s} transition (t vs t-{stride}): n={len(t)} nonzero best shift {np.mean(tmag > 0):.1%}, "
              f"|shift| p90={np.percentile(tmag, 90):.0f} max={tmag.max():.0f} px, "
              f"residual best/zero median={np.median(t[:, 3] / np.maximum(t[:, 4], 1e-6)):.3f}", flush=True)
        print(f"{'':14s} drift (t vs t-{args.span - 1}): n={len(d)} nonzero {np.mean(dmag > 0):.1%}, "
              f"|shift| median={np.median(dmag):.1f} p90={np.percentile(dmag, 90):.1f} max={dmag.max():.0f} px, "
              f"residual best/zero median={np.median(d[:, 3] / np.maximum(d[:, 4], 1e-6)):.3f}; "
              f"frames with drift >= 2 px: {[int(f) for f in d[dmag >= 2, 0][:15]]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
