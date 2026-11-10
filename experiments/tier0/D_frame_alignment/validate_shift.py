"""Step 1b of test D: is the phase-correlation shift real? For every transition, the luma residual
``mean |frame - previous|`` at 192x108 is measured at zero shift, at the phase-correlation shift
(rounded to the 192x108 pixel, 4 px at 768x432) and at the best integer shift of an exhaustive
search within ``--radius`` 192x108 pixels. A real pan shows a best shift away from zero that cuts
the residual well below the zero-shift one; a static camera shows best shift 0 and a ratio near 1.
Writes ``residual_<clip>.csv``; prints the per-clip summary.
"""
from __future__ import annotations

import argparse
import csv
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
SCALE = 4
SMALL = (MODEL[0] // SCALE, MODEL[1] // SCALE)


def luma_small(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    model = cv2.resize(gray, MODEL, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(model, SMALL, interpolation=cv2.INTER_AREA).astype(np.float32)


def residual(frame: np.ndarray, previous: np.ndarray, dx: int, dy: int) -> float:
    """mean |frame - previous shifted by (dx, dy)| over the overlap (previous moved so that
    previous[y, x] lands at [y + dy, x + dx], the motion phaseCorrelate reports)."""
    h, w = frame.shape
    xs = slice(max(0, dx), min(w, w + dx))
    ys = slice(max(0, dy), min(h, h + dy))
    xp = slice(max(0, -dx), min(w, w - dx))
    yp = slice(max(0, -dy), min(h, h - dy))
    return float(np.abs(frame[ys, xs] - previous[yp, xp]).mean())


def best_shift(frame: np.ndarray, previous: np.ndarray, radius: int):
    best = (residual(frame, previous, 0, 0), 0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            value = residual(frame, previous, dx, dy)
            if value < best[0]:
                best = (value, dx, dy)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="+", default=["grid_match8", "grid_match16", "grid_match22", "video1", "video6"])
    parser.add_argument("--radius", type=int, default=8, help="search radius in 192x108 px (8 = 32 px at 768)")
    parser.add_argument("--videos", default=str(REPOSITORY / "finetune" / "videos"))
    parser.add_argument("--out", default=str(HERE))
    parser.add_argument("--every", type=int, default=3, help="measure every n-th transition (the search is slow)")
    args = parser.parse_args()
    window = cv2.createHanningWindow(SMALL, cv2.CV_32F)
    for clip in args.clips:
        capture = cv2.VideoCapture(str(Path(args.videos) / f"{clip}.mp4"))
        stride = require_phase_stride(capture.get(cv2.CAP_PROP_FPS))
        history, rows, index = [], [], 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            small = luma_small(frame)
            history.append(small)
            if len(history) > stride:
                previous = history.pop(0)
                if index % args.every == 0:
                    (pdx, pdy), response = cv2.phaseCorrelate(previous, small, window)
                    r0 = residual(small, previous, 0, 0)
                    rp = residual(small, previous, int(round(pdx)), int(round(pdy)))
                    rb, bdx, bdy = best_shift(small, previous, args.radius)
                    rows.append((index, pdx * SCALE, pdy * SCALE, response, bdx * SCALE, bdy * SCALE, r0, rp, rb))
            index += 1
        capture.release()
        arr = np.array([row[4:] for row in rows])
        best_mag = np.hypot(arr[:, 0], arr[:, 1])
        ratio_best = arr[:, 4] / np.maximum(arr[:, 2], 1e-6)
        ratio_pc = arr[:, 3] / np.maximum(arr[:, 2], 1e-6)
        moving = best_mag >= SCALE          # best integer shift at least one 192x108 pixel away from zero
        print(f"{clip:14s} n={len(rows):4d} best-shift nonzero: {moving.mean():.1%} "
              f"(|best| median={np.median(best_mag):.1f} p90={np.percentile(best_mag, 90):.1f} px@768) | "
              f"residual ratio best/zero: median={np.median(ratio_best):.3f} p10={np.percentile(ratio_best, 10):.3f} "
              f"| phasecorr/zero: median={np.median(ratio_pc):.3f} p90={np.percentile(ratio_pc, 90):.3f} "
              f"| zero residual median={np.median(arr[:, 2]):.2f}", flush=True)
        if moving.any():
            sub = ratio_best[moving]
            print(f"{'':14s} on the {moving.sum()} nonzero-best transitions: ratio median={np.median(sub):.3f}, "
                  f"share with ratio < 0.7: {(sub < 0.7).mean():.1%}; frames: "
                  f"{[rows[i][0] for i in np.flatnonzero(moving)[:12]]}")
        with open(Path(args.out) / f"residual_{clip}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "pc_dx", "pc_dy", "response", "best_dx", "best_dy", "res_zero", "res_pc", "res_best"])
            for row in rows:
                writer.writerow([row[0]] + [f"{v:.3f}" for v in row[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
