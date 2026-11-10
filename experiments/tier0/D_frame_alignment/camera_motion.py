"""Step 1 of test D: how much does the camera move between a transition's two frames?

For each clip, every source frame is paired with the frame ``stride`` before it (the transition
``streamnet.transition`` sees: 2 source frames back at 60 FPS, 1 at 30), both reduced to a luma
picture at the model's 768x432 and then to a small one (96x54 by default; ``--scale`` picks the
reduction), and OpenCV's ``phaseCorrelate`` (Hanning window, sub-pixel peak) estimates the global
translation. Shifts are reported in 768x432 pixels. Writes ``shifts_<clip>.csv`` (frame, dx, dy,
response) and prints the distribution per clip and the estimate's CPU cost.

    python experiments/tier0/D_frame_alignment/camera_motion.py --clips grid_match8 grid_match16 grid_match22 video1 video6
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "system"))

from tennis_tracker.frame_rate import require_phase_stride  # noqa: E402

MODEL = (768, 432)


def small_luma(frame_bgr: np.ndarray, scale: int, window_cache: dict) -> np.ndarray:
    """The frame as float32 luma at 768x432 / scale, Hanning-windowed as phaseCorrelate wants."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    model = cv2.resize(gray, MODEL, interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(model, (MODEL[0] // scale, MODEL[1] // scale), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def measure(video: Path, scale: int, limit: int = 0):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    stride = require_phase_stride(fps)
    size = (MODEL[0] // scale, MODEL[1] // scale)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    history = []
    rows = []
    cost = 0.0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or (limit and index >= limit):
            break
        small = small_luma(frame, scale, {})
        history.append(small)
        if len(history) > stride:
            previous = history[-stride - 1]
            started = time.perf_counter()
            (dx, dy), response = cv2.phaseCorrelate(previous, small, window)
            cost += time.perf_counter() - started
            rows.append((index, dx * scale, dy * scale, response))
            del history[0]
        index += 1
    capture.release()
    return stride, rows, cost / max(len(rows), 1)


def describe(name: str, rows, stride: int, cost: float) -> str:
    mag = np.hypot([row[1] for row in rows], [row[2] for row in rows])
    q = np.percentile(mag, [50, 75, 90, 99])
    above = (mag > 0.5).mean()
    above2 = (mag > 2.0).mean()
    return (f"{name:14s} stride={stride} n={len(mag):5d} |shift| px@768: median={q[0]:.2f} p75={q[1]:.2f} "
            f"p90={q[2]:.2f} p99={q[3]:.2f} max={mag.max():.2f} | >0.5px: {above:.1%}  >2px: {above2:.1%} "
            f"| phaseCorrelate {cost * 1000:.3f} ms/frame")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="+", default=["grid_match8", "grid_match16", "grid_match22", "video1", "video6"])
    parser.add_argument("--scale", type=int, default=8, help="768x432 reduced by this: 8 gives 96x54, 4 gives 192x108")
    parser.add_argument("--videos", default=str(REPOSITORY / "finetune" / "videos"))
    parser.add_argument("--out", default=str(HERE))
    parser.add_argument("--limit", type=int, default=0, help="frames per clip (0: all)")
    args = parser.parse_args()
    out = Path(args.out)
    for clip in args.clips:
        video = Path(args.videos) / f"{clip}.mp4"
        stride, rows, cost = measure(video, args.scale, args.limit)
        print(describe(clip, rows, stride, cost), flush=True)
        with open(out / f"shifts_{clip}_s{args.scale}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "dx_px768", "dy_px768", "response"])
            for frame, dx, dy, response in rows:
                writer.writerow([frame, f"{dx:.3f}", f"{dy:.3f}", f"{response:.4f}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
