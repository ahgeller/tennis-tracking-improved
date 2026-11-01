"""Step 2 of test D: are e60's misses where the camera moves? For every labelled visible frame of
a clip, the recorded pipeline run's answer (hit within 10 px at 1080p, else miss) is bucketed by
the camera's pan speed at that frame: the best integer shift of the 768x432 luma between the
frame and the one ``--span`` source frames earlier (exhaustive search within ``--radius`` px),
divided by the transitions in that span, in 768 px per transition. Also the per-transition
integer shift (t against t - stride) for the same frames.

    python experiments/tier0/D_frame_alignment/misses_by_shift.py --run <opt1_gate>/e60_0.42 --clips grid_match8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(REPOSITORY))
import finetune  # noqa: E402,F401  (puts system/ on the path)
from finetune.data.labels import read_labels  # noqa: E402
from finetune.data.workspace import video_meta  # noqa: E402
from finetune.data.detector_metrics import HIT_PX, scale_for_width  # noqa: E402
from tennis_tracker.frame_rate import require_phase_stride  # noqa: E402

MODEL = (768, 432)
BUCKETS = [(0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, math.inf)]


def luma(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), MODEL, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def best_shift(frame: np.ndarray, previous: np.ndarray, radius: int):
    h, w = frame.shape
    inner = frame[radius:h - radius, radius:w - radius]
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            moved = previous[radius - dy:h - radius - dy, radius - dx:w - radius - dx]
            value = float(np.abs(inner - moved).mean())
            if best is None or value < best[0]:
                best = (value, dx, dy)
    return best[1], best[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="folder of <clip>_gridtracknet.json files (a pipeline run)")
    parser.add_argument("--clips", nargs="+", default=["grid_match8", "grid_match16", "grid_match22"])
    parser.add_argument("--span", type=int, default=30)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--videos", default=str(REPOSITORY / "finetune" / "videos"))
    parser.add_argument("--labels", default=str(REPOSITORY / "finetune" / "labels"))
    args = parser.parse_args()
    for clip in args.clips:
        video = Path(args.videos) / f"{clip}.mp4"
        meta = video_meta(video)
        labels = read_labels(Path(args.labels) / f"{clip}_ball.csv", meta.width, meta.height)
        payload = json.load(open(Path(args.run) / f"{clip}_gridtracknet.json"))
        rows = {row["frame"]: row for row in payload["frames"]}
        scale = scale_for_width(meta.width)
        stride = require_phase_stride(meta.fps)
        wanted = {frame for frame, label in labels.items() if label is not None}
        capture = cv2.VideoCapture(str(video))
        history, index = [], 0
        speed, step = {}, {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pic = luma(frame)
            history.append(pic)
            if len(history) > args.span:
                history.pop(0)
            if index in wanted and len(history) > stride:
                dx, dy = best_shift(pic, history[-stride - 1], 3)
                step[index] = math.hypot(dx, dy)
                if len(history) == args.span:
                    dx, dy = best_shift(pic, history[0], args.radius)
                    speed[index] = math.hypot(dx, dy) / ((args.span - 1) / stride)
            index += 1
        capture.release()
        print(f"\n{clip}: {len(wanted)} visible labels, {len(speed)} with a pan-speed estimate "
              f"(span {args.span - 1} source frames, radius {args.radius} px)")
        print(f"{'pan px/transition':>20s} {'n':>5s} {'hits':>5s} {'recall':>7s}")
        for low, high in BUCKETS:
            frames = [f for f, s in speed.items() if (s == 0.0 if high == 0.0 else low < s <= high)]
            hits = 0
            for f in frames:
                row = rows.get(f)
                label = labels[f]
                if row and row.get("present") and math.hypot(row["x"] - label[0], row["y"] - label[1]) * scale <= HIT_PX:
                    hits += 1
            name = "0" if high == 0.0 else f"({low}, {high}]"
            print(f"{name:>20s} {len(frames):5d} {hits:5d} {hits / len(frames) if frames else float('nan'):7.3f}")
        nonzero = [f for f, s in step.items() if s > 0]
        hits_nz = sum(1 for f in nonzero if rows.get(f, {}).get("present")
                      and math.hypot(rows[f]["x"] - labels[f][0], rows[f]["y"] - labels[f][1]) * scale <= HIT_PX)
        zero = [f for f, s in step.items() if s == 0]
        hits_z = sum(1 for f in zero if rows.get(f, {}).get("present")
                     and math.hypot(rows[f]["x"] - labels[f][0], rows[f]["y"] - labels[f][1]) * scale <= HIT_PX)
        print(f"per-transition integer shift (t vs t-{stride}): zero n={len(zero)} recall={hits_z / max(len(zero), 1):.3f}; "
              f"nonzero n={len(nonzero)} recall={hits_nz / max(len(nonzero), 1):.3f}; "
              f"|shift| values: {sorted(set(round(s, 2) for s in step.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
