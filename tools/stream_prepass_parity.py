"""Compare the streamed prepass of a streaming ball model (``stream_s4``) with the windowed one, unit
by unit and frame by frame, on a real clip.

    python tools/stream_prepass_parity.py --weights system/models/candidates/stream_s4_e60.pt --out output/stream_parity

Four prepasses run over the clip at confidence 0, so every covered frame reports its peak, and
every unit's 15-channel output is recorded as the prepass hands it to the heat ring:
W windowed (TensorRT unit engine), S streamed (TensorRT encoder and decoder engines), T streamed
(PyTorch FP16 halves, the fallback) and R windowed in float32 PyTorch (the reference). It writes
``parity.json``: per pair the largest heat and offset differences over the units, and per frame
whether a detection appears, moves or crosses the 0.3 / 0.4 / 0.5 thresholds. It changes no
timing, so it needs no GPU lock; the engines it builds are cached as the tracker's are.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

from tennis_tracker.config import Config  # noqa: E402
from tennis_tracker.engines import ball_gpu  # noqa: E402
from tennis_tracker.engines.ball_detectors import GridTrackNetBallDetector  # noqa: E402

RUNS = {"W": dict(ball_prepass_streaming=False), "S": {}, "T": dict(ball_prepass_runtime="torch"),
        "R": dict(ball_prepass_streaming=False, ball_prepass_runtime="torch")}
PAIRS = (("W", "S"), ("W", "R"), ("S", "R"), ("T", "R"), ("T", "S"))
Peak = Optional[Tuple[float, float, float]]


class Float32:
    """The detector's FP16 network run in float32 (the reference)."""

    def __init__(self, model: Any):
        self.model = model.float()
        for name in ("ARCH", "GRID_COLS", "GRID_ROWS", "INPUT_CHANNELS", "FRAMES_PER_UNIT"):
            setattr(self, name, getattr(model, name))

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.float())


def run(label: str, weights: Path, video: Path) -> Tuple[List[Peak], Dict[Tuple[int, ...], torch.Tensor]]:
    """One prepass: each frame's peak (x, y, confidence) and each unit's output (FP16, on the host)."""
    capture = cv2.VideoCapture(str(video))
    fps, width, height = capture.get(cv2.CAP_PROP_FPS), int(capture.get(3)), int(capture.get(4))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    detector = GridTrackNetBallDetector(str(weights), Config(conf=0.0, device="0", **RUNS[label]))
    if label == "R":
        detector.model = Float32(detector.model)
    units: Dict[Tuple[int, ...], torch.Tensor] = {}
    add = ball_gpu.HeatRing.add

    def recording(ring: Any, output: torch.Tensor, cut: Any) -> None:
        for row, indices in enumerate(cut):
            units[tuple(indices)] = output[row].detach().to(torch.float16).cpu()
        add(ring, output, cut)

    with patch.object(ball_gpu.HeatRing, "add", recording):
        rows = detector.detect_clip(video, fps, width, height, total)
    peaks: List[Peak] = []
    for row in rows:
        box, confidence = row[0] if row else (None, None)
        peaks.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, confidence) if box is not None else None)
    return peaks, units


def compare_units(first: Dict[Tuple[int, ...], torch.Tensor], second: Dict[Tuple[int, ...], torch.Tensor]) -> Dict[str, Any]:
    if first.keys() != second.keys():
        raise RuntimeError(f"the runs decoded different units ({len(first)} and {len(second)})")
    heat, offsets = [], []
    for key in first:
        difference = (first[key].float() - second[key].float()).abs()
        heat.append(float(difference[0::3].max()))
        offsets.append(max(float(difference[1::3].max()), float(difference[2::3].max())))
    return {"units": len(first), "max_heat": max(heat), "p99_unit_max_heat": float(np.percentile(heat, 99)),
            "median_unit_max_heat": float(np.median(heat)), "max_offset": max(offsets)}


def compare_frames(first: List[Peak], second: List[Peak]) -> Dict[str, Any]:
    both = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    moved = [float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in both]
    result: Dict[str, Any] = {
        "frames": len(first), "covered": len(both),
        "presence_differs": sum((a is None) != (b is None) for a, b in zip(first, second)),
        "peak_moved_over_4px": sum(move > 4 for move in moved), "max_peak_move_px": max(moved, default=0.0),
        "max_conf_diff": max((abs(a[2] - b[2]) for a, b in both), default=0.0)}
    for threshold in (0.3, 0.4, 0.5):
        on = [[peak is not None and peak[2] >= threshold for peak in run] for run in (first, second)]
        result[f"detections@{threshold}"] = [sum(on[0]), sum(on[1])]
        result[f"differ@{threshold}"] = sum(a != b for a, b in zip(*on))
        result[f"relocated@{threshold}"] = sum(1 for (a, b), p, q in zip(zip(first, second), *on)
                                              if p and q and np.hypot(a[0] - b[0], a[1] - b[1]) > 4)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="a streaming model's .pt record")
    parser.add_argument("--video", type=Path, default=ROOT / "system" / "sample" / "pomona.mp4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {}
    for label in RUNS:
        runs[label] = run(label, args.weights, args.video)
        torch.cuda.empty_cache()
    report = {f"{a}-{b}": {"units": compare_units(runs[a][1], runs[b][1]), "frames": compare_frames(runs[a][0], runs[b][0])}
              for a, b in PAIRS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
