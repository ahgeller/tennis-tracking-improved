"""Test C, step 1: run the stream_s4 prepass over the mixed_v1 holdout and dump, per frame, the top
K local maxima of the averaged heat grid instead of the one argmax the tracker keeps.

The detector is the tracker's own (``finetune.data.detect.ball_detector`` at conf 0, the same
object raw mode scores). The only change is a wrapper around ``ball_gpu.HeatRing.take``: before
the ring hands back its argmax peaks, the same mean grids go through a 3x3 non-maximum
suppression and a top-K, and those rows are read back beside the tracker's. Nothing in the
tracker is edited. Rank 1 of the dump equals the tracker's answer (checked in score.py).

Output: one ``<clip>.npz`` per clip in --out with
  frame  (N,)      source frame index
  conf   (N, K)    confidence, 0 where fewer than K peaks cleared --min-conf (sorted descending)
  x, y   (N, K)    the decoded point in native pixels (the tracker's decode; y offset 0, as Config)
and the clip's width / height / fps.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
import finetune  # noqa: E402,F401  puts system/ on sys.path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from finetune.data.detect import ball_detector  # noqa: E402
from finetune.data.workspace import video_meta  # noqa: E402
from finetune.scoring.archive import list_clips  # noqa: E402
from tennis_tracker.engines import ball_gpu  # noqa: E402

WEIGHTS = REPO / "system" / "models" / "candidates" / "stream_s4_e60.pt"


def topk_peaks(means: torch.Tensor, k: int, min_conf: float) -> torch.Tensor:
    """``(N, K, 4)``: per frame the K highest 3x3 local maxima of the heat channel at or above
    `min_conf` (confidence, x offset, y offset, flat cell), confidence 0 and cell -1 for the rows
    below K. Ties in a plateau keep the first cell, as the argmax does."""
    heat = means[:, 0:1]                                            # (N, 1, rows, cols)
    pooled = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    keep = (heat >= pooled) & (heat >= min_conf)
    # An exact plateau keeps every cell of it; at the 0.05+ level in a float32 mean that is rare,
    # and topk takes the lowest index first, as the argmax does.
    scores = torch.where(keep, heat, torch.zeros_like(heat)).flatten(1)          # (N, cells)
    conf, cells = scores.topk(k, dim=1)                                         # sorted descending
    flat = means.flatten(2)                                                      # (N, 3, cells)
    picked = flat.gather(2, cells.unsqueeze(1).expand(-1, 3, -1))               # (N, 3, K)
    picked = picked.permute(0, 2, 1).contiguous()                                # (N, K, 3)
    empty = conf <= 0
    picked[..., 0] = torch.where(empty, torch.zeros_like(conf), picked[..., 0])
    cell_f = torch.where(empty, torch.full_like(conf, -1.0), cells.to(torch.float32))
    return torch.cat([picked, cell_f.unsqueeze(2)], dim=2)


class PeakTap:
    """Wraps ``HeatRing.take`` for the life of one clip, collecting (frames, Readback of top-K)."""

    def __init__(self, k: int, min_conf: float):
        self.k, self.min_conf = k, min_conf
        self.pending: List[tuple] = []
        self.grid = None

    def install(self) -> None:
        tap = self
        original = ball_gpu.HeatRing.take

        def take(ring, indices):
            slots = torch.tensor([index % ring.slots for index in indices], dtype=torch.long).to(ring.device, non_blocking=True)
            sums = ring.sums.index_select(0, slots)
            means = sums * ring.counts.index_select(0, slots).reciprocal().view(-1, 1, 1, 1)
            tap.grid = (int(means.shape[-2]), int(means.shape[-1]))
            tap.pending.append((list(indices), ball_gpu.Readback(topk_peaks(means, tap.k, tap.min_conf))))
            return original(ring, indices)

        self._original = original
        ball_gpu.HeatRing.take = take

    def uninstall(self) -> None:
        ball_gpu.HeatRing.take = self._original

    def collect(self, width: int, height: int) -> Dict[str, np.ndarray]:
        rows, cols = self.grid
        frames, peaks = [], []
        for indices, readback in self.pending:
            values = readback.wait()
            frames.extend(indices)
            peaks.append(np.array(values, dtype=np.float32))
        self.pending.clear()
        frame = np.array(frames, dtype=np.int64)
        stacked = np.concatenate(peaks, axis=0) if peaks else np.zeros((0, self.k, 4), np.float32)
        order = np.argsort(frame, kind="stable")
        frame, stacked = frame[order], stacked[order]
        conf = stacked[..., 0].astype(np.float32)
        cell = stacked[..., 3]
        row = np.floor_divide(np.maximum(cell, 0), cols)
        col = np.maximum(cell, 0) - row * cols
        # The tracker's decode (ball_gpu.peak_points), in float32 then float64 the same way.
        x = ((stacked[..., 1].astype(np.float64) + col) * width / cols).astype(np.float32)
        y = ((stacked[..., 2].astype(np.float64) + row) * height / rows).astype(np.float32)
        x[conf <= 0] = np.nan
        y[conf <= 0] = np.nan
        return {"frame": frame, "conf": conf, "x": x, "y": y}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(HERE.parent / "peaks"))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--holdout", default="mixed_v1")
    parser.add_argument("--clips", nargs="*", default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = REPO / "finetune"
    clips = list_clips(args.clips, archive, False, args.holdout)
    print(f"[dump] {len(clips)} clips, weights {args.weights}, K={args.k}, min conf {args.min_conf}")
    detector = ball_detector(Path(args.weights), conf=0.0)
    tap = PeakTap(args.k, args.min_conf)
    tap.install()
    try:
        for video, csv in clips:
            meta = video_meta(video)
            started = time.perf_counter()
            found = detector.detect_clip(video, meta.fps, meta.width, meta.height, meta.total)
            data = tap.collect(meta.width, meta.height)
            # The tracker's own rank-1 answer, for the identity check in score.py.
            tracker_conf = np.zeros(len(found), np.float32)
            tracker_x = np.full(len(found), np.nan, np.float32)
            tracker_y = np.full(len(found), np.nan, np.float32)
            for index, detections in enumerate(found):
                if detections:
                    box, confidence = detections[0]
                    tracker_conf[index] = confidence
                    tracker_x[index] = (box[0] + box[2]) / 2.0
                    tracker_y[index] = (box[1] + box[3]) / 2.0
            np.savez_compressed(out / f"{video.stem}.npz", width=meta.width, height=meta.height, fps=meta.fps,
                                total=len(found), tracker_conf=tracker_conf, tracker_x=tracker_x, tracker_y=tracker_y,
                                **data)
            print(f"[dump] {video.stem}: {len(found)} frames, {len(data['frame'])} answered, "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    finally:
        tap.uninstall()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
