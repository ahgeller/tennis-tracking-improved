"""Score a pipeline run's court keypoints against the court labels (``labels/<clip>_court.csv``).

    python -m finetune.scoring.court_score --out-dir output/label_candidates [--holdout mixed_v1]
        [--compare before.log]

On every labelled frame each labelled ground point (the engine's fourteen, ``court_labels``) is set
against the run's point of the same name on that frame. Per clip and pooled:

- ``points``: labelled ground points; ``found``: the share of them the run gave;
- ``px_med`` / ``px_p90``: the pixel error at 1080p scale;
- ``m_med`` / ``m_p90``: the same error on the court plane in metres: the run's point carried onto
  the court through the labelled frame's own homography (fitted to its labelled ground points,
  four or more), against where that point is on a real court (``COURT_METRES``). Frames with fewer
  than four labelled ground points have no plane and give pixels only;
- ``frames``: labelled frames, and how many of them have no court from the run at all.

The net posts are labelled for a court model with geometry (their height pins the camera down);
the engine does not give them, so they are counted (``posts``) and not scored.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

import cv2
import numpy as np

from finetune.data import workspace as ws
from finetune.data.court_labels import COURT_METRES, GROUND_POINTS, CourtLabel, CourtPoints, engine_points, read_court_labels
from finetune.data.detector_metrics import scale_for_width
from finetune.scoring.label_scoring import (ALL, LabelledClip, add_run_arguments, finish, label_archive, labelled_clips,
                                            median, metric, metric_line, no_labels, percentile, share)
from finetune.scoring.tracker_events import court_keypoints_by_frame, read_run, run_json

KEYS = ("found", "px_med", "px_p90", "m_med", "m_p90")
MIN_PLANE_POINTS = 4


def plane_homography(points: CourtPoints) -> Optional[np.ndarray]:
    """Image -> court metres from a frame's labelled ground points, None with fewer than four."""
    names = [name for name in GROUND_POINTS if name in points]
    if len(names) < MIN_PLANE_POINTS:
        return None
    image = np.array([points[name] for name in names], np.float32)
    court = np.array([COURT_METRES[name][:2] for name in names], np.float32)
    matrix, _ = cv2.findHomography(image, court, 0)
    return matrix


def to_court(matrix: np.ndarray, point) -> np.ndarray:
    return cv2.perspectiveTransform(np.array([[point]], np.float32), matrix)[0, 0]


class ClipCourtScore(NamedTuple):
    clip: str
    frames: int
    frames_without_court: int
    points: int
    found: int
    px_errors: List[float]
    m_errors: List[float]
    posts: int


def score_clip(clip: str, labels: Sequence[CourtLabel], run: dict, px_scale: float) -> ClipCourtScore:
    points = found = posts = without = 0
    px, metres = [], []
    for label in labels:
        predicted = engine_points(run.get(label.frame))
        without += not predicted
        homography = plane_homography(label.points)
        for name, (x, y) in label.points.items():
            if name not in GROUND_POINTS:
                posts += 1
                continue
            points += 1
            if name not in predicted:
                continue
            found += 1
            px_, py_ = predicted[name]
            px.append(math.hypot(px_ - x, py_ - y) * px_scale)
            if homography is not None:
                cx, cy = to_court(homography, predicted[name])
                mx, my = COURT_METRES[name][:2]
                metres.append(math.hypot(float(cx) - mx, float(cy) - my))
    return ClipCourtScore(clip, len(labels), without, points, found, px, metres, posts)


def pooled(scores: Sequence[ClipCourtScore]) -> ClipCourtScore:
    return ClipCourtScore(ALL, sum(s.frames for s in scores), sum(s.frames_without_court for s in scores),
                          sum(s.points for s in scores), sum(s.found for s in scores),
                          [v for s in scores for v in s.px_errors], [v for s in scores for v in s.m_errors],
                          sum(s.posts for s in scores))


def format_line(score: ClipCourtScore) -> str:
    return metric_line(score.clip, [
        [f"frames={score.frames:3d}", f"no_court={score.frames_without_court:3d}", f"points={score.points:4d}",
         metric("found", share(score.found, score.points))],
        [metric("px_med", median(score.px_errors), "5.1f", "px"), metric("px_p90", percentile(score.px_errors, 0.9), "5.1f", "px")],
        [metric("m_med", median(score.m_errors), "5.2f", "m"), metric("m_p90", percentile(score.m_errors, 0.9), "5.2f", "m")],
        [f"posts={score.posts:3d}"],
    ])


def score_run(clips: Sequence[LabelledClip], out_dir: Path, model: str) -> List[str]:
    lines = [f"=== court keypoints [{model}] ==="]
    scores = []
    for clip, _, court_path in clips:
        payload = read_run(run_json(out_dir, clip, model))
        score = score_clip(clip, read_court_labels(court_path), court_keypoints_by_frame(payload),
                           scale_for_width(int(payload["video"]["width"])))
        scores.append(score)
        lines.append(format_line(score))
    lines.append(format_line(pooled(scores)))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    archive = args.archive or label_archive()
    clips = labelled_clips(args.clips, archive, args.include_excluded, args.holdout, ws.court_csv)
    if not clips:
        return no_labels("court", archive, "labels/<clip>_court.csv")
    print(f"[archive] {archive}  ({len(clips)} clip{'' if len(clips) == 1 else 's'} with court labels: {', '.join(clip for clip, _, _ in clips)})")
    try:
        lines = [line for model in args.models for line in score_run(clips, args.out_dir, model)]
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return finish(lines, args.compare, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
