"""Score a pipeline run's court keypoints against the court labels (``labels/<clip>_court.csv``).

    python -m finetune.scoring.court_score --out-dir output/label_candidates [--holdout mixed_v1]
        [--compare before.log]

On every labelled frame each labelled ground point (the engine's fourteen, ``court_labels``) is set
against the run's point of the same name on that frame. Per clip and pooled:

- ``points``: labelled ground points; ``found``: the share of them the run gave;
- ``px_med`` / ``px_p90``: the pixel error at 1080p scale;
- ``m_med`` / ``m_p90``: the same error on the court plane in metres: the run's point carried onto
  the court through the labelled frame's own homography (fitted to its labelled ground points),
  against where that point is on a real court (``COURT_METRES``). Only a plane the labels pin down
  is used (``fit_plane``): four or more ground points spanning at least ``MIN_SPAN_SHARE`` of the
  court (four along one baseline fit a homography exactly and put the rest of the court anywhere),
  and a fit that puts each of them within ``MAX_FIT_M`` of where it belongs (one that misses its own
  points has a mislabelled one). Other frames give pixels only, are counted (``no_plane``) and
  say why in an indented note;
- ``frames``: labelled frames, and how many of them have no court from the run at all.

The net posts are labelled for a court model with geometry (their height pins the camera down);
the engine does not give them, so they are counted (``posts``) and not scored.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from finetune.data import workspace as ws
from finetune.data.court_labels import (COURT_LENGTH_M, COURT_METRES, DOUBLES_WIDTH_M, GROUND_POINTS, CourtLabel, CourtPoints,
                                        engine_points, read_court_labels)
from finetune.data.detector_metrics import scale_for_width
from finetune.scoring.label_scoring import (ALL, LabelledClip, median, metric, metric_line, percentile, run_scorer, share)
from finetune.scoring.tracker_events import court_keypoints_by_frame, read_run, run_json

KEYS = ("found", "px_med", "px_p90", "m_med", "m_p90")
MIN_PLANE_POINTS = 4
MIN_SPAN_SHARE = 0.25     # the labelled ground points' hull covers at least this share of the doubles court
MAX_FIT_M = 0.25          # ... and the plane fitted to them puts each within this of where it belongs


def fit_plane(points: CourtPoints) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """(image -> court metres from a frame's labelled ground points, None) when they pin the plane
    down; (None, why not) when they do not: too few, spanning too little of the court, or a fit
    that misses one of them (see the module docstring)."""
    names = [name for name in GROUND_POINTS if name in points]
    if len(names) < MIN_PLANE_POINTS:
        return None, f"{len(names)} ground points labelled, a plane needs {MIN_PLANE_POINTS}"
    image = np.array([points[name] for name in names], np.float32)
    court = np.array([COURT_METRES[name][:2] for name in names], np.float32)
    span = float(cv2.contourArea(cv2.convexHull(court)))
    needed = MIN_SPAN_SHARE * COURT_LENGTH_M * DOUBLES_WIDTH_M
    if span < needed:
        return None, f"the points labelled span {span:.0f} m2 of the court, under {needed:.0f}: too little to pin the plane"
    matrix, _ = cv2.findHomography(image, court, 0)
    if matrix is None or not np.isfinite(matrix).all():
        return None, "no plane fits the points labelled"
    misses = [float(np.hypot(*(to_court(matrix, point) - spot))) for point, spot in zip(image, court)]
    worst = int(np.argmax(misses))
    if misses[worst] > MAX_FIT_M:
        return None, (f"the plane fitted misses its own point {names[worst]} by {misses[worst]:.2f} m "
                      f"(over {MAX_FIT_M} m): a point is mislabelled")
    return matrix, None


def plane_homography(points: CourtPoints) -> Optional[np.ndarray]:
    """The plane of ``fit_plane``, None when the points do not pin it down."""
    return fit_plane(points)[0]


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
    frames_off_plane: int = 0           # labelled frames whose points do not pin the court plane: pixels only
    notes: Tuple[str, ...] = ()         # why, per frame


def score_clip(clip: str, labels: Sequence[CourtLabel], run: dict, px_scale: float) -> ClipCourtScore:
    points = found = posts = without = 0
    px, metres, notes = [], [], []
    for label in labels:
        predicted = engine_points(run.get(label.frame))
        without += not predicted
        homography, reason = fit_plane(label.points)
        if reason is not None:
            notes.append(f"frame {label.frame}: {reason}")
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
    return ClipCourtScore(clip, len(labels), without, points, found, px, metres, posts, len(notes), tuple(notes))


def pooled(scores: Sequence[ClipCourtScore]) -> ClipCourtScore:
    return ClipCourtScore(ALL, sum(s.frames for s in scores), sum(s.frames_without_court for s in scores),
                          sum(s.points for s in scores), sum(s.found for s in scores),
                          [v for s in scores for v in s.px_errors], [v for s in scores for v in s.m_errors],
                          sum(s.posts for s in scores), sum(s.frames_off_plane for s in scores))


def format_line(score: ClipCourtScore) -> str:
    return metric_line(score.clip, [
        [f"frames={score.frames:3d}", f"no_court={score.frames_without_court:3d}", f"points={score.points:4d}",
         metric("found", share(score.found, score.points))],
        [metric("px_med", median(score.px_errors), "5.1f", "px"), metric("px_p90", percentile(score.px_errors, 0.9), "5.1f", "px")],
        [metric("m_med", median(score.m_errors), "5.2f", "m"), metric("m_p90", percentile(score.m_errors, 0.9), "5.2f", "m"),
         f"no_plane={score.frames_off_plane:3d}"],
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
        lines.extend(f"    {clip}: no court plane on {note}" for note in score.notes)
    lines.append(format_line(pooled(scores)))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_scorer(argv, __doc__ or "", "court", ws.court_csv,
                      lambda clips, out_dir, model, args: score_run(clips, out_dir, model), KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
