"""Score a pipeline run's racket hits against the hand-labelled ones (``labels/<clip>_hits.csv``).

    python -m finetune.scoring.hits_score --out-dir output/label_candidates [--holdout mixed_v1]
        [--tolerance-s 0.1] [--compare before.log]

The run's hits are its racket contacts (``tracker_events.tracker_hits``: ``court_events.contacts``,
each with its hitter's half of the court), matched one to one to the labels within the tolerance
exactly as the landing scorer matches landings (``court_events_score.match_landings``: greedy,
closest pair first; a label between two frames is half a frame from either). Per clip and pooled:
labels, hits found, matched, precision, recall, the median and p90 timing error in frames (the
label's own frames: a half-frame label is half a frame from either neighbour), the hitter accuracy
over the matched hits where both sides name one (with how many that is), and the median distance at
1080p scale between the ball where the contact was judged and the labelled contact point (the
label's own, else the ball labels', ``event_labels.resolve_hit_points``).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

from finetune.data import workspace as ws
from finetune.data.bounce_labels import landing_text
from finetune.data.detector_metrics import scale_for_width
from finetune.data.event_labels import HitLabel, read_hit_labels, resolve_hit_points
from finetune.data.labels import read_labels
from finetune.scoring.court_events_score import TOLERANCE_S, match_landings
from finetune.scoring.label_scoring import (ALL, LabelledClip, add_run_arguments, finish, label_archive, labelled_clips,
                                            median, metric, metric_line, no_labels, percentile, share)
from finetune.scoring.tracker_events import TrackerHit, court_block, read_run, run_fps, run_json, tracker_hits
from tennis_tracker.court.court_scene import seconds_to_frames

KEYS = ("precision", "recall", "dt_med", "hitter", "px_med")


class ClipHitScore(NamedTuple):
    clip: str
    labels: int
    found: int
    timing_frames: List[float]          # |hit - label| per matched pair
    hitter_right: int
    hitter_judged: int
    px_errors: List[float]
    unmatched_labels: List[HitLabel]
    unmatched_hits: List[TrackerHit]

    @property
    def matched(self) -> int:
        return len(self.timing_frames)


def score_clip(clip: str, labels: Sequence[HitLabel], hits: Sequence[TrackerHit], fps: float, tolerance_s: float,
               px_scale: float) -> ClipHitScore:
    tolerance = max(0, seconds_to_frames(tolerance_s, fps))
    pairs = match_landings([label.frame for label in labels], [hit.frame for hit in hits], tolerance)
    timing, px = [], []
    right = judged = 0
    for li, hi in pairs:
        label, hit = labels[li], hits[hi]
        timing.append(abs(hit.frame - label.frame))
        if label.hitter is not None and hit.hitter is not None:
            judged += 1
            right += label.hitter == hit.hitter
        if label.point is not None and hit.ball is not None:
            px.append(math.hypot(hit.ball[0] - label.point[0], hit.ball[1] - label.point[1]) * px_scale)
    used_labels, used_hits = {li for li, _ in pairs}, {hi for _, hi in pairs}
    return ClipHitScore(clip, len(labels), len(hits), timing, right, judged, px,
                        [label for k, label in enumerate(labels) if k not in used_labels],
                        [hit for k, hit in enumerate(hits) if k not in used_hits])


def pooled(scores: Sequence[ClipHitScore]) -> ClipHitScore:
    return ClipHitScore(ALL, sum(s.labels for s in scores), sum(s.found for s in scores),
                        [v for s in scores for v in s.timing_frames], sum(s.hitter_right for s in scores),
                        sum(s.hitter_judged for s in scores), [v for s in scores for v in s.px_errors], [], [])


def format_line(score: ClipHitScore) -> str:
    return metric_line(score.clip, [
        [f"labels={score.labels:4d}", f"found={score.found:4d}", f"matched={score.matched:4d}",
         metric("precision", share(score.matched, score.found)), metric("recall", share(score.matched, score.labels))],
        [metric("dt_med", median(score.timing_frames), "4.1f", "fr"), metric("dt_p90", percentile(score.timing_frames, 0.9), "4.1f", "fr")],
        [metric("hitter", share(score.hitter_right, score.hitter_judged)), f"hitter_n={score.hitter_judged:3d}"],
        [metric("px_med", median(score.px_errors), "5.1f", "px")],
    ])


def unmatched_lines(score: ClipHitScore, fps: float) -> List[str]:
    lines = []
    if score.unmatched_labels:
        lines.append("    unmatched labels: " + ", ".join(
            f"{landing_text(label.frame)} ({label.frame / fps:.2f}s)" for label in score.unmatched_labels))
    if score.unmatched_hits:
        lines.append("    unmatched hits:   " + ", ".join(
            f"{hit.frame} ({hit.frame / fps:.2f}s, {hit.source})" for hit in score.unmatched_hits))
    return lines


def score_run(clips: Sequence[LabelledClip], out_dir: Path, model: str, tolerance_s: float) -> List[str]:
    """The report for one model's run; FileNotFoundError or ValueError name a JSON missing or
    written without --court."""
    lines = [f"=== hits [{model}] tolerance={tolerance_s:.2f}s ==="]
    scores = []
    for clip, csv_path, hits_path in clips:
        path = run_json(out_dir, clip, model)
        payload = read_run(path)
        block = court_block(payload, path)
        fps, width, height = run_fps(payload), int(payload["video"]["width"]), int(payload["video"]["height"])
        labels = resolve_hit_points(read_hit_labels(hits_path), read_labels(csv_path, width, height))
        score = score_clip(clip, labels, tracker_hits(payload, block), fps, tolerance_s, scale_for_width(width))
        scores.append(score)
        lines.append(format_line(score))
        lines.extend(unmatched_lines(score, fps))
    lines.append(format_line(pooled(scores)))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--tolerance-s", type=float, default=TOLERANCE_S,
                        help=f"a hit within this many seconds of a label matches it (default {TOLERANCE_S})")
    args = parser.parse_args(argv)
    archive = args.archive or label_archive()
    clips = labelled_clips(args.clips, archive, args.include_excluded, args.holdout, ws.hits_csv)
    if not clips:
        return no_labels("hit", archive, "labels/<clip>_hits.csv")
    print(f"[archive] {archive}  ({len(clips)} clip{'' if len(clips) == 1 else 's'} with hit labels: {', '.join(clip for clip, _, _ in clips)})")
    try:
        lines = [line for model in args.models for line in score_run(clips, args.out_dir, model, args.tolerance_s)]
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return finish(lines, args.compare, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
