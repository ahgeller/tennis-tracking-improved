"""Score a pipeline run's rallies against the hand-labelled ones (``labels/<clip>_rallies.csv``).

    python -m finetune.scoring.rallies_score --out-dir output/label_candidates [--holdout mixed_v1]
        [--compare before.log]

The run's rallies are its rally legs grouped into points (``tracker_events.rallies_from_legs``).
Only labelled rallies with both ends marked are scored. Each labelled rally is paired with at most
one of the run's, greedily by overlap (intersection over union of the two frame spans, the largest
first, overlapping ones only). Per clip and pooled: rallies labelled, rallies found, the mean
overlap over the labelled rallies (an unpaired one counts 0: the per-rally overlap), recall and
precision at an overlap of ``MATCH_IOU`` or more, and the median and p90 start and end errors in
seconds over the paired rallies. The tracker starts a rally at the serve's hit and the label at the
toss, so a start error of a second or so is the two definitions, not a miss.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

from finetune.data import workspace as ws
from finetune.data.event_labels import read_rally_labels
from finetune.scoring.label_scoring import (ALL, LabelledClip, add_run_arguments, finish, label_archive, labelled_clips,
                                            median, metric, metric_line, no_labels, percentile, share)
from finetune.scoring.tracker_events import court_block, read_run, run_fps, run_json, tracker_rallies

MATCH_IOU = 0.5
KEYS = ("iou_mean", "recall@.5", "precision@.5", "start_med", "end_med")

Span = Tuple[int, int]


def iou(a: Span, b: Span) -> float:
    """Intersection over union of two inclusive frame spans."""
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / float((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def pair_rallies(labels: Sequence[Span], found: Sequence[Span]) -> List[Tuple[int, int, float]]:
    """(label index, found index, overlap) pairs, one to one, the largest overlaps first."""
    candidates = sorted(((iou(label, other), li, fi) for li, label in enumerate(labels)
                         for fi, other in enumerate(found)), reverse=True)
    pairs, used_labels, used_found = [], set(), set()
    for overlap, li, fi in candidates:
        if overlap <= 0.0:
            break
        if li in used_labels or fi in used_found:
            continue
        pairs.append((li, fi, overlap))
        used_labels.add(li)
        used_found.add(fi)
    return sorted(pairs)


class ClipRallyScore(NamedTuple):
    clip: str
    labels: int
    found: int
    overlaps: List[float]               # per labelled rally; 0 for one left unpaired
    start_errors_s: List[float]         # |found - label| per pair
    end_errors_s: List[float]
    good: int                           # pairs with an overlap of MATCH_IOU or more


def score_clip(clip: str, labels: Sequence[Span], found: Sequence[Span], fps: float) -> ClipRallyScore:
    pairs = pair_rallies(labels, found)
    overlaps = [0.0] * len(labels)
    starts, ends = [], []
    for li, fi, overlap in pairs:
        overlaps[li] = overlap
        starts.append(abs(found[fi][0] - labels[li][0]) / fps)
        ends.append(abs(found[fi][1] - labels[li][1]) / fps)
    return ClipRallyScore(clip, len(labels), len(found), overlaps, starts, ends,
                          sum(1 for _, _, overlap in pairs if overlap >= MATCH_IOU))


def pooled(scores: Sequence[ClipRallyScore]) -> ClipRallyScore:
    return ClipRallyScore(ALL, sum(s.labels for s in scores), sum(s.found for s in scores),
                          [v for s in scores for v in s.overlaps], [v for s in scores for v in s.start_errors_s],
                          [v for s in scores for v in s.end_errors_s], sum(s.good for s in scores))


def format_line(score: ClipRallyScore) -> str:
    mean = sum(score.overlaps) / len(score.overlaps) if score.overlaps else None
    return metric_line(score.clip, [
        [f"rallies={score.labels:3d}", f"found={score.found:3d}", metric("iou_mean", mean),
         metric("recall@.5", share(score.good, score.labels)), metric("precision@.5", share(score.good, score.found))],
        [metric("start_med", median(score.start_errors_s), "5.2f", "s"),
         metric("start_p90", percentile(score.start_errors_s, 0.9), "5.2f", "s")],
        [metric("end_med", median(score.end_errors_s), "5.2f", "s"),
         metric("end_p90", percentile(score.end_errors_s, 0.9), "5.2f", "s")],
    ])


def score_run(clips: Sequence[LabelledClip], out_dir: Path, model: str) -> List[str]:
    lines = [f"=== rallies [{model}] match at overlap >= {MATCH_IOU} ==="]
    scores = []
    for clip, _, rallies_path in clips:
        path = run_json(out_dir, clip, model)
        payload = read_run(path)
        block = court_block(payload, path)
        fps = run_fps(payload)
        labels = [(rally.start, rally.end) for rally in read_rally_labels(rallies_path) if rally.complete]
        score = score_clip(clip, labels, tracker_rallies(block, fps), fps)   # type: ignore[arg-type]
        scores.append(score)
        lines.append(format_line(score))
    lines.append(format_line(pooled(scores)))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    archive = args.archive or label_archive()
    clips = labelled_clips(args.clips, archive, args.include_excluded, args.holdout, ws.rallies_csv)
    if not clips:
        return no_labels("rally", archive, "labels/<clip>_rallies.csv")
    print(f"[archive] {archive}  ({len(clips)} clip{'' if len(clips) == 1 else 's'} with rally labels: {', '.join(clip for clip, _, _ in clips)})")
    try:
        lines = [line for model in args.models for line in score_run(clips, args.out_dir, model)]
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return finish(lines, args.compare, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
