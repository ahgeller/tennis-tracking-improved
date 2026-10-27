"""The detector metric a training run is judged by: the raw five-frame prepass (GridTrackNet or
HeatmapNet, by the weights) over whole clips scored as ``evaluate_archive.py`` scores it (``DetectorMetrics``: recall / wrong-object /
false-alarm), and the two rules read from it.

``epoch_score`` ranks the epochs of one run: recall minus the wrong-object rate minus the
false-alarm rate, the three rates at equal weight (every correct ball is worth one, every
wrong-object output and every false alarm costs one). ``best.npz`` (``best.pt``) is the scored epoch with the
highest score, a scalar with no anchor, so an early epoch with odd rates (a scratch run's first
scored epoch detected nothing: 0 / 0 / 0) cannot bar every later one, and a little recall is
traded for a large false-alarm drop (R0's epoch 9, 0.936 / 0.007 / 0.230, beats its epoch 1,
0.938 / 0.011 / 0.305). ``promotion_improves`` is the other question, asked once at the end of a
fine-tune: do the best epoch's weights beat the weights it started from? It is a guard rail,
asymmetric and anchored on the incumbent: a fine-tune with more wrong-object outputs or false
alarms than its starting weights (beyond METRIC_SLACK) is not shipped, whatever its recall.

The prepass is the tracker's own (``GridTrackNetBallDetector.detect_clip`` through
``finetune.data.detect``, the path ``evaluate_archive.py`` takes too), so the winner is judged on
exactly what the tracker will do: the same five-frame units per phase, the same trailing window,
the same first-answer-wins rule and y offset. There is no second copy of that protocol here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Set

import torch

from finetune.data.detect import ball_detector, detect_points
from finetune.data.detector_metrics import DetectorMetrics, scale_for_width
from finetune.data.workspace import video_meta
from finetune.training.clips import Clip
from finetune.training.frame_cache import TrainingUnit, training_targets

METRIC_SLACK = 0.005       # a candidate may be worse by this on wrong/false-alarm rate: 1 frame in 200
WRONG_PENALTY = 2.0        # ... and each unit of extra wrong-object rate costs this much recall
FALSE_ALARM_CONF = 0.3     # a detection at or above this on a no-ball frame makes its unit a hard negative


def score_weights(weights: Path, clips: List[Clip], device: torch.device, threshold: float = 0.5) -> DetectorMetrics:
    """Raw-detector recall / wrong-object / false-alarm of ``weights`` on whole clips (the
    evaluate_archive metric). The detector runs on CUDA only; ``device`` names which card
    (``cuda`` alone is card 0)."""
    detector = ball_detector(weights, conf=threshold, device=str(device.index or 0))
    pooled = []
    for clip, video, csv_path, _ in clips:
        meta = video_meta(video)
        labels = training_targets(csv_path, meta.width, meta.height)
        scale = scale_for_width(meta.width)
        points = detect_points(detector, video, meta)
        pairs = []
        for frame, label in sorted(labels.items()):
            point = points.get(frame)
            pairs.append((None if label is None else (label[0] * scale, label[1] * scale),
                          None if point is None else (point[0] * scale, point[1] * scale)))
        pooled.append(pairs)
    return DetectorMetrics.of_clips(pooled, 1.0)


def false_alarm_units(weights: Path, clips: List[Clip], samples: Sequence[TrainingUnit], device: torch.device,
                      threshold: float = FALSE_ALARM_CONF) -> Set[int]:
    """The indices into ``samples`` of the units in which the detector on ``weights`` fired, at or
    above ``threshold``, on a frame labelled as having no ball: one prepass over each clip the
    units come from, the hard negatives ``--oversample-false-alarms`` weights up.

    Every such firing counts, the bridged ones too (``detector_metrics.bridged_alarms``: the ball
    drawn through a one-to-three-frame occlusion the labels mark invisible). Those frames hold the
    ball, so weighting them up teaches the network to stay silent on it; a later change may leave
    them out of the hard negatives once the recipe has been measured without them."""
    detector = ball_detector(weights, conf=threshold, device=str(device.index or 0))
    videos = {clip: video for clip, video, _, _ in clips}
    fired: Dict[str, Set[int]] = {}
    for clip in sorted({sample["clip"] for sample in samples}):
        fired[clip] = set(detect_points(detector, videos[clip], video_meta(videos[clip])))
    return {index for index, sample in enumerate(samples)
            if any(point is None and frame in fired[sample["clip"]]
                   for frame, point in zip(sample["indices"], sample["points"]))}


def epoch_score(metrics: DetectorMetrics) -> float:
    """The scalar an epoch is ranked by: recall minus the wrong-object rate minus the false-alarm
    rate. Always finite: the three rates are shares (0 on an empty set), unlike the pixel
    distances, which are NaN when nothing was detected and are not read here."""
    return metrics.recall - metrics.wrong - metrics.false_alarm


def promotion_improves(candidate: DetectorMetrics, incumbent: DetectorMetrics) -> bool:
    """Does the candidate beat the incumbent: more correct ball with no more wrong-object or
    false-alarm outputs (within METRIC_SLACK, 1 frame in 200)? Asymmetric: a tie keeps the
    incumbent. The fine-tune's guard rail against its starting weights (``pick_winner``), not the
    rule the epochs of a run are ranked by (``epoch_score``)."""
    if candidate.wrong > incumbent.wrong + METRIC_SLACK:
        return False
    if candidate.false_alarm > incumbent.false_alarm + METRIC_SLACK:
        return False
    return candidate.recall - WRONG_PENALTY * max(0.0, candidate.wrong - incumbent.wrong) > incumbent.recall
