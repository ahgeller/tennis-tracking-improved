"""The detector metric: how a set of (label, prediction) pairs is scored, and the thresholds at
which a label and a raw detection are said to agree.

Distances are measured in 1080p pixels (a 1920-wide frame as it is; a smaller frame scaled up),
so clips of different sizes pool. ``DetectorMetrics.of`` is the one scorer: evaluate_archive.py
reports it, train_gridtracknet.py picks the better of the trained and the starting weights by it,
and ``winner.json`` stores it as a dict with these very field names (``wrong`` is the wrong-object
rate everywhere; no reader has a second spelling). The agreement thresholds are
``finetune.data.alignment``'s (the label audit and realign.py).

A prediction on a frame labelled invisible is a false alarm. Most of the pipeline's are not the
ball placed where it is not: the labels mark a ball behind a player, the net post or a frame the
labeller could not place as invisible for one to three frames, and the tracker draws the ball
straight through the occlusion. ``bridged`` counts those: an output inside a run of at most
``BRIDGE_MAX_RUN`` invisible labels with visible labels on both sides, within ``BRIDGE_PX`` (1080p)
of the straight line between the two flanking labels. ``false_alarm`` stays the total, bridged
included, so a log from before the split still compares; ``false_alarm - bridged`` is the share
that is the ball somewhere it was not. Runs are counted over consecutive pairs, so a caller
pooling clips hands ``of_clips`` one sequence per clip: a clip's first labels never flank the
previous clip's last.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

REFERENCE_WIDTH = 1920.0    # the frame size pixel distances are quoted at ...
REFERENCE_HEIGHT = 1080.0   # ... (1080p); a 720p clip's pixel counts 1.5 times
HIT_PX = 10.0               # a prediction counts as correct within this many pixels
WRONG_PX = 30.0             # beyond this it is tracking something else
BRIDGE_PX = 10.0            # a false alarm this close to the line between its flanking labels bridged an occlusion
BRIDGE_MAX_RUN = 3          # ... when the run of invisible labels it sits in is at most this long

AGREE_PX = 15.0             # a label and a raw detection agree within this many 1080p pixels
FAST_PX = 6.0               # a label that moved this far (1080p px) since the previous row can tell one frame from the next
MIN_FAST = 20               # fewer fast labels than this and every label is judged instead
DETECTION_CONF = 0.5        # detections at or above this confidence count as the detector's answer

Point = Optional[Tuple[float, float]]
LabelledPrediction = Tuple[Point, Point]     # (label or None for no ball, prediction or None for a miss)


def scale_for_width(width: int) -> float:
    """Multiplier that takes this frame's pixels to 1080p pixels."""
    return REFERENCE_WIDTH / width


def scale_for_height(height: int) -> float:
    return REFERENCE_HEIGHT / height


def _segment_distance(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance from ``point`` to the segment ``a``-``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 <= 0.0 else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
    return math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy))


def bridged_alarms(pairs: Sequence[LabelledPrediction], scale: float) -> List[bool]:
    """Per pair, whether it is a false alarm that bridged an occlusion: a prediction on an
    invisible label inside a run of at most ``BRIDGE_MAX_RUN`` invisible labels flanked by visible
    ones, within ``BRIDGE_PX`` (after ``scale``) of the line between the flanking labels."""
    flags = [False] * len(pairs)
    index = 0
    while index < len(pairs):
        if pairs[index][0] is not None:
            index += 1
            continue
        end = index
        while end < len(pairs) and pairs[end][0] is None:
            end += 1
        if 0 < index and end < len(pairs) and end - index <= BRIDGE_MAX_RUN:
            before, after = pairs[index - 1][0], pairs[end][0]
            for position in range(index, end):
                prediction = pairs[position][1]
                if prediction is not None and _segment_distance(prediction, before, after) * scale <= BRIDGE_PX:
                    flags[position] = True
        index = end
    return flags


class _Tally(NamedTuple):
    """The counts one sequence of pairs adds to the metric."""
    hit: int
    wrong: int
    miss: int
    false_alarm: int
    bridged: int
    quiet: int
    errors: List[float]

    @classmethod
    def of(cls, pairs: Sequence[LabelledPrediction], scale: float) -> "_Tally":
        hit = wrong = miss = false_alarm = bridged = quiet = 0
        errors = []
        for (label, prediction), is_bridged in zip(pairs, bridged_alarms(pairs, scale)):
            if label is None:
                if prediction is None:
                    quiet += 1
                else:
                    false_alarm += 1
                    bridged += is_bridged
                continue
            if prediction is None:
                miss += 1
                continue
            error = math.hypot(prediction[0] - label[0], prediction[1] - label[1]) * scale
            errors.append(error)
            if error <= HIT_PX:
                hit += 1
            elif error > WRONG_PX:
                wrong += 1
        return cls(hit, wrong, miss, false_alarm, bridged, quiet, errors)

    def plus(self, other: "_Tally") -> "_Tally":
        return _Tally(*(a + b for a, b in zip(self[:-1], other[:-1])), self.errors + other.errors)


class DetectorMetrics(NamedTuple):
    """How a detector did on a set of labelled frames: the rates are shares of the visible (or,
    for ``false_alarm`` and ``bridged``, the invisible) labels, the distances 1080p pixels over the
    frames where both a label and a prediction exist."""
    visible: int          # labels with a ball
    recall: float         # share of them the prediction hit within HIT_PX
    wrong: float          # share of them the prediction was more than WRONG_PX from: another object
    miss: float           # share of them with no prediction
    invisible: int        # labels without a ball
    false_alarm: float    # share of those with a prediction anyway, the bridged ones included
    median_px: float      # over the frames with both (NaN when there are none)
    p90_px: float
    bridged: float = 0.0  # share of the invisible labels whose prediction bridged an occlusion (module docstring)

    @property
    def false_alarm_count(self) -> int:
        return round(self.false_alarm * self.invisible)

    @property
    def bridged_count(self) -> int:
        return round(self.bridged * self.invisible)

    @classmethod
    def of(cls, pairs: Sequence[LabelledPrediction], scale: float) -> "DetectorMetrics":
        """Score ``pairs``, one clip's in frame order, each distance multiplied by ``scale`` to
        1080p pixels."""
        return cls.of_clips([pairs], scale)

    @classmethod
    def of_clips(cls, clips: Sequence[Sequence[LabelledPrediction]], scale: float) -> "DetectorMetrics":
        """Score several clips' pairs pooled, each clip's in frame order so the occlusion runs are
        found inside one clip."""
        tally = _Tally(0, 0, 0, 0, 0, 0, [])
        for pairs in clips:
            tally = tally.plus(_Tally.of(pairs, scale))
        visible = tally.miss + len(tally.errors)
        invisible = tally.false_alarm + tally.quiet
        errors = sorted(tally.errors)
        return cls(
            visible=visible,
            recall=tally.hit / max(visible, 1),
            wrong=tally.wrong / max(visible, 1),
            miss=tally.miss / max(visible, 1),
            invisible=invisible,
            false_alarm=tally.false_alarm / max(invisible, 1),
            median_px=errors[len(errors) // 2] if errors else float("nan"),
            p90_px=errors[int(0.9 * len(errors))] if errors else float("nan"),
            bridged=tally.bridged / max(invisible, 1),
        )
