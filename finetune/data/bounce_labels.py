"""The landing labels: ``labels/<clip>_bounces.csv``, the frames where the ball hit the ground.

A file has the ball label files' columns, ``frame,ball_x,ball_y`` (``finetune.data.labels``):
``frame_###`` names the zero-based source frame of the landing, and ``frame_###.5`` a landing
between that frame and the next, for a ball that is in the air on both (``event_rows`` reads
and writes exactly those two forms); the coordinates are the landing point in
native image pixels. Both coordinate cells may be left blank: the point is then the clip's ball
label at that frame, or, when the frame has none (the ball hidden, or the clip labelled on the
other 60 FPS phase), the midpoint of the ball labels on the frames either side of it, or the one
of them there is; a landing between two frames takes the same midpoint of the two
(``resolve_points``). One coordinate without the other is an error. The file
lists every landing of every rally, serves and faults included; a frame within the scorer's
tolerance (``finetune.scoring.court_events_score``, 0.1 s) of the true landing is close enough, so
either 60 FPS phase will do.

It is a file of its own, not a column of ``_ball.csv``: ``labels.write_labels`` rewrites that file
with exactly its columns. ``workspace.bounce_csv`` names it; the files are small and are versioned
with the scorer that reads them.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Mapping, NamedTuple

from finetune.data.event_rows import read_event_rows, write_event_rows
from finetune.data.labels import LabelPoint


class BounceLabel(NamedTuple):
    """One labelled landing: its source frame (an int, or a half frame ending .5 for a landing
    between two frames) and its point, None when the row left it blank."""

    frame: float
    point: LabelPoint = None


def between_frames(first: int, second: int) -> float:
    """The frame of a landing halfway between two label frames: the half frame between consecutive
    frames (3.5), the whole frame between two a step apart (413 between 412 and 414 on a 60 FPS
    clip labelled every other frame), as an int when it is whole."""
    midpoint = (first + second) / 2.0
    return int(midpoint) if midpoint.is_integer() else midpoint


def read_bounce_labels(path: Path) -> List[BounceLabel]:
    """Every landing in the file, in frame order. A frame listed twice, a frame cell that is not
    ``frame_<index>`` or ``frame_<index>.5``, a coordinate that does not parse or a lone coordinate
    is a ``LabelFormatError`` naming the file (``event_rows.read_event_rows``)."""
    return [BounceLabel(row.frame, row.point) for row in read_event_rows(path)]


def write_bounce_labels(path: Path, labels: Iterable[BounceLabel]) -> int:
    """Write the landings, sorted by frame, atomically, keeping any column of the file it replaces
    that is not a landing's (``event_rows.write_event_rows``); a landing without a point gets blank
    coordinate cells. Returns the row count. A frame given twice is a ValueError before anything is
    written: ``read_bounce_labels`` would refuse the file."""
    return write_event_rows(path, [(label.frame, label.point, ()) for label in labels], what="landing")


def point_from_ball_labels(frame: float, ball_labels: Mapping[int, LabelPoint]) -> LabelPoint:
    """Where the ball labels put a landing that has no point of its own (see ``resolve_points``):
    the label on its frame, else the midpoint of the labels on the frames either side of it (the
    two a half frame lies between), or the one of them there is."""
    whole = int(math.floor(frame))
    if frame == whole:
        found = ball_labels.get(whole)
        if found is not None:
            return float(found[0]), float(found[1])
        either_side = (ball_labels.get(whole - 1), ball_labels.get(whole + 1))
    else:
        either_side = (ball_labels.get(whole), ball_labels.get(whole + 1))
    labelled = [point for point in either_side if point is not None]
    if not labelled:
        return None
    return (sum(float(point[0]) for point in labelled) / len(labelled),
            sum(float(point[1]) for point in labelled) / len(labelled))


def resolve_points(labels: Iterable[BounceLabel], ball_labels: Mapping[int, LabelPoint]) -> List[BounceLabel]:
    """The landings with every blank point filled from the clip's ball labels (``labels.read_labels``):
    the label at the landing's frame, else the midpoint of the labels on the frame before and the
    frame after it (the ball hidden there, or the clip labelled on the other 60 FPS phase), or the
    one of them there is; a landing between two frames takes the same midpoint of the two. A landing
    with no visible ball label on any of those frames keeps ``None``; its timing is still scored."""
    return [BounceLabel(label.frame, point_from_ball_labels(label.frame, ball_labels) if label.point is None else label.point)
            for label in labels]
