"""The court labels: ``labels/<clip>_court.csv``, where the court's keypoints are on a few frames.

The court engine answers fourteen keypoints per frame (the tracking JSON's ``court_keypoints``, in
the semantic order ``tennis_tracker.court.court_geometry`` documents). The label file names the
same fourteen by name, ``COURT_POINTS[:14]`` in that order, and two more the engine does not
answer: the tops of the net posts (``net_post_left``, ``net_post_right``), where the net cord
meets each post, for the court model with geometry to come.

``frame,point,x,y``: ``frame`` is a ``frame_###`` cell, ``point`` a name of ``COURT_POINTS``,
``x,y`` its position in native image pixels (pixel k's centre at k + 0.5, as in the ball labels).
A frame in the file was checked as a whole: every point it lists is where the person put (or
left) it, and a point it leaves out was not visible on that frame (outside the picture, or
hidden). A frame not in the file has no court label.

``COURT_METRES`` places every point on the court in metres (ITF dimensions): x across from the
left doubles sideline, y along from the far baseline, z up (zero for the fourteen ground points,
which are the ones a ground-plane homography is fitted to). The court scorer
(``finetune.scoring.court_score``) uses them to turn a pixel error into metres on the court.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Tuple

from finetune.data.fileio import atomic_csv
from finetune.data.labels import COORDINATE_DECIMALS, LabelFormatError, frame_index, frame_name

COURT_COLUMNS = ("frame", "point", "x", "y")
# The engine's fourteen in its semantic order, then the two net posts it does not answer.
COURT_POINTS = (
    "far_left", "far_right", "near_left", "near_right",
    "far_left_singles", "near_left_singles", "far_right_singles", "near_right_singles",
    "far_service_left", "far_service_right", "near_service_left", "near_service_right",
    "far_centre", "near_centre",
    "net_post_left", "net_post_right",
)
ENGINE_POINTS = 14                # the first fourteen are the engine's, in its order
GROUND_POINTS = COURT_POINTS[:ENGINE_POINTS]

COURT_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_INSET_M = 1.37            # each alley's width
SERVICE_FROM_NET_M = 6.40
NET_POST_OUTSIDE_M = 0.914        # the net posts stand this far outside the doubles sidelines
NET_POST_HEIGHT_M = 1.07

_HALF = COURT_LENGTH_M / 2.0
_FAR_SERVICE, _NEAR_SERVICE = _HALF - SERVICE_FROM_NET_M, _HALF + SERVICE_FROM_NET_M
_LEFT_SINGLES, _RIGHT_SINGLES = SINGLES_INSET_M, DOUBLES_WIDTH_M - SINGLES_INSET_M
COURT_METRES: Dict[str, Tuple[float, float, float]] = {
    "far_left": (0.0, 0.0, 0.0), "far_right": (DOUBLES_WIDTH_M, 0.0, 0.0),
    "near_left": (0.0, COURT_LENGTH_M, 0.0), "near_right": (DOUBLES_WIDTH_M, COURT_LENGTH_M, 0.0),
    "far_left_singles": (_LEFT_SINGLES, 0.0, 0.0), "near_left_singles": (_LEFT_SINGLES, COURT_LENGTH_M, 0.0),
    "far_right_singles": (_RIGHT_SINGLES, 0.0, 0.0), "near_right_singles": (_RIGHT_SINGLES, COURT_LENGTH_M, 0.0),
    "far_service_left": (_LEFT_SINGLES, _FAR_SERVICE, 0.0), "far_service_right": (_RIGHT_SINGLES, _FAR_SERVICE, 0.0),
    "near_service_left": (_LEFT_SINGLES, _NEAR_SERVICE, 0.0), "near_service_right": (_RIGHT_SINGLES, _NEAR_SERVICE, 0.0),
    "far_centre": (DOUBLES_WIDTH_M / 2.0, _FAR_SERVICE, 0.0), "near_centre": (DOUBLES_WIDTH_M / 2.0, _NEAR_SERVICE, 0.0),
    "net_post_left": (-NET_POST_OUTSIDE_M, _HALF, NET_POST_HEIGHT_M),
    "net_post_right": (DOUBLES_WIDTH_M + NET_POST_OUTSIDE_M, _HALF, NET_POST_HEIGHT_M),
}

CourtPoints = Dict[str, Tuple[float, float]]      # point name -> (x, y) in native pixels


class CourtLabel(NamedTuple):
    """One labelled frame: its source frame and the points placed on it."""

    frame: int
    points: CourtPoints


def point_of(cell: str) -> str:
    """The point a ``point`` cell names (``COURT_POINTS``); anything else is a LabelFormatError."""
    text = (cell or "").strip().lower()
    if text not in COURT_POINTS:
        raise LabelFormatError(f"bad point cell {cell!r}; expected one of {', '.join(COURT_POINTS)}")
    return text


def engine_points(keypoints) -> CourtPoints:
    """The engine's flat ``[x0, y0, x1, y1, ...]`` keypoints as named points, the ones it did not
    detect (at or below zero on both axes, ``court_geometry.court_keypoint``) or that are not
    finite left out. Empty for None or a short list."""
    points: CourtPoints = {}
    if not keypoints:
        return points
    for index, name in enumerate(GROUND_POINTS):
        if 2 * index + 1 >= len(keypoints):
            break
        x, y = float(keypoints[2 * index]), float(keypoints[2 * index + 1])
        if x != x or y != y or abs(x) == float("inf") or abs(y) == float("inf") or (x <= 0.0 and y <= 0.0):
            continue
        points[name] = (x, y)
    return points


def read_court_labels(path: Path) -> List[CourtLabel]:
    """Every labelled frame, in frame order, its points in ``COURT_POINTS`` order. A cell that
    does not parse, or a point given twice on one frame, is a LabelFormatError naming the file."""
    by_frame: Dict[int, CourtPoints] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(COURT_COLUMNS).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(COURT_COLUMNS)}")
        for record in reader:
            if not any((value or "").strip() for value in record.values()):
                continue
            try:
                frame, name = frame_index(record.get("frame") or ""), point_of(record.get("point") or "")
                point = (float(record["x"]), float(record["y"]))
            except (LabelFormatError, TypeError, ValueError) as error:
                raise LabelFormatError(f"{path}: {error}") from error
            points = by_frame.setdefault(frame, {})
            if name in points:
                raise LabelFormatError(f"{path}: frame {frame} gives {name} twice")
            points[name] = point
    return [CourtLabel(frame, {name: by_frame[frame][name] for name in COURT_POINTS if name in by_frame[frame]})
            for frame in sorted(by_frame)]


def write_court_labels(path: Path, labels: Iterable[CourtLabel]) -> int:
    """Write the labelled frames, in frame order and each frame's points in ``COURT_POINTS``
    order, atomically. Returns the row count. A frame with no points is a ValueError: it would
    read back as no label at all, not as a frame where nothing was visible."""
    ordered = sorted(labels, key=lambda label: label.frame)
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.frame == later.frame:
            raise ValueError(f"frame {later.frame} is given twice")
    for frame, points in ordered:
        if not points:
            raise ValueError(f"frame {frame} has no court points; leave it out instead")
        unknown = set(points) - set(COURT_POINTS)
        if unknown:
            raise ValueError(f"frame {frame} names points the file has no name for: {', '.join(sorted(unknown))}")
    rows = 0
    with atomic_csv(path) as writer:
        writer.writerow(list(COURT_COLUMNS))
        for frame, points in ordered:
            for name in COURT_POINTS:
                if name in points:
                    x, y = points[name]
                    writer.writerow([frame_name(frame), name, f"{x:.{COORDINATE_DECIMALS}f}", f"{y:.{COORDINATE_DECIMALS}f}"])
                    rows += 1
    return rows


def labels_by_frame(labels: Iterable[CourtLabel]) -> Mapping[int, CourtPoints]:
    return {label.frame: dict(label.points) for label in labels}
