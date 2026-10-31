"""The hit and rally labels: ``labels/<clip>_hits.csv`` and ``labels/<clip>_rallies.csv``.

A hit file lists the frames where a racket hit the ball, serves included, one row each, with
the columns ``frame,ball_x,ball_y,hitter``: a landing file's columns (``bounce_labels``) and the
hitter, read exactly as a landing is. ``frame`` is ``frame_###`` for the zero-based source frame of
the contact, ``frame_###.5`` for a contact between that frame and the next (the ball still coming
in on one, already going out on the other); ``ball_x,ball_y`` is where the ball met the racket in
native image pixels, both blank when it is the ball label of that frame (``resolve_hit_points``
then takes it from the ball labels, as ``bounce_labels.resolve_points`` does for a landing).
``hitter`` says which player hit it, ``near`` (the player on the half of the court nearer the
camera) or ``far``, and is blank when it was not said. The hit scorer
(``finetune.scoring.hits_score``) matches the tracker's racket contacts to these.

A rally file lists the rallies, one row each, with the columns ``start,end``: ``start`` is the
frame the point begins on (the server's toss leaves the hand; for a serve with no toss in
view, the serve's hit), ``end`` the frame it is over (the ball is dead: its second bounce, the
net, the fence, or the frame it leaves play for good), both ``frame_###`` cells. Either cell may
be blank while a rally is still being marked; a rally with a blank cell is kept in the file and
left out by the scorer (``finetune.scoring.rallies_score``). Rallies are listed in start order
and never overlap.

Both are files of their own beside the ball labels and the landings, written by the label
tool's ``.`` and ``[`` / ``]`` keys (``finetune/labeling/LABEL_TOOL.md``). A clip with nothing
marked has no file: the scorers read any file, empty included, as the whole truth about its clip.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Literal, Mapping, NamedTuple, Optional, Sequence, Tuple, get_args

from finetune.data.bounce_labels import landing_frame, landing_name, landing_text, point_cells, point_from_ball_labels
from finetune.data.fileio import atomic_csv
from finetune.data.labels import COLUMNS, COORDINATE_DECIMALS, LabelFormatError, LabelPoint, frame_index, frame_name

Hitter = Literal["near", "far"]
HITTERS: Tuple[Hitter, ...] = get_args(Hitter)
HIT_COLUMNS = COLUMNS + ("hitter",)
RALLY_COLUMNS = ("start", "end")


class HitLabel(NamedTuple):
    """One labelled hit: its source frame (an int, or a half frame ending .5 for a contact
    between two frames), where the ball met the racket (None when the row left it blank) and who
    hit it (None when that was not said)."""

    frame: float
    point: LabelPoint = None
    hitter: Optional[Hitter] = None


class RallyLabel(NamedTuple):
    """One labelled rally: the frames it starts and ends on, None for a cell not yet marked."""

    start: Optional[int]
    end: Optional[int]

    @property
    def complete(self) -> bool:
        return self.start is not None and self.end is not None


def _records(path: Path, columns: Sequence[str]) -> List[dict]:
    """The file's non-blank rows; a file without `columns` is a LabelFormatError naming it."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(columns)}")
        return [record for record in reader if any((value or "").strip() for value in record.values())]


# --------------------------------------------------------------------------- hits

def hitter_of(cell: str) -> Optional[Hitter]:
    """The hitter a ``hitter`` cell names: ``near``, ``far``, or None for a blank cell; anything
    else is a LabelFormatError."""
    text = (cell or "").strip().lower()
    if not text:
        return None
    for hitter in HITTERS:
        if text == hitter:
            return hitter
    raise LabelFormatError(f"bad hitter cell {cell!r}; expected {' or '.join(HITTERS)} or blank")


def read_hit_labels(path: Path) -> List[HitLabel]:
    """Every hit in the file, in frame order. A frame cell that is not ``frame_<index>`` or
    ``frame_<index>.5``, a hitter that is not ``near`` / ``far`` / blank, or a frame listed twice
    is a LabelFormatError naming the file."""
    labels = []
    for record in _records(path, HIT_COLUMNS):
        try:
            frame = landing_frame(record.get("frame") or "")
            labels.append(HitLabel(frame, point_cells(record, path), hitter_of(record.get("hitter") or "")))
        except LabelFormatError as error:
            raise LabelFormatError(str(error) if str(error).startswith(str(path)) else f"{path}: {error}") from error
    labels.sort(key=lambda label: label.frame)
    for earlier, later in zip(labels, labels[1:]):
        if earlier.frame == later.frame:
            raise LabelFormatError(f"{path}: frame {landing_text(later.frame)} is listed twice")
    return labels


def write_hit_labels(path: Path, labels: Iterable[HitLabel]) -> int:
    """Write the hits, sorted by frame, atomically. Returns the row count. A frame given twice is
    a ValueError before anything is written: ``read_hit_labels`` would refuse the file."""
    ordered = sorted(labels, key=lambda label: label.frame)
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.frame == later.frame:
            raise ValueError(f"frame {landing_text(later.frame)} is given twice; a file holds one hit per frame")
    with atomic_csv(path) as writer:
        writer.writerow(list(HIT_COLUMNS))
        for frame, point, hitter in ordered:
            cells = ("", "") if point is None else (f"{point[0]:.{COORDINATE_DECIMALS}f}", f"{point[1]:.{COORDINATE_DECIMALS}f}")
            writer.writerow([landing_name(frame), *cells, hitter or ""])
    return len(ordered)


def resolve_hit_points(labels: Iterable[HitLabel], ball_labels: Mapping[int, LabelPoint]) -> List[HitLabel]:
    """The hits with every blank point filled from the clip's ball labels, as
    ``bounce_labels.resolve_points`` fills a landing's: the label at its frame, else the midpoint of
    the labels either side; None where there is none."""
    return [label if label.point is not None else label._replace(point=point_from_ball_labels(label.frame, ball_labels))
            for label in labels]


# --------------------------------------------------------------------------- rallies

def _rally_cell(cell: str) -> Optional[int]:
    text = (cell or "").strip()
    return frame_index(text) if text else None


def check_rallies(rallies: Sequence[RallyLabel]) -> None:
    """Raise ValueError unless every rally has at least one end marked, starts before it ends,
    and the rallies are in order without overlapping (an open end reaches up to the next rally)."""
    previous: Optional[RallyLabel] = None
    for rally in rallies:
        if rally.start is None and rally.end is None:
            raise ValueError("a rally needs its start or its end")
        if rally.complete and rally.start >= rally.end:   # type: ignore[operator]
            raise ValueError(f"the rally from {rally.start} to {rally.end} does not start before it ends")
        if previous is not None:
            last = previous.end if previous.end is not None else previous.start
            first = rally.start if rally.start is not None else rally.end
            if first <= last:   # type: ignore[operator]
                raise ValueError(f"the rallies at {previous.start}-{previous.end} and {rally.start}-{rally.end} overlap")
        previous = rally


def read_rally_labels(path: Path) -> List[RallyLabel]:
    """Every rally in the file, in order. A cell that is not ``frame_<index>`` or blank, a row with
    neither cell, a rally that does not start before it ends and rallies out of order or
    overlapping are a LabelFormatError naming the file."""
    rallies = []
    for record in _records(path, RALLY_COLUMNS):
        try:
            rallies.append(RallyLabel(_rally_cell(record.get("start") or ""), _rally_cell(record.get("end") or "")))
        except LabelFormatError as error:
            raise LabelFormatError(f"{path}: {error}") from error
    try:
        check_rallies(rallies)
    except ValueError as error:
        raise LabelFormatError(f"{path}: {error}") from error
    return rallies


def write_rally_labels(path: Path, rallies: Sequence[RallyLabel]) -> int:
    """Write the rallies, atomically, as given. Returns the row count. Rallies that
    ``read_rally_labels`` would refuse are a ValueError before anything is written."""
    check_rallies(rallies)
    with atomic_csv(path) as writer:
        writer.writerow(list(RALLY_COLUMNS))
        for start, end in rallies:
            writer.writerow([frame_name(start) if start is not None else "", frame_name(end) if end is not None else ""])
    return len(rallies)


def pair_rally_marks(starts: Iterable[int], ends: Iterable[int]) -> List[RallyLabel]:
    """The rallies a set of start marks and end marks make, in order: each start with the first end
    after it, unless another start comes first, in which case the earlier start is a rally whose
    end is not marked yet; an end with no start before it since the last end is a rally whose start
    is not marked. A frame marked both ways counts as an end (the label tool never marks one so)."""
    end_frames = set(ends)
    marks = sorted({(frame, "end") for frame in end_frames} |
                   {(frame, "start") for frame in starts if frame not in end_frames})
    rallies: List[RallyLabel] = []
    open_start: Optional[int] = None
    for frame, kind in marks:
        if kind == "start":
            if open_start is not None:
                rallies.append(RallyLabel(open_start, None))
            open_start = frame
        else:
            rallies.append(RallyLabel(open_start, frame))
            open_start = None
    if open_start is not None:
        rallies.append(RallyLabel(open_start, None))
    return rallies
