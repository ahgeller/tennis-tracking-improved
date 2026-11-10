"""The file format the landing and hit labels share (``bounce_labels``, ``event_labels``): one row
per event, a ``frame`` cell naming its frame (``frame_###``, or ``frame_###.5`` between that frame
and the next), the point cells ``ball_x,ball_y`` (both blank for the ball label's point), and any
columns of the event's own after them (a hit's ``hitter``).

A file may carry columns no reader here knows (a note somebody added in a spreadsheet). They are
kept: ``write_event_rows`` copies them from the file it replaces, row by row (``kept_columns``), so a
save never drops what it does not understand. The rally file keeps its own the same way.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

from finetune.data.fileio import atomic_csv
from finetune.data.labels import COLUMNS, COORDINATE_DECIMALS, LabelFormatError, LabelPoint, frame_name

# An event's frame cell: frame_<index>, or frame_<index>.5 for one between that frame and the next.
EVENT_FRAME_PATTERN = re.compile(r"frame_(\d+)(\.5)?")


class EventRow(NamedTuple):
    """One row of an event file: its frame (an int, or a half frame), its point (None when both
    cells are blank) and the cells of the event's own columns, by name."""

    frame: float
    point: LabelPoint
    cells: Dict[str, str]


def event_frame(cell: str) -> float:
    """The frame a ``frame`` cell names: an int for ``frame_###``, the half frame for ``frame_###.5``.
    As strict as ``labels.frame_index``: a cell that is not exactly one of those two forms
    (``frame_12.4``, ``frame_12.50``, ``frame_.5``) is an error, not an event moved to whatever frame
    could be fished out of it."""
    match = EVENT_FRAME_PATTERN.fullmatch((cell or "").strip())
    if match is None:
        raise LabelFormatError(f"bad frame cell {cell!r}; expected frame_<index> or frame_<index>.5")
    whole = int(match.group(1))
    return whole + 0.5 if match.group(2) else whole


def event_name(frame: float) -> str:
    """An event's ``frame`` cell: ``frame_###`` for a whole frame, ``frame_###.5`` for one between
    two. Any other fraction is a ValueError: the file has no way to say it."""
    whole = math.floor(frame)
    if frame == whole:
        return frame_name(int(whole))
    if frame - whole == 0.5:
        return frame_name(int(whole)) + ".5"
    raise ValueError(f"an event sits on a frame or halfway between two, not at {frame!r}")


def event_text(frame: float) -> str:
    """An event's frame as the reports print it: ``412``, or ``412.5`` between two frames."""
    return str(int(frame)) if float(frame).is_integer() else f"{frame:.1f}"


def point_cells(record: Mapping[str, str], path: Path) -> LabelPoint:
    """A row's ``ball_x``, ``ball_y`` cells as a point, None when both are blank; one without the
    other, or one that does not parse, is a LabelFormatError naming `path`."""
    x_cell, y_cell = (record.get("ball_x") or "").strip(), (record.get("ball_y") or "").strip()
    if not x_cell and not y_cell:
        return None
    if not x_cell or not y_cell:
        raise LabelFormatError(f"{path}: {record.get('frame')!r} gives one coordinate without the other")
    try:
        return float(x_cell), float(y_cell)
    except ValueError as error:
        raise LabelFormatError(f"{path}: {record.get('frame')!r}: {error}") from error


def read_records(path: Path, columns: Sequence[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    """(the file's columns, its non-blank rows); a file without `columns` is a LabelFormatError
    naming it."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = list(reader.fieldnames or ())
        if not set(columns).issubset(names):
            raise LabelFormatError(f"{path}: columns are {names}, expected {','.join(columns)}")
        return names, [{key: value or "" for key, value in record.items() if key is not None}
                       for record in reader if any((value or "").strip() for value in record.values() if isinstance(value, str))]


def read_event_rows(path: Path, own: Sequence[str] = ()) -> List[EventRow]:
    """Every row of an event file with the columns ``frame,ball_x,ball_y`` and `own`, in frame
    order. A frame cell that is not ``frame_<index>`` or ``frame_<index>.5``, a coordinate that does
    not parse, a lone coordinate or a frame listed twice is a LabelFormatError naming the file."""
    rows: List[EventRow] = []
    for record in read_records(path, COLUMNS + tuple(own))[1]:
        try:
            frame = event_frame(record.get("frame") or "")
        except LabelFormatError as error:
            raise LabelFormatError(f"{path}: {error}") from error
        rows.append(EventRow(frame, point_cells(record, path), {name: record.get(name) or "" for name in own}))
    rows.sort(key=lambda row: row.frame)
    for earlier, later in zip(rows, rows[1:]):
        if earlier.frame == later.frame:
            raise LabelFormatError(f"{path}: frame {event_text(later.frame)} is listed twice")
    return rows


def kept_columns(path: Path, columns: Sequence[str],
                 key: Callable[[Mapping[str, str]], str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """The columns the file at `path` has beyond `columns`, and their cells by each row's `key`:
    what a rewrite of the file carries over. None of either for a missing or unreadable file."""
    try:
        names, records = read_records(path, ())
    except (OSError, LabelFormatError, csv.Error):
        return [], {}
    extra = [name for name in names if name not in columns]
    if not extra:
        return [], {}
    return extra, {key(record): [record.get(name) or "" for name in extra] for record in records}


def write_rows(path: Path, columns: Sequence[str], rows: Iterable[Sequence[str]],
               key: Callable[[Mapping[str, str]], str]) -> int:
    """Write `rows` (cells in `columns` order) atomically, carrying over the columns the file it
    replaces has beyond `columns`, cell by cell for the rows whose `key` it had. Returns the row
    count."""
    extra, cells_of = kept_columns(path, columns, key)
    written = 0
    with atomic_csv(path) as writer:
        writer.writerow([*columns, *extra])
        for row in rows:
            own = list(row)
            writer.writerow([*own, *cells_of.get(key(dict(zip(columns, own))), [""] * len(extra))])
            written += 1
    return written


def by_frame_cell(record: Mapping[str, str]) -> str:
    return (record.get("frame") or "").strip()


def write_event_rows(path: Path, rows: Iterable[Tuple[float, LabelPoint, Sequence[str]]], own: Sequence[str] = (),
                     what: str = "event") -> int:
    """Write events, sorted by frame, atomically: ``frame,ball_x,ball_y`` then the cells of `own`
    (blank coordinates for an event without a point), and whatever columns the replaced file had
    beyond those. A frame given twice is a ValueError before anything is written: the reader would
    refuse the file (`what` names an event in the message). Returns the row count."""
    ordered = sorted(rows, key=lambda row: row[0])
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier[0] == later[0]:
            raise ValueError(f"frame {event_text(later[0])} is given twice; a file holds one {what} per frame")
    cells = []
    for frame, point, own_cells in ordered:
        xy = ("", "") if point is None else (f"{point[0]:.{COORDINATE_DECIMALS}f}", f"{point[1]:.{COORDINATE_DECIMALS}f}")
        cells.append([event_name(frame), *xy, *own_cells])
    return write_rows(path, COLUMNS + tuple(own), cells, by_frame_cell)
