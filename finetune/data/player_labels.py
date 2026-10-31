"""The player identity labels: ``labels/<clip>_players.csv``, who each tracked person is.

The tracker follows every person it sees with a box and a track id (ByteTrack's, per frame, in
the tracking JSON's ``player_boxes``), and a track id names nobody: the same player gets a new one
after an occlusion, and two people can swap one. The label tool shows a pipeline run's tracks as
segments (``finetune.labeling.label_players``) and the person labelling says who each segment is.
What it says is written per frame, so the labels outlive the run they were made on:

``frame,player,x1,y1,x2,y2,track``: ``frame`` is a ``frame_###`` cell (the zero-based source
frame), ``player`` one of ``P1`` to ``P4`` (the players of the clip, numbered as the person
labelling chose) or ``none`` for a person who is not a player (a spectator, a ball kid, a player on
the next court, an umpire), ``x1,y1,x2,y2`` the tracked box in native image pixels, and ``track``
the run's track id the box came from (a note on where the box came from; the scorer never reads
it). A frame holds one row per labelled box; boxes the person has not assigned yet are not in the
file at all, so a frame without rows is unknown, not empty.

The identity scorer (``finetune.scoring.players_score``) matches another run's boxes to these by
overlap, frame by frame.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, NamedTuple, Tuple

from finetune.data.fileio import atomic_csv
from finetune.data.labels import COORDINATE_DECIMALS, LabelFormatError, frame_index, frame_name

PLAYERS = ("P1", "P2", "P3", "P4")
NOT_PLAYER = "none"
IDENTITIES = PLAYERS + (NOT_PLAYER,)
PLAYER_COLUMNS = ("frame", "player", "x1", "y1", "x2", "y2", "track")

Box = Tuple[float, float, float, float]


class PlayerBoxLabel(NamedTuple):
    """One labelled box: its frame, who it is (``IDENTITIES``), the box and the track it came from."""

    frame: int
    player: str
    box: Box
    track: str = ""


def identity_of(cell: str) -> str:
    """The identity a ``player`` cell names (``P1``..``P4`` or ``none``, any case); anything else is
    a LabelFormatError."""
    text = (cell or "").strip()
    for identity in IDENTITIES:
        if text.lower() == identity.lower():
            return identity
    raise LabelFormatError(f"bad player cell {cell!r}; expected one of {', '.join(IDENTITIES)}")


def read_player_labels(path: Path) -> List[PlayerBoxLabel]:
    """Every labelled box, in frame order. A cell that does not parse, a box with no area, or one
    track's box given twice on a frame is a LabelFormatError naming the file."""
    labels: List[PlayerBoxLabel] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(PLAYER_COLUMNS[:-1]).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(PLAYER_COLUMNS)}")
        for record in reader:
            if not any((value or "").strip() for value in record.values()):
                continue
            try:
                x1, y1, x2, y2 = (float(record[key]) for key in ("x1", "y1", "x2", "y2"))
                label = PlayerBoxLabel(frame_index(record.get("frame") or ""), identity_of(record.get("player") or ""),
                                       (x1, y1, x2, y2), (record.get("track") or "").strip())
            except (LabelFormatError, TypeError, ValueError) as error:
                raise LabelFormatError(f"{path}: {error}") from error
            if not (x2 > x1 and y2 > y1):
                raise LabelFormatError(f"{path}: frame {label.frame} has a box with no area: {label.box}")
            labels.append(label)
    labels.sort(key=lambda label: (label.frame, label.track, label.box))
    seen = set()
    for label in labels:
        if label.track and (label.frame, label.track) in seen:
            raise LabelFormatError(f"{path}: track {label.track} has two boxes on frame {label.frame}")
        seen.add((label.frame, label.track))
    return labels


def write_player_labels(path: Path, labels: Iterable[PlayerBoxLabel]) -> int:
    """Write the labelled boxes, in frame order, atomically. Returns the row count."""
    ordered = sorted(labels, key=lambda label: (label.frame, label.track, label.box))
    with atomic_csv(path) as writer:
        writer.writerow(list(PLAYER_COLUMNS))
        for label in ordered:
            writer.writerow([frame_name(label.frame), label.player,
                             *(f"{value:.{COORDINATE_DECIMALS}f}" for value in label.box), label.track])
    return len(ordered)
