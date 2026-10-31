"""What the label tool reads and writes: a clip's rows, the review and flag sidecars, its landings,
hits and rallies, the candidates rejected, the player and court labels, and the settings file.
The on-disk formats are those of ``finetune.data.labels``, ``finetune.data.bounce_labels``,
``finetune.data.event_labels``, ``finetune.data.player_labels`` and ``finetune.data.court_labels``;
this module turns them into ``Row`` objects and the tool's maps and back. ``load_*`` functions read
and return; ``apply_*`` functions write what a sidecar says onto the rows they are given; ``save_*``
functions write what they are given to the path they are given, an empty file (its header) when
there is nothing. Which kinds are written, and when an emptied file is removed instead, is
``label_saving``'s."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from finetune.data.bounce_labels import BounceLabel, read_bounce_labels, write_bounce_labels
from finetune.data.court_labels import CourtLabel, CourtPoints, read_court_labels, write_court_labels
from finetune.data.event_labels import (Hitter, HitLabel, RallyLabel, pair_rally_marks, read_hit_labels,
                                        read_rally_labels, write_hit_labels, write_rally_labels)
from finetune.data.fileio import write_atomic
from finetune.data.labels import (FLAG_KINDS, FlagKind, Flags, LabelFormatError, LabelPoint, answered_point,
                                  format_point, frame_list, is_visible, read_flags, read_json_object, read_label_rows,
                                  read_review, write_flags, write_label_cells)
from finetune.data.player_labels import PlayerBoxLabel, read_player_labels, write_player_labels

CONFIDENT_CONF = 0.70       # a draft at or above this confidence is not "uncertain"
SETTINGS_NAME = "label_tool.settings.json"             # in the labels folder: filter, inset, trail


class Row:
    __slots__ = ("frame", "x", "y", "visible", "source", "conf", "reviewed", "edited", "suspect",
                 "history", "clip")
    revision = 0                      # bumped by every write to any row; Tool.render_state keys on it

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        Row.revision += 1

    def __init__(self, frame: int, x: float, y: float, visible: bool, source: str, conf: float):
        self.frame, self.x, self.y, self.visible = frame, x, y, visible
        self.clip = ""                # which clip this row belongs to, set by Source
        self.source, self.conf = source, conf
        self.reviewed = False         # a person has dealt with this frame: confirmed it (SPACE) or edited it
        self.edited = False           # ... and in particular changed its label (a click, v, a nudge)
        self.suspect = False          # flagged by ft.py check --audit --mark; answered by SPACE or an edit
        self.history: List[tuple] = []

    @property
    def uncertain(self) -> bool:
        """Worth stopping on: no confident guess, or an audit flag you have not answered yet - which
        reopens a frame you had already reviewed, since catching those is what an audit is for."""
        return self.suspect or not (self.visible and self.conf >= CONFIDENT_CONF)

    def snapshot(self) -> None:
        self.history.append((self.x, self.y, self.visible, self.reviewed, self.edited, self.suspect,
                             self.source))

    def undo(self) -> None:
        if self.history:
            self.x, self.y, self.visible, self.reviewed, self.edited, self.suspect, self.source = \
                self.history.pop()


def load_rows(csv_path: Path, width: int, height: int, origin: str = "det") -> List[Row]:
    """origin names where a row with no source column of its own came from: "csv" for labels that
    arrived with the clip, "det" for ones pretrack.py guessed. Only pretrack writes a conf column,
    and the tool drops it on save, so without this a hand-labelled clip read back as "det 1.00"."""
    rows = []
    for record in read_label_rows(csv_path):
        visible = is_visible(record.x, record.y, width, height)
        rows.append(Row(record.frame, record.x if visible else width / 2.0, record.y if visible else height / 2.0,
                        visible, record.source or (origin if visible else "none"),
                        record.conf if record.conf is not None else 0.0))
    rows.sort(key=lambda row: row.frame)
    return rows


def save_rows(csv_path: Path, rows: List[Row], width: int, height: int) -> None:
    write_label_cells(csv_path, [(row.frame, *format_point((row.x, row.y) if row.visible else None, width, height))
                                 for row in rows])


def save_flags(path: Path, rows: List[Row]) -> None:
    """Drop the flags you have answered from the flag file, and remember each answer.

    A flag is a question, and confirming the label (SPACE) answers it just as well as changing it
    does. A flag on a frame you had reviewed before the audit stays open until you look at it
    again: catching mistakes in reviewed frames is what an audit is for. Answered frames go in an
    "answered" map with the label they were answered at, so a later audit does not raise the same
    question about the same label again (label_audit.flag_frames reads it). A damaged file is its
    LabelFormatError, naming the file; an older audit's ``frames`` list reads as ``check``.
    """
    if not path.is_file():
        return
    flags: Flags = read_flags(path)
    kept, answered = _answered_flags(flags, rows)
    if all(kept[kind] == flags[kind] for kind in FLAG_KINDS) and answered == flags["answered"]:
        return
    write_flags(path, {**flags, **kept, "answered": answered})


def _answered_flags(flags: Flags, rows: List[Row]) -> Tuple[Dict[FlagKind, List[int]], Dict[str, Optional[List[float]]]]:
    """(the flags of each kind still open on these rows, the answers so far plus the ones the
    rows give). A flag on a frame the rows do not hold stays open."""
    by_frame = {row.frame: row for row in rows}
    open_now = {row.frame for row in rows if row.suspect}
    kept: Dict[FlagKind, List[int]] = {kind: [f for f in flags[kind] if f in open_now or f not in by_frame]
                                       for kind in FLAG_KINDS}
    answered = dict(flags["answered"])
    flagged = {frame for kind in FLAG_KINDS for frame in flags[kind]}
    for frame in sorted(flagged & (set(by_frame) - open_now)):      # flagged, held, and no longer open: answered
        row = by_frame[frame]
        answered[str(frame)] = answered_point((row.x, row.y) if row.visible else None)
    return kept, answered


def save_review(path: Path, rows: List[Row], cursor: int, zoom: float) -> None:
    write_atomic(path, json.dumps({
        "cursor": cursor,
        "frame": rows[cursor].frame if 0 <= cursor < len(rows) else None,   # resume by frame, not index
        "zoom": zoom,
        "reviewed": [row.frame for row in rows if row.reviewed],
        "edited": [row.frame for row in rows if row.edited],
        "evidence": {str(row.frame): {"source": row.source, "conf": row.conf} for row in rows},
    }))


def apply_flags(path: Path, rows: List[Row]) -> int:
    """Mark the frames a previous audit flagged on the rows; returns how many. They are the ones
    worth your time first. A flag file that cannot be read is an error naming it
    (LabelFormatError), not an empty answer."""
    flags: Flags = read_flags(path)
    frames = set(flags["unplaceable"]) | set(flags["check"])
    for row in rows:
        row.suspect = row.frame in frames
    return sum(row.suspect for row in rows)


def apply_review(path: Path, rows: List[Row]) -> Optional[dict]:
    """Mark the rows the review sidecar says are reviewed or edited (and restore their evidence),
    and return the sidecar; None when there is none yet. An edited frame counts as reviewed, as
    it does for every reader of the sidecar (labels.reviewed_frames). A damaged file, or a frame
    list that is not one, is an error naming it (LabelFormatError): the tool never starts afresh
    over somebody's record of what they reviewed. Odd evidence inside a parseable file is
    tolerated frame by frame."""
    state = read_review(path)
    if state is None:
        return None
    reviewed, edited = set(frame_list(state, "reviewed", path)), set(frame_list(state, "edited", path))
    evidence_of = state.get("evidence") if isinstance(state.get("evidence"), dict) else {}
    for row in rows:
        row.edited = row.frame in edited
        row.reviewed = row.frame in reviewed or row.edited
        evidence = evidence_of.get(str(row.frame))
        if isinstance(evidence, dict) and "source" in evidence and "conf" in evidence:
            try:
                row.source, row.conf = str(evidence["source"]), float(evidence["conf"])
            except (TypeError, ValueError):
                # Evidence is a note on where a draft came from, not a label: one odd entry is
                # reported and skipped so the reviewed/edited record of the whole clip still loads.
                print(f"[warn] {path.name}: frame {row.frame} has unreadable evidence {evidence!r}; ignored")
    return state


# A clip's landings (``bounce_labels``): frame -> the landing point its row gave, None for a row
# that left it blank, which means the ball label of that frame. A landing between two frames is
# keyed by the half frame between them (3.5) and, on a clip labelled every other frame, by the
# unlabelled whole frame between (``bounce_labels.between_frames``). The tool marks frames, so a
# landing it adds is a None until a click places one between two frames; a point written by hand
# is carried through a save unchanged.
Landings = Dict[float, LabelPoint]


def load_landings(path: Path) -> Landings:
    """The clip's landings from its landing file; none when there is no file. A damaged file is
    its LabelFormatError, naming the file: a record of where the ball landed is never replaced
    by an empty one."""
    if not path.is_file():
        return {}
    return {label.frame: label.point for label in read_bounce_labels(path)}


def save_landings(path: Path, landings: Landings) -> None:
    write_bounce_labels(path, (BounceLabel(frame, point) for frame, point in landings.items()))


class HitMark(NamedTuple):
    """One hit the tool holds: where the racket met the ball (None until a click places it, as a
    landing's point) and who hit it (None until , says so; never taken from the tracker, whose
    hitter the scorer measures against this one)."""

    point: LabelPoint = None
    hitter: Optional[Hitter] = None


# A clip's hits (``event_labels``), keyed as its landings are (``Landings``): frame -> its HitMark.
Hits = Dict[float, HitMark]


def load_hits(path: Path) -> Hits:
    """The clip's hits; none when there is no file. A damaged file is its LabelFormatError."""
    if not path.is_file():
        return {}
    return {label.frame: HitMark(label.point, label.hitter) for label in read_hit_labels(path)}


def no_point(moved: Optional[LabelPoint]) -> LabelPoint:
    """A landing placed or moved between two frames (``LabelSession.toggle_mark``): no point yet."""
    return None


def blank_hit(moved: Optional[HitMark]) -> HitMark:
    """A hit placed, or moved between two frames: no point yet, and the hitter it had."""
    return HitMark(None, moved.hitter if moved is not None else None)


def save_hits(path: Path, hits: Hits) -> None:
    write_hit_labels(path, (HitLabel(frame, mark.point, mark.hitter) for frame, mark in hits.items()))


def load_rallies(path: Path) -> Tuple[Set[int], Set[int]]:
    """The frames the clip's rallies start on and end on (the tool marks the two apart and pairs
    them on save, ``event_labels.pair_rally_marks``); none when there is no file."""
    if not path.is_file():
        return set(), set()
    rallies = read_rally_labels(path)
    return ({rally.start for rally in rallies if rally.start is not None},
            {rally.end for rally in rallies if rally.end is not None})


def save_rallies(path: Path, starts: Set[int], ends: Set[int]) -> None:
    rallies: List[RallyLabel] = pair_rally_marks(starts, ends)
    write_rally_labels(path, rallies)


# The candidates rejected (``label_candidates``): candidate kind -> the frames of the ones rejected.
Rejected = Dict[str, Set[int]]


def load_event_review(path: Path) -> Rejected:
    """The candidates the person labelling rejected, by kind; none when there is no file. A
    damaged file is a LabelFormatError naming it: a record of what was judged is never replaced by
    an empty one."""
    state = read_json_object(path, "event review file")
    if state is None:
        return {}
    rejected = state.get("rejected") or {}
    if not isinstance(rejected, dict):
        raise LabelFormatError(f"{path}: 'rejected' is not an object of frame lists")
    return {str(kind): set(frame_list(rejected, kind, path)) for kind in rejected}


def save_event_review(path: Path, rejected: Rejected) -> None:
    """Write the candidates rejected into the event review file, keeping whatever else it holds."""
    try:
        state = read_json_object(path, "event review file") or {}
    except LabelFormatError:
        state = {}                    # a rescue copy over a damaged one: what this session knows
    kept = {kind: sorted(frames) for kind, frames in sorted(rejected.items()) if frames}
    write_atomic(path, json.dumps({**state, "rejected": kept}, indent=1))


def load_players(path: Path) -> List[PlayerBoxLabel]:
    """The clip's player identity labels; none when there is no file."""
    return read_player_labels(path) if path.is_file() else []


def save_players(path: Path, labels: List[PlayerBoxLabel]) -> None:
    write_player_labels(path, labels)


# A clip's court labels (``court_labels``): frame -> the points placed on it.
CourtFrames = Dict[int, CourtPoints]


def load_court(path: Path) -> CourtFrames:
    """The clip's court labels; none when there is no file."""
    return {label.frame: dict(label.points) for label in read_court_labels(path)} if path.is_file() else {}


def save_court(path: Path, frames: CourtFrames) -> None:
    write_court_labels(path, [CourtLabel(frame, points) for frame, points in frames.items() if points])


# The settings file holds the toggles worth keeping between sessions: filter, inset, grid size, trail.
Settings = Dict[str, Union[str, bool, int]]


def load_settings(path: Path) -> Settings:
    """The toggles worth keeping between sessions. Missing or broken file: the defaults."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return blob if isinstance(blob, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(path: Path, settings: Settings) -> None:
    try:
        write_atomic(path, json.dumps(settings, indent=1))
    except OSError as error:
        # A convenience file: losing it costs the next session its remembered toggles, never any
        # labels, so the tool warns and carries on rather than failing an edit or a quit over it.
        print(f"[warn] could not save settings to {path}: {error}")
