"""Which of a clip's label files a label tool save writes, and how it keeps from losing anybody's work.

A clip's labels come in kinds, each its own file or files (``KINDS``): the ball labels, the review
state, the landings, hits, rallies, the candidates rejected, the player and the court labels. An
edit says which kind it changed (``LabelSession.touch``), and a save writes only those kinds, so a
court session never rewrites the ball labels from what it read at the start, and a ball edit never
rewrites a landing file somebody else is keeping.

Two sessions may have one clip open (a ball session and a court session, say). Every file a kind
writes is fingerprinted when it is read (``Source.remember``); a save that finds the file changed
since is refused for that kind: nothing is written over, this session's version goes to a rescue
copy beside it (``workspace.unsaved_copy``) and a warning names both, once. The kind stays unsaved,
and later saves only refresh the rescue copy.

Each kind is written on its own: a file another program holds open fails its kind alone, and is
retried at the next save; whatever is still unsaved when the tool closes goes to its rescue copy.

A kind with nothing left in it (the last landing taken off) removes its file only when the file
held something: the tool emptied it. A file that was already empty, its header alone, says "none
in this clip" and is left as it is; a scorer reads it so.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, NamedTuple, Optional, Tuple

from finetune.data.event_rows import read_records
from finetune.data.labels import LabelFormatError, read_json_object
from finetune.data.workspace import unsaved_copy
from finetune.labeling.label_store import (save_court, save_event_review, save_flags, save_hits, save_landings,
                                           save_players, save_rallies, save_review, save_rows)

if TYPE_CHECKING:
    from finetune.labeling.label_source import Source

# The kinds of label a clip has, in the order they are saved.
KINDS = ("ball", "review", "landings", "hits", "rallies", "rejected", "players", "court")
WHAT = {"ball": "ball labels", "review": "review state", "landings": "landings", "hits": "hits",
        "rallies": "rallies", "rejected": "rejected candidates", "players": "player labels", "court": "court labels"}


class Part(NamedTuple):
    """One file a kind writes: where, how (to any path, so the same write makes the rescue copy),
    whether there is nothing to write, and whether a change on disk since it was read stops the
    write (the flag file is merged at write time instead, ``label_store.save_flags``)."""

    path: Path
    write: Callable[[Path], None]
    empty: bool = False
    guarded: bool = True


def fingerprint(path: Path) -> Optional[str]:
    """What the file holds, as a hash; None when there is no file."""
    try:
        return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
    except FileNotFoundError:
        return None


def parts(source: "Source", kind: str) -> List[Part]:
    """The files `kind` writes for `source`."""
    if kind == "ball":
        return [Part(source.out_csv, lambda path: save_rows(path, source.rows, source.width, source.height))]
    if kind == "review":
        return [Part(source.review, lambda path: save_review(path, source.rows, source.cursor, source.zoom)),
                Part(source.suspects, lambda path: save_flags(path, source.rows), guarded=False)]
    if kind == "landings":
        return [Part(source.bounces, lambda path: save_landings(path, source.landings), not source.landings)]
    if kind == "hits":
        return [Part(source.hits_path, lambda path: save_hits(path, source.hits), not source.hits)]
    if kind == "rallies":
        return [Part(source.rallies_path, lambda path: save_rallies(path, source.rally_starts, source.rally_ends),
                     not (source.rally_starts or source.rally_ends))]
    if kind == "rejected":
        return [Part(source.events_review, lambda path: save_event_review(path, source.rejected),
                     not any(source.rejected.values()) and not event_review_extras(source.events_review))]
    if kind == "players":
        return [Part(source.players_path, lambda path: save_players(path, source.players or []), not source.players)]
    if kind == "court":
        return [Part(source.court_path, lambda path: save_court(path, source.court or {}),
                     not any((source.court or {}).values()))]
    raise ValueError(f"no such kind of label: {kind!r}")


def holds_rows(path: Path) -> bool:
    """Whether a label file says something: a CSV a row past its header, the event review file a
    candidate rejected."""
    if path.suffix == ".json":
        state = read_json_object(path, "event review file") or {}
        return any(state.get("rejected", {}).values()) if isinstance(state.get("rejected"), dict) else bool(state)
    return bool(read_records(path, ())[1])


def event_review_extras(path: Path) -> bool:
    """Whether the event review file holds more than the candidates rejected (kept by a rewrite)."""
    try:
        return bool(set(read_json_object(path, "event review file") or {}) - {"rejected"})
    except LabelFormatError:
        return True                   # damaged: never removed, the write says what is wrong


def save_kind(source: "Source", kind: str) -> List[Path]:
    """Write `kind`'s files, unless one of them changed on disk since it was read: then nothing is
    written and those files are returned. A failed write is its OSError (or LabelFormatError, for
    a file merged at write time that was damaged meanwhile)."""
    plan = parts(source, kind)
    moved = [part.path for part in plan if part.guarded and fingerprint(part.path) != source.disk.get(part.path)]
    if moved:
        return moved
    for part in plan:
        if not part.empty:
            part.write(part.path)
        elif part.path.is_file() and holds_rows(part.path):
            part.path.unlink()        # the tool took the last one off: no file, not one saying "none"
        if part.guarded:
            source.disk[part.path] = fingerprint(part.path)
    return []


def rescue_kind(source: "Source", kind: str) -> List[Path]:
    """Write `kind` to its rescue copies beside the real files; the copies written."""
    copies = []
    for part in parts(source, kind):
        if part.guarded:
            part.write(unsaved_copy(part.path))
            copies.append(unsaved_copy(part.path))
    return copies


def save_source(source: "Source") -> Tuple[List[str], List[str]]:
    """Write every kind `source` has changed, each on its own (``save_kind``): (the kinds written,
    the kinds still unsaved). A kind refused because its file changed on disk is warned about once
    and rescued, then only rescued at every later save."""
    written, unsaved = [], []
    for kind in [kind for kind in KINDS if kind in source.changed]:
        if kind in source.conflicted:
            try:
                rescue_kind(source, kind)
            except OSError as error:
                print(f"[warn] {source.clip}: could not refresh the rescue copy of the {WHAT[kind]}: {error}")
            unsaved.append(kind)
            continue
        try:
            moved = save_kind(source, kind)
        except (OSError, LabelFormatError) as error:
            # Another program has the file open, most likely. The edits are kept in memory and the
            # next save tries again; one stuck kind does not stop the others.
            print(f"[warn] could not save the {WHAT[kind]} of {source.clip}: {error} - close the file wherever it "
                  f"is open; the edits are kept and saving is retried")
            unsaved.append(kind)
            continue
        if moved:
            source.conflicted.add(kind)
            unsaved.append(kind)
            try:
                copies = ", ".join(path.name for path in rescue_kind(source, kind))
            except OSError as error:
                copies = f"nowhere: the rescue copy failed too ({error})"
            print(f"[warn] {', '.join(path.name for path in moved)} changed on disk since this session read it "
                  f"(another session on {source.clip}?) and was not written over. This session's {WHAT[kind]} "
                  f"are in {copies}; reopen the clip to carry on from what is on disk.")
            continue
        source.changed.discard(kind)
        written.append(kind)
        if kind == "review":
            source.saved_cursor = source.cursor
    return written, unsaved


def rescue_source(source: "Source") -> None:
    """At exit: every kind still unsaved goes to its rescue copies (``rescue_kind``), so closing the
    tool never throws edits away."""
    for kind in [kind for kind in KINDS if kind in source.changed]:
        try:
            copies = rescue_kind(source, kind)
        except OSError as error:
            # The last resort has failed and the tool is closing: there is nowhere left to put
            # these, so say so plainly and go on to rescue the rest.
            print(f"[error] {source.clip}: the unsaved {WHAT[kind]} are lost, their rescue copy failed too: {error}")
            continue
        print(f"[save] the {WHAT[kind]} of {source.clip} could not be written; they are in "
              f"{', '.join(path.name for path in copies)}")
