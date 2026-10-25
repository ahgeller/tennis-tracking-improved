"""Writing workspace files so a crash, a power cut or a locked file never leaves a torn one.

Every label CSV, sidecar JSON, manifest and settings file in the workspace, and every weights
file the tracker runs, goes through here: the content is written to a temporary file beside the
target, flushed to disk, and then moved over the target, so the target is always either the old
file or the complete new one. Windows refuses the move while another program (Excel, an editor,
a virus scan) has the target open, so the move is retried for a moment before the error is
raised.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING, BinaryIO, Iterator, TextIO

if TYPE_CHECKING:
    from _csv import _writer as CsvWriter     # what csv.writer() returns; csv.writer itself is a function

REPLACE_ATTEMPTS = 5       # tries before a locked target is given up on ...
REPLACE_PAUSE = 0.1        # ... and the seconds waited between them


def replace_file(temporary: Path, path: Path) -> None:
    """Move a finished temporary file over `path`, retrying a Windows sharing lock briefly."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_PAUSE)


def temporary_for(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


@contextmanager
def atomic_handle(path: Path, mode: str, **options) -> Iterator[IO]:
    """A handle opened with `mode` (and `options`, as ``Path.open`` takes them) on a temporary file
    beside `path`, whose content lands at `path` only once it is complete and on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_for(path)
    with temporary.open(mode, **options) as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    replace_file(temporary, path)


@contextmanager
def atomic_text(path: Path, newline: str = "") -> Iterator[TextIO]:
    """A text handle whose content lands at `path` only once it is complete and on disk."""
    with atomic_handle(path, "w", encoding="utf-8", newline=newline) as handle:
        yield handle


@contextmanager
def atomic_binary(path: Path) -> Iterator[BinaryIO]:
    """A binary handle whose content lands at `path` only once it is complete and on disk (what
    ``torch.save`` and ``np.save`` write to, for a checkpoint or weights)."""
    with atomic_handle(path, "wb") as handle:
        yield handle


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file, flushed to disk first."""
    with atomic_text(path) as handle:
        handle.write(text)


@contextmanager
def atomic_csv(path: Path) -> Iterator["CsvWriter"]:
    """A csv.writer whose rows land at `path` only once every row is written and on disk."""
    with atomic_text(path, newline="") as handle:
        yield csv.writer(handle)


def copy_atomic(source: Path, path: Path) -> None:
    """Copy `source` over `path` through a temporary file beside it, flushed to disk first, so a
    reader of `path` sees the old file or the complete copy and nothing in between."""
    with source.open("rb") as reader, atomic_binary(path) as writer:
        shutil.copyfileobj(reader, writer)


def backup_copy(path: Path, tag: str) -> Path:
    """Copy `path` to `<name>.before_<tag>_<timestamp>` beside it and return the copy's path. Every
    bulk write over a workspace file (an import, a shift, a realignment, a promotion) makes one."""
    stamp = f"{dt.datetime.now():%Y%m%d_%H%M%S_%f}"
    backup = path.with_name(f"{path.name}.before_{tag}_{stamp}")
    shutil.copyfile(path, backup)
    return backup
