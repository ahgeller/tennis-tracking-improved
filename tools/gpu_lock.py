"""Run one command while holding the shared GPU benchmark lock.

Several agents and people share one RTX 5080 while building optimization_test.
A timing taken while another job uses the GPU is wrong (the first research round
measured 54.8 s for a run that takes 26-28 s alone), so every timed run goes
through this script:

    python tools/gpu_lock.py -- python clean_tracker.py --input ... --info

The lock is a directory (``mkdir`` is atomic on NTFS and POSIX). A lock older
than ``--stale-minutes`` is assumed abandoned by a crashed holder and taken over.
Correctness runs that do not report timings do not need the lock.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def _main_checkout() -> Path:
    """The main working tree of this repository, which every git worktree of it shares: a lock
    beside this script would be one lock per worktree, and agents in different worktrees would
    time their runs over each other. A checkout's ``.git`` is a directory; a worktree's is a file
    naming ``<main>/.git/worktrees/<name>``."""
    here = Path(__file__).resolve().parent.parent
    marker = here / ".git"
    if marker.is_file():
        text = marker.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            if gitdir.parent.name == "worktrees":
                return gitdir.parent.parent.parent
    return here


LOCK = _main_checkout() / ".gpu_bench_lock"


def acquire(stale_minutes: float, poll_seconds: float) -> None:
    waited = 0.0
    while True:
        try:
            LOCK.mkdir()
            (LOCK / "owner.txt").write_text(f"pid {os.getpid()} since {time.ctime()}\n{' '.join(sys.argv)}\n")
            return
        except FileExistsError:
            age = time.time() - LOCK.stat().st_mtime
            if age > stale_minutes * 60:
                print(f"[gpu_lock] taking over a stale lock ({age / 60:.0f} min old)", file=sys.stderr)
                release()
                continue
            if waited == 0:
                owner = (LOCK / "owner.txt").read_text() if (LOCK / "owner.txt").exists() else "?"
                print(f"[gpu_lock] waiting for: {owner.strip()}", file=sys.stderr)
            time.sleep(poll_seconds)
            waited += poll_seconds


def release() -> None:
    for child in LOCK.glob("*"):
        child.unlink(missing_ok=True)
    try:
        LOCK.rmdir()
    except FileNotFoundError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stale-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    acquire(args.stale_minutes, args.poll_seconds)
    try:
        return subprocess.call(command)
    finally:
        release()


if __name__ == "__main__":
    raise SystemExit(main())
