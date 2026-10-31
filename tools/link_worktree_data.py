"""Give a git worktree of optimization_test the untracked data it needs to run.

A fresh worktree has the code but none of the gitignored assets: the model weights
and engines, the sample clips, the training videos, the 768x432 frame cache and the
imported public labels. Run this from inside the worktree:

    python tools/link_worktree_data.py

Large, read-mostly folders (`finetune/videos`, `finetune/cache`) become directory
junctions to the main copy, so they cost no disk space and a frame the trainer
adds to the cache is shared. Small files are copied, so a worktree can change its
own weights without touching the main copy. Existing files are never overwritten.

The candidate weights (`system/models/candidates`) and the TensorRT engine cache
(`system/models/engine_cache`) are copied from the reference worktree,
`phases/ABplus` (`--reference` names another), before the main copy's
`system/models` fills in the rest. Each worktree would otherwise build its own
stream_s4 engine on first use, and two builds of the same weights choose different
FP16 tactics, so their detections differ in the fourth decimal: outputs compare
across worktrees only on the same engine file (audit/records/opt_speed_motion.md).
They are copied, not junctioned, so a worktree that builds an engine for weights of
its own keeps it to itself. An engine a worktree has already built is kept; delete
it to take the reference's.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

MAIN = Path(r"C:\Users\Andrew\Desktop\clean_version\optimization_test")
REFERENCE = MAIN / "phases" / "ABplus"
HERE = Path(__file__).resolve().parent.parent

JUNCTIONS = ("finetune/videos", "finetune/cache")
REFERENCE_DIRS = ("system/models/candidates", "system/models/engine_cache")
COPY_DIRS = ("system/models", "system/sample", "finetune/models")
COPY_GLOBS = (("finetune/labels", "*"),)
COPY_FILES = ("finetune/.training-python",)


def junction(here: Path, main: Path, rel: str) -> None:
    target, link = main / rel, here / rel
    if link.exists():
        print(f"keep     {rel}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    print(f"junction {rel} -> {target}")


def copy_missing(src: Path, dst: Path) -> int:
    count = 0
    if not src.is_dir():
        return count
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        out = dst / item.relative_to(src)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)
        count += 1
    return count


def copy_reference_models(here: Path, reference: Path) -> None:
    """The reference worktree's candidate weights and engines, copied into ``here``."""
    if here.resolve() == reference.resolve():
        print(f"keep     {', '.join(REFERENCE_DIRS)} (this is the reference worktree)")
        return
    if not reference.is_dir():
        print(f"missing  reference worktree {reference}: candidates and engines come from the main copy only")
        return
    for rel in REFERENCE_DIRS:
        print(f"copied   {copy_missing(reference / rel, here / rel):5d} files into {rel} from {reference}")


def copy_data(here: Path, main: Path, reference: Path) -> None:
    """Everything this script copies (not the junctions), the reference's models first."""
    copy_reference_models(here, reference)
    for rel in COPY_DIRS:
        print(f"copied   {copy_missing(main / rel, here / rel):5d} files into {rel}")
    for rel, pattern in COPY_GLOBS:
        n = 0
        for item in (main / rel).glob(pattern):
            out = here / rel / item.name
            if item.is_file() and not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, out)
                n += 1
        print(f"copied   {n:5d} files into {rel}")
    for rel in COPY_FILES:
        if (main / rel).exists() and not (here / rel).exists():
            shutil.copy2(main / rel, here / rel)
            print(f"copied   {rel}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help=f"the worktree whose candidates and engine cache are copied (default {REFERENCE})")
    args = parser.parse_args(argv)
    if HERE == MAIN:
        print("this is the main copy; run the script from inside a worktree")
        return 1
    for rel in JUNCTIONS:
        junction(HERE, MAIN, rel)
    copy_data(HERE, MAIN, args.reference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
