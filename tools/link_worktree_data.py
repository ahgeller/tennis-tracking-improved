"""Give a git worktree of optimization_test the untracked data it needs to run.

A fresh worktree has the code but none of the gitignored assets: the model weights
and engines, the sample clips, the training videos, the 768x432 frame cache and the
imported public labels. Run this from inside the worktree:

    python tools/link_worktree_data.py

Large, read-mostly folders (`finetune/videos`, `finetune/cache`) become directory
junctions to the main copy, so they cost no disk space and a frame the trainer
adds to the cache is shared. Small files are copied, so a worktree can change its
own weights without touching the main copy. Existing files are never overwritten.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\Andrew\Desktop\clean_version\optimization_test")
HERE = Path(__file__).resolve().parent.parent

JUNCTIONS = ("finetune/videos", "finetune/cache")
COPY_DIRS = ("system/models", "system/sample", "finetune/models")
COPY_GLOBS = (("finetune/labels", "*"),)
COPY_FILES = ("finetune/.training-python",)


def junction(rel: str) -> None:
    target, link = MAIN / rel, HERE / rel
    if link.exists():
        print(f"keep     {rel}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    print(f"junction {rel} -> {target}")


def copy_missing(src: Path, dst: Path) -> int:
    count = 0
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


def main() -> int:
    if HERE == MAIN:
        print("this is the main copy; run the script from inside a worktree")
        return 1
    for rel in JUNCTIONS:
        junction(rel)
    for rel in COPY_DIRS:
        print(f"copied   {copy_missing(MAIN / rel, HERE / rel):5d} files into {rel}")
    for rel, pattern in COPY_GLOBS:
        n = 0
        for item in (MAIN / rel).glob(pattern):
            out = HERE / rel / item.name
            if item.is_file() and not out.exists():
                shutil.copy2(item, out)
                n += 1
        print(f"copied   {n:5d} files into {rel}")
    for rel in COPY_FILES:
        if (MAIN / rel).exists() and not (HERE / rel).exists():
            shutil.copy2(MAIN / rel, HERE / rel)
            print(f"copied   {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
