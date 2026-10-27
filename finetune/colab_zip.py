"""The one zip the Colab notebook unpacks: ``python finetune/ft.py colab-zip``.

``GridTrackNet_FineTune_Colab.ipynb`` runs the trainer on Google Drive, where everything it reads
or writes sits in the one folder ``MyDrive/tennis_finetune/`` (the zips, and ``runs/<TAG>/`` for
the outputs). It unpacks a single ``tennis_finetune_colab.zip`` whose entries all sit under
``tennis_finetune/``:

    tennis_finetune/videos/<every video in the workspace>
    tennis_finetune/labels/<every file in the workspace labels folder>
    tennis_finetune/clips.csv, review_status.json, holdouts.json, exclude.txt
    tennis_finetune/models/ball_detector.pt                  (the tracker's weights: the start)
    tennis_finetune/code/<the trainer's import closure>      (the notebook's REQUIRED list)

``--code-only`` writes ``tennis_finetune_code.zip`` with the records and the code alone; the
notebook lays it over the big zip's copy when it is on Drive, so a code fix is a small upload.

``labels/`` is copied whole, sidecars, drafts and ``.before_*`` backups included, so the zip
is also a complete backup of the hand labels. ``clips.csv`` and ``review_status.json`` must be
present: without them the trainer on Colab treats every clip as untrusted and trains on nothing.
Entries are stored, not deflated: the videos are already compressed, and deflating 900 MB of
them costs minutes for nothing.

``staged_files`` reads the REQUIRED list out of the notebook, so the zip and
``tests/test_colab_notebook.py`` (which checks that list against the trainer's imports) agree
on what code goes in. Like ``ft.py``, this imports only the standard library and
``finetune.data``, so it loads under any interpreter.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from finetune.data import workspace as ws
from finetune.data.fileio import replace_file, temporary_for
from finetune.data.workspace import videos_in
from tennis_tracker.paths import BALL_WEIGHTS, ROOT

NOTEBOOK = ROOT / "finetune" / "GridTrackNet_FineTune_Colab.ipynb"
TOP = "tennis_finetune"                                 # the folder every entry sits under
DEFAULT_OUT = ROOT / "output" / "tennis_finetune_colab.zip"
DEFAULT_CODE_OUT = ROOT / "output" / "tennis_finetune_code.zip"      # --code-only: the code and records alone
REQUIRED_RECORDS = (ws.MANIFEST_NAME, ws.REVIEW_STATUS_NAME)      # missing: the trainer trusts no clip
OPTIONAL_RECORDS = (ws.HOLDOUTS_NAME, ws.EXCLUDE_NAME)


def staged_files() -> List[str]:
    """The REQUIRED list in the notebook's staging cell: the repository files, relative to the
    root with forward slashes, that the trainer imports on Colab."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "REQUIRED = [" in source:
            tree = ast.parse(source.replace("!pip", "pass  # pip"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "REQUIRED":
                    return ast.literal_eval(node.value)
    raise ValueError(f"{NOTEBOOK.name} has no staging cell with a REQUIRED list")


class ColabZipError(Exception):
    """A file the notebook cannot do without is missing."""


@dataclass
class ZipPlan:
    """Which files go in, by section: the entry name under ``TOP`` -> the file on disk."""
    sections: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)        # optional records that were not there

    def add(self, section: str, entry: str, path: Path) -> None:
        self.sections.setdefault(section, {})[f"{TOP}/{entry}"] = path

    @property
    def entries(self) -> Dict[str, Path]:
        return {entry: path for files in self.sections.values() for entry, path in files.items()}


def plan_zip(code_only: bool = False) -> ZipPlan:
    """Everything the zip will hold, from the workspace, the tracker's weights and the notebook's
    REQUIRED list; a ``ColabZipError`` when a required record, the weights or a staged file is missing.
    With `code_only` just the records and the code: the small zip the notebook lays over the big
    one, so a code fix never means uploading the videos again."""
    layout = ws.current()
    plan = ZipPlan()
    if not code_only:
        for video in videos_in().values():
            plan.add("videos", f"videos/{video.name}", video)
        if layout.labels.is_dir():
            for path in sorted(layout.labels.iterdir()):
                if path.is_file():
                    plan.add("labels", f"labels/{path.name}", path)
    for name in REQUIRED_RECORDS:
        if not (layout.root / name).is_file():
            raise ColabZipError(f"{layout.root / name} is missing; without it the trainer on Colab trusts no clip")
        plan.add("records", name, layout.root / name)
    for name in OPTIONAL_RECORDS:
        if (layout.root / name).is_file():
            plan.add("records", name, layout.root / name)
        else:
            plan.skipped.append(name)
    if not code_only:
        if not layout.repo_weights.is_file():
            raise ColabZipError(f"{layout.repo_weights} is missing; the notebook starts from the tracker's weights")
        plan.add("models", f"models/{BALL_WEIGHTS.name}", layout.repo_weights)
    for name in staged_files():
        if not (ROOT / name).is_file():
            raise ColabZipError(f"{name} is in the notebook's REQUIRED list but not in the repository")
        plan.add("code", f"code/{name}", ROOT / name)
    return plan


def write_zip(plan: ZipPlan, out: Path) -> int:
    """Store the plan's files at `out`, written beside it first and moved over when complete, so
    `out` is never a half-written archive and a failure leaves no temporary. Returns the bytes
    stored."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_for(out)
    stored = 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for entry, path in plan.entries.items():
                archive.write(path, entry)
                stored += path.stat().st_size
        replace_file(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return stored


def build_colab_zip(out: Path = DEFAULT_OUT, code_only: bool = False) -> Path:
    """Plan, write and report the zip; returns its path."""
    started = time.perf_counter()
    plan = plan_zip(code_only)
    for name in plan.skipped:
        print(f"[skip] no {name} in {ws.current().root}")
    stored = write_zip(plan, out)
    for section, files in plan.sections.items():
        print(f"  {section:8s} {len(files):5d} files  {sum(p.stat().st_size for p in files.values()) / 2**20:8.1f} MB")
    print(f"{out}: {len(plan.entries)} entries, {stored / 2**20:.1f} MB stored, {time.perf_counter() - started:.1f} s")
    name = DEFAULT_CODE_OUT.name if code_only else DEFAULT_OUT.name
    print(f"next: upload it to MyDrive/tennis_finetune/ as {name} and run the notebook")
    return out


def cmd_colab_zip(args: argparse.Namespace) -> int:
    """``ft.py colab-zip [--out PATH] [--code-only]``: 0 with the zip written, 1 with a ``[fail]`` line
    when a file the notebook cannot do without is missing."""
    default = DEFAULT_CODE_OUT if args.code_only else DEFAULT_OUT
    try:
        build_colab_zip(Path(args.out) if args.out else default, args.code_only)
    except ColabZipError as error:
        print(f"[fail] {error}")
        return 1
    return 0
