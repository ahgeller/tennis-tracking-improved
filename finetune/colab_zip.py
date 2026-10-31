"""The one zip the Colab notebooks unpack: ``python finetune/ft.py colab-zip``.

``Optimization_Train_Colab.ipynb`` (and the older ``GridTrackNet_FineTune_Colab.ipynb``) run the trainer on Google Drive, where everything it reads
or writes sits in the one folder ``MyDrive/tennis_finetune/`` (the zips, and ``runs/<TAG>/`` for
the outputs). It unpacks a single ``tennis_finetune_colab.zip`` whose entries all sit under
``tennis_finetune/``:

    tennis_finetune/videos/<every video in the workspace>
    tennis_finetune/labels/<every file in the workspace labels folder>
    tennis_finetune/clips.csv, review_status.json, holdouts.json, exclude.txt
    tennis_finetune/models/ball_detector.pt                  (the workspace's model: the start and the teacher)
    tennis_finetune/pseudo/videos/, pseudo/labels/*_pseudo.csv  (when the workspace has pseudo-labels)
    tennis_finetune/code/<the trainer's import closure>      (``code_files``)
    tennis_finetune/code/code_manifest.json                  (that list, which the notebooks check)

``code/`` is laid out as the repository is: ``finetune/`` at its top and the tracker's modules under
``system/``, which ``finetune/__init__.py`` puts on ``sys.path`` there as it does here.

``--code-only`` writes ``tennis_finetune_code.zip`` with the records and the code alone; the
notebook lays it over the big zip's copy when it is on Drive, so a code fix is a small upload.

``labels/`` is copied whole, sidecars, drafts and ``.before_*`` backups included, so the zip
is also a complete backup of the hand labels. ``clips.csv`` and ``review_status.json`` must be
present: without them the trainer on Colab treats every clip as untrusted and trains on nothing.
Entries are stored, not deflated: the videos are already compressed, and deflating 900 MB of
them costs minutes for nothing.

The code is derived from the code, not listed by hand: ``code_files`` walks the import closure
of ``CODE_ROOTS`` (the trainer and the pseudo-label tool) through their source with ``ast``, every
``finetune.*``, ``tennis_tracker.*`` and ``ball_in_play_selector.*`` import at module level or
inside a function, relative ones resolved, and the packages above every module. A network a new
``ball_models`` architecture adds, or a new import in the trainer, goes in without an edit
here; ``tests/test_colab_notebook.py`` imports the trainer from the staged files alone, as Colab
does. Like ``ft.py``, this imports only the standard library and ``finetune.data``, so it loads
under any interpreter.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set

from finetune import REPOSITORY
from finetune.data import workspace as ws
from finetune.data.fileio import replace_file, temporary_for
from finetune.data.workspace import videos_in
from tennis_tracker.paths import HEATMAP_WEIGHTS

NOTEBOOK = REPOSITORY / "finetune" / "GridTrackNet_FineTune_Colab.ipynb"
OPTIMIZATION_NOTEBOOK = REPOSITORY / "finetune" / "Optimization_Train_Colab.ipynb"
CODE_ROOTS = ("finetune.training.train_gridtracknet", "finetune.training.pseudo_labels")
PACKAGES = ("finetune", "tennis_tracker", "ball_in_play_selector")     # the repository's own top-level packages
CODE_MANIFEST = "code_manifest.json"                   # under code/: the staged files, which the notebooks check
PSEUDO = "pseudo"                                      # the workspace's pseudo-label folder (pseudo_labels.PSEUDO_DIR)
PSEUDO_SUFFIX = "_pseudo.csv"
TOP = "tennis_finetune"                                 # the folder every entry sits under
DEFAULT_OUT = REPOSITORY / "output" / "tennis_finetune_colab.zip"
DEFAULT_CODE_OUT = REPOSITORY / "output" / "tennis_finetune_code.zip"      # --code-only: the code and records alone
REQUIRED_RECORDS = (ws.MANIFEST_NAME, ws.REVIEW_STATUS_NAME)      # missing: the trainer trusts no clip
OPTIONAL_RECORDS = (ws.HOLDOUTS_NAME, ws.EXCLUDE_NAME)


# --------------------------------------------------------------------------- the code's import closure

def package_root(name: str, repository: Path = REPOSITORY) -> Path:
    """The folder a module's top-level package sits in: the repository for finetune, system/ for the
    tracker's packages."""
    return repository if name.split(".")[0] == "finetune" else repository / "system"


def module_path(name: str, repository: Path = REPOSITORY) -> Path:
    """The file of the module `name`: ``<root>/a/b.py``, or ``<root>/a/b/__init__.py`` for a package.
    One that is neither is a ValueError naming it (an import of a module that does not exist)."""
    root = package_root(name, repository)
    for path in (root / (name.replace(".", "/") + ".py"), root / name.replace(".", "/") / "__init__.py"):
        if path.is_file():
            return path
    raise ValueError(f"no module file for {name} under {root}")


def module_name(path: Path, repository: Path = REPOSITORY) -> str:
    """The dotted name of a module file under the repository (``system/tennis_tracker/config.py`` ->
    ``tennis_tracker.config``; a package's ``__init__.py`` is the package)."""
    system = repository / "system"
    parts = path.relative_to(system if path.is_relative_to(system) else repository).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def internal_imports(path: Path, repository: Path = REPOSITORY) -> Iterator[str]:
    """Every module of ``PACKAGES`` that `path` imports, at module level or inside a function,
    relative imports (``from .config import Config``) resolved against `path`, and the submodule
    a ``from package import module`` names."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PACKAGES:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative: the package `path` sits in, `level - 1` levels up
                package = module_name(path.parent / "__init__.py", repository).split(".")
                base = ".".join(package[:len(package) - (node.level - 1)])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if not module or module.split(".")[0] not in PACKAGES:
                continue
            yield module
            for alias in node.names:
                if (package_root(module, repository) / module.replace(".", "/") / f"{alias.name}.py").is_file():
                    yield f"{module}.{alias.name}"


def import_closure(roots: Iterable[str], repository: Path = REPOSITORY) -> Set[str]:
    """Every module `roots` import, transitively, with every package above each one (importing a
    submodule runs its packages' ``__init__``)."""
    seen: Set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(internal_imports(module_path(name, repository), repository))
    for name in list(seen):
        parts = name.split(".")
        seen.update(".".join(parts[:depth]) for depth in range(1, len(parts)))
    return seen


def code_files(roots: Iterable[str] = CODE_ROOTS, repository: Path = REPOSITORY) -> List[str]:
    """The files of `roots`' import closure, relative to the repository root (``system/...`` for
    the tracker's) with forward slashes, sorted: what ``code/`` holds."""
    return sorted(module_path(name, repository).relative_to(repository).as_posix()
                  for name in import_closure(roots, repository))


def staged_files() -> List[str]:
    """The code the zip stages: ``code_files`` of ``CODE_ROOTS``."""
    return code_files()


class ColabZipError(Exception):
    """A file the notebook cannot do without is missing."""


@dataclass
class ZipPlan:
    """Which files go in, by section: the entry name under ``TOP`` -> the file on disk; and the
    files written from memory (``code_manifest.json``)."""
    sections: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)        # optional records that were not there
    generated: Dict[str, str] = field(default_factory=dict)

    def add(self, section: str, entry: str, path: Path) -> None:
        self.sections.setdefault(section, {})[f"{TOP}/{entry}"] = path

    @property
    def entries(self) -> Dict[str, Path]:
        return {entry: path for files in self.sections.values() for entry, path in files.items()}


def plan_zip(code_only: bool = False) -> ZipPlan:
    """Everything the zip will hold, from the workspace, the workspace's model and the notebook's
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
            raise ColabZipError(f"{layout.repo_weights} is missing; the notebook starts from the workspace's model")
        plan.add("models", f"models/{HEATMAP_WEIGHTS.name}", layout.repo_weights)
        pseudo = layout.root / PSEUDO
        for clip, video in (videos_in(pseudo / "videos") if (pseudo / "videos").is_dir() else {}).items():
            labels = pseudo / "labels" / f"{clip}{PSEUDO_SUFFIX}"
            if labels.is_file():              # an unlabelled pseudo video trains nothing: left out
                plan.add("pseudo", f"{PSEUDO}/videos/{video.name}", video)
                plan.add("pseudo", f"{PSEUDO}/labels/{labels.name}", labels)
    try:
        code = staged_files()
    except ValueError as error:
        raise ColabZipError(f"the trainer's code imports a module that is not in the repository: {error}") from error
    for name in code:
        plan.add("code", f"code/{name}", REPOSITORY / name)
    plan.generated[f"{TOP}/code/{CODE_MANIFEST}"] = json.dumps(code, indent=1)
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
            for entry, text in plan.generated.items():
                archive.writestr(entry, text)
                stored += len(text.encode("utf-8"))
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
    print(f"{out}: {len(plan.entries) + len(plan.generated)} entries, {stored / 2**20:.1f} MB stored, {time.perf_counter() - started:.1f} s")
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
