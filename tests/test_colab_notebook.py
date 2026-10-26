"""The Colab notebook stages every repository module the trainer imports.

train_gridtracknet.py imports a handful of finetune.* and tennis_tracker.* modules (some inside
functions); the notebook's staging cell lists the files it needs as REQUIRED (``finetune.colab_zip.staged_files``
reads it; ``ft.py colab-zip`` packs those files). This walks the trainer's internal import closure
from the source and checks the two agree, so a new import in the trainer fails here instead of
with ModuleNotFoundError on Colab. The settings cell is checked for the trainer flags the notebook
must pass: it runs both the held-out selection run and the ``--all-clips`` run that follows it."""
import ast
import json
from pathlib import Path
from typing import Iterator, Set

from finetune.colab_zip import NOTEBOOK, staged_files
from tennis_tracker.paths import ROOT

PACKAGES = ("finetune", "tennis_tracker", "ball_in_play_selector")
TRAINER = "finetune.training.train_gridtracknet"


def module_path(name: str) -> Path:
    path = ROOT / (name.replace(".", "/") + ".py")
    return path if path.is_file() else ROOT / name.replace(".", "/") / "__init__.py"


def module_name(path: Path) -> str:
    """The dotted name of a module file under the repository (``tennis_tracker/config.py`` ->
    ``tennis_tracker.config``; a package's ``__init__.py`` is the package)."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def internal_imports(path: Path) -> Iterator[str]:
    """Every finetune.* / tennis_tracker.* module `path` imports, at module level or inside a
    function, relative imports (``from .config import Config``) resolved against `path`."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PACKAGES:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative: the package `path` sits in, `level - 1` levels up
                package = module_name(path.parent / "__init__.py").split(".")
                base = ".".join(package[:len(package) - (node.level - 1)])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if not module or module.split(".")[0] not in PACKAGES:
                continue
            yield module
            for alias in node.names:            # `from finetune import refine` names a module too
                if (ROOT / module.replace(".", "/") / f"{alias.name}.py").is_file():
                    yield f"{module}.{alias.name}"


def import_closure(module: str) -> Set[str]:
    seen: Set[str] = set()
    pending = [module]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(internal_imports(module_path(name)))
    for name in list(seen):                      # importing a submodule imports every package above it
        parts = name.split(".")
        seen.update(".".join(parts[:depth]) for depth in range(1, len(parts)))
    return seen


def test_the_notebook_stages_the_trainers_whole_import_closure():
    needed = {module_path(name).relative_to(ROOT).as_posix() for name in import_closure(TRAINER)}
    staged = staged_files()
    assert set(staged) == needed, f"missing from the notebook: {sorted(needed - set(staged))}; " \
                                  f"staged but not imported: {sorted(set(staged) - needed)}"
    assert all((ROOT / name).is_file() for name in staged)


def test_the_notebook_runs_the_trainer_as_a_module_from_the_staged_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "'-m', 'finetune.training.train_gridtracknet'" in code
    assert "cwd=str(CODE)" in code
    assert "CODE / 'train_gridtracknet.py'" not in code        # no longer copied to code/ on its own


def code_cell_starting(title: str) -> str:
    """The source of the notebook's one code cell whose first line is `title`, parsed first (``!pip``
    lines stand in as ``pass``) so a cell Colab could not run fails here."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    matches = [source for source in cells if source.startswith(title + "\n")]
    assert len(matches) == 1, f"{len(matches)} code cells start with {title!r}"
    ast.parse(matches[0].replace("!pip", "pass  # pip"))
    return matches[0]


def test_the_settings_cell_exposes_the_all_clips_run():
    """SELECTION_RUN names a finished held-out run on Drive; set, the cell passes --all-clips and
    --selection-run to the trainer and leaves the held-out flags and --score-every out, and
    EPOCHS 0 leaves --epochs to the selection run's best epoch."""
    settings = code_cell_starting("#@title Training settings")
    assert 'SELECTION_RUN = ""          #@param {type:"string"}' in settings
    assert "'--all-clips', '--selection-run'" in settings
    assert "if not (all_clips and EPOCHS == 0):" in settings
    for held_out_only in ("'--holdout'", "'--val-clips'", "'--score-every'"):
        assert settings.count(held_out_only) == 1 and settings.index(held_out_only) > settings.index("else:")


def test_the_settings_cell_drops_the_learning_rate_on_a_resume_whose_lr_differs():
    """A resume (checkpoint.pt in the run folder) whose LR is not the rate the run's config.json
    was last training at (the latest lr_schedule entry, or lr for a run from before the schedule)
    passes --lr-drop, and only then: the trainer refuses a raise and a drop to the same rate."""
    settings = code_cell_starting("#@title Training settings")
    assert settings.count("'--lr-drop'") == 1 and settings.index("'--lr-drop'") > settings.index("cmd.append('--resume')")
    assert "(config.get('lr_schedule') or [{'lr': config.get('lr')}])[-1]['lr']" in settings
    assert "if LR != rate:" in settings
    assert "# Adadelta; 0.3 for a gentle fine-tune. On a resume it may be lowered (not raised)" in settings


def test_the_settings_cell_exposes_the_architectures_and_names_the_files_by_them():
    """ARCH picks the network: the cell passes --arch and gives --save the architecture's suffix
    (.pt for heatmap_s4 and memory_s4). START_WEIGHTS is a name in the zip's models/ folder or a
    path under runs/ on Drive (an earlier run's best.pt), and a fine-tune's must be the
    architecture's own file, judged by the record inside a .pt (two architectures share the
    suffix) with memory_s4 also taking its heatmap_s4 parent, said early, which the trainer
    refuses anyway; the staging cell lists the modules the switch imports."""
    settings = code_cell_starting("#@title Training settings")
    assert 'ARCH = "gridtracknet"       #@param ["gridtracknet", "heatmap_s4", "memory_s4"]' in settings
    assert "'--arch', ARCH," in settings and "'--save', str(RUN / ('winner' + SUFFIX))" in settings
    assert "SUFFIX = '.npz' if ARCH == 'gridtracknet' else '.pt'" in settings
    assert "START = DATA / 'models' / START_WEIGHTS" in settings and "START = WORK / START_WEIGHTS" in settings
    assert "'--weights', str(START)," in settings and "'--weights', str(DATA" not in settings
    assert "START.suffix != SUFFIX" not in settings                    # the suffix no longer names the architecture
    assert "start_arch = 'gridtracknet' if START.suffix == '.npz' else torch.load(START, map_location='cpu', " \
           "weights_only=True).get('arch')" in settings
    assert "allowed = {ARCH, 'heatmap_s4'} if ARCH == 'memory_s4' else {ARCH}" in settings
    assert "if start_arch not in allowed:" in settings
    assert {"tennis_tracker/heatmapnet.py", "tennis_tracker/memorynet.py", "tennis_tracker/ball_models.py"} <= set(staged_files())
    text = "".join("".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    assert 'START_WEIGHTS = "<heat run>/best.pt"' in text and "starts the memory fresh" in text


def test_everything_on_drive_lives_under_one_folder():
    """The notebook reads the zips from and writes the runs to MyDrive/tennis_finetune/ alone:
    nothing at the top of Drive, and the missing-zip message names the folder."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "DRIVE_DIR = DRIVE / 'tennis_finetune'" in code
    assert "ZIP = DRIVE_DIR / 'tennis_finetune_colab.zip'" in code
    assert "CODE_ZIP = DRIVE_DIR / 'tennis_finetune_code.zip'" in code
    assert "WORK = DRIVE_DIR / 'runs'" in code
    assert "to the folder MyDrive/tennis_finetune/ in your Google Drive first" in code
    assert code.count("DRIVE / ") == 1                     # DRIVE_DIR itself: no other path hangs off the top of Drive
    assert "tennis_finetune_runs" not in text and "top of your Google Drive" not in text
    assert "MyDrive/tennis_finetune/runs/<TAG>/" in text


def test_every_code_cell_parses():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]).replace("!pip", "pass  # pip"))


def test_the_all_clips_run_keeps_the_selection_runs_architecture_and_the_table_names_the_file():
    """The settings cell always passes --arch, so the all-clips run cannot inherit the selection
    run's architecture from the trainer: it reads the run's config.json and refuses another ARCH
    itself, in a line that says what to set, and the after-run text says so; the run cell's
    best-epoch mark names the file by the architecture's suffix."""
    settings = code_cell_starting("#@title Training settings")
    assert "selected = json.loads((SELECTION / 'config.json').read_text(encoding='utf-8')).get('arch') or 'gridtracknet'" in settings
    assert "if selected != ARCH:" in settings and "trained {selected}: set ARCH to it" in settings
    run = code_cell_starting("# Runs the trainer and streams its log here and to Drive (RUN/colab.log). The trainer prints "
                             "a step line every few")
    assert "f' <- best{SUFFIX}'" in run and "' <- best.npz'" not in run
    text = "".join("".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    assert "keep `ARCH` at the selection run's" in text and "inherits the architecture" not in text
