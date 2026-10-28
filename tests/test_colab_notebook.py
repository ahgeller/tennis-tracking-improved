"""The Colab notebooks stage every repository module the trainer imports, and the older notebook's
settings.

``ft.py colab-zip`` derives the code it packs from the code (``finetune.colab_zip.code_files``:
the import closure of the trainer and the pseudo-label tool, function-level imports included)
and writes the list to ``code/code_manifest.json``, which both notebooks check before running.
These tests keep that derivation honest: the walker is checked on a small tree of its own, and
the staged files alone, copied apart from the repository as Colab has them, must import the
trainer and pass its self-test, so a module the walker missed fails here instead of with
ModuleNotFoundError on Colab. The settings cell of ``GridTrackNet_FineTune_Colab.ipynb`` is
checked for the trainer flags it must pass; ``test_optimization_notebook.py`` runs the new one."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from finetune import REPOSITORY
from finetune.colab_zip import NOTEBOOK, OPTIMIZATION_NOTEBOOK, code_files, import_closure, staged_files


def write(root: Path, name: str, text: str = "") -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")


def test_the_walker_follows_every_kind_of_import_and_stops_at_the_repository(tmp_path):
    write(tmp_path, "finetune/__init__.py")
    write(tmp_path, "finetune/training/__init__.py")
    write(tmp_path, "finetune/training/trainer.py",
          "import os\nfrom finetune.training import helper\nfrom . import sibling\n"
          "def later():\n    from tennis_tracker.engines.box import Box\n")
    write(tmp_path, "finetune/training/helper.py", "import numpy\n")
    write(tmp_path, "finetune/training/sibling.py", "from ..data.io import read\n")
    write(tmp_path, "finetune/data/__init__.py")
    write(tmp_path, "finetune/data/io.py")
    write(tmp_path, "finetune/data/unused.py")
    write(tmp_path, "system/tennis_tracker/__init__.py")
    write(tmp_path, "system/tennis_tracker/engines/__init__.py", "from .box import Box\n")
    write(tmp_path, "system/tennis_tracker/engines/box.py", "from tennis_tracker import paths\n")
    write(tmp_path, "system/tennis_tracker/paths.py")
    assert code_files(["finetune.training.trainer"], tmp_path) == [
        "finetune/__init__.py", "finetune/data/__init__.py", "finetune/data/io.py", "finetune/training/__init__.py",
        "finetune/training/helper.py", "finetune/training/sibling.py", "finetune/training/trainer.py",
        "system/tennis_tracker/__init__.py", "system/tennis_tracker/engines/__init__.py",
        "system/tennis_tracker/engines/box.py", "system/tennis_tracker/paths.py"]
    write(tmp_path, "finetune/training/helper.py", "from finetune.training import gone_module\nimport finetune.nowhere\n")
    with pytest.raises(ValueError, match="no module file for finetune.nowhere"):
        import_closure(["finetune.training.trainer"], tmp_path)


def test_the_staged_code_alone_imports_and_self_tests_the_trainer(tmp_path):
    """The Colab situation: only the staged files, laid out as the zip lays them, no repository on
    the path. The trainer's self-test runs and the pseudo-label tool and every architecture's
    module import; the roots and the new training modules are in the list."""
    staged = staged_files()
    assert {"finetune/training/train_gridtracknet.py", "finetune/training/distill.py", "finetune/training/copy_paste.py",
            "finetune/training/pseudo_labels.py", "system/tennis_tracker/ball_models.py"} <= set(staged)
    for name in staged:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / name, tmp_path / name)
    script = ("import sys; sys.path.insert(0, '.'); import finetune\n"
              "import finetune.training.pseudo_labels\n"
              "from tennis_tracker.ball_models import ARCHITECTURES, model_class\n"
              "for arch in ARCHITECTURES: model_class(arch)\n"
              "assert all(not m.__file__.startswith(sys.argv[1]) for m in list(sys.modules.values()) "
              "if getattr(m, '__file__', None) and m.__name__.split('.')[0] in ('finetune', 'tennis_tracker'))\n"
              "from finetune.training.train_gridtracknet import main\n"
              "raise SystemExit(main(['--self-test']))\n")
    result = subprocess.run([sys.executable, "-I", "-c", script, str(REPOSITORY)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train_gridtracknet self-test passed" in result.stdout


@pytest.mark.parametrize("notebook", [NOTEBOOK, OPTIMIZATION_NOTEBOOK])
def test_the_notebooks_check_the_code_against_the_zips_manifest_not_a_list_of_their_own(notebook):
    cells = ["".join(cell["source"]) for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
             if cell["cell_type"] == "code"]
    staging = [source for source in cells if "REQUIRED = " in source]
    assert len(staging) == 1
    assert "REQUIRED = json.loads((CODE / 'code_manifest.json').read_text(encoding='utf-8'))" in staging[0]
    assert "REQUIRED = [" not in staging[0]


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
    (.pt for heatmap_s4). START_WEIGHTS is a name in the zip's models/ folder or a path under
    runs/ on Drive (an earlier run's best.pt), and a fine-tune's must be the architecture's own
    file, judged by the record inside a .pt, said early, which the trainer refuses anyway; the
    staging cell lists the modules the switch imports."""
    settings = code_cell_starting("#@title Training settings")
    assert 'ARCH = "heatmap_s4"         #@param ["gridtracknet", "heatmap_s4"]' in settings
    assert "'--arch', ARCH," in settings and "'--save', str(RUN / ('winner' + SUFFIX))" in settings
    assert "SUFFIX = '.npz' if ARCH == 'gridtracknet' else '.pt'" in settings
    assert "START = DATA / 'models' / START_WEIGHTS" in settings and "START = WORK / START_WEIGHTS" in settings
    assert "'--weights', str(START)," in settings and "'--weights', str(DATA" not in settings
    assert "START.suffix != SUFFIX" not in settings                    # the suffix no longer names the architecture
    assert "start_arch = 'gridtracknet' if START.suffix == '.npz' else torch.load(START, map_location='cpu', " \
           "weights_only=True).get('arch')" in settings
    assert "allowed = {ARCH}" in settings
    assert "if start_arch not in allowed:" in settings
    assert {"system/tennis_tracker/heatmapnet.py", "system/tennis_tracker/ball_models.py"} <= set(staged_files())
    text = "".join("".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    for gone in ("memory_s4", "heatmap_s2", "memorynet", "heatmapnet_s2"):      # removed 2026-10-26, see the README
        assert gone not in text, gone


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
