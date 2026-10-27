"""The layout the train branch keeps: the tracker in ``system/`` (a copy of ``main``, see AGENTS.md
for the invariant and the port recipe), the workspace beside it. What can be checked without
git history: the tracker's packages are the ones under ``system/``, nothing at the root shadows
them, and nothing under ``system/`` reaches back into the workspace, which ``main`` does not have."""
import ast
import sys
from pathlib import Path

import evaluation
import tennis_tracker
from finetune import REPOSITORY, SYSTEM
from tennis_tracker import paths


def test_the_tracker_packages_are_the_ones_under_system():
    assert str(SYSTEM) in sys.path
    assert paths.ROOT == SYSTEM and paths.MODELS == SYSTEM / "models"
    for package in (tennis_tracker, evaluation):
        assert package.__file__ is not None and SYSTEM in Path(package.__file__).parents


def test_nothing_at_the_root_shadows_a_system_package_or_script():
    """A root ``evaluation/`` or ``tennis_tracker/`` would be imported instead of the system's wherever
    the root comes first on sys.path (pytest's pythonpath, a root script's own folder). ``tests`` is the one name both sides use: the root's folder is
    not a package, so ``tests`` stays the system's."""
    shared = {entry.name for entry in SYSTEM.iterdir()
              if (entry.is_dir() and (entry / "__init__.py").is_file()) or entry.suffix == ".py"}
    assert shared >= {"tennis_tracker", "ball_in_play_selector", "evaluation", "tests", "clean_tracker.py"}
    assert not {name for name in shared - {"tests"} if (REPOSITORY / name).exists()}
    assert not (REPOSITORY / "tests" / "__init__.py").exists()


def test_nothing_under_system_imports_the_workspace():
    """``system/`` is ``main``'s tree, and ``main`` has no ``finetune`` package."""
    offenders = []
    for path in SYSTEM.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            if any(name.split(".")[0] == "finetune" for name in names):
                offenders.append(path.relative_to(SYSTEM).as_posix())
    assert not offenders
