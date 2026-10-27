"""Make the repository root and ``system/`` importable so the workspace suites see ``finetune`` and
the tracker's packages as the scripts do, and the fixtures they share. Tests are plain pytest
functions with fixtures and ``parametrize``; ``unittest.TestCase`` is not used.

This folder is not a package (no ``__init__.py``): ``system/tests`` is the package ``tests``, the
tracker's own suites exactly as on ``main``, and both folders are collected in one run (the root
``pytest.ini``). Here pytest imports each module by its file name with this folder on
``sys.path``, so ``tests._helpers`` is the tracker's helpers and ``_workspace_helpers`` this
folder's. A fixture of ``system/tests/conftest.py`` reaches only the suites under it.
"""
import sys
from pathlib import Path

import pytest

# The bootstrap: pytest.ini's pythonpath does the same, but a test run started another way
# (an IDE on one file, say) still needs the repository on sys.path before the project imports;
# importing finetune then puts system/ there too.
_REPOSITORY = Path(__file__).resolve().parent.parent
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from finetune import REPOSITORY, SYSTEM  # noqa: E402  (needs the bootstrap above)
from tennis_tracker.paths import ROOT  # noqa: E402

assert REPOSITORY == _REPOSITORY, "tests/ must sit directly under the repository finetune/ is in"
assert ROOT == SYSTEM, "the tracker these suites import must be the one under system/"


@pytest.fixture
def workspace_dir(tmp_path, monkeypatch):
    """Point the fine-tuning workspace (``finetune.data.workspace``) at a fresh folder under tmp_path,
    with empty ``labels/`` and ``videos/`` and the tracker's weights redirected to ``models/tracker.pt``
    under it. The finetune modules read every path from ``workspace.current()`` when called, so
    replacing that one function moves all of them; no real label, video or model is touched.
    Returns the folder."""
    from finetune.data import workspace

    root = tmp_path / "workspace"
    (root / "labels").mkdir(parents=True)
    (root / "videos").mkdir()
    relocated = workspace.Workspace(root, repo_weights=root / "models" / "tracker.pt")
    monkeypatch.setattr(workspace, "current", lambda: relocated)
    return root
