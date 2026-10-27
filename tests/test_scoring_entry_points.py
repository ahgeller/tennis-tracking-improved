"""The workspace scorers and the root scripts that front them: the documented commands keep
working from the repository root, every script's main takes argv and returns a status, and the
scorers and the workspace use the tracker's model paths."""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from finetune import REPOSITORY
from tennis_tracker import paths


def test_root_scripts_front_the_scoring_modules():
    import evaluate_archive
    import score_court_events
    import track_breaks
    from finetune.scoring import archive, court_events_score
    from finetune.scoring import track_breaks as track_breaks_module

    assert track_breaks.main is track_breaks_module.main
    assert evaluate_archive.main is archive.main
    assert score_court_events.main is court_events_score.main


def test_every_workspace_main_takes_argv_and_returns_a_status():
    """main(argv=None) -> int everywhere, invoked as raise SystemExit(main()): a test can drive a
    module without patching sys.argv, and the exit status is the return value, never implicit."""
    from finetune.data import import_data, realign
    from finetune.scoring import archive, court_events_score, track_breaks
    from finetune.training import pretrack, train_gridtracknet
    for module in (archive, court_events_score, track_breaks, import_data, realign, pretrack, train_gridtracknet):
        signature = inspect.signature(module.main)
        assert list(signature.parameters) == ["argv"] and signature.return_annotation in (int, "int"), module.__name__
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.rstrip().endswith("raise SystemExit(main())"), module.__name__
    with pytest.raises(SystemExit) as stop:
        archive.main(["--mode", "nope"])
    assert stop.value.code == 2
    with pytest.raises(SystemExit):
        train_gridtracknet.main(["--epochs", "x"])
    assert train_gridtracknet.main(["--self-test"]) == 0


@pytest.mark.parametrize("cwd", [REPOSITORY, REPOSITORY / "finetune"])
def test_the_documented_commands_answer_help_from_any_folder(cwd):
    for script in ("evaluate_archive.py", "score_court_events.py", "track_breaks.py"):
        done = subprocess.run([sys.executable, str(REPOSITORY / script), "--help"], cwd=cwd, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "usage:" in done.stdout
    done = subprocess.run([sys.executable, "-m", "finetune.scoring.archive", "--help"], cwd=REPOSITORY,
                          capture_output=True, text=True)
    assert done.returncode == 0 and "--archive" in done.stdout


def test_the_scorers_and_the_workspace_use_the_shared_model_paths():
    from finetune.data import workspace
    from finetune.scoring import archive

    assert archive.default_archive() == workspace.current().root or archive.ARCHIVE_VARIABLE in os.environ
    assert archive.MODELS["gridtracknet"] == paths.BALL_WEIGHTS
    assert workspace.current().repo_weights == paths.BALL_WEIGHTS
