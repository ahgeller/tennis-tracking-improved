"""The evaluation package and the root scripts that front the workspace scorers: the documented
commands keep working, `validate` and `parity` run as ``python -m``, and the gate scores with
the tracking scorer."""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from tennis_tracker.paths import ROOT


def test_root_scripts_front_the_package_modules():
    import evaluate_archive
    import score_court_events
    import track_breaks
    from evaluation import archive, court_events_score, parity, validate
    from evaluation import track_breaks as track_breaks_module

    assert track_breaks.main is track_breaks_module.main
    assert evaluate_archive.main is archive.main
    assert score_court_events.main is court_events_score.main
    assert parity.validate is validate.validate              # the parity gate scores with the tracking scorer


def test_every_module_main_takes_argv_and_returns_a_status():
    """main(argv=None) -> int everywhere, invoked as raise SystemExit(main()): a test can drive a
    module without patching sys.argv, and the exit status is the return value, never implicit."""
    from evaluation import archive, court_events_score, parity, track_breaks, validate
    from finetune.data import import_data, realign
    from finetune.training import pretrack, train_gridtracknet
    for module in (archive, court_events_score, parity, track_breaks, validate, import_data, realign, pretrack,
                   train_gridtracknet):
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


def test_the_documented_commands_still_answer_help():
    for script in ("evaluate_archive.py", "score_court_events.py"):
        done = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "usage:" in done.stdout
    done = subprocess.run([sys.executable, "-m", "evaluation.archive", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and "--archive" in done.stdout


@pytest.mark.parametrize("module", ["evaluation.validate", "evaluation.parity"])
def test_the_documented_commands_answer_help(module):
    done = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def test_archive_scorer_uses_the_workspace_and_the_shared_model_paths():
    from evaluation import archive
    from finetune.data import workspace
    from tennis_tracker import paths

    assert archive.default_archive() == workspace.current().root or archive.ARCHIVE_VARIABLE in __import__("os").environ
    assert archive.MODELS["gridtracknet"] == paths.BALL_WEIGHTS
