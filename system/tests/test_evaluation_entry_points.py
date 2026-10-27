"""The evaluation package's two modules run from the repository root as ``python -m``: the
documented commands keep working, and the gate scores with the tracking scorer."""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from tennis_tracker.paths import ROOT


def test_the_gate_scores_with_the_tracking_scorer():
    from evaluation import parity, validate

    assert parity.validate is validate.validate


def test_every_module_main_takes_argv_and_returns_a_status():
    """main(argv=None) -> int everywhere, invoked as raise SystemExit(main()): a test can drive a
    module without patching sys.argv, and the exit status is the return value, never implicit."""
    from evaluation import parity, validate
    for module in (parity, validate):
        signature = inspect.signature(module.main)
        assert list(signature.parameters) == ["argv"] and signature.return_annotation in (int, "int"), module.__name__
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.rstrip().endswith("raise SystemExit(main())"), module.__name__
    with pytest.raises(SystemExit) as stop:
        validate.main(["--nope"])
    assert stop.value.code == 2


@pytest.mark.parametrize("module", ["evaluation.validate", "evaluation.parity"])
def test_the_documented_commands_answer_help(module):
    done = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout
