"""The evaluation package and the root scripts that front it: the documented commands keep
working, and the package's two modules are the ones they name."""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from tennis_tracker.paths import ROOT


def test_root_scripts_front_the_package_modules():
    import check_parity
    import validate_tracking
    from evaluation import parity, validate

    assert validate_tracking.main is validate.main
    assert check_parity.main is parity.main
    assert parity.validate is validate.validate              # check_parity gates on the tracking scorer


def test_every_script_main_takes_argv_and_returns_a_status():
    """main(argv=None) -> int everywhere, invoked as raise SystemExit(main()): a test can drive a
    script without patching sys.argv, and the exit status is the return value, never implicit."""
    from evaluation import parity, validate
    for module in (parity, validate):
        signature = inspect.signature(module.main)
        assert list(signature.parameters) == ["argv"] and signature.return_annotation in (int, "int"), module.__name__
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.rstrip().endswith("raise SystemExit(main())"), module.__name__
    with pytest.raises(SystemExit) as stop:
        validate.main(["--nope"])
    assert stop.value.code == 2


def test_the_documented_commands_still_answer_help():
    for script in ("validate_tracking.py", "check_parity.py"):
        done = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "usage:" in done.stdout
