"""finetune/Optimization_Train_Colab.ipynb: its first three cells run here against a simulated
Drive (a folder holding a zip ``ft.py colab-zip`` built from a tiny workspace, the runtime's disk
another folder, ``google.colab`` a stub), and the command the settings cell builds is one the
trainer's own parser accepts with the flags the settings ask for."""
import ast
import json
import sys
import types
from pathlib import Path

import pytest

from finetune import ft
from finetune.colab_zip import OPTIMIZATION_NOTEBOOK
from finetune.data import workspace
from finetune.training.train_gridtracknet import build_parser
from tennis_tracker.ball_models import ARCHITECTURES


def code_cells() -> list:
    notebook = json.loads(OPTIMIZATION_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def runnable(source: str) -> str:
    """A cell as plain Python: the ``!pip`` line (Colab shell) left out."""
    return "\n".join("pass  # " + line if line.lstrip().startswith("!") else line for line in source.splitlines())


def test_every_code_cell_parses_and_the_settings_name_every_architecture_ball_models_has():
    for source in code_cells():
        ast.parse(runnable(source))
    settings = next(source for source in code_cells() if source.startswith("#@title Training settings\n"))
    options = ast.literal_eval(settings.split("#@param", 1)[1].split("\n", 1)[0].strip())
    assert set(ARCHITECTURES) <= set(options) and options[0] == "heatmap_s4"


def fill(root: Path) -> None:
    (root / "videos" / "rally.mp4").write_bytes(b"not decoded here")
    (root / "labels" / "rally_ball.csv").write_text("frame,ball_x,ball_y\n", encoding="utf-8")
    layout = workspace.current()
    layout.repo_weights.parent.mkdir(parents=True, exist_ok=True)
    layout.repo_weights.write_bytes(b"weights")
    layout.manifest.write_text("clip,source\nrally,custom\n", encoding="utf-8")
    layout.review_status.write_text(json.dumps({"verified_clips": ["rally"]}), encoding="utf-8")
    (root / "pseudo" / "videos").mkdir(parents=True)
    (root / "pseudo" / "labels").mkdir()
    (root / "pseudo" / "videos" / "court5_N.mp4").write_bytes(b"video")
    (root / "pseudo" / "labels" / "court5_N_pseudo.csv").write_text("frame,ball_x,ball_y,state\n", encoding="utf-8")


def simulated(source: str, drive: Path, local: Path) -> str:
    return (source.replace("Path('/content/drive/MyDrive')", repr(drive).replace("WindowsPath", "Path").replace("PosixPath", "Path"))
            .replace("Path('/content')", repr(local).replace("WindowsPath", "Path").replace("PosixPath", "Path")))


def test_the_first_cells_unpack_the_zip_from_drive_check_the_code_and_build_the_trainers_command(
        workspace_dir, tmp_path, monkeypatch, capsys):
    fill(workspace_dir)
    drive, local = tmp_path / "MyDrive", tmp_path / "content"
    (drive / "tennis_finetune").mkdir(parents=True)
    local.mkdir()
    assert ft.run_cli(["colab-zip", "--out", str(drive / "tennis_finetune" / "tennis_finetune_colab.zip")]) == 0
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    mount, staging, settings = code_cells()[:3]
    namespace: dict = {}
    exec(compile(simulated(runnable(mount), drive, local), "mount", "exec"), namespace)
    data = local / "tennis_finetune"
    assert (data / "videos" / "rally.mp4").is_file() and (data / ".unpacked_from").is_file()
    assert "1 labelled-workspace videos, 1 pseudo-labelled clips" in capsys.readouterr().out
    exec(compile(runnable(staging), "staging", "exec"), namespace)
    assert namespace["REQUIRED"] and namespace["CODE"] == data / "code"
    exec(compile(runnable(settings), "settings", "exec"), namespace)
    cmd = namespace["cmd"]
    assert cmd[1:4] == ["-u", "-m", "finetune.training.train_gridtracknet"]
    args = build_parser().parse_args(cmd[4:])
    assert args.arch == "heatmap_s4" and args.from_scratch and args.distill and args.distill_loss == "qfl"
    assert args.teacher == data / "models" / "ball_detector.pt" and args.teacher_maps == "online"
    assert args.augment == ["basic", "translate", "copypaste"] and args.holdout == "mixed_v1"
    assert args.pseudo == data / "pseudo" and args.pseudo_share == pytest.approx(0.3)
    assert args.save.name == "winner.pt" and args.run_dir == drive / "tennis_finetune" / "runs" / "opt1"
    assert not args.resume
    # Run again: the zip is unchanged, so nothing is unpacked twice; a checkpoint makes it a resume.
    (args.run_dir / "checkpoint.pt").write_bytes(b"")
    (args.run_dir / "config.json").write_text(json.dumps({"lr": 1.0}), encoding="utf-8")
    exec(compile(simulated(runnable(mount), drive, local), "mount", "exec"), namespace)
    assert "unpacking" not in capsys.readouterr().out
    exec(compile(runnable(settings), "settings", "exec"), namespace)
    assert "--resume" in namespace["cmd"] and "--lr-drop" not in namespace["cmd"]


def test_an_architecture_the_staged_code_lacks_is_refused_in_one_line(workspace_dir, tmp_path, monkeypatch):
    fill(workspace_dir)
    drive, local = tmp_path / "MyDrive", tmp_path / "content"
    (drive / "tennis_finetune").mkdir(parents=True)
    local.mkdir()
    assert ft.run_cli(["colab-zip", "--out", str(drive / "tennis_finetune" / "tennis_finetune_colab.zip")]) == 0
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    mount, staging, settings = code_cells()[:3]
    namespace: dict = {}
    exec(compile(simulated(runnable(mount), drive, local), "mount", "exec"), namespace)
    exec(compile(runnable(staging), "staging", "exec"), namespace)
    with pytest.raises(SystemExit, match="not in the staged ball_models"):
        exec(compile(runnable(settings).replace('ARCH = "heatmap_s4"', 'ARCH = "no_such_net"'), "settings", "exec"), namespace)
