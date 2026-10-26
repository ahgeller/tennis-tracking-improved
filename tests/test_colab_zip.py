"""finetune/colab_zip.py, the ``ft.py colab-zip`` command, on the ``workspace_dir`` fixture's
workspace: which entries the zip holds (every video, the whole labels folder, the records, the
weights and the notebook's REQUIRED code), that they are stored uncompressed and never decoded,
and that a missing clips.csv or review_status.json is an error."""
import json
import re
import zipfile

from finetune import ft
from finetune.colab_zip import DEFAULT_OUT, TOP, staged_files
from finetune.data import workspace


def fill_workspace(root, with_records=True):
    """A tiny workspace: two fake videos (bytes, not decodable), a .bak that is not a video, a
    label file with its sidecars and a backup, the records and the tracker's weights."""
    (root / "videos" / "rally.mp4").write_bytes(b"not really mp4")
    (root / "videos" / "serve.avi").write_bytes(b"not really avi")
    (root / "videos" / "rally.mp4.bak").write_bytes(b"old")
    for name in ("rally_ball.csv", "rally_ball.review.json", "rally_ball.suspect.json", "serve_ball.csv.draft",
                 "rally_bounces.csv", "rally_ball.csv.before_halfpixel_fix"):
        (root / "labels" / name).write_text("frame_000,1,2\n", encoding="utf-8")
    (root / "labels" / "nested").mkdir()                          # a folder in labels/: not a file, not packed
    layout = workspace.current()
    layout.repo_weights.parent.mkdir(parents=True, exist_ok=True)
    layout.repo_weights.write_bytes(b"weights")
    if with_records:
        layout.manifest.write_text("clip,source\nrally,custom\n", encoding="utf-8")
        layout.review_status.write_text(json.dumps({"verified_clips": ["rally"]}), encoding="utf-8")
        layout.holdouts.write_text("{}", encoding="utf-8")


def test_the_zip_holds_every_section_the_notebook_unpacks(workspace_dir, tmp_path, capsys):
    fill_workspace(workspace_dir)
    out = tmp_path / "deep" / "colab.zip"
    assert ft.run_cli(["colab-zip", "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert archive.read(f"{TOP}/videos/rally.mp4") == b"not really mp4"
        assert archive.read(f"{TOP}/models/gridtracknet_weights_torch.npz") == b"weights"
    assert {n for n in names if n.startswith(f"{TOP}/videos/")} == {f"{TOP}/videos/rally.mp4", f"{TOP}/videos/serve.avi"}
    assert {n for n in names if n.startswith(f"{TOP}/labels/")} == {
        f"{TOP}/labels/{name}" for name in ("rally_ball.csv", "rally_ball.review.json", "rally_ball.suspect.json",
                                            "serve_ball.csv.draft", "rally_bounces.csv", "rally_ball.csv.before_halfpixel_fix")}
    assert {f"{TOP}/clips.csv", f"{TOP}/review_status.json", f"{TOP}/holdouts.json"} <= names
    assert f"{TOP}/exclude.txt" not in names                      # optional and absent: skipped, said so
    assert {n for n in names if n.startswith(f"{TOP}/code/")} == {f"{TOP}/code/{name}" for name in staged_files()}
    assert len(names) == 2 + 6 + 3 + 1 + len(staged_files())
    assert not list(out.parent.glob("*.tmp"))
    out_text = capsys.readouterr().out
    assert "[skip] no exclude.txt" in out_text
    assert re.search(r"videos\s+2 files", out_text) and re.search(r"labels\s+6 files", out_text) and re.search(r"records\s+3 files", out_text)
    assert "MB stored" in out_text and "tennis_finetune_colab.zip" in out_text


def test_a_missing_record_the_trainer_needs_is_an_error(workspace_dir, tmp_path, capsys):
    fill_workspace(workspace_dir, with_records=False)
    out = tmp_path / "colab.zip"
    assert ft.run_cli(["colab-zip", "--out", str(out)]) == 1
    failure = capsys.readouterr().out
    assert "[fail]" in failure and "clips.csv is missing" in failure
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))
    workspace.current().manifest.write_text("clip,source\n", encoding="utf-8")
    assert ft.run_cli(["colab-zip", "--out", str(out)]) == 1
    assert "review_status.json is missing" in capsys.readouterr().out
    assert not out.exists()


def test_the_default_output_sits_under_the_repository_output_folder():
    assert DEFAULT_OUT.name == "tennis_finetune_colab.zip" and DEFAULT_OUT.parent.name == "output"


def test_code_only_holds_the_records_and_the_code_and_nothing_heavy(workspace_dir, tmp_path, capsys):
    fill_workspace(workspace_dir)
    out = tmp_path / "code.zip"
    assert ft.run_cli(["colab-zip", "--code-only", "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
    assert {f"{TOP}/clips.csv", f"{TOP}/review_status.json", f"{TOP}/holdouts.json"} <= names
    assert names >= {f"{TOP}/code/{name}" for name in staged_files()}
    assert not any(name.startswith((f"{TOP}/videos/", f"{TOP}/labels/", f"{TOP}/models/")) for name in names)
    assert "tennis_finetune_code.zip" in capsys.readouterr().out
