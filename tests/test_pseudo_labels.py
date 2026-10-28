"""finetune/training/pseudo_labels.py: the measured filter on a pipeline payload, the label file,
the commands, and pseudo-labelled clips through the frame cache into units with ``valid``."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from finetune.data import workspace as ws
from finetune.training import pseudo_labels
from finetune.training.frame_cache import prepare_clip, prepare_pseudo
from finetune.training.pseudo_labels import (BALL, IGNORE, NONE, QUIET_STEPS, frame_states, pseudo_csv, read_pseudo_labels,
                                             write_pseudo_labels)
from tests._helpers import write_clip


def payload(frames, tracks, total=200, width=1920):
    return {"video": {"width": width, "height": 1080, "fps": 60.0, "total_frames": total}, "tracks": tracks, "frames": frames}


def present(frame, x=500.0, y=300.0, conf=0.9, kind="measurement", track=1, drift=0.0):
    return {"frame": frame, "present": True, "x": x + drift, "y": y, "position_kind": kind, "track_id": track,
            "measurement": {"x": x, "y": y, "conf": conf}}


def track(track_id=1, frames=range(40, 60), timeline=True, observations=20):
    return {"track_id": track_id, "in_timeline": timeline, "num_obs": observations,
            "observations": [{"frame": frame} for frame in frames]}


def test_the_filter_keeps_sure_balls_and_sure_absences_and_ignores_the_rest():
    rows = [present(40), present(42, kind="recovered"), present(44, conf=0.6), present(46, drift=4.0),
            present(48, track=2), present(50, track=3), present(52, drift=2.0)]
    tracks = [track(), track(2, timeline=False), track(3, observations=5)]
    states = frame_states(payload(rows, tracks), 2)
    assert len(states) == 100 and min(states) == 0 and max(states) == 198
    assert states[40] == (BALL, (500.0, 300.0)) and states[52] == (BALL, (502.0, 300.0))
    assert [states[f][0] for f in (42, 44, 46, 48, 50)] == [IGNORE] * 5     # recovered, unsure, drifted, off-timeline, short
    reach = QUIET_STEPS * 2
    assert states[40 - reach - 2] == (NONE, None) and states[40 - reach][0] == IGNORE   # a detection within reach
    assert states[58 + reach + 2] == (NONE, None) and states[58 + reach][0] == IGNORE
    scaled = frame_states(payload([present(40, drift=2.0)], [track()], width=960), 2)
    assert scaled[40][0] == IGNORE                                            # 2 px at 960 wide is 4 px at 1080p


def test_the_label_file_round_trips_and_refuses_what_it_does_not_know(tmp_path):
    path = tmp_path / "c_pseudo.csv"
    write_pseudo_labels(path, {4: (NONE, None), 0: (BALL, (10.0, 2000.0)), 2: (IGNORE, None)})
    assert path.read_text(encoding="utf-8").splitlines()[0] == "frame,ball_x,ball_y,state"
    targets, ignored = read_pseudo_labels(path, 1920, 1080)
    assert targets == {0: [10.0, 1079.0], 2: None, 4: None} and ignored == {2}      # clamped onto the image
    with pytest.raises(ValueError, match="unknown pseudo-label state"):
        write_pseudo_labels(path, {0: ("maybe", None)})
    path.write_text("frame,ball_x,ball_y,state\n0,,,ball\n", encoding="utf-8")
    with pytest.raises(ValueError, match="without coordinates"):
        read_pseudo_labels(path, 1920, 1080)


def pseudo_clip(root: Path, clip: str = "court5_N", count: int = 30) -> Path:
    """A 30 FPS clip under `root`/videos with a pseudo-label file: frames 0-9 balls, 10-14 ignored,
    15-29 absent."""
    (root / "videos").mkdir(parents=True, exist_ok=True)
    video = write_clip(root / "videos" / f"{clip}.avi", count, (64, 36))
    rows = {frame: (BALL, (float(frame), 5.0)) if frame < 10 else (IGNORE, None) if frame < 15 else (NONE, None)
            for frame in range(count)}
    write_pseudo_labels(pseudo_csv(root, clip), rows)
    return video


def test_a_pseudo_clip_caches_into_units_with_valid_and_none_all_ignored(tmp_path):
    root = tmp_path / "pseudo"
    video = pseudo_clip(root)
    record = prepare_clip(("court5_N", video, pseudo_csv(root, "court5_N"), "pseudo"), tmp_path / "cache" / "pseudo",
                          False, "pseudo/")
    units = record["samples"]
    assert units and all(unit["frames"][0].startswith("pseudo/frames/court5_N/") for unit in units)
    assert all(any(unit["valid"]) for unit in units) and all(len(unit["valid"]) == 5 for unit in units)
    middle = next(unit for unit in units if unit["indices"][0] == 8)
    assert middle["valid"] == [True, True, False, False, False] and middle["points"][2] is None
    assert not any(unit["indices"][0] in (10, 11) for unit in units)        # all five frames ignored: dropped
    assert (tmp_path / "cache" / "pseudo" / "frames" / "court5_N").is_dir()


def test_prepare_pseudo_tags_the_source_and_refuses_an_empty_folder(tmp_path, capsys):
    root = tmp_path / "pseudo"
    pseudo_clip(root)
    write_clip(root / "videos" / "unlabelled.avi", 10, (64, 36))
    units = prepare_pseudo(root, tmp_path / "cache", False, 1)
    assert {unit["source"] for unit in units} == {"pseudo"} and {unit["clip"] for unit in units} == {"court5_N"}
    assert "1 pseudo-labelled clips ready" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no video"):
        prepare_pseudo(tmp_path / "empty", tmp_path / "cache", False, 1)


def test_the_commands_add_label_from_a_saved_run_and_report(tmp_path, capsys):
    workspace = ws.Workspace(tmp_path / "ws")
    source = write_clip(tmp_path / "Court 5 N.avi", 12, (64, 36))
    slow = write_clip(tmp_path / "slow.avi", 12, (64, 36), fps=45.0)
    with patch.object(ws, "current", return_value=workspace):
        assert pseudo_labels.main(["add", str(source), str(slow)]) == 1          # the 45 FPS one is refused
        out = capsys.readouterr().out
        assert "[add ] Court_5_N.avi" in out and "neither 30 nor 60" in out and "ffmpeg" in out
        labels = workspace.root / "pseudo" / "labels"
        labels.mkdir(parents=True)
        frames = [present(f, x=30.0, y=10.0) for f in range(12)]
        (labels / "Court_5_N_pipeline.json").write_text(json.dumps(
            payload(frames, [track(frames=range(12))], total=12, width=64)), encoding="utf-8")
        with patch("tennis_tracker.headless.run_headless") as pipeline:
            assert pseudo_labels.main(["label", "--reuse"]) == 0
        pipeline.assert_not_called()
        assert "[pseudo] Court_5_N_pseudo.csv: ball 12 (100%)" in capsys.readouterr().out
        assert pseudo_labels.main(["status"]) == 0
        assert "12 label frames: ball 12, none 0, ignore 0" in capsys.readouterr().out

