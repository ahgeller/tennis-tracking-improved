"""finetune/training/pretrack.py's draft writing: one row per label frame through the label
format's own writer (``labels.write_draft``), on a temporary folder. The detector itself is
exercised in test_label_workflow (raw_rows) and the script's workspace handling in test_workspace."""
from pathlib import Path

from finetune.data.labels import read_label_rows, read_labels
from types import SimpleNamespace
from unittest.mock import patch

from finetune.training import pretrack
from finetune.training.pretrack import RAW_DRAFT_CONF, write_draft
from tennis_tracker import headless
from tennis_tracker.ball_weights import HEATMAP_S4, STREAM_S4, UNKNOWN_CONF


def test_write_draft_covers_every_label_frame_in_the_label_format(tmp_path, capsys):
    """What pretrack.py wrote itself before: the guess where there is one, "no ball" otherwise."""
    draft = tmp_path / "clip_ball.csv.draft"
    guesses = {0: (10.123, 20.5, "det", 0.9), 1: (5.0, 5.0, "det", 0.8), 4: (-3.0, 2000.0, "interp", 0.25)}
    assert write_draft(guesses, draft, 2, 640, 360, 7) == 2                   # frame 1 is not a label frame at stride 2
    assert draft.read_text(encoding="utf-8").splitlines() == [
        "frame,ball_x,ball_y,source,conf", "frame_000,10.12,20.50,det,0.90", "frame_002,639,0,none,0.00",
        "frame_004,0.00,359.00,interp,0.25", "frame_006,639,0,none,0.00"]
    assert read_labels(draft, 640, 360) == {0: (10.12, 20.5), 2: None, 4: (0.0, 359.0), 6: None}
    assert [row.source for row in read_label_rows(draft)] == ["det", "none", "interp", "none"]
    assert capsys.readouterr().out == ""


def test_write_draft_says_when_a_guess_sits_in_the_no_ball_corner(tmp_path, capsys):
    draft = tmp_path / "clip_ball.csv.draft"
    assert write_draft({0: (639.0, 1.0, "det", 0.7)}, draft, 1, 640, 360, 1) == 1
    assert "1 guess(es) sit in the top-right corner" in capsys.readouterr().out
    assert read_labels(draft, 640, 360) == {0: None}                           # what every reader makes of it
    assert isinstance(draft, Path)


def test_pipeline_drafts_use_the_threshold_the_tracker_runs_the_weights_at(tmp_path):
    """The pipeline source drafts what the tracker would show, so without --conf its threshold is
    the one the tracker runs those weights at (it was a copy of the default once, left at 0.50 when
    the default became 0.40); a given threshold is used as given."""
    seen = []

    def run_headless(video, *, model_path, conf, **_):
        seen.append(conf)
        return SimpleNamespace(frame_rows=lambda: [])

    with patch.object(headless, "run_headless", run_headless):
        for weights in (STREAM_S4.path, HEATMAP_S4.path, tmp_path / "other.pt"):
            pretrack.pipeline_rows(tmp_path / "clip.mp4", weights, None, "0", tmp_path / "t.json")
        pretrack.pipeline_rows(tmp_path / "clip.mp4", STREAM_S4.path, 0.3, "0", tmp_path / "t.json")
    assert seen == [STREAM_S4.conf, HEATMAP_S4.conf, UNKNOWN_CONF, 0.3]
    assert RAW_DRAFT_CONF < min(seen[:3])                                      # raw drafts keep unsure guesses too
