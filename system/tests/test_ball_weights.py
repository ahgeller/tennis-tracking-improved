"""tennis_tracker.ball_weights: each known weights file's gated threshold, and the error for a missing one."""
import pytest

from tennis_tracker import ball_weights, paths, pipeline
from tennis_tracker.ball_weights import (HEATMAP_S4_WEIGHTS, STREAM_S4_WEIGHTS, UNKNOWN_CONF, missing_weights,
                                         operating_conf, unknown_weights_note)
from tennis_tracker.config import Config


def test_each_known_file_runs_at_its_gated_threshold_and_any_other_at_the_heatmap_networks(tmp_path, monkeypatch):
    assert operating_conf(paths.STREAM_WEIGHTS) == STREAM_S4_WEIGHTS.conf == 0.42      # the one literal
    assert operating_conf(paths.HEATMAP_WEIGHTS) == HEATMAP_S4_WEIGHTS.conf == 0.40
    assert Config().conf == operating_conf(paths.BALL_WEIGHTS)                         # the default's, by default
    monkeypatch.chdir(paths.ROOT)                                            # a relative path is the same file
    assert operating_conf("models/candidates/stream_s4_e60.pt") == STREAM_S4_WEIGHTS.conf
    assert operating_conf(str(paths.ROOT / "models" / "." / "ball_detector.pt")) == 0.40
    assert operating_conf(tmp_path / "stream_s4_e60.pt") == UNKNOWN_CONF == 0.40     # a copy elsewhere is not known


def test_a_missing_known_file_says_what_it_is_and_how_to_run_the_other_detector():
    message = str(missing_weights(paths.STREAM_WEIGHTS))
    assert str(paths.STREAM_WEIGHTS) in message
    assert "stream_s4" in message and "train branch" in message and "models/README.md" in message
    assert "--weights models/ball_detector.pt" in message
    assert "--weights models/candidates/stream_s4_e60.pt" in str(missing_weights(paths.HEATMAP_WEIGHTS))


def test_a_missing_unknown_file_names_only_the_path(tmp_path):
    expected = f"Ball-detector weights not found: {(tmp_path / 'x.pt').resolve()}"
    assert str(missing_weights(tmp_path / "x.pt")) == expected


def test_the_pipeline_refuses_missing_default_weights_with_that_error(tmp_path, monkeypatch):
    missing = tmp_path / "candidates" / "stream_s4_e60.pt"
    monkeypatch.setattr(ball_weights, "KNOWN", (STREAM_S4_WEIGHTS._replace(path=missing), HEATMAP_S4_WEIGHTS))
    with pytest.raises(FileNotFoundError, match="stream_s4 ball detector") as info:
        pipeline._build_ball_detector(Config(model_path=str(missing)))
    assert "--weights models/ball_detector.pt" in str(info.value)


def test_unknown_weights_are_named_with_the_threshold_they_run_at(tmp_path):
    """A retrained file elsewhere runs at UNKNOWN_CONF; the note says so, and nothing for known files."""
    note = unknown_weights_note(tmp_path / "stream_s4_retrained.pt")
    assert note is not None and "stream_s4_retrained.pt" in note and f"{UNKNOWN_CONF:.2f}" in note and "--conf" in note
    assert unknown_weights_note(paths.STREAM_WEIGHTS) is None and unknown_weights_note(paths.HEATMAP_WEIGHTS) is None
