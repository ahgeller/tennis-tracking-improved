"""The visibility gate (``Config.ball_visibility_gate``): the selector refuses its interpolated and
motion fills on frames whose visibility score (the stream_s4 visibility head's per-frame answer)
is below the gate. Off (0, the default) nothing changes: the selector's output is the one it gave
before and the tracking JSON has no ``visibility`` column. On this tree no prepass computes the
score, so a run with the gate on is refused at start-up; the selector, the JSON column and the
configuration are tested here, the prepass that reads the head on ``phase-vis-head2``."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ball_in_play_selector import select_ball_in_play
from tennis_tracker import headless
from tennis_tracker.config import Config
from tennis_tracker.engines import ball_detectors
from tennis_tracker.export import RunResult
from tests.test_selector_scenarios import HEIGHT, WIDTH, _moving_ball

# --- the selector ------------------------------------------------------------------------------


def _gap_run(visibility=None, gate=0.0):
    """A moving ball whose frames 1 and 2 the detector missed: the selector fills them."""
    detections, masks = _moving_ball()
    with_gap = detections[:1] + [[], []] + detections[3:]
    results, _, _ = select_ball_in_play(with_gap, 30.0, WIDTH, HEIGHT, boost_masks=masks, raw_motions=masks,
                                        visibility=visibility, visibility_gate=gate)
    return results


def _rows(results):
    return [None if r is None else (round(r.cx, 3), round(r.cy, 3), r.source, r.conf) for r in results]


def test_the_gate_off_leaves_the_selector_s_output_identical_whatever_the_scores():
    plain = _gap_run()
    assert plain[1] is not None and plain[2] is not None and plain[1].source in ("interp", "motion")
    assert _rows(_gap_run(visibility=[0.0] * 12, gate=0.0)) == _rows(plain)          # 0 is off
    assert _rows(_gap_run(visibility=None, gate=0.5)) == _rows(plain)                # no scores: nothing to refuse


def test_the_gate_refuses_a_fill_on_a_low_visibility_frame_and_keeps_one_on_a_high_visibility_frame():
    plain = _gap_run()
    scores = [0.9] * 12
    scores[1] = 0.1                                                      # the head calls frame 1 empty
    gated = _gap_run(visibility=scores, gate=0.3)
    assert gated[1] is None and gated[2] is not None
    # The rest of the trajectory is the same track: every other frame keeps its source and its
    # place (the refit smooths each point over its neighbours, so a missing row moves them a hair).
    assert [r.source for r in gated if r is not None] == [r.source for r in plain if r is not None and r is not plain[1]]
    for before, after in zip(plain[3:], gated[3:]):
        assert (after.cx, after.cy) == pytest.approx((before.cx, before.cy), abs=0.1)
    # A detection is never refused: the gate is about fills.
    scores[0] = 0.0
    assert _gap_run(visibility=scores, gate=0.3)[0].source == "det"
    # A frame without a score (None, or past the list) is not refused.
    scores[1] = None
    assert _gap_run(visibility=scores[:2], gate=0.3)[1] is not None


# --- the detector ------------------------------------------------------------------------------


def test_a_run_asking_for_the_gate_is_refused_at_start_up_on_this_tree():
    """No prepass here reads the visibility head (the streaming prepass decodes through engines
    exported without it), so the gate on is an error at start-up, never a silent no-op; off, a
    detector answers no scores."""
    for gate in (0.3, 1.5):
        with pytest.raises(ValueError, match="ball_visibility_gate"):
            ball_detectors.refuse_visibility_gate(Config(ball_visibility_gate=gate))
    ball_detectors.refuse_visibility_gate(Config())
    assert ball_detectors.BallDetectorBackend().visibility_scores() is None
    with patch.object(torch.cuda, "is_available", return_value=True):
        with pytest.raises(ValueError, match="ball_visibility_gate"):
            ball_detectors.GridTrackNetBallDetector("w.pt", Config(ball_visibility_gate=0.3))
        with pytest.raises(ValueError, match="ball_visibility_gate"):
            ball_detectors.TensorRTRuntimeBallDetector("ball.engine", Config(ball_visibility_gate=0.3), None)


# --- the tracking JSON and the configuration ----------------------------------------------------------


def test_the_frame_rows_carry_the_visibility_score_only_when_the_run_computed_it():
    base = dict(per_frame=[None, None], chosen_track=None, all_tracks=[], frame_count=2, court_keypoints=[],
                player_boxes=[], last_valid_court_keypoints=None, elapsed_sec=0.0, timing={}, pass2_frames_rendered=0,
                outputs={})
    assert all("visibility" not in row for row in RunResult(**base).frame_rows())
    rows = RunResult(**base, visibility=[0.25, None]).frame_rows()
    assert rows[0]["visibility"] == 0.25 and "visibility" not in rows[1]


def test_the_headless_config_passes_the_gate_through_and_leaves_it_off_by_default():
    assert Config().ball_visibility_gate == 0.0
    cfg = headless.headless_config(Path("v.mp4"), model_path=Path("w.pt"), conf=0.4, device="0", tracking_json=Path("v.json"))
    assert cfg.ball_visibility_gate == 0.0
    gated = headless.headless_config(Path("v.mp4"), model_path=Path("w.pt"), conf=0.4, device="0", tracking_json=Path("v.json"),
                                     visibility_gate=0.3)
    assert gated.ball_visibility_gate == 0.3
