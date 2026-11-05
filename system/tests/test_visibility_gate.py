"""The visibility gate (``Config.ball_visibility_gate``): the stream_s4 visibility head's per-frame
answer, averaged over the units covering a frame as the heat grids are, refuses the selector's
interpolated and motion fills on frames the head calls empty. Off (0, the default) nothing
changes: the head is never computed, the selector's output is the one it gave before and the
tracking JSON has no ``visibility`` column."""
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from ball_in_play_selector import select_ball_in_play
from tennis_tracker import headless, streamnet
from tennis_tracker.ball_models import deployed
from tennis_tracker.config import Config
from tennis_tracker.engines import ball_trt
from tennis_tracker.export import RunResult
from tests._helpers import Capture, CpuGridTrackNet, StandInModel
from tests.test_ball_detectors import GridStandIn, _FPS_FOR_STRIDE
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


class HeadedStandIn(GridStandIn):
    """GridStandIn with a visibility head: the logit for the frame at position k of a unit is the
    frame's index minus 10 (the index is the frame tensor's value)."""

    has_vis_head = True

    def __init__(self):
        super().__init__()
        self.vis_calls = 0

    def __call__(self, batch, return_vis=False):
        grids = self.answer(batch)
        if not return_vis:
            return grids
        self.vis_calls += 1
        logits = torch.stack([batch[:, 3 * k, 0, 0] - 10.0 for k in range(5)], dim=1)
        return grids, logits


class HeadedEngine(HeadedStandIn):
    """The stand-in as a static-batch engine (``ball_trt.BallEngine``'s surface)."""

    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.stream = None

    def answer(self, batch):
        assert batch.shape[0] == self.batch
        return super().answer(batch)


def _detector(model, window_step, gate):
    def frame_tensor(frame, device):
        return torch.full((3, 1, 1), float(frame[0, 0, 0]))
    cfg = Config(device="0", conf=0.25, gridtracknet_window_step=window_step, ball_visibility_gate=gate)
    return CpuGridTrackNet(cfg, model=model, decode=None, frame_tensor=frame_tensor)


def test_a_model_without_the_head_is_refused_at_start_up_when_the_gate_is_on():
    with pytest.raises(ValueError, match="ball_visibility_gate"):
        _detector(GridStandIn(), 1, 0.3)
    with pytest.raises(ValueError, match="ball_visibility_gate"):
        _detector(HeadedStandIn(), 1, 1.5)                               # a probability, in (0, 1)
    assert _detector(GridStandIn(), 1, 0.0).visibility_scores() is None  # off: no head wanted, none computed


@pytest.mark.parametrize("step, stride, count", [(1, 1, 12), (1, 2, 25), (5, 1, 13), (5, 2, 31)])
def test_the_prepass_averages_the_head_s_logits_over_the_units_covering_a_frame(step, stride, count):
    """Every frame's score is the sigmoid of the mean of the logits the units holding it gave;
    the stand-in's logit is the frame index minus 10 in every unit, so the mean is that. The
    engine path (two outputs, a static batch) answers the same, and with the gate off the head is
    never asked."""
    for engine in (None, HeadedEngine(3)):
        model = HeadedStandIn()
        detector = _detector(model, step, 0.3)
        detector.engine = engine
        with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), patch.object(torch.cuda, "synchronize"):
            results = detector.detect_clip(Path("s.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
        scores = detector.visibility_scores()
        assert len(results) == len(scores) == count
        for index, score in enumerate(scores):
            assert score == pytest.approx(1.0 / (1.0 + math.exp(-(index - 10.0))), abs=1e-5), index
        asked = model.vis_calls if engine is None else engine.vis_calls
        assert asked > 0
    off = _detector(HeadedStandIn(), step, 0.0)
    with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), patch.object(torch.cuda, "synchronize"):
        off.detect_clip(Path("s.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
    assert off.model.vis_calls == 0 and off.visibility_scores() is None


# --- the engine export ----------------------------------------------------------------------------


def test_the_export_is_unchanged_without_the_gate_and_adds_the_head_as_a_second_output_with_it():
    onnx = pytest.importorskip("onnx")
    model = deployed(streamnet.StreamNet(vis_head=True)).half()
    plain = onnx.load_from_string(ball_trt.export_onnx(model, 1, torch.device("cpu")))
    assert [output.name for output in plain.graph.output] == [ball_trt.OUTPUT_NAME]
    headed = onnx.load_from_string(ball_trt.export_onnx(model, 1, torch.device("cpu"), visibility=True))
    assert [output.name for output in headed.graph.output] == [ball_trt.OUTPUT_NAME, ball_trt.VISIBILITY_OUTPUT_NAME]
    assert ball_trt.OUTPUT_NAME < ball_trt.VISIBILITY_OUTPUT_NAME       # the session returns outputs sorted by name
    dims = [d.dim_value for d in headed.graph.output[1].type.tensor_type.shape.dim]
    assert dims == [1, streamnet.VIS_LOGITS]


def test_the_engine_cache_key_and_name_change_only_when_the_head_is_exported(tmp_path):
    weights = tmp_path / "ball.pt"
    weights.write_bytes(b"weights")
    properties = SimpleNamespace(name="RTX test", major=12, minor=0)
    cuda = torch.device("cuda:0")
    with patch.object(ball_trt, "import_tensorrt", return_value=SimpleNamespace(__version__="10.15.1.29")), \
         patch.object(torch.cuda, "get_device_properties", return_value=properties):
        plain = ball_trt.engine_key(weights, "stream_s4", 4, cuda)
        assert ball_trt.engine_key(weights, "stream_s4", 4, cuda, visibility=False) == plain
        assert ball_trt.engine_key(weights, "stream_s4", 4, cuda, visibility=True) != plain
        assert ball_trt.engine_path(weights, "stream_s4", 4, tmp_path, cuda).name == f"ball-stream_s4-b4-{plain}.engine"
        assert ball_trt.engine_path(weights, "stream_s4", 4, tmp_path, cuda, visibility=True).name.startswith("ball-stream_s4-b4-vis-")
        model = SimpleNamespace(ARCH="stream_s4")
        with patch.object(ball_trt, "export_onnx", return_value=b"onnx") as export, \
             patch.object(ball_trt, "build_serialized_engine", return_value=b"plan"):
            ball_trt.ensure_ball_engine(weights, model, 4, tmp_path / "cache", cuda)
            export.assert_called_with(model, 4, cuda, visibility=False)
            ball_trt.ensure_ball_engine(weights, model, 4, tmp_path / "cache", cuda, visibility=True)
            export.assert_called_with(model, 4, cuda, visibility=True)


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
