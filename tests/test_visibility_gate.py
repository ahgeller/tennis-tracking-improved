"""The visibility gate (``Config.ball_visibility_gate``): the selector refuses its interpolated and
motion fills on frames whose visibility score (the stream_s4 visibility head's per-frame answer)
is below the gate. Off (0, the default) nothing changes: the selector's output is the one it gave
before and the tracking JSON has no ``visibility`` column. The streaming prepass computes the
score from stream_s4 weights that carry the head (the decoder half answers it beside the grids);
a run with the gate on over anything else is refused at start-up."""
import copy
import math
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest
import torch

from ball_in_play_selector import select_ball_in_play
from tennis_tracker import ball_models, headless, heatmapnet, streamnet
from tennis_tracker.config import FOREGROUND, PROCESS, Config
from tennis_tracker.engines import ball_detectors
from tennis_tracker.engines.ball_gpu import LogitRing
from tennis_tracker.engines.ball_prepass_process import PrepassRows
from tennis_tracker.export import RunResult
from tests._helpers import Capture, CpuGridTrackNet
from tests.test_ball_detectors import STREAM_SIZE, _bare_detector, _stream_frame, _stream_net
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


def _peak_net():
    """``_stream_net`` with a peak visibility head whose logits vary from frame to frame."""
    model = _stream_net()
    model.enable_vis_head("peak")
    torch.manual_seed(7)
    with torch.no_grad():
        model.vis.out.weight.normal_(0.0, 0.02)
        model.vis.frame_bias.normal_(0.0, 0.5)
    return model.eval()


def test_a_backend_or_weights_that_cannot_answer_the_gate_refuse_it_at_start_up(tmp_path):
    """The gate on is an error at start-up wherever no score would be computed, never a silent
    no-op: the single-frame engine, weights without the head, a network that does not stream, the
    streaming prepass switched off, and a gate that is no probability. Off, nothing is checked."""
    for gate in (0.3, 1.5):
        with pytest.raises(ValueError, match="ball_visibility_gate"):
            ball_detectors.refuse_visibility_gate(Config(ball_visibility_gate=gate))
    ball_detectors.refuse_visibility_gate(Config())
    assert ball_detectors.BallDetectorBackend().visibility_scores() is None
    headless_net, peak, heat = tmp_path / "plain.pt", tmp_path / "peak.pt", tmp_path / "heat.pt"
    heatmapnet.save_model(streamnet.StreamNet(), headless_net)
    heatmapnet.save_model(_peak_net(), peak)
    heatmapnet.save_model(heatmapnet.HeatmapNet(), heat)
    on = Config(ball_visibility_gate=0.3)
    ball_detectors.require_visibility_head(on, peak)
    ball_detectors.require_visibility_head(Config(), tmp_path / "missing.pt")           # off: the file is not read
    with pytest.raises(ValueError, match="has none"):
        ball_detectors.require_visibility_head(on, headless_net)
    with pytest.raises(ValueError, match="heatmap_s4"):
        ball_detectors.require_visibility_head(on, heat)
    with pytest.raises(ValueError, match="ball_prepass_streaming is off"):
        ball_detectors.require_visibility_head(Config(ball_visibility_gate=0.3, ball_prepass_streaming=False), peak)
    with pytest.raises(ValueError, match="probability below 1"):
        ball_detectors.require_visibility_head(Config(ball_visibility_gate=1.5), peak)
    with patch.object(torch.cuda, "is_available", return_value=True):
        with pytest.raises(ValueError, match="has none"):
            ball_detectors.GridTrackNetBallDetector(str(headless_net), on)
        with pytest.raises(ValueError, match="ball_visibility_gate"):
            ball_detectors.TensorRTRuntimeBallDetector("ball.engine", on, None)


def test_the_deployed_form_of_a_record_with_the_head_keeps_the_head(tmp_path):
    """``load_deployed`` folds the network a record with the visibility head describes and keeps
    the head (it once built a network without one and failed on the head's keys); the folded
    network's logits are the trainable one's to rounding, and its halves' decoder answers them."""
    model, weights = _peak_net(), tmp_path / "peak.pt"
    heatmapnet.save_model(model, weights)
    deployed = ball_models.load_deployed(weights, torch.device("cpu")).float()
    assert deployed.vis_pool == "peak"
    unit = torch.rand((2, 15, *STREAM_SIZE), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        grids, logits = model(unit, return_vis=True)
        folded_grids, folded_logits = deployed(unit, return_vis=True)
        encoder, decoder = deployed.streaming_halves(visibility=True)
        frames = unit.reshape(2, 5, 3, *STREAM_SIZE)
        features = encoder(frames[:, 1:].reshape(8, 3, *STREAM_SIZE), frames[:, :-1].reshape(8, 3, *STREAM_SIZE))
        halves_grids, halves_logits = decoder(*(level.reshape(2, -1, *level.shape[2:]) for level in features))
    # The folded weights were rounded to FP16 once (load_deployed), so the answers agree to that rounding.
    assert torch.allclose(folded_grids, grids, atol=2e-2) and torch.allclose(folded_logits, logits, atol=2e-2)
    assert torch.equal(halves_grids, folded_grids) and torch.equal(halves_logits, folded_logits)
    assert logits.std() > 0.05                                              # a head that says something
    with pytest.raises(ValueError, match="no visibility head"):
        streamnet.StreamNet().streaming_halves(visibility=True)
    plain = streamnet.StreamNet().eval().streaming_halves()[1]
    assert isinstance(plain(*(torch.zeros((1, 4 * channels, STREAM_SIZE[0] // stride, STREAM_SIZE[1] // stride))
                              for channels, stride in zip(streamnet.ENCODER_CHANNELS[1:], (4, 8, 16)))), torch.Tensor)


def test_the_logit_ring_averages_a_frame_s_logits_over_the_units_holding_it():
    ring = LogitRing(16, torch.device("cpu"))
    ring.add(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0], [9.0] * 5]),
             [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])                              # the third row is padding
    assert ring.take([0, 1, 5, 2]).tolist() == [1.0, 6.0, 50.0, 11.5]
    ring.add(torch.tensor([[7.0] * 5]), [[16, 21, 22, 23, 24]])               # frame 16 reuses frame 0's cleared slot
    assert ring.take([16]).tolist() == [7.0]


@pytest.mark.parametrize("step, stride, count, batch", [(1, 1, 14, 3), (1, 2, 27, 4), (2, 1, 13, 2), (5, 2, 33, 4),
                                                        (1, 2, 7, 4)])
def test_the_streamed_prepass_scores_each_frame_with_the_mean_of_its_units_logits(monkeypatch, step, stride, count,
                                                                                    batch):
    """The gate on, the real network streamed on the CPU through its halves: a frame's score is the
    sigmoid of the mean logit the head gives it across the units the prepass ran over it (with one
    unit per frame, the first unit that answered it), the detections are the ones the gate off
    gives, and a frame no unit covers has no score."""
    monkeypatch.setattr(ball_detectors, "_UNITS_PER_INFERENCE", batch)
    monkeypatch.setattr(ball_detectors, "HEIGHT", STREAM_SIZE[0])
    monkeypatch.setattr(ball_detectors, "WIDTH", STREAM_SIZE[1])
    monkeypatch.setattr(ball_detectors, "require_visibility_head", lambda cfg, weights: None)
    model = _peak_net()
    fps = {1: 30.0, 2: 60.0}[stride]
    runs = {}
    for gate in (0.0, 0.4):
        cfg = Config(device="0", conf=0.0, gridtracknet_window_step=step, ball_visibility_gate=gate)
        detector = CpuGridTrackNet(cfg, model=copy.deepcopy(model), decode=None, frame_tensor=_stream_frame)
        assert detector.stream_halves is not None and detector.stream_halves.visibility == bool(gate)
        with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
             patch.object(torch.cuda, "synchronize"):
            runs[gate] = (detector.detect_clip(Path("s.mp4"), fps, 1920, 1080, count), detector.visibility_scores())
    assert runs[0.0][1] is None and runs[0.4][0] == runs[0.0][0]
    scores = runs[0.4][1]
    assert scores is not None and len(scores) == count
    # The reference: the folded network on every unit the prepass cuts, phase by phase.
    reference = ball_models.deployed(copy.deepcopy(model))
    sums, counts, first = [0.0] * count, [0] * count, {}
    with torch.inference_mode():
        for phase in range(stride):
            frames = list(range(phase, count, stride))
            starts = list(range(0, len(frames) - 4, step))
            if len(frames) >= 5 and starts[-1] != len(frames) - 5:
                starts.append(len(frames) - 5)                              # the trailing window
            for start in starts:
                unit = frames[start:start + 5]
                stacked = torch.cat([_stream_frame(torch.full((1, 1, 1), index), None) for index in unit]).unsqueeze(0)
                logits = reference(stacked, return_vis=True)[1][0].tolist()
                for index, logit in zip(unit, logits):
                    sums[index] += logit
                    counts[index] += 1
                    first.setdefault(index, logit)
    for index, score in enumerate(scores):
        if not counts[index]:
            assert score is None
            continue
        logit = sums[index] / counts[index] if step != 5 else first[index]
        assert score == pytest.approx(1.0 / (1.0 + math.exp(-logit)), abs=1e-4), index
    assert len({round(score, 3) for score in scores if score is not None}) > 1 or count < 9


def test_the_prepass_process_s_scores_arrive_beside_its_rows():
    """The parent of a prepass process keeps the scores each ``PrepassRows`` carries, frame for
    frame beside the rows; rows without scores (a child that sent none) count as frames without."""
    rows = [[([0.0, 0.0, 1.0, 1.0], 0.9)] for _ in range(6)]

    class Remote:
        def run(self, video_path, fps, width, height, total, on_rows, on_end):
            def feed():
                on_rows(PrepassRows(0, rows[:4], [0.1, 0.2, None, 0.4]))
                on_rows(PrepassRows(4, rows[4:]))
                on_end(None)
            threading.Thread(target=feed).start()

    detector = _bare_detector(Config(device="0", ball_prepass_mode=PROCESS, ball_visibility_gate=0.3))
    detector._gate = 0.3
    detector._remote = Remote()
    assert detector.detect_clip(Path("s.mp4"), 30.0, 64, 36, 6) == rows
    assert detector.visibility_scores() == [0.1, 0.2, None, 0.4, None, None]
    assert detector.visibility_rows(1, 3) == [0.2, None]
    off = _bare_detector(Config(device="0", ball_prepass_mode=FOREGROUND))
    assert off.visibility_scores() is None and off.visibility_rows(0, 2) is None



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
