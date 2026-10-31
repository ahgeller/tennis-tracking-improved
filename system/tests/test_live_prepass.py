"""The prepass fed one frame at a time (``GridTrackNetBallDetector.open_stream``), as the live
tracker feeds it: the batched prepass's answers, each frame published within its lookahead."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tennis_tracker.config import Config
from tests._helpers import Capture, CpuGridTrackNet
from tests.test_ball_detectors import GridStandIn, StaticBatchEngine, StreamStandIn

_FPS = {1: 30.0, 2: 60.0}


def _detector(step, stride, published, engine=None, halves=None):
    cfg = Config(device="0", conf=0.25, gridtracknet_window_step=step)
    detector = CpuGridTrackNet(cfg, model=GridStandIn(), decode=None, on_publish=published.append,
                               frame_tensor=lambda frame, device: torch.full((3, 1, 1), float(frame[0, 0, 0])))
    detector.engine, detector.stream_halves = engine, halves
    return detector


@pytest.mark.parametrize("network", ["model", "engine", "streaming"])
@pytest.mark.parametrize("step, stride, count", [(1, 2, 41), (1, 1, 23), (2, 2, 33), (5, 1, 21), (3, 2, 9)])
def test_a_prepass_fed_frame_by_frame_answers_as_the_batched_one(network, step, stride, count):
    """Pushed one frame at a time and run eagerly, every frame gets the batched prepass's answer on
    the same network; mid-clip each frame is final once ``lookahead_frames`` more have been pushed,
    and never published as final before its last unit ran."""
    def make(published):
        if network == "engine":
            return _detector(step, stride, published, engine=StaticBatchEngine(4))
        if network == "streaming":
            return _detector(step, stride, published, halves=StreamStandIn(4, stride).halves())
        return _detector(step, stride, published)

    batched_published = []
    batched = make(batched_published)
    with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), patch.object(torch.cuda, "synchronize"):
        expected = batched.detect_clip(Path("s.mp4"), _FPS[stride], 1920, 1080, count)

    published = []
    live = make(published)
    stream = live.open_stream(_FPS[stride], 1920, 1080)
    assert stream.lookahead_frames == 4 * stride
    capture = Capture(count)
    with live.stream_scope():
        for index in range(count):
            stream.push(capture.read()[1])
            ready = live.ready_upto
            # A frame is final once its last unit is cut: lookahead frames later, or earlier at the start.
            assert ready >= index + 1 - stream.lookahead_frames or index < 5 * stride
            assert all(live.precomputed[frame] == expected[frame] for frame in range(ready))
        stream.finish()
    assert live.ready_upto == count and live.precomputed == expected
    assert published == sorted(published)


def test_a_prepass_in_its_own_process_cannot_be_fed_frames():
    published = []
    detector = _detector(1, 1, published)
    detector._remote = object()
    with pytest.raises(RuntimeError, match="own process"):
        detector.open_stream(30.0, 64, 36)


def test_a_streamed_frame_from_the_host_reaches_the_model_unchanged():
    """``push`` takes a host frame as the clip reader would hand it."""
    seen = []
    detector = _detector(5, 1, [])
    detector._frame_tensor = lambda frame, device: seen.append(int(frame[0, 0, 0])) or torch.zeros((3, 1, 1))
    stream = detector.open_stream(30.0, 2, 2)
    for value in range(3):
        stream.push(np.full((2, 2, 3), value, np.uint8))
    assert seen == [0, 1, 2]
