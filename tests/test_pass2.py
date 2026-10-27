"""tennis_tracker/pass2.py: where the renderer's frames come from (the RAM cache pass 1 filled and the
second decode past it) and its per-frame court model, with the golden harness's fakes."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tennis_tracker import pass2
from tennis_tracker.config import Config
from tests._helpers import Capture
from tests.test_pipeline_run import (  # noqa: F401
    FRAMES, FULL_OVERRIDES, FakeCourtDetector, assert_json_close, clip, load_golden, run_pipeline,
)


class SeekableCapture(Capture):
    """``Capture`` that seeks with ``CAP_PROP_POS_FRAMES``; with ``seeks=False`` a set is ignored,
    as by a container that cannot seek exactly."""

    opened = []

    def __init__(self, count: int, seeks: bool = True):
        super().__init__(count)
        self.seeks = seeks
        self.opened.append(self)

    def set(self, prop, value) -> bool:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        if self.seeks:
            self.index = int(value)
        return self.seeks

    def get(self, prop) -> float:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        return float(self.index)

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok


@pytest.fixture
def captures(monkeypatch):
    """Route pass 2's reopen to a 10-frame ``SeekableCapture``; the list of captures opened."""
    SeekableCapture.opened = []
    monkeypatch.setattr(pass2, "cv2", SimpleNamespace(VideoCapture=lambda path: SeekableCapture(10),
                                                      CAP_PROP_POS_FRAMES=cv2.CAP_PROP_POS_FRAMES))
    return SeekableCapture.opened


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(cfg=Config(), video=SimpleNamespace(path="clip.mp4"))


def _cached(count: int):
    """What pass 1 would have cached of ``Capture``'s clip: its first ``count`` frames."""
    return [np.full((2, 2, 3), value, np.uint8) for value in range(count)]


def _frames(source: pass2._FrameSource, count: int):
    """``(frame value, owned)`` for frames ``0`` to ``count - 1``, ``source`` released afterwards."""
    try:
        out = []
        for fi in range(count):
            frame, owned = source.read(fi)
            out.append((None if frame is None else int(frame[0, 0, 0]), owned))
        return out
    finally:
        source.release()


# --------------------------------------------------------------------------- _FrameSource

def test_frame_source_serves_the_cached_prefix_and_decodes_the_rest_from_a_seek(captures, capsys):
    source = pass2._FrameSource(_ctx(), _cached(4), 10)
    assert "[pass 2] Using RAM frame cache (4 of 10 frames), decoding the rest" in capsys.readouterr().out
    assert _frames(source, 10) == [(value, value >= 4) for value in range(10)]   # cached frames are not the renderer's
    assert len(captures) == 1                                                      # one open, seeked to frame 4


def test_frame_source_decodes_from_the_start_when_the_seek_does_not_land(monkeypatch, captures, capsys):
    monkeypatch.setattr(pass2.cv2, "VideoCapture", lambda path: SeekableCapture(10, seeks=False))
    source = pass2._FrameSource(_ctx(), _cached(4), 10)
    assert ("[pass 2] The container did not seek to frame 4 (landed on 0); decoding from the start and "
            "skipping the 4 cached frames") in capsys.readouterr().out
    assert _frames(source, 10) == [(value, value >= 4) for value in range(10)]
    assert len(captures) == 2                                                      # the failed seek's capture was released


def test_frame_source_uses_the_cache_alone_when_it_holds_the_clip(captures, capsys):
    source = pass2._FrameSource(_ctx(), _cached(10), 10)
    assert "[pass 2] Using RAM frame cache (10 frames)" in capsys.readouterr().out
    assert _frames(source, 10) == [(value, False) for value in range(10)]
    assert captures == []


def test_frame_source_decodes_everything_without_a_cache(captures, capsys):
    source = pass2._FrameSource(_ctx(), None, 10)
    assert "RAM frame cache" not in capsys.readouterr().out
    assert _frames(source, 11) == [(value, True) for value in range(10)] + [(None, True)]
    assert len(captures) == 1


def test_a_partial_cache_renders_the_same_run(tmp_path, monkeypatch, fake_cuda, capsys, clip):
    """With a budget for 15 of the clip's 40 frames the run says so in both passes, renders every
    frame and leaves the tracking JSON as the golden has it."""
    result = run_pipeline(tmp_path, monkeypatch, clip, **dict(FULL_OVERRIDES, pass2_cache_max_mb=10))
    out = capsys.readouterr().out
    assert "[pass 1] RAM frame cache: 15 of 40 frames fit the 10 MiB budget" in out
    assert "[pass 2] Using RAM frame cache (15 of 40 frames), decoding the rest" in out
    assert result["json"]["summary"]["pass2_frames_rendered"] == FRAMES
    assert all(len(hashes) == FRAMES for hashes in result["streams"].values())
    assert_json_close(result["json"], load_golden()["json"], "a partial cache changed the tracking JSON")


def test_frames_before_the_first_court_detection_are_drawn_without_a_court_model(
        tmp_path, monkeypatch, fake_cuda, clip):
    # The ground model used to start from the clip's last keypoints, so the opening frames of a
    # panning clip drew the net line and the minimap with the end-of-clip homography.
    has_model = []
    draw_court_layer = pass2.draw_court_layer

    def recording(targets, cfg, court_kps, ground_model, court_polygon):
        has_model.append(ground_model is not None)
        draw_court_layer(targets, cfg, court_kps, ground_model, court_polygon)

    monkeypatch.setattr(pass2, "draw_court_layer", recording)
    monkeypatch.setattr(FakeCourtDetector, "first_detection", 5)
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES)
    assert len(has_model) == result["raw_json"]["video"]["total_frames"]
    assert has_model[:5] == [False] * 5 and all(has_model[5:])
