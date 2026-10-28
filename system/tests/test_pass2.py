"""tennis_tracker/pass2.py: where the renderer's frames come from (the RAM cache pass 1 filled and the
second decode past it) and its per-frame court model, with the golden harness's fakes."""
import shutil
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ball_in_play_selector import Detection, FrameResult
from tennis_tracker import pass2, video_decode
from tennis_tracker.config import Config
from tennis_tracker.pass1 import Pass1Result
from tennis_tracker.render.trails import track_color
from tennis_tracker.run_context import CourtEvents, Selection
from tests._helpers import Capture
from tests.test_pipeline_run import (  # noqa: F401
    FRAMES, FULL_OVERRIDES, FakeCourtDetector, assert_json_close, clip, load_golden, run_pipeline,
)


class UnseekableCapture(Capture):
    """``Capture`` that refuses a seek: pass 2 must reach its start by decoding, since OpenCV's
    seek lands on another picture on a variable-frame-rate clip."""

    opened = []

    def __init__(self, count: int):
        super().__init__(count)
        self.grabbed = 0
        self.opened.append(self)

    def set(self, prop, value) -> bool:
        raise AssertionError("pass 2 seeked with OpenCV")

    def grab(self) -> bool:
        ok, _ = self.read()
        self.grabbed += ok
        return ok


@pytest.fixture
def captures(monkeypatch):
    """Route pass 2's reopen (``video_decode``'s OpenCV decoder) to a 10-frame ``UnseekableCapture``;
    the list of captures opened."""
    UnseekableCapture.opened = []
    monkeypatch.setattr(video_decode, "nvc", None)
    monkeypatch.setattr(video_decode.cv2, "VideoCapture", lambda path: UnseekableCapture(10))
    return UnseekableCapture.opened


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(cfg=Config(), video=SimpleNamespace(path="clip.mp4"))


def _cached(count: int):
    """What pass 1 would have cached of ``Capture``'s clip: its first ``count`` frames."""
    return [np.full((2, 2, 3), value, np.uint8) for value in range(count)]


def _frames(source: pass2._FrameSource, count: int):
    """The frame values for frames ``0`` to ``count - 1``, ``source`` released afterwards."""
    try:
        return [None if (frame := source.read(fi)) is None else int(frame[0, 0, 0]) for fi in range(count)]
    finally:
        source.release()


# --------------------------------------------------------------------------- _FrameSource

def test_frame_source_serves_the_cached_prefix_and_decodes_the_rest_from_the_first_frame(captures, capsys):
    """Past the cache, OpenCV decodes the clip from its first frame and drops the cached ones
    (grabbed, never converted), with one open and no seek."""
    cached = _cached(4)
    source = pass2._FrameSource(_ctx(), cached, 10)
    assert "[pass 2] Using RAM frame cache (4 of 10 frames), decoding the rest (opencv)" in capsys.readouterr().out
    assert _frames(source, 10) == list(range(10))
    assert len(captures) == 1 and captures[0].grabbed == 4
    assert cached == [None] * 4                    # pass 2 is the cache's last reader: each frame leaves as it is served


def test_frame_source_refuses_a_clip_it_cannot_reopen(monkeypatch, captures):
    monkeypatch.setattr(video_decode.cv2, "VideoCapture",
                        lambda path: SimpleNamespace(isOpened=lambda: False, release=lambda: None))
    with pytest.raises(RuntimeError, match="reopen the input video for pass 2"):
        pass2._FrameSource(_ctx(), _cached(4), 10)


def _painted_index(frame: np.ndarray) -> int:
    """The frame index ``_index_painted_clip`` wrote into the picture: one 32 px block per bit."""
    return sum(1 << bit for bit in range(8) if frame[16, bit * 32 + 16].mean() > 128)


def _index_painted_clip(path, count: int) -> None:
    """A ``count``-frame H.264 mp4 whose timestamps run at 60 fps for the first 100 frames and
    slower after (a variable frame rate, as phone recordings have), each frame's index painted into
    its pixels in binary.  The B-frames and the rate change are what make a timestamp seek land on
    another picture than the one ``CAP_PROP_POS_FRAMES`` names."""
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "256x64", "-r", "60",
            "-i", "-", "-vf", "setpts='if(lt(N,100),N,N+N/3)/60/TB'", "-fps_mode", "vfr", "-c:v", "libx264",
            "-g", "50", "-bf", "3", "-pix_fmt", "yuv420p", "-crf", "10", str(path)]
    encoder = subprocess.Popen(args, stdin=subprocess.PIPE)
    for index in range(count):
        frame = np.zeros((64, 256, 3), np.uint8)
        for bit in range(8):
            if (index >> bit) & 1:
                frame[:, bit * 32:(bit + 1) * 32] = 255
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to encode the clip")
@pytest.mark.parametrize("gpu_decode", [False, True])
@pytest.mark.parametrize("n_cached", [50, 100, 150])
def test_frame_source_puts_every_frame_of_a_variable_frame_rate_clip_on_its_own_index(tmp_path, capsys, n_cached,
                                                                                       gpu_decode):
    """Pass 1 fills the cache from a sequential decode; pass 2's source, continuing from where
    the cache ends, must hand out the same pictures in the same order on a clip whose seeks land
    on the wrong picture while reporting the frame asked for (with OpenCV 4.9's FFmpeg backend the
    seek to frame 49 of this clip decodes frame 57 and the seek to 99 decodes 123; the seek to 149
    lands), and find the start without decoding the clip again: by NVDEC's frame index, or by
    OpenCV's walk to the cached pictures.  The whole test, encode included, stays under ten
    seconds."""
    if gpu_decode and not video_decode.nvdec_available():
        pytest.skip("needs PyNvVideoCodec and a CUDA device")
    started = time.perf_counter()
    path = tmp_path / "vfr.mp4"
    _index_painted_clip(path, 200)
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert [_painted_index(frame) for frame in frames] == list(range(200))      # the clip decodes in order
    ctx = SimpleNamespace(cfg=Config(gpu_decode=gpu_decode), video=SimpleNamespace(path=str(path)))
    source = pass2._FrameSource(ctx, frames[:n_cached], len(frames))
    try:
        served = [source.read(fi) for fi in range(len(frames))]
    finally:
        source.release()
    out = capsys.readouterr().out
    assert f"decoding the rest ({'nvdec' if gpu_decode else 'opencv'})" in out
    assert [_painted_index(frame) for frame in served] == list(range(200))
    assert time.perf_counter() - started < 10.0


def test_frame_source_uses_the_cache_alone_when_it_holds_the_clip(captures, capsys):
    source = pass2._FrameSource(_ctx(), _cached(10), 10)
    assert "[pass 2] Using RAM frame cache (10 frames)" in capsys.readouterr().out
    assert _frames(source, 11) == list(range(10)) + [None]
    assert captures == []


def test_frame_source_decodes_everything_without_a_cache(captures, capsys):
    source = pass2._FrameSource(_ctx(), None, 10)
    out = capsys.readouterr().out
    assert "RAM frame cache" not in out and "[pass 2] Decoding the clip (opencv)" in out
    assert _frames(source, 11) == list(range(10)) + [None]
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
    draw = pass2.CourtLayer.draw

    def recording(self, targets, court_kps, ground_model, model_key):
        has_model.append(ground_model is not None)
        draw(self, targets, court_kps, ground_model, model_key)

    monkeypatch.setattr(pass2.CourtLayer, "draw", recording)
    monkeypatch.setattr(FakeCourtDetector, "first_detection", 5)
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES)
    assert len(has_model) == result["raw_json"]["video"]["total_frames"]
    assert has_model[:5] == [False] * 5 and all(has_model[5:])


# --------------------------------------------------------------------------- show_all_tracks

WIDTH, HEIGHT = 640, 360


def _track(track_id, score, *points):
    return SimpleNamespace(track_id=track_id, score=score,
                           observations=[Detection(f, x, y, x - 4, y - 4, x + 4, y + 4, 0.9, 64.0, False)
                                         for f, x, y in points])


def _renderer(cfg: Config, frame_count: int, tracks, chosen):
    """A ``_FrameRenderer`` over ``frame_count`` blank frames with no court, players, detections or
    ROIs, the chosen track's observations as the selected results and no track-debug writer."""
    ctx = SimpleNamespace(cfg=cfg, video=SimpleNamespace(fps=30.0, width=WIDTH, height=HEIGHT),
                          timing=SimpleNamespace(enabled=False))
    pass1 = Pass1Result(detections=[[] for _ in range(frame_count)], rois=[None] * frame_count,
                        player_boxes=[{} for _ in range(frame_count)], court_keypoints=[None] * frame_count)
    chosen_at = {o.frame: o for o in chosen.observations}
    per_frame = [FrameResult(cx=o.cx, cy=o.cy, conf=0.9, source="det") if (o := chosen_at.get(fi)) else None
                 for fi in range(frame_count)]
    selection = Selection(per_frame=per_frame, chosen_track=chosen, all_tracks=list(tracks), court_polygon=None,
                          last_court_keypoints=None, display_path={})
    writers = SimpleNamespace(main=object(), motion_debug=None, track_debug=None)
    return pass2._FrameRenderer(ctx, pass1, selection, CourtEvents(), writers)


def _main_frames(show_all_tracks: bool, with_rejected: bool = True):
    """The tracking frames of a three-frame clip with a chosen track and, unless ``with_rejected``
    is off, a rejected one."""
    chosen = _track(1, 30.0, (0, 100.0, 100.0), (1, 110.0, 100.0), (2, 120.0, 100.0))
    rejected = _track(2, 5.0, (0, 400.0, 250.0), (1, 400.0, 260.0), (2, 400.0, 270.0))
    tracks = [chosen, rejected] if with_rejected else [chosen]
    renderer = _renderer(Config(show_all_tracks=show_all_tracks), 3, tracks, chosen)
    frames = []
    for fi in range(3):
        canvas = renderer._draw(fi, np.zeros((HEIGHT, WIDTH, 3), np.uint8), renderer.selection.per_frame[fi],
                                None, draw_in_place=True)
        assert canvas.debug is None                        # no track-debug video was asked for
        frames.append(canvas.main)
    return frames


def _holds(frame, color) -> bool:
    return bool((frame == np.array(color, np.uint8)).all(axis=2).any())


def _tinted(frame, color):
    """Pixels of ``frame`` in ``color`` at half strength or more: the colour itself, and the
    anti-aliased 1 px trail, which lands at about nine tenths of it."""
    color = np.array(color, np.uint8)
    return (frame >= color // 2).all(axis=2) & (frame <= color).all(axis=2) & frame.any(axis=2)


def test_show_all_tracks_puts_the_rejected_track_on_the_tracking_frame_without_the_debug_video():
    off, on = _main_frames(False)[-1], _main_frames(True)[-1]
    assert not _holds(off, track_color(2))
    assert _holds(on, track_color(2))
    # Its ring sits where the selector saw it this frame and its trail runs up from there.
    assert (on[268, 400] == np.array(track_color(2), np.uint8)).all()
    assert _tinted(on, track_color(2))[250:266, 400].all()
    # The chosen track's marker still sits on top of its own candidate colour at the ball.
    assert not (on[100, 120] == np.array(track_color(1), np.uint8)).all()
    assert np.array_equal(on[100, 120], off[100, 120])


def test_without_show_all_tracks_the_rejected_track_leaves_no_trace_on_the_tracking_frame():
    frames = _main_frames(False)
    alone = _main_frames(False, with_rejected=False)
    assert all(np.array_equal(a, b) for a, b in zip(frames, alone))
    assert not any(_holds(frame, track_color(2)) for frame in frames)
    assert all(frame.any() for frame in frames)                      # the chosen marker is still drawn
