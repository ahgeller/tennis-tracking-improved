"""The ball backends' interface and the GridTrackNet prepass on a fake decoder; no GPU or weights."""
import copy
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tennis_tracker import streamnet
from tennis_tracker.engines import ball_detectors
from tennis_tracker.engines.ball_gpu import StreamHalves, transition_slots
from tennis_tracker.engines.ball_prepass_process import PrepassRows
from tennis_tracker.engines.ball_detectors import (
    BallDetectorBackend,
    CachedDetections,
    GridTrackNetBallDetector,
    TensorRTRuntimeBallDetector,
)
from tennis_tracker.config import AUTO, FOREGROUND, PROCESS, THREAD, Config
from tennis_tracker.cuda_kernels import KernelUnavailable
from tests._helpers import Capture, CpuGridTrackNet, StandInModel


def test_ball_boxes_are_clipped_onto_the_frame():
    preds = torch.tensor([[[-2.0, 1.0, 5.0, 3.0, 0.9, 0.0]]])      # a postprocessed row past the left edge
    [(box, conf)] = ball_detectors.decode_ball_predictions(preds, (10.0, 10.0), 0.5, "postprocessed",
                                                           frame_size=(40, 25))
    assert box == [0.0, 10.0, 39.0, 24.0] and conf == pytest.approx(0.9)


def test_backend_interface_and_cached_token():
    backend = BallDetectorBackend()
    assert not backend.supports_cuda_frame()
    assert backend.prepare_video(Path("clip.mp4"), 30.0, 64, 32, 10) is None   # nothing to prepare per frame
    with pytest.raises(NotImplementedError):
        backend.detect(np.zeros((4, 4, 3), np.uint8), 0)
    assert GridTrackNetBallDetector.uses_raw_frames and not TensorRTRuntimeBallDetector.uses_raw_frames
    token = CachedDetections(3, [([1.0, 2.0, 3.0, 4.0], 0.5)])
    assert GridTrackNetBallDetector.detect_async_finish(None, token) == token.detections


def test_gridtracknet_constructor_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        GridTrackNetBallDetector("unused.npz", Config())


# --- the prepass on a fake decoder -------------------------------------------

class RaisingCapture(Capture):
    """A capture whose decoder fails on frame ``fail_at``."""

    def __init__(self, count, fail_at):
        super().__init__(count)
        self.fail_at = fail_at

    def read(self):
        if self.index == self.fail_at:
            raise MemoryError("decoder out of memory")
        return super().read()


# The stand-in models below answer no grid (their output is whatever shape the test likes and
# an injected decoder reads it), so they run the prepass at one unit per frame, where the
# model's answer goes straight to the decoder; the averaging path (a window step below 5, the
# default) reshapes a real 15-channel grid and is exercised with GridStandIn further down.
ONE_UNIT_PER_FRAME = 5


def _bare_detector(cfg=None) -> GridTrackNetBallDetector:
    """A GridTrackNet backend without its constructor: the model, decoder and
    frame encoder are stand-ins that find a ball at (10, 20) on every frame."""
    detector = GridTrackNetBallDetector.__new__(GridTrackNetBallDetector)
    detector.cfg = cfg or Config(device="0", gridtracknet_window_step=ONE_UNIT_PER_FRAME, ball_prepass_mode=FOREGROUND)
    # The constructor resolves AUTO from the weights, which a bare detector has none of: tests name the mode.
    assert detector.cfg.ball_prepass_mode != AUTO, "a bare detector needs an explicit ball_prepass_mode"
    detector.prepass_mode = detector.cfg.ball_prepass_mode
    detector.device = torch.device("cpu")
    detector.model = StandInModel(lambda batch: batch)
    detector._frame_tensor = lambda frame, device: torch.zeros((1, 3))
    detector._decode = lambda output, width, height, conf: [((10.0, 20.0), 0.9)] * (int(output.shape[0]) * 5)
    detector.engine = None
    detector.stream_halves = None
    detector._resizers = {}
    detector._nv12_resizers = {}
    detector._nv12_kernel = None
    detector._remote = None
    detector.on_publish = None
    detector.precomputed = None
    detector._ready_upto = 0
    detector._done = False
    detector._ready = threading.Condition()
    detector._worker = None
    detector._worker_error = None
    return detector


def test_frame_index_is_required():
    detector = _bare_detector()
    detector.precomputed = [[]]
    detector._done = True
    with pytest.raises(ValueError, match="frame_index"):
        detector.detect(np.zeros((2, 2, 3), np.uint8), None)
    with pytest.raises(RuntimeError, match="prepare_video"):
        _bare_detector().detect(np.zeros((2, 2, 3), np.uint8), 0)


def test_prepass_grows_the_cache_past_an_under_reported_frame_count(monkeypatch, fake_cuda, capsys):
    detector = _bare_detector()
    monkeypatch.setattr(ball_detectors.cv2, "VideoCapture", lambda path: Capture(7))
    results = detector.detect_clip(Path("clip.mp4"), 30.0, 1920, 1080, total=3)
    assert len(results) == 7                      # the hint said 3; every decoded frame has an answer
    assert all(len(frame) == 1 for frame in results)
    assert [detector.detect(None, i)[0][1] for i in range(7)] == [0.9] * 7
    with pytest.raises(IndexError, match="decoded 7 frames; source frame 7"):
        detector.detections_at(7)
    out = capsys.readouterr().out
    assert "container reported 3 frames, 7 decoded" in out


def test_foreground_prepass_failure_raises_instead_of_returning_empty_frames(monkeypatch, fake_cuda):
    detector = _bare_detector()
    monkeypatch.setattr(ball_detectors.cv2, "VideoCapture", lambda path: RaisingCapture(7, fail_at=2))
    with pytest.raises(RuntimeError, match="prepass failed") as info:
        detector.detect_clip(Path("clip.mp4"), 30.0, 1920, 1080, total=7)
    assert isinstance(info.value.__cause__, MemoryError)


def test_background_prepass_failure_reaches_the_consumer(monkeypatch, fake_cuda):
    detector = _bare_detector(Config(device="0", ball_prepass_mode=THREAD,
                                     gridtracknet_window_step=ONE_UNIT_PER_FRAME))
    monkeypatch.setattr(ball_detectors.cv2, "VideoCapture", lambda path: RaisingCapture(7, fail_at=2))
    detector.prepare_video(Path("clip.mp4"), 30.0, 1920, 1080, total=7)
    with pytest.raises(RuntimeError, match="prepass failed"):
        detector.detections_at(5)


# --- the prepass through the real constructor on the CPU ----------------------

# Frame rates whose phase stride is 1 and 2 (see tennis_tracker.frame_rate).
_FPS_FOR_STRIDE = {1: 30.0, 2: 60.0}


@pytest.mark.parametrize("stride, count", [(1, 23), (2, 47)])
def test_finalized_misses_are_immutable(stride, count):
    seen = set()

    def decode(batch, width, height, threshold):
        decoded = []
        for unit in batch:
            for value in unit:
                index = int(value)
                decoded.append(((float(index), 20.) if index in seen else None, .9))
                seen.add(index)
        return decoded
    publications = []
    detector = None

    def publish(upto):
        for previous in publications:
            assert detector.precomputed[:len(previous)] == previous
        publications.append([list(row) for row in detector.precomputed[:upto]])
    detector = CpuGridTrackNet(
        Config(device="0", gridtracknet_window_step=ONE_UNIT_PER_FRAME), model=lambda batch: batch, decode=decode,
        frame_tensor=lambda frame, device: torch.tensor([float(frame[0, 0, 0])]),
        on_publish=publish,
    )
    with patch.object(cv2, "VideoCapture", return_value=Capture(count)), \
         patch.object(torch.cuda, "synchronize"):
        results = detector.detect_clip(Path("synthetic.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
    assert len(results) == count
    assert not any(results)
    assert [detector.detections_at(index) for index in range(count)] == [[]] * count


# --- overlapping units, averaged ----------------------------------------------

class GridStandIn(StandInModel):
    """A stateless stand-in answering on a 1x2 grid: for the frame at position k of a unit,
    confidence 0.2 * (k + 1) in column (frame index) % 2 (the index is the frame tensor's value),
    x offset 0.25 and y offset 0.5 everywhere."""

    GRID_COLS, GRID_ROWS = 2, 1

    def __init__(self):
        super().__init__(self.answer)
        self.batches = []

    def answer(self, batch):
        self.batches.append(batch)
        output = torch.zeros((batch.shape[0], 15, 1, 2))
        for row in range(batch.shape[0]):
            for k in range(5):
                col = int(batch[row, 3 * k, 0, 0]) % 2
                output[row, 3 * k, 0, col] = 0.2 * (k + 1)
                output[row, 3 * k + 1] = 0.25
                output[row, 3 * k + 2] = 0.5
        return output


def _grid_detector(model, window_step, publications):
    """A CPU prepass on `model` with the real decoder; a frame's tensor is (3, 1, 1) holding the
    frame's index."""
    def frame_tensor(frame, device):
        return torch.full((3, 1, 1), float(frame[0, 0, 0]))
    cfg = Config(device="0", conf=0.25, gridtracknet_window_step=window_step)
    return CpuGridTrackNet(cfg, model=model, decode=None, frame_tensor=frame_tensor,
                           on_publish=publications.append)


def _centres(results):
    """Each frame's detection as its box centre (x, y), None where there is none."""
    return [((d[0][0][0] + d[0][0][2]) / 2, (d[0][0][1] + d[0][0][3]) / 2) if d else None for d in results]


@pytest.mark.parametrize("stride, count", [(1, 12), (2, 25)])
def test_a_window_step_of_one_averages_the_five_units_covering_a_frame(stride, count):
    """With gridtracknet_window_step 1 a unit starts at every frame of a phase, so an interior
    frame is answered by five units (at positions 0 to 4 in them) and its confidence is their
    mean, 0.6; the clip's first frame is in one unit (0.2, below the 0.25 threshold: absent) and
    its second in two (0.3). Every published prefix stays as it was, and the step-5 prepass on the
    same model answers each frame from the one unit holding it."""
    published, plain_published = [], []
    with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), patch.object(torch.cuda, "synchronize"):
        dense = _grid_detector(GridStandIn(), 1, published)
        results = dense.detect_clip(Path("s.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
        plain = _grid_detector(GridStandIn(), 5, plain_published)
        single = plain.detect_clip(Path("s.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
    assert len(results) == count
    confidences = [d[0][1] if d else None for d in results]
    for index in range(count):
        position = index // stride                                   # the frame's count within its phase
        per_phase = len(range(index % stride, count, stride))        # frames in that phase
        positions = [k for k in range(5) if 0 <= position - k and position - k + 4 <= per_phase - 1]
        expected = sum(0.2 * (k + 1) for k in positions) / len(positions)
        if expected < 0.25:
            assert confidences[index] is None, index
        else:
            assert confidences[index] == pytest.approx(expected, abs=1e-6), index
            assert _centres(results)[index] == pytest.approx(((0.25 + index % 2) * 960, 0.5 * 1080))
    assert published == sorted(published) and published[-1] == count
    interior = [i for i in range(count) if (i // stride) % 5 != 0 and i // stride < 5 * (len(range(i % stride, count, stride)) // 5)]
    assert all(single[i][0][1] == pytest.approx(0.2 * ((i // stride) % 5 + 1)) for i in interior)


def test_the_window_step_is_the_setting_and_is_refused_outside_one_to_five(capsys):
    """window_step is Config.gridtracknet_window_step as set (the default is 1); the construction
    line names a step below 5 and says nothing of one at 5; a step outside 1 to 5 is refused."""
    detector = CpuGridTrackNet(Config(device="0", gridtracknet_window_step=2), model=lambda b: b, decode=lambda *a: [],
                               frame_tensor=lambda f, d: torch.zeros((3, 1, 1)))
    assert detector.window_step() == 2 and ", a unit every 2 frame(s)" in capsys.readouterr().out
    plain = CpuGridTrackNet(Config(device="0", gridtracknet_window_step=5), model=lambda b: b, decode=lambda *a: [],
                            frame_tensor=lambda f, d: torch.zeros((3, 1, 1)))
    assert plain.window_step() == 5 and "a unit every" not in capsys.readouterr().out
    for step in (0, 6):
        with pytest.raises(ValueError, match=f"gridtracknet_window_step must be 1 to 5, not {step}"):
            CpuGridTrackNet(Config(device="0", gridtracknet_window_step=step), model=lambda b: b, decode=lambda *a: [],
                            frame_tensor=lambda f, d: torch.zeros((3, 1, 1)))


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_unit_streams_cut_a_unit_every_step_frames_and_cover_the_tail(step):
    """_UnitStream cuts units starting every `step` frames, reports the next unit's first frame
    while it is known, and its trailing window is the last five frames exactly when the last cut
    did not end on the last frame."""
    stream = ball_detectors._UnitStream(step)
    starts = []
    for index in range(17):
        unit = stream.push((torch.full((3,), float(index)), index))
        if unit is not None:
            assert unit[1] == list(range(index - 4, index + 1))
            starts.append(index - 4)
        if stream.pending_start is not None:
            assert stream.pending_start == (starts[-1] + step if starts else 0)
    assert starts == list(range(0, 13, step))
    trailing = stream.trailing_window()
    if starts[-1] + 4 == 16:
        assert trailing is None
    else:
        assert trailing is not None and trailing[1] == list(range(12, 17))


class StaticBatchEngine(GridStandIn):
    """GridStandIn as an engine of a static batch: it refuses any other batch size."""

    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.stream = None

    def answer(self, batch):
        assert batch.shape[0] == self.batch
        return super().answer(batch)


@pytest.mark.parametrize("step, stride, count", [(1, 2, 25), (5, 1, 23), (5, 2, 31), (2, 1, 14)])
def test_a_static_batch_engine_answers_what_the_pytorch_model_answers(step, stride, count):
    """The engine path pads its last call with copies of the last unit and splits trailing windows
    that overflow a call: every frame gets the answer the PyTorch path gives, on the device-side
    peaks (no host decoder) for both the averaged and the one-unit-per-frame prepass."""
    answers = []
    for engine in (None, StaticBatchEngine(3)):
        published = []
        detector = _grid_detector(GridStandIn(), step, published)
        detector.engine = engine
        with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
             patch.object(torch.cuda, "synchronize"):
            answers.append(detector.detect_clip(Path("s.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count))
        assert published == sorted(published) and published[-1] == count
    assert answers[0] == answers[1] and sum(map(bool, answers[0])) > count // 2


# --- a streaming network ----------------------------------------------------

class StreamStandIn:
    """GridStandIn as a streaming network's halves of a static `batch`: a transition's feature is
    the index of the frame it ends at, and the decoder checks each unit's four are consecutive
    frames of one phase (`stride` apart) before answering as GridStandIn does on that unit."""

    def __init__(self, batch, stride):
        self.batch, self.stride = batch, stride
        self.grid = GridStandIn()
        self.decoded = []

    def halves(self):
        return StreamHalves(encode=self.encode, decode=self.decode, batch=self.batch)

    def encode(self, frames, before):
        assert frames.shape[0] == before.shape[0] == self.batch
        return (frames[:, :1].clone(),)

    def decode(self, ends):
        assert ends.shape == (self.batch, 4, 1, 1)
        ends = ends[:, :, 0, 0]
        assert torch.equal(ends[:, 1:] - ends[:, :-1], torch.full((self.batch, 3), float(self.stride)))
        self.decoded.append(ends[:, -1].tolist())
        indices = torch.cat([ends[:, :1] - self.stride, ends], dim=1)
        return self.grid.answer(indices.view(self.batch, 5, 1, 1, 1).expand(-1, -1, 3, -1, -1).reshape(self.batch, 15, 1, 1))


@pytest.mark.parametrize("batch", [1, 2, 4, 7])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_the_streamed_prepass_decodes_every_unit_the_windowed_one_runs(step, stride, batch):
    """Across window steps, phases and batches, the ring hands each unit its own four transitions
    (the decoder checks them), the slots ``transition_slots`` sizes are never overwritten before a
    unit is decoded although the ring wraps several times, and every frame's answer, from the clip's
    first units to its trailing windows, is the windowed prepass's on the same static batch."""
    count = 3 * transition_slots(batch, step, stride) + 2       # the ring wraps more than twice
    answers, streamed = [], StreamStandIn(batch, stride)
    for halves in (None, streamed.halves()):
        published = []
        cfg = Config(device="0", conf=0.25, gridtracknet_window_step=step, gridtracknet_source_stride=stride)
        detector = CpuGridTrackNet(cfg, model=GridStandIn(), decode=None, on_publish=published.append,
                                   frame_tensor=lambda frame, device: torch.full((3, 1, 1), float(frame[0, 0, 0])))
        detector.engine = StaticBatchEngine(batch) if halves is None else None
        detector.stream_halves = halves
        with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
             patch.object(torch.cuda, "synchronize"):
            answers.append(detector.detect_clip(Path("s.mp4"), 30.0, 1920, 1080, count))
        assert published == sorted(published) and published[-1] == count
    assert answers[0] == answers[1] and sum(map(bool, answers[0])) > count // 2
    if step == 1:                               # a unit ends at every frame from each phase's fifth on
        assert {end for call in streamed.decoded for end in call} == set(range(4 * stride, count))


STREAM_SIZE = (32, 48)          # multiples of StreamNet's total stride, 16


def _stream_net():
    """A StreamNet with BatchNorm statistics off the identity and a head whose heat varies across
    the grid, so every frame has a clear peak to compare."""
    torch.manual_seed(5)
    model = streamnet.StreamNet()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.25, 2.25)
                module.weight.uniform_(0.5, 1.5)
        model.head.weight.normal_(0.0, 0.05)
    return model.eval()


def _stream_frame(frame, device):
    """A random picture seeded by the frame's index, the model input's shape at STREAM_SIZE."""
    return torch.rand((3, *STREAM_SIZE), generator=torch.Generator().manual_seed(int(frame[0, 0, 0])))


@pytest.mark.parametrize("step, stride, count, batch", [
    (1, 2, 47, 4), (1, 1, 23, 3), (2, 2, 31, 4), (5, 2, 33, 4), (3, 1, 14, 1),
    (1, 2, 9, 4),               # one phase gets a single unit, the other none
    (1, 2, 7, 4),               # no phase reaches five frames: no unit at all
])
def test_a_streaming_stream_net_answers_as_its_windowed_prepass(monkeypatch, step, stride, count, batch):
    """The real network, streamed on the CPU through its PyTorch halves (deployed, so folded),
    gives every frame the detection the windowed prepass gives with the trainable form: the same
    point and confidence to float32 rounding, and nothing on a frame no unit covers."""
    monkeypatch.setattr(ball_detectors, "_UNITS_PER_INFERENCE", batch)
    monkeypatch.setattr(ball_detectors, "HEIGHT", STREAM_SIZE[0])
    monkeypatch.setattr(ball_detectors, "WIDTH", STREAM_SIZE[1])
    model = _stream_net()
    answers = []
    # The windowed constructor warms the network up on an FP16 unit, which the CPU cannot convolve.
    unit_forward = StandInModel(lambda batch: model(batch.float()))
    for streaming in (False, True):
        cfg = Config(device="0", conf=0.0, gridtracknet_window_step=step, ball_prepass_streaming=streaming)
        network = copy.deepcopy(model) if streaming else unit_forward
        detector = CpuGridTrackNet(cfg, model=network, decode=None, frame_tensor=_stream_frame)
        assert (detector.stream_halves is not None) == streaming and detector.engine is None
        with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
             patch.object(torch.cuda, "synchronize"):
            answers.append(detector.detect_clip(Path("s.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count))
    windowed, streamed = answers
    assert len(windowed) == len(streamed) == count
    for index, (plain, stream) in enumerate(zip(windowed, streamed)):
        covered = len(range(index % stride, count, stride)) >= 5
        assert bool(plain) == bool(stream) == covered, index
        if covered:
            (box, confidence), = plain
            (stream_box, stream_confidence), = stream
            assert stream_box == pytest.approx(box, abs=1e-3) and stream_confidence == pytest.approx(confidence, abs=1e-5)


class FakeRemote:
    """A ``RemotePrepass`` that answers from a script on a thread: rows in two runs, then the end."""

    def __init__(self, rows, error=None):
        self.rows, self.error, self.closed = rows, error, False

    def run(self, video_path, fps, width, height, total, on_rows, on_end):
        def feed():
            half = len(self.rows) // 2
            on_rows(PrepassRows(0, self.rows[:half]))
            on_rows(PrepassRows(half, self.rows[half:]))
            on_end(self.error)
        threading.Thread(target=feed).start()

    def close(self):
        self.closed = True


def _remote_detector(remote):
    detector = _bare_detector(Config(device="0", ball_prepass_mode=PROCESS))
    detector.on_publish = None
    detector._remote = remote
    return detector


def test_a_prepass_process_answers_through_the_same_cache():
    rows = [[] if index % 3 else [([1.0, 2.0, 3.0, 4.0], 0.5)] for index in range(7)]
    detector = _remote_detector(FakeRemote(rows))
    assert detector.detect_clip(Path("clip.mp4"), 30.0, 1920, 1080, total=7) == rows
    detector.prepare_video(Path("clip.mp4"), 30.0, 1920, 1080, total=7, background=True)
    assert detector.detections_at(6) == rows[6]
    with pytest.raises(IndexError):
        detector.detections_at(7)
    remote = detector._remote
    detector.close()
    assert remote.closed and detector._remote is None


def test_a_prepass_process_failure_reaches_the_consumer():
    detector = _remote_detector(FakeRemote([[]] * 4, error=RuntimeError("the prepass process failed: boom")))
    with pytest.raises(RuntimeError, match="prepass failed"):
        detector.detect_clip(Path("clip.mp4"), 30.0, 1920, 1080, total=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="decodes into pinned memory for a CUDA upload")
def test_the_pinned_reader_gives_cv2s_frames_on_the_device(tmp_path):
    """More frames than the reader has buffers, so every buffer is decoded into again after its
    upload: each frame on the device is the one cv2 reads."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (96, 64))
    rng = np.random.default_rng(3)
    for _ in range(9):
        writer.write(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    writer.release()
    device = torch.device("cuda")
    reader = ball_detectors.open_frames(path, device)
    assert isinstance(reader, ball_detectors.PinnedCapture)
    expected = cv2.VideoCapture(str(path))
    got = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        assert isinstance(frame, torch.Tensor) and frame.device.type == "cuda"
        got.append(frame.clone())
    reader.release()
    for frame in got:
        ok, host = expected.read()
        assert ok and np.array_equal(frame.cpu().numpy(), host)
    assert not expected.read()[0] and len(got) == 9
    expected.release()


class FakeNvdec:
    """Stands in for ``video_decode.NvdecDecoder``: records how it was opened, or refuses the clip."""

    opened: list = []
    refuse = False

    def __init__(self, path, device=None, nv12=False):
        if FakeNvdec.refuse:
            raise RuntimeError("NVDEC does not decode clip.mp4")
        FakeNvdec.opened.append((path, device, nv12))


def test_the_prepass_reads_nvdec_pictures_and_falls_back_to_cv2(monkeypatch, capsys):
    """With ``ball_prepass_gpu_decode`` the prepass opens NVDEC for NV12 pictures once the NV12
    kernel loads; a kernel that cannot load, or a clip NVDEC refuses, is reported and cv2 reads
    the clip; with the option off NVDEC is never tried."""
    monkeypatch.setattr(ball_detectors.video_decode, "nvdec_available", lambda: True)
    monkeypatch.setattr(ball_detectors.video_decode, "NvdecDecoder", FakeNvdec)
    monkeypatch.setattr(ball_detectors, "open_frames", lambda path, device: "cv2 reader")
    kernel = object()
    loads = []
    monkeypatch.setattr(ball_detectors, "load_nv12_kernel", lambda device, cache: loads.append(cache) or kernel)
    FakeNvdec.opened, FakeNvdec.refuse = [], False
    detector = _bare_detector(Config(device="0", ball_prepass_mode=FOREGROUND))
    detector.device = torch.device("cuda", 0)
    assert isinstance(detector._open_frames(Path("clip.mp4")), FakeNvdec)
    assert isinstance(detector._open_frames(Path("clip.mp4")), FakeNvdec)
    assert FakeNvdec.opened == [(Path("clip.mp4"), detector.device, True)] * 2
    assert detector._nv12_kernel is kernel and loads == [Path(detector.cfg.ball_engine_dir)]   # loaded once

    FakeNvdec.refuse = True
    assert detector._open_frames(Path("clip.mp4")) == "cv2 reader"
    assert "NVDEC does not decode clip.mp4; the prepass decodes on the CPU" in capsys.readouterr().out

    def unavailable(device, cache):
        raise KernelUnavailable("cannot load the nvrtc library")

    FakeNvdec.refuse = False
    monkeypatch.setattr(ball_detectors, "load_nv12_kernel", unavailable)
    detector._nv12_kernel = None
    assert detector._open_frames(Path("clip.mp4")) == "cv2 reader"
    assert "cannot load the nvrtc library; the prepass decodes on the CPU" in capsys.readouterr().out

    FakeNvdec.opened = []
    detector.cfg = Config(device="0", ball_prepass_gpu_decode=False)
    assert detector._open_frames(Path("clip.mp4")) == "cv2 reader" and FakeNvdec.opened == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the NV12 kernel runs on a CUDA device")
def test_an_nv12_picture_and_its_bgr_frame_give_the_same_model_input():
    device = torch.device("cuda")
    detector = _bare_detector()
    try:
        detector._nv12_kernel = ball_detectors.load_nv12_kernel(device, None)
    except KernelUnavailable as error:
        pytest.skip(f"no NVRTC here: {error}")
    planes = torch.randint(0, 256, (1620, 1920), dtype=torch.uint8).to(device)
    picture = ball_detectors.video_decode.Nv12Frame(planes, True)
    bgr = ball_detectors.video_decode.nv12_to_bgr(planes, True)
    from_nv12 = detector._model_input(picture, device)
    assert torch.equal(from_nv12, detector._model_input(bgr, device))
    assert torch.equal(from_nv12, detector._model_input(bgr.cpu().numpy(), device))
