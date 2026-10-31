"""Ball detector backends: the five-frame prepass (GridTrackNet or HeatmapNet) and the
single-frame TensorRT engine.

Both sit behind ``BallDetectorBackend``: ``GridTrackNetBallDetector`` (the default, a
five-frame temporal model run as a whole-clip prepass; ``_prepass`` keeps source-frame
indices and temporal order; the model is whichever ``ball_models.load_ball_model`` builds
for ``Config.model_path``, GridTrackNet for ``.npz`` and HeatmapNet for ``.pt``, run as a
cached TensorRT engine from ``ball_trt`` or as the PyTorch model, a network that streams as its
two halves over ``ball_gpu.TransitionRing``) and the single-frame
Ultralytics TensorRT engine (``Config.model_path`` ending in ``.engine``), driven through
``TensorRTRuntimeSession`` from ``engine_session``.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import cv2
import numpy as np
import torch
from ball_in_play_selector import RawDetection

from ..config import Config
from .ball_gpu import (HeatRing, ModelResize, Nv12ModelResize, Readback, StreamHalves, TransitionRing, grid_peaks,
                      load_nv12_kernel, peak_points, transition_slots)
from .ball_prepass_process import PrepassRows, RemotePrepass
from .engine_boxes import OutputMode, decode_engine_boxes, engine_output_mode
from .engine_session import TensorRTRuntimeSession
from .. import video_decode
from ..cuda_kernels import Kernel
from ..ball_models import arch_of, load_ball_model, load_deployed, model_class
from ..frame_rate import require_phase_stride
from ..gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH, GridPoint

# A BGR HWC uint8 frame, either a host array or a CUDA tensor of the same layout.
FrameInput = Union[np.ndarray, torch.Tensor]
# What the prepass reads: such a frame, or an NV12 picture NVDEC decoded (converted as it is resized).
PrepassFrame = Union[np.ndarray, torch.Tensor, video_decode.Nv12Frame]
# The prepass's pieces: one frame's value with its source index, and a five-frame unit (the
# frames' values, in order) with the source indices it covers. A value is the frame's input tensor,
# or for a streaming network the number ``TransitionRing.push`` gave it (None for a phase's first).
_FrameValue = Union[torch.Tensor, int, None]
_PrepassFrame = Tuple[_FrameValue, int]
_PrepassUnit = Tuple[List[_FrameValue], List[int]]

_BALL_NMS_IOU = 0.45
# GridTrackNet gives a point; the box handed to the selector is this fraction
# of the frame diagonal around it, and its vertical offset is set at 1080p.
_BOX_RADIUS_FRAC = 0.003
_MIN_BOX_RADIUS_PX = 2.0
_REFERENCE_HEIGHT_PX = 1080.0
_UNITS_PER_INFERENCE = 4            # units per call of the PyTorch model
# Inference calls whose peaks may still be on their way to the host before the prepass waits for
# the oldest: enough that the device never idles while the host reads the next frames.
_READBACKS_IN_FLIGHT = 3
_DEFAULT_NAMES: Dict[int, str] = {0: "ball"}
RUNTIMES = ("tensorrt", "torch")
# Where the prepass runs (``Config.ball_prepass_mode``; ``AUTO`` resolves by the network, ``prepass_mode``).
AUTO, FOREGROUND, THREAD, PROCESS = "auto", "foreground", "thread", "process"
PREPASS_MODES = (AUTO, FOREGROUND, THREAD, PROCESS)


def prepass_mode(cfg: Config, weights: Path) -> str:
    """Where the prepass over `weights` runs: ``Config.ball_prepass_mode``, with ``AUTO`` resolved to
    ``PROCESS`` for a network that streams (``streaming_halves``, while ``Config.ball_prepass_streaming``
    is on) and ``FOREGROUND`` for any other. Another mode is a ValueError."""
    mode = cfg.ball_prepass_mode
    if mode not in PREPASS_MODES:
        raise ValueError(f"ball_prepass_mode must be one of {PREPASS_MODES}, not {mode!r}")
    if mode != AUTO:
        return mode
    streams = cfg.ball_prepass_streaming and callable(getattr(model_class(arch_of(weights)), "streaming_halves", None))
    return PROCESS if streams else FOREGROUND


class FrameReader:
    """What the prepass reads a clip through: ``read()`` returns ``(ok, frame)`` like
    ``cv2.VideoCapture.read``, the frame a BGR HWC uint8 host array or a CUDA tensor of that
    layout, and ``release()`` frees the decoder. ``open_frames`` is the one place a reader is
    made, so another decoder (NVDEC) replaces ``cv2.VideoCapture`` there alone."""

    def read(self) -> Tuple[bool, Optional[FrameInput]]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class PinnedCapture(FrameReader):
    """``cv2.VideoCapture`` decoding each frame straight into one of a few page-locked host buffers,
    which go up to `device` on the current stream without a wait: no pageable frame is copied into
    pinned memory first (1.2 ms a 1080p frame read and uploaded, against 3.6 ms through
    ``Tensor.pin_memory``; ``audit/records/opt_ball_prepass.md``). A buffer is decoded into again
    only once the upload that read it has finished. A frame cv2 could not decode in place (another
    size) is returned as its host array."""

    def __init__(self, capture: Any, device: torch.device, slots: int = 4):
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._capture, self._device = capture, device
        self._buffers = [torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True) for _ in range(slots)]
        self._uploaded: List[Optional[torch.cuda.Event]] = [None] * slots
        self._next = 0

    def read(self) -> Tuple[bool, Optional[FrameInput]]:
        slot = self._next
        self._next = (slot + 1) % len(self._buffers)
        uploaded = self._uploaded[slot]
        if uploaded is not None:
            uploaded.synchronize()
        host = self._buffers[slot]
        ok, frame = self._capture.read(host.numpy())
        if not ok or frame is None:
            return False, None
        if frame.ctypes.data != host.data_ptr() or tuple(frame.shape) != tuple(host.shape):
            return True, frame
        on_device = host.to(self._device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._uploaded[slot] = event
        return True, on_device

    def release(self) -> None:
        self._capture.release()


def open_frames(video_path: Path, device: Optional[torch.device] = None) -> Any:
    """A ``FrameReader`` over `video_path`, decoding into pinned memory and uploading to `device`
    when that is a CUDA device; one that cannot be opened is a RuntimeError."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video for GridTrackNet: {video_path}")
    if device is not None and device.type == "cuda" and capture.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
        return PinnedCapture(capture, device)
    return capture


class BallDetectorBackend:
    """Swappable ball detector interface.

    ``detect`` returns the frame's detections as ``(xyxy_box, confidence)``
    pairs in source pixels.  ``detect_async_start`` / ``detect_async_finish``
    split that into launch and collect so the pipeline can overlap the next
    frame's preprocessing with this frame's inference; the pending token is a
    backend-specific dataclass (``CachedDetections`` or ``TensorRTBallPending``)
    that only the same backend's ``detect_async_finish`` understands.
    ``frame_index`` is the zero-based source frame of ``frame_bgr``; a
    backend that caches a whole clip answers for that frame.

    ``prepare_video`` is called once per clip before the first frame; a
    per-frame backend has nothing to prepare.
    """

    uses_raw_frames = False
    use_async = False
    async_slots = 1
    # Where a whole-clip backend's prepass runs (``prepass_mode``); a per-frame backend has none.
    prepass_mode = FOREGROUND

    def prepare_video(self, video_path: Path, fps: float, width: int, height: int, total: int) -> None:
        """Hook for whole-clip backends; ``total`` is the container's frame count hint."""

    def close(self) -> None:
        """Release what the backend holds beyond this process's memory (a prepass process)."""

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[RawDetection]:
        raise NotImplementedError

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> Any:
        raise NotImplementedError

    def detect_async_finish(self, pending: Any) -> List[RawDetection]:
        raise NotImplementedError

    def supports_cuda_frame(self) -> bool:
        return False


@dataclass
class CachedDetections:
    """GridTrackNet's pending token: the detections are final when it is issued."""

    frame_index: int
    detections: List[RawDetection]


@dataclass
class _Flush:
    """One flush of the prepass, waiting for its peaks to reach the host: the frames below ``upto``
    are final once it is applied. ``frames`` are the averaged frames ``peaks`` answers, one row
    each; ``units`` (the one-unit-per-frame prepass) are the units it answers, five rows each."""

    upto: int
    peaks: Optional[Readback] = None
    frames: Sequence[int] = ()
    units: Sequence[Sequence[int]] = ()
    points: Optional[List[GridPoint]] = None      # decoded on the host already (``_decode``)


class GridTrackNetBallDetector(BallDetectorBackend):
    """Run the five-frame model at its native 30-FPS cadence.

    The model is GridTrackNet or HeatmapNet by ``weights_path``
    (``ball_models.load_ball_model``); both take the same 15-channel unit and
    answer in the same layout, so everything here reads the model's own
    ``INPUT_CHANNELS`` and grid and nothing asks which it holds. It runs as a
    TensorRT FP16 engine (``ball_trt``, ``Config.ball_prepass_runtime``
    "tensorrt", batches of ``Config.ball_engine_batch`` units, the last one
    padded) or as the PyTorch model (four units a call).

    A model with ``streaming_halves`` (``streamnet``, ``stream_s4``) streams instead when
    ``Config.ball_prepass_streaming`` is on: each frame's transition is encoded once and every
    unit the windowed prepass would run is decoded from its four cached transitions
    (``ball_gpu.TransitionRing``), the halves as two TensorRT engines (``ball_trt``) or, should
    those be unavailable, as PyTorch modules. The units, their batching and everything after the
    network are the windowed prepass's.

    ``Config.gridtracknet_window_step`` below 5 (the default is 1) starts a
    unit every that many frames of a phase instead of every 5, so a frame is
    answered by up to five overlapping units, and the answers a frame gets are
    averaged grid by grid before it is decoded once (``ball_gpu.HeatRing``).

    The clip is decoded by NVDEC where it can (``Config.ball_prepass_gpu_decode``;
    each NV12 picture becomes the model input in one kernel, ``ball_gpu.Nv12ModelResize``),
    else by cv2. Every frame is resized to the model's input on the GPU (the bytes
    OpenCV's resize gives of OpenCV's picture, either way) and each frame's peak is found there, so
    only four floats a frame come back, through pinned memory a few calls
    behind the device.

    ``prepare_video`` decodes the whole clip once (before pass 1, on a worker thread or in a child
    process: ``prepass_mode``, ``Config.ball_prepass_mode``) and caches one
    detection per decoded source frame; ``detections_at(i)`` returns frame
    ``i``'s result, waiting for the prepass if it is behind, and raises once
    the prepass has finished without ever decoding frame ``i``.

    ``detect`` / ``detect_async_start`` never read the frame they are given:
    the prepass already decoded the clip.  They answer for ``frame_index``,
    so skipping frames is safe and call order never matters.
    """

    uses_raw_frames = True
    use_async = False
    async_slots = 1

    def __init__(self, weights_path: str, cfg: Config):
        if not torch.cuda.is_available():
            raise RuntimeError("GridTrackNet requires CUDA-enabled torch.")
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}")
        self._frame_tensor: Callable[[PrepassFrame, torch.device], torch.Tensor] = self._model_input
        # The one-unit-per-frame prepass finds each frame's peak on the device; a decoder set here,
        # ``gridtracknet.decode_predictions``'s signature, decodes the model's output on the host
        # instead (the tests' seam for stand-in models that answer no grid).
        self._decode: Optional[Callable[[torch.Tensor, int, int, float], List[GridPoint]]] = None
        self._resizers: Dict[Tuple[int, int], ModelResize] = {}
        self._nv12_resizers: Dict[Tuple[int, int, bool], Nv12ModelResize] = {}
        self._nv12_kernel: Optional[Kernel] = None
        window_step = self.window_step()
        # One entry per decoded source frame, appended by the prepass.
        self.precomputed: Optional[List[List[RawDetection]]] = None
        # Progressive prepass state: frames < ready_upto are final; ``_done``
        # once the prepass has released its capture (with or without an error).
        self._ready_upto = 0
        self._done = False
        self._ready = threading.Condition()
        self._worker: Optional[threading.Thread] = None
        self._worker_error: Optional[BaseException] = None
        # Called with every newly published ``upto`` (the prepass process sends the rows on).
        self.on_publish: Optional[Callable[[int], None]] = None
        self._remote: Optional[RemotePrepass] = None
        self.model: Any = None
        self.engine: Any = None
        self.stream_halves: Optional[StreamHalves] = None
        self.prepass_mode = prepass_mode(cfg, Path(weights_path))
        if self.prepass_mode == PROCESS:
            # The child builds the model and the engines, starting now; this process only reads its answers.
            self._remote = RemotePrepass(weights_path, cfg)
            print(f"[detector] the ball prepass runs in its own process ({Path(weights_path).name}, "
                  f"pid {self._remote.pid})")
            return
        self.model = load_ball_model(Path(weights_path), self.device)
        self.stream_halves = self._open_stream(Path(weights_path))
        if self.stream_halves is None:
            self.engine = self._open_engine(Path(weights_path))
        halves = self.stream_halves
        if halves is not None and halves.stream is None:    # the PyTorch halves run: let cuDNN pick its kernels now
            with torch.inference_mode():
                frame = torch.zeros((halves.batch, 3, HEIGHT, WIDTH), device=self.device,
                                    dtype=next(self.model.parameters()).dtype)
                halves.decode(*(level.repeat(1, FRAMES_PER_UNIT - 1, 1, 1) for level in halves.encode(frame, frame)))
            torch.cuda.synchronize(self.device)
        elif self.engine is None and halves is None:        # the PyTorch model runs: let cuDNN pick its kernels now
            channels = int(getattr(self.model, "INPUT_CHANNELS"))
            with torch.inference_mode():
                self.model(torch.zeros((1, channels, HEIGHT, WIDTH), device=self.device, dtype=torch.float16))
            torch.cuda.synchronize(self.device)
        windows = "" if window_step == 5 else f", a unit every {window_step} frame(s)"
        runtime = f"TensorRT FP16, {self.engine.batch} units a call" if self.engine is not None else "FP16"
        if halves is not None:
            engines = "TensorRT FP16 encoder and decoder" if halves.stream is not None else "FP16"
            runtime = f"{engines} streaming, {halves.batch} a call"
        print(f"[detector] {self.model.ARCH} {runtime}, {self.model.GRID_COLS}x{self.model.GRID_ROWS} grid "
              f"(confidence={cfg.conf:.2f}, device={self.device}{windows})")

    def _open_engine(self, weights: Path) -> Any:
        """The cached TensorRT engine for `weights` (``ball_trt``) when ``Config.ball_prepass_runtime``
        asks for one and the device is a GPU, else None (the PyTorch model runs). A runtime that is
        neither is a ValueError; an engine TensorRT cannot provide is reported and None returned."""
        runtime = self.cfg.ball_prepass_runtime
        if runtime not in RUNTIMES:
            raise ValueError(f"ball_prepass_runtime must be one of {RUNTIMES}, not {runtime!r}")
        if runtime != "tensorrt" or self.device.type != "cuda":
            return None
        try:
            from .ball_trt import BallEngine, ensure_ball_engine
            path = ensure_ball_engine(weights, self.model, int(self.cfg.ball_engine_batch),
                                      Path(self.cfg.ball_engine_dir), self.device)
            return BallEngine(path, self.model, self.device)
        except (RuntimeError, OSError, ImportError) as error:
            print(f"[detector] TensorRT engine unavailable, running the PyTorch model ({error})")
            return None

    def _open_stream(self, weights: Path) -> Optional[StreamHalves]:
        """The model's streaming halves (``streaming_halves``, of the network ``ball_models.load_deployed``
        folds from the weights, which replaces the model) when it has them and
        ``Config.ball_prepass_streaming`` is on, else None (the model runs unit by unit). They run as cached TensorRT engines when ``Config.ball_prepass_runtime`` and the
        device allow, else as the PyTorch modules, a TensorRT failure reported as ``_open_engine``
        reports it."""
        if not self.cfg.ball_prepass_streaming or not callable(getattr(self.model, "streaming_halves", None)):
            return None
        runtime = self.cfg.ball_prepass_runtime
        if runtime not in RUNTIMES:
            raise ValueError(f"ball_prepass_runtime must be one of {RUNTIMES}, not {runtime!r}")
        self.model = load_deployed(weights, self.device)
        encoder, decoder = self.model.streaming_halves()
        if runtime == "tensorrt" and self.device.type == "cuda":
            batch = int(self.cfg.ball_engine_batch)
            try:
                from .ball_trt import ensure_stream_engines, open_stream_engines
                paths = ensure_stream_engines(weights, self.model.ARCH, encoder, decoder, batch,
                                              Path(self.cfg.ball_engine_dir), self.device)
                return open_stream_engines(paths, self.device)
            except (RuntimeError, OSError, ImportError) as error:
                print(f"[detector] TensorRT engines unavailable, streaming the PyTorch model ({error})")
        return StreamHalves(encoder, decoder, _UNITS_PER_INFERENCE)

    def window_step(self) -> int:
        """The frames of a phase between the prepass's unit starts: ``Config.gridtracknet_window_step``,
        which must be 1 to 5 (a ValueError otherwise)."""
        window_step = int(self.cfg.gridtracknet_window_step)
        if not 1 <= window_step <= 5:
            raise ValueError(f"gridtracknet_window_step must be 1 to 5, not {window_step}")
        return window_step

    def source_stride_for(self, fps: float) -> int:
        override = self.cfg.gridtracknet_source_stride
        if override:
            return max(1, int(override))
        try:
            return require_phase_stride(fps)
        except ValueError as error:
            raise RuntimeError(str(error)) from error

    def _open_frames(self, video_path: Path) -> Any:
        """The prepass's reader: NVDEC's NV12 pictures on the device when ``Config.ball_prepass_gpu_decode``
        asks for them, NVDEC opens the clip and the NV12 kernel loads (``ball_gpu.Nv12ModelResize``),
        else cv2 through ``open_frames``; a fallback is reported.

        NVDEC with the BGR conversion (``nv12_to_bgr``) and then ``ModelResize`` would be slower than
        cv2: the prepass is bound by the GPU there, and the conversion's int32 passes over the whole
        picture added 1.3 s on pomona (``audit/records/opt_phaseA_followup.md``)."""
        if self.cfg.ball_prepass_gpu_decode and self.device.type == "cuda" and video_decode.nvdec_available():
            try:
                if self._nv12_kernel is None:
                    self._nv12_kernel = load_nv12_kernel(self.device, Path(self.cfg.ball_engine_dir))
                return video_decode.NvdecDecoder(video_path, device=self.device, nv12=True)
            except RuntimeError as error:          # KernelUnavailable is one
                print(f"[detector] {str(error).splitlines()[0]}; the prepass decodes on the CPU")
        return open_frames(video_path, self.device)

    def _model_input(self, frame: PrepassFrame, device: torch.device) -> torch.Tensor:
        """One frame as the model's (3, HEIGHT, WIDTH) RGB FP16 input on `device`: an NV12 picture
        converted and resized in one kernel (``Nv12ModelResize``), or a BGR HWC uint8 frame (host
        array or CUDA tensor; a host frame goes up as uint8 through pinned memory without waiting)
        resized by ``ModelResize``. Both give the bytes OpenCV's picture resized by OpenCV gives."""
        if isinstance(frame, video_decode.Nv12Frame):
            key = (frame.width, frame.height, frame.full_range)
            nv12_resize = self._nv12_resizers.get(key)
            if nv12_resize is None:
                assert self._nv12_kernel is not None
                nv12_resize = self._nv12_resizers[key] = Nv12ModelResize(
                    frame.width, frame.height, WIDTH, HEIGHT, frame.full_range, device, self._nv12_kernel)
            return nv12_resize(frame)
        if isinstance(frame, torch.Tensor):
            source = frame.to(device)
        else:
            source = torch.from_numpy(np.ascontiguousarray(frame))
            if device.type == "cuda":
                source = source.pin_memory().to(device, non_blocking=True)
        size = (int(source.shape[1]), int(source.shape[0]))
        resize = self._resizers.get(size)
        if resize is None:
            resize = self._resizers[size] = ModelResize(size[0], size[1], WIDTH, HEIGHT, device)
        return resize(source)

    def prepare_video(self, video_path: Path, fps: float, width: int, height: int, total: int,
                      background: Optional[bool] = None) -> None:
        """Decode the video once and cache one detection per decoded frame.

        ``total`` is the container's frame count, only a hint: the cache
        grows with the frames actually decoded, and a mismatch is reported.
        ``background`` (default: unless ``prepass_mode`` is ``FOREGROUND``)
        returns at once, the prepass running on a worker thread or in the
        prepass process, so pass 1 can start on frame 0 while the detector is
        still ahead of it; ``detect()`` then blocks only if pass 1 catches up.
        Results are identical to the synchronous pass, whose failure raises
        from here.
        """
        source_stride = self.source_stride_for(fps)
        cached: List[List[RawDetection]] = []
        self.precomputed = cached
        self._ready_upto = 0
        self._done = False
        self._worker_error = None
        if background is None:
            background = self.prepass_mode != FOREGROUND
        if self._remote is not None:
            self._run_remote(self._remote, video_path, fps, width, height, total, cached)
            if not background:
                with self._ready:
                    while not self._done:
                        self._ready.wait(timeout=0.5)
                self._raise_if_failed()
            return
        capture = self._open_frames(video_path)
        args =(capture, cached, source_stride, width, height, int(total))
        if background:
            self._worker = threading.Thread(target=self._prepass, args=args, daemon=True, name="gridtracknet-prepass")
            self._worker.start()
        else:
            self._worker = None
            self._prepass(*args)
            self._raise_if_failed()

    def _run_remote(self, remote: RemotePrepass, video_path: Path, fps: float, width: int, height: int, total: int,
                    cached: List[List[RawDetection]]) -> None:
        """Start the prepass process on the clip: its final rows land in `cached` in order and are
        published as they arrive; its end, or its error, finishes the prepass here."""
        def on_rows(message: PrepassRows) -> None:
            del cached[message.start:]
            cached.extend(message.rows)
            self._publish(message.start + len(message.rows))

        def on_end(error: Optional[BaseException]) -> None:
            self._worker_error = error
            self._finish()

        remote.run(video_path, fps, width, height, total, on_rows, on_end)

    def close(self) -> None:
        """Stop the prepass process, if the prepass runs in one."""
        if self._remote is not None:
            self._remote.close()
            self._remote = None

    def _publish(self, upto: int) -> None:
        with self._ready:
            self._ready_upto = max(self._ready_upto, upto)
            self._ready.notify_all()
        if self.on_publish is not None:
            self.on_publish(upto)

    def _finish(self) -> None:
        with self._ready:
            self._done = True
            self._ready.notify_all()

    def _raise_if_failed(self) -> None:
        error = self._worker_error
        if error is None:
            return
        if not isinstance(error, Exception):   # KeyboardInterrupt, SystemExit: pass through
            raise error
        raise RuntimeError("GridTrackNet prepass failed") from error

    def _work_stream(self) -> Any:
        """The CUDA stream the prepass works on: the engine's (so its output is read in order without
        a wait), else a stream of its own on the worker thread so it overlaps pass 1 on the default
        stream; None, the current stream, in the foreground or off the GPU."""
        if self.device.type != "cuda":
            return None
        if self.engine is not None:
            return self.engine.stream
        if self.stream_halves is not None and self.stream_halves.stream is not None:
            return self.stream_halves.stream
        return torch.cuda.Stream(device=self.device) if self._worker is not None else None

    def _prepass(self, capture: Any, cached: List[List[RawDetection]], source_stride: int, width: int, height: int,
                 total_hint: int) -> None:
        # One independent stream of five-frame units per phase. The model wants 30 FPS
        # spacing, so at 60 FPS the odd frames form a second, equally valid unit stream;
        # running only phase 0 leaves every odd frame unobserved. ``_UnitStream`` cuts a
        # phase's units every ``window_step`` of its frames from the last five it saw. A streaming
        # network's units carry their frames' transition numbers in the ring instead of the frames.
        window_step = self.window_step()
        streams = [_UnitStream(window_step) for _ in range(source_stride)]
        batches: List[_PrepassUnit] = []
        engine = self.engine
        network = engine if engine is not None else self.model
        halves = self.stream_halves
        static = engine if engine is not None else halves            # a network of a static batch
        batch_units = static.batch if static is not None else _UNITS_PER_INFERENCE
        transitions = None
        if halves is not None:
            transitions = TransitionRing(halves, transition_slots(batch_units, window_step, source_stride), source_stride)
        averaging = window_step != 5
        # The frames summed in the ring and not yet decoded.
        covered: Set[int] = set()
        ring: Optional[HeatRing] = None
        finalized: Set[int] = set()
        in_flight: Deque[_Flush] = deque()
        frame_index = inference_calls = detected = 0
        grid = (0, 0)
        started = time.perf_counter()
        y_offset = self.cfg.gridtracknet_y_offset_px * height / _REFERENCE_HEIGHT_PX
        radius = max(_MIN_BOX_RADIUS_PX, _BOX_RADIUS_FRAC * np.hypot(width, height))

        def store(index: int, found: GridPoint) -> None:
            nonlocal detected
            point, confidence = found
            if point is None:
                return
            x, y = point[0], point[1] + y_offset
            cached[index] = [([x - radius, y - radius, x + radius, y + radius], confidence)]
            detected += 1

        def apply(flush: _Flush) -> None:
            # The peaks become points on the host; with overlapping units each decoded frame is
            # final, otherwise the first unit to answer a source frame finalises it and a later
            # overlapping unit (the clip's trailing window) is ignored.
            points = flush.points
            if flush.peaks is not None:
                rows, cols = grid
                points = peak_points(flush.peaks.wait(), rows, cols, width, height, self.cfg.conf)
            if points is not None:
                if averaging:
                    for index, found in zip(flush.frames, points):
                        store(index, found)
                else:
                    for unit_index, indices in enumerate(flush.units):
                        for index, found in zip(indices, points[unit_index * FRAMES_PER_UNIT:]):
                            if index not in finalized:
                                finalized.add(index)
                                store(index, found)
            self._publish(flush.upto)

        def drain(everything: bool = False) -> None:
            while in_flight and (everything or len(in_flight) > _READBACKS_IN_FLIGHT or in_flight[0].peaks is None
                                 or in_flight[0].peaks.ready()):
                apply(in_flight.popleft())

        def flush(final: bool = False) -> None:
            nonlocal inference_calls, ring, grid
            # Every frame below the oldest unit not yet cut is final (a mid-clip call cannot be
            # sure of later frames: the next unit may still cover them).
            pending_min = min((stream.pending_start for stream in streams if stream.pending_start is not None),
                              default=frame_index + 1)
            upto = frame_index if final else min(pending_min, frame_index + 1)
            record = _Flush(upto)
            if batches:
                units = [indices for _, indices in batches]
                if transitions is not None:     # pads its static batch itself
                    output = transitions.decode([cast(List[Optional[int]], values) for values, _ in batches])
                else:
                    frames = [cast(torch.Tensor, tensor) for tensors, _ in batches for tensor in tensors]
                    if engine is not None:      # a static batch: pad with the last unit
                        frames += frames[-FRAMES_PER_UNIT:] * (batch_units - len(batches))
                    stacked = torch.cat(frames).view(len(frames) // FRAMES_PER_UNIT, -1, *frames[0].shape[1:])
                    output = network(stacked)
                grid = (int(output.shape[-2]), int(output.shape[-1]))
                if averaging:
                    if ring is None:
                        ring = HeatRing(_ring_slots(batch_units, window_step, source_stride), *grid, output.device)
                    for indices in units:
                        for index in indices:
                            covered.add(index)
                    ring.check_span(min(covered), frame_index)
                    ring.add(output, units)
                elif self._decode is not None:
                    record.points, record.units = self._decode(output[:len(units)], width, height, self.cfg.conf), units
                else:
                    record.peaks = Readback(grid_peaks(output[:len(units)].reshape(-1, 3, *grid)))
                    record.units = units
                inference_calls += 1
                batches.clear()
            if ring is not None:
                ready = sorted(index for index in covered if index < upto)
                if ready:
                    for index in ready:
                        covered.discard(index)
                    record.peaks, record.frames = Readback(ring.take(ready)), ready
            in_flight.append(record)
            drain(everything=final)

        try:
            stream = self._work_stream()
            stream_scope = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with stream_scope, torch.inference_mode():
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    cached.append([])
                    phase = frame_index % source_stride
                    tensor = self._frame_tensor(frame, self.device)
                    value: _FrameValue = tensor if transitions is None else transitions.push(tensor, phase)
                    unit = streams[phase].push((value, frame_index))
                    if unit is not None:
                        batches.append(unit)
                        if len(batches) == batch_units:
                            flush()
                    frame_index += 1
                batches.extend(window for window in (stream.trailing_window() for stream in streams) if window is not None)
                while static is not None and len(batches) > batch_units:    # trailing windows past a static batch
                    rest = batches[batch_units:]
                    del batches[batch_units:]
                    flush()
                    batches.extend(rest)
                flush(final=True)
                if stream is not None:
                    stream.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
        except BaseException as error:  # surfaced to the consumer by _raise_if_failed
            self._worker_error = error
        finally:
            capture.release()
            self._publish(len(cached))
            self._finish()

        elapsed = time.perf_counter() - started
        decode = getattr(capture, "backend", video_decode.BACKEND_OPENCV)
        self._report_prepass(frame_index, detected, inference_calls, source_stride, elapsed, total_hint, decode)

    def _report_prepass(self, frame_index: int, detected: int, inference_calls: int, source_stride: int,
                        elapsed: float, total_hint: int, decode: str = video_decode.BACKEND_OPENCV) -> None:
        """The prepass's console line, and a second when the decoded frame count differs from the
        container's (the decoded count is the one every index refers to)."""
        print(
            f"[detector] GridTrackNet prepass: {frame_index} source frames, {detected} detections, "
            f"{inference_calls} batched calls across {source_stride} phase(s) "
            f"in {elapsed:.1f}s ({frame_index / max(elapsed, 1e-9):.1f} source fps, {decode} decode)"
            + (" [overlapped with pass 1]" if self._worker is not None else "")
        )
        if self._worker_error is None and frame_index != total_hint:
            print(f"[detector] GridTrackNet prepass: the container reported {total_hint} frames, "
                  f"{frame_index} decoded; the decoded count is used")

    def wait_until_ready(self, index: int) -> None:
        """Block until frame ``index`` is final or the prepass has finished; raises if it failed."""
        with self._ready:
            while self._ready_upto <= index and not self._done:
                self._ready.wait(timeout=0.5)
        self._raise_if_failed()

    def detect_clip(self, video_path: Path, fps: float, width: int, height: int,
                    total: int) -> List[List[RawDetection]]:
        """Run the prepass in the foreground and return each decoded frame's detections.

        Index ``i`` of the result holds frame ``i``'s detections: an empty list
        when the ball was not found, else one ``(xyxy_box, confidence)`` pair.
        A prepass that fails raises instead of returning partial results.
        """
        self.prepare_video(video_path, fps, width, height, total, background=False)
        return list(self.precomputed or [])

    def detections_at(self, index: int) -> List[RawDetection]:
        """Detections for source frame ``index``, waiting for the prepass if it
        is behind.  Raises ``IndexError`` once the prepass has finished and
        decoded no such frame."""
        if self.precomputed is None:
            raise RuntimeError("GridTrackNet.prepare_video() must run before detection.")
        self.wait_until_ready(index)
        if index >= len(self.precomputed):
            raise IndexError(
                f"GridTrackNet prepass decoded {len(self.precomputed)} frames; source frame {index} does not exist"
            )
        return self.precomputed[index]

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[RawDetection]:
        """Detections for source frame ``frame_index``.  ``frame_bgr`` is never
        read: the prepass already decoded the clip."""
        return self.detect_async_start(frame_bgr, frame_index).detections

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> CachedDetections:
        if frame_index is None:
            raise ValueError("GridTrackNet answers per source frame: frame_index is required")
        index = int(frame_index)
        return CachedDetections(index, self.detections_at(index))

    def detect_async_finish(self, pending: CachedDetections) -> List[RawDetection]:
        return pending.detections


def _ring_slots(batch_units: int, window_step: int, source_stride: int) -> int:
    """Slots enough for ``HeatRing`` to hold every frame the prepass has summed and not decoded:
    the frames one call's units start across (``batch_units`` units, a unit every ``window_step``
    frames of a phase, so ``batch_units * window_step`` source frames whatever the phases) and one
    unit's length on either side, doubled for headroom; ``HeatRing.check_span`` guards it."""
    return 2 * (batch_units * window_step + 2 * FRAMES_PER_UNIT * source_stride)


class _UnitStream:
    """One phase's frames cut into five-frame units: a unit starts every `step` frames of the
    phase (5: units that do not overlap, each frame answered once; 1: a unit at every frame,
    each frame in up to five). ``push`` takes the phase's next frame and returns the unit it
    completes, if any; ``trailing_window`` is the phase's last five frames when the clip ended
    between unit starts (a unit left short), so every frame is covered; ``pending_start`` is the
    source index of the first frame of the next unit to cut, None while that frame has not been
    read: frames below it are covered by no unit still to come."""

    def __init__(self, step: int):
        self.step = step
        self.recent: Deque[_PrepassFrame] = deque(maxlen=5)
        self.seen = 0               # frames of the phase pushed so far
        self.next_cut = 5           # the count at which the next unit completes
        self.pending_start: Optional[int] = None

    def push(self, item: _PrepassFrame) -> Optional[_PrepassUnit]:
        self.recent.append(item)
        self.seen += 1
        if self.seen == self.next_cut - 4:          # the next unit's first frame
            self.pending_start = item[1]
        if self.seen != self.next_cut:
            return None
        self.next_cut += self.step
        self.pending_start = None
        first = self.next_cut - 4                   # the count of the next unit's first frame
        if first <= self.seen:                      # already read: in the window just cut
            self.pending_start = self.recent[first - self.seen - 1][1]
        return _window(self.recent)

    def trailing_window(self) -> Optional[_PrepassUnit]:
        if self.seen >= 5 and self.seen != self.next_cut - self.step:
            return _window(self.recent)
        return None


def _window(frames: Deque[_PrepassFrame]) -> _PrepassUnit:
    return [value for value, _ in frames], [index for _, index in frames]



def decode_ball_predictions(
    preds: torch.Tensor,
    scale: Tuple[float, float],
    conf_threshold: float,
    output_mode: OutputMode,
    cls_filter: Optional[Sequence[int]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> List[RawDetection]:
    """Turn one engine output tensor into ``(xyxy_box, confidence)`` pairs in source pixels
    (the arguments of ``decode_engine_boxes``, minus the ball's fixed NMS threshold)."""
    rows = decode_engine_boxes(preds, scale, conf_threshold, _BALL_NMS_IOU, output_mode, cls_filter, frame_size)
    return [([float(row[0]), float(row[1]), float(row[2]), float(row[3])], float(row[4])) for row in rows]


@dataclass
class TensorRTBallPending:
    """An inference in flight on the ball engine's stream."""

    outputs: List[torch.Tensor]
    scale: Tuple[float, float]
    event: Optional[torch.cuda.Event]
    # Keeps the engine input buffer alive until the inference reading it is done.
    input_tensor: torch.Tensor
    # The source frame's (width, height), which the decoded boxes are clipped onto.
    frame_size: Tuple[int, int]


class TensorRTRuntimeBallDetector(BallDetectorBackend):
    """Single-frame ball detector for an Ultralytics-exported TensorRT engine.

    Selected when ``Config.model_path`` ends in ``.engine`` (``build_engines.py``
    builds one from ``models/ball.pt``); GridTrackNet is the default backend.
    Each frame is one inference, launched on the session's stream with
    ``cfg.trt_async_slots`` output buffers when ``cfg.trt_async_execute`` is
    set, so the next frame's preprocessing overlaps this frame's inference.
    The caller resolved the engine path, so a missing or broken engine raises.
    ``device`` defaults to ``cfg.device`` (CUDA 0 when that is not an index).
    """

    def __init__(
        self,
        engine_path: str,
        cfg: Config,
        ball_cls_id: Optional[int],
        names: Optional[Dict[int, str]] = None,
        device: Optional[torch.device] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT runtime backend requires CUDA.")
        self.cfg = cfg
        self.names = names or dict(_DEFAULT_NAMES)
        self.ball_cls_id = ball_cls_id
        self.cls_filter = [ball_cls_id] if ball_cls_id is not None else None
        if device is None:
            device = torch.device(f"cuda:{cfg.device}") if cfg.device.isdigit() else torch.device("cuda:0")
        self.session = TensorRTRuntimeSession(
            engine_path,
            device=device,
            async_execute=cfg.trt_async_execute,
            output_slots=max(1, cfg.trt_async_slots),
        )
        self.device = self.session.device
        self.use_async = self.session.use_async
        self.async_slots = self.session.output_slot_count

        probe = torch.zeros(
            (1, 3, self.session.input_h, self.session.input_w),
            device=self.device,
            dtype=torch.float16 if self.session.fp16 else torch.float32,
        )
        outputs = self.session.forward_tensor(probe, out_slot=0)
        self.session.wait()
        self.output_mode: OutputMode = engine_output_mode(outputs[0])
        print(
            f"[detector] TensorRT ball engine, {self.output_mode} "
            f"(fp16={self.session.fp16}, async={self.use_async}, slots={self.async_slots}, "
            f"device={self.device}, preprocess=gpu_resize_norm)"
        )

    def supports_cuda_frame(self) -> bool:
        return self.device.type == "cuda"

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[RawDetection]:
        return self.detect_async_finish(self.detect_async_start(frame_bgr, frame_index))

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> TensorRTBallPending:
        """Launch inference on ``frame_bgr``; ``frame_index`` is part of the
        interface and not needed here (every call runs its own frame)."""
        out_slot = self.session.next_output_slot()
        frame_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))   # HWC on either path
        t = None
        if isinstance(frame_bgr, torch.Tensor):
            t, scale = self.session.preprocess_cuda_hwc_frame(frame_bgr)
            if t is None:
                frame_bgr = frame_bgr.detach().cpu().numpy()
        if t is None or scale is None:
            t, scale = self.session.preprocess_frame(np.asarray(frame_bgr))
        outputs = self.session.forward_tensor(t, out_slot=out_slot)
        return TensorRTBallPending(outputs, scale, self.session.completion_event(), t, frame_size)

    def detect_async_finish(self, pending: TensorRTBallPending) -> List[RawDetection]:
        self.session.wait(pending.event)
        return decode_ball_predictions(
            pending.outputs[0], pending.scale, self.cfg.conf, self.output_mode, self.cls_filter,
            frame_size=pending.frame_size,
        )

