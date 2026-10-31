"""Frames as a camera delivers them: one at a time, each stamped with when it arrived, and a stated
policy for a tracker that cannot keep up.

``LiveSource`` decodes on a thread of its own, from a video file or a camera index:

- ``realtime`` pacing hands a file's frames over at the clip's frame rate, as a camera would (a
  camera is always realtime): frame ``i`` arrives ``i / fps`` seconds after the first, and its
  arrival time is that moment, not when the tracker got round to it;
- ``fast`` pacing hands each frame over as soon as the tracker asks, with back-pressure, so nothing
  is ever dropped and the arrival time is when the frame was handed over (the throughput mode).

``FrameQueue`` holds the frames that arrived and were not taken yet. Its ``DropPolicy``:

- ``never``: every frame waits its turn, however far behind the tracker falls. The backlog is
  reported (``backlog_s``, ``max_backlog``) so a run that fell behind says so;
- ``backlog``: once more than ``max_backlog`` frames are waiting, the oldest are dropped, in
  groups of ``group`` consecutive frames (the prepass's phase count: at 60 FPS a pair, one frame of
  each phase, so each phase keeps its own spacing). Every dropped source index is recorded.

Frames are decoded into page-locked buffers where the capture allows, so they go up to the GPU
without another copy (``video_io.pinned_source``).
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

PACINGS = ("realtime", "fast")
_FAST_QUEUE_FRAMES = 4      # frames decoded ahead of the tracker in fast pacing


class DropPolicy(str, Enum):
    NEVER = "never"
    BACKLOG = "backlog"


@dataclass
class ArrivedFrame:
    """One source frame as it arrived: its zero-based source index, the BGR HWC uint8 picture and
    the ``time.perf_counter`` moment it became available."""

    index: int
    image: np.ndarray
    arrived: float


@dataclass
class FrameQueue:
    """The frames that arrived and were not taken, under a ``DropPolicy`` (see the module
    docstring). ``put`` never blocks except under ``capacity`` (fast pacing's back-pressure);
    ``get`` blocks until a frame or the end (None)."""

    policy: DropPolicy = DropPolicy.NEVER
    max_backlog: int = 0
    group: int = 1
    capacity: Optional[int] = None
    dropped: List[int] = field(default_factory=list)
    max_waiting: int = 0
    _frames: Deque[ArrivedFrame] = field(default_factory=collections.deque)
    _ended: bool = False
    _error: Optional[BaseException] = None
    _lock: threading.Condition = field(default_factory=threading.Condition)

    def __post_init__(self) -> None:
        if self.policy is DropPolicy.BACKLOG and self.max_backlog < self.group:
            raise ValueError(f"a backlog of {self.max_backlog} frames cannot hold a group of {self.group}")

    def put(self, frame: ArrivedFrame, stop: Optional[threading.Event] = None) -> bool:
        """Queue `frame`, dropping the oldest groups past the backlog; False when `stop` was set
        while waiting for room (fast pacing only)."""
        with self._lock:
            while self.capacity is not None and len(self._frames) >= self.capacity:
                if stop is not None and stop.is_set():
                    return False
                self._lock.wait(timeout=0.05)
            self._frames.append(frame)
            if self.policy is DropPolicy.BACKLOG:
                while len(self._frames) > self.max_backlog:
                    for _ in range(self.group):
                        self.dropped.append(self._frames.popleft().index)
            self.max_waiting = max(self.max_waiting, len(self._frames))
            self._lock.notify_all()
            return True

    def end(self, error: Optional[BaseException] = None) -> None:
        with self._lock:
            self._ended, self._error = True, error
            self._lock.notify_all()

    def get(self) -> Optional[ArrivedFrame]:
        """The oldest waiting frame, None once the source ended and every frame was taken; a
        source that failed raises its error then."""
        with self._lock:
            while not self._frames and not self._ended:
                self._lock.wait()
            if self._frames:
                frame = self._frames.popleft()
                self._lock.notify_all()
                return frame
            if self._error is not None:
                raise RuntimeError("the live frame source failed") from self._error
            return None

    @property
    def waiting(self) -> int:
        with self._lock:
            return len(self._frames)


def open_capture(video: Optional[str], camera: Optional[int]) -> Any:
    """The ``cv2.VideoCapture`` for a file or a camera index; one that cannot be opened is a RuntimeError."""
    capture = cv2.VideoCapture(int(camera)) if camera is not None else cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {'camera ' + str(camera) if camera is not None else video}")
    return capture


class LiveSource:
    """Decode `capture` on a thread and deliver its frames through `queue` (see the module docstring).
    ``fps`` paces a file in realtime; ``camera`` captures are realtime by nature and never paced.
    ``clock`` and ``sleep`` are the timing seams the tests replace.

    With `hold_after` the source delivers that many frames and waits for ``release()`` before the
    next: the tracker builds its per-stream state from the first frames (pass 1's frame window and
    its CUDA graphs: a quarter of a second), and a realtime file's clock starts at the release, as
    a camera would be opened once the tracker is ready."""

    def __init__(self, capture: Any, queue: FrameQueue, fps: float, pacing: str = "realtime", camera: bool = False,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 hold_after: int = 0) -> None:
        if pacing not in PACINGS:
            raise ValueError(f"pacing must be one of {PACINGS}, not {pacing!r}")
        self.capture, self.queue = capture, queue
        self.period = 1.0 / max(float(fps), 1e-6)
        self.pacing, self.camera = pacing, bool(camera)
        self.clock, self.sleep = clock, sleep
        self.started: Optional[float] = None
        self._stop = threading.Event()
        self.hold_after = int(hold_after)
        self._released = threading.Event()
        if self.hold_after <= 0:
            self._released.set()
        self._pinned: Optional[Tuple[int, int]] = None
        height, width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if height > 0 and width > 0 and torch.cuda.is_available():
            self._pinned = (height, width)
        self._thread = threading.Thread(target=self._run, name="live-source", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def release(self) -> None:
        """Deliver the frames after the first `hold_after` (see the class docstring)."""
        self._released.set()

    def stop(self) -> None:
        """Stop decoding (the frames queued so far stay queued) and release the capture."""
        self._stop.set()
        self._released.set()
        self._thread.join()
        self.capture.release()

    def _read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._pinned is None:
            return self.capture.read()
        buffer = torch.empty((*self._pinned, 3), dtype=torch.uint8, pin_memory=True).numpy()
        ok, frame = self.capture.read(buffer)
        if ok and frame is not None and frame.ctypes.data != buffer.ctypes.data:
            self._pinned = None             # the capture allocates its own frames: read them as they come
        return ok, frame

    def _run(self) -> None:
        error: Optional[BaseException] = None
        try:
            index = 0
            while not self._stop.is_set():
                if index == self.hold_after and not self._released.is_set():
                    self._released.wait()
                    if self._stop.is_set():
                        break
                    if self.started is not None:      # the clock starts again: frame `index` is due now
                        self.started = self.clock() - index * self.period
                ok, image = self._read()
                if not ok or image is None:
                    break
                now = self.clock()
                if self.started is None:
                    self.started = now
                arrived = now
                if self.pacing == "realtime" and not self.camera:
                    due = self.started + index * self.period
                    if due > now:
                        self.sleep(due - now)
                    arrived = max(due, now)
                if not self.queue.put(ArrivedFrame(index, image, arrived), self._stop):
                    break
                index += 1
        except BaseException as caught:  # noqa: BLE001 - handed to the consumer by FrameQueue.get
            error = caught
        finally:
            self.queue.end(error)


def make_queue(pacing: str, policy: DropPolicy, max_backlog: int, group: int) -> FrameQueue:
    """The queue a source of this pacing delivers into: fast pacing waits for the tracker (a few
    frames ahead, nothing dropped); realtime takes the policy."""
    if pacing == "fast":
        return FrameQueue(DropPolicy.NEVER, capacity=_FAST_QUEUE_FRAMES)
    return FrameQueue(policy, max_backlog=max_backlog, group=group)
