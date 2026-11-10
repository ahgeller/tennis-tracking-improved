"""The fixed-lag selector in a child process, so its runs never hold up pass 1.

A selector run is plain Python and NumPy (``select_ball_in_play``), tens of milliseconds over a
window of seconds. On pass 1's thread every run stalls the frames behind it; on another thread it
still shares the interpreter lock with pass 1. ``SelectorProcess`` runs ``evidence.EvidenceFeed``
in a child process instead: the parent sends each committed frame's evidence (about 60 KB at
1080p, the packed masks) and the ghosts found with it, the child appends it to its own copy and
runs the selector when the lag says so, and sends the committed rows back as they are made.

The child's feed is exactly ``evidence.replay``'s, so it commits what the selector run in line
commits (``tests/test_live_selector_process.py``). Frames come back in order; ``finish`` ends the
stream, waits for the last rows and returns the stream's tracks and run count.

The request queue is bounded (``_MAX_PENDING_FRAMES``): a selector that falls behind holds the
tracker back instead of queueing without end. The child ignores Ctrl-C, which the console sends
to every process attached to it: the tracker stops its source on Ctrl-C and still needs the child
for the final flush. A child that died or failed is reported, and ``close`` then abandons what the
queues still hold, so the tracker's exit never waits on a pipe nobody reads.
"""
from __future__ import annotations

import multiprocessing
import queue as queue_module
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ball_in_play_selector import FrameResult, Track

from .evidence import EvidenceFeed, FrameEvidence, GhostEntry
from .fixed_lag import LagSettings

_FINISH = "finish"
_FRAME = "frame"
_ROWS = "rows"
_READY = "ready"
_DONE = "done"
_FAILED = "failed"
# Frames sent and not yet taken by the child: 4 s at 60 FPS (about 15 MB of packed masks at 1080p).
_MAX_PENDING_FRAMES = 240
_PUT_POLL_S = 0.5

RowsCallback = Callable[[int, List[Optional[FrameResult]], bool], None]


@dataclass
class SelectorOutput:
    """What the child sent back at the end: the stream's tracks and how the runs went."""

    tracks: Dict[int, Track] = field(default_factory=dict)
    runs: int = 0
    run_seconds: List[float] = field(default_factory=list)


def _serve(requests: Any, replies: Any, fps: float, width: int, height: int, settings: LagSettings) -> None:
    """The child: feed every frame it is sent and send back each run's rows."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # the tracker handles Ctrl-C (module docstring)
    try:
        feed = EvidenceFeed(fps, width, height, settings, keep=False)
        spent: List[float] = []
        replies.put((_READY,))

        def send(done: List[int], started: float, final: bool = False) -> None:
            if done:
                spent.append(time.perf_counter() - started)
                replies.put((_ROWS, done[0], [feed.selector.committed[frame] for frame in done], final))

        while True:
            message = requests.get()
            if message[0] == _FINISH:
                started = time.perf_counter()
                send(feed.finish(), started, final=True)
                replies.put((_DONE, feed.selector.tracks, feed.selector.runs, spent))
                return
            _, evidence, ghosts = message
            started = time.perf_counter()
            send(feed.add(evidence, ghosts), started)
    except BaseException:                                   # the parent reports it
        replies.put((_FAILED, traceback.format_exc()))


class SelectorProcess:
    """The fixed-lag selector in a child process (module docstring). ``add`` is called per committed
    frame, in order; ``on_rows(first, rows, final)`` is called on a receiving thread as rows come
    back (`final`: the rows ``finish`` committed)."""

    def __init__(self, fps: float, width: int, height: int, settings: LagSettings,
                 on_rows: Optional[RowsCallback] = None) -> None:
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue(maxsize=_MAX_PENDING_FRAMES)
        self._replies = context.Queue()
        self._process = context.Process(target=_serve, name="live-selector", daemon=True,
                                        args=(self._requests, self._replies, fps, width, height, settings))
        self._process.start()
        self.output = SelectorOutput()
        self._on_rows = on_rows
        self._next_row = 0                  # the frame the next rows must start at
        self._error: Optional[str] = None
        self._done = threading.Event()
        self._ready = threading.Event()
        self._receiver = threading.Thread(target=self._receive, name="live-selector-rows", daemon=True)
        self._receiver.start()

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Wait until the child has started (a spawned interpreter imports the tracker: seconds),
        so the stream's first frames do not wait for it."""
        deadline = time.perf_counter() + timeout
        while not self._ready.wait(0.1):
            if self._done.is_set() or time.perf_counter() > deadline:
                self._raise()
                raise RuntimeError("the live selector process did not start")
        self._raise()

    def add(self, evidence: FrameEvidence, ghosts: Sequence[GhostEntry] = ()) -> None:
        """Send the next committed frame; waits while the child is ``_MAX_PENDING_FRAMES`` behind."""
        self._put((_FRAME, evidence, list(ghosts)))

    def finish(self, timeout: float = 120.0) -> SelectorOutput:
        """End the stream: the child commits what is left; the tracks and the runs. `timeout` is
        for the child's last run (the caller scales it with the stream when that run covers it all)."""
        self._put((_FINISH,))
        if not self._done.wait(timeout):
            raise RuntimeError("the live selector process did not finish")
        self._raise()
        self._process.join(timeout=10)
        return self.output

    def close(self) -> None:
        """Stop the child. One that did not finish cleanly (it died, failed or was given up on)
        may leave messages in either queue: they are abandoned, not flushed, so exiting does not
        wait on them."""
        if not (self._done.is_set() and self._error is None):
            self._requests.cancel_join_thread()
            self._replies.cancel_join_thread()
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=10)

    def _put(self, message: Tuple[Any, ...]) -> None:
        while True:
            self._raise()
            try:
                self._requests.put(message, timeout=_PUT_POLL_S)
                return
            except queue_module.Full:
                continue

    def _raise(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"the live selector process failed:\n{self._error}")
        if not self._done.is_set() and not self._process.is_alive() and self._process.exitcode not in (None, 0):
            raise RuntimeError(f"the live selector process exited with code {self._process.exitcode}")

    def _fail(self, error: str) -> None:
        self._error = self._error or error
        self._done.set()

    def _receive(self) -> None:
        while True:
            try:
                message = self._replies.get(timeout=0.5)
            except queue_module.Empty:
                if not self._process.is_alive() and self._replies.empty():
                    self._fail(f"the process exited with code {self._process.exitcode}")
                    return
                continue
            kind = message[0]
            if kind == _READY:
                self._ready.set()
            elif kind == _ROWS:
                _, first, rows, final = message
                if first != self._next_row:
                    self._fail(f"rows for frame {first} arrived after {self._next_row} committed")
                    return
                self._next_row += len(rows)
                if self._on_rows is not None:
                    try:
                        self._on_rows(first, rows, final)
                    except BaseException:
                        self._fail(traceback.format_exc())
                        return
            elif kind == _DONE:
                _, self.output.tracks, self.output.runs, self.output.run_seconds = message
                self._done.set()
                return
            else:
                self._fail(message[1])
                return


def run_in_process(frames: Sequence[Tuple[FrameEvidence, Sequence[GhostEntry]]], fps: float, width: int,
                   height: int, settings: LagSettings) -> Tuple[List[Optional[FrameResult]], SelectorOutput]:
    """Feed `frames` through a ``SelectorProcess`` and finish it (for tests and tools): every
    committed row in frame order, and the output."""
    rows: List[Optional[FrameResult]] = []
    process = SelectorProcess(fps, width, height, settings, on_rows=lambda _first, new, _final: rows.extend(new))
    try:
        for evidence, ghosts in frames:
            process.add(evidence, ghosts)
        return rows, process.finish()
    finally:
        process.close()
