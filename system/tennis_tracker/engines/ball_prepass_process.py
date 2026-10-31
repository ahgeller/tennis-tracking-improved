"""The ball prepass in a process of its own, beside pass 1 and from the start of init.

On a worker thread the prepass and pass 1 share one interpreter lock, and every time either
gives it up for a CUDA call or a decode it may wait to get it back (Python's GIL hands over at most
every ``sys.getswitchinterval()``, 5 ms): the measured cost was pass 1 at 84-91 fps while the
prepass ran against 146-176 fps after it (``audit/records/optimization_plan_2026-10.md``). A child
process has its own interpreter and CUDA context, so the two never wait on each other's Python, and
pass 1's CUDA graphs can be captured while the child works.

``RemotePrepass`` starts ``python -m tennis_tracker.engines.ball_prepass_process`` and hands it the
detector to build as soon as it connects, so the child loads its engines while the parent builds
its own. ``run`` hands it a clip and calls ``on_rows`` from a reader thread with each newly final
run of frames (a ``PrepassRows``), then ``on_end`` with None or the child's error.

The two talk over a ``multiprocessing.connection`` on a loopback socket with a random
authentication key; what crosses it is the setup, one ``ClipRequest`` a clip and a few floats a
frame back. The messages are dataclasses, so a field (a crop gate's state or window per frame) can
be added without touching the plumbing.

Lifecycle: the parent owns the child. ``close`` stops it, waiting for an idle child and
terminating one that is still working (a failed or interrupted run); the child ignores Ctrl-C,
which the parent handles, and exits by itself when the parent process ends, however it ends
(``_exit_with_parent``). A child that dies before it connects, or fails to build its detector,
fails the parent's ``run`` with its reason.
"""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from ball_in_play_selector import RawDetection

from ..config import Config

_AUTHKEY_ENV = "TENNIS_PREPASS_AUTHKEY"
_MODULE = "tennis_tracker.engines.ball_prepass_process"
_ROOT = Path(__file__).resolve().parents[2]
CONNECT_TIMEOUT_S = 60.0           # from the child's start to its connection: its interpreter and imports
STOP_TIMEOUT_S = 10.0              # an idle child's exit after it is told to stop
_POLL_S = 0.2

Rows = List[List[RawDetection]]


@dataclass(frozen=True)
class ClipRequest:
    """One clip for the child's prepass, as ``GridTrackNetBallDetector.prepare_video`` takes it."""
    video: str
    fps: float
    width: int
    height: int
    total: int


@dataclass(frozen=True)
class PrepassRows:
    """A newly final run of frames, ``start`` onwards: each frame's detections, as the prepass
    caches them. A per-frame field the parent should see goes here beside ``rows``."""
    start: int
    rows: Rows = field(default_factory=list)


@dataclass(frozen=True)
class PrepassEnd:
    """The end of a clip: ``error`` None when it finished, else the child's traceback."""
    error: Optional[str] = None


class RemotePrepass:
    """A prepass child process for `weights` under `cfg` (see the module docstring)."""

    def __init__(self, weights: str, cfg: Config):
        authkey = secrets.token_bytes(16)
        # A loopback socket: a Windows named pipe (the default family) refused the child's connection
        # now and then (WaitNamedPipe: file not found while the listener's instance existed).
        self._listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=authkey)
        self._address: Tuple[str, int] = self._listener.address
        self._authkey = authkey
        self._setup = (str(weights), asdict(cfg))
        host, port = self._address
        environment = dict(os.environ, **{_AUTHKEY_ENV: authkey.hex()})
        self._child = subprocess.Popen(
            [sys.executable, "-u", "-m", _MODULE, f"{host}:{port}", str(os.getpid())],
            cwd=str(_ROOT), env=environment, stdin=subprocess.DEVNULL)
        self._connection: Optional[Any] = None
        self._accept_error: Optional[BaseException] = None
        self._connected = threading.Event()
        self._closed = False
        self._reader: Optional[threading.Thread] = None
        self._acceptor = threading.Thread(target=self._accept, daemon=True, name="prepass-process-accept")
        self._acceptor.start()

    @property
    def pid(self) -> int:
        return int(self._child.pid)

    def _accept(self) -> None:
        """Take the child's connection and send it the detector to build at once."""
        try:
            connection = self._listener.accept()
            connection.send(self._setup)
            self._connection = connection
        except BaseException as error:      # noqa: BLE001 - reported by _wait_connected
            self._accept_error = error
        finally:
            self._connected.set()

    def _wait_connected(self) -> Any:
        """The connection to the child; a RuntimeError when the child exits or times out before it
        connects, or the connection failed."""
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while not self._connected.wait(_POLL_S):
            code = self._child.poll()
            if code is not None:
                raise RuntimeError(f"the prepass process exited with code {code} before it connected")
            if time.monotonic() > deadline:
                raise RuntimeError(f"the prepass process did not connect within {CONNECT_TIMEOUT_S:.0f} s")
        if self._connection is None:
            raise RuntimeError(f"the prepass process could not connect ({self._accept_error!r})")
        return self._connection

    def run(self, video_path: Path, fps: float, width: int, height: int, total: int,
            on_rows: Callable[[PrepassRows], None], on_end: Callable[[Optional[BaseException]], None]) -> None:
        """Start the child's prepass over a clip and return at once: a reader thread sends the clip
        when the child has connected, calls `on_rows` with each run of final frames and `on_end`
        with None or the error (the child's, or its loss). A clip waits for the one before it."""
        if self._closed:
            raise RuntimeError("the prepass process is closed")
        if self._reader is not None:
            self._reader.join()
        request = ClipRequest(str(video_path), float(fps), int(width), int(height), int(total))
        self._reader = threading.Thread(target=self._read, args=(request, on_rows, on_end), daemon=True,
                                        name="prepass-process-reader")
        self._reader.start()

    def _read(self, request: ClipRequest, on_rows: Callable[[PrepassRows], None],
              on_end: Callable[[Optional[BaseException]], None]) -> None:
        error: Optional[BaseException] = None
        try:
            connection = self._wait_connected()
            connection.send(request)
            while True:
                message = connection.recv()
                if isinstance(message, PrepassRows):
                    on_rows(message)
                    continue
                if isinstance(message, PrepassEnd) and message.error is not None:
                    error = RuntimeError(f"the prepass process failed:\n{message.error}")
                elif not isinstance(message, PrepassEnd):
                    error = RuntimeError(f"the prepass process sent {type(message).__name__}")
                break
        except (EOFError, OSError) as lost:
            error = RuntimeError(f"the prepass process exited ({lost!r}; exit code {self._child.poll()})")
        except BaseException as failure:  # noqa: BLE001 - a lost child or an on_rows bug still ends the prepass
            error = failure
        on_end(error)

    def close(self) -> None:
        """Stop the child and wait for it; safe to call more than once. A child that finished its
        clip is told to stop; any other (still building its detector or in a clip: the run failed or
        was interrupted) is terminated, which loses nothing (the engine and kernel caches are
        written atomically)."""
        if self._closed:
            return
        self._closed = True
        if self._reader is not None:        # a clip whose rows are all in ends a moment later
            self._reader.join(timeout=_POLL_S)
        connection = self._connection
        if connection is not None and self._reader is not None and not self._reader.is_alive():
            try:
                connection.send(None)
            except OSError:
                pass
        else:
            self._child.terminate()
        try:
            self._child.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._child.kill()
            self._child.wait()
        if not self._connected.is_set():    # wake the acceptor: the child can no longer connect
            try:
                Client(self._address, family="AF_INET", authkey=self._authkey).close()
            except OSError:
                pass
        self._acceptor.join(timeout=STOP_TIMEOUT_S)
        if self._reader is not None:
            self._reader.join(timeout=STOP_TIMEOUT_S)
        if self._connection is not None:
            self._connection.close()
        self._listener.close()


def _exit_with_parent(parent: int) -> None:
    """End this process as soon as the parent process ends, however it ends (killed, crashed): a
    daemon thread waits on the parent's handle on Windows, and polls the parent elsewhere."""
    def watch() -> None:
        if os.name == "nt":
            import ctypes
            synchronize, infinite = 0x00100000, 0xFFFFFFFF
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(synchronize, False, parent)
            if handle:              # no handle: the parent is gone already
                kernel32.WaitForSingleObject(handle, infinite)
        else:
            while os.getppid() == parent:
                time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="prepass-parent-watch").start()


def _serve(address: str, parent: int) -> None:
    """The child: build the detector the parent names, then run each clip it sends in the
    foreground, sending every newly final run of frames as it is published. A detector that cannot
    be built answers every clip with its error; a lost parent ends the process quietly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)        # the parent handles Ctrl-C and stops this process
    _exit_with_parent(parent)
    host, port = address.rsplit(":", 1)
    connection = Client((host, int(port)), family="AF_INET", authkey=bytes.fromhex(os.environ[_AUTHKEY_ENV]))
    try:
        _answer(connection)
    except (EOFError, OSError):
        return                      # the parent closed the connection or went away


def _answer(connection: Any) -> None:
    weights, fields = connection.recv()
    build_error: Optional[str] = None
    detector: Any = None
    try:
        from .ball_detectors import FOREGROUND, GridTrackNetBallDetector
        cfg = replace(Config(**fields), ball_prepass_mode=FOREGROUND)
        detector = GridTrackNetBallDetector(weights, cfg)
    except Exception:  # noqa: BLE001 - reported to the parent with each clip
        build_error = traceback.format_exc()
    sent = [0]

    def publish(upto: int) -> None:
        rows = (detector.precomputed or [])[sent[0]:upto]
        if rows:
            connection.send(PrepassRows(sent[0], rows))
            sent[0] += len(rows)

    if detector is not None:
        detector.on_publish = publish
    while True:
        request: Optional[ClipRequest] = connection.recv()
        if request is None:
            return
        if build_error is not None:
            connection.send(PrepassEnd(build_error))
            continue
        sent[0] = 0
        try:
            detector.detect_clip(Path(request.video), request.fps, request.width, request.height, request.total)
        except Exception:  # noqa: BLE001 - reported to the parent (a lost parent fails this send too)
            connection.send(PrepassEnd(traceback.format_exc()))
            continue
        connection.send(PrepassEnd())


if __name__ == "__main__":
    # Run the module's own copy, not __main__'s, so the messages pickle under the name the parent imports.
    from tennis_tracker.engines import ball_prepass_process
    ball_prepass_process._serve(sys.argv[1], int(sys.argv[2]))
