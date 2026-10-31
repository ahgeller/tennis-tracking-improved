"""The ball prepass in a process of its own, beside pass 1 and from the start of init.

On a worker thread the prepass and pass 1 share one interpreter lock, and every time either
gives it up for a CUDA call or a decode it may wait to get it back (Python's GIL hands over at most
every ``sys.getswitchinterval()``, 5 ms): the measured cost was pass 1 at 84-91 fps while the
prepass ran against 146-176 fps after it (``audit/records/optimization_plan_2026-10.md``). A child
process has its own interpreter and CUDA context, so the two never wait on each other's Python, and
pass 1's CUDA graphs can be captured while the child works.

``RemotePrepass`` starts ``python -m tennis_tracker.engines.ball_prepass_process`` and hands it the
detector to build as soon as it connects (``PrepassSetup``), so the child loads its engines while
the parent builds its own. ``run`` hands it a clip (``ClipRequest``) and calls ``on_rows`` from a
reader thread with each newly final run of frames (``PrepassRows``), then ``on_end`` with None or
the child's error (from ``PrepassEnd``).

The two talk over a ``multiprocessing.connection`` on a loopback socket with a random
authentication key; what crosses it is the setup, one ``ClipRequest`` a clip and a few floats a
frame back. The messages are dataclasses, so a field (a crop gate's state or window per frame) can
be added without touching the plumbing. The parent accepts on a socket of its own and runs the
key's handshake under a timeout, so a stray client on the port (a port scan, a client that never
speaks) is dropped and the child's connection still taken.

Lifecycle: the parent owns the child. ``stop`` tells an idle child to exit (``StopRequest``) and
terminates one that is still working (a failed or interrupted run); ``close`` also waits for it.
The child ignores Ctrl-C, which the parent handles, and exits by itself when the parent process
ends, however it ends (``_exit_with_parent``). A child that dies before it connects, or fails to
build its detector, fails the parent's ``run`` with its reason.
"""
from __future__ import annotations

import os
import secrets
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection, answer_challenge, deliver_challenge
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ball_in_play_selector import RawDetection

from ..config import FOREGROUND, Config

_AUTHKEY_ENV = "TENNIS_PREPASS_AUTHKEY"
_MODULE = "tennis_tracker.engines.ball_prepass_process"
_ROOT = Path(__file__).resolve().parents[2]
CONNECT_TIMEOUT_S = 60.0           # from the child's start to its connection: its interpreter and imports
HANDSHAKE_TIMEOUT_S = 2.0          # a connecting client's key handshake; a silent stray is dropped after it
STOP_TIMEOUT_S = 10.0              # an idle child's exit after it is told to stop
_POLL_S = 0.2
_HEADER = struct.Struct("!i")      # multiprocessing.connection's message framing

Rows = List[List[RawDetection]]


@dataclass(frozen=True)
class PrepassSetup:
    """The detector the child builds: the weights and the parent's ``Config`` as a dict."""
    weights: str
    config: Dict[str, Any]


@dataclass(frozen=True)
class ClipRequest:
    """One clip for the child's prepass, as ``GridTrackNetBallDetector.prepare_video`` takes it."""
    video: str
    fps: float
    width: int
    height: int
    total: int


@dataclass(frozen=True)
class StopRequest:
    """Tells an idle child to exit."""


@dataclass(frozen=True)
class PrepassRows:
    """A newly final run of frames, ``start`` onwards: each frame's detections, as the prepass
    caches them. A per-frame field the parent should see goes here beside ``rows``."""
    start: int
    rows: Rows


@dataclass(frozen=True)
class PrepassEnd:
    """The end of a clip: ``error`` None when it finished, else the child's traceback."""
    error: Optional[str] = None


class _Framed:
    """A socket under a timeout, framed as ``multiprocessing.connection`` frames messages: what the
    key handshake (``deliver_challenge``, ``answer_challenge``) needs before the socket becomes a
    ``Connection``, whose own reads cannot time out."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send_bytes(self, data: bytes) -> None:
        self.sock.sendall(_HEADER.pack(len(data)) + bytes(data))

    def recv_bytes(self, maxlength: Optional[int] = None) -> bytes:
        size, = _HEADER.unpack(self._exactly(_HEADER.size))
        if size < 0 or (maxlength is not None and size > maxlength):
            raise OSError(f"bad message length {size}")
        return self._exactly(size)

    def _exactly(self, size: int) -> bytes:
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise EOFError("the client closed the connection")
            data += chunk
        return data


class RemotePrepass:
    """A prepass child process for `weights` under `cfg` (see the module docstring)."""

    def __init__(self, weights: str, cfg: Config):
        authkey = secrets.token_bytes(16)
        # A loopback socket: a Windows named pipe (the default family) refused the child's connection
        # now and then (WaitNamedPipe: file not found while the listener's instance existed).
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(_POLL_S)
        self._address: Tuple[str, int] = self._server.getsockname()[:2]
        self._authkey = authkey
        self._setup = PrepassSetup(str(weights), asdict(cfg))
        host, port = self._address
        environment = dict(os.environ, **{_AUTHKEY_ENV: authkey.hex()})
        self._child = subprocess.Popen(
            [sys.executable, "-u", "-m", _MODULE, f"{host}:{port}", str(os.getpid())],
            cwd=str(_ROOT), env=environment, stdin=subprocess.DEVNULL)
        self._connection: Optional[Connection] = None
        self._accept_error: Optional[BaseException] = None
        self._connected = threading.Event()        # set when the acceptor is done, connected or not
        self._closing = threading.Event()
        self._stopped = False
        self._closed = False
        self._reader: Optional[threading.Thread] = None
        self._acceptor = threading.Thread(target=self._accept, daemon=True, name="prepass-process-accept")
        self._acceptor.start()

    @property
    def pid(self) -> int:
        return int(self._child.pid)

    def _accept(self) -> None:
        """Take the child's connection and send it the detector to build at once; a client that
        fails the handshake is dropped and the next one taken, until the child connects, exits or
        ``close`` is called."""
        try:
            while not self._closing.is_set() and self._child.poll() is None:
                try:
                    sock, _ = self._server.accept()
                except socket.timeout:
                    continue
                connection = self._handshake(sock)
                if connection is not None:
                    connection.send(self._setup)
                    self._connection = connection
                    return
        except BaseException as error:      # noqa: BLE001 - reported by _wait_connected
            self._accept_error = error
        finally:
            self._connected.set()

    def _handshake(self, sock: socket.socket) -> Optional[Connection]:
        """The key handshake with a new client, as ``Listener.accept`` does it but under
        ``HANDSHAKE_TIMEOUT_S``: the connection, or None for a client without the key."""
        sock.settimeout(HANDSHAKE_TIMEOUT_S)
        framed = _Framed(sock)
        try:
            deliver_challenge(framed, self._authkey)        # type: ignore[arg-type]
            answer_challenge(framed, self._authkey)         # type: ignore[arg-type]
        except (AuthenticationError, EOFError, OSError):    # socket.timeout is an OSError
            sock.close()
            return None
        sock.settimeout(None)
        return Connection(sock.detach())

    def _wait_connected(self) -> Connection:
        """The connection to the child; a RuntimeError when the child exits or times out before it
        connects, the prepass is closed, or accepting failed."""
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while not self._connected.wait(_POLL_S):
            if time.monotonic() > deadline:
                raise RuntimeError(f"the prepass process did not connect within {CONNECT_TIMEOUT_S:.0f} s")
        if self._connection is not None:
            return self._connection
        code = self._child.poll()
        if code is not None:
            raise RuntimeError(f"the prepass process exited with code {code} before it connected")
        if self._closing.is_set():
            raise RuntimeError("the prepass process was closed before it connected")
        raise RuntimeError(f"the prepass process could not connect ({self._accept_error!r})")

    def run(self, video_path: Path, fps: float, width: int, height: int, total: int,
            on_rows: Callable[[PrepassRows], None], on_end: Callable[[Optional[BaseException]], None]) -> None:
        """Start the child's prepass over a clip and return at once: a reader thread sends the clip
        when the child has connected, calls `on_rows` with each run of final frames and `on_end`
        with None or the error (the child's, or its loss). A clip waits for the one before it."""
        if self._stopped:
            raise RuntimeError("the prepass process is stopped")
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
            message = connection.recv()
            while isinstance(message, PrepassRows):
                on_rows(message)
                message = connection.recv()
            if not isinstance(message, PrepassEnd):
                error = RuntimeError(f"the prepass process sent {type(message).__name__}")
            elif message.error is not None:
                error = RuntimeError(f"the prepass process failed:\n{message.error}")
        except (EOFError, OSError) as lost:
            error = RuntimeError(f"the prepass process exited ({lost!r}; exit code {self._child.poll()})")
        except BaseException as failure:  # noqa: BLE001 - a lost child or an on_rows bug still ends the prepass
            error = failure
        on_end(error)

    def stop(self) -> None:
        """Tell the child to end without waiting for it; safe to call more than once, and ``close``
        waits. A child that finished its clip is told to stop, so it exits while the parent goes on
        (its CUDA teardown takes most of a second); any other (still building its detector or in a
        clip: the run failed or was interrupted) is terminated, which loses nothing (the engine and
        kernel caches are written atomically)."""
        if self._stopped:
            return
        self._stopped = True
        self._closing.set()                 # an acceptor still waiting for the child gives up
        if self._reader is not None:        # a clip whose rows are all in ends a moment later
            self._reader.join(timeout=_POLL_S)
        connection = self._connection
        if connection is not None and self._reader is not None and not self._reader.is_alive():
            try:
                connection.send(StopRequest())
            except OSError:
                pass
        else:
            self._child.terminate()

    def close(self) -> None:
        """Stop the child (``stop``) and wait for it to exit; safe to call more than once. Bounded:
        the acceptor polls, so nothing here waits on a client."""
        if self._closed:
            return
        self.stop()
        self._closed = True
        try:
            self._child.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._child.kill()
            self._child.wait()
        self._acceptor.join(timeout=STOP_TIMEOUT_S)
        if self._reader is not None:
            self._reader.join(timeout=STOP_TIMEOUT_S)
        if self._connection is not None:
            self._connection.close()
        self._server.close()


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
    """The child's side of the protocol over `connection` (what ``Client`` returns)."""
    setup = connection.recv()
    if not isinstance(setup, PrepassSetup):
        raise OSError(f"the parent sent {type(setup).__name__} before the setup")
    build_error: Optional[str] = None
    detector: Any = None
    try:
        from .ball_detectors import GridTrackNetBallDetector
        cfg = replace(Config(**setup.config), ball_prepass_mode=FOREGROUND)
        detector = GridTrackNetBallDetector(setup.weights, cfg)
    except Exception:  # noqa: BLE001 - reported to the parent with each clip
        build_error = traceback.format_exc()
    sent = 0

    def publish(upto: int) -> None:
        nonlocal sent
        rows = (detector.precomputed or [])[sent:upto]
        if rows:
            connection.send(PrepassRows(sent, rows))
            sent += len(rows)

    if detector is not None:
        detector.on_publish = publish
    while True:
        request = connection.recv()
        if isinstance(request, StopRequest):
            return
        if not isinstance(request, ClipRequest):
            connection.send(PrepassEnd(f"the parent sent {type(request).__name__}, not a clip"))
            continue
        if build_error is not None:
            connection.send(PrepassEnd(build_error))
            continue
        sent = 0
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
