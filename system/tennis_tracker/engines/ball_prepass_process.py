"""The ball prepass in a process of its own, so it never holds the interpreter lock pass 1 runs on.

On a worker thread the prepass and pass 1 share one interpreter lock, and every time either
gives it up for a CUDA call or a decode it may wait to get it back (Python's GIL hands over at most
every ``sys.getswitchinterval()``, 5 ms): the measured cost was pass 1 at 84-91 fps while the
prepass ran against 146-176 fps after it (``audit/records/optimization_plan_2026-10.md``). A child
process has its own interpreter and CUDA context, so the two never wait on each other's Python.

``RemotePrepass`` starts ``python -m tennis_tracker.engines.ball_prepass_process`` (only the
detector's imports, not the tracker's), which builds the detector (the cached engine loads while
the parent builds its own engines) and waits; ``run`` hands it a clip and calls ``on_rows`` from a
reader thread with each newly final run of frames, ``(start, rows)``, then ``on_end`` with None
or the child's error. The two talk over a ``multiprocessing.connection`` pipe with a random
authentication key; what crosses it is a few floats a frame.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from ..config import Config

_AUTHKEY_ENV = "TENNIS_PREPASS_AUTHKEY"
_ROOT = Path(__file__).resolve().parents[2]


class RemotePrepass:
    """A prepass child process for `weights` under `cfg` (see the module docstring)."""

    def __init__(self, weights: str, cfg: Config):
        authkey = secrets.token_bytes(16)
        self._listener = Listener(authkey=authkey)
        environment = dict(os.environ, **{_AUTHKEY_ENV: authkey.hex()})
        self._child = subprocess.Popen(
            [sys.executable, "-u", "-m", "tennis_tracker.engines.ball_prepass_process", str(self._listener.address)],
            cwd=str(_ROOT), env=environment)
        self._weights, self._cfg = weights, cfg
        self._connection: Optional[Any] = None
        self._reader: Optional[threading.Thread] = None

    def _connect(self) -> Any:
        """The pipe to the child, sending it the detector to build on first use."""
        if self._connection is None:
            self._connection = self._listener.accept()
            self._connection.send((self._weights, asdict(self._cfg)))
        return self._connection

    def run(self, video_path: Path, fps: float, width: int, height: int, total: int,
            on_rows: Callable[[int, List[Any]], None], on_end: Callable[[Optional[BaseException]], None]) -> None:
        """Start the child's prepass over a clip; `on_rows` and `on_end` are called from a reader thread."""
        connection = self._connect()
        connection.send((str(video_path), fps, width, height, total))

        def read() -> None:
            error: Optional[BaseException] = None
            try:
                while True:
                    kind, payload = connection.recv()
                    if kind == "rows":
                        on_rows(*payload)
                    elif kind == "done":
                        break
                    else:
                        error = RuntimeError(f"the prepass process failed: {payload}")
                        break
            except (EOFError, OSError) as lost:
                error = RuntimeError(f"the prepass process exited ({lost})")
            on_end(error)

        self._reader = threading.Thread(target=read, daemon=True, name="prepass-process-reader")
        self._reader.start()

    def close(self) -> None:
        """Stop the child (it exits when its pipe closes) and wait for it."""
        if self._connection is not None:
            try:
                self._connection.send(None)
            except OSError:
                pass
            self._connection.close()
        self._listener.close()
        try:
            self._child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._child.kill()


def _serve(address: str) -> None:
    """The child: build the detector the parent names, then run each clip it sends in the foreground,
    sending every newly final run of frames as it is published."""
    from .ball_detectors import GridTrackNetBallDetector

    connection = Client(address, authkey=bytes.fromhex(os.environ[_AUTHKEY_ENV]))
    weights, fields = connection.recv()
    cfg = replace(Config(**fields), gridtracknet_prepass_process=False, gridtracknet_prepass_background=False)
    detector = GridTrackNetBallDetector(weights, cfg)
    sent = [0]

    def publish(upto: int) -> None:
        rows = (detector.precomputed or [])[sent[0]:upto]
        if rows:
            connection.send(("rows", (sent[0], rows)))
            sent[0] += len(rows)

    detector.on_publish = publish
    while True:
        try:
            request: Optional[Tuple[str, float, int, int, int]] = connection.recv()
        except EOFError:
            return
        if request is None:
            return
        video, fps, width, height, total = request
        sent[0] = 0
        try:
            detector.detect_clip(Path(video), fps, width, height, total)
        except Exception as error:  # reported to the parent, which raises it to the consumer
            cause = error.__cause__ or error
            connection.send(("failed", f"{type(cause).__name__}: {cause}"))
            continue
        connection.send(("done", None))


if __name__ == "__main__":
    _serve(sys.argv[1])
