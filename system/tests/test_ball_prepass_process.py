"""The prepass process: where ``Config.ball_prepass_mode`` puts the prepass, and a real child's
lifecycle (its errors reach the parent, ``stop`` and ``close`` end it, and it never outlives the parent).

The lifecycle tests start real child processes, which import torch: a few seconds each. The
child fails to build its detector on purpose (weights that do not exist), so no GPU is needed;
the one test that compares a real prepass in the child with the in-process one needs CUDA and the
stream_s4 weights."""
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pytest
import torch

from tennis_tracker import paths
from tennis_tracker.ball_models import HEATMAP_S4, STREAM_S4
from tennis_tracker.config import AUTO, FOREGROUND, PROCESS, THREAD, Config
from tennis_tracker.engines import ball_prepass_process
from tennis_tracker.engines.ball_detectors import GridTrackNetBallDetector, prepass_mode
from tennis_tracker.engines.ball_prepass_process import RemotePrepass

ROOT = Path(__file__).resolve().parents[1]
CHILD_EXIT_S = 30.0


def _record(tmp_path: Path, arch: str) -> Path:
    path = tmp_path / f"{arch}.pt"
    torch.save({"arch": arch, "state_dict": {}}, path)
    return path


@pytest.mark.parametrize("arch, streaming, expected", [
    (STREAM_S4, True, PROCESS),             # a network that streams: its prepass beside pass 1
    (STREAM_S4, False, FOREGROUND),         # ... unless it runs unit by unit
    (HEATMAP_S4, True, FOREGROUND),         # the windowed networks: before pass 1
])
def test_auto_puts_a_streaming_networks_prepass_in_a_process_and_any_other_in_the_foreground(
        tmp_path, arch, streaming, expected):
    cfg = Config(ball_prepass_mode=AUTO, ball_prepass_streaming=streaming)
    assert prepass_mode(cfg, _record(tmp_path, arch)) == expected


def test_auto_reads_an_npz_as_gridtracknet_without_opening_it(tmp_path):
    assert prepass_mode(Config(), tmp_path / "missing.npz") == FOREGROUND


@pytest.mark.parametrize("mode", [FOREGROUND, THREAD, PROCESS])
def test_an_explicit_mode_is_kept_for_any_network(tmp_path, mode):
    assert prepass_mode(Config(ball_prepass_mode=mode), _record(tmp_path, HEATMAP_S4)) == mode


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ball_prepass_mode"):
        prepass_mode(Config(ball_prepass_mode="background"), _record(tmp_path, HEATMAP_S4))


def test_the_default_detector_runs_its_prepass_in_a_process():
    if not paths.BALL_WEIGHTS.is_file():
        pytest.skip("the default stream_s4 weights are not here")
    assert prepass_mode(Config(), paths.BALL_WEIGHTS) == PROCESS


# --- a real child --------------------------------------------------------------------------


def _gone(pid: int, within: float = CHILD_EXIT_S) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:        # gone, also between a check and the next
            return True
        time.sleep(0.1)
    return False


class Ends:
    """``run``'s callbacks: the rows it was given and how it ended."""

    def __init__(self):
        self.rows, self.error, self.ended = [], None, threading.Event()

    def on_rows(self, message):
        self.rows.append(message)

    def on_end(self, error):
        self.error = error
        self.ended.set()


def test_a_child_that_cannot_build_its_detector_reports_why_and_stops_when_closed(tmp_path):
    remote = RemotePrepass(str(tmp_path / "missing.pt"), Config(device="0"))
    ends = Ends()
    remote.run(tmp_path / "clip.mp4", 30.0, 64, 48, 10, ends.on_rows, ends.on_end)
    assert ends.ended.wait(CHILD_EXIT_S)
    assert isinstance(ends.error, RuntimeError) and "the prepass process failed" in str(ends.error)
    assert "missing.pt" in str(ends.error) and "Traceback" in str(ends.error)     # the child's own traceback
    assert ends.rows == []
    second = Ends()                                  # the child is still there and answers a second clip
    remote.run(tmp_path / "clip.mp4", 30.0, 64, 48, 10, second.on_rows, second.on_end)
    assert second.ended.wait(CHILD_EXIT_S) and "missing.pt" in str(second.error)
    pid = remote.pid
    remote.stop()                                                # returns at once; the child exits by itself
    remote.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        remote.run(tmp_path / "clip.mp4", 30.0, 64, 48, 10, ends.on_rows, ends.on_end)
    assert _gone(pid) and remote._child.wait(CHILD_EXIT_S) == 0
    remote.close()
    remote.close()                                               # a second close does nothing
    assert not remote._acceptor.is_alive()


def test_a_child_that_dies_before_it_connects_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ball_prepass_process, "_MODULE", "tennis_tracker.no_such_module")
    remote = RemotePrepass(str(tmp_path / "missing.pt"), Config(device="0"))
    ends = Ends()
    remote.run(tmp_path / "clip.mp4", 30.0, 64, 48, 10, ends.on_rows, ends.on_end)
    assert ends.ended.wait(CHILD_EXIT_S)
    assert "exited with code 1 before it connected" in str(ends.error)
    remote.close()
    assert not remote._acceptor.is_alive()


def test_closing_a_child_that_has_not_finished_terminates_it_at_once(tmp_path):
    """Closed before its clip ends (an error or Ctrl-C in the parent), or before any clip: the child
    is terminated rather than waited for."""
    remote = RemotePrepass(str(tmp_path / "missing.pt"), Config(device="0"))
    pid = remote.pid
    started = time.monotonic()
    remote.close()
    assert time.monotonic() - started < ball_prepass_process.STOP_TIMEOUT_S and _gone(pid, 5.0)
    assert not remote._acceptor.is_alive()


def test_the_child_ignores_ctrl_c_and_leaves_it_to_the_parent():
    """The console sends Ctrl-C to every process in it; the child's handler is SIG_IGN, so it does
    not die with a traceback of its own while the parent stops it."""
    code = textwrap.dedent("""
        import signal
        from tennis_tracker.engines import ball_prepass_process as bpp

        def refuse(*args, **kwargs):
            raise ConnectionRefusedError

        bpp._exit_with_parent = lambda parent: None
        bpp.Client = refuse
        try:
            bpp._serve("127.0.0.1:1", 0)
        except ConnectionRefusedError:
            pass
        print(signal.getsignal(signal.SIGINT) is signal.SIG_IGN)
    """)
    environment = dict(os.environ, TENNIS_PREPASS_AUTHKEY="00")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True,
                          timeout=60)
    assert done.stdout.strip() == "True", done.stderr


def test_the_child_exits_when_its_parent_is_killed(tmp_path):
    """A parent killed outright never closes its prepass: the child sees it go and exits."""
    code = textwrap.dedent(f"""
        import sys, time
        from tennis_tracker.config import Config
        from tennis_tracker.engines.ball_prepass_process import RemotePrepass
        remote = RemotePrepass({str(tmp_path / 'missing.pt')!r}, Config(device="0"))
        print(remote.pid, flush=True)
        time.sleep(600)
    """)
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        child = int(parent.stdout.readline())
        assert psutil.pid_exists(child)
        parent.kill()
        parent.wait()
        assert _gone(child)
    finally:
        parent.kill()


def _parent_script(body: str, timeout: float = 90.0) -> str:
    """Run `body` in a fresh interpreter (a RemotePrepass over weights that do not exist, as
    `remote`, and `tmp` its folder) and return what it printed; a hang fails at `timeout`."""
    code = textwrap.dedent("""
        import socket, sys, tempfile, threading, time
        from pathlib import Path
        from tennis_tracker.config import Config
        from tennis_tracker.engines import ball_prepass_process as bpp
        tmp = Path(tempfile.mkdtemp())
        ended, box = threading.Event(), {}

        def on_end(error):
            box["error"] = error
            ended.set()

        def closes_within(seconds):
            done = threading.Event()
            threading.Thread(target=lambda: (remote.close(), done.set()), daemon=True).start()
            return done.wait(seconds)
    """) + textwrap.dedent(body)
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.parametrize("stray", ["leaves at once", "stays silent"])
def test_a_stray_client_on_the_port_does_not_keep_the_child_from_connecting(stray):
    """A port scan (connect and go) or a client that never speaks is dropped after the handshake
    fails or times out; the child still connects, its answer arrives, and close returns."""
    out = _parent_script(f"""
        remote = bpp.RemotePrepass(str(tmp / "missing.pt"), Config(device="0"))
        stray = socket.create_connection(remote._address)
        if {stray == "leaves at once"!r}:
            stray.close()
        remote.run(tmp / "clip.mp4", 30.0, 64, 48, 10, lambda message: None, on_end)
        print("ended", ended.wait(60), "missing.pt" in str(box.get("error")))
        print("closed", closes_within(20))
    """)
    assert "ended True True" in out and "closed True" in out


def test_close_returns_while_a_silent_client_holds_the_handshake_and_the_child_never_connects():
    """The child exits without connecting while a stray holds the acceptor in its handshake: close
    still returns, since nothing in it waits on a client."""
    out = _parent_script("""
        bpp._MODULE = "this"                    # a child that prints and exits without connecting
        remote = bpp.RemotePrepass(str(tmp / "missing.pt"), Config(device="0"))
        stray = socket.create_connection(remote._address)
        time.sleep(1.0)
        started = time.monotonic()
        print("closed", closes_within(20), "acceptor", remote._acceptor.is_alive())
        remote2 = bpp.RemotePrepass(str(tmp / "missing.pt"), Config(device="0"))
        remote2.run(tmp / "clip.mp4", 30.0, 64, 48, 10, lambda message: None, on_end)
        print("ended", ended.wait(30), "before it connected" in str(box.get("error")))
        remote2.close()
    """)
    assert "closed True acceptor False" in out and "ended True True" in out


# --- the same answers ----------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available() or not paths.STREAM_WEIGHTS.is_file(),
                    reason="runs the stream_s4 prepass on the GPU")
def test_the_prepass_in_the_child_gives_the_in_process_answers_frame_for_frame(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (320, 180))
    rng = np.random.default_rng(5)
    background = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    for index in range(40):
        frame = background.copy()
        cv2.circle(frame, (20 + 7 * index, 40 + 2 * index), 3, (240, 255, 255), -1)
        writer.write(frame)
    writer.release()
    answers = {}
    for mode in (FOREGROUND, PROCESS):
        cfg = Config(device="0", conf=0.0, ball_prepass_mode=mode, ball_prepass_gpu_decode=False)
        detector = GridTrackNetBallDetector(str(paths.STREAM_WEIGHTS), cfg)
        try:
            answers[mode] = detector.detect_clip(path, 60.0, 320, 180, 40)
        finally:
            detector.close()
    assert len(answers[FOREGROUND]) == 40 and any(answers[FOREGROUND])
    assert answers[PROCESS] == answers[FOREGROUND]
