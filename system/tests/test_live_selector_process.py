"""The fixed-lag selector in its child process (``tennis_tracker.live.selector_process``) commits
what the one in line commits."""
import dataclasses
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tennis_tracker.live.evidence import RecordedEvidence, frame_evidence_of, replay
from tennis_tracker.live.fixed_lag import LagSettings
from tennis_tracker.live.selector_process import run_in_process
from tests._helpers import RALLY_FPS as FPS, RALLY_HEIGHT as HEIGHT, RALLY_WIDTH as WIDTH, rally as _rally


def test_the_selector_process_commits_what_the_selector_in_line_commits(quiet_spawn):
    detections, masks = _rally()
    ghosts = [(70, 66, (300, 40, 400, 140)), (100, 90, (0, 0, 640, 360))]
    recorded = RecordedEvidence(FPS, WIDTH, HEIGHT, detections=list(detections), boost_masks=list(masks),
                                raw_motions=list(masks), player_boxes=[{} for _ in detections],
                                court_keypoints=[None for _ in detections], ghosts=ghosts)
    settings = LagSettings(delay=8, history=30, every=4)
    inline = replay(recorded, settings)
    frames = [(frame_evidence_of(recorded, frame), [g for g in ghosts if g[0] == frame]) for frame in range(len(detections))]
    rows, output = run_in_process(frames, FPS, WIDTH, HEIGHT, settings)
    assert len(rows) == len(detections)
    assert [None if row is None else dataclasses.astuple(row) for row in rows] == \
        [None if row is None else dataclasses.astuple(row) for row in inline.committed]
    assert output.runs == inline.runs and len(output.run_seconds) == inline.runs
    assert sorted(output.tracks) == sorted(inline.tracks)


def test_a_failing_selector_process_reaches_the_tracker(quiet_spawn):
    """A child that fails (here: a frame that is not evidence) is reported, not waited on forever."""
    from tennis_tracker.live.selector_process import SelectorProcess
    process = SelectorProcess(FPS, WIDTH, HEIGHT, LagSettings(delay=2, history=4))
    try:
        process.wait_ready()
        process.add("not evidence", [])
        with pytest.raises(RuntimeError, match="selector process failed"):
            process.finish(timeout=60)
    finally:
        process.close()


_DIES_MID_STREAM = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    from tennis_tracker.live.fixed_lag import LagSettings
    from tennis_tracker.live.selector_process import SelectorProcess
    if __name__ == "__main__":
        process = SelectorProcess(60.0, 1920, 1080, LagSettings(delay=30, history=60, every=12))
        process.wait_ready()
        process._process.terminate()          # the child dies: killed, out of memory, crashed
        try:
            for _ in range(1000):
                process.add(([], None, None, {{"pad": b"x" * 60000}}, None))
                time.sleep(0.005)
        except RuntimeError as error:
            print("reported:", str(error).splitlines()[0], flush=True)
        finally:
            process.close()
""")


def test_a_tracker_whose_selector_process_died_still_exits(tmp_path):
    """The parent's queue feeder thread must not keep the interpreter from exiting once the child
    is gone with frames still in the pipe (a regression: it waited on the full pipe forever)."""
    script = tmp_path / "dies_mid_stream.py"
    script.write_text(_DIES_MID_STREAM.format(root=str(Path(__file__).resolve().parent.parent)), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "reported: the live selector process exited with code" in done.stdout
