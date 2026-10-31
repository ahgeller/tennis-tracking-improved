"""The fixed-lag selector in its child process (``tennis_tracker.live.selector_process``) commits
what the one in line commits."""
import dataclasses
import faulthandler
import sys

import pytest

from tennis_tracker.live.evidence import RecordedEvidence, frame_evidence, replay
from tennis_tracker.live.fixed_lag import LagSettings
from tennis_tracker.live.selector_process import run_in_process
from tests.test_live_fixed_lag import FPS, HEIGHT, WIDTH, _rally


@pytest.fixture
def quiet_spawn():
    """Windows' CreateProcess raises and handles an access violation inside the call, which
    pytest's fault handler would print as a crash; it is off while the child starts."""
    enabled = faulthandler.is_enabled()
    if enabled and sys.platform == "win32":
        faulthandler.disable()
    yield
    if enabled and sys.platform == "win32":
        faulthandler.enable()


def test_the_selector_process_commits_what_the_selector_in_line_commits(quiet_spawn):
    detections, masks = _rally()
    ghosts = [(70, 66, (300, 40, 400, 140)), (100, 90, (0, 0, 640, 360))]
    recorded = RecordedEvidence(FPS, WIDTH, HEIGHT, detections=list(detections), boost_masks=list(masks),
                                raw_motions=list(masks), player_boxes=[{} for _ in detections],
                                court_keypoints=[None for _ in detections], ghosts=ghosts)
    settings = LagSettings(delay=8, history=30, every=4)
    inline = replay(recorded, settings)
    frames = [(frame_evidence(recorded, frame), [g for g in ghosts if g[0] == frame]) for frame in range(len(detections))]
    output = run_in_process(frames, FPS, WIDTH, HEIGHT, settings)
    assert len(output.committed) == len(detections) == len(output.received_at)
    assert [None if row is None else dataclasses.astuple(row) for row in output.committed] == \
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
