"""The live frame source (``tennis_tracker.live.source``): arrival times, pacing and the drop policy."""
import threading

import numpy as np
import pytest

from tennis_tracker.live.source import ArrivedFrame, DropPolicy, FrameQueue, LiveSource, make_queue


def _frame(index, arrived=0.0):
    return ArrivedFrame(index, np.zeros((2, 2, 3), np.uint8), arrived)


def _drain(queue):
    taken = []
    while True:
        frame = queue.get()
        if frame is None:
            return taken
        taken.append(frame.index)


def test_never_dropping_keeps_every_frame_and_reports_the_backlog():
    queue = FrameQueue(DropPolicy.NEVER)
    for index in range(50):
        queue.put(_frame(index))
    queue.end()
    assert queue.max_waiting == 50 and not queue.dropped
    assert _drain(queue) == list(range(50))


@pytest.mark.parametrize("group", [1, 2, 3])
def test_a_backlog_drops_the_oldest_frames_a_phase_group_at_a_time(group):
    """Past the budget the oldest frames go, ``group`` consecutive ones at a time (one of each
    phase), so every phase keeps its own order; each dropped index is recorded once."""
    queue = FrameQueue(DropPolicy.BACKLOG, max_backlog=6, group=group)
    for index in range(40):
        queue.put(_frame(index))
        assert queue.waiting <= 6
    queue.end()
    kept = _drain(queue)
    assert sorted(kept + queue.dropped) == list(range(40))
    assert kept == sorted(kept) and len(queue.dropped) % group == 0
    for start in range(0, len(queue.dropped), group):        # whole groups of consecutive frames
        run = queue.dropped[start:start + group]
        assert run == list(range(run[0], run[0] + group))
    assert kept[-1] == 39


def test_a_tracker_that_keeps_up_drops_nothing():
    queue = FrameQueue(DropPolicy.BACKLOG, max_backlog=4, group=2)
    for index in range(30):
        queue.put(_frame(index))
        assert queue.get().index == index
    assert not queue.dropped


def test_a_backlog_smaller_than_a_group_is_refused():
    with pytest.raises(ValueError):
        FrameQueue(DropPolicy.BACKLOG, max_backlog=1, group=2)


def test_fast_pacing_waits_for_the_tracker_instead_of_dropping():
    queue = make_queue("fast", DropPolicy.BACKLOG, max_backlog=2, group=2)
    stop = threading.Event()
    for index in range(4):
        assert queue.put(_frame(index), stop)
    stop.set()
    assert not queue.put(_frame(4), stop)                     # full, and the source was stopped
    assert [queue.get().index for _ in range(4)] == [0, 1, 2, 3] and not queue.dropped


class _Capture:
    """A ``cv2.VideoCapture`` stand-in with `count` frames and no size (so no pinned buffers)."""

    def __init__(self, count, fail_at=None):
        self.count, self.index, self.fail_at, self.released = count, 0, fail_at, False

    def get(self, _prop):
        return 0

    def read(self, image=None):
        if self.index == self.fail_at:
            raise OSError("decoder broke")
        if self.index >= self.count:
            return False, None
        self.index += 1
        return True, np.full((2, 2, 3), self.index - 1, np.uint8)

    def release(self):
        self.released = True


class _Clock:
    """A clock that only moves when slept on (and by `step` per reading)."""

    def __init__(self, step=0.0):
        self.now, self.step, self.slept = 100.0, step, []

    def __call__(self):
        self.now += self.step
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_a_realtime_file_arrives_at_its_frame_rate():
    """Frame ``i`` of a realtime file arrives ``i / fps`` after the first, however fast it decodes."""
    clock = _Clock(step=0.001)
    queue = FrameQueue(DropPolicy.NEVER)
    source = LiveSource(_Capture(10), queue, fps=50.0, pacing="realtime", clock=clock, sleep=clock.sleep)
    source.start()
    frames = [queue.get() for _ in range(10)]
    assert queue.get() is None
    source.stop()
    start = frames[0].arrived
    assert [f.index for f in frames] == list(range(10))
    assert [round(f.arrived - start, 6) for f in frames] == [round(i / 50.0, 6) for i in range(10)]
    assert len(clock.slept) == 9 and source.capture.released


def test_a_slow_decoder_arrives_late_and_never_early():
    """A frame decoded after its due time arrives when it was decoded, not when it was due."""
    clock = _Clock(step=0.05)                                  # 50 ms per reading against a 20 ms period
    queue = FrameQueue(DropPolicy.NEVER)
    source = LiveSource(_Capture(5), queue, fps=50.0, pacing="realtime", clock=clock, sleep=clock.sleep)
    source.start()
    frames = _drain_frames(queue)
    source.stop()
    gaps = np.diff([f.arrived for f in frames])
    assert np.allclose(gaps, 0.05) and not clock.slept


def _drain_frames(queue):
    frames = []
    while (frame := queue.get()) is not None:
        frames.append(frame)
    return frames


def test_a_camera_is_never_paced():
    clock = _Clock(step=0.001)
    queue = FrameQueue(DropPolicy.NEVER)
    source = LiveSource(_Capture(6), queue, fps=30.0, pacing="realtime", camera=True, clock=clock, sleep=clock.sleep)
    source.start()
    assert len(_drain_frames(queue)) == 6
    source.stop()
    assert not clock.slept


def test_a_failing_decoder_reaches_the_tracker():
    queue = FrameQueue(DropPolicy.NEVER)
    source = LiveSource(_Capture(10, fail_at=3), queue, fps=30.0, pacing="fast")
    source.start()
    assert [queue.get().index for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="live frame source failed"):
        queue.get()
    source.stop()


def test_an_unknown_pacing_is_refused():
    with pytest.raises(ValueError):
        LiveSource(_Capture(1), FrameQueue(), fps=30.0, pacing="slow")
