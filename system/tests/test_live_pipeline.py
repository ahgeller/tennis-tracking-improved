"""The live run's own logic (``tennis_tracker.live.pipeline``), without the GPU: dropped frames are
source frames to the selector, the Phase C seam, Ctrl-C's stop and the options' checks."""
import contextlib
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ball_in_play_selector import PackedMask

from tennis_tracker.live.evidence import EMPTY_EVIDENCE, replay
from tennis_tracker.live.fixed_lag import LagSettings
from tennis_tracker.live.pipeline import LiveOptions, LiveStats, _LiveRun, _Pump, _stop_on_interrupt, camera_name
from tennis_tracker.live.source import ArrivedFrame, DropPolicy, FrameQueue
from tennis_tracker.motion import BoostMasks
from tennis_tracker.pass1 import Pass1Result
from tests._helpers import RALLY_FPS, RALLY_HEIGHT, RALLY_WIDTH, rally, same_row

DROPPED = {40, 41, 90, 91, 92, 93}


class _Pending:
    """A boost mask pass 1 left unfiltered (a ``motion.PendingBoost``): resolves to `mask`, and
    records the processed frame it was resolved for in `resolved`."""

    def __init__(self, processed, mask, resolved):
        self.processed, self.mask, self.resolved = processed, mask, resolved
        self.nbytes = int(mask.bits.nbytes)

    def resolve(self):
        self.resolved.append(self.processed)
        return self.mask


def _run_with_drops(selector_process=False, resolved=None):
    """A ``_LiveRun`` fed as pass 1 feeds it, with the source frames in ``DROPPED`` never seen:
    pass 1 numbers the others 0, 1, 2... A ghost found as processed frame 60 was committed lies on
    processed frame 55. With a `resolved` list, pass 1's boost masks are stored as pass 1 stores
    them, unfiltered (``BoostMasks.defer``), and each records its processed frame there when it is
    filtered."""
    detections, masks = rally()
    seen = [frame for frame in range(len(detections)) if frame not in DROPPED]
    boost_masks = [masks[f] for f in seen]
    if resolved is not None:
        boost_masks = BoostMasks()
        for processed, f in enumerate(seen):
            if masks[f] is None:
                boost_masks.append(None)
            else:
                boost_masks.defer(_Pending(processed, masks[f], resolved))
    result = Pass1Result(detections=[detections[f] for f in seen], boost_masks=boost_masks,
                         raw_motions=[masks[f] for f in seen], player_boxes=[{} for _ in seen],
                         court_keypoints=[None for _ in seen])
    pump = SimpleNamespace(source_index=seen, arrivals=[100.0 + f / RALLY_FPS for f in seen],
                           queue=FrameQueue(DropPolicy.NEVER))
    ctx = SimpleNamespace(video=SimpleNamespace(fps=RALLY_FPS, width=RALLY_WIDTH, height=RALLY_HEIGHT))
    settings = LagSettings(delay=8, history=30, every=4)
    stats = LiveStats(fps=RALLY_FPS, delay_frames=8, every_frames=4, history_frames=30)
    run = _LiveRun(ctx, pump, LiveOptions(record_evidence=True, report_every_s=0.0, selector_process=selector_process),
                   stats, settings)
    run.attach(result)
    ghost = (300, 40, 400, 140)
    for processed in range(len(seen)):
        run.committed_frame(processed, [(55, ghost)] if processed == 60 else [])
    run.finish()
    run.close()
    return run, seen, result, settings, ghost


def test_after_a_drop_the_selector_and_the_record_see_source_frames():
    run, seen, result, settings, ghost = _run_with_drops()
    recorded, total = run.recorded, len(seen) + len(DROPPED)
    assert len(run.committed) == len(recorded.detections) == run.fed == total
    assert recorded.source_index == seen
    for frame in DROPPED:
        evidence = (recorded.detections[frame], recorded.boost_masks[frame], recorded.raw_motions[frame],
                    recorded.player_boxes[frame], recorded.court_keypoints[frame])
        assert evidence == EMPTY_EVIDENCE and run.arrivals[frame] is None
    assert recorded.ghosts == [(seen[60], seen[55], ghost)]          # remapped to source frames
    assert run.stats.evidence_frames == seen
    assert run.stats.report()["slowest_evidence"][0]["frame"] in seen
    # The run commits what the selector fed the record commits: a replay is the run.
    replayed = replay(recorded, settings)
    assert all(same_row(a, b) for a, b in zip(run.committed, replayed.committed))
    for track in run.tracks.values():
        assert all(obs.frame < total and obs.frame not in DROPPED for obs in track.observations)
    # Latencies are for frames that arrived; pass 1 kept none of its masks.
    assert len(run.stats.latencies_s) + run.stats.flushed_at_end == len(seen)
    assert all(mask is None for mask in result.boost_masks + result.raw_motions)


def test_committing_a_frame_hands_on_its_boost_mask_filtered_once_and_keeps_none(quiet_spawn):
    """Pass 1 stores each boost mask unfiltered until it is first read: committing a frame hands
    the selector and the record its filtered mask, filtered once, and keeps nothing behind."""
    eager, *_ = _run_with_drops()
    for selector_process in (False, True):
        resolved = []
        run, seen, result, _, _ = _run_with_drops(selector_process=selector_process, resolved=resolved)
        filled = [processed for processed, f in enumerate(seen) if rally()[1][f] is not None]
        assert resolved == filled
        assert all(mask is None or isinstance(mask, PackedMask) for mask in run.recorded.boost_masks)
        assert all((a is None and b is None) or np.array_equal(a.unpack(), b.unpack())
                   for a, b in zip(run.recorded.boost_masks, eager.recorded.boost_masks, strict=True))
        assert all(same_row(a, b) for a, b in zip(run.committed, eager.committed))
        assert all(mask is None for mask in result.boost_masks)


def test_after_a_drop_the_selector_process_commits_what_the_selector_in_line_does(quiet_spawn):
    inline, *_ = _run_with_drops(selector_process=False)
    child, *_ = _run_with_drops(selector_process=True)
    assert len(child.committed) == len(inline.committed)
    assert all(same_row(a, b) for a, b in zip(child.committed, inline.committed))


class _Prepass:
    lookahead_frames = 4

    def __init__(self):
        self.pushed = []

    def push(self, image):
        self.pushed.append(image)

    def finish(self):
        pass


class _Detector:
    device = SimpleNamespace(type="cpu")

    def __init__(self):
        self.prepass = _Prepass()

    def open_stream(self, fps, width, height):
        return self.prepass

    def stream_scope(self):
        return contextlib.nullcontext()


def _queue_of(count):
    queue = FrameQueue(DropPolicy.NEVER)
    for index in range(count):
        queue.put(ArrivedFrame(index, np.full((2, 2, 3), index, np.uint8), float(index)))
    queue.end()
    return queue


def test_a_region_gate_sees_every_frame_before_the_ball_model_does():
    asked = []
    detector = _Detector()
    pump = _Pump(_queue_of(3), detector, 30.0, 2, 2, region_for=lambda frame: asked.append(frame.index))
    while pump.pull():
        pass
    assert asked == [0, 1, 2] and [int(image[0, 0, 0]) for image in detector.prepass.pushed] == [0, 1, 2]
    assert pump.source_index == [0, 1, 2]


def test_a_region_is_refused_until_the_prepass_can_take_one():
    pump = _Pump(_queue_of(1), _Detector(), 30.0, 2, 2, region_for=lambda frame: (0, 0, 1, 1))
    with pytest.raises(NotImplementedError, match="Phase C"):
        pump.pull()


class _Source:
    def __init__(self):
        self.interrupted = False

    def request_stop(self):
        self.interrupted = True


def test_ctrl_c_stops_the_source_and_a_second_aborts():
    source = _Source()
    before = signal.getsignal(signal.SIGINT)
    with _stop_on_interrupt(source):
        signal.raise_signal(signal.SIGINT)
        assert source.interrupted                 # the run carries on and finishes what is in
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is before


def test_a_camera_cannot_be_paced_fast_and_is_named_as_the_input():
    with pytest.raises(ValueError, match="camera"):
        LiveOptions(camera=0, pacing="fast")
    assert LiveOptions(camera=0).pacing == "realtime"
    assert camera_name(1) == "camera:1"


def test_a_live_config_runs_the_prepass_in_the_foreground_and_writes_no_video():
    """Live feeds the prepass frame by frame on the tracking thread, whatever the config asked for
    (a streaming network's AUTO is a prepass process), so pass 1's CUDA graphs stay on."""
    from tennis_tracker.config import FOREGROUND, PROCESS, Config
    from tennis_tracker.engines.ball_detectors import prepass_mode
    from tennis_tracker.live.pipeline import live_config
    from tennis_tracker.pass1 import use_cuda_graphs
    asked = Config(ball_prepass_mode=PROCESS, save_tracking_video=True, save_motion_debug=True,
                   save_detector_input_debug=True, save_track_debug_video=True, motion_diagnostics=True)
    cfg = live_config(asked)
    assert cfg.ball_prepass_mode == FOREGROUND == prepass_mode(cfg, Path("stream.pt"))   # the weights are not read
    assert not (cfg.save_tracking_video or cfg.save_motion_debug or cfg.save_detector_input_debug
                or cfg.save_track_debug_video or cfg.motion_diagnostics)
    assert use_cuda_graphs(cfg, SimpleNamespace(prepass_mode=FOREGROUND)) == bool(cfg.pass1_cuda_graphs)
    assert (cfg.conf, cfg.model_path) == (asked.conf, asked.model_path)
