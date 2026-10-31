"""The live tracker: frames arrive one at a time and each frame's ball output is committed a fixed
number of frames later. Nothing waits for the whole clip.

``run_live(cfg, options)`` drives the offline stages, unchanged, one frame at a time:

1. ``LiveSource`` decodes each frame once, on its own thread, and stamps when it arrived
   (``source``: realtime or as fast as possible, a file or a camera, and the drop policy).
2. ``_Pump`` takes it, pushes it into the ball prepass (``GridTrackNetBallDetector.open_stream``:
   the same engines, ``TransitionRing`` and heat ring, each unit run as soon as it is cut) and
   holds it for pass 1. A frame's detections are final ``PrepassStream.lookahead_frames`` frames
   after it (8 at 60 FPS, 4 at 30 FPS).
3. Pass 1 (``pass1.Pass1Collector``, the offline loop itself) trails the prepass: it pulls frames
   and detections on demand, and both pull the pump, so it runs as far behind the newest frame
   as the prepass's lookahead and the motion gate's next frame need, and no farther.
4. Each frame pass 1 commits (``Pass1Collector.committed``) goes to the ``FixedLagSelector``,
   which commits every frame's output ``delay`` frames after it (``fixed_lag``), in its own process
   by default (``selector_process``).

**Frame indices.** The prepass and pass 1 see the frames that were not dropped, numbered as they
were processed: their motion, ROI tracking and the prepass's phases run on that sequence, so across
a drop they see two frames further apart in time than a frame period. The selector, the evidence
record and the output see source frames: a dropped frame is fed to the selector as
``evidence.EMPTY_EVIDENCE`` and is a row with no ball in the JSON, and the ghost ROIs and the
latency report are given by source frame.

The tracking JSON is the offline one (``export.write_tracking_json``). Its ``tracks`` are the
stream tracks that own a committed row, each as the last selector run that committed one of its
rows judged it (so its observations are those of that run's window); a track the selector never
committed a row of is not kept, as offline keeps every track. Beside it ``<json stem>.live.json``
holds the run's latency report (``LiveStats``) and, with ``record_evidence``,
``<json stem>.evidence.pkl`` the selector's input for ``tools/live_lag_sweep.py``. The court events,
when asked for, are computed once the stream has ended, over the committed output (not live yet).

Ctrl-C stops the source: the frames already in are tracked, every frame is committed and the
outputs are written. A second Ctrl-C aborts.

**Phase C seam.** ``_Pump._push_ball`` is the one place a frame reaches the ball model, before
anything is run on it. A ``RegionGate`` (``run_live``'s `region_for`) is called there with the
``ArrivedFrame`` (source index, picture, arrival time) and returns the region the model should
look at, or None for the whole frame. Anything else the gate decides from (the committed rows,
pass 1's ROI tracker) it must hold itself; v1 passes none. A region is refused for now: the
prepass has no region-aware push (``audit/records/opt_live.md``, "Phase C hook").
"""
from __future__ import annotations

import collections
import contextlib
import gc
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Protocol, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult

from ..config import FOREGROUND, Config
from ..court.court_geometry import last_valid_court_keypoints
from ..engines.ball_detectors import BallDetectorBackend, CachedDetections, FrameInput, GridTrackNetBallDetector
from ..engines.court_detector import CourtDetector
from ..engines.player_detector import PlayerDetector, prefetch_bytetrack
from ..export import RunResult, write_tracking_json
from ..frame_window import FrameWindow
from ..pass1 import Box, Pass1Result, Pass1Collector, use_cuda_graphs
from ..pipeline import build_ball_detector, compute_court_events, assemble_result, open_cuda_device, video_info
from ..run_context import RunContext, Selection, Timing
from ..utils import print_platform_report
from ..video_io import pinned_source
from .evidence import (EMPTY_EVIDENCE, EVIDENCE_SUFFIX, EvidenceFeed, FrameEvidence, GhostEntry, RecordedEvidence,
                       frame_evidence_of, save_evidence)
from .fixed_lag import DEFAULT_DELAY_S, DEFAULT_EVERY_S, DEFAULT_HISTORY_S, LagSettings
from .selector_process import SelectorProcess
from .source import ArrivedFrame, DropPolicy, FrameQueue, LiveSource, make_queue, measure_fps, open_capture

LIVE_REPORT_SUFFIX = ".live.json"
_CAMERA_FPS_FRAMES = 30          # frames a camera's rate is measured over before the stream
_FINISH_TIMEOUT_S = 120.0        # the selector's last run, plus as long again as the stream when it
                                 # is the whole clip's (``delay`` None)

Region = Tuple[int, int, int, int]          # x1, y1, x2, y2 in source pixels


class RegionGate(Protocol):
    """The Phase C seam (module docstring): where the ball model should look in a frame, None for
    the whole frame."""

    def __call__(self, frame: ArrivedFrame) -> Optional[Region]: ...


def camera_name(index: int) -> str:
    """What a camera is called where a run names its input (the tracking JSON's ``video.input``)."""
    return f"camera:{index}"


@dataclass(frozen=True)
class LiveOptions:
    """How a live run is fed and when it commits (``fixed_lag``'s defaults). ``delay_s`` None
    commits only at the end: the offline answer. A camera's frame rate is measured unless
    ``camera_fps`` gives it; a camera cannot be paced ``fast`` (it would wait for the tracker
    and lose frames unseen)."""

    delay_s: Optional[float] = DEFAULT_DELAY_S
    history_s: float = DEFAULT_HISTORY_S
    every_s: float = DEFAULT_EVERY_S
    pacing: str = "realtime"
    drop: DropPolicy = DropPolicy.BACKLOG
    max_backlog_s: float = 0.5
    camera: Optional[int] = None
    camera_fps: Optional[float] = None
    record_evidence: bool = False
    report_every_s: float = 5.0
    selector_process: bool = True        # the selector in its own process (``selector_process``)

    def __post_init__(self) -> None:
        if self.camera is not None and self.pacing == "fast":
            raise ValueError("a camera is realtime: fast pacing would make it wait for the tracker and lose frames")


@dataclass
class LiveStats:
    """What a live run measured. A frame's latency runs from its arrival to its committed output;
    the frames the stream's end flushed (less than ``delay`` after them ever arrived) are counted
    apart and not in the latencies."""

    fps: float
    delay_frames: Optional[int]
    every_frames: int
    history_frames: int
    lookahead_frames: int = 0
    frames: int = 0
    dropped: List[int] = field(default_factory=list)
    flushed_at_end: int = 0
    latencies_s: List[float] = field(default_factory=list)
    evidence_latencies_s: List[float] = field(default_factory=list)   # arrival -> pass 1 committed it
    evidence_frames: List[int] = field(default_factory=list)          # their source frames
    selector_runs: int = 0
    run_seconds: List[float] = field(default_factory=list)      # each selector run's time
    max_waiting: int = 0
    first_arrival: float = 0.0
    last_commit: float = 0.0
    stopped: bool = False                # ended by Ctrl-C

    @property
    def source_seconds(self) -> float:
        return (self.frames + len(self.dropped)) / max(self.fps, 1e-9)

    @property
    def wall_seconds(self) -> float:
        return max(self.last_commit - self.first_arrival, 1e-9)

    def report(self) -> Dict[str, Any]:
        def quantiles(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {"median_ms": None, "p99_ms": None, "max_ms": None}
            ordered = np.sort(np.asarray(values))
            return {"median_ms": 1000 * float(np.median(ordered)),
                    "p99_ms": 1000 * float(np.quantile(ordered, 0.99)), "max_ms": 1000 * float(ordered[-1])}

        return {
            "fps": self.fps, "delay_frames": self.delay_frames, "every_frames": self.every_frames,
            "history_frames": self.history_frames, "lookahead_frames": self.lookahead_frames,
            "frames": self.frames, "dropped": len(self.dropped), "dropped_frames": self.dropped,
            "stopped": self.stopped,
            "flushed_at_end": self.flushed_at_end, "latency": quantiles(self.latencies_s),
            "evidence_latency": quantiles(self.evidence_latencies_s),
            # The latency spikes: the source frames pass 1 committed latest after they arrived.
            "slowest_evidence": [{"frame": self.evidence_frames[index], "ms": 1000 * self.evidence_latencies_s[index]}
                                 for index in np.argsort(self.evidence_latencies_s)[::-1][:10]],
            "selector_runs": self.selector_runs,
            "selector_ms_per_run": 1000 * float(np.mean(self.run_seconds)) if self.run_seconds else None,
            "selector_p99_ms": 1000 * float(np.quantile(self.run_seconds, 0.99)) if self.run_seconds else None,
            "selector_max_ms": 1000 * max(self.run_seconds) if self.run_seconds else None,
            "max_waiting_frames": self.max_waiting,
            "source_seconds": self.source_seconds, "wall_seconds": self.wall_seconds,
            "realtime_factor": self.source_seconds / self.wall_seconds,
        }


class _Pump:
    """Moves frames from the source into the prepass and on to pass 1, on demand (see the module
    docstring): ``pull`` takes the next frame that arrived, or ends the prepass at the stream's end."""

    def __init__(self, queue: FrameQueue, detector: GridTrackNetBallDetector, fps: float, width: int,
                 height: int, region_for: Optional[RegionGate] = None) -> None:
        self.queue, self.detector = queue, detector
        self.region_for = region_for
        self.prepass = detector.open_stream(fps, width, height)
        self.waiting: Deque[np.ndarray] = collections.deque()   # pushed, not yet read by pass 1
        self.arrivals: List[float] = []      # per processed frame
        self.source_index: List[int] = []    # per processed frame
        self.ended = False

    def pull(self) -> bool:
        """Take one more frame; False once the stream has ended (the prepass is then finished)."""
        if self.ended:
            return False
        frame = self.queue.get()
        with self.detector.stream_scope():
            if frame is None:
                self.prepass.finish()
                self.ended = True
                return False
            self._push_ball(frame)
        self.arrivals.append(frame.arrived)
        self.source_index.append(frame.index)
        self.waiting.append(frame.image)
        return True

    def _push_ball(self, frame: ArrivedFrame) -> None:
        """The frame reaches the ball model here, and only here (the Phase C seam, module docstring):
        uploaded from its pinned buffer when it has one, and pushed whole."""
        if self.region_for is not None and self.region_for(frame) is not None:
            raise NotImplementedError("the prepass has no region-aware push yet (Phase C): a gate may only "
                                      "return None, the whole frame")
        pinned = pinned_source(frame.image)
        image: FrameInput = frame.image
        if pinned is not None and self.detector.device.type == "cuda":
            image = pinned.to(self.detector.device, non_blocking=True)
        self.prepass.push(image)


class _LiveBall(BallDetectorBackend):
    """Pass 1's ball detector: the prepass's answer for a frame, pulling frames through the pump
    until that frame is final."""

    uses_raw_frames = True

    def __init__(self, pump: _Pump) -> None:
        self.pump = pump

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> CachedDetections:
        detector = self.pump.detector
        while detector.ready_upto <= frame_index and self.pump.pull():
            pass
        return CachedDetections(frame_index, detector.detections_at(frame_index))

    def detect_async_finish(self, pending: CachedDetections) -> List[Any]:
        return pending.detections

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[Any]:
        return self.detect_async_start(frame_bgr, frame_index).detections


class _PumpReader:
    """Pass 1's frame reader: the frames the pump has pushed, in order, pulling when none waits."""

    def __init__(self, pump: _Pump) -> None:
        self.pump = pump

    def read(self) -> Optional[np.ndarray]:
        while not self.pump.waiting and self.pump.pull():
            pass
        return self.pump.waiting.popleft() if self.pump.waiting else None


class _LiveCollector(Pass1Collector):
    """Pass 1's loop, with each committed frame and the ghost ROIs found with it handed on
    (``evidence.GhostEraser`` has the rule the selector applies them by)."""

    def __init__(self, ctx: RunContext, window: FrameWindow, run: "_LiveRun") -> None:
        super().__init__(ctx, window, None)
        self.run_state = run

    def committed(self, frame_index: int, ghosts: List[Tuple[int, Box]]) -> None:
        self.run_state.committed_frame(frame_index, ghosts)


class _LiveRun:
    """The run's state after pass 1: the selector (in line or in its process), the evidence record
    and the stats, all by source frame (module docstring). Both selectors are fed exactly the
    evidence the record holds (``evidence.EvidenceFeed``), so a replay of the record commits what
    the run committed."""

    def __init__(self, ctx: RunContext, pump: _Pump, options: LiveOptions, stats: LiveStats,
                 settings: LagSettings) -> None:
        self.ctx, self.pump, self.options, self.stats, self.settings = ctx, pump, options, stats, settings
        video = ctx.video
        self.result: Optional[Pass1Result] = None
        self.recorded = RecordedEvidence(video.fps, video.width, video.height) if options.record_evidence else None
        self.process: Optional[SelectorProcess] = None
        self.feed: Optional[EvidenceFeed] = None
        if options.selector_process:
            self.process = SelectorProcess(video.fps, video.width, video.height, settings, on_rows=self._rows)
        else:
            self.feed = EvidenceFeed(video.fps, video.width, video.height, settings, keep=False)
        self.committed: List[Optional[FrameResult]] = []
        self.arrivals: List[Optional[float]] = []      # per source frame fed; None: dropped
        self.tracks: Dict[int, Any] = {}
        self.next_report = 0.0
        self._lock = threading.Lock()

    @property
    def fed(self) -> int:
        """Source frames handed to the selector so far."""
        return len(self.arrivals)

    def attach(self, result: Pass1Result) -> None:
        self.result = result

    def committed_frame(self, frame: int, ghosts: List[Tuple[int, Box]]) -> None:
        """Pass 1 committed processed frame `frame` (and found `ghosts` on earlier ones): hand it to
        the selector as its source frame, after an empty one for each source frame dropped since."""
        assert self.result is not None
        result, pump = self.result, self.pump
        source = pump.source_index[frame]
        self.stats.evidence_latencies_s.append(time.perf_counter() - pump.arrivals[frame])
        self.stats.evidence_frames.append(source)
        while self.fed < source:
            self._feed(EMPTY_EVIDENCE, [], None)
        entries: List[GhostEntry] = [(source, pump.source_index[ghost_frame], box) for ghost_frame, box in ghosts]
        self._feed(frame_evidence_of(result, frame), entries, pump.arrivals[frame])
        # Pass 1 never reads a committed frame's masks again (only how many there are), and the
        # selector and the record hold their own references: a long stream keeps none of them here.
        result.boost_masks[frame] = None
        result.raw_motions[frame] = None
        self._report(time.perf_counter())

    def _feed(self, evidence: FrameEvidence, ghosts: List[GhostEntry], arrival: Optional[float]) -> None:
        source = self.fed
        self.arrivals.append(arrival)
        if self.recorded is not None:
            self.recorded.add(evidence)
            self.recorded.ghosts.extend(ghosts)
            if arrival is not None:
                self.recorded.source_index.append(source)
        if self.process is not None:
            self.process.add(evidence, ghosts)
            return
        assert self.feed is not None
        t0 = time.perf_counter()
        runs = self.feed.selector.runs
        done = self.feed.add(evidence, ghosts)
        if self.feed.selector.runs != runs:
            self.stats.run_seconds.append(time.perf_counter() - t0)
        if done:
            self._rows(done[0], [self.feed.selector.committed[index] for index in done], False)

    def finish(self) -> None:
        """The stream has ended: commit the rest; every row and the stream's tracks."""
        if self.process is not None:
            timeout = _FINISH_TIMEOUT_S
            if self.settings.delay is None:
                timeout += self.fed / max(self.ctx.video.fps, 1e-9)
            output = self.process.finish(timeout)
            self.tracks, self.stats.selector_runs = output.tracks, output.runs
            self.stats.run_seconds = list(output.run_seconds)
            return
        assert self.feed is not None
        t0 = time.perf_counter()
        done = self.feed.finish()
        if done:
            self.stats.run_seconds.append(time.perf_counter() - t0)
            self._rows(done[0], [self.feed.selector.committed[index] for index in done], True)
        self.tracks, self.stats.selector_runs = self.feed.selector.tracks, self.feed.selector.runs

    def close(self) -> None:
        if self.process is not None:
            self.process.close()

    def _rows(self, first: int, rows: List[Optional[FrameResult]], final: bool) -> None:
        """Rows committed for source frames ``first...``: stamp their latency (on the receiving
        thread when the selector runs in its process). The rows the stream's end flushed (`final`)
        had fewer than ``delay`` frames after them and are counted apart; a dropped frame has none."""
        now = time.perf_counter()
        with self._lock:
            if first != len(self.committed):
                raise RuntimeError(f"live rows for frame {first} after {len(self.committed)} committed")
            self.committed.extend(rows)
            arrivals = self.arrivals[first:first + len(rows)]
            if final:
                self.stats.flushed_at_end += sum(arrival is not None for arrival in arrivals)
            else:
                self.stats.latencies_s.extend(now - arrival for arrival in arrivals if arrival is not None)
            self.stats.last_commit = now

    def _report(self, now: float) -> None:
        if self.options.report_every_s <= 0 or now < self.next_report:
            return
        self.next_report = now + self.options.report_every_s
        stats, queue = self.stats, self.pump.queue
        recent = stats.latencies_s[-200:]
        latency = f"{1000 * float(np.median(recent)):.0f} ms" if recent else "-"
        behind = queue.waiting
        print(f"[live] {len(self.pump.arrivals)} frames in, {len(self.committed)} out, latency {latency}, "
              f"{behind} waiting ({behind / max(self.ctx.video.fps, 1e-9):.2f} s behind), "
              f"{len(queue.dropped)} dropped")


@contextlib.contextmanager
def _stop_on_interrupt(source: LiveSource) -> Iterator[None]:
    """Ctrl-C stops the source (the run then finishes the frames already in); a second aborts.
    Only the main thread can take signals; elsewhere Ctrl-C keeps its default."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(_signum: int, _frame: Any) -> None:
        if source.interrupted:
            raise KeyboardInterrupt
        # os.write: print could re-enter a stdout write the signal interrupted.
        os.write(1, b"\n[live] Ctrl-C: stopping the source and finishing the frames already in "
                    b"(Ctrl-C again aborts)\n")
        source.request_stop()

    previous = signal.signal(signal.SIGINT, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def live_config(cfg: Config) -> Config:
    """`cfg` for a live run: no video or debug output written, and the prepass in the foreground
    (``FOREGROUND``), fed frame by frame on the tracking thread (``open_stream``): a thread or a
    process prepass reads the clip itself."""
    return replace(cfg, save_tracking_video=False, save_motion_debug=False, save_detector_input_debug=False,
                   save_track_debug_video=False, motion_diagnostics=False, ball_prepass_mode=FOREGROUND)


def _open_live(cfg: Config, options: LiveOptions) -> RunContext:
    """The config resolved for a CUDA device, the detectors and the capture, as ``pipeline._open_inputs``
    builds them (``open_cuda_device``, ``video_info``), for a live run: no video written, the prepass
    on this thread, a camera named as the input and its frame rate measured."""
    started_at = time.time()
    timing = Timing(cfg.info_timing)
    init_t0 = timing.start()
    if cfg.player_model_path:
        prefetch_bytetrack()
    camera = options.camera
    cfg = live_config(cfg)
    if camera is not None:
        cfg = replace(cfg, input_video=camera_name(camera))
    cfg, device = open_cuda_device(cfg)
    detector = build_ball_detector(cfg)
    if not isinstance(detector, GridTrackNetBallDetector):
        raise RuntimeError("The live tracker runs the five-frame ball model (.pt or .npz weights)")
    capture = open_capture(cfg.input_video if camera is None else None, camera)
    if camera is None:
        video = video_info(Path(cfg.input_video), capture)
        print(f"[init] Live {video.path.name}: {video.width}x{video.height} @ {video.fps:.1f}fps, {options.pacing}")
    else:
        reported = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = options.camera_fps or measure_fps(capture, _CAMERA_FPS_FRAMES)
        video = video_info(Path(camera_name(camera)), capture, fps=fps)
        print(f"[init] Live camera {camera}: {video.width}x{video.height} @ {video.fps:.2f}fps "
              f"({'given' if options.camera_fps else f'measured over {_CAMERA_FPS_FRAMES} frames'}; "
              f"it reports {reported:.2f})")
    court_detector = CourtDetector(cfg)
    player_detector = PlayerDetector(cfg, court_keypoints=None)
    print_platform_report(cfg, None)
    return RunContext(cfg=cfg, video=video, capture=capture, device=device, encoder=None, detector=detector,
                      court_detector=court_detector, player_detector=player_detector, timing=timing,
                      started_at=started_at, init_t0=init_t0)


def run_live(cfg: Config, options: LiveOptions = LiveOptions(),
             region_for: Optional[RegionGate] = None) -> Tuple[RunResult, LiveStats]:
    """Track the ball live (module docstring); writes ``cfg.tracking_json`` and its live report and
    returns the result, indexed by source frame, with the run's measurements. `region_for` is the
    Phase C seam (None in v1: the whole frame)."""
    ctx = _open_live(cfg, options)
    cfg, video = ctx.cfg, ctx.video
    detector = ctx.detector
    assert isinstance(detector, GridTrackNetBallDetector)
    stride = detector.source_stride_for(video.fps)        # the prepass's phases: a drop takes one of each
    queue = make_queue(options.pacing, options.drop, max(stride, int(round(options.max_backlog_s * video.fps))), stride)
    settings = LagSettings.at_fps(video.fps, options.delay_s, options.history_s, options.every_s)
    stats = LiveStats(fps=video.fps, delay_frames=settings.delay, every_frames=settings.every,
                      history_frames=settings.history)
    pump = _Pump(queue, detector, video.fps, video.width, video.height, region_for)
    stats.lookahead_frames = pump.prepass.lookahead_frames
    run = _LiveRun(ctx, pump, options, stats, settings)
    live_ctx = replace(ctx, detector=_LiveBall(pump))
    # Pass 1's frame window is built from the first two frames; a file's source holds the rest until
    # it is (a camera keeps capturing: what the tracker cannot take then is dropped and reported).
    camera = options.camera is not None
    source = LiveSource(ctx.capture, queue, video.fps, options.pacing, camera=camera, hold_after=0 if camera else 2)
    reader = _PumpReader(pump)
    try:
        with _stop_on_interrupt(source):
            if run.process is not None:
                run.process.wait_ready()
            source.start()
            try:
                first = reader.read()
                if first is None:
                    raise RuntimeError(f"No frames from {video.path}")
                stats.first_arrival = pump.arrivals[0]
                window = FrameWindow(ctx.device, cfg.enable_preprocess, first, reader.read(),
                                     cuda_graphs=use_cuda_graphs(cfg, detector))
                ctx.timing.add("init_total", ctx.init_t0)
                print(f"[live] committing {settings.delay if settings.delay is not None else 'all'} frame(s) behind, "
                      f"re-selecting every {settings.every} over {settings.history} frames of history"
                      f"{' in its own process' if options.selector_process else ''}; "
                      f"detections final {stats.lookahead_frames} frames behind")
                collector = _LiveCollector(live_ctx, window, run)
                run.attach(collector.result)
                # Everything built so far (the models, engines, CUDA graphs: millions of objects) lives
                # for the whole stream; out of the collector's reach, a full collection no longer walks
                # it, which otherwise stalls the tracking thread 80-170 ms at a time.
                gc.collect()
                gc.freeze()
                source.release()
                pass1 = collector.run(reader)       # type: ignore[arg-type]   # reads like ThreadedFrameReader
            finally:
                source.stop()
                gc.unfreeze()
            stats.stopped = source.interrupted
            run.finish()
    finally:
        run.close()
    stats.frames = len(pump.arrivals)
    stats.dropped = list(queue.dropped)
    stats.max_waiting = queue.max_waiting
    result = _finish(ctx, pass1, run, pump.source_index, stats)
    if run.recorded is not None and cfg.tracking_json:
        save_evidence(Path(cfg.tracking_json).with_suffix(EVIDENCE_SUFFIX), run.recorded)
    return result, stats


def _by_source_frame(values: List[Any], source_index: List[int], total: int, empty: Any) -> List[Any]:
    out = [empty() if callable(empty) else empty for _ in range(total)]
    for processed, index in enumerate(source_index):
        out[index] = values[processed]
    return out


def _finish(ctx: RunContext, pass1: Pass1Result, run: _LiveRun, source_index: List[int],
            stats: LiveStats) -> RunResult:
    """The result by source frame, the court events over it when asked, the tracking JSON and the
    live report."""
    cfg = ctx.cfg
    total = len(run.committed)
    assert total == run.fed and (not source_index or source_index[-1] + 1 == total)
    court_keypoints = _by_source_frame(pass1.court_keypoints, source_index, total, None)
    player_boxes = _by_source_frame(pass1.player_boxes, source_index, total, dict)
    last_court = last_valid_court_keypoints(court_keypoints)
    tracks = sorted(run.tracks.values(), key=lambda track: track.track_id)
    for track in tracks:
        track.in_timeline = True
    by_source = Pass1Result(player_boxes=player_boxes, court_keypoints=court_keypoints)
    selection = Selection(run.committed, None, tracks, None, None if last_court is None else list(last_court), {})
    events = compute_court_events(ctx, by_source, selection)
    outputs = {"tracking_json": Path(cfg.tracking_json)} if cfg.tracking_json else {}
    result = assemble_result(ctx, by_source, selection, events, total, 0, outputs)
    report = stats.report()
    if cfg.tracking_json:
        write_tracking_json(ctx, by_source, result)
        Path(cfg.tracking_json).with_suffix(LIVE_REPORT_SUFFIX).write_text(json.dumps(report, indent=2),
                                                                           encoding="utf-8")
    latency = report["latency"]
    print(f"\n[live] {stats.frames} frames, {len(stats.dropped)} dropped, {stats.flushed_at_end} flushed at the end"
          f"{', stopped by Ctrl-C' if stats.stopped else ''}")
    if latency["median_ms"] is not None:
        print(f"[live] latency median {latency['median_ms']:.0f} ms, p99 {latency['p99_ms']:.0f} ms, "
              f"max {latency['max_ms']:.0f} ms")
    if stats.run_seconds:
        print(f"[live] selector {report['selector_ms_per_run']:.1f} ms a run (p99 {report['selector_p99_ms']:.0f} ms, "
              f"max {report['selector_max_ms']:.0f} ms, {stats.selector_runs} runs); "
              f"at most {stats.max_waiting} frame(s) waited")
    print(f"[live] realtime factor {report['realtime_factor']:.2f} "
          f"({report['source_seconds']:.1f} s of video in {report['wall_seconds']:.1f} s)")
    filled = result.filled_frames
    print(f"[done] {filled}/{total} frames filled ({100 * filled / max(1, total):.1f}%)")
    if cfg.tracking_json:
        print(f"[done] Tracking JSON: {cfg.tracking_json}")
    return result
