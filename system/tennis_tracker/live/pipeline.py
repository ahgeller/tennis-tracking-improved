"""The live tracker: frames arrive one at a time and each frame's ball output is committed a fixed
number of frames later. Nothing waits for the whole clip.

``run_live(cfg, options)`` drives the offline stages, unchanged, one frame at a time:

1. ``LiveSource`` decodes each frame once, on its own thread, and stamps when it arrived
   (``source``: realtime or as fast as possible, a file or a camera, and the drop policy).
2. ``_Pump`` takes it, pushes it into the ball prepass (``GridTrackNetBallDetector.open_stream``:
   the same engines, ``TransitionRing`` and heat ring, each unit run as soon as it is cut) and
   holds it for pass 1. A frame's detections are final ``PrepassStream.lookahead_frames`` frames
   after it (8 at 60 FPS, 4 at 30 FPS).
3. Pass 1 (``pass1._Pass1Collector``, the offline loop itself) trails the prepass: it pulls frames
   and detections on demand, and both pull the pump, so it runs as far behind the newest frame
   as the prepass's lookahead and the motion gate's next frame need, and no farther.
4. Each frame pass 1 commits goes to the ``FixedLagSelector``, which commits the output of every
   frame ``delay`` frames after it (``fixed_lag``).

The tracking JSON is the offline one (``export.write_tracking_json``), indexed by source frame;
a dropped frame is a row with no ball and no evidence. Beside it ``<json stem>.live.json`` holds
the run's latency report (``LiveStats``) and, with ``record_evidence``, ``<json stem>.evidence.pkl``
the selector's input for ``tools/live_lag_sweep.py``. The court events, when asked for, are
computed once the stream has ended, over the committed output (not live yet).

**Phase C seam.** ``_Pump._push_ball`` is the one place a frame reaches the ball model, before
anything is run on it, with its index and the ball's committed and pending state at hand: a gate
that chooses where the model looks (``roi_gate`` on ``phase-C``) decides there and passes the
region with the frame. v1 always looks at the whole frame.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ball_in_play_selector import FrameResult

from ..config import Config
from ..court.court_geometry import last_valid_court_keypoints
from ..engines.ball_detectors import BallDetectorBackend, CachedDetections, FrameInput, GridTrackNetBallDetector
from ..engines.court_detector import CourtDetector
from ..engines.player_detector import PlayerDetector, prefetch_bytetrack
from ..export import RunResult, write_tracking_json
from ..frame_rate import require_phase_stride
from ..frame_window import FrameWindow
from ..pass1 import FrameRecord, Pass1Result, _Pass1Collector, _use_cuda_graphs
from ..pipeline import _build_ball_detector, _court_events
from ..run_context import RunContext, Selection, Timing, VideoInfo
from ..utils import detect_device, print_platform_report, resolve_runtime
from ..video_io import pinned_source
from .evidence import EVIDENCE_SUFFIX, EvidenceFeed, FrameEvidence, GhostEntry, RecordedEvidence, save_evidence
from .fixed_lag import LagSettings
from .selector_process import SelectorProcess
from .source import ArrivedFrame, DropPolicy, FrameQueue, LiveSource, make_queue, open_capture

_DEFAULT_FPS = 30.0              # a camera that reports no rate
LIVE_REPORT_SUFFIX = ".live.json"
# Pass 1 only appends masks (``pass1.erase_ghost_rois``, which reads them back, is offline's); once a
# frame is handed to the selector, the masks of frames this far behind it are let go.
_PASS1_LOOKBACK = 8


@dataclass(frozen=True)
class LiveOptions:
    """How a live run is fed and when it commits (``audit/records/opt_live.md`` has the evidence for
    the defaults). ``delay_s`` None commits only at the end: the offline answer."""

    delay_s: Optional[float] = 0.5
    history_s: float = 1.0
    every_s: float = 0.2
    pacing: str = "realtime"
    drop: DropPolicy = DropPolicy.BACKLOG
    max_backlog_s: float = 0.25
    camera: Optional[int] = None
    record_evidence: bool = False
    report_every_s: float = 5.0
    selector_process: bool = True        # the selector in its own process (``selector_process``)


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
    selector_runs: int = 0
    run_seconds: List[float] = field(default_factory=list)      # each selector run's time
    max_waiting: int = 0
    first_arrival: float = 0.0
    last_commit: float = 0.0

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
            "flushed_at_end": self.flushed_at_end, "latency": quantiles(self.latencies_s),
            "evidence_latency": quantiles(self.evidence_latencies_s),
            # The latency spikes: the processed frames pass 1 committed latest after they arrived.
            "slowest_evidence": [{"frame": int(index), "ms": 1000 * self.evidence_latencies_s[index]}
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
                 height: int) -> None:
        self.queue, self.detector = queue, detector
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


class _LiveCollector(_Pass1Collector):
    """Pass 1's loop, with each committed frame handed on (``on_commit``) and the ghost ROIs it finds
    logged with it (``evidence.GhostEraser`` has the rule the selector applies them by)."""

    def __init__(self, ctx: RunContext, window: FrameWindow, run: "_LiveRun") -> None:
        super().__init__(ctx, window, None)
        self.run_state = run
        self._ghost_lengths: Dict[int, int] = {}
        found = self.roi_tracker.update_from_dets

        def update_from_dets(dets: Any, frame_index: int) -> Any:
            dropped = found(dets, frame_index)
            ghosts = self.result.ghost_rois
            self._ghost_lengths = {entry[0]: len(ghosts[entry[0]]) for entry in dropped if 0 <= entry[0] < len(ghosts)}
            return dropped

        self.roi_tracker.update_from_dets = update_from_dets      # type: ignore[method-assign]

    def _commit(self, record: FrameRecord, dets: Any) -> None:
        self._ghost_lengths = {}
        super()._commit(record, dets)
        ghosts = self.result.ghost_rois
        new = [(record.frame_index, frame, box) for frame, before in self._ghost_lengths.items()
               for box in ghosts[frame][before:]]
        self.run_state.committed_frame(record.frame_index, new)


class _LiveRun:
    """The run's state after pass 1: the selector (in line or in its process), the evidence record
    and the stats. Both selectors are fed exactly the evidence the record holds
    (``evidence.EvidenceFeed``), so a replay of the record commits what the run committed."""

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
        self.tracks: Dict[int, Any] = {}
        self.released = 0
        self.next_report = 0.0
        self._lock = threading.Lock()

    def attach(self, result: Pass1Result) -> None:
        self.result = result

    def committed_frame(self, frame: int, ghosts: List[GhostEntry]) -> None:
        """Pass 1 committed `frame` (and found `ghosts`): hand it to the selector."""
        assert self.result is not None
        result = self.result
        now = time.perf_counter()
        self.stats.evidence_latencies_s.append(now - self.pump.arrivals[frame])
        evidence: FrameEvidence = (result.detections[frame], result.boost_masks[frame], result.raw_motions[frame],
                                   result.player_boxes[frame], result.court_keypoints[frame])
        if self.recorded is not None:
            self.recorded.add_frame(result, frame)
            self.recorded.ghosts.extend(ghosts)
        if self.process is not None:
            self.process.add(evidence, ghosts)
        else:
            assert self.feed is not None
            t0 = time.perf_counter()
            runs = self.feed.selector.runs
            done = self.feed.add(evidence, ghosts)
            if self.feed.selector.runs != runs:
                self.stats.run_seconds.append(time.perf_counter() - t0)
            if done:
                self._rows(done[0], [self.feed.selector.committed[index] for index in done], False)
        self._release(frame)
        self._report(time.perf_counter())

    def finish(self) -> None:
        """The stream has ended: commit the rest; every row and the stream's tracks."""
        if self.process is not None:
            output = self.process.finish()
            self.process.close()
            self.tracks, self.stats.selector_runs = output.tracks, output.runs
            self.stats.run_seconds = list(output.run_seconds)
        else:
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
        """Rows committed for frames ``first...``: stamp their latency (on the receiving thread when
        the selector runs in its process). The rows the stream's end flushed (`final`) had fewer than
        ``delay`` frames after them and are counted apart."""
        now = time.perf_counter()
        with self._lock:
            if first != len(self.committed):
                raise RuntimeError(f"live rows for frame {first} after {len(self.committed)} committed")
            self.committed.extend(rows)
            if final:
                self.stats.flushed_at_end += len(rows)
            else:
                self.stats.latencies_s.extend(now - self.pump.arrivals[index] for index in range(first, first + len(rows)))
            self.stats.last_commit = now

    def _release(self, frame: int) -> None:
        """Without an evidence record, let go of pass 1's masks once they are sent and older than any
        frame pass 1 still reads (a long stream keeps a bounded window of them)."""
        assert self.result is not None
        if self.recorded is not None:
            return
        upto = max(0, frame - _PASS1_LOOKBACK)
        for index in range(self.released, upto):
            self.result.boost_masks[index] = None
            self.result.raw_motions[index] = None
        self.released = max(self.released, upto)

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


def _open_live(cfg: Config, options: LiveOptions) -> Tuple[RunContext, Any]:
    """The config resolved for a CUDA device, the detectors and the capture, as ``pipeline._open_inputs``
    builds them, for a live run (no video written, the prepass on this thread)."""
    started_at = time.time()
    timing = Timing(cfg.info_timing)
    init_t0 = timing.start()
    if cfg.player_model_path:
        prefetch_bytetrack()
    device_str, _ = detect_device(cfg.device)
    if not str(device_str).isdigit():
        raise RuntimeError("The live tracker requires an NVIDIA CUDA device index such as --device 0.")
    device_index = int(device_str)
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} is unavailable.")
    cfg = replace(cfg, save_tracking_video=False, save_motion_debug=False, save_detector_input_debug=False,
                  save_track_debug_video=False, motion_diagnostics=False,
                  gridtracknet_prepass_background=False, gridtracknet_prepass_process=False)
    cfg, _ = resolve_runtime(cfg, str(device_index), writes_video=False)
    torch.backends.cudnn.benchmark = True
    detector = _build_ball_detector(cfg)
    if not isinstance(detector, GridTrackNetBallDetector):
        raise RuntimeError("The live tracker runs the five-frame ball model (.pt or .npz weights)")
    capture = open_capture(cfg.input_video if options.camera is None else None, options.camera)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or _DEFAULT_FPS
    video = VideoInfo(path=Path(cfg.input_video if options.camera is None else f"camera{options.camera}"), fps=fps,
                      width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      frame_count_hint=max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    print(f"[init] Live {'camera ' + str(options.camera) if options.camera is not None else video.path.name}: "
          f"{video.width}x{video.height} @ {video.fps:.1f}fps, {options.pacing}")
    court_detector = CourtDetector(cfg)
    player_detector = PlayerDetector(cfg, court_keypoints=None)
    print_platform_report(cfg, None)
    ctx = RunContext(cfg=cfg, video=video, capture=capture, device=torch.device(f"cuda:{device_index}"),
                     encoder=None, detector=detector, court_detector=court_detector,
                     player_detector=player_detector, timing=timing, started_at=started_at, init_t0=init_t0)
    return ctx, capture


def run_live(cfg: Config, options: LiveOptions = LiveOptions()) -> Tuple[RunResult, LiveStats]:
    """Track the ball live (module docstring); writes ``cfg.tracking_json`` and its live report and
    returns the result, indexed by source frame, with the run's measurements."""
    ctx, capture = _open_live(cfg, options)
    cfg, video = ctx.cfg, ctx.video
    detector = ctx.detector
    assert isinstance(detector, GridTrackNetBallDetector)
    stride = require_phase_stride(video.fps)
    queue = make_queue(options.pacing, options.drop, max(stride, int(round(options.max_backlog_s * video.fps))), stride)
    settings = LagSettings.at_fps(video.fps, options.delay_s, options.history_s, options.every_s)
    stats = LiveStats(fps=video.fps, delay_frames=settings.delay, every_frames=settings.every,
                      history_frames=settings.history)
    pump = _Pump(queue, detector, video.fps, video.width, video.height)
    stats.lookahead_frames = pump.prepass.lookahead_frames
    run = _LiveRun(ctx, pump, options, stats, settings)
    live_ctx = replace(ctx, detector=_LiveBall(pump))
    source = LiveSource(capture, queue, video.fps, options.pacing, camera=options.camera is not None)
    reader = _PumpReader(pump)
    try:
        if run.process is not None:
            run.process.wait_ready()
        source.start()
        try:
            first = reader.read()
            if first is None:
                raise RuntimeError(f"No frames from {video.path}")
            stats.first_arrival = pump.arrivals[0]
            window = FrameWindow(ctx.device, cfg.enable_preprocess, first, reader.read(), cuda_graphs=_use_cuda_graphs(cfg))
            ctx.timing.add("init_total", ctx.init_t0)
            print(f"[live] committing {settings.delay if settings.delay is not None else 'all'} frame(s) behind, "
                  f"re-selecting every {settings.every} over {settings.history} frames of history"
                  f"{' in its own process' if options.selector_process else ''}; "
                  f"detections final {stats.lookahead_frames} frames behind")
            collector = _LiveCollector(live_ctx, window, run)
            run.attach(collector.result)
            pass1 = collector.run(reader)       # type: ignore[arg-type]   # reads like ThreadedFrameReader
        finally:
            source.stop()
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
    total = (source_index[-1] + 1) if source_index else 0
    per_frame = _by_source_frame(run.committed, source_index, total, None)
    court_keypoints = _by_source_frame(pass1.court_keypoints, source_index, total, None)
    player_boxes = _by_source_frame(pass1.player_boxes, source_index, total, dict)
    last_court = last_valid_court_keypoints(court_keypoints)
    last_court = None if last_court is None else list(last_court)
    tracks = sorted(run.tracks.values(), key=lambda track: track.track_id)
    for track in tracks:
        track.in_timeline = True
    by_source = Pass1Result(player_boxes=player_boxes, court_keypoints=court_keypoints)
    selection = Selection(per_frame, None, tracks, None, last_court, {})
    events = _court_events(ctx, by_source, selection)
    result = RunResult(
        per_frame=per_frame, chosen_track=None, all_tracks=tracks, frame_count=total,
        court_keypoints=court_keypoints, player_boxes=player_boxes,
        last_valid_court_keypoints=last_court,
        elapsed_sec=ctx.elapsed(), timing=dict(ctx.timing.values), pass2_frames_rendered=0,
        outputs={"tracking_json": Path(cfg.tracking_json)} if cfg.tracking_json else {}, court_events=events)
    report = stats.report()
    if cfg.tracking_json:
        write_tracking_json(ctx, by_source, result)
        Path(cfg.tracking_json).with_suffix(LIVE_REPORT_SUFFIX).write_text(json.dumps(report, indent=2),
                                                                           encoding="utf-8")
    latency = report["latency"]
    print(f"\n[live] {stats.frames} frames, {len(stats.dropped)} dropped, {stats.flushed_at_end} flushed at the end")
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

