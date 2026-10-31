"""Pass 1: decode the clip once, collecting detections and motion evidence per frame.

For every source frame, in order: fold it into the background model, run the
court and player detectors (every frame or on a schedule), compute the motion
masks inside the ROI tracker's regions, hand the detector its input, and
store the frame's evidence.  Ball inference for frame N overlaps the
preprocessing of frame N+1; a frame is committed when its detections arrive.
"""
from __future__ import annotations

import collections
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from ball_in_play_selector import PackedMask, RawDetection

from .court.court_geometry import court_keypoints_key
from .court.court_scene import PlayerBoxes
from .cuda_kernels import KernelUnavailable
from .config import THREAD
from .engines.ball_detectors import BallDetectorBackend, FrameInput
from .engines.player_detector import PlayerPending, interpolate_player_boxes
from .frame_window import FrameWindow
from .motion import (
    BackgroundModel, BoostMasks, DetectorInputRequest, MotionMasks, PreprocessResult, build_court_side_protect_mask,
    preprocess_frame_cuda,
)
from .motion_gpu import MotionKernels
from .run_context import RunContext
from .roi_tracking import ROIMotionTracker
from .video_io import ThreadedFrameReader, VideoWriter, pinned_source

Box = Tuple[int, int, int, int]

# A track that has not seen a detection for more than this many frames is not
# "visible" for the skip-frame decision.
_VISIBLE_MAX_FRAMES_SINCE_DET = 3
# Two ROI boxes are the same box when every edge is within this many pixels.
_ROI_MATCH_TOLERANCE_PX = 2
_BYTES_PER_PIXEL = 3
_MIB = 1024.0 * 1024.0

try:
    import psutil
except ImportError:          # optional: without it the automatic cache budget is off
    psutil = None


@dataclass
class Pass1Result:
    """Everything pass 1 learned, one entry per decoded frame in source order."""

    detections: List[List[RawDetection]] = field(default_factory=list)
    boost_masks: BoostMasks = field(default_factory=BoostMasks)              # filtered when first read
    raw_motions: List[Optional[PackedMask]] = field(default_factory=list)
    rois: List[Optional[List[Box]]] = field(default_factory=list)          # visual ROI boxes
    ghost_rois: List[List[Box]] = field(default_factory=list)              # ROIs of tracks that timed out
    player_boxes: List[PlayerBoxes] = field(default_factory=list)          # tracked player id -> box
    court_keypoints: List[Optional[List[float]]] = field(default_factory=list)
    cached_frames: Optional[List[np.ndarray]] = None   # the clip's first frames, as many as fit the RAM budget
    roi_frames: int = 0            # frames whose motion was computed inside ROIs (counted per ROI)
    fullframe_frames: int = 0
    skipped_detect_frames: int = 0

    @property
    def frame_count(self) -> int:
        return len(self.detections)


@dataclass
class FrameRecord:
    """A frame's evidence, held until its ball detections arrive.  The motion masks are still on
    their way to the host (``MotionMasks``) and the player boxes are read at commit time, after
    the frame's player inference has been finished."""

    frame_index: int
    masks: Optional[MotionMasks]
    rois: Optional[List[Box]]
    court_keypoints: Optional[List[float]]
    detector_frame: Optional[np.ndarray]
    detector_frame_cuda: Optional[torch.Tensor]


@dataclass
class PendingDetection:
    handle: Any
    record: FrameRecord


@dataclass
class _Preprocessed:
    """What preprocessing produced for a frame: the detector's input, the masks, the ROIs and
    the ball inference it may already have started."""

    detector_frame: Optional[np.ndarray]
    detector_frame_cuda: Optional[torch.Tensor] = None
    masks: Optional[MotionMasks] = None
    rois_visual: Optional[List[Box]] = None
    det_pending: Any = None

    @property
    def detector_input(self) -> Optional[FrameInput]:
        return self.detector_frame_cuda if self.detector_frame_cuda is not None else self.detector_frame


def pass2_cache_budget_mb(cfg) -> float:
    """How much RAM pass 2's frame cache may use.

    ``pass2_cache_max_mb`` > 0 is an explicit override.  Otherwise take half of
    the memory actually free right now, capped by ``pass2_cache_ceiling_mb`` --
    a fixed budget cannot know whether a 12 GiB clip fits on this machine.
    Without psutil that cannot be measured, so the budget is 0.
    """
    explicit = cfg.pass2_cache_max_mb
    if explicit > 0:
        return float(explicit)
    if psutil is None:
        return 0.0          # cannot tell how much is free: do not risk the machine
    ceiling = float(max(1, cfg.pass2_cache_ceiling_mb))
    free_mb = psutil.virtual_memory().available / _MIB
    return min(0.5 * free_mb, ceiling)


def cache_frame_capacity(budget_mb: float, width: int, height: int) -> int:
    """How many decoded ``width`` x ``height`` frames a cache budget of ``budget_mb`` holds."""
    return int(budget_mb * _MIB // (width * height * _BYTES_PER_PIXEL))


class FrameCache:
    """The decoded frames kept for pass 2: the clip's first ``capacity`` frames, the share of the
    clip the RAM budget holds.  Pass 2 serves these from memory and decodes only the rest, so a
    budget smaller than the clip still saves that share of the second decode.  ``frames`` is the
    list ``Pass1Result.cached_frames`` hands to pass 2; ``add`` fills it in source order and
    stops at the capacity, or earlier when the machine runs out of memory."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.frames: List[np.ndarray] = []

    @property
    def full(self) -> bool:
        return len(self.frames) >= self.capacity

    def add(self, frame: np.ndarray) -> None:
        """Keep ``frame`` while the cache lasts; the first frame it cannot keep ends the prefix."""
        if self.full:
            return
        try:
            # A frame decoded into pinned memory is copied out of it: the clip's frames would
            # otherwise hold gigabytes of page-locked RAM.
            self.frames.append(frame.copy() if pinned_source(frame) is not None else frame)
        except MemoryError:
            self.capacity = len(self.frames)
            print(f"[pass 1] RAM frame cache stopped at {self.capacity} frames (MemoryError)")


def _new_frame_cache(ctx: RunContext) -> Optional[FrameCache]:
    """A frame cache sized to the RAM budget, else None (with a note).  The cache holds as many of
    the clip's frames as the budget allows, the whole clip when it fits."""
    cfg, video = ctx.cfg, ctx.video
    if not (ctx.needs_pass2_outputs and cfg.cache_input_frames_pass2 and video.frame_count_hint > 0):
        return None
    est_mb = float(video.frame_count_hint) * video.width * video.height * _BYTES_PER_PIXEL / _MIB
    budget_mb = pass2_cache_budget_mb(cfg)
    capacity = cache_frame_capacity(budget_mb, video.width, video.height)
    if est_mb <= budget_mb:
        print(f"[pass 1] RAM frame cache enabled ({est_mb:.0f} MiB est, {budget_mb:.0f} MiB budget)")
        return FrameCache(capacity)
    if capacity > 0:
        print(f"[pass 1] RAM frame cache: {capacity} of {video.frame_count_hint} frames fit the "
              f"{budget_mb:.0f} MiB budget")
        return FrameCache(capacity)
    if psutil is None and cfg.pass2_cache_max_mb <= 0:
        print("[pass 1] RAM frame cache disabled (psutil not installed; set pass2_cache_max_mb to enable it)")
    else:
        print(f"[pass 1] RAM frame cache disabled ({budget_mb:.0f} MiB budget holds no frame of {est_mb:.0f} MiB)")
    return None


def _same_box(a: Box, b: Box) -> bool:
    return all(abs(int(a[i]) - int(b[i])) <= _ROI_MATCH_TOLERANCE_PX for i in range(4))


class Pass1Collector:
    """The pass-1 loop and the per-frame state it carries between iterations."""

    def __init__(self, ctx: RunContext, window: FrameWindow,
                 detector_input_writer: Optional[VideoWriter]) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.video = ctx.video
        self.timing = ctx.timing
        self.detector = ctx.detector
        self.court_detector = ctx.court_detector
        self.player_detector = ctx.player_detector
        self.window = window
        self.detector_input_writer = detector_input_writer
        self.background = BackgroundModel(self.cfg, ctx.device, cuda_graphs=use_cuda_graphs(self.cfg, ctx.detector))
        self.motion_kernels = _load_motion_kernels(self.cfg, ctx.device)
        self.roi_tracker = ROIMotionTracker(self.cfg, self.video.width, self.video.height, self.video.fps)
        self.pre_cuda_perf: Optional[Dict[str, float]] = (
            {"pre_cuda_total": 0.0, "pre_cuda_raw_d2h": 0.0, "pre_cuda_cc_filter": 0.0}
            if self.timing.enabled else None)
        # Court-side protect mask, rebuilt only when the court keypoints move.
        self._protect_key: Optional[Tuple[int, ...]] = None
        self._protect_mask: Optional[np.ndarray] = None
        self._protect_mask_cuda: Optional[torch.Tensor] = None
        # Skip-frame detection and the aux (court/player) schedule.
        self.skip_n = max(1, self.cfg.skip_frame_detect)
        self.aux_force_interval = max(1, self.cfg.aux_force_interval)
        self.last_aux_detect_frame = -10 ** 9
        self.use_cuda_detector_input = self.cfg.enable_preprocess and self.detector.supports_cuda_frame()
        self.can_overlap_ball_inference = bool(self.detector.use_async and int(self.detector.async_slots) >= 2)
        if self.use_cuda_detector_input:
            print("[pass 1] CUDA zero-copy preprocess->detector path enabled")
        self._frame_cache = _new_frame_cache(ctx)
        self.result = Pass1Result(cached_frames=self._frame_cache.frames if self._frame_cache is not None else None)
        self._progress_frame, self._progress_elapsed = 0, 0.0   # where the last progress line was
        self._player_pending: Deque[PlayerPending] = collections.deque()   # launched, not yet finished
        self._players_tracked: List[bool] = []    # per committed frame: did the player tracker run on it

    # -- per-frame steps -------------------------------------------------------

    def _should_detect(self, frame_index: int) -> bool:
        """Skip-frame detection: every Nth frame, and between them only while no
        ROI track has the ball in sight (so a lost ball is reacquired at once)."""
        if self.skip_n <= 1 or frame_index % self.skip_n == 0:
            return True
        if not self.cfg.skip_frame_require_roi:
            return False
        visible = any(track.frames_since_det <= _VISIBLE_MAX_FRAMES_SINCE_DET
                      for track in self.roi_tracker.tracks)
        return not visible

    def _aux_detect(self, frame_index: int, frame: np.ndarray,
                    detect_this_frame: bool) -> Tuple[Optional[List[float]], Optional[PlayerPending]]:
        """Court keypoints for this frame and the started player inference (or None)."""
        t0 = self.timing.start()
        run_aux = True
        if self.cfg.aux_detect_on_detect_frames and not detect_this_frame:
            run_aux = (frame_index - self.last_aux_detect_frame) >= self.aux_force_interval
        player_pending = None
        if run_aux:
            court_kps = self.court_detector.detect(frame, frame_index=frame_index)
            self.player_detector.set_court_keypoints(court_kps)
            gpu = self.window.curr_gpu
            player_pending = self.player_detector.detect_async_start(
                frame, frame_index=frame_index, frame_gpu_chw=gpu.chw if gpu is not None else None)
            self.last_aux_detect_frame = frame_index
        else:
            court_kps = self.court_detector.keypoints
            self.player_detector.set_court_keypoints(court_kps)
        self.timing.add("pass1_aux_detect", t0)
        return court_kps, player_pending

    def _protect_mask_for(self, court_kps: Optional[List[float]]) -> Optional[torch.Tensor]:
        """The CUDA court-side protect mask for these keypoints (cached per keypoint set)."""
        t0 = self.timing.start()
        key = court_keypoints_key(court_kps)
        if key != self._protect_key:
            self._protect_mask = build_court_side_protect_mask(self.video.height, self.video.width, court_kps)
            self._protect_key = key
            self._protect_mask_cuda = (
                None if self._protect_mask is None
                else torch.from_numpy(self._protect_mask).to(self.ctx.device))
        self.timing.add("pass1_pre_mask_build", t0)
        return self._protect_mask_cuda

    def _preprocess(self, frame_index: int, court_kps: Optional[List[float]],
                    detect_this_frame: bool) -> Tuple[PreprocessResult, Optional[List[Box]]]:
        """Motion masks and detector input for the current frame.

        Returns ``(preprocess_result, visual_rois)``.  The court-side protect mask only shapes
        the detector input, so it is built only for a frame that asks for one.
        """
        request = self._detector_input_request(detect_this_frame)
        protect_mask_cuda = self._protect_mask_for(court_kps) if (request.cpu or request.cuda) else None
        rois, rois_visual = self.roi_tracker.next_rois(frame_index)
        if rois:
            self.result.roi_frames += len(rois)
        else:
            self.result.fullframe_frames += 1
        pre = preprocess_frame_cuda(self.window, self.background, self.cfg, request=request,
                                    rois=rois, protect_mask=protect_mask_cuda, perf=self.pre_cuda_perf,
                                    kernels=self.motion_kernels)
        return pre, rois_visual

    def _detector_input_request(self, detect_this_frame: bool) -> DetectorInputRequest:
        """Which detector inputs this frame's preprocessing must build.

        A detector that reads raw frames (GridTrackNet) gets neither, and the
        enhanced frame is then built only for the detector-input debug video
        (undimmed, from the gated motion).  The TensorRT engine gets a CUDA
        frame when it is detecting this frame and can take one, else a host
        frame; the static background is left undimmed while an ROI track has
        the ball in sight (``cfg.skip_preprocess_dim``).
        """
        raw_frames = self.detector.uses_raw_frames
        ball_in_sight = len(self.roi_tracker.tracks) > 0 and self.roi_tracker.ball_visible
        return DetectorInputRequest(
            cpu=self.detector_input_writer is not None or (not self.use_cuda_detector_input and not raw_frames),
            cuda=self.use_cuda_detector_input and detect_this_frame,
            dim_static=not raw_frames and not (self.cfg.skip_preprocess_dim and ball_in_sight),
            wide_boost=not raw_frames,
        )

    # -- committing frames ------------------------------------------------------

    def _finish_players(self, through_frame: int) -> None:
        """Finish the player inferences launched for frames up to ``through_frame``, in order.

        A frame's player inference is launched in its own iteration and finished in the next,
        just before the frame is committed, so the GPU has a whole frame of other work to run it
        behind instead of the loop waiting for it; ByteTrack still sees every frame in order, and
        a frame's boxes are the tracker's state after its own frame, as before.
        """
        while self._player_pending and self._player_pending[0].frame_index <= through_frame:
            t0 = self.timing.start()
            self.player_detector.detect_async_finish(self._player_pending.popleft())
            self.timing.add("pass1_aux_wait", t0)

    def _commit(self, record: FrameRecord, dets: List[RawDetection]) -> None:
        """Append the frame's evidence; retire the ROIs of tracks that just timed out."""
        self._finish_players(record.frame_index)
        t0 = self.timing.start()
        out = self.result
        dropped_ghost_rois = self.roi_tracker.update_from_dets(dets, record.frame_index)
        out.detections.append(dets)
        masks = record.masks
        out.boost_masks.append(masks.deferred_boost() if masks is not None else None)
        out.raw_motions.append(masks.packed_raw() if masks is not None else None)
        # A track that timed out never had the ball: its coasting ROIs on
        # earlier frames are moved from the visual ROI list to the ghost list,
        # and erase_ghost_rois() clears their motion in one batch after pass 1.
        found: List[Tuple[int, Box]] = []
        for ghost_frame, rx1, ry1, rx2, ry2 in dropped_ghost_rois:
            if not (0 <= ghost_frame < len(out.raw_motions)) or ghost_frame == record.frame_index:
                continue
            if out.rois[ghost_frame] is None:
                continue
            ghost = (rx1, ry1, rx2, ry2)
            kept = [box for box in out.rois[ghost_frame] if not _same_box(box, ghost)]
            if len(kept) != len(out.rois[ghost_frame]):
                out.ghost_rois[ghost_frame].append(ghost)
                found.append((ghost_frame, ghost))
            out.rois[ghost_frame] = kept if kept else None
        out.rois.append(record.rois)
        out.ghost_rois.append([])
        # A frame the player tracker skips (cfg.player_detect_interval) is filled after the loop.
        out.player_boxes.append(self.player_detector.player_boxes_by_slot())
        self._players_tracked.append(self.player_detector.tracks_frame(record.frame_index))
        out.court_keypoints.append(record.court_keypoints)
        if self.detector_input_writer is not None:
            # Stream the detector input straight to its writer rather than keep it.
            if record.detector_frame is not None:
                self.detector_input_writer.write(record.detector_frame)
            elif record.detector_frame_cuda is not None:
                self.detector_input_writer.write(record.detector_frame_cuda.detach().cpu().numpy())
        self.timing.add("pass1_store", t0)
        self.committed(record.frame_index, found)

    def committed(self, frame_index: int, ghosts: List[Tuple[int, Box]]) -> None:
        """Called once frame `frame_index`'s evidence is stored, with the ghost ROIs ``(frame, box)``
        added to earlier frames as it was; for a subclass that hands frames on (the live tracker)."""

    def _finish_pending(self, pending: PendingDetection) -> None:
        t0 = self.timing.start()
        dets = self.detector.detect_async_finish(pending.handle)
        self.timing.add("pass1_ball_detect", t0)
        self._commit(pending.record, dets)

    # -- the loop ---------------------------------------------------------------

    def _preprocess_frame(self, frame_index: int, frame: np.ndarray, court_kps: Optional[List[float]],
                          detect_this_frame: bool, prev_pending: Optional[PendingDetection]
                          ) -> Tuple[_Preprocessed, Optional[PendingDetection]]:
        """Preprocess the frame and start its ball inference as soon as the input is ready, so
        it overlaps the CPU work that follows. Returns ``prev_pending`` unchanged, or None when a
        backend without output slots had to finish the previous frame first."""
        pre, rois_visual = self._preprocess(frame_index, court_kps, detect_this_frame)
        out = _Preprocessed(
            detector_frame=frame if self.detector.uses_raw_frames else pre.detector_frame,
            detector_frame_cuda=pre.detector_frame_cuda,
            masks=pre.masks, rois_visual=rois_visual)
        if detect_this_frame:
            if prev_pending is not None and not self.can_overlap_ball_inference:
                self._finish_pending(prev_pending)
                prev_pending = None
            out.det_pending = self.detector.detect_async_start(out.detector_input, frame_index=frame_index)
        return out, prev_pending

    def _queue_detection(self, record: FrameRecord, pre: _Preprocessed,
                         detect_this_frame: bool) -> Optional[PendingDetection]:
        """The frame's ball inference, started now when preprocessing did not; a frame the
        skip schedule leaves out is committed at once with no detections."""
        if not detect_this_frame:
            self.result.skipped_detect_frames += 1
            self._commit(record, [])
            return None
        det_pending = pre.det_pending
        if det_pending is None:
            det_t0 = self.timing.start()
            det_pending = self.detector.detect_async_start(pre.detector_input, frame_index=record.frame_index)
            self.timing.add("pass1_ball_detect", det_t0)
        return PendingDetection(det_pending, record)

    def _progress(self, frame_index: int) -> None:
        """Every ``progress_every`` frames: the average since the run began (init included, so it
        climbs for a while) and the rate over the frames since the previous line, which is the
        speed pass 1 is actually running at."""
        if self.cfg.progress_every and frame_index % self.cfg.progress_every == 0:
            elapsed = self.ctx.elapsed()
            recent = (frame_index - self._progress_frame) / max(elapsed - self._progress_elapsed, 1e-9)
            self._progress_frame, self._progress_elapsed = frame_index, elapsed
            print(f"  [pass1 {frame_index}/{self.video.frame_count_hint}] "
                  f"{frame_index/elapsed:.1f} fps avg, {recent:.1f} fps now")

    def run(self, reader: ThreadedFrameReader) -> Pass1Result:
        cfg, window, timing = self.cfg, self.window, self.timing
        pending: Optional[PendingDetection] = None
        frame_index = 0
        while not window.exhausted:
            frame = window.curr
            # Finish the previous frame's inference after this frame's
            # preprocessing, so the two overlap.
            prev_pending, pending = pending, None
            if self._frame_cache is not None:
                self._frame_cache.add(frame)
            if cfg.enable_preprocess:
                self.background.update(window.curr_gpu.vs)
            detect_this_frame = self._should_detect(frame_index)
            court_kps, player_pending = self._aux_detect(frame_index, frame, detect_this_frame)
            if player_pending is not None:
                self._player_pending.append(player_pending)

            pre_t0 = timing.start()
            pre = _Preprocessed(detector_frame=frame)
            if cfg.enable_preprocess:
                pre, prev_pending = self._preprocess_frame(frame_index, frame, court_kps, detect_this_frame, prev_pending)
            timing.add("pass1_preprocess", pre_t0)

            if prev_pending is not None:
                self._finish_pending(prev_pending)

            record = FrameRecord(
                frame_index=frame_index,
                masks=pre.masks,
                rois=pre.rois_visual if pre.rois_visual is not None else self.roi_tracker.last_rois,
                court_keypoints=court_kps,
                detector_frame=pre.detector_frame,
                detector_frame_cuda=pre.detector_frame_cuda,
            )
            pending = self._queue_detection(record, pre, detect_this_frame)

            frame_index += 1
            self._progress(frame_index)
            slide_t0 = timing.start()
            window.slide(reader.read())
            timing.add("pass1_slide", slide_t0)

        if pending is not None:
            self._finish_pending(pending)
        self._finish_players(frame_index)
        if not all(self._players_tracked):
            self.result.player_boxes = interpolate_player_boxes(self.result.player_boxes, self._players_tracked)
        return self.result


def _open_detector_input_writer(ctx: RunContext) -> Optional[VideoWriter]:
    cfg, video = ctx.cfg, ctx.video
    if not cfg.save_detector_input_debug:
        return None
    os.makedirs(os.path.dirname(cfg.output_detector_input_debug_path) or ".", exist_ok=True)
    return VideoWriter(cfg.output_detector_input_debug_path, video.fps, video.width, video.height,
                       cfg, encoder=ctx.encoder)


def use_cuda_graphs(cfg, detector: BallDetectorBackend) -> bool:
    """Whether pass 1 replays its fixed-shape GPU work from CUDA graphs: ``cfg.pass1_cuda_graphs``,
    unless the prepass runs on a worker thread of this process, whose CUDA work during a capture
    would invalidate it (a prepass process has its own context)."""
    return bool(cfg.pass1_cuda_graphs) and detector.prepass_mode != THREAD


def _pinned_frame_shape(ctx: RunContext) -> Optional[Tuple[int, int]]:
    """The ``(height, width)`` pass 1's reader decodes into pinned buffers when the frames go to the
    GPU, so their upload needs no copy into a pinned buffer on this thread; else None."""
    if not (ctx.cfg.enable_preprocess and ctx.device.type == "cuda"):
        return None
    return ctx.video.height, ctx.video.width


def _load_motion_kernels(cfg, device: torch.device) -> Optional[MotionKernels]:
    """The motion-mask kernels (``motion_gpu``) when ``cfg.motion_kernels`` asks for them and the
    device is a GPU, else None (the PyTorch ops build the same masks); a kernel that cannot be
    loaded is reported and the PyTorch ops run."""
    if not (cfg.enable_preprocess and cfg.motion_kernels and device.type == "cuda"):
        return None
    try:
        return MotionKernels.load(device, Path(cfg.ball_engine_dir))
    except KernelUnavailable as error:
        # Only loading falls back: a launch that fails later, mid-clip, raises to the caller rather
        # than switch paths half way through the masks.
        reason = str(error).partition("\n")[0]
        print(f"[pass 1] motion kernels unavailable, building the masks with PyTorch ops ({reason})")
        return None


def collect_pass1(ctx: RunContext) -> Pass1Result:
    """Decode the clip and collect per-frame evidence; raises RuntimeError on an empty clip."""
    cfg, timing = ctx.cfg, ctx.timing
    print("[preprocess] CUDA S+V motion path" if cfg.enable_preprocess else "[preprocess] disabled")
    reader = ThreadedFrameReader(ctx.capture, prefetch=max(2, cfg.frame_reader_prefetch),
                                 pinned_shape=_pinned_frame_shape(ctx))
    first = reader.read()
    if first is None:
        reader.release()
        raise RuntimeError(f"No frames in video: {ctx.video.path}")
    second = reader.read()
    window = FrameWindow(ctx.device, cfg.enable_preprocess, first, second,
                         cuda_graphs=use_cuda_graphs(cfg, ctx.detector))
    timing.add("init_total", ctx.init_t0)
    pass1_t0 = timing.start()
    print("[pass 1] Collecting detections...")
    detector_input_writer = _open_detector_input_writer(ctx)
    collector = Pass1Collector(ctx, window, detector_input_writer)
    try:
        result = collector.run(reader)
    finally:
        reader.release()
        if detector_input_writer is not None:
            detector_input_writer.close()
    timing.add("pass1_total", pass1_t0)
    perf = collector.pre_cuda_perf
    if perf is not None and float(perf.get("pre_cuda_total", 0.0)) > 0.0:
        pre_total = float(perf["pre_cuda_total"])
        pre_d2h = float(perf.get("pre_cuda_raw_d2h", 0.0))
        pre_cc = float(perf.get("pre_cuda_cc_filter", 0.0))
        timing.set("pass1_pre_cuda_total", pre_total)
        timing.set("pass1_pre_cuda_d2h", pre_d2h)
        timing.set("pass1_pre_cuda_cc", pre_cc)
        timing.set("pass1_pre_cuda_gpu_other", max(0.0, pre_total - pre_d2h - pre_cc))

    n = result.frame_count
    t_pass1 = ctx.elapsed()
    det_frames = sum(1 for d in result.detections if d)
    print(f"[pass 1] Done: {n} frames in {t_pass1:.1f}s "
          f"({n/t_pass1:.1f} fps), detections in {det_frames} frames")
    if collector.skip_n > 1:
        skipped = result.skipped_detect_frames
        print(f"[pass 1] Skip-frame detection: {skipped}/{n} frames skipped "
              f"(every {collector.skip_n}th frame, {100.0*skipped/max(n,1):.1f}% saved)")
    if cfg.roi_motion_enabled:
        print(f"[pass 1] ROI motion: {result.roi_frames} ROI frames, "
              f"{result.fullframe_frames} full-frame frames")
    return result


def erase_ghost_rois(result: Pass1Result) -> int:
    """Clear the motion inside every ghost ROI and repack those frames' masks.

    A ghost ROI is the coasting region of an ROI track that timed out without
    reacquiring the ball; motion in it was never ball evidence.  Returns the
    number of frames whose masks changed.
    """
    print("[pass 1] Executing batched ghost ROI deletion...")
    t0 = time.perf_counter()
    erased_frames = 0
    for frame_index, ghosts in enumerate(result.ghost_rois):
        if ghosts and erase_ghosts_in_frame(result, frame_index, ghosts):
            erased_frames += 1
    print(f"[pass 1] Batched ghost deletion cleaned {erased_frames} frames in {time.perf_counter() - t0:.2f}s")
    return erased_frames


def _clear_boxes(mask: Optional[np.ndarray], boxes: List[Box]) -> bool:
    """Zero ``boxes`` in ``mask``; whether any of them held a set pixel."""
    changed = False
    for rx1, ry1, rx2, ry2 in boxes:
        if mask is not None and np.any(mask[ry1:ry2, rx1:rx2]):
            mask[ry1:ry2, rx1:rx2] = 0
            changed = True
    return changed


def erase_ghosts_in_frame(result: Pass1Result, frame_index: int, ghosts: List[Box]) -> bool:
    """Clear the ghost ROIs in one frame's raw and boost masks and repack the ones that changed;
    whether either mask changed."""
    raw = result.raw_motions[frame_index] if frame_index < len(result.raw_motions) else None
    boost = result.boost_masks[frame_index] if frame_index < len(result.boost_masks) else None
    if raw is None and boost is None:
        return False
    raw_u8 = raw.unpack() if raw is not None else None
    boost_u8 = boost.unpack() if boost is not None else None
    changed = _clear_boxes(raw_u8, ghosts) | _clear_boxes(boost_u8, ghosts)
    if not changed:
        return False
    if raw_u8 is not None:
        result.raw_motions[frame_index] = PackedMask.pack(raw_u8, roi=raw.roi)
    if boost_u8 is not None:
        result.boost_masks[frame_index] = PackedMask.pack(boost_u8, roi=boost.roi)
    return True
