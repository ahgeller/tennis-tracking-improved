"""Pass 2: render the tracking video and the optional debug videos from pass 1's evidence.

Frames come from the RAM cache pass 1 filled (the clip's first frames, as many
as fit the budget) and, past it, from a second decode of the clip.  Every
frame gets the court, players, the selected ball and its trail (and, with
``Config.show_all_tracks``, every candidate track lightly underneath); the
track-debug video adds raw detections, ROIs, every candidate track with its id
and score, and the chosen track's display path; the motion-debug video shows
the masks.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .court.court_geometry import MIN_KEYPOINT_VALUES, build_ground_projection_model, court_keypoints_key
from .court.court_minimap import draw_court_minimap
from .motion import build_court_side_protect_mask
from .render.motion_debug import draw_motion_masks, draw_selected_ball
from .render.motion_trails import RawMotionTrails
from .pass1 import Pass1Result
from .render.court_layer import CourtLayer
from .render.rendering import (
    Canvas, MarkerSizer, SearchCentreSmoother, draw_hud, draw_motion_rois, draw_player_boxes, draw_raw_detections,
    draw_selected_result,
)
from .render.trails import DEBUG_TRACK_STYLE, LIGHT_TRACK_STYLE, BallTrail, CandidateTrackTrails, DisplayPathTrail
from .run_context import CourtEvents, RunContext, Selection
from .video_decode import Decoder, OpenCVDecoder, open_decoder
from .video_io import ThreadedFrameReader, VideoWriter

COLOR_ROI_THIN = (0, 255, 0)


def _open_writer(ctx: RunContext, path: str) -> VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    video = ctx.video
    return VideoWriter(path, video.fps, video.width, video.height, ctx.cfg, encoder=ctx.encoder)


def _open_decoder(ctx: RunContext, cached: Sequence[np.ndarray]) -> Decoder:
    """The clip reopened at the frame after the cached prefix, frame-exact: NVDEC seeks by its
    frame index, OpenCV walks from an early seek to the cached pictures (``video_decode``).

    OpenCV's own seek lands on another picture on a variable-frame-rate clip (pomona's seek to
    frame 689 decodes 683), which pass 2 used to answer by decoding the whole clip again.
    """
    try:
        decoder = open_decoder(ctx.video.path, start_frame=len(cached), host=True, reference=cached,
                               prefer_nvdec=ctx.cfg.gpu_decode)
    except RuntimeError as error:
        raise RuntimeError(f"Could not reopen the input video for pass 2: {ctx.video.path}") from error
    if isinstance(decoder, OpenCVDecoder) and decoder.synced_by == "seek-missed":
        print(f"[pass 2] The cached frames were not found near the seek to frame {len(cached)}; "
              f"decoding from the start and skipping the {len(cached)} cached frames")
    return decoder


class _FrameSource:
    """Pass-2 frames: the prefix of the clip pass 1 cached, from RAM, and the rest from a decode
    that starts where the cache ends.

    The decode runs on a reader thread ahead of the renderer, as in pass 1, so a frame's
    decoding overlaps the drawing of the one before it.  Pass 2 is the cache's last reader: every
    frame ``read`` hands out is the renderer's to draw on, and a cached frame leaves the cache as
    it is handed out, so its memory is freed once it is written.
    """

    def __init__(self, ctx: RunContext, cached: Optional[List[np.ndarray]], frame_count: int) -> None:
        self._cached: List[Optional[np.ndarray]] = cached if cached is not None else []
        self._n_cached = len(self._cached)
        self._reader: Optional[ThreadedFrameReader] = None
        if cached is not None and self._n_cached >= frame_count:
            print(f"[pass 2] Using RAM frame cache ({self._n_cached} frames)")
            return
        decoder = _open_decoder(ctx, cached or [])
        if self._n_cached:
            print(f"[pass 2] Using RAM frame cache ({self._n_cached} of {frame_count} frames), "
                  f"decoding the rest ({decoder.backend})")
        else:
            print(f"[pass 2] Decoding the clip ({decoder.backend})")
        self._reader = ThreadedFrameReader(decoder, prefetch=max(2, ctx.cfg.frame_reader_prefetch))

    def read(self, frame_index: int) -> Optional[np.ndarray]:
        """Frame ``frame_index``, the renderer's to draw on; None once the decode runs dry."""
        if frame_index < self._n_cached:
            frame, self._cached[frame_index] = self._cached[frame_index], None
            return frame
        return self._reader.read() if self._reader is not None else None

    def release(self) -> None:
        if self._reader is not None:
            self._reader.release()


class _GroundModelCache:
    """The court-plane homography for the latest valid keypoints, rebuilt only when they move.

    It starts empty: before the first court detection there is no court plane, so nothing is
    drawn with one. Seeding it with the clip's last keypoints put the end-of-clip homography on
    the opening frames, wrong whenever the camera moved.
    """

    def __init__(self) -> None:
        self.keypoints: Optional[List[float]] = None
        self._key: Optional[Tuple[int, ...]] = None     # court_keypoints_key of the model's keypoints
        self.model = None

    @property
    def key(self) -> Optional[Tuple[int, ...]]:
        """What the current model was built from (``court_keypoints_key`` of its keypoints)."""
        return self._key

    def update(self, court_kps: Optional[List[float]]):
        if court_kps and len(court_kps) >= MIN_KEYPOINT_VALUES:
            self.keypoints = court_kps
        key = court_keypoints_key(self.keypoints)
        if key != self._key:
            self.model = build_ground_projection_model(self.keypoints)
            self._key = key
        return self.model


class _ProtectMaskCache:
    """The court-side protect mask for the motion-debug overlay, per keypoint set."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self._key: Optional[Tuple[int, ...]] = None     # court_keypoints_key of the mask's keypoints
        self.mask: Optional[np.ndarray] = None
        self._first = True

    def get(self, court_kps: Optional[List[float]]) -> Optional[np.ndarray]:
        key = court_keypoints_key(court_kps)
        if self._first or key != self._key:
            self.mask = build_court_side_protect_mask(self.height, self.width, court_kps) if court_kps else None
            self._key = key
            self._first = False
        return self.mask


class _Writers:
    """The video outputs this run asked for: the tracking video, the motion-debug video and the
    track-debug video, each None when it is off."""

    def __init__(self, ctx: RunContext) -> None:
        cfg = ctx.cfg
        self.main = self.motion_debug = self.track_debug = None
        if cfg.save_tracking_video:
            self.main = _open_writer(ctx, cfg.output_video)
            print(f"[writer] Tracking encoder: {self.main.encoder}")
        if cfg.save_motion_debug:
            self.motion_debug = _open_writer(ctx, cfg.output_debug_path)
        if cfg.save_track_debug_video:
            self.track_debug = _open_writer(ctx, cfg.output_track_debug_path)

    def close(self) -> None:
        for w in (self.main, self.motion_debug, self.track_debug):
            if w is not None:
                w.close()


class _FrameRenderer:
    """Draws and writes one frame after another, carrying the trails, the court-model caches
    and the marker sizes from frame to frame."""

    def __init__(self, ctx: RunContext, pass1: Pass1Result, selection: Selection, events: CourtEvents,
                 writers: _Writers) -> None:
        cfg, video = ctx.cfg, ctx.video
        self.cfg, self.video, self.timing = cfg, video, ctx.timing
        self.pass1, self.selection, self.events, self.writers = pass1, selection, events, writers
        self.ground_model = _GroundModelCache()
        self.court_layer = CourtLayer(cfg, video.height, video.width, selection.court_polygon)
        self.protect_masks = _ProtectMaskCache(video.height, video.width)
        self.sizer = MarkerSizer(cfg)
        self.search_smoother = SearchCentreSmoother()
        self.ball_trail = BallTrail(cfg, video.fps, video.width, video.height)
        self.path_trail = DisplayPathTrail(video.fps, video.width)
        # A single-detection track is a record for the diagnostics, not a trail: its one dot would
        # scatter the detector's strays over the video.
        self.candidate_trails = CandidateTrackTrails([t for t in selection.all_tracks if len(t.observations) >= 2],
                                                     selection.chosen_track, video.fps)
        self.raw_motion_trails = RawMotionTrails(video.width)

    def _draw(self, fi: int, frame: np.ndarray, result, court_kps, draw_in_place: bool) -> Canvas:
        """The tracking frame and, when the track-debug video is on, its debug twin.

        ``draw_in_place`` draws the tracking frame onto ``frame`` itself instead of a copy: a
        decoded frame nothing else reads (6 MB copied per frame otherwise).
        """
        cfg, pass1, selection = self.cfg, self.pass1, self.selection
        canvas = Canvas(frame if draw_in_place else frame.copy(),
                        frame.copy() if self.writers.track_debug is not None else None)
        model = self.ground_model.update(court_kps)

        self.court_layer.draw(canvas.both, court_kps, model, self.ground_model.key)
        if cfg.draw_players:
            draw_player_boxes(canvas.both, pass1.player_boxes[fi])
        # Rejected proposals and ROIs belong only in the track-debug view.  Candidate tracks go
        # there too and, with show_all_tracks, lightly on the tracking frame; either way under
        # the chosen result and its trail so the final choice sits on top.
        draw_raw_detections(canvas.debug_only, pass1.detections[fi])
        draw_motion_rois(canvas.debug_only, pass1.rois[fi])
        if canvas.debug is not None or cfg.show_all_tracks:
            self.candidate_trails.advance(fi)
        if canvas.debug is not None:
            self.candidate_trails.draw(canvas.debug, fi, DEBUG_TRACK_STYLE)
        if cfg.show_all_tracks:
            self.candidate_trails.draw(canvas.main, fi, LIGHT_TRACK_STYLE)
        if result is not None:
            draw_selected_result(canvas, result, fi, self.sizer, self.search_smoother, cfg.draw_search_regions)
        # The trail's break limits follow the court's extent on screen; until the first detection
        # they use the clip's last keypoints, which is the one place the static court still serves.
        self.ball_trail.advance(result, self.ground_model.keypoints or selection.last_court_keypoints)
        self.ball_trail.draw(canvas, cfg.draw_ball_trail)
        if cfg.draw_court_minimap:
            draw_court_minimap(canvas.main, model, pass1.player_boxes[fi], self.events.rally_legs, fi, self.video.fps,
                               contact_events=self.events.contacts)

        # The track-debug stream follows the chosen track's display path while
        # the selector has nothing; the current result keeps it continuous.
        path_point = selection.display_path.get(fi)
        if path_point is None and result is not None:
            path_point = (float(result.cx), float(result.cy), True)
        self.path_trail.advance(path_point)
        if canvas.debug is not None and result is None:
            self.path_trail.draw(canvas.debug, path_point)
        draw_hud(canvas, fi, result, selection.chosen_track)
        return canvas

    def _write_motion_debug(self, fi: int, frame: np.ndarray, result, court_kps) -> None:
        cfg, pass1, timing = self.cfg, self.pass1, self.timing
        write_t0 = timing.start()
        raw_m = pass1.raw_motions[fi].unpack() if pass1.raw_motions[fi] is not None else None
        boost_m = pass1.boost_masks[fi].unpack() if pass1.boost_masks[fi] is not None else None
        rois = pass1.rois[fi]
        vis = draw_motion_masks(frame, raw_m, boost_m, cfg, court_keypoints=court_kps,
                                protect_mask=self.protect_masks.get(court_kps), rois=rois)
        probe_style = cfg.debug_probe_motion_style
        if not probe_style:
            for rx1, ry1, rx2, ry2 in rois or ():
                cv2.rectangle(vis, (int(rx1), int(ry1)), (int(rx2), int(ry2)), COLOR_ROI_THIN, 1)
        if result is not None:
            draw_selected_ball(vis, result, self.sizer.radius)
        self.raw_motion_trails.advance(fi, None if probe_style else rois, boost_m)
        self.raw_motion_trails.draw(vis, fi)
        self.writers.motion_debug.write(vis)
        timing.add("pass2_write_debug", write_t0)

    def render(self, fi: int, frame: np.ndarray, owned: bool = False) -> None:
        """Draw frame ``fi`` and write it to every output that is on.  ``owned``: ``frame`` is
        the renderer's to draw on (nothing else holds it), which saves the copy unless the
        motion-debug video needs the undrawn frame."""
        timing, writers = self.timing, self.writers
        render_t0 = timing.start()
        debug_write_elapsed = 0.0
        result = self.selection.per_frame[fi]
        court_kps = self.pass1.court_keypoints[fi]
        canvas = self._draw(fi, frame, result, court_kps, draw_in_place=owned and writers.motion_debug is None)

        if writers.track_debug is not None:
            write_t0 = timing.start()
            writers.track_debug.write(canvas.debug)
            debug_write_elapsed += timing.add("pass2_write_track_debug", write_t0)
        if timing.enabled:
            timing.values["pass2_render"] = (timing.values.get("pass2_render", 0.0)
                                             + max(0.0, time.perf_counter() - render_t0 - debug_write_elapsed))
        if writers.main is not None:
            write_t0 = timing.start()
            writers.main.write(canvas.main)
            timing.add("pass2_write_main", write_t0)
        if writers.motion_debug is not None:
            self._write_motion_debug(fi, frame, result, court_kps)


def render_pass2(ctx: RunContext, pass1: Pass1Result, selection: Selection, events: CourtEvents) -> int:
    """Write the selected video outputs; returns the number of frames rendered (0 when none is on)."""
    cfg, timing = ctx.cfg, ctx.timing
    n = pass1.frame_count
    if ctx.needs_pass2_outputs:
        print("[pass 2] Rendering selected video output(s)...")
    else:
        print("[pass 2] Skipped (no video outputs selected)")
    pass2_t0 = timing.start()

    writers = _Writers(ctx)
    rendered = 0
    if not ctx.needs_pass2_outputs:
        timing.add("pass2_total", pass2_t0)
        return rendered

    source = _FrameSource(ctx, pass1.cached_frames, n)
    renderer = _FrameRenderer(ctx, pass1, selection, events, writers)
    try:
        for fi in range(n):
            read_t0 = timing.start()
            frame = source.read(fi)
            if frame is None:
                break
            timing.add("pass2_read", read_t0)
            renderer.render(fi, frame, owned=True)
            rendered += 1
            if cfg.progress_every and (fi + 1) % cfg.progress_every == 0:
                print(f"  [pass2 {fi+1}/{n}] rendering...")
    finally:
        source.release()
        writers.close()
    if rendered < n:
        print(f"[pass 2] Warning: the second decode gave {rendered} of the {n} frames pass 1 saw; "
              f"the last {n - rendered} frame(s) are missing from the video output(s)")
    timing.add("pass2_total", pass2_t0)
    return rendered
