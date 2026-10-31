"""The tracking run: open inputs, pass 1, selection, court events, pass 2, export.

``run(cfg)`` is the entry point ``clean_tracker.py`` and ``evaluate_archive.py``
call.  Each stage takes and returns explicit data (see ``run_context``,
``pass1`` and ``pass2``); this module only orders them and reports.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import cv2
import torch
from ball_in_play_selector import select_ball_in_play

from .config import THREAD, Config
from .court.court_geometry import build_court_polygon, last_valid_court_keypoints
from .court.bounce_events import infer_kinematic_bounces, predict_bounce_events
from .court.contact_events import detect_player_contacts
from .court.court_scene import CourtScene
from .court.rally_legs import assemble_rally_legs
from .ball_models import WEIGHTS_SUFFIX
from .ball_weights import missing_weights
from .engines.ball_detectors import BallDetectorBackend, GridTrackNetBallDetector, TensorRTRuntimeBallDetector
from .engines.court_detector import CourtDetector
from .engines.engine_files import find_ball_class_id_from_names, read_engine_class_names, resolve_engine_path
from .export import RunResult, write_tracking_json
from .pass1 import Pass1Result, collect_pass1, erase_ghost_rois
from .pass2 import render_pass2
from .engines.player_detector import PlayerDetector, prefetch_bytetrack
from .render.display_path import build_display_path
from .run_context import CourtEvents, RunContext, Selection, Timing, VideoInfo
from .render.summaries import print_selector_track_summary, print_timing_summary
from .utils import EncoderChoice, choose_encoder, detect_device, print_platform_report, resolve_runtime
from .video_io import check_writable_video_path

_DEFAULT_FPS = 24.0                 # when the container reports none
PREPASS_WEIGHTS = frozenset(WEIGHTS_SUFFIX.values())     # the suffixes the five-frame prepass detector loads
# The display path interpolates across at least this much time, within these frame bounds.
_DISPLAY_PATH_GAP_SECONDS = 0.60
_DISPLAY_PATH_GAP_MIN_FRAMES, _DISPLAY_PATH_GAP_MAX_FRAMES = 10, 24


class _Output(NamedTuple):
    """One file a run can write: its role in ``RunResult.outputs``, the Config
    field that turns it on, the Config field that names its path, and how the
    run reports it."""

    role: str
    flag: str
    path: str
    label: str


_OUTPUTS = (
    _Output("tracking_json", "tracking_json", "tracking_json", "Tracking JSON"),
    _Output("tracking_video", "save_tracking_video", "output_video", "Tracking video"),
    _Output("motion_debug", "save_motion_debug", "output_debug_path", "Motion debug video"),
    _Output("detector_input", "save_detector_input_debug", "output_detector_input_debug_path",
            "Detector-input video"),
    _Output("track_debug", "save_track_debug_video", "output_track_debug_path", "Track-debug video"),
)


def _written_outputs(cfg: Config) -> Dict[str, Path]:
    """The files this run writes, by role, in the order they are reported."""
    return {output.role: Path(getattr(cfg, output.path)) for output in _OUTPUTS if getattr(cfg, output.flag)}


def _video_outputs(cfg: Config) -> Dict[str, Path]:
    """The videos this run writes, by role: every output but the tracking JSON."""
    return {role: path for role, path in _written_outputs(cfg).items() if role != "tracking_json"}


def _output_label(role: str) -> str:
    return next(output.label for output in _OUTPUTS if output.role == role)


def validate_io_paths(cfg: Config) -> Path:
    """The resolved input path; raises when an output would overwrite it or
    another output, or when a video output cannot be written (see
    ``video_io.check_writable_video_path``), so a bad path fails here and not
    at the first frame of pass 2."""
    input_path = Path(cfg.input_video).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    seen: Dict[Path, str] = {}
    videos = _video_outputs(cfg)
    for role, value in _written_outputs(cfg).items():
        label = _output_label(role)
        path = value.expanduser().resolve()
        if path == input_path:
            raise ValueError(f"{label} must not overwrite the input video: {path}")
        if path in seen:
            raise ValueError(f"{label} and {seen[path]} resolve to the same output path: {path}")
        if role in videos:
            check_writable_video_path(path, label)
        seen[path] = label
    return input_path


def _build_ball_detector(cfg: Config) -> BallDetectorBackend:
    """The five-frame prepass detector for ``.npz`` (GridTrackNet) or ``.pt`` (HeatmapNet)
    weights, else the TensorRT engine next to ``model_path``."""
    model_path = Path(cfg.model_path).expanduser().resolve()
    if model_path.suffix.lower() in PREPASS_WEIGHTS:
        if not model_path.is_file():
            raise missing_weights(model_path)
        return GridTrackNetBallDetector(str(model_path), cfg)
    engine_path = resolve_engine_path(cfg.model_path)
    if engine_path is None:
        raise RuntimeError(
            f"Ball TensorRT engine not found for model_path='{cfg.model_path}'. "
            "Expected a .engine file."
        )
    names = read_engine_class_names(engine_path)
    ball_cls_id, ball_cls_name = find_ball_class_id_from_names(names, cfg.ball_class_name)
    print(f"[init] Model classes: {names}")
    print(f"[init] Ball class: id={ball_cls_id}, name='{ball_cls_name}'")
    return TensorRTRuntimeBallDetector(str(engine_path), cfg, ball_cls_id, names=names)


class _EncoderProbe:
    """``utils.choose_encoder`` for a CUDA run, on a thread of its own; ``result`` waits for it and
    raises what it raised."""

    def __init__(self, cfg: Config) -> None:
        self._choice: Optional[EncoderChoice] = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, args=(cfg,), name="encoder-probe", daemon=True)
        self._thread.start()

    def _run(self, cfg: Config) -> None:
        try:
            self._choice = choose_encoder(cfg, is_cuda=True)
        except BaseException as error:  # noqa: BLE001 - re-raised by result()
            self._error = error

    def result(self) -> EncoderChoice:
        self._thread.join()
        if self._error is not None:
            raise self._error
        assert self._choice is not None
        return self._choice


class _RunSetup(NamedTuple):
    """What a run settles before its detectors are built (``_setup_run``)."""
    cfg: Config                     # with the runtime resolved
    input_path: Path
    device: torch.device
    encoder_probe: Optional[_EncoderProbe]
    timing: Timing
    started_at: float
    init_t0: float


def _setup_run(cfg: Config) -> _RunSetup:
    """Validate the config and pick the device; start the encoder probe and boxmot's import."""
    input_path = validate_io_paths(cfg)
    started_at = time.time()
    timing = Timing(cfg.info_timing)
    init_t0 = timing.start()
    if cfg.player_model_path:
        prefetch_bytetrack()        # overlaps boxmot's import with everything below

    device_str, _ = detect_device(cfg.device)
    if not str(device_str).isdigit():
        raise RuntimeError("The tracking runtime requires an NVIDIA CUDA device index such as --device 0.")
    device_index = int(device_str)
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} is unavailable.")
    # The caller's Config is not changed. A run that writes no video gets no encoder: FFmpeg is
    # not probed or dry-run for it. A run that does probes on a thread (FFmpeg's encoder list and an
    # NVENC dry run, ~0.4 s of subprocesses) while the detectors are built; the platform report
    # follows once the choice is in.
    cfg, _ = resolve_runtime(cfg, str(device_index), writes_video=False)
    encoder_probe = _EncoderProbe(cfg) if _video_outputs(cfg) else None
    # The run owns the process: let cuDNN autotune the fixed-shape convolutions.
    torch.backends.cudnn.benchmark = True
    return _RunSetup(cfg, input_path, torch.device(f"cuda:{device_index}"), encoder_probe, timing, started_at, init_t0)


def _open_inputs(setup: _RunSetup, detector: BallDetectorBackend) -> RunContext:
    """Open the clip, start the ball prepass and build the court and player detectors. The caller
    owns `detector` (and closes it); the clip is released here if this fails."""
    cfg, input_path = setup.cfg, setup.input_path
    cap = cv2.VideoCapture(str(input_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        video = VideoInfo(
            path=input_path,
            fps=cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count_hint=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        print(f"[init] Video: {video.width}x{video.height} @ {video.fps:.1f}fps, ~{video.frame_count_hint} frames")
        # A prepass on a worker thread starts after the court and player engines are built and the
        # player engine's CUDA graph is captured: a capture is invalidated by CUDA work from another
        # thread. A prepass in the foreground runs first instead, so boxmot's import (prefetch_bytetrack,
        # ~2 s that PlayerDetector waits for) finishes behind it; so does a prepass process, which returns
        # at once and runs beside the engines' builds (its own CUDA context cannot break a capture here).
        threaded_prepass = detector.prepass_mode == THREAD
        if not threaded_prepass:
            detector.prepare_video(input_path, video.fps, video.width, video.height, video.frame_count_hint)
        court_detector = CourtDetector(cfg)
        player_detector = PlayerDetector(cfg, court_keypoints=None, fps=video.fps)
        if threaded_prepass:
            detector.prepare_video(input_path, video.fps, video.width, video.height, video.frame_count_hint)
        encoder = setup.encoder_probe.result() if setup.encoder_probe is not None else None
        print_platform_report(cfg, encoder)
    except BaseException:
        cap.release()
        raise
    return RunContext(
        cfg=cfg,
        video=video,
        capture=cap,
        device=setup.device,
        encoder=encoder,
        detector=detector,
        court_detector=court_detector,
        player_detector=player_detector,
        timing=setup.timing,
        started_at=setup.started_at,
        init_t0=setup.init_t0,
    )


def _select(ctx: RunContext, pass1: Pass1Result) -> Selection:
    """Run the ball-in-play selector over pass 1's evidence and derive the display path."""
    cfg, video, timing = ctx.cfg, ctx.video, ctx.timing
    n = pass1.frame_count
    selector_t0 = timing.start()
    poly_t0 = timing.start()
    court_polygon, last_kps = build_court_polygon(pass1.court_keypoints)
    timing.add("selector_build_poly", poly_t0)

    print("[selector] Running ball-in-play selection...")
    select_t0 = timing.start()
    filtered_before = pass1.boost_masks.filter_seconds
    per_frame, chosen_track, all_tracks = select_ball_in_play(
        pass1.detections, video.fps, video.width, video.height,
        court_polygon=court_polygon,
        boost_masks=pass1.boost_masks,
        raw_motions=pass1.raw_motions,
        player_boxes_by_frame=pass1.player_boxes,
        debug=cfg.print_selector_tracks,
        court_keypoints_by_frame=pass1.court_keypoints)
    timing.add("selector_select", select_t0)
    # The boost masks pass 1 left unfiltered are filtered as the selector first reads them.
    timing.set("selector_boost_cc", pass1.boost_masks.filter_seconds - filtered_before)

    print(f"[selector] {len(all_tracks)} total tracks")
    post_t0 = timing.start()
    if cfg.print_selector_tracks:
        print_selector_track_summary(all_tracks, chosen_track, total_frames=n,
                                     limit=max(cfg.selector_track_limit, 0))
    # The selector owns its bounded tail; the display path is for the debug video only.
    path_gap = max(
        max(1, cfg.display_path_max_gap),
        min(_DISPLAY_PATH_GAP_MAX_FRAMES,
            max(_DISPLAY_PATH_GAP_MIN_FRAMES, int(round(float(video.fps) * _DISPLAY_PATH_GAP_SECONDS)))),
    )
    display_path = build_display_path(chosen_track, n, max_interp_gap=path_gap,
                                      frame_w=video.width, frame_h=video.height)
    if cfg.save_track_debug_video:
        # Where the chosen track's path is sparse, fill from the final per-frame
        # output so late-rally segments stay visible.  Never from other tracks:
        # the path must only reflect the chosen track's positions.
        for fi, res in enumerate(per_frame):
            if fi in display_path or res is None:
                continue
            display_path[fi] = (res.cx, res.cy, res.source in ("det", "motion"))
    timing.add("selector_post", post_t0)
    timing.add("selector_total", selector_t0)
    return Selection(per_frame, chosen_track, all_tracks, court_polygon, last_kps, display_path)


def _court_events(ctx: RunContext, pass1: Pass1Result, selection: Selection) -> CourtEvents:
    """Bounces, contacts and rally legs for the minimap and the tracking JSON; not computed
    when neither ``cfg.court_events`` nor ``cfg.draw_court_minimap`` asks for them."""
    cfg, video = ctx.cfg, ctx.video
    if not (cfg.court_events or cfg.draw_court_minimap):
        return CourtEvents()
    if not cfg.bounce_model_path:
        raise RuntimeError("--court requires Config.bounce_model_path")
    t0 = time.perf_counter()
    scene = CourtScene(selection.per_frame, pass1.player_boxes, video.fps, video.width, video.height)
    bounces = predict_bounce_events(scene, cfg.bounce_model_path)
    contacts = detect_player_contacts(scene)
    kinematic_bounces = infer_kinematic_bounces(scene)
    rally_legs = assemble_rally_legs(scene, pass1.court_keypoints, contacts, bounces, kinematic_bounces)
    inferred_legs = sum(leg["quality"] == "inferred" for leg in rally_legs)
    volleys = sum(leg["outcome"] == "volley" for leg in rally_legs)
    print(
        f"[court events] {len(bounces.confirmed)} confirmed bounce(s) "
        f"({len(bounces.candidates)} candidate, {len(kinematic_bounces)} kinematic), "
        f"{len(contacts)} player contact(s), {len(rally_legs)} leg(s) "
        f"({inferred_legs} inferred, {volleys} volley) "
        f"in {time.perf_counter() - t0:.3f}s"
    )
    return CourtEvents(computed=True, bounces=list(bounces.confirmed), bounce_candidates=list(bounces.candidates),
                       kinematic_bounces=list(kinematic_bounces), contacts=list(contacts), rally_legs=list(rally_legs))


def _finish(ctx: RunContext, pass1: Pass1Result, selection: Selection, events: CourtEvents,
            rendered: int) -> RunResult:
    """Assemble the result, write it as the tracking JSON and report what was produced."""
    cfg = ctx.cfg
    n = pass1.frame_count
    elapsed = ctx.elapsed()
    result = RunResult(
        per_frame=list(selection.per_frame),
        chosen_track=selection.chosen_track,
        all_tracks=list(selection.all_tracks),
        frame_count=n,
        court_keypoints=list(pass1.court_keypoints),
        player_boxes=list(pass1.player_boxes),
        last_valid_court_keypoints=last_valid_court_keypoints(pass1.court_keypoints),
        elapsed_sec=elapsed,
        timing=dict(ctx.timing.values),
        pass2_frames_rendered=rendered,
        outputs=_written_outputs(cfg),
        court_events=events,
    )
    if cfg.tracking_json:
        write_tracking_json(ctx, pass1, result)
    filled = result.filled_frames
    print(f"\n[done] {filled}/{n} frames filled ({100*filled/max(1,n):.1f}%)")
    print(f"[done] {elapsed:.1f}s total")
    for role, path in result.outputs.items():
        print(f"[done] {_output_label(role)}: {path}")
    if ctx.timing.enabled:
        print_timing_summary(ctx.timing.values, elapsed, n, rendered)
    return result


def run(cfg: Config) -> RunResult:
    """Track the ball through ``cfg.input_video``, write the configured outputs
    and return what was found (see ``RunResult``)."""
    setup = _setup_run(cfg)
    detector = _build_ball_detector(setup.cfg)
    try:    # whatever ends the run (an init failure, Ctrl-C, an error), a prepass process is stopped
        ctx = _open_inputs(setup, detector)
        pass1 = collect_pass1(ctx)
        detector.finish_prepass()           # a prepass process exits while the selector runs
        erase_ghost_rois(pass1)
        selection = _select(ctx, pass1)
        events = _court_events(ctx, pass1, selection)
        rendered = render_pass2(ctx, pass1, selection, events)
        return _finish(ctx, pass1, selection, events, rendered)
    finally:
        detector.close()
