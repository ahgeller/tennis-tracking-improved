from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from tennis_tracker.ball_weights import DEFAULT_CONF
from tennis_tracker.paths import COURT_ENGINE, BALL_WEIGHTS, MODELS, PLAYER_ENGINE

# Where the ball prepass runs (Config.ball_prepass_mode; engines/ball_detectors.prepass_mode resolves AUTO).
PrepassMode = Literal["auto", "foreground", "thread", "process"]
AUTO: PrepassMode = "auto"
FOREGROUND: PrepassMode = "foreground"
THREAD: PrepassMode = "thread"
PROCESS: PrepassMode = "process"
PREPASS_MODES: Tuple[PrepassMode, ...] = (AUTO, FOREGROUND, THREAD, PROCESS)


@dataclass
class Config:
    # paths
    input_video: str = r"input_videos/TMP.mp4"
    output_video: str = "output_videos/prof_test.mp4"
    # The ball detector: a .pt record (``ball_models`` reads its architecture; the default is stream_s4),
    # .npz GridTrackNet, .engine TensorRT.
    model_path: str = str(BALL_WEIGHTS)

    # Ball detector
    ball_class_name: Optional[str] = None
    # The default detector's gated threshold (ball_weights.py; a caller naming other weights passes theirs,
    # ball_weights.operating_conf). heatmap_s4 runs at 0.40 (0.50 until 2026-10-27: on mixed_v1 0.40 gave +92
    # hits for +2 false alarms), stream_s4 at 0.42.
    conf: float = DEFAULT_CONF
    device: str = "auto"
    # Shift applied to GridTrackNet points, in px at 1080p (positive = down).
    # Measured on the 41 hand-verified clips video13-53: with 0 the median error
    # is 2.3 px (p90 5.8) at every confidence threshold; with 1.6 it was 3.0 px
    # (p90 6.5), the detections sitting a median 2 px below the labels.
    gridtracknet_y_offset_px: float = 0.0
    gridtracknet_source_stride: Optional[int] = None  # None = 2 at 60 FPS / 1 at 30 FPS (the training cadence)
    # Where the five-frame prepass runs: "foreground" (before pass 1, alone), "thread" (a worker thread,
    # beside pass 1), "process" (a child process, engines/ball_prepass_process.py, beside pass 1 and from
    # the start of init) or "auto": the process for a network that streams (stream_s4 with
    # ball_prepass_streaming), the foreground for the others. The answers are the same in every mode.
    # In the child, stream_s4's prepass leaves init (1.0 to 2.4 s shorter than heatmap_s4's foreground
    # one, audit/records/opt_abplus_s4.md) and costs pass 1 up to 0.4 s beside it; the windowed
    # heatmap_s4 prepass, ~4 s, slowed pass 1 by more than that (opt_ball_prepass.md item 3,
    # opt_abplus.md item 3).
    ball_prepass_mode: PrepassMode = AUTO
    # Frames of a phase between one five-frame unit's start and the next: 1 starts a unit at every
    # frame and averages the up-to-five answers each frame gets (the default since 2026-10-26: on
    # heat4's weights, held out, false alarm 0.085 to 0.073 at recall 0.894 to 0.900, and through the
    # pipeline 0.178 to 0.164 at 0.898 to 0.906, for 1.5x the prepass time); 5 answers every frame
    # from one unit, the prepass as it was before.
    gridtracknet_window_step: int = 1
    # What runs the prepass's network: "tensorrt", an FP16 engine built from the weights on first
    # use and cached in ball_engine_dir (engines/ball_trt.py; 1.2 to 1.6 ms a unit against 3.8 to 4.1 ms
    # eager on the throttled RTX 5080 laptop), or "torch", the FP16 PyTorch model. TensorRT falls back to
    # torch, with a message, when it is not installed or cannot build the model.
    ball_prepass_runtime: str = "tensorrt"
    ball_engine_batch: int = 4          # units per engine call (the batch sweep in audit/records/opt_ball_prepass.md)
    # A network that can stream (stream_s4) encodes each frame's transition once and decodes every
    # unit from the cached features (ball_gpu.TransitionRing), as two engines; off, it runs unit by
    # unit like any other. The units are the same and their answers agree to FP16 rounding
    # (audit/records/opt_stream_prepass.md on the optimization branch).
    ball_prepass_streaming: bool = True
    ball_engine_dir: str = str(MODELS / "engine_cache")
    # The prepass decodes with NVDEC (PyNvVideoCodec) where it opens the clip, and its frames stay on
    # the GPU; else cv2 decodes into pinned memory. Both give the same bytes (video_decode.py).
    ball_prepass_gpu_decode: bool = True

    # TensorRT
    trt_async_execute: bool = True
    trt_async_slots: int = 3
    # Replay the per-frame player engine launch from a CUDA graph (see TensorRTRuntimeSession):
    # same kernels, 0.02 ms of CPU per frame instead of 1.9 ms. Needs trt_async_execute.
    trt_cuda_graph: bool = True
    # Replay pass 1's fixed-shape per-frame GPU work (each frame's conversion and motion planes, the
    # background update) from CUDA graphs: the same kernels, a fraction of the CPU launch cost
    # (frame_window.GraphedFrameConversion). Off while the prepass runs on a worker thread, whose
    # CUDA work would break a capture.
    pass1_cuda_graphs: bool = True
    # Build pass 1's motion masks with two CUDA kernels (tennis_tracker/motion_gpu.py) instead of some
    # forty PyTorch ops a box: the same bits, a few launches (audit/records/opt_abplus.md). Off, or
    # when NVRTC cannot build them, the PyTorch ops run.
    motion_kernels: bool = True

    # Player & court detection
    player_model_path: Optional[str] = str(PLAYER_ENGINE)
    court_model_path: Optional[str] = str(COURT_ENGINE)
    # Above 1, the player engine and ByteTrack run on every Nth frame only and pass 1
    # interpolates the boxes between (PlayerDetector). Kept at 1: N=3 cut pass 1 by about
    # 1.2 ms/frame but lost one of 27 labelled landing hits (audit/records/opt_pass1.md).
    player_detect_interval: int = 1
    court_detect_interval: int = 400
    court_conf: float = 0.10
    print_court_raw: bool = False
    court_remap_semantic_14: bool = False
    court_points_only: bool = False
    court_draw_indices: bool = False
    player_conf: float = 0.21
    player_iou: float = 0.10
    draw_players: bool = True
    draw_court: bool = True
    # Court events: ball landings, racket contacts and rally legs from the selected track
    # (``tennis_tracker/court``), exported as the tracking JSON's ``court_events`` block.
    # Drawing the minimap implies computing them; --court turns both on.
    court_events: bool = False
    draw_court_minimap: bool = False
    bounce_model_path: Optional[str] = None       # the CatBoost bounce model, needed when they are computed
    num_players: int = 4

    # Detector-input preprocessing (HSV boost of motion blobs, dimming of static
    # pixels). GridTrackNet reads raw frames, so these only shape the TensorRT
    # ball-engine input and the debug views.
    enable_preprocess: bool = True
    pre_sat_boost: float = 1.55
    pre_val_boost: float = 1.04
    pre_hue_shift: float = 0.18
    dim_static: float = 0.88
    static_sat_scale: float = 0.75
    motion_dilate: int = 5

    # Background model and motion mask. The background is an exponential mean
    # and variance of the S and V channels; a pixel is motion when its squared
    # difference exceeds motion_thresh^2 + motion_k_std^2 * variance.
    background_alpha: float = 0.02         # background/variance update rate at static pixels
    # Update rate at pixels that were motion on the previous frame. 0.015 is
    # below background_alpha, so moving objects are absorbed into the
    # background more slowly there; a parked ball still settles within a few
    # seconds instead of being frozen as a permanent ghost.
    motion_freeze_alpha: float = 0.015
    motion_thresh: float = 11.0            # additive floor of the threshold, in 0-255 units
    motion_k_std: float = 3.0              # std-dev multiplier of the variance term; balanced against motion_thresh so variance suppresses static-ball halos
    motion_v_min: float = 40.0             # V below this is never motion; low enough to keep dim balls in shadow
    motion_raw_temporal_gate: bool = True  # frame-to-frame delta test before raw foreground becomes motion evidence
    motion_raw_temporal_hi: float = 18.0
    motion_raw_temporal_lo: float = 8.0
    motion_raw_temporal_very_hi: float = 36.0
    motion_raw_close_size: int = 2
    motion_raw_open_size: int = 0
    # Components above boost_max_blob_area px are not ball-sized: a 1200 px
    # blob (radius ~19 px) overshoots the ball.
    boost_max_blob_area: int = 600
    boost_min_blob_area: int = 0

    # Skip-frame ball detection: run the detector every Nth frame and let the
    # selector interpolate the gaps.
    skip_frame_detect: int = 1          # 1 = every frame
    skip_frame_require_roi: bool = True
    skip_preprocess_dim: bool = True
    aux_detect_on_detect_frames: bool = True
    aux_force_interval: int = 6
    frame_reader_prefetch: int = 8
    # Pass 2 decodes past the frame cache with NVDEC (PyNvVideoCodec) where it opens the clip, else
    # OpenCV (tennis_tracker/video_decode.py; both give the same pixels). Off: pass 2 wants host
    # frames, and NVDEC's GPU work slowed its NVENC writes (pomona pass 2: 3.84 s with NVDEC,
    # 3.62 s with OpenCV, median of 3 locked runs). NVDEC pays where the frames stay on the GPU.
    gpu_decode: bool = False

    # ROI-based motion - only compute motion/CC near ball
    roi_motion_enabled: bool = True
    roi_visible_radius_frac: float = 0.1
    roi_lost_radius_frac: float = 0.2
    roi_lost_expand_per_frame: float = 0.0022 # grow lost ROI by this * diag per lost frame
    roi_max_radius_frac: float = 0.085       # cap ROI radius
    roi_motion_bleed_frac: float = 0.0       # extend the underlying cropping boundary for motion processing to capture long blurred streaks without affecting the tight visual box
    roi_fullframe_interval: int = 15        # while lost, probe full frame every N frames; 0 disables

    # Blob shape filtering of the boost mask.
    blob_shape_filter: bool = True
    blob_max_aspect: float = 4.0
    blob_preserve_tiny: bool = True       # components up to blob_tiny_max_area px bypass the aspect test (a tennis ball is a few px across)
    blob_tiny_max_area: int = 120

    # outputs / debug
    save_tracking_video: bool = True
    save_motion_debug: bool = False
    save_detector_input_debug: bool = False
    output_debug_path: str = "output_videos/prof_test_motion_debug.mp4"
    output_detector_input_debug_path: str = "output_videos/prof_test_detector_input_debug.mp4"
    debug_show_raw_motion: bool = False
    # Per-frame motion diagnostics in the tracking JSON.  Nothing in the
    # normal path reads them and they cost ~1.7 ms/frame (mask unpack plus
    # findContours on every frame), so they are opt-in.
    motion_diagnostics: bool = False
    debug_probe_motion_style: bool = True
    # Track-debug video: court geometry, every candidate track, the raw
    # detections, the motion ROIs and the chosen track's display path.
    save_track_debug_video: bool = False
    output_track_debug_path: str = "output_videos/prof_test_track_debug.mp4"
    # Longest gap (frames) the chosen track's display path is interpolated
    # across in that video; pass 2 raises it to 0.6 s at the clip's FPS.
    display_path_max_gap: int = 12
    print_selector_tracks: bool = True
    selector_track_limit: int = 0  # 0 = print all
    draw_search_regions: bool = False
    draw_ball_trail: bool = True
    # Every selector track on the tracking video too, each in its track colour, under the
    # chosen track's trail and marker (``--showtracks``; the track-debug video always has them).
    show_all_tracks: bool = False

    # trail rendering / switch guards
    trail_hard_switch_x_frac: float = 0.30
    trail_hard_switch_y_frac: float = 0.30
    ball_marker_box_scale: float = 0.22
    ball_marker_min_radius: int = 3
    ball_marker_max_radius: int = 6

    # output encoding
    use_nvenc: bool = True
    nvenc_preset: str = "p1"
    nvenc_bitrate: str = "8M"
    use_async_writer: bool = True
    async_queue: int = 16
    cache_input_frames_pass2: bool = True  # keep decoded frames in RAM to avoid second decode
    # 0 sizes the pass-2 frame cache from the RAM free at start-up (half of
    # it, capped by pass2_cache_ceiling_mb). The cache is a prefix of the clip:
    # pass 1 keeps frames while they fit the budget and pass 2 decodes only the
    # rest, so a budget smaller than the clip (a 1080p clip needs 12-22 GiB)
    # still saves that share of the second decode.
    pass2_cache_max_mb: int = 0
    pass2_cache_ceiling_mb: int = 4096
    progress_every: int = 200
    info_timing: bool = False
    tracking_json: Optional[str] = None
