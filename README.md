# Triton Ball Tennis

A GPU tennis-video tracker built around GridTrackNet temporal ball detection,
TensorRT player and court detection, and conservative motion/physics recovery.
It produces an annotated MP4 and per-frame tracking JSON.

## Demo

[![GridTrackNet tennis tracking demo](media/tracking_demo.gif)](media/tracking_demo.mp4)

The overlay distinguishes where each selected point came from:

- **Green (`DET`):** a position supported by a GridTrackNet detection (may be smoothed)
- **Orange (`MOT`):** motion-supported recovery
- **Cyan (`PHY`):** short bounded interpolation or physics recovery

Raw detector coordinates are preserved in the JSON `measurement` field; `x/y`
are the selected, possibly smoothed position and `position_kind` distinguishes
measurement, filtered and recovered positions. Recovery is deliberately conservative: an
unresolved detector gap remains a gap instead of becoming a long predicted path.

## Pipeline

- GridTrackNet runs on raw RGB frames in five-frame temporal units at its native
  30 FPS cadence (30 or 60 FPS input is supported).
- Player and court models run through TensorRT.
- Motion masks support selection and short recovery but are not used to alter
  GridTrackNet's input.
- The ball-in-play selector rejects static objects and implausible track switches,
  closes safe one-frame holes, and exports the selected source for every frame.
- The renderer draws a short anti-aliased trail without inventing points across
  lost sections.

## Requirements

- Python 3.10 and an NVIDIA CUDA GPU
- CUDA-enabled PyTorch (install it from the PyTorch index for your CUDA version)
- NumPy, OpenCV (pip's `opencv-python` is enough)
- TensorRT 10 for the player and court engines
- BoxMOT for player tracking (required with the player engine; unset the engine path to run without players)
- CatBoost, only for `--court`
- SciPy and psutil, optional (motion-debug trail smoothing; automatic frame-cache sizing)
- Ultralytics, only for exporting new engines with `build_trt_engines.py`
- pytest, for the tests and `clean_tracker.py --self-test`
- FFmpeg recommended for faster video encoding

`requirements.txt` lists these with the versions the last review ran with
(recorded in `audit/records/snapshot.json`); it is not a full lock file:

```powershell
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

FilterPy is not needed: the selector's Kalman filter is plain NumPy.

The repository includes the GridTrackNet weight export, the CatBoost bounce
model and the TensorRT engines used by the default pipeline. TensorRT engines
are platform-specific and may need to be rebuilt for a different GPU or
TensorRT version.

## Usage

Run the bundled Pomona sample:

```powershell
python clean_tracker.py
```

Run another 30 or 60 FPS match:

```powershell
python clean_tracker.py --input path\to\match.mp4 --output output\tracking.mp4 --tracking-json output\tracking.json
```

Export JSON without rendering a video:

```powershell
python clean_tracker.py --input path\to\match.mp4 --tracking-json output\tracking.json --no-video
```

Use another CUDA device or detector threshold:

```powershell
python clean_tracker.py --device 1 --conf 0.55
```

## Fine-tuning GridTrackNet

The bundled GridTrackNet weights contain the latest fine-tuned checkpoint used
to produce the demo. The labelling and training workspace that produced them
(the label tool, the trainer, the Colab notebook, the hand-corrected labels and
the detector scorers over those labels) lives on the `train` branch, which
carries this tracker as well.

## Validation

Run the deterministic tracking checks and the synthetic regression tests:

```powershell
python clean_tracker.py --self-test
python -m pytest -q
```

Evaluate against annotations:

```powershell
python clean_tracker.py --no-video --annotations sample\pomona_annotations.json
```

`check_parity.py` is a strict gate against the frozen Pomona benchmark, and
`validate_tracking.py` scores an exported tracking JSON against hand labels.
Scoring the detector or the pipeline on the workspace's hand-labelled clips
(`evaluate_archive.py`) is on the `train` branch.

## Project structure

```text
clean_tracker.py                         Command-line entry point
validate_tracking.py                     Tracking JSON vs. hand labels (evaluation/validate.py)
check_parity.py                          Frozen Pomona benchmark gate (evaluation/parity.py)
evaluation/                              The scorer and the gate behind those two entry points
build_trt_engines.py                     YOLO checkpoint -> ONNX -> TensorRT engine builder
requirements.txt                         Python packages (see Requirements)
tennis_tracker/pipeline.py               Run entry point: orders the stages, calls the selector
tennis_tracker/run_context.py            Data the stages pass each other
tennis_tracker/pass1.py                  Decode, motion evidence, ROIs and detections per frame
tennis_tracker/frame_window.py           Previous/current/next frames for pass 1's temporal gate
tennis_tracker/pass2.py                  Render the tracking and debug videos
tennis_tracker/export.py                 What a run returns, and the tracking JSON export
tennis_tracker/tracking_schema.py        The tracking JSON's shape, shared by the writer and the scorers
tennis_tracker/motion.py                 CUDA background model and motion masks
tennis_tracker/roi_tracking.py           ROI tracking for motion search
tennis_tracker/headless.py               Running the tracker from another program, nothing drawn
tennis_tracker/gridtracknet.py           GridTrackNet model and decoder
tennis_tracker/video_io.py               Frame reader, video writer, CUDA uploads
tennis_tracker/config.py                 The run's Config; paths.py: model asset paths, spelled once
tennis_tracker/frame_rate.py             Frame-rate rules; utils.py: host-side helpers (FFmpeg, device)
tennis_tracker/engines/                  The TensorRT engines and the detector backends:
  ball_detectors.py                        GridTrackNet prepass and the TensorRT ball engine
  player_detector.py                       TensorRT player detection with ByteTrack association
  court_detector.py                        Court keypoint engine, run every N frames
  engine_session.py                        TensorRT session shared by the engine backends
  engine_files.py                          Engine files on disk: metadata, class names, I/O tensors
  engine_boxes.py                          Decoding detection-engine outputs into boxes
tennis_tracker/court/                    The court and what happens on it:
  court_geometry.py                        Court keypoints, side guides, polygon, homography
  court_scene.py                           CourtScene, the event types and the player-box queries
  ball_turns.py                            Where the ball track turns: sample windows, short gaps
  bounce_events.py                         CatBoost bounces and kinematic landing candidates
  contact_events.py                        Player contacts
  rally_legs.py                            Schematic rally legs on the court plane
  court_drawing.py                         Court keypoints, lines, net line and polygon
  court_minimap.py                         Court minimap drawing
tennis_tracker/render/                   Drawing for pass 2:
  rendering.py                             Frame drawing: canvas, markers, HUD
  trails.py                                Ball, display-path and candidate-track trails
  display_path.py                          The chosen track's display path
  motion_debug.py                          Motion-debug video drawing: masks, selected ball
  motion_trails.py                         Motion-debug video's raw-motion trails
  summaries.py                             Console summaries printed at the end of a run
ball_in_play_selector/                   Track construction, scoring, and recovery
models/gridtracknet_weights_torch.npz    GridTrackNet inference weights
models/ctb_regr_bounce.cbm               CatBoost bounce model (--court)
models/                                  Player and court TensorRT engines
sample/                                  Sample video and validation annotations
tests/                                   Synthetic regression tests (pytest)
audit/                                   Review reports and verification scripts
audit/records/                           Their logs and results (records, not code)
output/                                  Generated videos, JSON, and reports (ignored)
```

## Current limitation

GridTrackNet can still miss a blurred ball or select a player/background feature.
The conservative selector prevents many false continuations, but correcting raw
green detector errors requires fine-tuning the model on additional labeled clips.
