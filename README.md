# Triton Ball Tennis

A GPU tennis-video tracker built around five-frame temporal ball detection
(HeatmapNet, with GridTrackNet still runnable), TensorRT player and court
detection, and conservative motion/physics recovery.
It produces an annotated MP4 and per-frame tracking JSON.

## Demo

[![GridTrackNet tennis tracking demo](media/tracking_demo.gif)](media/tracking_demo.mp4)

The overlay distinguishes where each selected point came from:

- **Green (`DET`):** a position supported by a ball detection (may be smoothed)
- **Orange (`MOT`):** motion-supported recovery
- **Cyan (`PHY`):** short bounded interpolation or physics recovery

Raw detector coordinates are preserved in the JSON `measurement` field; `x/y`
are the selected, possibly smoothed position and `position_kind` distinguishes
measurement, filtered and recovered positions. Recovery is deliberately conservative: an
unresolved detector gap remains a gap instead of becoming a long predicted path.

## Pipeline

- The ball detector runs on raw RGB frames in five-frame temporal units at its
  native 30 FPS cadence (30 or 60 FPS input is supported).
- Player and court models run through TensorRT.
- Motion masks support selection and short recovery but are not used to alter
  the ball detector's input.
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
- Ultralytics, only for exporting new engines with `python -m tennis_tracker.engines.build_engines`
- pytest, for the tests and `clean_tracker.py --self-test`
- FFmpeg recommended for faster video encoding

`requirements.txt` lists these with the versions the last review ran with
(recorded in `audit/records/snapshot.json`); it is not a full lock file:

```powershell
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

FilterPy is not needed: the selector's Kalman filter is plain NumPy.

The repository includes the ball-detector weights (`models/ball_detector.pt`, and
the GridTrackNet export in `models/legacy/`), the CatBoost bounce model and the
TensorRT engines used by the default pipeline. TensorRT engines
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

See what the selector chose from:

```powershell
python clean_tracker.py --input path	o\match.mp4 --showtracks
```

`--showtracks` draws every selector track on the tracking video in its own colour, chosen
or not, under the chosen track's green trail and marker.

## Ball detector weights: two networks

The tracker runs either of two ball-detector networks and picks by the weights
file. Nothing else changes: same video, same JSON, same selector.

| weights file | network | status |
|---|---|---|
| `.pt` naming `heatmap_s4` (default: `models/ball_detector.pt`) | HeatmapNet, 192x108 heat map (`tennis_tracker/heatmapnet.py`) | the bundled detector |
| `.npz` (`models/legacy/gridtracknet_finetuned.npz`) | GridTrackNet, 48x27 grid (`tennis_tracker/gridtracknet.py`) | the previous default, run with `--weights` |

```powershell
python clean_tracker.py --input path\to\match.mp4                                                    bundled HeatmapNet
python clean_tracker.py --input path\to\match.mp4 --weights models\legacy\gridtracknet_finetuned.npz  GridTrackNet
```

The tracker picks by the suffix and, for a `.pt`, by the `arch` the file names
inside; a `.pt` naming an architecture that no longer exists is refused by name.

The prepass starts a five-frame unit at every frame of a phase and averages the
answers each frame gets before decoding it once (`Config.gridtracknet_window_step`,
1; the default since 2026-10-26, when it measured fewer false alarms at higher
recall on held-out clips, raw and through the selector, for 1.5x the prepass
time). Setting it to 5 is the prepass as it was, one unit per frame. The training
playbook with that comparison is the README of the `train` branch; the section
below says what to delete should one of the two lose (the trainer items live on
`train`).

## Keeping one network

Architectures exist so that one can be chosen; the ones that lose are deleted,
not kept behind a flag. Two were removed on 2026-10-26 after the held-out
comparison: `memory_s4` (HeatmapNet with a convolutional GRU carrying a state
between consecutive units of one phase) and `heatmap_s2` (HeatmapNet with one
more decoder block, answering at stride 2). The heat control matched the memory
network and the stride-2 curve equalled the control's, so neither earned its
weight; the runs and their numbers are in the `train` branch's ledger,
`audit/records/gridtracknet_recipe_2026-10.md`, which keeps their history. A
`.pt` still names its architecture inside, so a record of either is refused by
name (`ball_models.arch_of`).

Two remain, and for each the whole of it is (the trainer's share lives on `train`):

- `gridtracknet`: `tennis_tracker/gridtracknet.py` stays (its loader, resize and
  decoder serve every architecture); its entry in `tennis_tracker/ball_models.py`
  (`GRIDTRACKNET` in `ARCHITECTURES`, `MODEL_CLASSES`, `WEIGHTS_SUFFIX`, `_MODULES`,
  the `.npz` branch of `arch_of`), the `GridTrackNet` class and its NPZ `load_model`,
  `models/legacy/`, and `tests/test_gridtracknet.py` with the GridTrackNet cases of
  the ball-models and detector tests. On `train`: `gridtracknet_loss` with the
  `.npz` branch of `save_weights`, the one-hot target, the `.npz` template handling
  of `--weights`, the notebook's `ARCH` option and the GridTrackNet cases of the
  trainer tests.
- `heatmap_s4`: `tennis_tracker/heatmapnet.py` (the `HeatmapNet` class and the
  `.pt` record functions), its entry in `ball_models` (`HEATMAP_S4` in
  `ARCHITECTURES`, `MODEL_CLASSES`, `WEIGHTS_SUFFIX`, `_MODULES`, the record branch
  of `arch_of`), `models/ball_detector.pt`, and `tests/test_heatmapnet.py` with the
  heatmap cases of the ball-models and detector tests. On `train`: `heatmap_loss`
  with the `.pt` branch of `save_weights`, the Gaussian target, the trainer's
  `--arch` choice and `HEAT_RECIPE`, the notebook's `ARCH` option and the heatmap
  cases of the trainer and loop tests.

With one `.pt` architecture left, `read_record` with `arch=None` and `arch_of`'s
record read stay only so that a stale record is refused by name; if that is not
wanted, `arch_of` can go back to the suffix alone.

The winner's weights are promoted by copying them over the default path in
`tennis_tracker/paths.py` (`BALL_WEIGHTS`, `models/ball_detector.pt`, what
`Config.model_path` defaults to), which `ft.py promote` on `train` does with a
backup. On 2026-10-26 that file became the heat-map control run's winner (ctrl1,
the ledger has the numbers); the GridTrackNet `.npz` files moved to
`models/legacy/` (`gridtracknet_finetuned.npz`, the previous default, and
`gridtracknet_upstream.npz`, the export it was fine-tuned from, untracked) beside
the old YOLO `ball.engine`, and `clean_tracker.py --weights` still runs any of
them. Record every promotion in the ledger with its held-out numbers.

## The bundled weights and the training workspace

The bundled weights, `models/ball_detector.pt`, are the heat-map network's
control run of 2026-10-26 (ctrl1: ten epochs of fine-tuning on the from-scratch
run heat4), trained on the `train` branch with the `mixed_v1` recordings held
out (held out, dense prepass: recall 0.929 at false alarm 0.083 at threshold
0.50; the ledger there, `audit/records/gridtracknet_recipe_2026-10.md`, has the
rest). `models/legacy/`
keeps the GridTrackNet files, among them the fine-tuned checkpoint that produced
the demo, and the old YOLO `ball.engine`. The labelling and training workspace
(the label tool, the trainer, the Colab notebook, the hand-corrected labels and
the detector scorers over those labels) lives on the `train` branch, which
carries this tracker as its `system/` folder.

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

`python -m evaluation.parity` is a strict gate against the frozen Pomona benchmark
(frozen on the GridTrackNet `.npz`, so with the default `.pt` it reports the new
detector's numbers as differences; it is not refreshed), and
`python -m evaluation.validate` scores an exported tracking JSON against hand
labels.
Scoring the detector or the pipeline on the workspace's hand-labelled clips
(`evaluate_archive.py`) is on the `train` branch.

## Project structure

```text
clean_tracker.py                         Command-line entry point
evaluation/                              Scoring, run as modules from the root:
  validate.py                              Tracking JSON vs. hand labels (python -m evaluation.validate)
  parity.py                                Frozen Pomona benchmark gate (python -m evaluation.parity)
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
tennis_tracker/ball_models.py            The ball-detector architectures: which a weights file names, building it
tennis_tracker/heatmapnet.py             HeatmapNet model and its .pt record (the default detector)
tennis_tracker/gridtracknet.py           GridTrackNet model, NPZ loader, and the decoder both share
tennis_tracker/video_io.py               Frame reader, video writer, CUDA uploads
tennis_tracker/config.py                 The run's Config; paths.py: model asset paths, spelled once
tennis_tracker/frame_rate.py             Frame-rate rules; utils.py: host-side helpers (FFmpeg, device)
tennis_tracker/engines/                  The TensorRT engines and the detector backends:
  ball_detectors.py                        Five-frame prepass (HeatmapNet or GridTrackNet) and the TensorRT ball engine
  player_detector.py                       TensorRT player detection with ByteTrack association
  court_detector.py                        Court keypoint engine, run every N frames
  engine_session.py                        TensorRT session shared by the engine backends
  engine_files.py                          Engine files on disk: metadata, class names, I/O tensors
  engine_boxes.py                          Decoding detection-engine outputs into boxes
  build_engines.py                         YOLO checkpoint -> ONNX -> TensorRT engine builder (python -m)
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
models/ball_detector.pt                  The ball detector the tracker runs (heatmap_s4)
models/legacy/                           GridTrackNet .npz files and the old YOLO ball.engine (--weights)
models/ctb_regr_bounce.cbm               CatBoost bounce model (--court)
models/                                  Player and court TensorRT engines
sample/                                  Sample video and validation annotations
tests/                                   Synthetic regression tests (pytest)
audit/                                   Review reports and verification scripts
audit/records/                           Their logs and results (records, not code)
output/                                  Generated videos, JSON, and reports (ignored)
```

## Current limitation

The ball detector can still miss a blurred ball or select a player/background feature.
The conservative selector prevents many false continuations, but correcting raw
green detector errors requires fine-tuning the model on additional labeled clips.
