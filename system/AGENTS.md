# Project guide

## Purpose and entry points

This is an offline GPU tennis-video tracker. `clean_tracker.py` runs detection,
selection and optional rendering (`--weights` points it at other ball-detector
weights, `.npz` or `.pt`; the default, `models/ball_detector.pt`, is the heat-map
network's control run of 2026-10-26, and `models/legacy/` keeps the GridTrackNet
`.npz` files and the old YOLO `ball.engine`). Outputs are an annotated MP4 and
per-frame JSON. It is not a calibrated 3D ball-flight or line-calling system.

The labelling and GridTrackNet fine-tuning workspace (`finetune/`, its launchers,
the Colab notebook, the hand-corrected labels, and the scorers over those labels:
`evaluate_archive.py`, `score_court_events.py`, `track_breaks.py`) lives on the
`train` branch, which carries this tracker as `system/`, an exact copy of this
tree. This branch, `main`, is the tracker alone; do not add workspace code here.

- `tennis_tracker/gridtracknet.py`: PyTorch port and NPZ loader. Five raw RGB
  frames at 768x432 become 15 input channels. Output is five confidence/x-offset/
  y-offset grids at 48x27. Width-axis normalization reproduces the upstream export;
  it is Keras BatchNormalization on that axis: gamma and beta train, mean and
  variance are running statistics (buffers).
- `tennis_tracker/heatmapnet.py`: the second ball-detector architecture, `HeatmapNet`,
  a U-shaped encoder-decoder with skip connections (`encode`, `decode`) answering on a
  192x108 grid (stride 4) in GridTrackNet's output layout, from the same 15-channel
  input; `.pt` weights (`{"arch": <ARCH>, "state_dict"}`).
  `tennis_tracker/ball_models.py` is the one front door: `arch_of` reads the
  architecture off `Config.model_path` (`.npz` GridTrackNet; a `.pt` names
  `heatmap_s4` inside, and a record naming another architecture is refused),
  `load_ball_model` builds it, and both classes carry `ARCH`, `GRID_COLS`,
  `GRID_ROWS`, `INPUT_CHANNELS` and `FRAMES_PER_UNIT`, so the detector and
  `decode_predictions` (which reads the grid off the output's shape) never ask which
  class they hold. The trainer that produced the `.pt` (`--arch heatmap_s4`: Gaussian
  target, CenterNet focal loss, from scratch or fine-tuned from a `.pt`) and its runs
  are on the `train` branch (its ledger, `audit/records/gridtracknet_recipe_2026-10.md`);
  `README.md` ("Keeping one network") says what to delete when an architecture loses,
  and that `memory_s4` and `heatmap_s2` were removed on 2026-10-26 after the held-out
  comparison (the ledger has the runs).
- The fast candidates on the same contract (`.pt` records, all registered in
  `ball_models`; untrained until the `train` branch runs them): `heatmap_lite.py`
  (`heatmap_lite`, HeatmapNet slimmed), `streamnet.py` (`stream_s4`: a shared encoder per
  frame transition, `encode_frame` once per frame and `decode` per unit from four cached
  transitions, equal to the unit forward pass; the prepass streams it through
  `streaming_halves` unless `Config.ball_prepass_streaming` is off) and `wasbnet.py` (`wasb_5f`, WASB-SBDT's
  small HRNet at stride 4). `repvgg.py` holds their RepVGG blocks and the fold to plain
  convolutions: `ball_models.deployed(model)` is the inference form (export it, not the
  trainable one `load_ball_model` returns); `ball_models.load_deployed(weights, device)` folds
  from the float32 record, so each fused weight is rounded to FP16 once. Measurements: `audit/records/opt_detector_archs.md`
  on the `optimization` branch.
- `tennis_tracker/engines/`: `ball_detectors.py`, the progressive GridTrackNet
  prepass and the single-frame TensorRT ball engine. At about 60 FPS, even and
  odd phases each run at about 30 FPS. Preserve source-frame indices and
  temporal order. `Config.gridtracknet_window_step` (1, the default since
  2026-10-26) is the frames of a phase between unit starts: below 5 a frame's
  overlapping units are averaged before one decode (`_UnitStream`, and
  `HeatRing` in `ball_gpu.py`); 5 is the old prepass, one unit per frame.
  `ball_gpu.py` is the prepass's device side (OpenCV's resize to the byte, the heat
  ring, peaks, `TransitionRing`, the cached transition features a streaming network's units are
  decoded from, and the one kernel that turns NVDEC's NV12 pictures into the model
  input with the same bytes; `tennis_tracker/cuda_kernels.py` compiles such kernels
  with the NVRTC PyTorch ships and caches the cubins; `video_decode.py` is NVDEC
  and OpenCV decoding with OpenCV's pixels); `ball_trt.py` builds and caches the ball model's TensorRT engine
  (`Config.ball_prepass_runtime`), or a streaming network's encoder and decoder engines; `ball_prepass_process.py` runs the prepass in a
  child process. The prepass runs before pass 1 by default
  (`audit/records/opt_ball_prepass.md` on the `optimization` branch has why).
  `engine_session.py`: the TensorRT session the ball, player
  and court engines share; `engine_files.py`: locating engine files, their
  Ultralytics metadata and I/O tensor table (no CUDA); `engine_boxes.py`:
  decoding detection-engine outputs into boxes for the ball and player
  backends. `build_engines.py`: YOLO checkpoint to ONNX to TensorRT engine, run
  as `python -m tennis_tracker.engines.build_engines`. `player_detector.py`:
  TensorRT players with ByteTrack association.
  `court_detector.py`: court keypoints every N frames, last good result cached.
  `tennis_tracker/paths.py`: the model asset paths every consumer (Config
  defaults, scripts) spells once. `headless.py`: the one Config and run for a
  caller that wants the tracker's answer and no video.
- `tennis_tracker/pipeline.py`: orders the stages, calls the selector and, with
  `--court` (`Config.court_events`, implied by the minimap), the court events,
  reports. `run_context.py`: data the stages pass each other (`CourtEvents`:
  the bounces, candidates, kinematic bounces, contacts and legs found). `pass1.py`
  (with `frame_window.py`): decoding, background/motion state, ROI collection,
  detector calls; the fixed-shape per-frame GPU work (frame conversion, motion
  planes, background update) replays from CUDA graphs (`Config.pass1_cuda_graphs`). `pass2.py`: rendering pass. `export.py`: `RunResult`, what a
  run returns, and the tracking JSON writer, whose `court_events` block carries
  every event with its provenance (`{}` when they were not computed).
  `tracking_schema.py`: the tracking JSON's shape (TypedDicts, typing only),
  which the writer and the scorers share. `motion.py`, `roi_tracking.py`: CUDA
  motion masks and ROI tracking.
- `tennis_tracker/court/`: `court_geometry.py`, keypoints, side guides, polygon,
  homography; the court-event stages, each reading one `CourtScene`:
  `court_scene.py` (`CourtScene`, the event types and the player-box queries),
  `ball_turns.py` (where the ball track turns: sample windows and short gaps),
  `bounce_events.py` (CatBoost/kinematic bounce candidates), `contact_events.py`
  (player contacts) and `rally_legs.py` (schematic rally legs);
  `court_drawing.py`, the court layers; `court_minimap.py`, the minimap.
- `tennis_tracker/render/`: `rendering.py` draws only: the canvas, colours,
  markers and HUD, with `trails.py` (ball, display-path and candidate trails),
  `display_path.py` (the chosen track's display path), `motion_debug.py`
  (the motion-debug video's masks and selected ball) and `motion_trails.py` (its
  raw-motion trails) beside it. `summaries.py`: the console track table
  and timing report.
- `ball_in_play_selector/`: candidate association (`association.py`), Kalman
  state, track scoring, timeline selection, bounded recovery, trajectory refit.
  Its changes are measured on the `train` branch (`finetune/scoring/replay_selector.py` and
  `audit/records/selector_robustness_2026-10.md` there). A minimal selector without the
  score and the timeline chain was built and rejected on 2026-10-27: it cannot choose
  between two balls on screen.
- `evaluation/`: `validate.py` scores a tracking JSON against annotations in native
  pixels (the parity gate freezes its numbers); `parity.py` is the Pomona gate.
  Both run as modules from the root: `python -m evaluation.validate` and
  `python -m evaluation.parity`.

## Environment and commands

Run commands from the repository root. The verified local interpreter is
`C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe`. Plain `python` currently
resolves to a different Python installation. Other machines should use their own
environment, not copy this user-specific path into application code.

Runtime dependencies include CUDA PyTorch, NumPy, OpenCV and TensorRT;
BoxMOT supplies player tracking. FilterPy is not one: the selector's Kalman filter
is NumPy (`ball_in_play_selector/physics.py`), and `tests/test_ball_kalman.py` only
compares against FilterPy when it is installed. CatBoost is needed for `--court`.
Ultralytics is used by engine export. FFmpeg/NVENC can accelerate rendering.
Versions and model/source hashes for the review are in `audit/records/snapshot.json`.
There is no complete locked installation manifest. TensorRT engines may be
GPU/version specific.

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m pytest -q
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --no-video --tracking-json output/new_review_tracking.json
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --court --no-video --tracking-json output/new_review_court_events.json
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m evaluation.parity
```

`python -m pytest -q` (pytest required) runs every test: all suites live in
`tests/` (`pytest.ini` sets `testpaths` and puts the root on the import path),
use synthetic inputs only and need no GPU. `clean_tracker.py --self-test` runs
the selector, court-event, ROI and GridTrackNet modules from `tests/` through
pytest. Put new tests in `tests/` as plain pytest functions (fixtures and
`parametrize`, not `unittest.TestCase`), shared synthetic fixtures in
`tests/_helpers.py`. `audit/` keeps the review's reports and verification
scripts; `audit/records/` holds their logs and results, and
`audit/records/review_checks_2026-09.py.txt` is the September 2026 defect
reproduction script, kept as text: it no longer runs.

Use distinct output paths for new runs. `python -m evaluation.parity` verifies a frozen Pomona
benchmark; the September 7, 2026 worktree fails several of its gates. Do not
silently refresh that baseline to make a change pass.

## Change discipline

Keep RGB detector inputs unchanged unless an experiment explicitly changes the
model contract. Keep observations distinguishable from fitted positions. Short
unsupported gaps should remain absent. Physics is presently an image-space
prior; do not present its extrapolations as measured 3D positions or speeds.

Do not promote or retrain weights as incidental cleanup; the weights in `models/`
are produced on the `train` branch. Architecture experiments remain separate from
correctness fixes; do not promote a model implicitly.

See `audit/PROJECT_REVIEW.md` before modifying evaluation, prepass publication,
motion masks or physics. `audit/records/review_checks_2026-09.py.txt` records
how the September 2026 defects were reproduced; its pointer to
`tests/test_fixes.py` is out of date: the tests that assert the fixes live next
to each module's tests (`tests/test_<module>.py`). Convert relevant checks to
desired-behavior regression tests when making fixes.

The disabled manual perspective scaling, homography gravity, color gate,
flicker suppression and raw-component experiments were subsequently removed.
Keep automatic FPS/resolution normalization and supported debug outputs.
See `audit/CLEANUP_RESULTS.md` for rationale and measured output differences.
