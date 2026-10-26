# Project guide

## Purpose and entry points

This is an offline GPU tennis-video tracker. `clean_tracker.py` runs detection,
selection and optional rendering (`--weights` points it at other ball-detector
weights, `.npz` or `.pt`). Outputs are an annotated MP4 and per-frame JSON.
It is not a calibrated 3D ball-flight or line-calling system.

The labelling and GridTrackNet fine-tuning workspace (`finetune/`, its launchers,
the Colab notebook, the hand-corrected labels, and the scorers over those labels:
`evaluate_archive.py`, `score_court_events.py`, `track_breaks.py`) lives on the
`train` branch, which carries this tracker as well. This branch, `main`, is the
tracker alone; do not add workspace code here.

- `tennis_tracker/gridtracknet.py`: PyTorch port and NPZ loader. Five raw RGB
  frames at 768x432 become 15 input channels. Output is five confidence/x-offset/
  y-offset grids at 48x27. Width-axis normalization reproduces the upstream export;
  it is Keras BatchNormalization on that axis: gamma and beta train, mean and
  variance are running statistics (buffers).
- `tennis_tracker/heatmapnet.py`: the second ball-detector architecture, `HeatmapNet`,
  a U-shaped encoder-decoder with skip connections (`encode`, `decode`) answering on a
  192x108 grid (stride 4) in GridTrackNet's output layout, from the same 15-channel
  input; `.pt` weights (`{"arch": <ARCH>, "state_dict"}`, the record `MemoryNet` shares).
  `tennis_tracker/memorynet.py`: the third, `MemoryNet(HeatmapNet)`, arch `memory_s4`: a
  convolutional GRU at the bottleneck carries a state between consecutive units of one
  phase; `forward(value, state=None) -> (output, new_state)`; `out` starts at zero so a
  network built `from_heatmap` answers as its parent until the memory trains.
  `tennis_tracker/ball_models.py` is the one front door: `arch_of` reads the
  architecture off `Config.model_path` (`.npz` GridTrackNet; a `.pt` names
  `heatmap_s4` or `memory_s4` inside), `load_ball_model` builds it, and all three
  classes carry `ARCH`, `STATEFUL`, `GRID_COLS`, `GRID_ROWS`, `INPUT_CHANNELS` and
  `FRAMES_PER_UNIT`, so the detector, `decode_predictions` (which reads the grid off the
  output's shape) and the trainer never ask which class they hold; the prepass and the
  trainer branch on `STATEFUL` alone (one unit per phase per call with the phase's state;
  chains of `--chain` units, `dataset.ChainDataset`, `loop.chain_loss`). The trainer's
  `--arch heatmap_s4` (Gaussian target, CenterNet focal loss) trains it from scratch or
  fine-tunes it from a `.pt`; `--arch memory_s4 --weights <heatmap_s4 .pt>` starts the
  memory network from its parent. No run of either has been made yet; `README.md`
  ("Keeping one network") says what to delete when an architecture loses.
- `tennis_tracker/engines/`: `ball_detectors.py`, the progressive GridTrackNet
  prepass and the single-frame TensorRT ball engine. At about 60 FPS, even and
  odd phases each run at about 30 FPS. Preserve source-frame indices and
  temporal order. `engine_session.py`: the TensorRT session the ball, player
  and court engines share; `engine_files.py`: locating engine files, their
  Ultralytics metadata and I/O tensor table (no CUDA); `engine_boxes.py`:
  decoding detection-engine outputs into boxes for the ball and player
  backends. `player_detector.py`: TensorRT players with ByteTrack association.
  `court_detector.py`: court keypoints every N frames, last good result cached.
  `tennis_tracker/paths.py`: the model asset paths every consumer (Config
  defaults, scripts) spells once. `headless.py`: the one Config and run for a
  caller that wants the tracker's answer and no video.
- `tennis_tracker/pipeline.py`: orders the stages, calls the selector and, with
  `--court` (`Config.court_events`, implied by the minimap), the court events,
  reports. `run_context.py`: data the stages pass each other (`CourtEvents`:
  the bounces, candidates, kinematic bounces, contacts and legs found). `pass1.py`
  (with `frame_window.py`): decoding, background/motion state, ROI collection,
  detector calls. `pass2.py`: rendering pass. `export.py`: `RunResult`, what a
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
- `evaluation/`: `validate.py` scores a tracking JSON against annotations in native
  pixels (`check_parity` freezes its numbers); `parity.py` is the Pomona gate.
  `validate_tracking.py` and `check_parity.py` at the root are thin entry points
  to them.
- `build_trt_engines.py`: YOLO checkpoint to ONNX to TensorRT engine.

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

Use distinct output paths for new runs. `check_parity.py` verifies a frozen Pomona
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
