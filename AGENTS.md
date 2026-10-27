# Project guide: the train branch

## Layout

This branch is the tennis-video tracker plus its labelling and GridTrackNet
fine-tuning workspace.

- `system/` is the tracker: `main`'s whole tree, byte for byte (its `README.md`,
  `AGENTS.md`, `tests/`, `audit/`, `models/`, `tennis_tracker/`,
  `ball_in_play_selector/`, `evaluation/` with `validate.py` and `parity.py`,
  `clean_tracker.py`). `system/AGENTS.md` is the map of that code and its change
  discipline; read it before touching anything under `system/`.
- Everything else is training and labelling: `finetune/` (the workspace, the trainer,
  the scorers), the root scripts `evaluate_archive.py`, `score_court_events.py` and
  `track_breaks.py` that front the scorers, `tests/` (the workspace suites),
  `audit/` (the workspace's ledgers and records) and the launchers
  `finetune.ps1` / `finetune.bat`.

**The invariant.** `system/` equals `main`: after every change is ported,
`git rev-parse main^{tree}` and `git rev-parse train:system` print the same hash
(`git diff --stat main^{tree} train:system` is empty). So `system/` holds nothing
`main` would not: no workspace code, no train-only docs (they go at the root), and
nothing under it imports `finetune` (`tests/test_system_layout.py` checks that and
that nothing at the root shadows a system package). A tracker change is made in one
place and ported to the other, as its own commit with the same message (in Git Bash:
a PowerShell pipe before 7.4 re-encodes the patch):

```sh
# train -> main: on main, the system/ part of train commits <a>..<b>
git diff --binary <a> <b> -- system | git apply -p2
# main -> train: on train, main commits <a>..<b>
git diff --binary <a> <b> | git apply --directory=system
```

**Imports.** The train code imports the tracker's packages (`tennis_tracker`,
`ball_in_play_selector`, `evaluation`) as top-level packages. `finetune/__init__.py`
is the one place that puts `system/` on `sys.path`, and every way in passes through
it: modules import `finetune.*`, and each script's `__main__` bootstrap puts the
repository root on the path and imports `finetune`. `tennis_tracker.paths.ROOT` is
therefore `system/` (the weights are `system/models/`); `finetune.REPOSITORY` is the
repository root, which holds `finetune/` and `output/`. The Colab code zip
(`finetune/colab_zip.py`) packs files at their repository paths, `system/` included,
so the same `__init__` sets the path up there.

## The workspace code

- `finetune/refiner/refine.py`: model-free sub-pixel ball suggestions for the label tool
  (candidates chosen by trajectory, trust score); `motion_blobs.py` measures what moved
  on one frame (the change map, its motion centroid and blobs).
- `finetune/refiner/cnn_refiner.py`: small CNN, trained only on frames the user reviewed, that places
  those suggestions where the user clicks (`refiner_training.py` runs the training: the folds,
  the "sure" threshold and the promotion check, on crops `refiner_crops.py` gathers, leaving out
  `exclude.txt` clips and the clips `review_status.json` lists as in progress (`in_progress_clips`),
  and networks `refiner_fitting.py` fits and scores; `hints.py`
  is their command line, the script `ft.py hints` runs). Versions live in
  `finetune/models/refiner/` (all kept, `current.txt` names the active one); retrain with
  `ft.py hints train` or menu option 10. Its self-check decides "trusted" (threshold set on
  folds of clips it never saw); refine.py still finds the ball and decides ball/no ball.
  In the label tool an unreviewed frame's suggestion is the mark SPACE takes (sure ones
  always, unsure ones within 3 px of the draft; see `Suggestions.offer_for` in
  `label_hints.py` for the measurement);
  `x` keeps the draft, `y` takes a hint beside a reviewed label. `ft.py hints sheet <clip>`
  (`refiner_sheet.py`) renders a zoomed PNG of where two versions disagree, for judging marks by eye. A trajectory
  prior on the hints was measured and makes them worse (see the module docstring).
- `finetune/ft.py`: workflow front door (launchers, script commands;
  `menu.py` is the interactive menu, `commands.py` has the in-process commands,
  `colab_zip.py` the `colab-zip` command, the one zip
  `GridTrackNet_FineTune_Colab.ipynb` unpacks, `interpreter.py` the venv
  hand-over). The rest of the package is five
  subpackages. `finetune/labeling/`: `label_tool.py` (manual edits: the
  actions the key table names, the window and the loop), split into
  `label_session.py` (the one owner of the sitting's state: the rows, the
  cursor and its walk, the view geometry, the toggles, the suggestions, which
  the views and handlers read and write directly),
  `label_cli.py` (its command line), `label_store.py` (a clip's rows and
  sidecars), `label_source.py` (one clip of a labelling run, and the prefetcher
  that decodes ahead of the cursor), `label_window.py` (window plumbing and the
  key tables), `label_cursor.py` (parking the OS pointer on the ball),
  `label_keys.py` (what the keys do), `label_mouse.py` (what the mouse does),
  `label_frame_view.py` (the single-frame picture and HUD), `label_grid.py` and
  `label_sheet.py` (the grid and trajectory-sheet views, each owning its state,
  picture and keys), `label_draw.py` (the filters, marks and text every view
  draws with) and `label_hints.py` (the suggestions). `finetune/refiner/`: the
  suggestion model above (`refine.py` with `motion_blobs.py`, `cnn_refiner.py` with its torch network class in
  `refiner_network.py`, `refiner_training.py` with `refiner_crops.py` and `refiner_fitting.py`,
  `refiner_sheet.py`, `hints.py`). `finetune/data/`: `labels.py`, the label
  file format; `label_check.py`, checks a label file against its video's facts;
  `label_audit.py`, checks labels against the raw detector; `label_flags.py`, the frames an
  audit flagged for a person (the flag sidecar the label tool and the menu read); `detect.py`,
  the raw GridTrackNet prepass over one clip (torch only when a detector is built), which
  the audit, `realign.py`, `pretrack.py`, `finetune/scoring/archive.py` and the trainer's winner
  metric all run; `bounce_labels.py`, the landing labels (`labels/<clip>_bounces.csv`, the
  frames where the ball landed, `frame_###.5` for a landing between two frames, points
  optional) the label tool's `/` key writes and the court-event scorer reads;
  `workspace.py`, where the workspace keeps things and a clip's video
  facts, and the one place `clips.csv` and `exclude.txt` are read and written
  (`TENNIS_WORKSPACE` relocates the workspace); `detector_metrics.py`, the detector metric (`DetectorMetrics`);
  `data_policy.py`, which clips may be trained, validated or scored on;
  `holdouts.py`, the named held-out sets of whole recordings in `finetune/holdouts.json`
  (`mixed_v1`: prof games 4 and 7, video1 to 10 and three grid matches, 24 clips), which
  the trainer's `--holdout NAME` holds out and the evaluator's `--holdout NAME` scores;
  `fileio.py`, atomic workspace writes; `alignment.py`, the one label-vs-detector
  agreement the audit and `realign.py` share; `import_data.py` (its importers;
  `import_sources.py` holds what they share) and `realign.py` (with the pure
  solver in `realign_solve.py`), source conversion and temporal realignment. `finetune/training/`:
  `train_gridtracknet.py` (the training run: its command line, data, epochs,
  the per-epoch `checkpoint.pt` that `--resume` continues from, and the winner;
  `finetune/train_gridtracknet.py` at the old path only hands over
  to it, for the command below), `loop.py` (the loss, one epoch, the validation
  loss and saving weights), `metrics.py` (the detector metric a run is judged
  by and the winner rule: `best.npz` is the epoch the metric prefers, scored after
  every `--score-every` epoch, and `false_alarm_units`, the hard negatives), `clips.py`
  (which clips it trains on and holds out: `--val-clips`, `--holdout NAME`, or nothing
  with `--all-clips`), `frame_cache.py` (the per-clip cache of frames and training
  units: lossless 768x432 PNGs through `gridtracknet.resize_to_model`, the tracker's
  own resize, keyed on the video, about 8 GB for the 64 trainable clips, measured
  7.9 GB), `dataset.py`
  (targets, the named augmentations behind `--augment`, the Dataset and the sampler
  with `--oversample-absent` and `--oversample-false-alarms`) and `pretrack.py` (drafts).
  `--arch heatmap_s4` trains the tracker's `HeatmapNet` (Gaussian target, CenterNet focal
  loss) from scratch or fine-tunes it from a `.pt`; its recipe is versioned (`HEAT_RECIPE`,
  `heat_recipe` in config.json: offsets at weight 1, the convolutions BatchNorm follows held
  at their Kaiming norm after every step by `loop.hold_conv_scale`, `loop.check_finite` after
  every epoch; heat1 to heat3, the runs that led there, are in the ledger). `README.md`
  ("Keeping one network") says what to delete when an architecture loses.
  `audit/records/gridtracknet_recipe_2026-10.md` is the ledger of the recipe runs.
- `finetune/scoring/`: the scorers over the workspace labels. `archive.py` scores the
  detector or the pipeline on workspace clips with `finetune/data/detector_metrics.py`
  (`pipeline_score.py` scores one pipeline run's clip; raw mode also prints a `macro`
  per-clip-mean line, one `source=<name>` line per source, the count beside each
  false-alarm rate and, with `--cell-histogram`, recall by position within the grid cell;
  `--window-step` scores either prepass of `Config.gridtracknet_window_step`),
  `court_events_score.py` scores a `--court` pipeline run's landings against the landing
  labels (tiers, a 0.1 s match, `--compare`, and `--recompute` to rerun the event stages on
  a run's frames without the GPU), `compare_eval_logs.py` sets two evaluator logs side by
  side, `track_breaks.py` buckets a pipeline run's misses by cause and `replay_selector.py`
  reruns the selector on the CPU over a pipeline run's recorded detections and scores it
  (the chain, every track, or the detector alone), the tool a selector change is measured
  with before a GPU run; `audit/records/selector_robustness_2026-10.md` is the ledger of
  the selector changes. The root
  `evaluate_archive.py`, `score_court_events.py` and `track_breaks.py` are thin entry points
  to `archive`, `court_events_score` and `track_breaks`; the rest run as
  `python -m finetune.scoring.<module>`.

## Environment and commands

Run commands from the repository root, except the tracker's own gate and suite,
which run from `system/` as they do on `main`. The verified local interpreter is
`C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe`. Plain `python` currently
resolves to a different Python installation. Other machines should use their own
environment, not copy this user-specific path into application code.

The tracker's dependencies are in `system/AGENTS.md`. The label tool also needs
conda-forge's Qt build of OpenCV (`requirements.txt`, the workspace's superset of
`system/requirements.txt`).

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m pytest -q
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' system/clean_tracker.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/train_gridtracknet.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' system/clean_tracker.py --no-video --tracking-json output/new_review_tracking.json
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode raw --clips video13
cd system; & 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m evaluation.parity
```

`python -m pytest -q` at the root (pytest required) runs every test: the workspace
suites in `tests/` and the tracker's in `system/tests/` (the root `pytest.ini` names
both and puts the root and `system/` on the import path); `cd system; python -m pytest -q`
runs the tracker's alone, as on `main`. All use synthetic inputs only and need no GPU.
Root `tests/` is not a package: `system/tests` is the package `tests`, so a workspace
suite imports the tracker's helpers as `tests._helpers` and the workspace's own
(the headless label tool, the trainer probes) from `_workspace_helpers`, and
`system/tests/conftest.py`'s fixtures (`fake_cuda`) reach only the tracker's suites.
Put new workspace tests in `tests/` as plain pytest functions (fixtures and
`parametrize`, not `unittest.TestCase`). `train_gridtracknet.py --self-test` is also
run by `tests/test_train_gridtracknet.py`.

Always specify `--archive finetune` for workspace evaluation: the evaluator can
otherwise prefer a separate Desktop archive. Use distinct output paths for new
runs. `python -m evaluation.parity` verifies a frozen Pomona benchmark; do not
silently refresh that baseline to make a change pass.

## Label review status - October 25, 2026

Every labelled clip in the workspace is reviewed and finished: the Pomona clip,
`video1` through `video53`, and every `grid_match` clip with a label file
(1 to 31 and 36 to 92, including the ones once listed as needing work).
`finetune/review_status.json` lists all 86 under `verified_clips` and has an
empty `in_progress_clips`; it is the one record of this, so a clip added later is
unreviewed until it is added there. The user confirmed this on 2026-10-25 after
watching `output/label_review_all.mp4`, a zoomed back-to-back pass over every
label (rendered by a one-off script, not part of the package).

Any reviewed clip may be used for measurement, validation and model selection;
video13 and up remain the standard set. These rules concern evaluation; do not
silently rewrite labels or training exclusions as part of a review. Label files
for grid_match and video13+ clips are gitignored, so back them up before any
bulk change.

## Data and change discipline

Label CSVs use `frame_###,ball_x,ball_y`, native image pixels, zero-based source
frame indices, and a top-right sentinel for invisible balls. Labels on 60 FPS
clips may use either odd or even phase. Missing labels are not absent-ball
labels. Reviewed, edited and suspect flags live in sidecar JSON files.

Preserve hand edits, phase, source mapping and backups. Do not revert uncommitted
edits to the training/labeling tools, labels or manifests, or promote/retrain
weights, as incidental cleanup. `clips.csv` notes, `exclude.txt` and review sidecars
convey different things: passing the format checker does not establish annotation
correctness.

Labels clicked in the label tool before commit 307f632 (2026-10-18) sat about half a
pixel up-left of the ball; commit 8c36b5e moved them back (backups:
`<csv>.before_halfpixel_fix`). Do not apply that correction again.

See `system/audit/PROJECT_REVIEW.md` before modifying evaluation, prepass publication,
label frame identity, realignment pins, motion masks or physics. The user has
authorized fixing the reviewed defects and running validation; convert relevant
checks to desired-behavior regression tests next to each module's tests. Architecture
experiments remain separate from the correctness fixes; do not promote a model
implicitly.
