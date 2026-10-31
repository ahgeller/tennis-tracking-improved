# AB+s4: stream_s4 by default, its prepass in a child process (2026-10-30)

Branch `phase-ABplus-s4` off `phase-ABplus-heatmap` at f43218a (the AB+heatmap baseline below).
The worktree is `phases/ABplus_s4`, made with `git worktree add` (treehouse cannot place a worktree
at a given path), data linked with `tools/link_worktree_data.py`, and
`system/models/candidates/stream_s4_e60.pt` copied from `phases/ABplus` (the link script does not
copy candidates). Tracker runs use the `tennis-opt` env, the evaluator and the root suite the
`tennis-analysis` env. Every timed and scoring run went through `tools/gpu_lock.py`.

## Results

### Shipped

| commit | change |
|---|---|
| 4a21ee3 | the real-engine streaming test skips without onnx (it failed in `tennis-analysis`) |
| bb07ecb | stream_s4 at 0.42 by default; its prepass in a child process from the start of init |
| 244dbf1 | the child is told to exit right after pass 1 and waited for at the end (0.7 s off pomona) |
| (this) | this record; the `Config.ball_prepass_mode` comment states the measured numbers |

### End to end, AB+heatmap (f43218a, heatmap_s4 at 0.40) vs AB+s4 (244dbf1, stream_s4 at 0.42)

`clean_tracker.py -i <clip> --info` from `system/`, each tree's default model, the eight variants
interleaved in one lock: a warm-up round, then 4 timed rounds, the order reversed each round.
Before and during each run `nvidia-smi` and every other python process's CPU were polled; no run
had a GPU neighbour or a busy python process, so none was discarded. Median [min-max], seconds:

| | pomona heatmap | pomona s4 | VolyAvw heatmap | VolyAvw s4 |
|---|---|---|---|---|
| `--no-video` total | 16.60 [16.50-17.30] | 16.60 [16.10-16.60] | 25.05 [24.80-26.20] | **23.10** [23.10-23.20] |
| - init | 6.68 [6.59-6.71] | **5.66** [5.35-5.92] | 8.38 [8.28-8.60] | **6.00** [4.39-6.05] |
| - prepass (in init / in the child) | 4.35 | 2.10 | 6.15 | 3.50 |
| - pass 1 | 9.14 [9.02-9.79] | 9.80 [9.77-10.27] | 15.05 [14.92-15.96] | 15.42 [15.18-16.91] |
| - selector | 0.64 | 0.76 | 1.39 | 1.45 |
| video total | 22.45 [22.20-24.30] | 22.85 [21.70-22.90] | 35.40 [34.60-35.90] | **32.95** [32.60-34.30] |
| - init | 6.73 | 5.72 | 8.38 | 5.83 |
| - pass 1 | 9.97 | 10.73 | 16.30 | 16.22 |
| - pass 2 | 4.91 | 5.30 | 9.28 | 9.33 |
| process wall, `--no-video` | 20.19 | 20.26 | 29.34 | 27.38 |

The laptop ran in its slow state for this suite (pomona `--no-video` was 11 s in the fast state
earlier in the day, for both trees). VolyAvw is 8% faster end to end (1.95 s, 2.45 s with video);
pomona is a tie (0.0 s, +0.4 s with video, inside the spread). The prepass leaves init in both
clips (1.0 s and 2.4 s off it), and pass 1 gives back 0.7 s on pomona and 0.4 s on VolyAvw
(next section). Process mode against stream_s4 in the foreground, single runs after the suite:
pomona 16.6 s against 18.3 s, VolyAvw 23.1 s against 25.8 s.

A first suite (17:54) overlapped another agent's unlocked CPU sweep and root pytest and a second
was slowed by the bench's own CPU polling; both were discarded.

### Accuracy gate

`evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1`, AB+s4 defaults (the
scorer prints `stream_s4_e60.pt at confidence 0.42`; the prepass in the child, 24 clips):

| | recall | wrong | false alarm (bridged / false) | median / p90 px | recall grid_match / video1-10 / video17+ |
|---|---|---|---|---|---|
| AB+heatmap, stream_s4 at 0.42 (baseline) | 0.953 | - | 95 (79 / 16) | - | .913 / .971 / .952 |
| AB+s4 default | **0.953** | 0.001 | **95** (79 / 16), 0.183 | 1.0 / 4.0 | **.913 / .971 / .952** |

### Identity: child process vs in-process

- The gate run again with `--prepass-mode foreground`: all 24 clips' `frames`, `tracks`,
  `chosen_track` and `court_events` equal the process-mode run's.
- pomona and VolyAvw, process mode against the prepass forced into the foreground: 0 of 2022 and 0
  of 3286 frames differ; with and without video, 0 differ.
- pomona AB+s4 default against AB+heatmap's tree run with `--weights stream_s4_e60.pt --conf 0.42`
  (foreground): 0 of 2022 frames differ.
- `test_ball_prepass_process.py`'s GPU test: `detect_clip` in the child and in-process give equal
  rows on a synthetic 40-frame 60 FPS clip.

### Off our court (`tools/off_court.py`, `--no-video`)

| clip | AB+s4 present / off | AB+heatmap present / off |
|---|---|---|
| VolyAvw | 2457 / 0 | 2578 / 0 |
| pomona | 1090 / 0 | 1075 / 0 |
| PomonaPitzer Women vs. UCSD | 3142 / 0 | - |

Vereinsduell (49878 frames) was not run.

## 1. Defaults

- `paths.STREAM_WEIGHTS` (`models/candidates/stream_s4_e60.pt`) is `BALL_WEIGHTS`, what
  `Config.model_path` defaults to. `paths.HEATMAP_WEIGHTS` (`models/ball_detector.pt`) stays the
  training workspace's model (`workspace.py`, `colab_zip.py`, `ft.py promote`).
- `tennis_tracker/ball_weights.py` keys each known file's gated threshold by path: stream_s4 0.42,
  heatmap_s4 0.40, 0.40 for any other file. `Config.conf` defaults to the default weights'
  threshold; `clean_tracker --conf` defaults to the threshold of the weights named (`--weights
  models/ball_detector.pt` runs at 0.40 without `--conf`); `--conf` still overrides. The archive
  scorer's `--gridtracknet-conf` and `pretrack`'s drafts follow the same rule.
- A missing known file fails with what it is, where it goes, and how to run the other detector
  (`ball_weights.missing_weights`).

## 2. The prepass process

`Config.ball_prepass_mode` (`auto`, `foreground`, `thread`, `process`) replaces the two
background/process flags; `auto` gives `process` to a network whose model class streams
(`streaming_halves`) with `ball_prepass_streaming` on, `foreground` to any other.

- **Overlap.** `RemotePrepass` starts the child when the detector is built and sends it the detector
  to build as soon as it connects, so the child loads its engines while the parent builds court,
  player and its own; `prepare_video` runs before court and player in process mode, so the clip goes
  over right after the video opens. Pass 1 consumes rows as they arrive (`PrepassRows`).
- **Protocol.** `multiprocessing.connection` on a loopback socket (named pipes refused connections
  now and then on Windows) with a random authkey passed in the environment. Messages are
  dataclasses (`ClipRequest`, `PrepassRows(start, rows)`, `PrepassEnd(error)`), pickled under the
  module's real name (the child's `__main__` calls the imported module's `_serve`).
- **Lifecycle.** `run` never blocks (a reader thread waits for the connection). A child that dies
  before it connects, fails to build its detector (it answers every clip with the traceback) or
  fails a clip ends the run with its reason. `stop` tells an idle child to exit and terminates a
  busy one; `close` stops and waits (kill after 10 s), and is idempotent. The pipeline calls
  `finish_prepass` after pass 1 (the child's CUDA teardown, 0.7 s, then overlaps the selector) and
  `close` whatever ends the run, init failures included. The child ignores Ctrl-C and exits when its
  parent ends however it ends (`OpenProcess`/`WaitForSingleObject` on Windows, `getppid` elsewhere).
  Interrupting real runs at 1, 4 and 7 s left no child behind.
- **CUDA graphs** in pass 1 are disabled only in `thread` mode.

## 3. Pass 1 is slower with stream_s4: why

Interleaved, pass 1 with stream_s4 is 0.66 s slower on pomona and 0.37 s on VolyAvw (`--no-video`).
Two causes:

- **Full-frame motion frames.** Pass 1 falls back to full-frame motion where it has no ROI, which
  follows the detections: on pomona stream_s4 gives 127 full-frame frames against heatmap_s4's 78,
  and the `cc` and `store` timers grow by 0.38 s and 0.56 s. On VolyAvw the counts are equal (53 and
  54) and so are those timers. This is the model's output, so it cannot change without changing
  outputs; cheaper full-frame CC and packing (AB+'s open item) would shrink it.
- **The child beside pass 1.** The remaining 0.3-0.4 s (VolyAvw, where the motion work is equal)
  is the child's prepass sharing the GPU and CPU with the first seconds of pass 1.

The earlier "1.4 s" (10.68 against 9.23) came from two different suites; measured interleaved in
one it is the above. Init is now floored by the player detector's import (boxmot pulls in
torchvision's `roi_align` and `torch._dynamo`, about 2.9 s), not the prepass; deferring ByteTrack so
pass 1 can start before that import ends was considered and not built.

## 4. Phase C hook

Phase C's gate (`roi_gate.py` on `phase-C`, states SEARCH/LOCK/CONTACT/COAST/IDLE, one 256x192
crop in model space) plugs in here without touching the parent/child plumbing:

- **Where.** In the child, in `GridTrackNetBallDetector._prepass` (`engines/ball_detectors.py`),
  where a unit is cut (`unit = streams[phase].push(...)`) and before it is batched
  (`batches.append`). The gate needs only the prepass's own finalised rows, so it lives wholly in
  the child; pass 1 does not change.
- **Decoder on a region.** `TransitionRing.decode` is `halves.decode(*gather(units))`;
  `TransitionRing.gather` returns the per-stride decoder inputs, which a region decoder slices at
  the crop (aligned to 16 px for the stride-16 maps) and decodes with an engine per crop shape,
  cached under its own `ball_trt` key.
- **The known issue.** The encoder caches whole-frame features, one per transition, reused by the
  units that share it. A crop that moves every frame breaks that cache. The fix options: hold the
  crop fixed per unit (so the cached transitions of a unit share one crop), or re-encode when the
  crop moves.
- **Protocol.** `PrepassRows` carries `start` and `rows`; a per-frame gate state or window goes
  beside `rows` as a new field, and nothing else in the plumbing changes.
- **Lag.** The gate observes rows after readback and batching (4 units a call); it must plan with
  that lag or flush before planning, as the Phase C record describes.

## Files

- New: `system/tennis_tracker/ball_weights.py`, `system/tests/test_ball_weights.py`,
  `system/tests/test_ball_prepass_process.py`, this record.
- Changed: `system/tennis_tracker/` `paths.py`, `config.py`, `pipeline.py`, `pass1.py`,
  `headless.py`, `engines/ball_detectors.py`, `engines/ball_prepass_process.py`,
  `engines/ball_gpu.py`; `system/clean_tracker.py`, `system/mypy.ini`, `system/AGENTS.md`,
  `system/README.md`, `system/models/README.md`; `finetune/` `scoring/archive.py`,
  `data/workspace.py`, `data/detect.py`, `training/pretrack.py`, `colab_zip.py`; tests in
  `system/tests/` and `tests/`.

## Tests

System suite (`tennis-opt`): 640 passed. Root suite (`tennis-analysis`, which runs both): 1362 passed,
11 skipped. mypy clean
in both.

## Open issues

- pomona gains nothing end to end: its init saving goes to pass 1's extra full-frame motion frames.
  Cheaper full-frame CC and packing would turn it into a win.
- Init is floored by the boxmot import (2.9 s); deferring ByteTrack would let pass 1 start earlier.
- The training workspace still calls `models/ball_detector.pt` the tracker's model in places, and
  `pseudo_labels`' `PIPELINE_CONF` 0.5 predates the gated thresholds.
- `tools/link_worktree_data.py` does not link `models/candidates/`, so a new worktree lacks the
  default weights until they are copied (the error says so).
