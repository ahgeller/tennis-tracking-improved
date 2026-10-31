# Live tracker v1 (phase-live)

Branch `phase-live` from `phase-ABplus-heatmap` (f43218a). Model stream_s4
(`system/models/candidates/stream_s4_e60.pt`) at conf 0.42. The design is in `live_design.md`; this
record has the measurements. Runs: tennis-opt for the tracker, tennis-analysis for the evaluator and
the suites. Every GPU or timed run was under `tools/gpu_lock.py`.

## Entry points

```powershell
# a file, frames arriving at its frame rate (default), or as fast as it goes
python system/clean_tracker.py --live -i clip.mp4 --weights system/models/candidates/stream_s4_e60.pt --conf 0.42
python system/clean_tracker.py --live --pace fast -i clip.mp4 ... --record-evidence
# a camera
python system/clean_tracker.py --camera 0 --weights ... --conf 0.42
# the held-out gate in live mode (fast pacing, evidence recorded beside each JSON)
python evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1 `
    --gridtracknet-weights system/models/candidates/stream_s4_e60.pt --gridtracknet-conf 0.42 --live --live-delay 0.5
# replay that evidence at other lags on the CPU
python tools/live_lag_sweep.py --evidence output/gate_live_L0.5 --delays 0 0.1 0.2 0.33 0.5 1 2 3 clip --history 1 2 4 8
```

Options: `--live-delay` (L, seconds or `clip`), `--live-history` (H), `--live-every` (K),
`--live-drop backlog|never`, `--live-backlog` (seconds of waiting frames before dropping),
`--live-selector-inline`. Beside the JSON, `<stem>.live.json` holds the run's report.

## Correctness before accuracy

| check | result |
|---|---|
| offline pipeline on this branch vs AB+heatmap (`output/gate_review_s4`), held-out gate | 0/13137 frames differ (present, x/y within 1e-6 px, track id); gate identical: recall 0.953, FA 95 (79/16), .971/.913/.952 |
| live prepass (eager, frame by frame) vs batched prepass, both decoded by OpenCV, pomona | 0/2022 detections differ (max difference 0.0) |
| live prepass vs the offline prepass (decoded by NVDEC), pomona | 949/2022 detections moved, by at most 0.007 px and 0.001 confidence; masks, players and court identical |
| `FixedLagSelector(delay=None)` on the 24 clips' offline evidence vs the offline JSONs | 0/13137 frames differ |
| replay of the live gate's recorded evidence (L=0.5 s, H=4 s, K=0.1 s) vs that live gate's JSONs | 0/13137 frames differ |
| selector in its process vs in line, pomona | 0/2022 frames differ |

So the selector's inputs live are the offline ones up to the decoder, and the replay tool predicts
a live run's output exactly: a sweep over replays is a sweep over live runs. The 1e-6 px tolerance
absorbs a 1e-13 px BLAS difference between the two conda envs (`opt_picker_rules.md` met the same).

## Choosing L: the sweep

The live gate run's recorded evidence (`output/gate_live_L0.5`), replayed by
`tools/live_lag_sweep.py` at every L and H with K = 0.1 s (`output/sweep_L_H.txt`). Offline
(L = clip): recall 0.953, FA 95 (79 bridged / 16 false), custom .971, grid .913, prof .952.
"differ" counts the frames whose output is present in one and not the other, or more than 0.5 px
apart, against the whole-clip answer.

| L | H | recall | FA (bridged/false) | custom | grid | prof | differ |
|---|---|---|---|---|---|---|---|
| 0 | 1 s | 0.880 | 99 (74/25) | .923 | .833 | .855 | 5758 |
| 0 | 4 s | 0.877 | 99 (74/25) | .922 | .820 | .855 | 5765 |
| 0.1 s | 1 s | 0.919 | 97 (78/19) | .951 | .878 | .904 | 3130 |
| 0.1 s | 4 s | 0.916 | 97 (78/19) | .950 | .864 | .904 | 3138 |
| 0.2 s | 1 s | 0.946 | 96 (79/17) | .963 | .901 | .948 | 1541 |
| 0.2 s | 4 s | 0.943 | 96 (79/17) | .962 | .890 | .948 | 1556 |
| 0.33 s | 1 s | 0.951 | 95 (79/16) | .966 | .911 | .953 | 283 |
| 0.33 s | 4 s | 0.949 | 95 (79/16) | .966 | .903 | .953 | 295 |
| 0.5 s | 1 s | 0.951 | 95 (79/16) | .967 | .911 | .952 | 69 |
| 0.5 s | 2 s | 0.951 | 95 (79/16) | .967 | .911 | .952 | 70 |
| 0.5 s | 4 s | 0.950 | 95 (79/16) | .967 | .908 | .952 | 76 |
| 0.5 s | 8 s | 0.949 | 95 (79/16) | .967 | .904 | .952 | 82 |
| 1 s | 1, 2, 4, 8 s | 0.953 | 95 (79/16) | .971 | .913 | .952 | 0 |
| 2 s, 3 s | 1, 2, 4, 8 s | 0.953 | 95 (79/16) | .971 | .913 | .952 | 0 |
| clip | - | 0.953 | 95 (79/16) | .971 | .913 | .952 | 0 |

The wrong-output rate is 0.001 from L = 0.2 s up (0.003 at L = 0) and the median error 1.0 px
everywhere.

- **Gate.** Within the gate means: pooled recall within 0.005 of offline, FA no higher than
  offline's 95, and no source more than 0.01 below its offline recall.
- **Smallest L within the gate: 0.33 s** (H = 1 s): recall 0.951, FA 95 (79/16), sources
  .966/.911/.953.
- **Smallest L that matches offline exactly: 1 s**, for every H from 1 s to 8 s: 0 of 13137 frames
  differ. At H = 0.5 s it is not exact (70 frames): one second of committed history is needed too.
- **H: shorter is no worse.** At every L, H = 1 s scores as well as or better than 2, 4 and 8 s,
  and it costs least (the window is H + L + K frames, and a selector run grows faster than
  linearly with its window). A long history gives the window's normalisations more frames
  against which to dilute a new track's short evidence: the track span over the window's length,
  and the timeline's coverage and gap terms. That delays the switch to the new track.

**Why the curve bends where it does.** Every selector decision that needs the future needs a
quarter to a third of a second of it, and none needs more than a second:

- *Selecting a new track* needs `select_min_obs_sec` = 0.25 s of its observations
  (`select_reacquire_obs_sec`, 0.16 s, to re-acquire one). Below that lag, the first frames of
  every rally are committed before the track that holds them can be selected. That is the large
  recall loss at L = 0 to 0.2 s. It is worst on grid, which has many short, broken tracks.
- *Gap fills* (bridged interpolation and motion recovery) need the detection after the gap. A
  frame in a gap can be filled only if the gap's far end arrives within L. At L = 0 nothing ahead
  is bridged (74 bridged false alarms instead of 79, and the motion fills are lost). The
  selector's small-gap window, `timeline_small_gap_sec` = 0.26 s, is the scale of the gaps that
  matter.
- *Choosing between two tracks* (a handover, a ball on the neighbouring court, a racket) depends
  on the tracks' scores, which grow with their length. The 69 to 283 frames that still differ at
  L = 0.33 to 0.5 s are such choices, made on less evidence than offline has.
- A track dies after 0.67 s without a detection (`max_gap_frames`), and a finished track can be
  stitched back within 1 s (`trail_stitch_gap_frames`). Past one second of lag, every decision the
  selector makes about a frame already has all the evidence the whole clip would give it, which
  is why L = 1 s is exact.

## Choosing K

At the candidate lags, K (the time between selector runs) and H = 0.5 s against 1 s
(`output/sweep_K.txt`):

| L | H | K | recall | FA | custom | grid | prof | differ | runs |
|---|---|---|---|---|---|---|---|---|---|
| 0.33 s | 1 s | 0.1 s | 0.951 | 95 | .966 | .911 | .953 | 283 | 2968 |
| 0.33 s | 1 s | 0.2 s | 0.951 | 95 | .967 | .911 | .952 | 168 | 1490 |
| 0.5 s | 0.5 s | 0.2 s | 0.951 | 95 | .967 | .913 | .952 | 100 | 1469 |
| 0.5 s | 1 s | 0.1 s | 0.951 | 95 | .967 | .911 | .952 | 69 | 2924 |
| **0.5 s** | **1 s** | **0.2 s** | **0.951** | **95 (79/16)** | **.967** | **.913** | **.952** | **51** | **1469** |
| 0.5 s | 1 s | 0.33 s | 0.952 | 95 | .969 | .913 | .952 | 28 | 885 |
| 1 s | 1 s | 0.1, 0.2, 0.33 s | 0.953 | 95 | .971 | .913 | .952 | 0 | 2804 / 1407 / 850 |

A frame waits L to L + K after pass 1 commits it, so a larger K also adds lag: on average K/2.
Doubling K from 0.1 s to 0.2 s halves the number of runs, and each run costs only about 10% more
(its window grows by K). So the selector's CPU time halves, and accuracy improves slightly,
because the average lag grows.

**Choice: L = 0.5 s, H = 1 s, K = 0.2 s** (the `LiveOptions` defaults).

- It is within the gate with margin: recall 0.951 (0.002 below offline), FA 95 (79/16) equal to
  offline, sources .967/.913/.952, and 51 of 13137 frames differ.
- Each frame is committed 0.5 to 0.7 s after pass 1 has it. That is 0.5 s less than the exact
  setting.
- It costs half the selector time of K = 0.1 s.
- `--live-delay 1` gives the offline answer exactly where a second of delay can be afforded.

## The held-out gate in live mode

These are real live runs, not replays: `evaluate_archive.py --mode pipeline --live` tracks each
clip frame by frame with fast pacing (`output/gate_live_final`, `output/gate_live_L0.5`).

| run | recall | wrong | FA (bridged/false) | median | custom | grid | prof |
|---|---|---|---|---|---|---|---|
| offline stream_s4 @0.42 (this branch = AB+heatmap) | 0.953 | 0.001 | 95 (79/16) | 1.0 px | .971 | .913 | .952 |
| **live, L 0.5 s, H 1 s, K 0.2 s (the defaults)** | **0.951** | **0.001** | **95 (79/16)** | **1.0 px** | **.967** | **.913** | **.952** |
| live, L 0.5 s, H 4 s, K 0.1 s (the first run) | 0.950 | 0.001 | 95 (79/16) | 1.0 px | .967 | .908 | .952 |

Both live runs score exactly what the replay predicted for their settings.

Off our court (`tools/off_court.py`; present frames in brackets). The offline column is
stream_s4 AB+ from `opt_picker_rules.md`:

| clip | offline AB+ | live (defaults) |
|---|---|---|
| VolyAvw | 40 (2509) | 0 (2457) |
| pomona | 0 (1090) | 0 (1090) |
| PomonaPitzer | 0 (3142) | 0 (3129) |
| Vereinsduell | 22 (32366) | 1 (32526) |

- **VolyAvw.** The 40 neighbouring-court frames offline are gone live. Window 2772-2865 has no
  output; window 1107-1137 has 30 frames, all on our court, as offline.
- **Vereinsduell.** The one live frame off our court is a single motion row (frame 24581, 5.7 m
  out).
- **How the runs were made.** VolyAvw, pomona and PomonaPitzer are the realtime runs below.
  Vereinsduell (27 minutes) used fast pacing; nothing was dropped.

## Latency and the per-frame budget

Realtime pacing (each frame is due at i/fps after the first, as from a camera), the defaults, on
the desktop RTX GPU (`system/output/live_realtime`). Latency is measured from a frame's arrival to
the moment its final row is out. "Evidence" is measured from arrival to the moment pass 1 has the
frame; the selector adds L to L + K on top. The 30 FPS clips are the 60 FPS ones with every
second frame (`ffmpeg -r 30`).

| clip | fps | frames | dropped | latency median / p99 / max | evidence median / p99 | selector ms a run (mean / p99 / max) | most frames waiting |
|---|---|---|---|---|---|---|---|
| pomona | 60 | 2022 | 0 | 781 / 1176 / 1478 ms | 151 / 524 ms | 29.7 / 60 / 66 | 28 |
| pomona | 30 | 1012 | 0 | 755 / 856 / 1357 ms | 152 / 175 ms | 14.4 / 34 / 36 | 2 |
| VolyAvw | 60 | 3286 | 0 | 787 / 901 / 1442 ms | 151 / 164 ms | 38.5 / 93 / 131 | 5 |
| VolyAvw | 30 | 1644 | 0 | 762 / 860 / 1352 ms | 152 / 163 ms | 19.4 / 44 / 47 | 2 |
| PomonaPitzer | 60 | 3674 | 0 | 785 / 931 / 1504 ms | 149 / 337 ms | 39.6 / 58 / 148 | 17 |

**Realtime factor.** Under realtime pacing the factor is 0.98 to 0.99 on every clip. The
tracker keeps up with the video, and the 1 to 2% is the start-up and the final flush. As fast
as it can go (fast pacing, `system/output/live_fast`):

| clip | 60 FPS | 30 FPS |
|---|---|---|
| pomona | 1.92 | 3.32 |
| VolyAvw | 1.98 | 3.80 |
| PomonaPitzer | 1.97 | - |
| Vereinsduell (27 min, 30 FPS) | - | 3.72 |

**Where the latency goes.**

| part | time |
|---|---|
| prepass lookahead | 4 phase frames (133 ms) |
| pass 1 | about 15 ms |
| L | 500 ms |
| waiting for the next selector run | up to K = 200 ms, 100 ms on average |
| the run itself | 15 to 40 ms |

The median of about 780 ms is this sum.

**Spikes (p99 and max).**

- **The max (1.35 to 1.5 s on every clip) is frames 0 and 1.** They arrive before pass 1 is built
  (about 0.25 s of frame window and CUDA graphs) and before the first selector run.
- **The 60 FPS p99 on pomona and PomonaPitzer is pass 1's heaviest motion.** For about 30 frames
  (pomona 588-618 and 1054-1084, PomonaPitzer around 2900) the tracking thread costs 26 to 30 ms a
  frame against 16.7 ms. A cProfile of the stretch: `connectedComponentsWithStats` in
  `filter_boost_mask` takes 8 ms a call, against under 1 ms elsewhere (the boost mask's bounding
  box spans most of the frame). The backlog peaks at 28 frames (0.47 s) there and drains right
  after. VolyAvw has no such stretch: p99 901 ms.
- **The selector is never the bottleneck.** Its worst run (148 ms) is under its 200 ms period,
  and it runs in its own process, so it never stalls pass 1.

**Start-up and drops.** The first realtime runs dropped 5 to 22 frames per clip. They had three
causes, all fixed in 0a3fcac:

| cause | effect | fix |
|---|---|---|
| a realtime file's clock started before pass 1 was built | frames 2-15 dropped at startup | the source holds a file's frames after the first two until pass 1 is built (`LiveSource(hold_after=2)`, `release()`) |
| the Python garbage collector's full collections walked every model and engine object | 80 to 190 ms stalls of the tracking thread | freeze the start-up objects (`gc.collect(); gc.freeze()`) before the stream |
| the 0.25 s backlog budget | the heavy-motion stretches (0.47 s at peak) dropped pairs of frames | the budget is now 0.5 s |

After the fixes nothing is dropped on any clip. A camera is not held (it keeps capturing), so its
first frames can still be dropped, and they are reported.

**Per-frame budget at 60 FPS (16.7 ms):**

- Tracking thread: a median of 7 to 8.5 ms a frame (prepass push and pass 1). It goes over budget
  only in the heavy-motion stretches above, which the backlog absorbs.
- Selector process: 30 to 40 ms every 200 ms on average (2 to 3 ms a frame).

**At 30 FPS (33.3 ms)** both have more than 3x headroom.

**Accuracy of the realtime runs.** No frame was dropped, so the realtime JSONs are what fast
pacing gives. Off our court: 0 frames on VolyAvw, pomona and PomonaPitzer (see the gate section).

## Two outputs (provisional plus final)

A two-output design would send a provisional point per frame at a small lag (L_p = 0.2 s:
recall 0.946, FA 96), then the final point at L_f = 1 s (exact). Assessment:

- **When it helps.** It is worth having only for a display that must react within about 0.3 s, and
  only if the consumer can revise what it has already shown.
- **How often it corrects.** At L_p = 0.2 s, 1541 of 13137 frames (12%) would change between
  provisional and final; at 0.33 s, 283 (2%). At 0.2 s that is a visible correction about once a
  rally.
- **Cost.** A second selector run every K frames, which doubles the selector time, already the
  largest CPU item. It also adds a second stream of rows to the JSON and the console, and needs a
  consumer contract for corrections. None of the existing consumers can use a row that may be
  revised: not the scorers, the court events or the review tool.
- **Verdict.** Not built in v1. One output at L = 0.5 s is within the gate and 0.5 s behind exact.
  A follow-up can add the provisional stream as a second `FixedLagSelector` fed by the same
  `EvidenceFeed`, with no other code, if a consumer asks for it.

## Phase C hook

The extension point is `run_live(cfg, options, region_for=gate)` (`system/tennis_tracker/live/pipeline.py`).
`_Pump._push_ball` calls the gate there: it is the one place a frame reaches the ball model, on
the tracking thread, as the frame is taken from the source and before anything else runs on it.

- **Type.** `RegionGate`, a callable `(ArrivedFrame) -> Optional[Region]`. `Region` is
  `(x1, y1, x2, y2)` in source pixels; None means the whole frame.
- **What is at hand at the call.** Only the `ArrivedFrame`: its source index, its BGR picture (in
  a page-locked buffer when the capture allows) and its arrival time. Anything else the gate
  decides from it must hold itself. The live run has two candidates, neither passed in v1:
  - `_LiveRun.committed`: the committed rows, L + K behind the newest frame.
  - pass 1's ROI tracker: its newest detections, the prepass's lookahead (4 phase frames)
    behind.
- **Out.** Today `PrepassStream.push` takes a whole frame, so a gate that returns a region is
  refused with NotImplementedError. A region-aware `push` is the missing half.
- **Constraint.** stream_s4 encodes each frame transition once (`TransitionRing`), and a unit
  decodes four cached transitions. A unit's region must therefore be the same for all four of its
  transitions, or its cached features stop matching. Either the ring keys its features by region,
  or the gate holds a region for at least a unit (four phase frames).
- **Not built.** The gate and the region-aware push. `tests/test_live_pipeline.py` checks that a
  gate sees every frame before the model does and that a region is refused.

## What is not live yet

- **Court events** (bounces, contacts, rally legs) are computed once at stream end over the
  committed rows. A fixed-lag version needs two detector steps of committed trajectory on each
  side of a bounce, plus the merge window: about 0.2 s more than L.
- **Provisional output**: see the two-output assessment.
- **Tracking video**: live writes no video. A renderer would have to draw each frame L late, from
  the committed rows.
- **Decoder**: live decodes once with OpenCV; offline decodes the prepass's frames with NVDEC.
  Their pixels differ slightly (see the correctness table). NVDEC frames would need a host copy
  for pass 1.
- **The prepass in its own process** (`gridtracknet_prepass_process`) is not used live: the stream
  runs on the tracking thread, and `open_stream` refuses a remote prepass.
- **The selector's window normalisation**: on a window, the selector's clip-length terms are
  window-length. This is measured above, not removed.
- **On-device**: iOS, Core ML and Swift are out of v1. Each CUDA/TensorRT stage would need a port:
  the prepass engines, the motion kernels and pass 1's CUDA graphs.

## Follow-ups

- Fixed-lag court events (above).
- A provisional stream, if a consumer needs one.
- An incremental selector that keeps its tracks and the timeline DP between runs. A run's cost
  grows faster than linearly with its window, so this would cut the selector's time by about the
  ratio of the window to K. It is a selector change, measured on `train`.
- Phase C region gate (above).

## Tests and checks

New tests, 46 in all:

| file | tests | what they check |
|---|---|---|
| `system/tests/test_live_prepass.py` | 17 | the push-by-push prepass equals the batched one; `_prepass`'s contract; readiness |
| `system/tests/test_live_fixed_lag.py` | 12 | the causal selector: whole-clip equals offline, commits never change, stable track ids, out-of-order runs refused, the replay equals the live feed |
| `system/tests/test_live_source.py` | 14 | the drop policy (BACKLOG drops whole phase groups, NEVER and fast pacing block, every drop reported); realtime pacing; the start-up hold |
| `system/tests/test_live_selector_process.py` | 2 | the selector process equals the selector in line, and it reports a failure |
| `system/tests/test_clean_tracker_cli.py` | 1 | `--live` and `--camera` reach the live tracker |

Suites, all under `tools/gpu_lock.py` at 0a3fcac:

- `system` (tennis-opt): 655 passed.
- Root (tennis-analysis): 1377 passed, 11 skipped. `test_ball_trt` now skips when `onnx` is not
  installed, as it is not in tennis-analysis; it used to fail there.
- mypy is clean: `system/mypy.ini` (23 files, the live modules added) and the root config (5 files).

Realtime and fast-paced live outputs are identical on the five latency clips (0/11638 frames
differ), so pacing changes timing only.
