# Speed round: player detection (2026-10-31)

Branch `speed-players` off `phase-ABplus-heatmap` at f43218a. Three ideas for the player
detector's cost, each judged alone and then together:

1. an output-identical player schedule;
2. player detection every other frame on 60 FPS clips only;
3. boxmot's import deferred or threaded so init or pass 1 starts sooner.

**Result: none was kept by this round's rule; idea 2 ships by a later decision.** Idea 1 has no
exact schedule (evidence below). Idea 2 saves 1.4-1.7 s on Pomona and 3.7 s on VolyAvw, and passes
the ball gate exactly. But on stream_s4 it adds one unmatched confirmed landing (labelled precision
0.272 to 0.269), so it fails the strict rule's "landing scores unchanged". Idea 3 keeps every
output identical but saves nothing measurable. The branch `speed-players` carries this record
only. The two implementations, with their tests, were kept as patches against f43218a under
`output/patches/` of the `speed_players` worktree (gitignored, not committed).

**Later decision (2026-10-31): idea 2 ships.** The owner had explicitly allowed this change, and
it was decided to keep it, knowing its cost (section 2): the ball gate is identical for both
models (all clips, every source, the 60 FPS subset), video6, the only held-out clip with landing
labels, is unchanged, and the saving is 1.4-1.7 s on Pomona and 3.7 s on VolyAvw. The cost, on
stream_s4 only: one extra confirmed landing on the training clip grid_match36 (frame 1142, from a
0.65 px move of a padded player box's edge) and two extra placed landings on PomonaPitzer, no hit
lost or gained. The shipped heat-map model's landing scores are unchanged. The patch was applied
on `phase-ABplus-heatmap` as its own commit, unchanged except for documentation and a command-line
switch: `Config.player_every_other_frame_fps = 55.0` (0 turns it off) and
`clean_tracker.py --players-every-frame`. Idea 3 stays out.

## How it was measured

- **Models**: the shipped `heatmap_s4` (`models/ball_detector.pt`, conf 0.40) and the candidate
  `stream_s4` (`models/candidates/stream_s4_e60.pt`, conf 0.42).
- **One TensorRT engine for stream_s4.** Every worktree builds the stream_s4 engine into its own
  `system/models/engine_cache`, and two builds of the same weights give different bytes and
  different detections. Every worktree here (base, idea 2, idea 3) used ABplus's build: encoder
  `...-encoder-b4-d3a601afe5634cc0.engine` md5 f5bfa32d, decoder `...-decoder-b4-96434428d24a6555.engine`
  md5 4ef6ca09. The base was a clean worktree of f43218a. An early base stream_s4 run on a
  worktree's own build did not match, and was discarded and re-run.
- **Gate**: `evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1 --court`,
  for both models, scored by source and on the 60 FPS subset. Landings: `score_court_events.py`
  on the 7 clips with landing labels (video6 is the only held-out one; grid_match4/36/38/47/55
  and PomonaPitzer are not), tiers confirmed, placed and all. Off-court: `tools/off_court.py` on
  pomona, VolyAvw and PomonaPitzer (`clean_tracker.py --court`).
- **Identity**: the tracking JSON's `frames` (ball rows, player boxes, court keypoints), `tracks`,
  `chosen_track`, `court_events` and `players`, compared field by field.
- **Timing**: `clean_tracker.py --info` from `system/`, the variants interleaved in one lock (a
  warm-up round, then 3 rounds, the order reversed each round), `[done]` total and the `[info]`
  rows as median [min-max] seconds. Before and after each run the other python processes' CPU
  time and the GPU's compute apps were sampled; a run in which another python process used more
  than 1 s of CPU or held the GPU would have been discarded (none was). Every GPU or timed run went
  through `tools/gpu_lock.py`.
- **Held-out frame rates**: 10 of the 24 held-out clips are about 60 FPS: grid_match8 (60.00),
  grid_match16 (59.98), video1-7 (60.00-60.17), video9 (60.10). The other 14 are 30 FPS:
  grid_match22, video8, video10, video17-23, video43-46. Samples: pomona 59.67, VolyAvw 59.54,
  PomonaPitzer 59.97. Of the landing clips outside the held-out set, grid_match36/38 are 59.94
  and grid_match4/47/55 are 30.

Baselines reproduced on f43218a:

| model | recall | FA (bridged/false) | custom / grid / prof | median px | 60 FPS subset: recall, FA, median |
|---|---|---|---|---|---|
| shipped @0.40 | 0.952 | 89 (79/10) | .966 / .896 / .962 | 1.0 | 0.956, 41 (36/5), 0.7 px |
| stream_s4 @0.42 | 0.953 | 95 (79/16) | .971 / .913 / .952 | 1.0 | 0.956, 41 (36/5), 0.7 px |

## 1. Output-identical player schedule: none exists

Every frame's player boxes are written to the tracking JSON (`frames[i].player_boxes`), so a
schedule is output-identical only if each skipped frame's boxes come out the same, and so do the
boxes of every frame after it.

- ByteTrack's (boxmot 16.0.10) output boxes are its Kalman means. Every update moves every active
  track's state, and each later frame's boxes depend on all earlier updates. Skipping a frame's
  update therefore changes that frame's boxes unless the tracker holds no track and the engine
  would find no player, and knowing the engine finds nothing means running it.
- **Replay** (`replay_skips.py`, scratch): the exact detections ByteTrack received on every frame
  were recorded from a full run (pomona 2022 frames, VolyAvw 3286). A replay reproduces the
  recorded boxes exactly. Then, for each frame t, the update at t alone was skipped (t keeps
  t-1's boxes, as pass 1 does for a frame without an update) and the tracker was replayed forward
  from its saved state, for up to 600 frames, looking for the first box that differs:

  | clip | frames tested | skip changes no box | frames with tracks and detections | first changed box |
  |---|---|---|---|---|
  | pomona | 2021 | 0 | 2021 | on the skipped frame itself, every time |
  | VolyAvw | 3285 | 0 | 3285 | on the skipped frame itself, every time |

  Every frame has both live tracks and detections, so the "empty tracker, no detections" case
  never occurs in these clips.
- **Exact reuse** of an engine result needs a frame bit-identical to the one before it. Decoded
  with OpenCV, none of the 10 held-out 60 FPS clips (8088 frames) or the 3 samples (8982 frames)
  has a single frame identical to the previous one.

So no schedule skips player detection without changing an output, and no result can be reused
exactly. Nothing was built.

## 2. Every other frame at 60 FPS: rejected by the round's rule, shipped by a later decision

The implementation (`output/patches/idea2_every_other_frame_60fps.patch`):

- `Config.player_every_other_frame_fps = 55.0`;
- `player_detect_interval(cfg, fps)` raises the interval to at least 2 when the clip's FPS is at or
  above that threshold, so 30 and 50 FPS clips are unchanged;
- the pipeline passes `video.fps` to `PlayerDetector`;
- the existing interval path (engine and ByteTrack on every other frame, `track_buffer` 30 updates
  so a lost track still lasts 60 frames, `interpolate_player_boxes` for the frames between)
  does the rest;
- tests for the rule at 30/50/59.5/60 FPS and for interpolation through the pipeline.

**Ball gate: identical numbers.** Every row is equal to three decimals for both models: all clips,
each source, the 60 FPS subset and the 30 FPS subset.

| model | base | idea 2 |
|---|---|---|
| shipped | 0.952, 89 (79/10), .966/.896/.962, 1.0 px | 0.952, 89 (79/10), .966/.896/.962, 1.0 px |
| shipped, 60 FPS subset | 0.956, 41 (36/5), 0.7 px | 0.956, 41 (36/5), 0.7 px |
| stream_s4 | 0.953, 95 (79/16), .971/.913/.952, 1.0 px | 0.953, 95 (79/16), .971/.913/.952, 1.0 px |
| stream_s4, 60 FPS subset | 0.956, 41 (36/5), 0.7 px | 0.956, 41 (36/5), 0.7 px |

**Outputs.**

- **30 FPS clips**: the 14 are identical once player ids are renumbered by first appearance:
  boxes, tracks and court events (both models). The raw ids differ only because ByteTrack numbers
  tracks from a process-global counter (`BaseTrack._count`) and the evaluator runs every clip in
  one process, so the 60 FPS clips before them create a different number of tracks.
- **60 FPS clips**: the ball rows are identical on all 10 held-out clips for both models. The
  player boxes are new values on every frame.
- **Samples and landing clips**: a few ball rows change because the selector and court events read
  player boxes:
  - VolyAvw stream_s4: 29 rows; 4 fewer filled frames (2457 to 2453);
  - grid_match38 shipped: 11 rows;
  - grid_match36 stream_s4: 11 rows.

  Contacts move by 1-6 px (VolyAvw shipped also loses one of 15), and a few rally legs start
  1-2 frames later.

**Landings** (`score_court_events.py`, all 7 labelled clips; the default review policy gives the
same totals):

| model, tier | base: found / hits / precision / recall / px | idea 2 |
|---|---|---|
| shipped, confirmed | 90 / 26 / .289 / .419 / 1.1 | 90 / 26 / .289 / .419 / 1.1 |
| shipped, placed | 115 / 30 / .261 / .484 / 1.1 | 115 / 30 / .261 / .484 / 1.1 |
| shipped, all | 114 / 30 / .263 / .484 / 1.1 | 114 / 30 / .263 / .484 / 1.1 |
| stream_s4, confirmed | 92 / 25 / .272 / .403 / 1.5 | **93** / 25 / **.269** / .403 / 1.5 |
| stream_s4, placed | 117 / 29 / .248 / .468 / 1.5 | **119** / 29 / **.244** / .468 / 1.5 |
| stream_s4, all | 115 / 29 / .252 / .468 / 1.5 | **117** / 29 / **.248** / .468 / 1.5 |

- video6, the one held-out clip with landing labels, is unchanged in every tier for both models.
- The extra confirmed landing is a model bounce at grid_match36 frame 1142 (59.94 FPS), with the
  same ball measurement in both runs (y 646.66). `bounce_events._off_player_events` drops a
  landing on a player box padded by a quarter of its height (`court_scene.inside_player`). The
  base's player box has its padded top at y 646.28, so the base drops the landing. Under idea 2,
  1142 is a tracked frame, but the Kalman state behind it has seen half the updates, and the
  padded top is at 646.93, 0.65 px lower, so idea 2 keeps it.
- The two extra placed landings are on PomonaPitzer (18 to 20 found, the same 17 hits).

**Off-court**: `off_court.py` reports 0 off-court frames on every sample for both models, before
and after. Its "present" count changes on VolyAvw stream_s4 (2457 to 2453), from the 4 frames above.

**Timing** (interleaved; the 60 FPS samples are where it acts):

| run | base, shipped | idea 2, shipped | base, stream_s4 | idea 2, stream_s4 |
|---|---|---|---|---|
| pomona `--no-video`, total | 14.70 [14.70-15.20] | **13.10** [12.80-13.40] | 16.00 [15.80-26.80] | **14.30** [14.30-14.30] |
| - pass 1 | 7.83 | 6.19 | 8.62 | 6.92 |
| VolyAvw `--no-video`, total | 23.00 [22.70-23.00] | **19.20** [19.20-19.40] | 22.80 [22.20-22.80] | **19.10** [18.90-19.10] |
| - pass 1 | 13.72 | 10.14 | 13.73 | 10.15 |
| pomona with video, total | 19.90 [19.70-20.10] | **18.30** [17.20-18.70] | 20.80 [20.70-21.30] | **19.40** [19.30-19.60] |
| - pass 1 / pass 2 | 8.64 / 4.36 | 6.91 / 4.38 | 9.18 / 4.33 | 7.62 / 4.39 |

The 26.80 s base stream_s4 round on Pomona is a slow-GPU-state outlier: no neighbour was flagged,
and the other two rounds were 15.80 and 16.00. Idea 2 saves 1.4-1.7 s on Pomona (7-11%) and
3.6-3.7 s on VolyAvw (16%), all of it in pass 1 (`pass1 aux wait(d2h)`, which holds the wait for
the player engine, falls by 1.1 s on Pomona and 2.7 s on VolyAvw). 30 FPS clips run unchanged code.

**Decision: rejected.** The ball gate, video6 and the off-court counts hold. On stream_s4, though,
the labelled landing precision falls in every tier (one extra false confirmed landing, two extra
placed ones), with no hit gained, and the rule asks for landing scores unchanged. The saving is
real (11% of a Pomona `--no-video` run, 16% of VolyAvw's). The prior record (`opt_pass1.md`,
"Players") found the same kind of cost at N=3 (a labelled landing lost). If the landing rule
were relaxed to "hits and recall not lower, median px not worse", this patch would pass as is.

**Later decision: shipped** (see the top of this record). The cost above is accepted as measured:
stream_s4 confirmed precision .272 to .269 and placed .248 to .244 on the labelled clips (none of
it on the held-out video6), the shipped model unchanged.

## 3. Deferred or threaded boxmot import: rejected (no saving)

On f43218a `prefetch_bytetrack()` already starts the import on a thread at the top of the run, so
it overlaps the CUDA context, the ball model and the engine builds. `PlayerDetector.__init__` joins
it. What is left to save is the part of the import still running when the detector is built.

- **The import's cost**, warm (torch, numpy and cv2 already imported): 3.11-3.33 s in five fresh
  processes and 4.98 s in a sixth, the first after a while (`python -X importtime`).
  `boxmot.reid.core` is 2.5 s of it (2.45-2.59, 4.29 in the slow run): the CLIP backbone module,
  mostly building its tokenizer, takes 1.9-2.1 s, torchvision 1.2 s (torch._dynamo
  1.0 s of that), `boxmot.trackers` 0.6 s; the ByteTrack module itself is under 0.01 s.
  It is Python bytecode holding the GIL, so a thread does not make it free: the foreground
  prepass takes 2.5 s with the import done and 4.1-4.4 s with it running alongside.
- **The change built** (`output/patches/idea3_deferred_boxmot.patch`, with tests):
  - `PlayerDetector` no longer waits for the import. It opens the engine and queues each frame's
    decoded detections.
  - ByteTrack is created and fed the queue in frame order once the import has finished. ByteTrack
    uses its image argument only for its shape, so the queue holds no frames.
  - Pass 1 reads the boxes once at the end (`boxes_by_frame`).
  - The outputs are identical to the base on pomona for both models.
- Two placements of the import were timed:
  - **dstart**: the import starts at the top of the run, as now, and the detector stops waiting.
  - **dafter**: the import starts after the foreground prepass, so the prepass runs alone.

| clip, model | base | dstart | dafter |
|---|---|---|---|
| pomona, shipped, total | 15.50 [15.50-15.60] | 15.50 [15.40-16.40] | 17.30 [17.10-18.00] |
| - init / pass 1 | 6.36 / 8.53 | 6.27 / 8.51 | 4.94 / 11.71 |
| pomona, stream_s4, total | 16.50 [15.80-16.90] | 16.50 [16.40-17.30] | 17.00 [17.00-17.20] |
| - init / pass 1 | 6.73 / 9.06 | 6.78 / 9.00 | 3.93 / 12.34 |
| video19 (30 FPS, short), shipped | 5.40 [5.30-5.50] | 5.40 [5.30-5.40] | 5.90 [5.90-6.00] |
| - init / pass 1 | 4.90 / 0.42 | 2.90 / 2.49 | 2.54 / 3.28 |
| video45 (30 FPS, short), shipped | 5.10 [5.10-8.00] | 5.00 [4.90-5.10] | 5.50 [5.40-5.60] |
| - init / pass 1 | 4.92 / 0.20 | 2.74 / 2.22 | 2.43 / 3.07 |

The same suites as idea 2 also timed idea 3 (dstart, the one that keeps the import's placement):

| run | base, shipped | idea 3, shipped | base, stream_s4 | idea 3, stream_s4 |
|---|---|---|---|---|
| VolyAvw `--no-video`, total | 23.00 [22.70-23.00] | 23.40 [22.60-26.90] | 22.80 [22.20-22.80] | 23.00 [22.80-23.20] |
| pomona with video, total | 19.90 [19.70-20.10] | 19.90 [19.70-20.00] | 20.80 [20.70-21.30] | 21.30 [21.10-22.00] |

On both, `pass1 player boxes` is 0.00 s: the import had finished before the detector was built.

- On Pomona the import has already finished when the detector is built (dstart queued 1 frame), so
  there is nothing left to hide.
- On short clips init does get about 2 s shorter, but pass 1 gets 2 s longer: the import still
  holds the GIL while pass 1 runs, and the queued association catches up at the end
  (`pass1 player boxes` 1.0-1.7 s).
- Starting the import after the prepass (dafter) makes the prepass faster but pass 1 slower by
  more, 0.4-1.8 s worse in total.

**Decision: rejected.** No variant saves anything measurable, and the change adds a queue and a
second tracker-creation path.

## All together

Nothing was kept by the round's rule, so there is no combination to measure here. Idea 2 with the
`speed-motion` round's changes is measured where they were merged: the "Speed round merged"
section of each tracker branch's record.

## Suites

Branch HEAD (this record and the test fix below; the tracker code is f43218a's):

- `cd system; python -m pytest -q` in `tennis-opt`: 609 passed.
- Root `python -m pytest -q` in `tennis-analysis`: 1331 passed, 11 skipped.
- mypy clean at the root (5 files) and in `system/` (18 files).

The root suite had 1 failure on f43218a:
`test_ball_trt.py::test_a_small_streaming_networks_engines_stream_what_its_forward_pass_answers`
fails in `tennis-analysis`, which has no `onnx`. `opt_picker_rules.md` already noted it as
environmental. onnx is optional (without it the prepass runs PyTorch), so the test now skips
without it, as it already did without CUDA or TensorRT. It still runs and passes in `tennis-opt`.
That is its own commit.

The idea 2 and idea 3 patches each passed `test_player_detector.py`, `test_pipeline_run.py` and
`test_pass1_stages.py` (43 and 42 tests) before they were taken off the branch.
