# Picker rules: option A of the track-picker comparison (2026-10-30)

Branch `picker-rules` off `phase-ABplus` at 0cb0e6c. Option A fixes the selector's track picker
with rules; option B (branch `picker-learned`) replaces it with a learned CatBoost picker. Both are
judged on the same scorecard, written down below exactly as given. Python is the `tennis-analysis`
env for the evaluator and the suites, `tennis-opt` for the sample-clip tracker runs (as AB+ did).
Every GPU run went through `tools/gpu_lock.py`.

Two models throughout: the shipped `heatmap_s4` (`models/ball_detector.pt`, conf 0.40) and the
candidate `stream_s4` (`models/candidates/stream_s4_e60.pt`, conf 0.42).

## The scorecard (identical for option B)

Held-out pipeline gate, run once at the end, for both models:

    python evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1 --gridtracknet-conf 0.40
    python evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1 --gridtracknet-conf 0.42 \
        --gridtracknet-weights system/models/candidates/stream_s4_e60.pt

Reported: recall, wrong, false alarms (bridged/false), median px, and recall per source
(custom/grid/prof). Baselines (AB+ HEAD):

| model | recall | FA (bridged/false) | custom | grid | prof |
|---|---|---|---|---|---|
| shipped @0.40 | 0.946 | 98 (86/12) | .964 | .891 | .953 |
| stream_s4 @0.42 | 0.947 | 104 (86/18) | .969 | .906 | .942 |

Gate: recall within 0.01 of the baseline, no source down by more than 0.010, median px no worse
than +0.3; false alarms and wrong should go down.

The off-our-court metric (below) on the sample clips VolyAvw, pomona, PomonaPitzer and
Vereinsduell, both models, before and after, plus the VolyAvw windows 2772-2865 (the
neighbouring-court track stream_s4 picked) and 1107-1137.

No tuning on the held-out clips: every rule and threshold was chosen on the 62 non-held-out
labelled clips (`finetune` archive minus `mixed_v1`) and the sample videos; the held-out gate ran
once, after the commits.

## The off-our-court metric (`tools/off_court.py`)

Written first and sent to the main session before any selector change. The definition is the
module docstring; in short:

- Every present output frame is given its supporting selector track: a detector row its own
  track, a fitted or motion row the detector row before it in its run of consecutive present
  frames (else the one after). A run with no detector row is a group of its own.
- A track's output frames form one group across the whole clip. The group is *off our court*
  when every one of its points lies more than 3 m beyond our doubles sidelines, sideways on the
  ground.
- The lateral distance of `(x, y)` is read on the point's image row, or on the far baseline's
  row when the point is above it: the two doubles sidelines (through corner keypoints 0-2 and
  1-3) give the court's edges on that row, the width there is 10.97 m. Court: the frame's own
  keypoints, else `last_valid_court_keypoints`.

Why this shape: seen from behind a baseline, an airborne ball stands straight above its ground
point, so one image position cannot say which court a ball is on; but our rally comes down to our
court at a bounce or a racket, while a ball on a neighbouring court stays beside that court, whose
nearest sideline is at least 3.66 m from ours (the ITF's minimum side run-off). A first version
segmented by runs and flagged a real wide ball of ours (PomonaPitzer 871-897, labelled, 4-5 m out,
cut off from its rally by one missing frame); grouping by track fixed it.

The selector rule below uses the same geometry on the track's observations, so the metric's drop on
VolyAvw is by construction. The gate measures what the rule costs.

## Results

### Held-out gate (once, at 19fdaf7)

| model | | recall | wrong | FA (bridged/false) | median | custom | grid | prof |
|---|---|---|---|---|---|---|---|---|
| shipped @0.40 | AB+ | 0.946 | 0.006 | 98 (86/12) | 1.0 px | .964 | .891 | .953 |
| | rules | **0.952** | **0.001** | **89** (79/10) | 1.0 px | .966 | .896 | .962 |
| stream_s4 @0.42 | AB+ | 0.947 | 0.004 | 104 (86/18) | 1.0 px | .969 | .906 | .942 |
| | rules | **0.953** | **0.001** | **95** (79/16) | 1.0 px | .971 | .913 | .952 |

Both pass: recall up 0.006, every source up, median unchanged, wrong and false alarms down.
Out-dirs `output/gate_ship`, `output/gate_s4`.

Interp rows on the held-out clips (hit / wrong / on an invisible label):

| model | AB+ | rules |
|---|---|---|
| shipped | 20 / 25 / 23 | 11 / 3 / 14 |
| stream_s4 | 13 / 17 / 42 | 7 / 4 / 33 |

22 of shipped's 25 wrong interp rows (13 of stream_s4's 17) were detections the refit had moved off
and relabelled, every one of them within 10 px of the label before the move. The interp rows left
are almost all single-hole fills; the ones on invisible labels are mostly *bridged* (on the line
through an occlusion), see "Rejected" for why they stay.

### Non-held-out clips (the tuning set: 62 labelled clips, real GPU runs)

Baseline: option B's same-commit runs (`phases/picker_learned/output/train_{shipped,s4}`, AB+ code).
New: `output/new_tune_{ship,s4}` at HEAD.

| model | | recall | wrong | FA (bridged/false) | custom | grid | prof |
|---|---|---|---|---|---|---|---|
| shipped | AB+ | 0.963 | 0.003 | 396 (297/99) | .966 | .957 | .976 |
| | rules | 0.969 | 0.001 | 342 (261/81) | .969 | .963 | .984 |
| stream_s4 | AB+ | 0.949 | 0.003 | 372 (287/85) | .963 | .939 | .970 |
| | rules | 0.956 | 0.001 | 321 (248/73) | .967 | .946 | .977 |

### Off our court, sample clips (output frames off our court; present frames in brackets)

| clip | shipped AB+ | shipped rules | stream_s4 AB+ | stream_s4 rules |
|---|---|---|---|---|
| VolyAvw | 0 (2590) | 0 (2578) | **40** (2509) | **0** (2457) |
| pomona | 0 (1075) | 0 (1075) | 0 (1090) | 0 (1090) |
| PomonaPitzer | 0 (3110) | 0 (3108) | 0 (3142) | 0 (3142) |
| Vereinsduell | 22 (34537) | 12 (34400) | 22 (32366) | 12 (32255) |

VolyAvw windows (present / off / on our court):

| window | shipped AB+ | shipped rules | stream_s4 AB+ | stream_s4 rules |
|---|---|---|---|---|
| 2772-2865 | 0 / 0 / 0 | 0 / 0 / 0 | 40 / 40 / 0 | 0 / 0 / 0 |
| 1107-1137 | 26 / 0 / 26 | 26 / 0 / 26 | 30 / 0 / 30 | 30 / 0 / 30 |

1107-1137 is on our court in every run: frames of the brief's "inside_strict_frac 0" track are a
dead ball of ours (low, beside our court), not a neighbouring court's. The output there is the
same before and after; the brief's concern came from `inside_strict_frac`, which is 0 for any ball
outside the polygon proper.

Vereinsduell, looked at frame by frame (no labels exist):
- 3282-3291 (vetoed now) is a ball rising past the far-right fence out of the picture, most likely
  ours leaving the venue rather than a neighbouring court's. A dead ball either way.
- 5943-5963 (still output, 3.1 m by the metric, 1.9 m by the selector) is a ball just left of our
  far-left corner, above the side wall. The two disagree because this clip's per-frame court
  keypoints sit visibly off the painted lines (the far baseline about 20 px low); the metric reads
  the frame's keypoints, the selector the scene's court polygon. Ambiguous; left alone.
- 34041 is one motion row after our own ball, tossed towards the bench (track 650), cut off from its
  track by a missing frame: a false flag of the metric's run grouping.

## Decisions

### 1. A track that never comes within 3 m of our sidelines is not selected (8528553)

`score_tracks` records `court_lateral_min_m`, the closest the track's observations come to our
doubles sidelines (same geometry as the metric, on the scene's court polygon), and
`selected_tracks` vetoes the track past `select_off_court_lateral_m` = 3.0 (`selection['off_court']`).
The near-court score term could not do it: `inside_frac` uses a margin of 0.12 x the image
diagonal, which reaches the neighbouring court, so the VolyAvw neighbour track scored 0.86 there.
Effect: stream_s4's 40 VolyAvw frames gone; no other sample clip changed except Vereinsduell's
3282-3291 (the CPU replay before the GPU runs showed the same: only VolyAvw stream_s4 moved). Without a four-corner court nothing
is vetoed.

### 2. Refit moves are bounded and a detection stays a detection (a39f4ae)

The idea is the parked WIP (08e8a21); the evidence is new. A spike or a strong detection off the
local fit used to move without a bound and, past about 44 px, was relabelled interp. On the tuning
set (shipped) those moves put 27 rows more than 30 px from the label, 26 of which had been within
10 px, and repaired 2. On the held-out gate they were 22 of shipped's 25 wrong interp rows.

### 3. Without a blob, only a hole of one detector step is filled (1f21def)

A straight fill with no motion blob is drawn only through one missing detector step (one frame at
30 FPS, two at 60). Longer blob-less fills on the tuning set (shipped) hit 22 labels for 54 false
alarms, 19 of them off the line. No feature separated the good ones (speed, anchor confidence, gap
length were tried). Frames where a blob is found are still placed.

### 4. A point already on its line stays exactly where it was (19fdaf7)

Robustness, found by the suites: on a straight run the refit's residue (~1e-14 px) differs between
BLAS builds and flipped `position_kind`, so the golden passed in one env only. Moves under 1e-6 px
are skipped. The golden fixture, made with numpy 2.5.3 / OpenCV 5.0.0 (no env here has them), is
regenerated with numpy 1.26.4 / OpenCV 4.9.0 and passes in both `tennis-analysis` and `tennis-opt`.

## Rejected

- **Dropping single-hole fills too.** On the tuning set (shipped, exact on the real outputs) it
  takes FA from 342 to 282 but recall from .963 to .962, and it cost one confirmed bounce of 23 on
  the 6 tuning clips with bounce labels (grid_match38, the landing at 514 lost its vertex). The
  invisible labels those fills land on are mostly bridged: the ball's line through a racket or the
  net, where an output is arguably right.
- **Dropping motion rows.** FA 282 -> 170 but recall .962 -> .958; the motion rows hit far more than
  they miss (tuning, shipped: 95 hits, 3 wrong, 112 invisible).
- **A per-frame "in the court picture" test** (not built). An airborne ball over our court appears anywhere above
  it, so a single image position cannot reject a neighbour without rejecting our lobs; hence the
  track-level, ground-distance rule.
- **Restoring the relabel with a tighter threshold.** Every relabelled row checked was a correct
  detection at a bounce or contact; there was nothing to tighten towards.
- **A lower or higher veto margin.** 3 m sits under the 3.66 m ITF minimum run-off; the nearest true
  wide ball of ours seen (PomonaPitzer 871-897) reaches 4-5 m out, but its track also comes down near
  our court, so its closest approach is well inside the margin. Not tuned further: one neighbour track in the data.

## Caveats

- `--reuse-json` rescores existing JSONs; it does not rerun the selector. Selector changes need full
  GPU runs (done here) or `finetune.scoring.replay_selector` (no motion rows) for relative numbers.
- The metric and the veto share geometry; the metric's VolyAvw drop is by construction.
- `system/tests/test_ball_trt.py` fails in `tennis-analysis` (no `onnx` package) and passes in
  `tennis-opt`; environmental, as on AB+.

## Tests

At HEAD: root `python -m pytest -q` in `tennis-analysis` 1325 passed, 10 skipped, 1 failed (the
`onnx` gap above); `cd system; python -m pytest -q` in `tennis-opt` 605 passed. mypy clean at the root
(5 files) and in `system/` (18 files). New scenario tests: the neighbour veto (4), fills over one
detector step at 30 and 60 FPS (2), the refit noise floor (1), the two WIP refit tests, and
`tests/test_off_court.py` (4).
