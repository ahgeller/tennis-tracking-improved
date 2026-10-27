# Selector robustness ledger, October 2026

**What this is for.** The design review of the ball-in-play selector (`ball_in_play_selector/`)
proposed a set of changes; each is its own commit on the branch `selector-robust` (off `train`)
so the owner can measure and drop any one of them. This file records, per commit, what changed and
what the CPU replay measured; the last section says which changes only a GPU run can judge, and
the commands.

**How the numbers are made.** `python -m evaluation.replay_selector <run_dir>` feeds the
detections a pipeline run recorded in its tracking JSONs (every track's observations, with the
motion flag the run gave each) back into the selector on the CPU, with the run's court polygon
and player boxes and no motion masks, and scores the output against the labels with the detector
metric (`finetune/data/detector_metrics.py`: a hit within 10 px at 1080p, wrong beyond 30 px, a
false alarm an output on an invisible label, split since 9fa1649 into `bridged`, an output on the
ball's line through a 1 to 3 frame occlusion, and `false`). The source run is the 2026-10-26
control (`output/ctrl_eval` in the main checkout: `evaluate_archive.py --archive finetune --mode
pipeline --holdout mixed_v1` at detector confidence 0.50, 24 clips, 8515 labelled-visible frames,
518 invisible). Fidelity: the control hit 7946 labels, the replayed selector 7923; 23 of the 24
hits lost are `motion` rows the replay cannot make without masks, and apart from those the two
differ on 7 frames. The replay cannot see detections below the run's 0.50 threshold.

Columns: hits / wrong / false alarms as `total (bridged + false)` / tracks, per clips.csv source
(`custom` = video1 to 10, 3569 visible, 207 invisible; `prof` = video17 to 23 and 43 to 46, 3354
and 162; `grid` = grid_match8, 16 and 22, 1592 and 149) and over ALL (8515 and 518).

## Commits

| commit | change | custom hits / wrong / fa | prof hits / wrong / fa | grid hits / wrong / fa | ALL hits / wrong / fa / tracks |
|---|---|---|---|---|---|
| control run (GPU, 0.50) | the run as written, with its motion rows | 3395 / 13 / 42 (31+11) | 3161 / 31 / 27 (27+0) | 1390 / 0 / 19 (19+0) | 7946 / 44 / 88 (77+11) / 134 |
| 9fa1649 | detector metric: false alarms split into bridged and false; measurement only | as above | as above | as above | as above |
| replay harness | `evaluation/replay_selector.py`; `core.output_trajectory` split from the chooser (no behaviour change, golden JSON unchanged); the baseline every row below compares to | 3384 / 9 / 36 (30+6) | 3158 / 33 / 26 (26+0) | 1381 / 0 / 19 (19+0) | 7923 / 42 / 81 (75+6) / 134 |
| 1b track_breaks | switch rule: adjacent rows of one track are the ball moving, not a switch (control: switches 202 to 67, the 135 were fast balls, 99 of them on the 720p prof clips); `chosen_pruned` counted through the tracks' `in_timeline` flag (control: 1 to 3, from `other_track`); the selector returns the chain's record as a pruned copy so a one-track chain's own record keeps every observation. Output unchanged | same | same | same | same |

## What the chooser is worth

The labels mark only the ball in play and the detector now fires almost only on it, so the owner
asked what track selection (scoring, the timeline chain) buys over taking every track, and over
the detector alone. Replay variants on the control JSONs (`--variant chain union raw`):

| variant | what is output | custom hits / wrong / fa | prof hits / wrong / fa | grid hits / wrong / fa | ALL hits / wrong / fa |
|---|---|---|---|---|---|
| raw | every recorded detection at 0.50, no selector | 3375 / 9 / 18 (9+9) | 3188 / 8 / 13 (13+0) | 1340 / 27 / 11 (4+7) | 7903 / 44 / 42 (26+16) |
| union | every track's observations, the fills and the refit over the union, no scoring, no chain | 3386 / 12 / 43 (30+13) | 3165 / 36 / 26 (26+0) | 1363 / 35 / 25 (18+7) | 7914 / 83 / 94 (74+20) |
| chain | the selector as it is | 3384 / 9 / 36 (30+6) | 3158 / 33 / 26 (26+0) | 1381 / 0 / 19 (19+0) | 7923 / 42 / 81 (75+6) |

Reading: against the detector alone the selector is worth +20 hits (the fills), the same wrong
count and +39 false alarms, 36 of them bridged occlusions (the fill through a short gap the
labels call invisible) and 10 fewer false outright. Taking every track instead of choosing costs
41 wrong-object frames (grid_match22 and grid_match16: dead-time tracks on people and the net
that the chain leaves out) and 14 false alarms outright, for 9 fewer hits. The chooser's value is
almost entirely in `grid`: on `custom` and `prof` the union and the chain are within a handful
of frames.

## For the owner's GPU run

(filled in as the commits land)
