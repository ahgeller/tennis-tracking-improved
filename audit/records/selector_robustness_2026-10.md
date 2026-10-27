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
| 1c single-detection tracks | `build_tracks` keeps tracks of one observation (schema 5): never selected, written to the JSON, left off the candidate trails; the replay then sees every detection of a new run. Output unchanged; the control JSONs predate it, so the replay row is the same | same | same | same | same |
| 2 two-level confidence | `SelectorConfig.start_conf` 0.50 over the detector floor `Config.conf` 0.40: a detection below it is `weak` (flagged per observation, schema 6), may not start a track or rejoin a stale one, and continues only a live track of `weak_continue_min_obs` (3) observations within `weak_gate_frac` (0.02 of the diagonal) of its Kalman prediction; otherwise dropped. The control JSONs hold nothing below 0.50, so the replay on them is unchanged; the 0.40 and 0.30 runs in `output/selector_baseline` (same 24 clips) are the measurement, see the rows below | same | same | same | same |
| 3a fill speed floors | `_MIN_FILL_SPEED_PXPF` and `_NEAR_PLAYER_FILL_SPEED_PXPF` scaled by the clip's diagonal (1080p reference) as well as its frame rate, and the plain floor 10 to 2 px/frame. Measured on the control: 10 scaled 7940 / 42 / 87 (81+6), 4 scaled 7944 / 42 / 89 (83+6), 2 scaled 7959 / 42 / 89 (83+6), 0 scaled 7960 / 42 / 89 (83+6); 2 keeps a floor and all but one of the hits | 3396 / 9 / 38 (32+6) | 3175 / 33 / 32 (32+0) | 1388 / 0 / 19 (19+0) | 7959 / 42 / 89 (83+6) / 134 |
| 4 speed ratio on observed steps (NOT adopted, no commit) | the association's speed-ratio gate judged against the median per-frame speed of the track's last same-phase steps instead of the Kalman velocity (position gating unchanged). Measured against the 3a row: 4 steps 7949 / 41 / 90 (83+7) / 142 tracks, 2 steps 7947 / 42 / 90 (83+7) / 131, 8 steps 7899 / 37 / 87 (81+6) / 170; the gate switched off entirely 7962 / 51 / 90 (83+7) / 75. The filter's velocity is the better reference (fewest splits for the wrong it removes), so the code was reverted; the gate itself is worth 9 wrong frames for 3 hits and 59 tracks | 3395 / 13 / 39 (32+7) | 3166 / 28 / 32 (32+0) | 1388 / 0 / 19 (19+0) | 7949 / 41 / 90 (83+7) / 142 |
| 5a dead code | removed: the score terms `total_score` never read (`jump_penalty`, `movement_quality`, `coverage_frac`, with `_motion_consistency`, `_jump_penalty` and the cached per-step velocities), the far-outside court fraction nothing read, the direction gate (`assoc_dir_cos_min = -1.0` never fired) and the area gate (the detector's box is synthetic, so the ratio is 1); `assoc_min_speed_for_dir` renamed `assoc_min_speed_for_ratio`; `BallKalmanFilter` docstring says what the filter does. Replay identical to 3a, golden frames identical (the three keys leave the breakdown) | same | same | same | same |
| 5b drop before refit | `_drop_implausible_fills` runs before `refine_trajectory`, so a fill that left the frame or jumped the court is not among the points the refit smooths towards. Replay identical to 5a on every clip line (the drop fires on rows the refit's runs did not contain) | same | same | same | same |
| 5c timeline gaps in seconds | `timeline_max_gap_frac` 0.85 and `timeline_small_gap_frac` 0.02 of the clip become `timeline_max_gap_sec` 11.0 and `timeline_small_gap_sec` 0.26 (those fractions of the 13 s median held-out clip, 1.6 to 30 s). Replay identical to 5b on every clip line: no control clip's chain met either limit | same | same | same | same |
| 5d pixel constants scaled | the jump gate's 120 px floor, the refit's kink speed (2.5 px/frame) and spike floor (10 px), the relabel floor and the Kalman bounce thresholds (vy > 2, residual < -10) are read in the clip's pixels (1080p reference, `core._px`, `physics._bounce_scale`). Measured one group at a time: spike and kink +2 hits, jump gate and bounce thresholds no change; scaling the refit's bounds, Huber width and outlier floor cost 17 hits on the 720p clips (p90 4.8 to 3.6 px but hits lost), so those stay native pixels | 3396 / 9 / 38 (32+6) | 3177 / 33 / 32 (32+0) | 1388 / 0 / 19 (19+0) | 7961 / 42 / 89 (83+6) / 134 |

The two-level rule, replayed on the runs that hold detections below 0.50 (`output/selector_baseline/conf_0.40`
and `conf_0.30` in the main checkout, the same 24 clips; the chain as it stands is the row above each):

| run | rule | custom hits / wrong / fa | prof hits / wrong / fa | grid hits / wrong / fa | ALL hits / wrong / fa / tracks |
|---|---|---|---|---|---|
| 0.40 | none (chain) | 3418 / 11 / 35 (29+6) | 3176 / 36 / 27 (26+1) | 1404 / 0 / 16 (16+0) | 7998 / 47 / 78 (71+7) / 137 |
| 0.40 | 0.50, 3 obs, 0.02 | 3418 / 9 / 37 (31+6) | 3173 / 36 / 26 (25+1) | 1402 / 0 / 16 (16+0) | 7993 / 45 / 79 (72+7) / 138 |
| 0.30 | none (chain) | 3449 / 10 / 36 (28+8) | 3195 / 42 / 40 (35+5) | 1433 / 0 / 17 (16+1) | 8077 / 52 / 93 (79+14) / 136 |
| 0.30 | 0.50, 3 obs, 0.02 | 3449 / 8 / 37 (30+7) | 3188 / 39 / 37 (34+3) | 1416 / 0 / 17 (16+1) | 8053 / 47 / 91 (80+11) / 139 |

Other settings (ALL, hits / wrong / fa): gate 0.03 or 0.05 at 3 observations, 0.40 run 7994 / 47 / 77, 0.30 run
8054 / 51 / 90 and 8070 / 51 / 91; 2 observations at 0.02, 7994 / 45 / 79 and 8055 / 47 / 91; at 0.05, 7995 / 47 /
77 and 8072 / 51 / 91. Reading: the rule is neutral at 0.40 (-5 hits, -2 wrong, +1 false alarm) and costs hits at
0.30 (-24 for -5 wrong); the hits it loses are tracks that began on a weak detection. It is kept at the asked
defaults for the owner's GPU run to judge; `start_conf` set to the detector floor switches it off.

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

The same three variants on the runs made at the lower detector thresholds (`output/selector_baseline`,
same 24 clips; the chain here is the selector as it stood at the harness commit):

| run | variant | custom hits / wrong / fa | prof hits / wrong / fa | grid hits / wrong / fa | ALL hits / wrong / fa |
|---|---|---|---|---|---|
| 0.40 | raw | 3427 / 10 / 29 (16+13) | 3222 / 9 / 17 (16+1) | 1401 / 29 / 13 (6+7) | 8050 / 48 / 59 (38+21) |
| 0.40 | chain | 3418 / 11 / 35 (29+6) | 3176 / 36 / 27 (26+1) | 1404 / 0 / 16 (16+0) | 7998 / 47 / 78 (71+7) |
| 0.30 | raw | 3468 / 15 / 37 (19+18) | 3264 / 14 / 38 (30+8) | 1432 / 30 / 19 (10+9) | 8164 / 59 / 94 (59+35) |
| 0.30 | chain | 3449 / 10 / 36 (28+8) | 3195 / 42 / 40 (35+5) | 1433 / 0 / 17 (16+1) | 8077 / 52 / 93 (79+14) |

At 0.40 the detector alone has 52 more hits than the chain and 19 fewer false alarms, for one more
wrong frame: below 0.50 the detector finds ball frames the selector then discards (its `prof`
tracks lose 46 hits to the chain's pruning and chooser), while the chain's fills add bridged
occlusions. The chooser's one clear win stays `grid`: 29 wrong frames at 0.40 and 30 at 0.30 that
the chain turns into 0.

The reading above that the chain's pruning and chooser discard the `prof` frames has the cause
wrong. Taking every track without choosing loses the same hits, and per frame none of the 52 raw
`prof` hits the selector misses at 0.40 is absent from the output: all are output more than 10 px
from the label, about two thirds still `det` and a third relabelled `interp`. The refit moves
them; it, not the chooser, is the next thing to measure.

### A minimal selector, built and rejected (2026-10-27)

Because the labels mark only the ball in play and the chooser bought little on them, a minimal
selector was built in place of the score and the timeline chain: every track kept unless fewer than
40% of its observations are on or near the court (the court term of the chain's score; every ball
track on the three runs is at 0.56 or above, the dead-time junk at 0.20 or below) or it has fewer
than 6 observations. Replayed on the selector-robust 0.40 run it tied the chain: 8005 hits / 45
wrong / 81 false alarms (74+7) against 8002 / 45 / 81, and it cost no time. On
`sample/VolyAvw-cut-merged-1771797159395.mp4`, which no label covers, it failed where two balls
are on screen at once: at about 10.5 s it kept a ball near the camera between points beside the
server's, and the trail crossed the court from one to the other; at about 18.5 s it kept the
server's ball and a loose ball by the fence and the fills swung between them. Choosing one of two
simultaneous tracks is what the score and the chain do, and the labelled clips hold no such
moment, so the replay could not see it. The chain stays the only selector. A labelled clip with a
second ball in dead time would let the metric catch this.

## For the owner's GPU run

The replay measures everything above except what needs a detector run: proposal 2 at the 0.40
floor on fresh detections (the control JSONs hold none below 0.50; the 0.40 and 0.30 runs are the
proxy), the motion rows the fills take from the masks (3a's lower floor lets more tails and gaps
reach the masks), and the 60 FPS phase behaviour of the association on new detections. From the
main checkout, on the `selector-robust` branch:

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1 --gridtracknet-conf 0.40 --out-dir output/selector_robust_eval
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' track_breaks.py output/selector_robust_eval --compare output/ctrl_eval/track_breaks.log
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m evaluation.compare_eval_logs output/ctrl_eval/pipeline.log output/selector_robust_eval/pipeline.log
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m evaluation.replay_selector output/selector_robust_eval --archive finetune --variant control chain union raw
```

Results (held-out `mixed_v1`, 24 clips, detector floor 0.40, `evaluate_archive.py --mode pipeline`):

| run | recall | wrong | false alarms (bridged + false) |
|---|---|---|---|
| unchanged selector (`train`, `output/selector_baseline/conf_0.40`, rescored with the bridged split: `output/selector_robust_eval/baseline040_rescored.log`) | 0.942 | 0.006 | 90 (81+9) |
| `selector-robust` at 402fe2e (`output/selector_robust_eval/pipeline.log`) | 0.945 | 0.006 | 95 (86+9) |

`track_breaks` against the unchanged 0.40 run: hits +26, no_track -19, other_track -7,
chosen_pruned +0. The replay on the new run's JSONs (`output/selector_robust_eval/replay.log`):
control 8048 hits / 50 wrong / 95 false alarms (86+9), chain 8002 / 45 / 81 (74+7), union 7996 /
93 / 93 (71+22), raw 8035 / 49 / 55 (37+18); the control is 46 hits over its own chain replay, which
cannot make the motion rows without masks. Each run's detector-alone row set beside it: the old
selector had 25 hits fewer than its run's detector alone (8025 against 8050), this one 13 more
(8048 against 8035), both cutting the false alarms outright from about 20 to 9. On
`sample/VolyAvw-cut-merged-1771797159395.mp4` it outputs 26 more frames than the old selector (38
short gap fills of 1 to 4 frames, 12 single frames dropped, none moved more than 15 px) and the
selector takes 1.08 s of a 58.8 s run against the old one's 1.10 s. Merged to `train` and `main` on
2026-10-27 as the tracker's selector.

The control was made at 0.50; `output/selector_baseline/conf_0.40/pipeline.log` is the 0.40 run
of the unchanged selector (ALL recall 0.942, 90 false alarms) and the fairer `compare_eval_logs`
first argument. The `track_breaks` switch column is not comparable to the control log's (1b
changed the rule: 202 became 67 on the control JSONs); rerun `track_breaks.py output/ctrl_eval/pipeline`
for a like-for-like control. The new run's JSONs are schema 6 and carry every single-detection
track and the `weak` flag, so the replay on them sees everything the selector saw, down to 0.40.
