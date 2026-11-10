# Test C: track-before-detect, the oracle check

Question: on the frames stream_s4 e60 misses on the mixed_v1 holdout, is the ball present in the
averaged heat grid as a weaker local maximum, so that a path scorer over several frames could pick
it up where the one-argmax prepass does not?

Setup: `dump_peaks.py` runs the tracker's own prepass (`finetune.data.detect.ball_detector`, e60 at
conf 0, TensorRT streaming halves, NVDEC) over the 24 holdout clips and wraps `HeatRing.take` so
the same mean grids also go through a 3x3 NMS and a top-5 above 0.05 (`peaks/<clip>.npz`, 704 KB
in all). `score.py` scores them against the labels at 1080p scale (`detector_metrics.HIT_PX` = 10 px)
and reads the e60_0.42 pipeline run's JSONs (opt1_gate folder) for the pipeline's misses and the
player boxes. No tracker code was changed. Every number below is measured.

## Sanity

- Rank 1 of the dump equals the tracker's own answer on every frame of every clip (checked in
  `score.py`, exact float equality where the argmax is at or above 0.05).
- The rank-1 sweep reproduces the recorded raw e60 numbers (`audit/records/opt_stream_prepass.md`):
  0.952 recall / 55 FA at 0.40 and 0.930 / 27 at 0.50. At the operating 0.42: 0.948 / 43 FA
  (22 bridged, 21 false). The e60_0.42 pipeline misses 404 of 8,515 (recall 0.953, matches the gate).

## Step 2: the oracle ceiling

Visible labels 8,515; a hit is a peak within 10 px (1080p).

| rule | recall |
|---|---|
| rank 1 at or above 0.42 (the raw metric today) | 0.948 |
| rank 1 at or above 0.05 | 0.9765 |
| any of top-2 at or above 0.05 | 0.9833 |
| any of top-5 at or above 0.20 | 0.9775 |
| any of top-5 at or above 0.05 (the ceiling) | **0.9868** (8,403 / 8,515) |

The ceiling clears 0.97, so step 4 ran. But the gap decomposes unevenly: of the 440 frames rank-1
at 0.42 misses, the ball is rank 1 but below 0.42 on 240 (151 of them between 0.30 and 0.42), rank
2 to 5 on 88, and not in the top 5 above 0.05 on 112. Rank 2+ is worth at most 1.0 point of recall;
the rest of the headroom is a threshold question (lowering to 0.30 already gives 0.965 / 101 FA).

Where the true ball sits on the 404 frames the pipeline missed: rank 1: 244, rank 2: 38, rank 3: 15,
rank 4: 11, rank 5: 2, absent: 94. Its confidence when present: 0.05-0.10: 24, 0.10-0.20: 51,
0.20-0.30: 58, 0.30-0.42: 106, at or above 0.42: 71 (the selector dropped or moved a confident
detection on those 71).

By source and speed (rank-1 at 0.42 vs the top-5 ceiling; speed = 1080p px per source frame from the
nearest labelled neighbour within 3 frames):

| group | labels | rank1@0.42 | ceiling | pipeline misses: rank 1 low / rank 2+ / absent |
|---|---|---|---|---|
| grid | 1,592 | 0.897 | 0.961 | 51 / 35 / 53 |
| own | 3,569 | 0.963 | 0.995 | 71 / 17 / 15 |
| prof | 3,354 | 0.957 | 0.990 | 122 / 14 / 26 |

| speed (px/frame) | labels | rank1@0.42 | ceiling | pipeline misses: rank 1 low / rank 2+ / absent |
|---|---|---|---|---|
| 0-6 | 3,202 | 0.936 | 0.987 | 105 / 43 / 38 |
| 6-30 | 4,243 | 0.964 | 0.995 | 97 / 15 / 17 |
| 30-70 | 953 | 0.953 | 0.976 | 26 / 5 / 15 |
| 70+ | 112 | 0.705 | 0.768 | 14 / 3 / 23 |

The fast balls (70+) are the one place the heat grid has nothing to offer: 23 of the 40 misses have
no peak within 10 px above 0.05 at all (grid_match22 and video46 hold 29 of those 40). Slow balls
(0-6, mostly held or at the apex near a player) are where rank 2+ recovers the most: 43 frames, but
that is 1.3% of the bucket.

## Step 3: junk on the 518 no-ball frames

| floor | peaks per frame (top-5 cap) | frames with any | inside a player box |
|---|---|---|---|
| 0.05 | 2.82 (1,463) | 425 / 518 | 633 (43%) |
| 0.20 | 0.37 (192) | 159 / 518 | 76 (40%) |
| 0.42 | 0.08 (43) | 43 / 518 | 13 (30%) |

On visible frames, excluding the ball's own peak, there are 2.33 other peaks per frame above 0.05
and 0.19 above 0.20. A floor of 0.05 therefore hands any path scorer about 3 distractors per
frame, 4 in 10 of them on a player.

## Step 4: the cheapest path scorer

Causal Viterbi over the last 8 source frames, states = (candidate at t-1, candidate at t) over the
top-5 peaks plus "none", gain = confidence, costs = constant-velocity-plus-gravity residual
(0.3 per 10 px, capped at 3 sigma; gravity = median labelled d2y per fps: 0.23 px/frame^2 at 60,
1.59 at 30), 0.35 for a none, 0.5 to re-acquire, 0.3 + speed term for a first link. Constants
fixed a priori, not tuned. Decision statistics on the best path ending at t: the summed path score,
and "lift": conf_t + lam * mean gain of the other 7 frames. Raw metric, no selector.

| rank-1 operating point | recall / FA | best path scorer at that FA: sum | lift 0.25 | lift 0.5 | lift 1.0 |
|---|---|---|---|---|---|
| 0.34 | 0.960 / 82 | 0.845 | 0.956 | 0.951 | 0.937 |
| 0.42 | 0.948 / 43 | 0.679 | 0.938 | 0.929 | 0.894 |
| 0.50 | 0.930 / 27 | 0.586 | 0.927 | 0.905 | 0.857 |

At matched recall, lift 0.25 needs 65 FA where rank-1 at 0.42 needs 43. Every variant loses, and
the loss shrinks as lam goes to 0, i.e. as the statistic degenerates into rank-1 confidence: the
trajectory term adds no separation between the low-confidence true ball and the low-confidence junk,
because at 2 to 3 distractors per frame there is nearly always a junk path that is also smooth
(most of its extra false alarms are bridged ones, the ball drawn across an occlusion). A tuned
version could close some of the gap but has 1.0 point of rank-2+ headroom to win in the first place,
and the pipeline's selector already is a path scorer over confident detections (0.953 / 95 with 79
bridged).

## Verdict: KILL

Track-before-detect on the top-K peaks does not pay. The oracle ceiling (0.987) looks generous, but of
the 328 frames it adds over rank-1 at 0.42, 240 (73%) are the ball at rank 1 below threshold, which
a threshold reaches without any path scoring, and only 88 (1.0 point of recall) are at rank 2 to 5;
the other 112 misses have no peak at all. The cheapest path scorer loses to rank-1
thresholding at every matched false-alarm count. What the numbers point to instead: calibration of
the 0.30-0.42 band (151 true balls there against 58 extra false alarms between 0.30 and 0.42, the
grid clips carrying most of both) and the detector's blindness to balls over 70 px/frame, which no
decode rule can recover.

## Files

- `run.py`: waits for the GPU (under 6000 MiB), runs `dump_peaks.py` then `score.py`; `--score-only` reuses `peaks/`.
- `dump_peaks.py`: the top-5 dump (wraps `ball_gpu.HeatRing.take`; writes `peaks/<clip>.npz`).
- `score.py`: steps 2 to 4; writes `results.json`, `per_frame.csv` (one row per labelled frame: rank-1
  conf, the true ball's rank and conf, speed, pipeline hit, peak counts, DP choice and score) and `score.log`.
- `dump.log`: the prepass log of the dump run.
