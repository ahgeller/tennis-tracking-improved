# Test A: the gate split into in-flight, held/static and speed bands

Verdict: **UNCLEAR, leaning no.** opt1 does not beat e60 on in-flight balls by more than the
noise band at equal false alarms. The two models trade places on which frames they hit (94
gained, 93 lost, paired per frame), and what looks like an in-flight edge (+20 hits, +0.0024) is
paid for by held balls (-19 hits) and sits far inside the training-noise band (0.026).

All numbers below are measured (`run.py` over the recorded gate JSONs; CPU only). The `all` row
reproduces the gate log for every run (e60_0.42: 8111 hits = recall 0.953, FA 95 = 79 / 16).

## Definitions

- in-flight: visible label that is not held/static. held/static: label speed < 6 px/frame
  (1080p, per source frame, from the neighbouring labels) AND inside a player box of the run's
  own JSON. 274 of the 8515 visible labels are held/static (3.2%).
- A false alarm (output on an invisible label) is classed by the output's own speed and
  position, since the label has none. The `invis` column of a sub-bucket counts only invisible
  frames with an output; quiet frames have no bucket.
- bridged = within 10 px of the line between the flanking visible labels (run of <= 3).

## Pooled scorecard (hits / visible, FA = bridged / false)

| run | all recall | all FA | in-flight recall | in-flight FA | held recall (n=274) | held FA |
|---|---|---|---|---|---|---|
| e60_0.42 | 0.953 (8111) | 95 (79/16) | 0.961 (7923/8241) | 91 (75/16) | 0.686 (188) | 4 (4/0) |
| opt1_0.38 | 0.953 (8111) | 92 (77/15) | 0.964 (7943/8241) | 90 (75/15) | 0.617 (169) | 2 (2/0) |
| e60_0.34 | 0.959 (8166) | 113 (90/23) | 0.967 (7970/8241) | 106 (83/23) | 0.715 (196) | 7 (7/0) |
| opt1_0.30 | 0.960 (8177) | 112 (89/23) | 0.970 (7996/8241) | 108 (86/22) | 0.661 (181) | 4 (3/1) |

At matched FA the in-flight gap is +0.0024 (opt1_0.38 vs e60_0.42, 90 vs 91 FA) and +0.0032
(opt1_0.30 vs e60_0.34, 108 vs 106 FA). Paired per frame, opt1_0.38 hits 89 in-flight labels
e60_0.42 misses and misses 69 it hits: McNemar chi2 = 400/158 = 2.5, p about 0.11 even treating
frames as independent, which they are not (misses come in runs). Not significant.

## Speed bands (pooled, e60_0.42 / opt1_0.38)

| band (px/frame) | visible | e60 recall | opt1 recall | e60 FA | opt1 FA |
|---|---|---|---|---|---|
| 0-6 | 2961 | 0.943 | 0.946 | 25 (20/5) | 19 (17/2) |
| 6-30 | 4344 | 0.972 | 0.969 | 61 (52/9) | 64 (54/10) |
| 30-45 | 648 | 0.961 | 0.958 | 6 (5/1) | 6 (5/1) |
| 45-70 | 405 | 0.904 | 0.916 | 3 (2/1) | 3 (1/2) |
| 70+ | 152 | 0.704 | 0.711 | 0 | 0 |

Per source group: `scorecard.csv` (grid / own / prof rows for every bucket and run). Grid is
weakest everywhere (in-flight 0.928 e60, 0.922 opt1; 70+ px/frame: 7 of 25 hit).

## Noise band

`C:\Users\Andrew\Desktop\label_backups\models\stream_s4_new\results.csv` scores the raw
held-out recall every second epoch. Recomputed sd of recall over the late epochs:

| window | epochs | mean | sd | sd of a difference of two runs | meaningful (95%) |
|---|---|---|---|---|---|
| >= 60 | 19 | 0.919 | 0.0137 | 0.019 | 0.038 |
| >= 70 | 14 | 0.924 | 0.0094 | 0.013 | 0.026 |
| >= 76 | 11 | 0.927 | 0.0070 | 0.010 | 0.020 |

The 0.013 the other agent quoted is the >= 60 window, which still includes the learning-rate
tail (epoch 60 is where the loss steps down). The honest band for a converged run is sd about
0.007-0.009, so a difference between two independently trained models is meaningful only above
about 0.02-0.026 recall (raw mode; the pipeline's selector damps it, estimate: the pipeline band
is somewhat narrower). The raw false-alarm count has sd about 3 of 518 across those epochs. The
binomial sd of one run's pipeline recall at n = 8515 is 0.0023, and it understates the truth
because misses are correlated in runs. Every opt1-vs-e60 difference in the table above
(0.000 to 0.003 recall, 1 to 3 FA) is well inside both bands.

## What surprised me

- 70+ px/frame balls (fast serves and smashes, 152 labels) are hit only 70%: the single biggest
  in-flight recall hole per label, and threshold does not help (0.704 -> 0.717 at 0.34). Nearly
  all are prof (109) and grid (25).
- Held/static balls (274) are hit 62-72%, but they produce almost no false alarms (2-7), so
  their 80-100 misses are a recall-only problem the product may not care about.
- The bridged false alarms sit overwhelmingly in the 6-30 px/frame band (52 of 79 at e60_0.42):
  the ball in a normal rally passing behind a player or the net post, drawn through.

## Files

- `run.py`: the scorer; `python experiments/tier0/A_scorecard_split/run.py` (about 20 s).
- `scorecard.csv`: every bucket x group x run. `noise_band.json`: the >= 70 window.
