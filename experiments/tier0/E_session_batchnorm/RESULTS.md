# Test E: per-session BatchNorm re-estimation for stream_s4 e60

**Verdict: KILL.** Re-estimating the BatchNorm running statistics on a clip's own frames (Schneider et
al. 2020, arXiv 2006.16971) with no training does not raise recall at equal false alarms. At a fixed
threshold it moves the whole operating curve: a little more recall and more false alarms, which is a
threshold change, not an improvement. At equal pooled false alarms every variant is below the baseline
(0.9450 to 0.9477 against 0.9486), and at equal false alarms within the grid group, the group this was
meant to help, grid recall falls by 0.008 to 0.017. Nothing reaches the bar (grid or prof +0.01 at equal
false alarms with own-camera not down by more than 0.003). Even the whole-clip upper bound, which a
live phone could not compute, fails it.

All numbers below are measured (raw detector, mixed_v1, 24 clips, 8,515 visible and 518 invisible
labels, PyTorch FP16 trainable form through the tracker's own prepass, window step 1). Nothing here is
an estimate except the last section, which is labelled.

## Sanity check: the PyTorch baseline reproduces TensorRT

`evaluate_archive.py --mode raw --holdout mixed_v1` on the deployed TensorRT engines
(`trt_raw_reference.log`) against this test's `baseline` rows (`rows_baseline.csv`, PyTorch FP16,
`ball_prepass_runtime="torch"`, `ball_prepass_streaming=False`):

| thr | TensorRT recall / FA (bridged+false) | PyTorch recall / FA |
|---|---|---|
| 0.30 | 0.965 / 101 (55+46) | 0.9648 / 101 (55+46) |
| 0.42 | 0.948 / 43 (22+21) | 0.9486 / 43 (22+21) |
| 0.50 | 0.930 / 27 (13+14) | 0.9304 / 27 (13+14) |

Per source at the reference's 0.40: TensorRT grid 0.901 / custom 0.966 / prof 0.963; PyTorch at 0.42
0.897 / 0.963 / 0.957 (the two are at different thresholds, the pooled sweep above is the like-for-like
check). The known pipeline numbers (recall 0.953, FA 95 = 79 + 16 at 0.42) are the selector's output,
not the raw detector's; the raw baseline at 0.42 is 0.9486 / 43.

## Variants

Per clip, independently: a float32 copy of the record, every BatchNorm (29 layers, 2,112 channels)
reset with `momentum=None` (cumulative average), train mode under `no_grad`, the clip's first N frames
fed as the five-frame units the prepass cuts (one per frame of each phase), then eval and FP16.

- `bn60`: N = 60 source frames (1 s at 60 FPS, 2 s at 30; 43 to 52 units)
- `bn300`: N = 300 source frames
- `bnall`: the whole clip (an upper bound; cheating for live use, since it sees the future)
- `bn300_p1`: `bn300` statistics blended 1:1 with the record's (the paper's source prior, weight 1)

## Pooled sweep (recall / FA count, bridged+false)

| thr | baseline | bn60 | bn300 | bnall (cheating) | bn300_p1 |
|---|---|---|---|---|---|
| 0.30 | 0.9648 / 101 | 0.9669 / 102 | 0.9670 / 101 | 0.9674 / 107 | 0.9677 / 108 |
| 0.34 | 0.9598 / 82 | 0.9627 / 82 | 0.9630 / 84 | 0.9636 / 82 | see summary.md |
| 0.38 | 0.9543 / 63 | 0.9569 / 66 | 0.9560 / 63 | 0.9571 / 61 | see summary.md |
| 0.42 | 0.9486 / 43 (22+21) | 0.9499 / 51 (25+26) | 0.9513 / 52 (27+25) | 0.9515 / 49 (28+21) | 0.9527 / 47 (26+21) |
| 0.46 | 0.9398 / 33 | 0.9427 / 41 | 0.9439 / 39 | 0.9445 / 38 | see summary.md |
| 0.50 | 0.9304 / 27 | 0.9341 / 27 | 0.9358 / 31 | 0.9353 / 33 | 0.9378 / 31 |

The full table (every 0.02 from 0.30 to 0.50, all variants) is in `summary.md`.

## Per source group at 0.42 (recall / FA / wrong-object rate)

| group | clips | vis | baseline | bn60 | bn300 | bnall | bn300_p1 |
|---|---|---|---|---|---|---|---|
| grid | 3 | 1592 | 0.8970 / 7 / 0.016 | 0.9070 / 14 / 0.020 | 0.9026 / 15 / 0.023 | 0.9008 / 11 / 0.022 | 0.9026 / 11 / 0.020 |
| custom | 10 | 3569 | 0.9633 / 22 / 0.004 | 0.9625 / 22 / 0.003 | 0.9672 / 21 / 0.001 | 0.9678 / 21 / 0.001 | 0.9667 / 21 / 0.003 |
| prof | 11 | 3354 | 0.9574 / 14 / 0.001 | 0.9568 / 15 / 0.001 | 0.9574 / 16 / 0.001 | 0.9583 / 17 / 0.001 | see summary.md |

Grid at 0.42 looks like +0.010 for bn60, but its false alarms double (7 to 14) and its wrong-object
rate rises: the detector fires more on everything.

## Equal false alarms (the comparison that counts)

Pooled: each variant at the lowest 0.01-grid threshold whose FA count is at most the baseline's 43.

| variant | thr | recall | FA | grid (delta) | custom (delta) | prof (delta) |
|---|---|---|---|---|---|---|
| baseline | 0.42 | 0.9486 | 43 | 0.8970 | 0.9633 | 0.9574 |
| bn60 | 0.45 | 0.9450 | 43 | 0.8989 (+0.0019) | 0.9583 (-0.0050) | 0.9529 (-0.0045) |
| bn300 | 0.45 | 0.9464 | 42 | 0.8989 (+0.0019) | 0.9619 (-0.0014) | 0.9526 (-0.0048) |
| bnall (cheating) | 0.44 | 0.9477 | 43 | 0.8964 (-0.0006) | 0.9644 (+0.0011) | 0.9544 (-0.0030) |
| bn300_p1 | 0.45 | 0.9473 | 42 | 0.8970 (+0.0000) | 0.9608 (-0.0025) | 0.9568 (-0.0006) |

Per group, each group's threshold matched to the baseline's FA count in that group (grid 7, custom 22,
prof 14):

| variant | grid thr / recall (delta) | custom thr / recall (delta) | prof thr / recall (delta) |
|---|---|---|---|
| bn60 | 0.49 / 0.8894 (-0.0075) | 0.42 / 0.9625 (-0.0008) | 0.43 / 0.9559 (-0.0015) |
| bn300 | 0.50 / 0.8863 (-0.0107) | 0.41 / 0.9683 (+0.0050) | 0.44 / 0.9541 (-0.0033) |
| bnall (cheating) | 0.52 / 0.8800 (-0.0170) | 0.41 / 0.9681 (+0.0048) | 0.44 / 0.9544 (-0.0030) |
| bn300_p1 | 0.49 / 0.8863 (-0.0107) | 0.42 / 0.9667 (+0.0034) | 0.45 / 0.9568 (-0.0006) |

The only positive movement is own-camera (+0.003 to +0.005), the group that needs no help and that the
weights trained on (video1 to 9 are not held out for e60). Grid, the target, gets worse the more the
statistics are adapted (bn60 -0.008, bn300 -0.011, whole clip -0.017).

## Per clip (recall delta at 0.42, from summary.md)

The adaptation is noisy clip by clip: video9 +0.016 / +0.069 / +0.065 (bn60 / bn300 / bnall),
grid_match8 +0.019 / +0.013 / +0.011 but with +7 / +8 / +4 false alarms on 22 invisible labels,
grid_match16 -0.009 / -0.016 / -0.014, video43 -0.032 / -0.024 / -0.024, video1 -0.016 / -0.012 /
-0.012. No source group moves together.

## Why it does not work here (interpretation, not measured)

The paper's gains come from corruptions (noise, blur, contrast) that shift every channel's first and
second moments the same way across a whole test set of i.i.d. images. A tennis clip is one scene:
most of its frames are court and background, and the detector's statistics in the trained record
already average over 86 clips of such scenes. Re-estimating on one clip mostly re-centres the
decoder's features around that clip's mean heat, which the sigmoid head reads as a confidence shift
(the curve moves along the threshold axis), and the ball, a 2 to 4 px event, is far too rare in the
per-frame statistics to change the moments that would matter for it. The difference fields (6 of the
9 encoder input channels) are also near zero on static pixels, so a clip's motion content, not its
camera, dominates their statistics, and a quiet first second (bn60) is an unrepresentative sample.

## If it had worked: applying the statistics on the phone (estimate)

Not needed given the verdict, recorded for completeness. The adapted statistics are 29 BatchNorm
layers x (mean, var) x 2,112 channels = 4,224 floats. Folding them into the deployed FP16 convolutions
means rewriting every fused kernel (`repvgg.fuse_for_inference`): 606,352 weights + 959 biases, 1.2 MB
in FP16, of which the encoder is 295,488 weights (0.6 MB). A Core ML ML Program cannot have its
weights rewritten at run time without recompiling, so the practical form would be a per-channel affine
(scale, shift) after each of the 14 fused convolutions (2 x 959 = 1,918 FP16 values, 3.8 KB), applied
as 14 extra elementwise layers on the Neural Engine, or folded into the model's own biases and
weights at session start and the ML Program re-loaded (seconds). Either way the adaptation itself
needs the trainable-form forward over 50 to 300 units in FP32, which is 0.2 to 1.5 GMAC per unit at
batch statistics and is only sensible at session start.

## What run.py does

`python experiments/tier0/E_session_batchnorm/run.py --score --analyze` (tennis-opt interpreter, from
the tests_AE root). `--score` loads the PyTorch trainable `stream_s4_e60.pt`, builds the tracker's
`GridTrackNetBallDetector` with `Config(ball_prepass_runtime="torch", ball_prepass_streaming=False,
conf=0.0)` so TensorRT and the folded network are bypassed and every candidate is kept, and for each
variant and each mixed_v1 clip adapts a fresh float32 copy on the clip's first N frames (`adapt`,
`clip_units`), hands the FP16 result to the detector and writes every labelled frame's label,
prediction and confidence to `rows_<variant>.csv`. `--analyze` scores the CSVs on the CPU with the
workspace's `DetectorMetrics` (the evaluate_archive metric, 1080p pixels, bridged alarms) and writes
`summary.md` / `summary.json`: the pooled sweep, the per-group and per-clip tables and the two
equal-false-alarm comparisons. Variants are `baseline`, `bn<N>`, `bnall`, `bn<N>_p<K>` (prior weight K).
No tracker, selector or trainer code was changed.

## Files

- `run.py`, `RESULTS.md`, `summary.md`, `summary.json`
- `rows_baseline.csv`, `rows_bn60.csv`, `rows_bn300.csv`, `rows_bnall.csv`, `rows_bn300_p1.csv` (0.5 MB each)
- `trt_raw_reference.log` (the TensorRT raw sweep the baseline is checked against), `score.log`, `score_blend.log`
