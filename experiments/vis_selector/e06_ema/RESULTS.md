# Visibility head as a fill gate: offline evidence (CPU, s4_fg1_e06_ema_vis.pt)

Numbers from the per-frame CSV `vis_logits.csv` (measured) and the e60 0.42 run re-scored with fills removed (an estimate of the gate inside the pipeline: the run's detections are e60's, the logits are this record's).

## (a) AUC of the visibility logit against the held-out labels

| group | visible | absent | AUC |
|---|---|---|---|
| grid | 1592 | 149 | 0.877 |
| own | 3569 | 207 | 0.888 |
| prof | 3354 | 162 | 0.907 |
| all | 8515 | 518 | 0.888 |

On the frames the run filled (the ones a gate judges): 114 visible, 69 absent, AUC 0.512.
On the frames the run output a detection: 8077 visible, 26 absent, AUC 0.785.

Absent labels in a run of at most 3 flanked by visible labels (what a bridged fill crosses): 160, AUC against the visible labels 0.744; in longer runs or at a clip's edge: 358, AUC 0.952.

Probability (sigmoid of the logit) quantiles, visible labels: p5 0.930, p25 0.980, median 0.993; absent labels: p50 0.909, p75 0.956, p95 0.995.

## Baseline (the recorded e60 0.42 run, as Tier 0 B read it)

recall 0.9526 (8111 hits of 8515), FA 95 of 518 (79 bridged / 16 false); fill rows: interp 7 hits / 33 FA, motion 83 hits / 36 FA, det 8021 hits / 26 FA.

## (b) Drop every fill below the probability threshold; (c) keep a fill that closes a single-frame hole

Columns: rows of the run below the threshold (fill FA = interp + motion false alarms, det FA = detector false alarms the gate cannot touch, fill hits = visible labels a fill got right, lost if dropped), then the re-scored run.

| threshold | fill FA below (interp/motion) | det FA below | fill hits below (interp/motion) | drop: recall / FA (bridged/false) | drop: grid / own / prof | keep-single: recall / FA | keep-single hits lost |
|---|---|---|---|---|---|---|---|
| 0.02 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.05 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.10 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.15 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.20 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.25 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.30 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.35 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.40 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.45 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.50 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.60 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |
| 0.70 | 0 (0/0) | 0 | 0 (0/0) | 0.9526 / 95 (79/16) | 0.913 / 0.971 / 0.952 | 0.9526 / 95 (79/16) | 0 |

## Interp-only and motion-only gates (drop)

| threshold | interp only: recall / FA | motion only: recall / FA |
|---|---|---|
| 0.10 | 0.9526 / 95 | 0.9526 / 95 |
| 0.20 | 0.9526 / 95 | 0.9526 / 95 |
| 0.30 | 0.9526 / 95 | 0.9526 / 95 |
| 0.40 | 0.9526 / 95 | 0.9526 / 95 |
| 0.50 | 0.9526 / 95 | 0.9526 / 95 |

## Where the run's fill rows sit on the probability axis

| bucket | fill FA | fill hits | det FA | det hits |
|---|---|---|---|---|
| [0.00, 0.05) | 0 | 0 | 0 | 0 |
| [0.05, 0.10) | 0 | 0 | 0 | 0 |
| [0.10, 0.20) | 0 | 0 | 0 | 0 |
| [0.20, 0.30) | 0 | 0 | 0 | 0 |
| [0.30, 0.50) | 0 | 0 | 0 | 0 |
| [0.50, 0.70) | 0 | 0 | 0 | 0 |
| [0.70, 0.90) | 11 | 11 | 2 | 111 |
| [0.90, 1.00) | 58 | 79 | 24 | 7910 |

Visible labels the run missed outright: 324; the head calls 324 of them visible at 0.5 (these are the frames a fill may still earn).
