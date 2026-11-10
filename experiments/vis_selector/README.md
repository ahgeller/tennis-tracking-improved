# The visibility head as a fill gate

`run.py` reproduces everything here on the CPU (no GPU, no tracker run; about 10 minutes for the
24 held-out clips, the logits are cached in `vis_logits.csv`). `RESULTS.md` is its report for
`s4_fg1_e10_vis.pt` (the epoch-10 live weights), `e06_ema/` the same for the epoch-6 EMA weights,
`gate_sweep.csv` the threshold sweep as a table.

## What was measured (facts)

- The head's AUC against the 9,033 held-out labels is 0.892 (e10) and 0.888 (e06 EMA), per source
  grid 0.885 / own 0.904 / prof 0.900: the Colab number holds on the CPU with the tracker's own
  frame schedule (a unit at every frame of each 30 FPS phase, the logits of the up to five units
  holding a frame averaged).
- The head is good at the ball being gone and poor at it being hidden. Against the visible
  labels, the 358 absent labels in long runs or at a clip's edge score AUC 0.964; the 160 absent
  labels inside a run of at most three flanked by visible labels, the occlusions a bridged fill
  crosses, score 0.729.
- On the 183 labelled frames the e60 0.42 run filled (114 fill hits, 69 fill false alarms, the
  rows the gate would judge) the AUC is 0.449 (e06 EMA: 0.512): no better than a coin. The fill
  false alarms and the fill hits sit on the same probabilities (46 of the 69 false alarms and 46
  of the 114 hits above 0.9).
- The head is biased towards visible: the absent labels' median probability is 0.639 (e06 EMA:
  0.909), the visible labels' 5th percentile 0.782.

## The threshold sweep (estimate: e60's detections, this record's logits)

No threshold up to 0.5 changes a single fill of the e60 run. At 0.60 one motion fill goes (a
false alarm and a hit: recall 0.9524, FA 94); at 0.70 five false alarms (all non-bridged motion
fills) and seven hits go (recall 0.9517, FA 90). Tier 0 B's "no interp" rule removed 33 false
alarms for 7 hits; the gate at 0.70 removes 5 for 7. Interp-only and motion-only gates do nothing
below 0.5 either. The "keep a single-frame hole" variant changes nothing.

## Verdict

With the epoch-10 head, the right operating point is off (`ball_visibility_gate` 0.0). The 69 fill
false alarms the gate was meant to remove are occlusions of one to three frames that the head,
pooled over a whole five-frame unit, calls visible as confidently as a real ball. The plumbing is
in place for the morning model; the one-line GPU check is whether its curve on the filled frames
has moved off 0.5. If it has not, the head's training target (every absent label weighted alike)
or its pooling (a per-frame answer from the stride-16 fused features, not the per-frame heat
channel) is the thing to change, not the gate.
