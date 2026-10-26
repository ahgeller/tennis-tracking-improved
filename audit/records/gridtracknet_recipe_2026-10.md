# GridTrackNet recipe ledger, October 2026

**What this is for.** The user will retrain GridTrackNet from scratch on the workspace dataset:
the bundled weights were trained on some of these very videos, so fine-tuning them is not the
goal and no weights from these runs are meant for deployment. The runs below fine-tune only
because a fine-tune is the cheap instrument (about 100 minutes each) for measuring which data
choices help on this footage: the resize rule, the augmentations, the hard-negative sampling,
the checkpoint rule. What is kept is the code and these results. A flag that helps a fine-tune
is likely, not certain, to help a scratch run; the scratch retrain should re-check the margins
it cares about on the same held-out set.

The runs behind the detector recipe (plan item 2): every experiment, its flags and seed, what it
measured on the held-out set `mixed_v1` (`finetune/holdouts.json`: prof games 4 and 7, video1 to
video10, grid_match8, 16 and 22; 24 clips, 9,033 labels, 518 of them absent-ball) and what was
decided from it. Logs sit under `output/<run>/` (untracked): `train.log`, `train/results.csv`,
`train/config.json`, `train/winner.json`, `eval_best.log`.

Rules, from the plan: a change is accepted when `promotion_improves` holds on the pooled
`conf>=0.50` line against R0, macro recall is not down, no `source=` line drops more than 0.010
recall and the median px is not up more than 0.3; R1 must also flatten the cell histogram; R4a
and R4b target false alarm down with recall within 0.005. Margins under 0.010 are repeated with
seed 11 and must keep their sign. R5 is every accepted change together and must beat R0 by the
same rule. Nothing is promoted here; `ft.py promote` is the user's call.

Machine: RTX 5080 Laptop GPU (16 GB, driver 617.42, CUDA 12.8 torch 2.10.0), on mains. The
Windows power mode was Balanced (the AC overlay), not Best performance; `nvidia-smi` reported
the SW power cap not active before each run (power limit 104 W of a 120 W maximum). Another
session ran the court-event pipeline evaluation on the same GPU during the early runs, so the
timings below are not comparable between runs, only the metrics are.

All runs: 10 epochs, batch 2, Adadelta lr 1.0, `--score-every 1`, fine-tuning the bundled
weights (`models/gridtracknet_weights_torch.npz`), training on the 40 clips outside the set
(prof games 2, 5, 6, 10; grid 1, 4, 7, 17, 19, 21, 23, 24, 26, 31), the PNG cache of commit 1.

## Runs

| run | commit | flags | seed | best epoch | pooled recall / wrong / false alarm (n) | macro recall | custom / prof / grid recall | med px | decision |
|---|---|---|---|---|---|---|---|---|---|
| E0 linear | 9b4985b | bundled weights, raw, `--holdout mixed_v1 --cell-histogram` | - | - | 0.925 / 0.012 / 0.210 (109/518) | 0.919 | 0.964 / 0.900 / 0.891 | 2.0 | the reference; cell spread 0.033 (x) and 0.034 (y); prepass 65 fps |
| E0 area | 9b4985b | same, `RESIZE_INTERPOLATION` flipped to INTER_AREA by `output/e0_area.py` | - | - | 0.931 / 0.009 / 0.205 (106/518) | 0.924 | 0.967 / 0.905 / 0.905 | 2.0 | +0.6 recall, under the 2 point bar, prepass 58 fps: the recipe stays INTER_LINEAR |
| R0 | 9b4985b | `--epochs 10 --seed 7` (basic augmentation), cancelled in epoch 10 | 7 | 1 (of 9 scored) | 0.938 / 0.011 / 0.305 (epoch 1); 0.936 / 0.007 / 0.230 (epoch 9) | - | - | 1.9 | baseline: recall +1 point over the bundled weights from epoch 1 on, false alarms up and falling with epochs; not scored by source (cancelled) |
| R1 to R5 | - | see `audit/gridtracknet_recipe_experiments.md` | - | - | - | - | - | - | not run: the chain was cancelled on 2026-10-21 at the user's request (ten more hours); the protocol and commands are in the doc |

## E0: the current weights, INTER_LINEAR against INTER_AREA

Full tables, the cell histogram and the per-source reading are in
`audit/gridtracknet_recipe_experiments.md`. Logs: `output/E0/eval_linear.log`, `output/E0/eval_area.log`.

## Notes

- The epoch-by-epoch held-out numbers of R0 are in the doc and in `output/R0/train/results.csv`.
  The checkpoint rule kept epoch 1 (highest recall) although false alarms fell by 0.075 by
  epoch 9; whether the rule should trade a little recall for a large false-alarm drop is an open
  question for the scratch retrain.
- Settled on 2026-10-25, after the scratch run on Colab showed the incumbent rule failing
  outright: its first scored epoch detected nothing (recall 0, wrong 0, false alarm 0), became the
  incumbent, and `promotion_improves` (more recall with wrong and false alarm not up by more than
  METRIC_SLACK) let nothing overtake it (epoch 4, 0.050 / 0.276 / 0.236, lost to "best epoch 2"),
  so `best.npz` would have stayed a detector that detects nothing for all 80 epochs and been
  saved as the winner. An incumbent-anchored rule is wrong for epoch selection in general: an
  early epoch with odd rates anchors every later one. The epochs of a run are now ranked by the
  scalar `epoch_score` = recall - wrong - false_alarm (equal weight: a correct ball is worth one,
  a wrong-object output or a false alarm costs one; a tie keeps the earlier epoch), which also
  answers the R0 question above: epoch 9 (0.936 / 0.007 / 0.230, score 0.699) now beats epoch 1
  (0.938 / 0.011 / 0.305, score 0.622). `promotion_improves` remains the fine-tune guard rail at
  the end of a run (the best epoch against the starting weights), where an anchor is the point.
  Every scored epoch's weights are kept as `epochs/epoch_NNN.npz` so a run can be re-ranked or
  re-scored after the fact, and a resumed run re-derives its best epoch from `results.csv`.
- 2026-10-25: the trainer can now lower the learning rate on a resume (`--resume --lr-drop --lr 0.1`;
  the optimizer's Adadelta accumulators are kept, only the rate changes, and `config.json` records
  every rate as `lr_schedule`, one `{"epoch": N, "lr": value}` entry per rate from epoch 1; a raise
  is refused). Why: the scratch run on Colab (Adadelta, lr 1.0 fixed, no schedule) had held-out
  recall near 0.82 by epoch 24 while the false-alarm rate swung between scored epochs (0.14, 0.38,
  0.21), the mark of a step that is now too large, and the plan is to stop near epoch 40 and carry
  the same run on at 0.1 for the last stretch rather than start over. `winner.json` of a later
  `--all-clips` run copies the schedule in (`selection_run.recipe.lr_schedule`) and the trainer
  says the schedule is not replayed; whether the drop helps is not yet measured, and the matching
  rule for the all-clips run (when to drop, and to what) is open until it is.
- 2026-10-25: a second architecture exists behind one switch. `tennis_tracker/heatmapnet.py`
  (`HeatmapNet`, arch id `heatmap_s4`): a U-shaped encoder-decoder with skip connections
  (stem 24 at full resolution, 48/96/192/256 at strides 2/4/8/16, decoder 192 and 96 at strides
  8 and 4) answering on a 192x108 grid in GridTrackNet's 15-channel output layout, from the same
  five stacked RGB frames; the ball is 2 to 4 px at the input and sub-pixel from stride 4 on, and
  the published work (WASB 2023, TrackNetV2/V3) measured the biggest gain from a fine heat-map
  head with skips. The weights file's suffix is the switch (`Config.model_path`: `.npz`
  GridTrackNet, `.pt` HeatmapNet; `tennis_tracker/ball_models.py`), the tracker's prepass, the
  decoder, `evaluate_archive.py --gridtracknet-weights` and the trainer's metric all follow it,
  and `train_gridtracknet.py --arch heatmap_s4 --from-scratch` trains it on a Gaussian heat
  target (sigma one cell) with CenterNet's penalty-reduced focal loss, writing `last.pt`,
  `best.pt`, `epochs/epoch_NNN.pt`; a fine-tune (`--weights runs/<tag>/best.pt`, no
  `--from-scratch`) carries a winner on to every clip and to new clips later, and `--weights`
  must then be the architecture's own file. GridTrackNet's runs are byte-for-byte unchanged. No
  `heatmap_s4` run has been made yet; when one is, it goes in the table above with the same
  held-out set and rule.
- 2026-10-26: a third architecture, `memory_s4` (`tennis_tracker/memorynet.py`, `MemoryNet`):
  HeatmapNet plus a convolutional GRU at the bottleneck (48x27, 256 channels, 3x3 gates) whose
  state is carried from one five-frame unit to the next of the same phase, so the network knows
  where the ball was before the five frames it looks at; the decoder takes the bottleneck plus a
  1x1 `out` of the state, which starts at zero, so a MemoryNet built from a HeatmapNet's weights
  (`from_heatmap`; the trainer's `--arch memory_s4 --weights <heatmap_s4 .pt>`) answers exactly
  as that HeatmapNet until the memory trains. The prepass runs a `STATEFUL` model one unit per
  phase per call in temporal order with the phase's state (reset at the clip's start; the
  trailing window reads it without updating it), publishing exactly what it published before;
  the trainer's example is a chain of `--chain` (4) consecutive non-overlapping units of one
  clip and phase, the state carried through it, the loss the mean of the units', one
  augmentation draw per chain (`reverse` reverses the unit order too), a chain's sampling
  weight the mean of its units' and `config.json` recording `chain`. A `.pt` now names its
  architecture inside and `ball_models.arch_of` reads it (the suffix is shared). The README's
  stage 3b says how to train it from the heat run's `best.pt` and judge it on the same
  commands, and "Keeping one network" what to delete when an architecture loses. No run yet.
