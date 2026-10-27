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
| heat1 | d6dc880 | `--arch heatmap_s4 --from-scratch --holdout mixed_v1 --epochs 40 --lr 1.0 --augment basic translate --score-every 2` (Colab A100, 4.4 min/epoch) | 7 | 12 (of 20 scored) | 0.866 / 0.005 / 0.075 (39/518) at 0.50; 0.920 / 0.011 / 0.168 at 0.30 | 0.869 | 0.904 / 0.873 / 0.768 | 3.4 | epoch 10 was 0.897 recall at 0.112 false alarm; from epoch 18 recall fell to 0.11 and from 24 to 0.000 while the training loss kept falling: the decoder's BatchNorm running variance (`up2.1`, `up2.4`) had overflowed float16 to infinity (25104 at epoch 12, 9088 running mean at 24), which silences the eval-mode network only. Fixed in recipe 2 below; the 3.4 px median (bundled 2.1) is the offset head untrained at weight 0.001 |
| heat2 | 747d1ed | `--arch heatmap_s4 --weights heat1/best.pt --holdout mixed_v1 --epochs 16 --lr 0.3 --augment basic translate --score-every 2` (recipe 2 fine-tune of heat1's epoch 12; Colab) | 7 | 4 (of 8 scored) | 0.901 / 0.005 / 0.102 at 0.50 (epochs 2 to 8: 0.889, 0.901, 0.902, 0.891 recall, 0.10 to 0.12 false alarm) | - | - | 1.0 | the offset fix proven: 3.4 px median to 1.0 at once; the `up2.1` running variance fell under weight decay at 0.3 (10110, 8657, 6730, then under the warning line); recall held at heat1's level, the false-alarm rate did not fall. Scored on the PC per source: pending |
| heat3 | 747d1ed | `--arch heatmap_s4 --from-scratch --holdout mixed_v1 --epochs 40 --lr 1.0 --augment basic translate --score-every 2` (recipe 2 from scratch; Colab A100, 4.3 min/epoch) | 7 | 16 (of 8 scored, in progress) | 0.831 / 0.006 / 0.054 at epoch 16; 0.883 / 0.014 / 0.104 at epoch 10 | - | - | 1.0 | the same climb as heat1 on the same schedule: `up2.4` running variance 9348 at epoch 11, 19625, 24039, 44242, 26936, 60577 at 16 (heat1's `best.pt`, epoch 12, held 25104), with the scored recall swinging 0.883, 0.775, 0.451, 0.831 while the training loss fell 0.44 to 0.41: weight decay 1e-4 holds the scale at 0.3 and loses to the step at 1.0. Fixed in recipe 3 below; the run was left going at 1.0 at the user's request |
| heat4 | a928f77 | `--arch heatmap_s4 --from-scratch --holdout mixed_v1 --epochs 40 --lr 1.0 --augment basic translate --score-every 2`, LR 0.1 from epoch 21 (recipe 3 from scratch; Colab A100, 4.4 min/epoch) | 7 | 26 (of 20 scored) | 0.894 / 0.007 / 0.083 (Colab, at 0.50); epoch 38 was 0.915 / 0.008 / 0.114, epoch 34 0.911 / 0.009 / 0.102 | - | - | 0.8 | no warning line in 40 epochs: the hold works at 1.0. At 1.0 the scored rows sat at 0.85 to 0.88 (epochs 8 to 20, bouncing 0.82 to 0.88); the drop to 0.1 at 21 lifted them to 0.89 to 0.92 at once and they stayed there. The winner rule prefers epoch 26's lower false alarm to 38's recall. PC scoring per source and with dense windows: done 2026-10-26 (the dense-window note below) |
| mem1 | 1da1550 | `--arch memory_s4 --weights heat4/best.pt --holdout mixed_v1 --epochs 10 --lr 0.3 --augment basic translate --score-every 2` (stage 3b: the memory network from heat4's winner; Colab A100, 7.3 min/epoch on chains of 4) | 7 | 4 (of 5 scored) | PC, raw at 0.50: 0.925 / 0.007 / 0.087 (45/518); Colab rows 0.919, 0.923, 0.920, 0.927, 0.914 recall at 0.08 to 0.11 false alarm | 0.929 | 0.954 / 0.942 / 0.822 | 0.8 | the memory takes effect in the first two epochs. Against heat4's best.pt with dense windows (0.900 / 0.005 / 0.073; 0.935 / 0.943 / 0.729): nine recall points on the grid clips, two on custom, prof level, false alarm one point higher. Through the pipeline (`--mode pipeline`): recall 0.906 to 0.932, false alarm 0.164 (85) to 0.193 (100), bad outputs on players 38 to 55. A stateful model takes one unit per five frames, so these are its tracker numbers. Control (ctrl1) pending |
| s2_1 | 1da1550 | `--arch heatmap_s2 --weights heat4/best.pt --holdout mixed_v1 --epochs 10 --lr 0.3 --augment basic translate --score-every 2` (stage 3c: the stride-2 network from heat4's winner; Colab A100, 4.5 min/epoch; the run was tagged `heat4_ft` on Drive by mistake, the folder is kept locally as s2_1) | 7 | 2 (of 5 scored) | PC, raw at 0.50: 0.884 / 0.003 / 0.039 (20/518); at 0.30: 0.940 / 0.005 / 0.106 (55); Colab rows 0.885, 0.891, 0.890, 0.902, 0.886 recall at 0.03 to 0.05 false alarm | 0.888 | 0.897 / 0.926 / 0.765 | 0.7 | a different operating point, not a loser: at 0.50 one recall point under heat4's dense 0.900 with the false alarms halved (0.073 to 0.039), at 0.30 the same 0.940 recall at 0.106 against 0.158. Custom down four points, grid up four, prof two under. Through the pipeline: recall 0.893 (heat4 0.906, mem1 0.932), false alarm 0.141 (73; heat4 85, mem1 100). Control: ctrl1 below |
| ctrl1 | 1da1550 | `--arch heatmap_s4 --weights heat4/best.pt --holdout mixed_v1 --epochs 10 --lr 0.3 --augment basic translate --score-every 2` (the control for mem1 and s2_1: the parent itself, 10 more epochs; Colab A100; on Drive the run is `heat4_ft`) | 7 | 6 (of 5 scored) | PC, raw at 0.50: 0.929 / 0.006 / 0.083 (43/518); at 0.30: 0.960 / 0.008 / 0.183 (95); Colab rows 0.886, 0.907, 0.928, 0.918, 0.921 recall at 0.05 to 0.09 false alarm | 0.932 | 0.946 / 0.953 / 0.842 | 0.7 | the control matches mem1 to the digit (0.929 / 0.083 against 0.925 / 0.087): mem1's gain was the ten epochs at 0.3, not the memory. Through the pipeline 0.933 / 0.170 (88), ahead of mem1's 0.932 / 0.193 and heat4's 0.906 / 0.164. s2_1 keeps the one thing the control lacks, half the false alarms at the same threshold (0.039 against 0.083; at 0.30, 0.106 against 0.183 at 0.940 against 0.960 recall). The trainer's guard rail (`promotion_improves`: no rise in false alarm) kept heat4's weights as the winner over epoch 6, so heat4's plateau at 0.1 was not the most the network could do: 0.3 for six more epochs added three recall points |
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
- 2026-10-26, heat1 (the first `heatmap_s4` run; the table above): 0.897 held-out recall at epoch
  10 and the best score at 12 (0.864 / 0.005 / 0.073), then a collapse to zero detections from
  epoch 24 while the training loss kept falling. On the PC the epoch 24 file had infinite
  running variances in the decoder's two last BatchNorms and running means of 9088: with
  BatchNorm after every convolution the loss cannot see the convolutions' scale, Adadelta at 1.0
  let it grow, the pre-normalisation activations passed float16's 65504 in some batches, and one
  such batch leaves the running variance infinite for good. Training mode uses the batch's own
  statistics, so only the eval-mode network (the prepass, the metric, the tracker) went silent.
  Also found: the 3.4 px median at 1080p against the bundled 2.1 is a quarter of a 10 px cell at
  random, the offset head untrained at GridTrackNet's weight of 0.001 (mean offset -0.12 px, no
  bias). Recipe 2 (`train_gridtracknet.HEAT_RECIPE`, recorded as `heat_recipe` in config.json;
  a heatmap run without it is refused as a resume): `loop.HEAT_OFFSET_WEIGHT` 1.0 (CenterNet's
  lambda_off), `loop.build_optimizer` putting the convolutions BatchNorm follows in an Adadelta
  group with weight decay 1e-4 (`NORMALISED_WEIGHT_DECAY`; GridTrackNet keeps its one undecayed
  group, byte-for-byte), `loop.check_finite` stopping an epoch that leaves a non-finite parameter
  or buffer instead of scoring zero for the rest of the run, and a warning line once a running
  variance passes a tenth of 65504. heat1's `best.pt` (recipe 1) stays a valid network: on the
  held-out set at threshold 0.30 it matches the bundled weights' recall with half their false
  alarms (0.920 / 0.168 against 0.919 / 0.326), at 3.6x the speed. heat2 is the same stage 1 on
  recipe 2.
- 2026-10-26, heat2 and heat3 (the table above): recipe 2 fixed the offsets (1.0 px median from
  the first scored epoch) and the fine-tune at 0.3 was calm, but the scratch run at 1.0 grew its
  decoder's running variance exactly as heat1 had, warning from epoch 11 and swinging the scored
  recall from epoch 12; weight decay on the normalised convolutions is a force the step outruns
  at the full rate. Recipe 3 (`HEAT_RECIPE` 3): the decay is gone, and `loop.hold_conv_scale`
  rescales every output channel's filter of the fourteen convolutions BatchNorm follows back to
  a Kaiming filter's norm (`loop.FILTER_NORM`, sqrt 2) after every optimizer step, which is the
  bounded weight normalisation of Hoffer, Banner, Golan and Soudry, "Norm matters" (NeurIPS
  2018): BatchNorm divides each channel's scale out, so the rescale leaves the network's answer
  unchanged (to BatchNorm's eps; `tests/test_training_loop.py` checks a weight scaled by 40 and a
  channel by 10 answer the same, and that an epoch leaves every filter at the norm), the
  pre-normalisation activations stay a fresh network's size at any learning rate, and the
  effective step no longer shrinks as the weights grow, which is what the rate was quietly
  doing. The optimizer is one plain Adadelta group again for every architecture. A fine-tune
  from weights that grew (heat3's decoder, 40x) is brought to the norm at the first step and
  its running statistics follow within the first few dozen steps of the epoch. The warning and
  `check_finite` stay as a check that should never fire. A recipe 2 run (heat2, heat3) is refused
  as a resume; it is continued as a fine-tune under a new tag. heat4 is stage 1 on recipe 3.
- 2026-10-26, dense windows (inference only, no training; branch `dense-windows`,
  `Config.gridtracknet_window_step`, `evaluate_archive.py --window-step`): a five-frame unit
  started at every frame of a phase instead of every fifth, each frame's up to five answers
  averaged grid by grid before one decode (`ball_detectors._UnitStream`, `_HeatAverage`,
  `gridtracknet.decode_grid`). WASB (BMVC 2023, Table 4) measured this as its largest row.
  heat1's `best.pt` on mixed_v1, raw: at 0.50, recall 0.866 to 0.871, false alarm 0.075 (39) to
  0.044 (23), wrong 0.005 to 0.001; at 0.30, recall 0.920 to 0.935, false alarm 0.168 to 0.133;
  prof 0.873 to 0.886, grid 0.768 to 0.773, custom 0.904 to 0.901; median px unchanged. The
  prepass over the 24 clips took 81 s against 55 s (the loader, not the network, is the cost;
  five times the units is 1.5x the time). Also tried and removed: every unit mirrored and the
  two answers averaged (x offsets mirrored back as one minus themselves): alone 0.866 / 0.069,
  on top of dense windows 0.873 / 0.042, within noise for 1.5x more time. Logs in
  `output/dense_windows/`.
- 2026-10-26, dense windows made the default (heat4's `best.pt`, mixed_v1, the PC; logs in
  `output/dense_windows/heat4/`). Raw at 0.50: recall 0.894 to 0.900, false alarm 0.085 (44) to
  0.073 (38), wrong 0.008 to 0.005; at 0.30: 0.925 / 0.212 to 0.940 / 0.158; prof 0.930 to 0.943,
  custom 0.933 to 0.935, grid 0.732 to 0.729; median 0.8 to 0.7 px. Through the pipeline
  (`--mode pipeline`, the selector included): recall 0.898 to 0.906, false alarm 0.178 (92) to
  0.164 (85), the bad outputs landing on players 52 to 38. The prepass took 192 s against 133 s
  over the 24 clips. Both checks better, so `Config.gridtracknet_window_step` is 1; a stateful
  model takes 5 on its own. Reverting is that one field back to 5, the prepass as it was.
- 2026-10-26, stage 3 verdict (mem1, s2_1 and the control ctrl1, the table above; PC scoring
  under `output/dense_windows/<run>/`): the control, heatmap_s4 fine-tuned ten epochs at 0.3 from
  heat4's `best.pt`, reached 0.929 / 0.083 at 0.50, where mem1 reached 0.925 / 0.087: the memory
  network's gain was the epochs. s2_1's threshold curve (`raw_curve.log`, eight thresholds) is the
  control's shifted along the threshold, equal at the control's false-alarm rate (about 0.93 at
  0.35) and at 0.94 recall (0.106 against 0.116), worse at 0.96 recall (0.274 against 0.183), for
  29% more compute. Neither child beats its parent: heatmap_s4 is the network, and the memory and
  stride-2 networks are to be removed per "Keeping one network". The control also showed heat4's
  plateau at 0.1 was not the most the network could do (0.900 to 0.929 dense in six epochs at
  0.3), so the stage 1 schedule drops to 0.3 and not 0.1 (`next steps.txt`, step 5).
- 2026-10-26, promotion: `models/ball_detector.pt` is ctrl1's `best.pt` (heatmap_s4, epoch 6 of the
  ten-epoch fine-tune at 0.3 from heat4's winner; sha256 b4512f3e848d1279...), the tracker's default
  (`tennis_tracker/paths.py`, `BALL_WEIGHTS`). Held out on mixed_v1, dense prepass, the PC: raw at
  0.50 recall 0.929 / wrong 0.006 / false alarm 0.083 (43/518), at 0.30 0.960 / 0.008 / 0.183;
  custom 0.946, prof 0.953, grid 0.842; through the pipeline 0.933 / 0.170 (88). It trained on the
  62 clips outside mixed_v1, so those 24 clips still measure it. The GridTrackNet files moved to
  `models/legacy/` (`gridtracknet_finetuned.npz`, the default until today, measured on 2026-10-21
  in `output/baseline_main_9cb984f`; `gridtracknet_upstream.npz`, the export it came from) with
  the old YOLO `ball.engine`; `clean_tracker.py --weights` runs any of them. The Pomona parity gate
  (`python -m evaluation.parity`) was frozen on the `.npz` and now reports the new detector's
  numbers as differences; it is not refreshed here. The memory and stride-2 networks were removed
  the same day (commit "Keeping one network").
