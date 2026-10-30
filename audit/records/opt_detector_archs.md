# Phase B: fast ball-detector architectures (2026-10-27)

Owner: the Phase B agent of the optimization plan (`optimization_plan_2026-10.md`, section 3.B).
Three new architectures registered in `system/tennis_tracker/ball_models.py`, all on the
existing contract (15-channel five-frame unit at 768x432 in, 15-channel heat + x/y offset layout
on the 192x108 grid out, `.pt` record `{"arch", "state_dict"}`, class attributes `ARCH`,
`GRID_COLS`, `GRID_ROWS`, `INPUT_CHANNELS`, `FRAMES_PER_UNIT`), so `decode_predictions`, the
trainer's heat-map loss, the scorer and the selector take them unchanged:

| id | module | what it is |
|---|---|---|
| `stream_s4` | `streamnet.py` | per-transition shared encoder, cached features, light fusion decoder |
| `heatmap_lite` | `heatmap_lite.py` | HeatmapNet topology, space-to-depth stem, quarter widths, RepVGG |
| `wasb_5f` | `wasbnet.py` | WASB-SBDT's small HRNet, stem strides 2,2, 15-channel head |

Shared blocks: `repvgg.py` (RepVGG block, BatchNorm fold, `fuse_for_inference`).
`ball_models.deployed(model)` returns the fused inference form; `load_ball_model` keeps
returning the trainable form because the trainer fine-tunes from it.

## Decisions and evidence

1. **Signed differences as polarity fields (StreamNet input).** TrackNetV5 (arXiv 2512.02789,
   eq. 1) splits the frame difference into `ReLU(I_t - I_t-1)` and `ReLU(I_t-1 - I_t)`; V4's
   absolute difference loses direction. The paper does not say gray or colour; per channel is
   chosen because it keeps the previous frame exactly recoverable (`prev = rgb - pos + neg`, a
   test checks it), so a transition loses nothing of the pair. MDD's learnable sigmoid mapping
   (eqs. 2-3) is left out (the first conv+BN+ReLU can threshold); untested whether it helps.
2. **The first frame's difference: none needed.** A difference from a frame outside the unit
   breaks the 15-channel contract; a zero difference for position 0 makes a frame's features
   depend on its position in the unit, so the encode-once cache would not hold. Instead a unit is
   its four transitions (f1|f0)..(f4|f3); each is the same function of two consecutive frames
   wherever it sits, so one encoder pass per frame of a phase serves every unit (novel, untested
   for accuracy). Measured: streaming (`stream_units`) equals the unit forward pass to < 1e-6
   (float32) in trainable and fused form (`test_streaming_answers_exactly_as_the_unit_forward_pass`).
3. **Pixel-unshuffle stem.** Lossless space-to-depth (YOLOv5 Focus; TResNet arXiv 2003.13630)
   replaces HeatmapNet's full-resolution stem (~2.8 GMAC of 24.8, measured by conv hooks).
4. **RepVGG, not QARepVGG.** RepVGG (arXiv 2101.03697) trains 3x3+BN, 1x1+BN and identity-BN
   branches and folds them into one 3x3 conv. QARepVGG (arXiv 2212.01593) exists for INT8
   quantisation, which the plan defers on sm_120. RepVGG's every branch is a bias-free conv
   followed by BN inside an `nn.Sequential`, the pattern `loop.hold_conv_scale` holds (the FP16
   overflow fix of heat1/heat3); a test checks every BN in all three networks is such a pair or a
   RepVGG identity.
5. **Fold before export, not in TensorRT.** TensorRT fuses conv + per-channel scale (Layer Fusion
   Catalog, https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/fusion-catalog.html)
   but cannot merge parallel conv branches. Measured (TRT FP16, batch 4, run 2): heatmap_lite
   fused 1.72 ms vs unfused 2.26 ms; stream_s4 3.27 vs 4.61 ms. wasb_5f has no branches and its
   only folds are conv-BN, which TRT does itself: 3.56 vs 3.45 ms (equal within noise).
   Fold exactness: float64 agreement < 1e-9 (tests); float32 deployed vs trainable max abs diff
   1.5e-5 (lite), 7.5e-6 (stream), 3.7e-5 (wasb) on real Pomona frames.
6. **Widths.** StreamNet encoder (16, 32, 64, 128) at strides 2/4/8/16 and decoder
   (128, 64, 48) at 16/8/4; HeatmapNet-Lite (16, 32, 64, 128) and decoder (64, 48), the plan's
   numbers. Not searched; the training runs decide whether they are enough.
7. **`wasb_5f` is included.** WASB's config (`src/configs/model/wasb.yaml`, MIT) ported
   faithfully: 64-ch stem, one Bottleneck(32x4), stages of two BasicBlocks with branches
   (16, 32, 64, 128). Measured 1.485M params, matching the plan's ~1.5M. Deviations to fit the
   contract: stem strides 2,2 (upstream 1,1 at 512x288, i.e. output at input resolution), 15-ch
   head, fusion upsampling to exact size because 432/32 is not an integer. So WASB's published
   F1 does not transfer; its weights do not load. HRNet's many thin branches make it the
   slowest of the three per GMAC.

## Measurements

`tools/bench_ball_archs.py` (random seeded weights, BN statistics calibrated on four real
Pomona units; ONNX opset 17 TorchScript exporter; TensorRT 10.15 Python builder FP16; batch 4 =
`_UNITS_PER_INFERENCE`; CUDA events, median of 7 x 200 calls). Both runs under the shared
main-copy lock (`C:\...\optimization_test\.gpu_bench_lock`): run 1 built and timed, run 2
re-timed the same engines (`--reuse-engines`). A first run taken before the lock fix (each
worktree had its own lock) is discarded. Other agents whose `gpu_lock.py` predates the fix may
still have overlapped these runs; the relative numbers agree across both runs.

| arch | params (M, deployed) | GMAC / unit | GMAC / frame, streaming | TRT ms / unit run 1 | run 2 | vs heatmap_s4 (run 2) |
|---|---:|---:|---:|---:|---:|---:|
| heatmap_s4 | 3.148 | 24.75 | 24.75 | 2.00 | 1.63 | 1.00 |
| heatmap_lite | 0.518 | 3.96 | 3.96 | 0.58 | 0.43 | 0.26 |
| stream_s4 (unit forward) | 0.607 | 6.51 | - | 0.89 | 0.82 | 0.50 |
| stream_s4 (streaming) | 0.607 | - | 2.64 (1.29 enc + 1.35 dec) | 0.42 | 0.38 | 0.23 |
| wasb_5f | 1.482 | 4.58 | 4.58 | 0.93 | 0.89 | 0.54 |

Streaming = per call, four new transitions encoded (encoder engine, batch 4: 0.76 ms) and four
units decoded from cached features (decoder engine, batch 4: 0.84 ms): 1.52 ms per four frames.
At `window_step` 1 a frame is one new unit, so "per frame" equals "per unit" for the others.
The plan's earlier heatmap_s4 TRT figure (0.91 ms/unit) was measured differently and before
the lock existed; the same-script ratios above are the comparison to use.

TensorRT FP16 vs PyTorch FP32 (deployed) max abs diff on real frames: heatmap_s4 0.028,
lite 0.052, stream 0.024 (halves 0.099), wasb 0.069. These are untrained networks with
calibrated statistics; re-check on trained weights before deployment.

## Smoke training and end to end

`train_gridtracknet.py --arch <id> --from-scratch --val-clips video13 --epoch-units 3200
--epochs 1 --batch-size 8 --workers 1 --lr 1.0` (400 steps, the shipped recipe incl.
`hold_conv_scale`), then `clean_tracker.py --input sample/pomona.mp4 --weights
output/smoke/<id>/last.pt --no-video`. Running-mean training loss at steps 1 / 200 / 400 and
the trainer's own held-out scoring (video13-16 at conf 0.5):

| id | loss 1 / 200 / 400 | val loss | held-out recall / wrong / FA | tracker on Pomona |
|---|---|---:|---|---|
| stream_s4 (Kaiming head) | 2530 / 165 / 84 | 1.46 | 0.287 / 0.006 / 0.000 | runs; 181 detections, 166 frames filled |
| heatmap_lite | 28.7 / 4.37 / 3.01 | 0.99 | 0.390 / 0.000 / 0.000 | runs; 302 detections, 320 frames filled |
| wasb_5f | 28.2 / 5.24 / 4.13 | 4.79 | 0.000 / 0.000 / 0.000 | runs; 0 detections |

All three train, save `.pt` records the scorer and the tracker load, and score end to end.
The losses fall for all three. These numbers prove the plumbing, not accuracy (400 steps
from scratch; ctrl1 trained for days).

**Head initialisation (decision 8, measured).** The stream_s4 smoke ran with a Kaiming head and
started at loss 2530. Measured on fresh networks: heat-logit std 0.89 (heatmap_s4), 1.71
(lite), 1.86 (stream), 4.59 (wasb, whose HRNet branch sums are unnormalised), so the mean fresh
heat was 0.18 and 0.30 for stream and wasb against the 0.10 prior. `repvgg.reset_head` now
draws the head from N(0, 0.001) as CenterNet does (`fill_fc_weights`,
https://github.com/xingyizhou/CenterNet): every new network starts at mean heat 0.100 (logit
std <= 0.024), and the heatmap_lite and wasb_5f smokes above, run with it, start at loss ~28.
heatmap_s4 is unchanged.

## Open issues

- Accuracy is unknown until the Colab runs (gates in the plan, section 3.B).
- wasb_5f learns slowest: val loss 4.8 against 1.0 to 1.5 and no detections after 400 steps.
  Its finest branch is 16 channels at stride 4 where upstream's runs at stride 1; give it a
  longer run before judging, and drop it first if it trails.
- TensorRT FP16 output of the untrained networks differs from FP32 by up to 0.1 (stream's
  split halves); re-check on trained weights.
- Environment: another agent's `cuda-python` install moved tennis-opt to numpy 2.2.6, which
  broke cv2 (`_ARRAY_API not found`); numpy 1.26.4 (the pin) was reinstalled at 19:45.
- The GPU lock polls and does not queue in order: these runs waited up to 50 min for it.
- The prepass does not yet use StreamNet's streaming API: `ball_detectors._prepass` would need a
  per-phase ring of four transition features (`encode_frame` once per frame, `decode` per unit).
  That is the prepass owner's code; `streamnet.stream_units` is the reference schedule.
  Done on 2026-10-30: `opt_stream_prepass.md`.
- The tracker's eager path loads the trainable form; call `ball_models.deployed(model)` after
  `load_ball_model` for inference (or export it) to get the fused graph.
