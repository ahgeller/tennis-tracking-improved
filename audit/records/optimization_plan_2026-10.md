# Optimization plan: a lightning-fast tracker (2026-10-27)

This is the research synthesis behind the `optimization` branch of `optimization_test`
(a copy of `clean_version_gridtracknet_restart`, the `train` layout: tracker in
`system/`, training workspace in `finetune/`). Nine research agents covered
architectures, GPU engineering, ROI tracking, training, physics, strokes,
unconventional ideas, the competitive landscape, and a real profile. Every
workstream on this branch traces back to a section here. "Measured" means measured
on this machine; everything else is an estimate until a workstream measures it.

## 1. Baseline (measured, RTX 5080 Laptop, idle GPU, two runs each)

Clip `system/sample/pomona.mp4`: 1920x1080, 59.67 fps, 2022 frames, 33.9 s, H.264 + AAC.

| stage | seconds | notes |
|---|---:|---|
| imports (before the pipeline timer) | ~3.7 | |
| init (TRT engines, ball warm-up, open video) | 3.3-4.5 | |
| pass 1 | 18.4 | 9.1 ms/frame |
| - motion preprocessing (GPU, many small kernels + syncs) | 6.3 | `countNonZero` 1.3 s, probe delta 1.6 s, CPU connected components, D2H per frame |
| - players (TRT launch 1.8 ms + wait 1.2 ms + ByteTrack 0.65 ms per frame, every frame) | ~6-7 | `preprocess_cuda_chw_frame` 1.3 ms self time |
| - decode/slide | 2.2 | |
| - store | 1.2 | |
| ball prepass (worker thread, overlaps pass 1) | 12.0 | alone: cv2 read 4.3, `.cpu()` sync 3.7, uploads 1.4, resize 1.3; GPU work ~5 |
| selector | 0.31 | not a bottleneck |
| pass 2 | 4.0 | render 2.5 (court layer: 86k `cv2.line`, 1.0), read 1.0, NVENC 0.15 |
| **total** | **26-28 pipeline, ~30 process** | `--no-video`: 22.5 / ~26 |

- GPU utilisation averages 31-36%: CPU, Python and CPU-GPU syncs are the limit.
- Pass 1 runs at 84-91 fps while the prepass runs and 146-176 fps after it: the
  prepass thread and pass 1 contend for the GIL.
- The video is decoded three times on the CPU (prepass, pass 1, pass 2). Pass 2 logs
  `seek to frame 690 ... does not match the cache; decoding from the start`.
- HeatmapNet per 5-frame unit, batch 4: eager FP16 NCHW 2.4-4.7 ms, channels_last
  1.85 ms, **TensorRT FP16 0.91 ms** (ONNX opset 17, max abs diff 0.004).
  `torch.compile` fails (no Triton on Windows).
- Decode alone: OpenCV 415-500 fps, ffmpeg CPU ~2200 fps, NVDEC ~1260 fps.

## 2. Findings several agents reached independently

1. **Each frame is encoded ~5 times.** `gridtracknet_window_step=1` starts a unit at
   every frame. HeatmapNet: ~3.1M params, ~25 GMAC per unit (decoder ~12.6, full-res
   stem ~2.8, each encoder stage ~2.6). Caching per-frame encoder features removes this.
2. **Profile before shrinking.** The ledger already noted the prepass loader, not the
   network, was the bottleneck (81 s vs 55 s); the profile agrees.
3. **Signed frame differences** (TrackNetV4 motion attention, TrackNetV5 MDD) raise
   tennis accuracy cheaply. The README says motion is deliberately not fed to the
   detector, so this must win on `mixed_v1` to land.
4. **The current weights already run on crops.** HeatmapNet is fully convolutional
   (`TOTAL_STRIDE=16`, no global ops): any crop that is a multiple of 16 works. All
   five frames of a unit must share one crop window (early fusion).

## 3. Phases

### A. Plumbing, no retraining (est. 3-6x end to end)
1. TensorRT FP16 HeatmapNet engine (static batch 8-16) through `engine_session.py`.
2. GPU pre/post-processing in the prepass: GPU resize + fp16, heat averaging as one
   ring buffer, argmax on GPU, 3 floats per frame back through pinned memory + events.
3. Prepass out of pass 1's GIL (own process with a shared ring, or one decode feeding both).
4. Decode once with NVDEC (PyNvVideoCodec: Windows wheels, Blackwell from driver
   576.52, NV12 DLPack only), fan frames out; fix the pass 2 seek fallback.
5. Pass 1 diet: motion at half resolution in uint8/fp16, async event-based mask
   downloads, players every 2nd frame with interpolation, CUDA Graphs for fixed shapes.
6. Pass 2: render the static court layer once; pipe NV12 not bgr24.
- Skip INT8/FP8 for now: sm_120 INT8/FP8 conv support is unconfirmed and FP8 conv
  needs channel counts in multiples of 16 (input is 15).

### B. A streaming student model (est. 10-18x less detector compute)
- **StreamNet** (`stream_s4`): per-frame encoder on RGB + signed diff, pixel-unshuffle
  stem, RepVGG blocks (16/32/64/128 ch at strides 2/4/8/16), features cached for five
  frames, light fusion decoder, same 192x108x15 output. ~1.4 GMAC/frame (estimate).
  Differs from the removed ConvGRU `memory_s4`: no recurrence, same five-frame window.
- **HeatmapNet-Lite** (`heatmap_lite`): same topology, widths (16,32,64,128), decoder
  (64,48), space-to-depth stem, RepVGG fused at export. ~2.5 GMAC/unit.
- **WASB-5f**: WASB's small HRNet (~1.5M params, MIT, F1 ~96) with a 15-ch stem and
  BlurBall blur heads. Accuracy-first alternative.
- Recipe: distil from the shipped model (FPD-style MSE on sigmoid heat maps + offsets,
  MGD feature distillation at stride 4 and 8), Noisy-Student pseudo-labels on unlabelled
  tennis (Mendeley 75m8vz7jr2, CalTennis) filtered by track consistency, copy-paste ball
  + speed-scaled blur augmentation, recipe 3 (`hold_conv_scale`), Adadelta 1.0 then 0.3.
- Gates (unchanged): `mixed_v1` raw at 0.50, `epoch_score` >= ctrl1's 0.840, no source
  group down > 0.010, median px <= +0.3, raw curve at equal false alarm,
  `--mode pipeline`, seed 11 repeat under 0.010 margin, prepass seconds on the 5080.
- Bar to beat (ctrl1): raw 0.50 recall 0.929 / wrong 0.006 / false alarm 0.083;
  own camera 0.946, prof 0.953, grid 0.842; pipeline 0.933 / 0.170.

### C. Look only where the ball can be (est. 3-5x on top)
- State machine SEARCH / LOCK / CONTACT / COAST / IDLE (TTNet global-then-local
  cascade; extends `roi_tracking.py`'s lock/coast/probe). LOCK = one 256x192 crop
  centred on the middle of the predicted path, never shifted within a unit.
- First zero-cost experiment: current weights on oracle crops of labelled frames.
- Physics gate: skip units mid-flight while the segment fit residual < ~2 px.
- Audio gate: onset detector for hits/bounces; skips dead time, snaps event times.
- Blur-streak velocity head (BlurBall) as a Kalman measurement.

### D. SwingVision-style stats
- `flight3d`: camera pose by PnP on the 14 court keypoints, segment-wise
  gravity + drag (+ Magnus) fits anchored at bounce z=0. Expect bounce position
  ~10-30 cm near side, 0.5-1 m far side, speed within ~5-10%, spin class only,
  line calls as in / out / uncertain. Never use ball pixel size for depth.
- Strokes only at contacts: rules (serve / volley / overhead), FH/BH from the ball's
  side of the player box (flipped for the far side and left-handers; untested), then a
  pose model must beat that heuristic. Spin class from arc shape + post-bounce kick.
- Scoring state machine on rally legs; doubles as dead-time removal.

## 4. Fact-check of the Google AI Mode conversation (summary)

True: on-device, Neural Engine, founder ex-Tesla Autopilot, patent US11893808B2
(neural 2D-to-3D using court reference features). Unverified or invented: "500 million
shots", "YOLO-like" detector. Wrong: stroke type from the Apple Watch (ITF PAT-25-037:
camera-only gives FH/BH/serve and spin), A13 requirement (A12), ball pixel size for
depth. Patent caution: explicit geometry (PnP + physics fit) is less likely to read on
the patent than learned 2D-to-3D regression; get a freedom-to-operate review before
anything commercial.

## 5. Sources

WASB arXiv 2311.05237; BlurBall 2509.18387; TrackNetV4 2409.14543; TrackNetV5
2512.02789; TTNet 2004.09927; TSM 1811.08383; FPD 1811.05419; MGD 2205.01529; Noisy
Student 1911.04252; DepGraph 2301.12900; RepVGG 2101.03697; copy-paste small objects
1902.07296; Where Is The Ball 2506.05763; TT3D 2504.10035; MonoTrack (CVPRW 2022);
BST 2502.21085; F3Set 2504.08222; ITF PAT-25-037; PyNvVideoCodec docs; NVIDIA ModelOpt.
