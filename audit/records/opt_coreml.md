# Phase E feasibility: the ball and player models as Core ML for an iPhone timing (2026-10-31)

Owner: the Core ML agent, branch `phase-coreml` (from `phase-live`). Plan: `optimization_plan_2026-10.md`
section 3.E, "first a feasibility test: time the converted ball model on a real iPhone in Xcode".

## What was built

`tools/export_coreml.py` writes `output/coreml_speed_test/`, a folder to carry to a Mac: five ML
Program packages (FP16 compute, FP16 tensors, fixed shapes at batch 1, iOS 17), samples of
`system/sample/pomona.mp4` (four five-frame units, and each unit's first frame as the player detector
takes it) with the PyTorch FP32 answers, `manifest.json`, and the Mac kit from `tools/coreml_speed_test/`
(`MAC_SPEED_TEST.md`, the Xcode steps; `validate.sh` / `validate_coreml.py`, one command that runs every
package with coremltools against the saved answers). `--models stream_s4 heatmap_s4 player` picks what is
exported (`EXPORTS`, default all); the exit status is 1 when a PC check fails, 2 when a source checkpoint
is missing or not the recorded one (the output folder is then left as it was).

| package | from | inputs | outputs | GMAC / call | size |
|---|---|---|---|---:|---:|
| `stream_s4_encoder` | `stream_s4_e60.pt` (md5 fa2358c6...), fused | `frame`, `previous` 1x3x432x768 | `quarter` 1x32x108x192, `eighth` 1x64x54x96, `bottom` 1x128x27x48 | 1.29 | 0.6 MB |
| `stream_s4_decoder` | same | `t1_quarter` .. `t4_bottom` (the four transitions' features, oldest first) | `grids` 1x15x108x192 | 1.35 | 0.6 MB |
| `stream_s4_unit` | same | `frames` 1x15x432x768 | `grids` | 6.51 | 1.2 MB |
| `heatmap_s4_unit` | `ball_detector.pt` (md5 ca1f188b...) | `frames` | `grids` | 24.75 | 6.0 MB |
| `player_detector` | `player.pt` (md5 4de317cc...), Conv and BatchNorm fused | `image` 1x3x640x640 | `predictions` 1x5x8400 (raw head, no NMS) | 3.16 | 5.1 MB |

The decoder takes the four transitions separately (not concatenated as the TensorRT decoder does) so an
app passes the encoder's cached outputs without a 9 MB copy per frame. Per frame at window step 1 the
tracker runs one encoder and one decoder call: stream_s4's phone cost is their sum. A unit's grids are not
the tracker's answer on their own: it averages each frame's grids over the up to five units that contain
it (`HeatRing`) before decoding the peak. At 60 FPS the player detector runs on every other frame, so its
per-frame cost is half a call.

## Where the player and court weights come from

Inside the repository there are no source weights for either: `player.engine` and `courtdetection.engine`
are bare TensorRT plans (no Ultralytics header). Outside it, three folders hold `player` and
`courtdetection` `.pt` / `.onnx`, byte-identical across the three (md5):

| file | md5 | what it is |
|---|---|---|
| `player.pt` | 4de317cc3e2cad7e9196fa438b439544 | YOLO11n detect, 1 class (player), trained at 640, Ultralytics 8.3.171 |
| `player.onnx` | edfb391d8703fe26435cf37b70e7fdbd | its export |
| `player.engine` (HoldsSomeTennis, Interview Prep) | 60d2d31dd45408361dd0dbb08bbd6dab | **the shipped `system/models/player.engine`, byte for byte** |
| `courtdetection.pt` | 833bd8d6024fdd8c0c4aac409695d1ca | YOLO26n-pose, 14 keypoints, end-to-end head, trained at 1280, Ultralytics 8.4.1 |
| `courtdetection.onnx` | b008aa59c1bdf78eefb64514f44e3065 | its export |
| `courtdetection.engine` (HoldsSomeTennis, Interview Prep) | 3d7e85e71777fad5f7c9bf940b8c5500 | 11.0 MB; the shipped one is 26.1 MB, md5 8b57ae924093beef80d9701a644c8378 |

Folders: `C:\Users\Andrew\Desktop\HoldsSomeTennis\models`, `C:\Users\Andrew\Desktop\Interview
Prep\triton-ball-tennis\models`, `C:\Users\Andrew\Desktop\tmp\pass1_perf\engines` (no engines). Both
shipped engines' SHA-256 equal `system/audit/records/snapshot.json`'s.

Run on eight pomona frames (0, 240, 600, 900, 1200, 1500, 1800, 2000), preprocessed by the tracker's own
`TensorRTRuntimeSession.preprocess_frame`, each engine and the `.pt` (Ultralytics, CUDA, FP32 and FP16)
on the identical input tensor:

- **Player: matches.** Same players on every frame (15 boxes) at the tracker's thresholds (0.21, NMS IoU
  0.10); `.pt` FP32 against the shipped engine: IoU >= 0.963, corners within 0.33 px of the 640 input
  (1.0 source px), scores within 0.001; `.pt` FP16: IoU >= 0.994.
- **Court: no match.** The shipped engine finds the court on all eight frames (conf 0.88 to 0.92); the
  `.pt` scores 0.0000 to 0.0001 on frames 0 to 1200 and, where both find it (1500 to 2000), its keypoints
  are 630 px from the engine's. The 11 MB engine beside the `.pt` agrees with it (keypoints within 1 px,
  box within 1.1 px, conf within 0.004), so the `.pt` is that engine's source and the shipped engine was
  built from other weights (2026-07-13), which no folder holds. The court model is not exported.

The player detector is exported from a gitignored copy at `system/models/sources/player.pt`; the export
refuses a checkpoint of another md5 unless `--allow-unmatched-player` is given (the manifest records the
md5 either way, and whether it is the matched one). `export_coreml.player_input` (the session's
preprocessing on the CPU) gives the CUDA session's input bit for bit on all eight frames, and a test
holds it bit-equal to `preprocess_frame` itself on the CPU. The package is the raw head, NMS left to the
app: Ultralytics' Core ML NMS (`nms=True`) wraps the model in a pipeline with an image input and its own
thresholds, which is not the tracker's tensor contract, and the tracker's NMS runs on the handful of
anchors above 0.21 (`engine_boxes._raw_head_rows`; the validator's copy is tested equal to it on random
heads). The boxes stay FP16 like the engine's output: the program computes them in FP16, so an FP32
output would only widen values already rounded.

## Conversion path

coremltools converts on macOS and Linux only: 9.0 (2025-11-10, the latest stable) ships no Windows
wheel, and the docs list macOS and Linux. So the export runs in WSL (Ubuntu 24.04): Miniforge in
`~/miniforge3`, env `coreml` (Python 3.12, coremltools 9.0, torch 2.7.0+cpu, the version 9.0 supports,
torchvision 0.22.0+cpu, numpy 2.0.2, opencv-python-headless 4.10.0.84, ultralytics 8.4.51 (as the
tracker's env; `pip check` notes it would like the non-headless OpenCV, which is not needed), pytest,
mypy, pyflakes, pycodestyle). Command:

```powershell
wsl -d Ubuntu --cd <worktree> -- /home/andrew/miniforge3/envs/coreml/bin/python tools/export_coreml.py
```

The input is the TorchScript trace of `ball_models.deployed` (stream_s4: RepVGG and BatchNorm folded
in float32; heatmap_s4 has no RepVGG, coremltools folds its BatchNorms) and of the Ultralytics model in
its own export mode (fused, `Detect.export`). The saved ball programs hold only `conv`, `relu`, `concat`,
`upsample_nearest_neighbor`, `sigmoid`, `pixel_unshuffle`, `sub`, `mul` (and in the unit, `reshape` /
`slice_by_index`): no `batch_norm` left.

## PC validation (Core ML predicts only on macOS)

On four units (source frames 240, 810, 1800 with the ball in view; 660 without), 20 frames, 15 of them
above the operating threshold (0.42 stream, 0.40 heatmap), each graph against the trainable FP32 network,
scored by the Mac validator's own functions:

| graph | heatmap_s4 unit | stream_s4 unit | encoder (max abs / rel) | decoder | encoder fp16 -> decoder chain | player (4 frames) |
|---|---|---|---|---|---|---|
| fused, eager FP32 | 0 | heat 0, offs 1e-5 | 0.015 / 4.4e-4 | heat 8e-5 | heat 8e-5 | exact |
| traced FP32 (converted graph) | 0 | heat 0, offs 1e-5 | 0.015 / 4.4e-4 | heat 8e-5 | heat 8e-5 | exact |
| traced FP16 (PyTorch CPU) | heat 0.0018, offs 0.026 | heat 0.0009, offs 0.006 | 0.047 / 1.6e-3 | heat 0.0006 | heat 0.0009 | 8 vs 8 boxes, IoU >= 0.958, <= 0.33 px |

Every ball row: heat-map argmax on the same cell in 20/20 frames (15/15 confident). The encoder's FP32
0.015 is not a network difference: the saved references are float16 (they are also the decoder's
inputs), and 0.015 is their rounding, the check's noise floor. The FP16 row is the PC's nearest stand-in
for Core ML's FP16; the Mac's Neural Engine is checked by `validate.sh` (heat within 0.05, offsets on
confident frames within 0.1, encoder levels within 5% relative, players' count equal and IoU >= 0.9).
Tests: `tests/test_export_coreml.py` (shapes, names, metadata, the fuse against the trainable network,
the Mac cases on the PyTorch graphs, a samples-only export's manifest, the CLI's refusals and exit codes,
the copied tracker logic against the tracker's own; the conversion itself where coremltools installs,
i.e. WSL).

To export the court model later: find the weights the 26 MB engine was built from (2026-07-13), check them
the same way, then add them as an `EXPORTS` entry as the player was (YOLO pose, end-to-end head: the
300x48 output already holds the chosen rows, no NMS).

## Open

- The iPhone numbers themselves: the owner runs `MAC_SPEED_TEST.md` on a Mac.
- Which layers the Neural Engine refuses (the encoder's `pixel_unshuffle` and difference fields, the
  unit's 5-D reshape) is only visible in Xcode's report.
- The court model, as above.
