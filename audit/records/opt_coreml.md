# Phase E feasibility: the ball models as Core ML for an iPhone timing (2026-10-31)

Owner: the Core ML agent, branch `phase-coreml` (from `phase-live`). Plan: `optimization_plan_2026-10.md`
section 3.E, "first a feasibility test: time the converted ball model on a real iPhone in Xcode".

## What was built

`tools/export_coreml.py` writes `output/coreml_speed_test/`, a folder to carry to a Mac: four ML
Program packages (FP16 compute, FP16 tensors, fixed shapes at batch 1, iOS 17), five-frame samples of
`system/sample/pomona.mp4` with the PyTorch FP32 answers, `manifest.json`, and the Mac kit from
`tools/coreml_speed_test/` (`MAC_SPEED_TEST.md`, the Xcode steps; `validate.sh` / `validate_coreml.py`,
one command that runs every package with coremltools against the saved answers).

| package | from | inputs | outputs | GMAC / call | size |
|---|---|---|---|---:|---:|
| `stream_s4_encoder` | `stream_s4_e60.pt` (md5 fa2358c6...), fused | `frame`, `previous` 1x3x432x768 | `quarter` 1x32x108x192, `eighth` 1x64x54x96, `bottom` 1x128x27x48 | 1.29 | 0.6 MB |
| `stream_s4_decoder` | same | `t1_quarter` .. `t4_bottom` (the four transitions' features, oldest first) | `grids` 1x15x108x192 | 1.35 | 0.6 MB |
| `stream_s4_unit` | same | `frames` 1x15x432x768 | `grids` | 6.51 | 1.2 MB |
| `heatmap_s4_unit` | `ball_detector.pt` (md5 ca1f188b...) | `frames` | `grids` | 24.75 | 6.0 MB |

The decoder takes the four transitions separately (not concatenated as the TensorRT decoder does) so an
app passes the encoder's cached outputs without a 9 MB copy per frame. Per frame at window step 1 the
tracker runs one encoder and one decoder call: stream_s4's phone cost is their sum.

## Conversion path

coremltools converts on macOS and Linux only: 9.0 (2025-11-10, the latest stable) ships no Windows
wheel, and the docs list macOS and Linux. So the export runs in WSL (Ubuntu 24.04): Miniforge in
`~/miniforge3`, env `coreml` (Python 3.12, coremltools 9.0, torch 2.7.0+cpu, the version 9.0 supports,
numpy 2.0.2, opencv-python-headless 4.10.0.84, pytest, mypy, pyflakes). Command:

```powershell
wsl -d Ubuntu --cd <worktree> -- /home/andrew/miniforge3/envs/coreml/bin/python tools/export_coreml.py
```

The input is the TorchScript trace of `ball_models.deployed` (stream_s4: RepVGG and BatchNorm folded
in float32; heatmap_s4 has no RepVGG, coremltools folds its BatchNorms). The saved programs hold only
`conv`, `relu`, `concat`, `upsample_nearest_neighbor`, `sigmoid`, `pixel_unshuffle`, `sub`, `mul` (and
in the unit, `reshape` / `slice_by_index`): no `batch_norm` left.

## PC validation (Core ML predicts only on macOS)

On four units (source frames 240, 810, 1800 with the ball in view; 660 without), 20 frames, 15 of them
above the operating threshold (0.42 stream, 0.40 heatmap), each graph against the trainable FP32 network:

| graph | heatmap_s4 unit | stream_s4 unit | encoder (max abs / rel) | decoder | encoder fp16 -> decoder chain |
|---|---|---|---|---|---|
| fused, eager FP32 | 0 | heat 0, offs 1e-5 | 0.015 / 4.4e-4 | heat 8e-5 | heat 8e-5 |
| traced FP32 (converted graph) | 0 | heat 0, offs 1e-5 | 0.015 / 4.4e-4 | heat 8e-5 | heat 8e-5 |
| traced FP16 (PyTorch CPU) | heat 0.0018, offs 0.026 | heat 0.0009, offs 0.006 | 0.047 / 1.6e-3 | heat 0.0006 | heat 0.0009 |

Every row: heat-map argmax on the same cell in 20/20 frames (15/15 confident). The FP16 row is the PC's
nearest stand-in for Core ML's FP16; the Neural Engine's own FP16 is checked by `validate.sh` on the Mac.
Tests: `tests/test_export_coreml.py` (shapes, names, metadata, the fuse against the trainable network,
the Mac cases on the PyTorch graphs; the conversion itself where coremltools installs, i.e. WSL).

## Player and court models

`player.engine` and `courtdetection.engine` are bare TensorRT plans (no Ultralytics metadata header).
No `.onnx`, `.pt` or other source weights for either exist in the repository, its worktrees or
`system/models`, and a TensorRT plan cannot be converted back. To export them later: obtain the
Ultralytics checkpoints the engines were built from (`models/player.pt`, a YOLO detector at 640;
`models/courtdetection.pt`, a YOLO pose model at 1280, per `build_engines.py`), then
`YOLO(pt).export(format="coreml", imgsz=..., half=True)` (Ultralytics' own Core ML export) or trace
and convert as `export_coreml.py` does. Without the checkpoints they must be retrained
(`system/models/README.md` names open starting points).

## Open

- The iPhone numbers themselves: the owner runs `MAC_SPEED_TEST.md` on a Mac.
- Which layers the Neural Engine refuses (the encoder's `pixel_unshuffle` and difference fields, the
  unit's 5-D reshape) is only visible in Xcode's report.
