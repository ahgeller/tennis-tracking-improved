# Ball prepass optimization (Phase A items 1-3), October 2026

Branch `worktree-agent-a6d684a271bfee9ad`, off `optimization` (93771f8). Machine: RTX 5080 Laptop,
torch 2.10.0+cu128, TensorRT 10.15.1.29, OpenCV 4.9.0, numpy 1.26.4, Python 3.10, Windows 11.

**Timing conditions.** Every timing here was taken through the main copy's shared lock
(`optimization_test/tools/gpu_lock.py`, lock `.gpu_bench_lock`), so none needed re-taking after
the coordinator's lock correction. The GPU sat thermally throttled (86-87 C, 1485-1665 MHz,
98% utilisation) through the whole session, and run-to-run spread on an identical command is up
to ~25% (orig `--no-video` measured 26.6 s and 30.5 s in one suite). Correctness and parity runs
were unlocked; their timings are not quoted.

## Decisions and evidence

| # | Decision | Evidence |
|---|----------|----------|
| 1 | TensorRT FP16 engine from an ONNX export (opset 17, TorchScript exporter, `dynamo=False`), FP16 input and output | TensorRT best practices (https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) and measurement: 1.17-1.58 ms a unit against 3.75-4.15 ms FP16 eager (table A) |
| 2 | Static batch 4 | Sequential sweep favoured 4 (1.167 ms/unit), but a 5-round round-robin that cancels thermal drift measured 2/4/8/16 within 6% of each other (table A). 4 publishes results sooner, matches the torch path's 4 units a call and needs the smallest heat ring |
| 3 | Engine cache key: recipe, weights sha256, arch, batch, input size, TRT version, torch version, GPU name and SM | An engine is only valid for the TRT version and GPU that built it (TRT docs, "Engine compatibility"); any other change rebuilds automatically. Build takes ~22 s once per weights |
| 4 | PyTorch fallback when TRT/onnx is missing or the build fails, or `ball_prepass_runtime="torch"` | Required by the brief; tested in `tests/test_ball_trt.py` |
| 5 | GPU resize as OpenCV's fixed-point INTER_LINEAR, not a float bilinear | OpenCV `modules/imgproc/src/resize.cpp` (11-bit coefficients, `VResizeLinearVec_32s8u` rounding). Measured on 20 Pomona frames: emulation 0 of 19.9M bytes differ; rounded float bilinear differs in 4.6% by 1 level; unrounded in 59.8% (max 0.625) |
| 6 | Heat average as a device ring (`HeatRing`), sums by `index_add_` latest position first, means by multiply with `reciprocal(counts)` | Reproduces the host summation order bit for bit; CUDA's division by a scalar is a multiply by the float32 reciprocal (PyTorch `BinaryDivTrueKernel`, verified by measurement) |
| 7 | Peaks found on the device, 4 floats a frame back through pinned memory with an event (`Readback`) | `torch.argmax` returns the first maximum like `np.argmax` (PyTorch docs); pinned + `non_blocking` copy per https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html |
| 8 | cv2 decodes straight into pinned buffers (`PinnedCapture`) | Measured (table B): 1.233 ms a frame read + upload against 3.591 ms via `Tensor.pin_memory()` and 6.066 ms pageable |
| 9 | Decode behind `FrameReader` / `open_frames` | The single swap point for an NVDEC reader |
| 10 | Overlap mode: see "Getting the prepass out of pass 1's way" | Measured (tables C, E) |

## A. Engine batch (ms per unit, HeatmapNet heatmap_s4, FP16 I/O)

Sequential sweep: B1 1.538, B2 1.297, B4 1.167, B8 1.543, B16 1.787 (B8 with FP32 I/O 1.502).
FP16 eager: B4 4.148, B8 3.750. Max output difference against eager: 0.004-0.005.
Round-robin, 5 rounds, median: B2 1.608, B4 1.577, B8 1.581, B16 1.522.

## B. Read and upload, ms a 1080p Pomona frame

| path | ms |
|---|---|
| read only | 1.485 |
| read + CPU resize (original) | 2.653 |
| read + pageable upload + GPU resize | 6.066 |
| read + `pin_memory()` + GPU resize | 3.591 |
| read into pinned + GPU resize | 1.233 |

(Read into pinned beats read alone because cv2 then writes into a reused buffer instead of
allocating a fresh frame.)

## Getting the prepass out of pass 1's way (item 3)

Options researched:
- **Worker thread** (the original design). PyTorch's bindings release the GIL during kernels, but
  Python's 5 ms switch interval (https://docs.python.org/3/library/sys.html#sys.setswitchinterval)
  and the convoy effect (https://bugs.python.org/issue7946) still serialise the host-side work of
  two busy threads.
- **Child process** (`gridtracknet_prepass_process`, `engines/ball_prepass_process.py`): no shared
  GIL, results sent back as rows over a loopback socket. A second process means a second CUDA
  context; without MPS (Linux only, https://docs.nvidia.com/deploy/mps/) contexts time-slice the
  GPU instead of sharing it.
- **CUDA streams / priorities**: the prepass already runs on its own stream (the engine's). The
  default stream has the lowest priority already, so giving pass 1 priority would need changes to
  pass 1's code, which this item does not own.
- **Serial** (prepass first, then pass 1 alone): no contention at all.

### C. End-to-end Pomona, interleaved suite G (the decisive one)

`clean_tracker.py --input sample/pomona.mp4`, one lock acquisition, runs interleaved in the order
listed. Before each run the suite listed other python processes on the GPU: one agent's
**unlocked** GPU job (pid 34032) was present for runs 3-9 ("busy" below). The log's
"[pass 1] Done ... fps" divides by time since start including init, so pass 1 fps here is
2022 / (pass 1 - init). With the serial default, init contains the 4 s prepass.

| run | GPU | total s | init s | prepass s | pass 1 fps |
|---|---|---|---|---|---|
| orig `--no-video` | clean | 22.6 | 2.7 | 12.5 (overlapped) | 104 |
| **serial `--no-video`** | clean | **21.1** | 6.8 | 4.0 | **147** |
| thread `--no-video` | busy | 24.0 | 2.9 | 8.7 | 98 |
| process `--no-video` | busy | 26.2 | 2.9 | 6.0 | 90 |
| orig video | busy | 33.6 | 3.4 | 15.6 | 87 |
| serial video | busy | 30.2 | 7.1 | 4.3 | 122 |
| orig `--no-video` | busy | 34.3 | 3.2 | 16.1 | 66 |
| serial `--no-video` | busy | 35.7 | 10.1 | 6.6 | 82 |
| thread `--no-video` | busy | 22.7 | 4.8 | 6.6 | 117 |
| process `--no-video` | clean | 21.5 | 3.2 | 5.7 | 114 |
| orig video | clean | 31.1 | 3.1 | 14.7 | 92 |
| **serial video** | clean | **28.7** | 7.1 | 4.2 | **133** |

Paired orig against serial: `--no-video` 22.6 -> 21.1 s clean (-7%), 34.3 -> 35.7 s busy;
video 31.1 -> 28.7 s clean (-8%), 33.6 -> 30.2 s busy. Suite F (video, unlocked GPU jobs not
checked) agreed: orig 32.6 s, serial 30.2 s, thread 36.0 s, process 39.5 s. Suite E was
discarded: two other agents' python processes were on the GPU throughout (orig `--no-video` 40.6 s).
Suite C (before the pinned reader) had orig/thread/process level within noise.

**Decision: serial** (`gridtracknet_prepass_background=False`). With the engine and the pinned
reader the prepass alone takes ~4 s; overlapped, it costs pass 1 more than that through GIL and
GPU time-slicing (pass 1 at 87-104 fps beside it, 122-147 fps alone). Serial ties process mode
on time and is the simplest, with no second CUDA context (+1.5-3 s init) and no IPC. Thread and
process modes stay as options. The win is small (~7%) because pass 1, not the prepass, is now
the critical path.

### Prepass alone (suite F, `prepass_bench.py`, 2 reps each)

Original prepass 10.6 / 10.0 s; new TRT prepass 11.7 / 7.5 s (first rep includes engine load and
cuDNN/TRT warm-up). Inside the pipeline (serial, init excluded) the new prepass measured
4.0-4.3 s on a clean GPU against the original's 12.5-15.6 s overlapped.

## Parity (Pomona, `--no-video` and video, original code against this branch)

- PyTorch runtime against original: same 1075 filled frames, 14 tracks, chosen track 1, same
  timeline; positions within 0.0029 px, confidence within 0.00024 (FP16 batch and stream noise; the
  resize itself is byte-exact).
- TensorRT default (thread, serial and process modes all identical) against original: same 1075
  filled frames, 14 tracks, chosen track 1, same timeline, `court_events` equal; positions within
  0.0085 px, confidence within 0.0019. One difference: non-chosen track 7 (track list index 11)
  gains two observations, frames 1614 and 1638, whose confidences are 0.40037 and 0.40043 against
  the 0.40 threshold: TRT FP16 noise at the boundary. Raw prepass detections 1118 -> 1120 for the
  same reason.

## Environment note

`pip install onnx` first pulled numpy 2.2.6, which breaks this cv2 build; numpy was put back to
1.26.4 within ~30 s (two of this item's baseline runs failed in that window; other agents' runs
then may have too). `onnx==1.23.2` with `ml_dtypes==0.5.4` is pinned in `system/requirements.txt`.

## Open issues

- Pass 1's first ~200 frames run at 18-34 fps in every variant (warm-up outside this item).
- Engine build costs ~22 s per new weights file (training runs that score many checkpoints pay
  it each time unless they set `ball_prepass_runtime="torch"`).
- Process mode's init overhead (+1.5-3 s, second CUDA context).
- Other agents ran GPU jobs outside the shared lock during several suites (E, G runs 3-9).
