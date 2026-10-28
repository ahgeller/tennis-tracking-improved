# Phase A follow-up: NVDEC prepass, pass-1 CUDA graphs, init

Speed only: no result changes. Base is `optimization` at afd9c8f. Clip is `system/sample/pomona.mp4`
(1920x1080, 2022 frames) unless stated. Python is the `tennis-opt` env (torch 2.10 cu128, TensorRT
10.15, numpy 1.26.4). Every timing below ran under `tools/gpu_lock.py`, with base and new interleaved.

## Results

### Parity

The tracking JSON was compared deeply against the base output, ignoring timing, paths and config
keys. The video was compared as bytes.

| clip (`--court`) | headline (filled, tracks, chosen, bounces, contacts) | exact differences | video |
|---|---|---|---|
| pomona, with video | equal (1075, 14, 1, [241,556,832,927,1019,1190,1288,1884,1967], 3) | 0 | byte-identical |
| pomona `--no-video` | equal | 0 | - |
| PomonaPitzer Women vs. UCSD `--no-video` | equal (3110, 12, 0, 17 bounces, 21) | 0 | - |
| Vereinsduell `--no-video` | equal (34537, 876, 0, ..., 299) | 0 | - |
| VolyAvw `--no-video` | equal | 0 | - |

I ran the same suite after each of the three commits; it was clean every time. The init commit's
own A/B runs (4 with video) were also each byte-identical to the base video.

### End to end

These are locked runs on pomona with `--court --info`: 4 rounds, base and new interleaved, video
and no-video. The table shows medians, and the per-run values are in the scratchpad logs.

The machine swings between a fast state and a slow state, about 1.6x apart, from run to run. The
GPU was idle and cool at the start of each run (nvidia-smi was logged). The swing hits both trees
alike, so the paired runs are the fair comparison.

| stage (s) | base `--no-video` | new `--no-video` | base video | new video |
|---|---|---|---|---|
| total | 21.30 | 16.50 | 25.00 | 20.95 |
| init (includes prepass) | 8.71 | 5.06 | 9.25 | 4.90 |
| prepass (as reported) | 5.10 | 3.85 | 5.45 | 3.40 |
| pass 1 total | 11.72 | 10.46 | 10.34 | 10.39 |
| pass 1 decode/slide | 2.36 | 1.10 | 1.98 | 1.08 |
| pass 1 preprocess | 6.88 | 7.42 | 6.39 | 7.54 |
| pass 1 aux wait | 2.58 | 2.69 | 2.47 | 2.70 |
| selector | 0.49 | 0.45 | 0.48 | 0.46 |
| pass 2 | - | - | 4.43 | 4.83 |

Paired totals:

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| no-video, base -> new | 25.2 -> 19.8 | 25.7 -> 19.9 | 17.4 -> 12.6 | 15.9 -> 13.2 |
| video, base -> new | 25.0 -> 16.5 | 25.0 -> 26.2 | 31.1 -> 25.3 | 19.6 -> 16.6 |

In video pair 2 the new run landed in the slow state and the base run in the fast one.

Two more locked suites of 3 rounds each, `timed_r1` and `timed_r2`, were run before the init
commit. Their medians:

- no-video: total 24.1 -> 15.5 and 25.6 -> 19.8;
- video: total 30.4 -> 26.7 and 31.4 -> 25.8.

The "prepass" figure in the final suite is the prepass's own wall time. Since the init commit,
boxmot's import runs alongside it and competes for the CPU, so the figure rose from 2.5 s to 3-4 s.
What counts is init, which fell. On its own, the prepass is 2.5 s against cv2's 3.7-4.1 s (see
below).

Pass 1's slide stage drops by half: 1.6 -> 0.85 s in the fast state and 3.1 -> 1.3 s in the slow
one. The preprocess and aux-wait timers each grow by about 0.4 s in matched pairs. Replayed graphs
return to Python at once, so I infer the GPU wait now lands at the next synchronising call. The
inference is that this time moves rather than being new work, since pass 1's total still falls
0.8-1.8 s in matched no-video pairs. Pass 1 stays CPU-bound: its GPU kernels total about 4.0 s
(profiler) against 8-15 s of wall time.

## 1. NVDEC decode for the ball prepass

**Built.**

- `NvdecDecoder(nv12=True)` (`video_decode.py`) hands out the decoder's NV12 pictures, copied out
  of its buffers, without conversion.
- `ball_gpu.Nv12ModelResize` turns each picture into the FP16 planar RGB model input with one CUDA
  kernel (`nv12_model_input`). The kernel converts and resizes in a single pass, and only at the
  resize taps.
- The kernel's arithmetic is the integer arithmetic of the two existing exact paths:
  - `nv12_to_bgr`: FFmpeg swscale's BT.601 fixed point. The coefficients come from
    `swscale_coefficients`, which is ported from `libswscale/yuv2rgb.c` (`ff_yuv2rgb_coeffs`, the
    `cy`/`oy`/`crv`/`cbu`/`cgu`/`cgv` setup), and the per-pixel steps follow
    `libswscale/x86/yuv_2_rgb.asm`.
  - `ModelResize`: OpenCV's `INTER_LINEAR` for 8-bit images, with `INTER_RESIZE_COEF_BITS = 11`
    and the rounding `(v + (1 << 21)) >> 22` (`modules/imgproc/src/resize.cpp`,
    `HResizeLinear`/`VResizeLinear` and `FixedPtCast`).
- `cuda_kernels.py` compiles the kernel with the NVRTC library that PyTorch ships
  (`nvrtc64_120_0.dll`), through ctypes, and loads it with the driver API (`cuModuleLoadData`,
  `cuModuleGetFunction`, `cuLaunchKernel`). The references are the
  [NVRTC docs](https://docs.nvidia.com/cuda/nvrtc/) and the
  [driver API module docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).
- The cubin is cached beside the TensorRT engines. Its key is the kernel name, the SM arch, the
  NVRTC version and a sha256 of the source.
- The cv2 pinned-memory reader stays as the fallback. It is used when
  `Config.ball_prepass_gpu_decode` is off, when NVDEC is unavailable, or when the kernel cannot be
  built or loaded; a fallback prints one `[detector]` line.

**Evidence.**

- Exactness:
  - The kernel output equals `ModelResize(nv12_to_bgr(...))` byte for byte on all 2022 pomona
    frames, and on random frames of 6 sizes x 2 ranges (`tests/test_ball_gpu.py`).
  - Prepass detections are identical to the cv2 path's.
- Locked prepass bench, warm and interleaved in one process:

  | path | runs (s) |
  |---|---|
  | cv2 | 4.13 / 3.69 / 3.79 |
  | NVDEC + fused kernel | 2.84 / 2.67 / 2.70 |

  Round 0 of the NVDEC path includes the 0.2-0.3 s NVRTC compile; a cached load takes 0.11 s. A
  launch costs about 34 us through ctypes.

**Rejected: NVDEC, then `nv12_to_bgr`, then `ModelResize`.** Locked, this ran 5.21 / 5.14 / 5.34 s
against cv2's 4.04 / 3.72 / 3.70 s.

- The cv2 prepass is CPU-bound, with the GPU about 60% busy.
- Moving the decode to the GPU makes the GPU the bottleneck. The profiled CUDA total went from
  3.18 s to 4.96 s, of which the int32 full-picture conversion (stack, cat and elementwise ops)
  was about 1.3 s, and the GPU power-throttled.
- The fused kernel touches only the resize taps, which is why it wins.

**Other rejections.**

| option | why rejected |
|---|---|
| NPP NV12->RGB | not swscale's rounding, so not bit-identical |
| CuPy RawKernel | needs numpy 2, and numpy must stay 1.26.4 |
| Triton | not installed |
| `torch.cuda._compile_kernel` | needs `CUDA_HOME` (a toolkit), which this env does not have |
| one decode shared by the prepass and pass 1 | holding the clip's frames costs about 12 GB |

## 2. CUDA graphs for pass 1

**Built.**

- `frame_window.GraphedFrameConversion` graphs a frame's conversion from HWC uint8 to CHW float and
  its motion planes. There are 4 slots, each with a static uint8 input, a captured graph and static
  outputs. The window holds 3 frames, so the slot a new frame takes was released one slide
  earlier.
- `PinnedFrameUploader.upload_into` copies the pinned frame straight into the slot's input.
- `motion.BackgroundModel` graphs the steady-state background update:
  - there is one graph per input address and stride, and the window's rotating slots give 4;
  - the previous motion mask is copied into one static buffer, which the graphs read;
  - the first frame, and any frame without a previous mask, run the original eager code.
- Capture uses `capture_error_mode="thread_local"`, as the
  [PyTorch CUDA graphs notes](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) and the
  `torch.cuda.graph` docs describe.
- A failed capture prints one `[pass 1]` line and runs that work eagerly.
- `Config.pass1_cuda_graphs` (on by default) is ignored while the prepass runs on a worker thread
  (`gridtracknet_prepass_background` without a process). Any CUDA work on another thread during a
  capture invalidates it; the docs list cross-thread work under capture as an error.

**Why it is bit-identical.** A graph replays the same kernels with the same launch parameters on
the same addresses (PyTorch docs: "replay ... the same kernels with the same arguments"). The tests
check this:

- `test_graphed_frames_are_the_eager_frames_bit_for_bit_while_in_the_window`;
- `test_the_graphed_background_update_is_the_eager_one_bit_for_bit`;
- `test_a_failed_capture_falls_back_to_eager_conversion`.

End to end, the JSON had 0 differences and the video was byte-identical.

**Evidence.**

- Locked microbench, per frame (conversion, planes and background update):
  - eager: 0.615 ms of CPU, 0.705 ms including the GPU;
  - graphed: 0.007 ms of CPU, 0.586 ms including the GPU.
- Capturing the 4 slots costs about 0.03 s once: 6 frames took 0.036 s graphed, against 0.010 s
  eager when warm.
- End to end, decode/slide fell from 1.6 to 0.85 ms per frame (fast state), as in the table above.

**Rejected: graphing the ROI work.** ROI crops change shape and position every frame, so a graph
per shape would almost never be replayed. Bucketing shapes to a fixed set would mean re-creating the
crop's edge-padding semantics at bucket sizes, which risks changing results for a CPU saving of a
few launches per ROI. Pass 1's GPU total is about 4 s against 8-15 s of wall time; what is left is
Python and launch overhead in the ROI, detector and tracker code, not anything graphable at a fixed
shape.

**Not done: graphing the player preprocess together with its engine.** The engine launch is already
replayed from a TensorRT CUDA graph (`trt_cuda_graph`). Only about 5 small preprocess launches per
detected frame remain, and that is every Nth frame, not every frame. This would need graphs keyed by
input address, and the expected saving is under 0.1 s per clip.

## 3. Init

**Measured breakdown.** Locked, with video, before the change, by wall-clock wrappers around each
step of `pipeline._open_inputs` (`profile_init.py`). Total `_open_inputs` was 6.35 s, after 2.1 s
of imports.

| step | time (s) |
|---|---|
| encoder probe (ffmpeg `-encoders` plus the NVENC dry run) | 0.42 |
| ball model and TRT engine | 0.44 |
| court detector | 0.20 |
| player detector | 2.35 |
| prepass reader open | 0.37 |
| prepass | 2.88 |

Almost all of the player detector's 2.35 s was spent waiting for boxmot's import
(`prefetch_bytetrack` starts it on a thread at the top of init). `python -X importtime` puts boxmot
at 3.1 s, of which 2.2 s is `boxmot.reid`: its CLIP backbone imports torchvision, which pulls in
`torch._dynamo` (1.2 s).

**Built.**

- A prepass on the main thread (the default) now runs before the court and player detectors are
  built, so the import finishes behind it; after the change the player detector took 0.14 s.
- A prepass on a worker thread still starts after them. This keeps the player engine's CUDA graph
  capture free of other threads' CUDA work, as before.
- The encoder probe runs on a thread (`_EncoderProbe`) while the detectors are built. The platform
  report prints once the choice is in, and a probe exception is re-raised on the main thread.
- Tests: `test_a_foreground_prepass_runs_before_the_engines_and_a_background_one_after` and
  `test_a_video_run_probes_the_encoder_alongside_init_and_raises_what_the_probe_raised`.

**Evidence.**

- Reorder alone (`--no-video`, 3 interleaved pairs, locked): init 6.48 / 6.32 / 6.27 -> 5.44 /
  5.38 / 5.24 s; total 19.6 / 19.3 / 19.2 -> 18.7 / 18.5 / 18.2 s.
- The prepass itself slows from 2.5 to 4.2 s, because the import competes with it for the CPU and
  GIL. It is still a net 1.0 s saved.
- The encoder thread on top of the reorder (video, 4 interleaved pairs, locked): init 5.67 / 5.83 /
  5.70 / 6.25 -> 5.62 / 5.39 / 5.61 / 5.41 s, a median of -0.26 s.
- The JSON was identical and the video byte-identical in every run.

**Not done.**

| idea | why not |
|---|---|
| Avoid boxmot's ReID import (import `boxmot.trackers.bytetrack` directly) | `boxmot/__init__` runs on any submodule import, so it would need a change to a third-party package or `sys.modules` stubbing. Too fragile for about 2 s that now mostly overlaps the prepass. |
| Overlap the 2.1 s import of torch and cv2 | it is on the critical path of everything after it |

## Files

- New:
  - `system/tennis_tracker/cuda_kernels.py`
  - `system/tests/test_cuda_kernels.py`
  - this record
- Changed:
  - `system/tennis_tracker/engines/ball_gpu.py`
  - `system/tennis_tracker/engines/ball_detectors.py`
  - `system/tennis_tracker/video_decode.py`
  - `system/tennis_tracker/frame_window.py`
  - `system/tennis_tracker/motion.py`
  - `system/tennis_tracker/pass1.py`
  - `system/tennis_tracker/video_io.py`
  - `system/tennis_tracker/pipeline.py`
  - `system/tennis_tracker/config.py` (`ball_prepass_gpu_decode`, `pass1_cuda_graphs`)
  - `system/mypy.ini`
  - `system/AGENTS.md`
  - tests: `test_ball_gpu.py`, `test_ball_detectors.py`, `test_video_decode.py`,
    `test_frame_window.py`, `test_pipeline_run.py`

## Open issues

- The machine's run-to-run swing between two speed states (about 1.6x) dominates single-run
  timings. The swing affects both trees alike, so compare paired runs.
- The NVRTC path depends on the DLL name PyTorch ships (`nvrtc64_120_0.dll` on Windows,
  `libnvrtc.so.12` on Linux). A torch built against another CUDA major version would make the
  kernel unavailable, and the prepass would fall back to cv2 with a printed note rather than fail.
- The cubin cache sits in `ball_engine_dir`. A read-only engine directory makes every run compile
  (0.2-0.3 s) rather than fail.
