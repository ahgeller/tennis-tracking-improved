# AB+: pass 1 speed on top of A+B (2026-10-30)

Branch `phase-ABplus` off `phase-AB` at ae270b3 (the AB baseline below). Speed only: the shipped
changes leave every output identical. Clip is `system/sample/pomona.mp4` (1920x1080, 2022 frames)
unless stated. Python is the `tennis-opt` env. Every timing and scoring run went through
`tools/gpu_lock.py`; before each timed run `nvidia-smi --query-compute-apps` was logged, and a run
with another python process on the GPU would have been discarded (none was). The laptop's RTX 5080
swings about 1.6x between a fast and a slow state, so every comparison is interleaved (a warm-up
round, then the order reversed each round) and reported as median [min-max].

Two models throughout: the shipped `heatmap_s4` (`models/ball_detector.pt`, conf 0.40) and the
candidate `stream_s4` (`models/candidates/stream_s4_e60.pt`, conf 0.42).

## Results

### Shipped

| commit | change | pass 1, shipped | pass 1, stream_s4 | outputs |
|---|---|---|---|---|
| 1b6b58f | motion masks with two NVRTC kernels | 11.83 [11.81-11.88] -> 10.21 [10.10-14.75] s | 12.74 [12.38-13.05] -> 10.94 [10.79-10.94] s | identical |
| 8cbca6b | frames decoded into pinned memory | 10.89 [10.79-11.45] -> 9.23 [9.12-9.70] s | 11.51 [10.54-13.94] -> 10.68 [8.75-11.48] s | identical |
| fc0e3f3, 68c01bb | review fixes (section 8) | - | - | identical |

The two shipped rows come from different suites, each interleaved against its own base: the
kernels row from section 1's, the pinned-decode row from section 8's (where the laptop ran in a
slower state, so its absolute seconds are higher).

### End to end, AB (ae270b3) vs AB+ (HEAD)

`clean_tracker.py --input sample/pomona.mp4 --info` from `system/`, the eight variants interleaved
in one lock, a warm-up round then 2 timed rounds, no GPU neighbour in any run. `[done]` total and
the `[info]` rows, median [min-max], seconds:

| | AB shipped | AB+ shipped | AB stream_s4 | AB+ stream_s4 |
|---|---|---|---|---|
| `--no-video` total | 18.30 [18.10-18.30] | **15.20** [14.70-15.20] | 20.00 [19.50-20.00] | **15.80** [15.70-15.80] |
| - init (incl. prepass) | 5.34 | 5.46 | 5.93 | 5.93 |
| - prepass | 4.10 | 4.20 | 4.50 | 4.50 |
| - pass 1 | 12.25 [12.04-12.25] | 8.96 [8.56-8.96] | 13.40 [13.10-13.40] | 9.22 [9.03-9.22] |
| - selector | 0.56 | 0.71 | 0.58 | 0.59 |
| video total | 23.70 [23.50-23.70] | **20.90** [20.70-20.90] | 25.40 [25.30-25.40] | **21.90** [21.70-21.90] |
| - pass 1 | 12.52 | 9.64 | 13.45 | 10.23 |
| - pass 2 | 4.94 | 5.02 | 5.14 | 4.76 |
| process wall (`--no-video`) | 21.72 | 18.96 | 23.62 | 19.29 |

Pass 1 is 23-31% faster; end to end 12-21%. Every run's tracking JSON equals its AB pair frame for
frame, and the last video runs' `tracking.mp4` are byte-identical.

### Parity

- Pass 1's outputs (detections, boost and raw masks bit for bit, ROIs, ghost ROIs, player boxes,
  court keypoints, ROI and full-frame counts) were dumped on pomona after each change and compared
  with AB's: 0 of 2022 frames differ, both models.
- The tracking JSON of every timed pomona run equals AB's frame for frame (both models).
- Accuracy gate (`evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1`):
  per-frame outputs identical to AB on every clip, for both models, after each commit.

### Accuracy gate (AB+ HEAD; AB's numbers are the same, every clip frame for frame)

| model | recall | wrong | false alarm (bridged / false) | median / p90 px | landed line / other / player |
|---|---|---|---|---|---|
| shipped @0.40 | 0.946 | 0.006 | 98/518 (86 / 12) | 1.0 / 4.0 | 39 / 60 / 47 |
| stream_s4 @0.42 | 0.947 | 0.004 | 104/518 (86 / 18) | 1.0 / 4.0 | 32 / 59 / 50 |

| source | shipped recall | shipped FA | stream_s4 recall | stream_s4 FA |
|---|---|---|---|---|
| custom | 0.964 | 42/207 | 0.969 | 46/207 |
| grid | 0.891 | 17/149 | 0.906 | 17/149 |
| prof | 0.953 | 39/162 | 0.942 | 41/162 |

### Off our court (`system/sample/`, `--no-video`)

A present output is off our court when its x lies outside the court's sidelines, extended to the
ball's image height, by more than a quarter of the court's width there (the frame's keypoints, else
the last valid ones). AB and AB+ give the same outputs frame for frame, so the counts are equal.

| clip | shipped present / off | stream_s4 present / off |
|---|---|---|
| VolyAvw | 2590 / 550 | 2509 / 582 |
| pomona | 1075 / 74 | 1090 / 89 |
| PomonaPitzer Women vs. UCSD | 3110 / 322 | 3142 / 324 |

The off-court outputs are almost all `det` sources on the neighbouring courts: the selector's
business (out of scope), reported only.

## 1. Motion on demand

**Measured first.** Wall-clock wrappers around pass 1's steps (CPU time, no syncs added), AB,
shipped model, pomona, locked, ms per frame:

| step | ms/frame |
|---|---|
| `_raw_motion_cuda` (foreground, probe deltas, blurs, temporal gate) | 1.22 |
| `_queue_masks` (closing per box, remembered mask, D2H start) | 0.63 |
| `MotionMasks._resolve` at commit (of which `filter_boost_mask`, the CC) | 0.50 (0.39) |
| `MotionMasks._pack` | 0.17 |
| `BackgroundModel.update` (CUDA graph) | 0.05 |

The `[info]` row `pass1 pre(cuda total)` was 2.36-2.39 ms/frame. The mask build is about forty
small PyTorch ops a box, so the 1.85 ms is launch overhead from Python, not GPU time.

Which parts are consumed where: replaying the selector on pomona's pass-1 dump with recording
lists, it reads the boost mask at 1165 of 2022 frames and the raw mask at 29.

**What cannot be made on demand.** The background model's next update reads this frame's cleaned
gated mask inside the ROIs (the slow-adaptation mask). So the foreground, the temporal gate and the
closing must run every frame, or the model and every later mask change. Only their cost can go.

**Built: two NVRTC kernels that give PyTorch's bits** (`tennis_tracker/motion_gpu.py`):

- `motion_gate`: one thread per pixel of a box computes the foreground (V and S against the
  background mean and variance), the two probe deltas, their 3x3 blurs and the temporal gate.
- `motion_morph`: one dilation or erosion of a box; the default closing is two launches. The last
  pass also sets the pixel in the background's slow-adaptation buffer
  (`BackgroundModel.motion_target`, the buffer the update's CUDA graphs read, so no copy).
- Every float step rounds as PyTorch's kernels do: `__fsub_rn`, `__fmul_rn`, `__fadd_rn` and
  `__fdiv_rn` are never contracted into FMAs (NVRTC contracts by default), a blur sums its taps in
  `avg_pool2d`'s order and divides by nine, and each scalar is the float32 PyTorch casts it to.
- `Config.motion_kernels` (on) chooses them; when NVRTC cannot build them pass 1 prints one line
  and the PyTorch ops run. Those ops stay the reference.
- Tests (`tests/test_motion_gpu.py`, CUDA only): the masks, packed masks, blobs and remembered mask
  equal the PyTorch path for every ROI layout (none, one, overlapping, frame edges), gate setting
  (on, off, no next frame) and cleaning (none, closing, closing plus opening); the background's mean
  and variance stay equal over ten frames with and without its CUDA graphs; and two tests put
  pixels exactly on the foreground and temporal thresholds (and one float32 step either side). An
  FMA in the threshold, a reversed blur order or a dropped channel each fail them (checked by
  mutation). A third test (added in review) picks pixels whose 3x3 sum rounds differently divided
  by nine and multiplied by a ninth and puts the gate's threshold there; replacing the division
  with that multiply fails it. (PyTorch divides a CUDA tensor by a scalar as a multiply by the
  reciprocal, so that test computes its reference on the host; `avg_pool2d` divides.)

**Evidence.** Interleaved, locked, `clean_tracker.py --no-video --info`, 3 timed rounds each:

| | AB shipped | kernels shipped | AB stream_s4 | kernels stream_s4 |
|---|---|---|---|---|
| total (s) | 17.80 [17.80-18.10] | 17.20 [16.20-20.80] | 19.00 [19.00-20.10] | 17.30 [17.20-17.60] |
| pass 1 total | 11.83 [11.81-11.88] | 10.21 [10.10-14.75] | 12.74 [12.38-13.05] | 10.94 [10.79-10.94] |
| pass 1 pre(cuda total) | 4.78 | 2.43 | 5.23 | 2.84 |
| pass 1 aux wait | 2.80 | 3.69 | 3.00 | 3.66 |

The one slow-state run (kernels shipped, round 2: 14.75 s, aux wait 8.1 s) hit the laptop's slow
state; the other two rounds give 10.10 and 10.21 s. Aux wait rises because pass 1 now reaches the
player engine's results sooner: the GPU work it used to hide behind launch overhead is now exposed.

**Not built: CC and packing on demand.** The boost-mask CC on the 733 frames the selector never
reads cost 348 ms of the 424 ms total (shipped, pomona); packing about 0.17 ms a frame. Deferring
them means handing the selector a lazy sequence of masks, and `erase_ghost_rois` clears ghost boxes
from the boost mask after its CC (clearing the raw mask first would change which blobs pass the
area test), so the laziness would have to replay that order. About 0.35 s of 15 s, for a change to
the selector's input type: left as an open item.

**Timer nesting** (verified in code). GridTrackNet's prepass detector is not asynchronous, so
`can_overlap_ball_inference` is False and the previous frame's `_finish_pending` -> `_commit` ->
`_finish_players`, the mask resolve and the store run inside `pass1_preprocess`. `pass1 preprocess`
therefore includes `aux wait` and `store/debug`; the rows are not additive.

## 2. Decode once

Pass 1 decodes with OpenCV on a reader thread; the prepass, serial and first by default, decodes
the clip again with NVDEC. Sharing one decode needs both consumers alive at once. With the serial
prepass (the default since Phase A) that means holding the clip's frames, about 12 GB on pomona
(`opt_phaseA_followup.md`). With an overlapped prepass it is `video_decode.SharedFrameSource`, but
overlap itself is what item 3 measures, and it loses for the shipped model. **Not built.**

**Built instead: decode into pinned memory** (8cbca6b). The reader thread decodes each frame
straight into a fresh pinned buffer from PyTorch's caching host allocator (OpenCV writes into a
buffer of the right shape and type), and `PinnedFrameUploader` copies from that buffer to the device
instead of first `np.copyto` into its own pinned slot on pass 1's thread. The allocator keeps a
buffer from reuse until its recorded copy is done. The pass 2 frame cache copies pinned frames out,
so a cached clip does not pin gigabytes. Interleaved against the kernels commit (section 8's suite,
3 timed rounds, median [min-max] s):

| | kernels shipped | + pinned shipped | kernels stream_s4 | + pinned stream_s4 |
|---|---|---|---|---|
| total | 17.60 [17.30-18.00] | 15.50 [15.30-16.80] | 18.30 [17.40-21.40] | 17.70 [15.10-18.80] |
| pass 1 | 10.89 [10.79-11.45] | 9.23 [9.12-9.70] | 11.51 [10.54-13.94] | 10.68 [8.75-11.48] |
| pass 1 decode/slide | 1.34 [1.33-1.38] | 0.68 [0.65-0.70] | 1.32 [1.22-1.53] | 0.75 [0.59-0.81] |

The slide halves for both models. The rest of the shipped model's gain is the reader thread no
longer allocating a 6 MB frame per read. The stream_s4 runs spread widely and overlap, so its pass 1
gain (0.8 s at the median) is less certain than the shipped model's (1.7 s, separated). Tests:
`test_frames_decoded_into_pinned_memory_upload_from_it_and_leave_the_cache_pageable` and
`test_frames_of_another_size_end_the_pinned_reads_with_one_warning`.

## 3. Child-process prepass, re-evaluated

`gridtracknet_prepass_background` and `gridtracknet_prepass_process` on, against the serial
default, on the AB+ tree. Interleaved, locked, 3 timed rounds:

| | serial shipped | process shipped | serial stream_s4 | process stream_s4 |
|---|---|---|---|---|
| total (s) | 14.90 [14.50-15.10] | 15.50 [14.90-15.50] | 15.90 [15.20-15.90] | 15.00 [14.20-15.10] |
| init | 5.55 | 3.70 | 5.91 | 3.61 |
| prepass (reported) | 4.20 | 4.20 | 4.40 | 2.30 |
| pass 1 total | 8.60 | 10.93 | 8.84 | 10.67 |

Outputs are identical either way. With the shipped windowed network the prepass still takes 4.2 s
alongside pass 1 and slows pass 1 by 2.3 s, a net loss of 0.6 s; with the streamed stream_s4 it
gains 0.9 s. **Rejected as the default** (the shipped model loses). If stream_s4 ships, turning the
process on is worth re-measuring then.

## 4. Pass 1 host overhead

After items 1 and 2 (stream_s4, locked profile, ms per frame): `_commit` 2.39, of which player
`detect_async_finish` 1.36 (ByteTrack 0.67, box decode 0.67); `MotionMasks._resolve` 0.70 (CC
0.56); `_pack` 0.25; `detect_async_start` 0.51; `_to_gpu` 0.29.

**Rejected, measured:**

| change | result |
|---|---|
| Player head thresholded on the device before the download | The player engine's head is `(1, 5, 8400)` FP16, one class: 84 KB, so there was nothing to save. Box decode 0.80 -> 1.08 ms/frame in the profile; reverted. |
| Player outputs copied to pinned memory on the engine's stream, waited by event (so the decode does not wait behind pass 1's queued work on the current stream) | aux wait 1.50-1.59 -> 1.53-1.57 ms/frame, box decode 0.80 -> 0.68: within noise. Reverted. On the saved outputs the decode alone is 0.18 ms; the rest in the pipeline is contention with the reader thread, not the GPU. |
| `sys.setswitchinterval` 1 ms or 0.5 ms (default 5 ms), for the GIL handoffs between pass 1, the reader and the prepass | total 15.40 [14.20-15.40] default, 14.60 [14.10-14.60] at 1 ms, 16.50 [15.80-16.50] at 0.5 ms (2 rounds): no consistent effect. Not built. |

ByteTrack (boxmot) is third-party; not touched.

## 5. Player scheduling

`aux_force_interval` only matters when the ball detector skips frames (`aux_detect_on_detect_frames`
with `skip_frame_detect` > 1); the default detects every frame, so it never applies. Not changed.

`player_detect_interval`, now that pass 1 is the limit, timed and gated (shipped model, 2
interleaved rounds):

| N | total (s) | pass 1 (s) |
|---|---|---|
| 1 | 15.40 [14.00-15.40] | 9.05 [7.96-9.05] |
| 2 | 12.70 [12.70-12.70] | 6.68 [6.52-6.68] |
| 3 | 12.20 [12.00-12.20] | 5.97 [5.79-5.97] |

Gate, against N=1 (AB+):

| | recall | false alarm | real false | landed player | clips that differ |
|---|---|---|---|---|---|
| shipped N=1 | 0.946 | 98/518 | 12 | 47 | - |
| shipped N=2 | 0.946 | 99/518 | 12 | 49 | grid_match8 (6 frames), video23 (4) |
| shipped N=3 | 0.946 | 100/518 | 12 | 44 | grid_match8 (17), video23 (27) |
| stream_s4 N=1 | 0.947 | 104/518 | 18 | 50 | - |
| stream_s4 N=2 | 0.947 | 105/518 | 18 | 49 | grid_match8 (6) |
| stream_s4 N=3 | 0.947 | 105/518 | 18 | 48 | grid_match8 (17), video23 (13) |

Every N > 1 raises the false alarms (by 1 or 2, all bridged), and shipped N=2 also adds two player
landings. **Rejected** under the gate; N stays 1. The time is real (2.7 s at N=2), so a player
schedule that keeps the outputs (for example tracking only while the ball is near a player) is an
open item.

## 6. Init

Init is 5.3-5.9 s, of which the serial prepass is 4.1-4.4 s; the rest is the engines (about 1 s).
Phase A measured the prepass alone at 2.5 s: it runs while boxmot's import (about 3 s, mostly
`boxmot.reid` pulling in torchvision and `torch._dynamo`) competes for the CPU and the GIL on its
thread. Avoiding that import needs a change to the third-party package or `sys.modules` stubs
(rejected in Phase A as fragile), and running the prepass beside pass 1 instead loses for the
shipped model (item 3). Nothing practical was left to build here.

## 7. Ready for C

Phase C's gate (`system/tennis_tracker/roi_gate.py` on `phase-C`, `audit/records/opt_roi.md`
there) decides per five-frame unit whether the network sees the full frame, a 256x192 LOCK crop,
a wide crop or nothing, from the ball's own track in the detector's finalised frames. Where it
plugs into AB+:

- **The prepass, not pass 1.** The gate needs only the detector's own results, so it sits in
  `GridTrackNetBallDetector._prepass` (`engines/ball_detectors.py`), between assembling a unit
  and batching it. Pass 1 consumes the prepass's per-frame rows as before; nothing in AB+'s pass 1
  changes. The motion kernels and pinned decode are independent of it.
- **Lag and batching.** The gate's `plan` expects every earlier frame observed; with
  `ball_engine_batch` 4 the prepass must flush before planning, or plan with a lag of N units, as
  the Phase C record describes. AB+ did not change the batching or the flush.
- **Windowed vs streamed.** The gate was designed against the windowed `heatmap_s4`, where a crop
  shrinks the whole network's input. The streamed `stream_s4` encodes each frame's transition once
  at full frame (`ball_gpu.TransitionRing`) and decodes units from the cached features: a crop
  there saves only the decoder (slice the cached feature maps at the crop) unless the encoder also
  runs per crop, which breaks the cache whenever the crop moves. Phase C should measure the decoder
  share of a streamed unit before porting the gate to it.
- **Kept compatible.** The prepass's `_UnitStream`, `TransitionRing`, the engine cache layout and
  the serial-first ordering are unchanged from AB. On `phase-C` the gate is a standalone module
  (`roi_gate.py` is the branch's only change under `system/tennis_tracker/`, off 93771f8), not
  wired into any prepass, so the wiring is new work against this prepass; engines per crop shape
  will need their own cache keys in `ball_trt`.
- **Not built** here, by the brief.

## 8. Review fixes (fc0e3f3, 68c01bb)

Two reviewers read ae270b3..4b0a77b. They found no serious bug: bit parity held on real pomona
frames at odd sizes, and the pinned buffers had neither a race nor a leak. Fixed:

- **Box bounds.** The gate kernel indexes the frame without checks, where PyTorch's slices clamp.
  ROIs are clipped to the size the container reports, so a clip that decodes at another size could
  have made the kernel write past the frame. `MotionKernels.gate` now raises `ValueError` for a
  box outside the frame, and `morph`'s shape check already refused a clamped crop. Test:
  `test_a_box_outside_the_frame_is_refused_before_any_launch`.
- **Fallback.** `load_kernel` turns a CUDA runtime error raised under it (a broken driver or
  device) into `KernelUnavailable`, so pass 1 falls back to the PyTorch ops; other exception types
  stay bugs and propagate. Only loading falls back: a launch that fails mid-clip raises (commented
  in `pass1._load_motion_kernels`). Test:
  `test_pass1_falls_back_to_pytorch_ops_when_the_kernels_cannot_load`.
- **Pinned size mismatch.** When `pinned_shape` did not match the decoded frames, the reader
  allocated a pinned buffer per frame that was never used. Now the first read that does not land
  in the buffer ends the pinned reads, with one warning when the size differs.
  `FrameCapture.read` takes the optional `image`; `OpenCVDecoder` passes it through and
  `NvdecDecoder` ignores it (it has its own buffers).
- **One code path.** `_raw_motion_cuda` and `_queue_masks` serve both paths. The shared preamble and
  tail are written once, and only the per-box work branches. `BackgroundModel._motion_buffer_for`
  makes the graphs' buffer. The probe's saturation scale (`255 * _PROBE_S_DIFF_SCALE`) is passed
  in through `GateScalars` instead of hard-coded.
- **Not done.** `video_io.py` was not added to `mypy.ini`: its writer has 5 existing type errors,
  none of them in the new code.

Verification after the fixes:

- Pass 1's pomona dump is identical to AB's.
- Suites: system 596 passed, 1 skipped; root 1323 passed, 1 skipped; mypy clean (18 and 5 files).
- Gate: per-frame outputs identical to AB on every clip, both models. Shipped @0.40: recall 0.946, FA 98/518 (86 bridged, 12 false), landed line/other/player 39/60/47. stream_s4 @0.42: recall 0.947, FA 104/518 (86, 18), landed 32/59/50.
- Timings: the suite in the table above. AB, the kernels commit and HEAD were interleaved, 3 timed
  rounds, median [min-max] s:

| | AB | kernels | HEAD |
|---|---|---|---|
| shipped total | 20.10 [19.40-21.30] | 17.60 [17.30-18.00] | 15.50 [15.30-16.80] |
| shipped pass 1 | 13.61 [12.99-14.88] | 10.89 [10.79-11.45] | 9.23 [9.12-9.70] |
| stream_s4 total | 22.10 [20.90-24.10] | 18.30 [17.40-21.40] | 17.70 [15.10-18.80] |
| stream_s4 pass 1 | 14.97 [13.15-15.94] | 11.51 [10.54-13.94] | 10.68 [8.75-11.48] |

This suite ran in a slower machine state than the final table at the top (AB shipped 20.1 s
against 18.3 s there). The ratios agree: pass 1 is 32% faster for the shipped model and 29% for
stream_s4.

## Files

- New: this record; `system/tennis_tracker/motion_gpu.py`; `system/tests/test_motion_gpu.py`.
- Changed: `system/tennis_tracker/motion.py` (kernel path, `BackgroundModel.motion_target`),
  `pass1.py` (kernel loading, pinned reader, cache copy), `config.py` (`motion_kernels`),
  `video_io.py` (`pinned_source`, the reader's pinned decode, the uploader), `video_decode.py`
  (`read(image)`), `cuda_kernels.py` (load failures), `mypy.ini`, `AGENTS.md`,
  `tests/test_frame_window.py`.

## Open issues

- CC and packing on demand (item 1, about 0.35 s on pomona).
- A player schedule that keeps outputs identical (item 5, up to 2.7 s).
- The child-process prepass for stream_s4 (item 3, 0.9 s) if stream_s4 ships.
- In-pipeline host costs are inflated about 3x over their isolated cost (player box decode 0.18 ms
  alone, 0.67 ms in pass 1) by contention with the reader thread; unexplained in detail.
