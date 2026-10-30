# Phase A + B: the prepass streams stream_s4

The Phase A prepass now runs the Phase B `stream_s4` detector in true streaming mode. Each source
frame's transition is encoded once and its features are cached in a small ring on the GPU. Every
unit the windowed prepass would run is decoded from its four cached transitions. The halves are two
TensorRT FP16 engines, cached the way the unit engine is.

Base is `optimization` at c9ec36b, on branch `phase-AB`. Weights are
`system/models/candidates/stream_s4_e60.pt` (gitignored). Clip is `system/sample/pomona.mp4`
(1920x1080, 2022 frames, 59.67 FPS, so 2 phases) unless stated. Python is the `tennis-opt` env:
torch 2.10.0+cu128, TensorRT 10.15.1.29, numpy 1.26.4, RTX 5080 Laptop GPU. Timings and scoring
ran under `tools/gpu_lock.py`. The shipped default weights (`models/ball_detector.pt`, heatmap_s4)
are unchanged; `--weights` selects the new model.

## Design

- **The schedule is the windowed one.** `_UnitStream` still cuts the units: window step, phases,
  the clip's first units and its trailing windows are all unchanged. A unit is always the last five
  frames its phase has seen, so its transitions are the phase's last four, at any window step. With
  a streaming network, the value a frame carries through `_UnitStream` is its transition number in
  the ring, not its tensor (None for a phase's first frame, which has no transition). Batching,
  `HeatRing`, `grid_peaks` and the pinned `Readback` of four floats a frame are untouched.
- **`ball_gpu.TransitionRing`.** `push(frame, phase)` copies the frame and the phase's previous
  frame into the encoder's static-batch input buffers. When `batch` transitions are waiting, it
  encodes them and copies the features into consecutive ring slots, wrapping at the end; there is
  one ring per stride (4, 8, 16). `decode(units)` first encodes any partial batch. It then uploads
  the units' slot indices from pinned memory (no stream wait) and gathers each stride with one
  `index_select` into the decoder's preallocated input. Last, it runs the decoder; a short call is
  padded with the last unit, as a static engine batch is. The prepass adds no per-frame host sync or
  Python loop, and every buffer is allocated once per clip.
- **Ring size.** `transition_slots(batch, step, stride)` is 15 slots at the defaults (batch 4,
  step 1, 2 phases), about 35 MB of FP16 features. `TransitionRing.decode` raises if a unit's
  transition was overwritten, so a ring that is too small fails loudly instead of answering wrong. I
  measured the smallest ring that still passes, over steps 1-5, 1-3 phases and batches 1, 2, 4, 7
  and 8: the formula always leaves at least 2 spare slots (exactly 2 at stride 1, 3-13 at stride 2, 4-8 at stride 3).
  Script: `slot_margin.py` in the scratchpad.
- **Two engines** (`ball_trt.ensure_stream_engines`, `open_stream_engines`). `StreamNet.streaming_halves`
  gives `StreamEncoder` (`encode_frame` as a module) and `StreamDecoder` (`_decode` on the four
  transitions concatenated per stride). The halves are exported from the deployed form. Their
  weights are folded in float32, then cast to FP16 once, because TensorRT cannot merge RepVGG's
  parallel branches (`opt_detector_archs.md`, decision 5). Each half is cached as its own file. Its
  key adds `part=encoder|decoder` to the existing key: weights digest, arch, batch, input size,
  TensorRT, torch, GPU name and compute capability, and recipe. The whole-model key has no part
  line, so it is the key it was before (a test recomputes it from the documented fields). Proof
  in practice: the windowed engine this tree built, `stream_s4_e60-stream_s4-b4-56c786801e724068`,
  has exactly the name the base tree built. Both halves run on one stream, which the prepass works
  on, at addresses bound per call; the engine sizes are 0.74 MB and 0.88 MB.
- **Selection and fallback.** The detector streams any model that has `streaming_halves`, unless
  `Config.ball_prepass_streaming` (default on) is off. If TensorRT cannot provide the engines, it
  prints `TensorRT engines unavailable, streaming the PyTorch model (...)` and streams the PyTorch
  halves. That mirrors the existing unit-engine fallback. `ball_prepass_runtime="torch"` and a
  non-CUDA device also take the PyTorch halves. Every other architecture takes its old path: the
  same unit engine, the same key and the same code. The only edit on that path builds the
  static-batch padding from the same last unit's tensors, which mypy needed.
- **The child process path** (`ball_prepass_process`) builds the same detector, so it streams too;
  I did not time it separately.

## 1. Parity on Pomona

Script: `parity.py` in the scratchpad. It ran 4 prepasses at confidence 0, so every covered frame
reports its peak, and recorded every unit's 15-channel output as the prepass handed it to the heat
ring (2014 units):

- W: windowed, TensorRT unit engine. This is the path before this change; it exports the
  trainable form.
- S: streamed, TensorRT encoder + decoder engines.
- T: streamed, PyTorch FP16 halves (the fallback).
- R: windowed, PyTorch FP32, trainable form (the reference).

Per unit, max absolute difference over all 15 channels and 192x108 cells:

| pair | max heat | p99 of per-unit max heat | median per-unit max heat | max offset (cell units) |
|---|---:|---:|---:|---:|
| W vs S | 0.0109 | 0.0056 | 0.0018 | 0.063 |
| W vs R | 0.0090 | 0.0049 | 0.0015 | 0.055 |
| **S vs R** | **0.0051** | 0.0037 | 0.0015 | 0.054 |
| T vs R | 0.0037 | 0.0024 | 0.0010 | 0.025 |
| T vs S | 0.0034 | 0.0027 | 0.0010 | 0.051 |

The streamed engines are closer to the FP32 reference than the windowed engine is (0.0051 against
0.0090). So W vs S is two FP16 roundings, not a streaming error. Folding before export plausibly
helps: TensorRT runs one convolution instead of three branches summed in FP16. A cell offset of
0.063 is 0.25 model px, or 0.63 source px. The schedule itself is exact: the CPU test below
matches the windowed prepass to float32 rounding.

Per frame, after the heat ring's averaging and the device peak:

| pair | presence differs | detections at 0.3 / 0.4 / 0.5 | frames that differ at 0.3 / 0.4 / 0.5 | same frame, > 4 px apart, above 0.3 | max confidence difference |
|---|---:|---|---|---:|---:|
| W vs S | 0 | 1250/1249, 1130/1130, 1049/1049 | 3, 0, 0 | 0 | 0.0039 |
| W vs R | 0 | 1250/1249, 1130/1129, 1049/1049 | 3, 1, 0 | 0 | 0.0029 |
| S vs R | 0 | 1249/1249, 1130/1129, 1049/1049 | 0, 1, 0 | 0 | 0.0027 |

At the tracker's 0.4 threshold and at 0.5, streamed and windowed detections are identical on every
frame. At 0.3, 3 frames sit within 0.004 of the threshold and flip. The windowed engine flips just
as many against the FP32 reference. Five frames move their peak by more than 4 px, all below 0.3:
the argmax jumps between near-equal background cells.

### Unit tests (no GPU)

- `test_ball_detectors.py::test_the_streamed_prepass_decodes_every_unit_the_windowed_one_runs`
  covers 60 cases: steps 1-5, 1-3 phases, batches 1, 2, 4 and 7, with clips long enough to wrap the
  ring more than twice. A stand-in decoder checks that each unit's four transitions are consecutive
  frames of one phase. Every frame's answer equals the windowed prepass on the same static batch,
  from the first units to the trailing windows. At step 1, a unit ends at every frame from each
  phase's fifth on.
- `test_a_streaming_stream_net_answers_as_its_windowed_prepass` runs a real StreamNet on the CPU.
  It has scrambled BN statistics and a grown head. The streamed side is deployed (folded); the
  windowed side is the trainable form. It covers 7 schedules, including short clips: one where a
  phase gets a single unit, one where no phase reaches five frames. Points agree to 1e-3 px and
  confidences to 1e-5, and frames that no unit covers stay empty.
- `test_ball_gpu.py`: the ring encodes each transition once, encodes full batches as they fill and
  partial ones at decode, and wraps (8 slots, 12 transitions, the span exactly at the boundary). It
  gathers each unit's four transitions in order, pads to the batch, and refuses an overwritten
  transition, more than `batch` units, or fewer than 4 slots.
- `test_ball_trt.py`: each half has its own key and file name, and the whole model's key is
  unchanged. One export serves both halves, which are built once and then found in the cache. The
  runtime choice: engines on a GPU; PyTorch halves on request, off the GPU, or on a TensorRT
  failure (with the message); no streaming when it is off, where the model stays unfolded as
  loaded, or for a model without halves.
- Mutation check: off-by-one slot indices fail 66 of the 72 streaming cases. The 6 that pass are
  the cases where no unit is decoded.

## 2. Accuracy

`evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1 --gridtracknet-weights
system/models/candidates/stream_s4_e60.pt --thresholds 0.3 0.4 0.5` (24 clips). Raw mode runs this
same prepass (`finetune/data/detect.py` builds a `GridTrackNetBallDetector` with the default
Config), so "before" means the base tree (windowed) and "after" means this tree (streamed).

| conf | before: recall / wrong / falseAlarm | after: recall / wrong / falseAlarm |
|---|---|---|
| 0.30 | 0.965 / 0.006 / 0.195 (101/518) | 0.965 / 0.006 / 0.195 (101/518) |
| 0.40 | 0.952 / 0.005 / 0.106 (55/518) | 0.953 / 0.005 / 0.108 (56/518) |
| 0.50 | 0.930 / 0.004 / 0.052 (27/518) | 0.930 / 0.004 / 0.052 (27/518) |

At 0.3 and 0.5 the pooled lines are identical. At 0.4, a few frames sit at the threshold and flip
under FP16 rounding: 2 more visible hits (8119 to 8121) and 1 more false alarm (video20, 3/10 to
4/10). In the per-clip table, 4 of 24 clips change in the third decimal. The before value is the
held-out figure the task quotes (0.930 / 0.052 at 0.5).

Tracker JSONs on Pomona (`clean_tracker.py --no-video`, the timing runs below): each path is
deterministic run to run, with 0 differences between its own runs.

- Streamed against windowed: selected frames go from 1102 to 1094, and the track count from 18 to
  19. On the 1094 frames both select, the median move is 0.001 px and the max 1.9 px.
- The 8 frames (1488, 1498-1504) are a far-court segment, y about 114. The detections the selector
  received there are the same set at 0.4: confidences move by at most 0.004, positions by about
  0.001 px.
- Even so, the selector's `speed_ratio` gate splits track 15 at frame 1490 in the streamed run, and
  the new 8-observation track is not selected. The selector is discontinuous at its gates, so FP16
  noise on the order of the windowed engine's own error can flip a decision. This is not a streaming
  error, but it is a real difference in the tracker's output on this clip.
- For scale, the shipped heatmap_s4 selects 1075 frames on the same clip.

Per the coordinator, the `--mode pipeline` gate and the threshold sweep are run by the main agent
from the main checkout, so they are not repeated here.

## 3. Speed

### Per call (isolated, CUDA events, median of 7 x 200 calls, locked)

Script: `latency.py` in the scratchpad.

| call | median ms | min - max |
|---|---:|---|
| windowed unit engine, 4 units (as shipped before: trainable form) | 3.80 | 3.22 - 4.29 |
| encoder engine, 4 transitions | 0.39 | 0.35 - 0.39 |
| decoder engine, 4 units | 0.55 | 0.52 - 0.76 |
| ring step for 4 frames: 4 pushes, encoder, index upload, 3 gathers, decoder (host included) | 1.21 | 1.11 - 1.25 |

Per frame at step 1, that is 0.95 ms windowed against 0.30 ms streamed, including the ring's copies,
gathers and Python: 3.1x. Engine time alone is 0.24 ms per frame (4.0x). Over Pomona's 2014 units
the network time drops from about 1.9 s to about 0.6 s.

### Prepass alone (locked, one process, interleaved, warm-up discarded, 5 runs each)

Script: `prepass_bench.py` in the scratchpad.

| | median s | min - max |
|---|---:|---|
| stream_s4 streamed | **1.85** | 1.74 - 2.11 |
| stream_s4 windowed | 2.38 | 2.34 - 3.77 |
| shipped heatmap_s4 | 2.74 | 2.69 - 2.98 |

The prepass saves 0.53 s, short of the 1.3 s of network time removed. The streamed prepass, at
0.9 ms a frame, is now bound by decode and the per-frame host work (NVDEC, the NV12 kernel launch,
`_UnitStream`, the ring's bookkeeping), not by the network.

### End to end (`clean_tracker.py -i sample/pomona.mp4 --weights ... --no-video`, from `system/`, locked)

Windowed is the base tree, c9ec36b. There was one warm-up round, then 3 rounds with the order
reversed every other round. The prepass figure is the one the tracker reports.

| | prepass r1 / r2 / r3 (s) | total r1 / r2 / r3 (s) |
|---|---|---|
| stream_s4 streamed | 2.4 / 8.1 / 4.6 | 12.8 / 30.3 / 26.1 |
| stream_s4 windowed | 2.9 / 8.5 / 4.2 | 12.7 / 26.8 / 23.6 |
| shipped heatmap_s4 | 3.2 / 3.2 / 4.2 | 12.8 / 12.7 / 20.1 |

Round 1 is the only round where the machine stayed in its fast state for all three runs. There,
the prepass ordering matches the isolated bench (2.4 < 2.9 < 3.2 s) and the totals are equal
(12.7-12.8 s). Rounds 2 and 3 caught the laptop's slow state; totals double, and in round 3 the
order is inverted. Nothing about the end-to-end totals can be concluded from these runs. As the
Phase A record found, the prepass in the tracker overlaps init and shares the CPU with boxmot's
import. The isolated prepass bench above is the number to use.

## Files

- New:
  - this record
- Changed:
  - `system/tennis_tracker/streamnet.py` (`streaming_halves`, `StreamEncoder`, `StreamDecoder`)
  - `system/tennis_tracker/engines/ball_gpu.py` (`StreamHalves`, `TransitionRing`, `transition_slots`)
  - `system/tennis_tracker/engines/ball_trt.py` (half keys and paths, `export_halves`,
    `ensure_stream_engines`, `open_stream_engines`)
  - `system/tennis_tracker/engines/ball_detectors.py` (`_open_stream`, the prepass's two branch
    points)
  - `system/tennis_tracker/config.py` (`ball_prepass_streaming`)
  - `system/AGENTS.md`
  - `audit/records/opt_detector_archs.md` (its open issue points here)
  - tests: `system/tests/test_ball_detectors.py`, `test_ball_gpu.py`, `test_ball_trt.py` and
    `_helpers.py` (`CpuGridTrackNet` takes a real network unwrapped)

Tests: `python -m pytest -q` passes in `system/` (548 passed, 1 skipped) and at the root (1275 passed, 1 skipped). `mypy`
is clean in `system/` (17 files) and at the root.

## Open issues

- The tracker's selection on Pomona differs by 8 frames: the gate flip described above. If the
  selector should be robust to detector noise of this size, that is a selector question. It applies
  to the windowed engine against FP32 as well.
- The streamed prepass is no longer network-bound. A further speedup would come from the
  per-frame host path, such as batching the NV12 conversion, not from the networks.
- The windowed stream_s4 engine still exports the trainable, unfolded form, because other
  architectures' paths were to stay untouched. Folding it would speed up `ball_prepass_streaming`
  off and the other RepVGG networks (heatmap_lite, wasb_5f), but it changes their engines' outputs
  to rounding.
- `HeatRing.add` and `take` still upload their indices from pageable memory (unchanged here). The
  ring's own index upload is pinned.
