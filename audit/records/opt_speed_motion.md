# Speed: pass 1's motion masks and host overhead (2026-10-31)

Branch `speed-motion` off `phase-ABplus-heatmap` at f43218a (the base below). Three ideas for pass
1, all speed only: every kept change leaves the outputs identical. Python is the `tennis-opt` env
for the tracker and `tennis-analysis` for the evaluator and the suites. Every timed, profiled or
scored run went through `tools/gpu_lock.py`. Another builder (`speed-players`) shared the machine, so
`timing.py` logged, for every run, the CPU time any other python process used during it and the
`nvidia-smi` compute apps; a run with another python process busy (over 0.5 s of CPU) was left out.

Two models throughout: shipped `heatmap_s4` (conf 0.40) and `stream_s4`
(`models/candidates/stream_s4_e60.pt`, conf 0.42).

## Results

| commit | idea | kept | pass 1, pomona shipped / s4 (s) | outputs |
|---|---|---|---|---|
| c9ecc05 | 1: whole-frame masks squeezed, packed without a frame scan | yes | 6.06 -> 5.62 / 6.67 -> 5.68 | identical |
| 1d637ec | 3: player head decoded channel by channel | yes | 5.62 -> 4.65 / 5.68 -> 4.82 (with 30a1f42) | identical |
| 30a1f42 | 3: player outputs copied to pinned memory on the engine stream | yes | (row above) | identical |
| 21f88bc | 2: boost filter left to its first reader | yes | 4.65 -> 4.34 / 4.82 -> 4.45 | identical |
| a47c753 | test fix: skip the StreamNet engine test without onnx | - | - | - |

### Identity

The gate (`evaluate_archive.py --archive finetune --mode pipeline --holdout mixed_v1 --court`, 24
clips) and `clean_tracker.py --no-video --court` on pomona and VolyAvw, both models, were run on the
base and on each tree; `frames` (with players and court keypoints), `tracks`, `chosen_track` and
`court_events` were compared JSON for JSON:

| tree | gate clips (2 models) | samples (2 models) |
|---|---|---|
| idea 1 (c9ecc05) | 48 / 48 equal | 4 / 4 equal |
| all ideas (21f88bc) | 48 / 48 equal | 4 / 4 equal |

The motion diagnostics and player boxes of the sample JSONs are equal too, every timed run's
tracking JSON (below) equals its base run's, and the `--video` runs' `tracking.mp4` are
byte-identical to the base's. These runs were made before 21f88bc dropped a timer update (see idea
2), which changes no output; 21f88bc's two pomona samples were checked again (section 5).

Gate, all trees (the base's numbers): shipped recall 0.952, FA 89/518 (79 bridged / 10 false),
median 1.0 px; stream_s4 recall 0.953, FA 95/518 (79 / 16), median 1.0 px.

**A trap found on the way.** The first stream_s4 comparison differed on every clip (conf in the
fourth decimal, a fraction of a pixel), with the gate's metrics equal. stream_s4's TensorRT engine is
built per worktree into `system/models/engine_cache` (not linked by `tools/link_worktree_data.py`),
and two builds of the same model choose different FP16 tactics. Every tree here was given the
engine `phases/ABplus` built (the one `speed_players` uses), and the base and idea-1 stream_s4 runs
were redone; after that they are equal. Compare stream_s4 outputs across worktrees only with the
same engine file.

### Speed

`timing.py`: `clean_tracker.py --info` from each tree's `system/`, the trees and both models
interleaved in one lock, a warm-up round then 3 timed rounds (order reversed each round), median
[min-max] seconds. Trees: base f43218a, idea 1 c9ecc05, ideas 1+3 30a1f42, all 21f88bc (before its
timer fix, which changes no work).

**pomona, `--no-video`** (no run left out)

| shipped | base | 1 | 1+3 | all |
|---|---|---|---|---|
| total | 11.80 [11.70-11.80] | 11.30 [11.10-11.40] | 10.40 [10.20-10.50] | **10.10** [10.00-10.20] |
| pass1 total | 6.06 [6.01-6.09] | 5.62 [5.44-5.62] | 4.65 [4.61-4.84] | **4.34** [4.29-4.39] |
| aux wait(d2h) | 2.49 | 2.48 | 1.71 | 1.70 |
| preprocess | 4.68 | 4.21 | 3.32 | 3.03 |
| pre(cuda total) | 1.68 | 1.32 | 1.25 | 0.98 |
| pre(cuda cc) | 0.64 | 0.30 | 0.30 | - (in the selector) |
| store/debug | 1.07 | 0.56 | 0.56 | 0.25 |
| decode/slide | 0.41 | 0.41 | 0.40 | 0.39 |
| selector total | 0.36 | 0.38 | 0.37 | 0.43 |

| stream_s4 | base | 1 | 1+3 | all |
|---|---|---|---|---|
| total | 12.10 [11.90-13.60] | 11.20 [11.20-11.30] | 10.30 [10.30-10.40] | **10.20** [9.80-10.20] |
| pass1 total | 6.67 [6.48-6.85] | 5.68 [5.65-5.73] | 4.82 [4.71-4.89] | **4.45** [4.30-4.51] |
| aux wait(d2h) | 2.52 | 2.46 | 1.66 | 1.68 |
| preprocess | 5.22 | 4.29 | 3.46 | 3.11 |
| pre(cuda total) | 2.06 | 1.40 | 1.39 | 1.03 |
| pre(cuda cc) | 0.97 | 0.37 | 0.37 | - |
| store/debug | 1.54 | 0.69 | 0.68 | 0.28 |
| decode/slide | 0.43 | 0.42 | 0.40 | 0.40 |
| selector total | 0.39 | 0.39 | 0.39 | 0.46 |

**VolyAvw, `--no-video`.** Two timed runs (1+3 stream_s4 round 2, all shipped round 2) had another
python process busy (0.6 s of CPU) and are left out; one more round of those two trees was run in a
second lock, so every cell has 3 or 4 clean runs.

| shipped | base | 1 | 1+3 | all |
|---|---|---|---|---|
| total | 18.10 [18.00-18.30] | 18.00 [17.80-18.60] | 16.65 [16.50-17.10] | **16.60** [16.30-17.20] |
| pass1 total | 10.21 [10.19-10.42] | 10.21 [9.99-10.42] | 8.81 [8.62-8.99] | **8.29** [8.16-8.96] |
| aux wait(d2h) | 5.08 | 5.06 | 3.81 | 3.78 |
| preprocess | 7.96 | 7.90 | 6.60 | 6.04 |
| pre(cuda cc) | 0.48 | 0.45 | 0.47 | - |
| store/debug | 0.95 | 0.86 | 0.88 | 0.38 |
| selector total | 0.84 | 0.84 | 0.86 | 1.15 |

| stream_s4 | base | 1 | 1+3 | all |
|---|---|---|---|---|
| total | 17.20 [17.10-17.20] | 17.10 [16.90-17.90] | 15.90 [15.50-16.00] | **15.40** [15.20-15.50] |
| pass1 total | 10.33 [10.12-10.36] | 10.17 [10.08-10.86] | 8.96 [8.68-9.05] | **8.26** [8.18-8.27] |
| aux wait(d2h) | 5.02 | 5.03 | 3.80 | 3.78 |
| preprocess | 7.94 | 7.89 | 6.64 | 6.03 |
| pre(cuda cc) | 0.46 | 0.47 | 0.49 | - |
| store/debug | 0.94 | 0.88 | 0.90 | 0.38 |
| selector total | 0.87 | 0.85 | 0.84 | 1.03 |

**pomona, with video** (base against all, no run left out):

| | base shipped | all shipped | base stream_s4 | all stream_s4 |
|---|---|---|---|---|
| total | 16.30 [16.30-16.40] | **15.30** [15.10-15.50] | 16.60 [16.60-16.70] | **15.00** [14.90-15.10] |
| pass1 total | 6.35 [6.34-6.51] | 5.08 [5.00-5.27] | 6.89 [6.89-7.02] | 5.01 [4.99-5.28] |
| pass2 total | 4.08 | 4.13 | 4.13 | 4.18 |
| selector total | 0.40 | 0.45 | 0.40 | 0.47 |

Pass 1 is 28-33% faster on pomona and 19-20% on VolyAvw; end to end 8-16% without video, 6-10%
with it. Idea 1 pays on pomona (78 and 127 full-frame frames) and is within noise on VolyAvw (54
and 53); idea 3 pays on every clip; idea 2 moves part of its work into the selector (below).

## 1. Cheaper whole-frame motion masks (kept, c9ecc05)

**Measured first** (pomona, stream_s4, base, instrumented): the 127 full-frame frames took 9.54 ms
each in `MotionMasks._resolve` (1.21 s in all) plus 3.8 ms each in packing (0.49 s); the 1895 ROI
frames took 0.16 ms each. A full-frame cleaned mask is sparse: median 1311 set pixels in 78
components, spread over a bounding box of about 1.4 MP. The component pass scanned that box,
`PackedMask.pack` scanned the frame again, and the frame-sized narrow mask was zeroed, maxed and
scanned for the boost.

**Built** (`tennis_tracker/motion.py`):

- The device reduces each whole-frame mask's row and column occupancy (`any` over each axis) and
  appends them to the same pinned D2H, so the host knows the bounding box without a scan.
- `_SqueezedMask` squeezes every run of empty rows and of empty columns to one. An 8-connected
  component never spans an empty row or column, so the squeezed image holds the frame's components
  whole and apart with their areas, widths and heights; moved back by one integer offset per axis,
  their centroids round to the same pixel (both are an integer sum over the area). When squeezing
  would keep over half the box, the box is used as it is.
- The component filter is vectorised (`_boost_discs`, shared with ROI frames), each disc is drawn on
  its own small stamp, and the boost is packed from the stamps' points (`_pack_stamps`: the bytes
  `np.packbits` would give, from the set bits). The raw mask is packed from the known box.

**Evidence.** Offline on the 127 dumped pomona crops: identical masks on 127 / 127, 9.02 ms -> 4.11
ms a frame counting the occupancy reduction on the CPU (on the device it is asynchronous). In the
pipeline `pre(cuda cc)` falls 0.64 -> 0.30 s (shipped) and 0.97 -> 0.37 s (stream_s4), `store`
1.07 -> 0.56 and 1.54 -> 0.69 s. Tests (`tests/test_motion.py`): the squeezed path gives the
frame path's packed masks, blobs and full arrays for six scattered masks with and without a wide
mask; `_boost_discs` equals the per-component loop it replaced, including aspects exactly on the
3.5, 4.5 and `blob_max_aspect` limits; stamps pack as discs drawn on the frame (radii 1 to 12, on
corners, overlapping); squeezing keeps components whole and apart. Mutations each fail them: no
gap rows, a disc margin one short, a centroid offset off by one, `<=` for `<` on either aspect
limit, `>` for `>=` on the area.

## 2. Filter only the frames the selector reads (kept, 21f88bc)

**The question.** The selector reads a frame's boost mask only where it has a detection or a motion
search. Recording which frames it reads (idea 1 in place): pomona 1165 / 2022 (shipped) and 1195
(stream_s4), VolyAvw 2794 / 3286 and 2776. The filter and the boost packing on frames never read
cost 0.31 s and 0.57 s on pomona, 0.14 s and 0.15 s on VolyAvw, measured inside pass 1.

**The constraints.** Ghost-ROI erasure clears ghost boxes in the filtered boost, so it must run after
the filter; a stored `MotionMasks` holds the pinned download buffer (two frame-sized masks on a
full-frame frame), so keeping them all until the selector is not possible; ROI boxes that overlap
are filtered each on its own crop, so the union mask is not enough to filter later.

**Built.** At commit, pass 1 packs the raw mask as before and stores, instead of the packed boost, a
`DeferredBoost`: its own copy of the ROI crops, or of the squeezed whole-frame mask (a few KB),
and the settings. `Pass1Result.boost_masks` is a `BoostMasks`, a `Sequence` that resolves an entry
the first time it is indexed and keeps the `PackedMask`. The selector's input type was already
`Sequence[Optional[MaskLike]]`, so it is unchanged; ghost erasure reads and writes through the same
indexing, so it erases the filtered mask exactly as before; pass 2's motion debug and the export's
motion diagnostics read through it too. When a frame's masks were already filtered in pass 1 (the
detector-input debug video), the packed boost is stored as before. Frames with no raw motion store
None, as before.

**Evidence.** Pass 1 falls 4.65 -> 4.34 s and 4.82 -> 4.45 s on pomona, 8.81 -> 8.29 s and 8.96 ->
8.26 s on VolyAvw. The selector grows by 0.06-0.07 s on pomona and 0.19-0.29 s on VolyAvw (the
filtering of the frames it reads), so end to end it saves 0.05-0.5 s. The saving in pass 1 is larger
than the unread frames' cost: the work that moved runs faster in the selector than in pass 1 (see
idea 3: pass 1's host work runs about twice as slow as the same work alone). The `pre(cuda cc)`
row now covers only frames filtered in pass 1 and is absent from the normal runs above. The first
version also added the deferred filter's time to pass 1's perf dict after pass 1 had reported it;
21f88bc drops that (nothing read it). Tests: a deferred boost resolves to `packed_boost` bit for bit for
overlapping ROIs whose crops disagree in the overlap, three scattered whole frames with a wide
mask and a dense whole frame (where the squeezed image is a view of the box), and keeps no view of
the download buffer; nothing is deferred without raw motion; `BoostMasks` filters an
entry once, on first read, and supports slicing, iteration and assignment. Dropping the cache, the
crop copy or the view copy each fail them.

## 3. Pass 1's host overhead (kept, 1d637ec and 30a1f42)

**The question.** The player box decode took 0.18 ms isolated against 0.67 ms in the pipeline. GIL,
CUDA sync stalls, the allocator, or the cache?

**Measured** (pomona, shipped, base, two runs each, wall time and the thread's own CPU time from
`QueryThreadCycleTime` around each step of `detect_async_finish`, ms per frame):

| step | wall | thread CPU |
|---|---|---|
| `session.wait` (stream waits on the launch's event) | 0.024 | 0.024 |
| D2H (`.float().cpu()` of the transposed head) | 0.20-0.22 | 0.20-0.21 |
| numpy decode (threshold, boxes, NMS) | 0.34-0.35 | 0.34-0.35 |
| decode total | 0.81 | 0.80 |
| ByteTrack update | 0.77-0.78 | 0.77-0.78 |

- Not the GIL: the thread's CPU time equals its wall time in every step, so it never waited for
  another thread.
- Not a CUDA sync stall: synchronising the device before the decode took 0.27-0.36 ms but left the
  D2H at 0.19-0.20 ms, so the transfer was not waiting for queued GPU work.
- The same numpy decode on 600 dumped player heads alone: 0.148 ms against 0.34-0.35 ms in the
  pipeline. Decoding each head three times in a row inside the pipeline (new decode): 0.257, 0.160,
  0.131 ms against 0.069 ms alone. So part is cold data (the first call), and the rest is the core
  itself running pass 1's host work about twice as slow as the same work alone (other threads and
  the prepass child on the CPU). Nothing waits; there is just more CPU work than the isolated
  number suggests, so the fix is less work.
- The allocator: the old D2H allocated a device copy, a float copy and a host tensor each frame;
  both fixes remove them, which the numbers cannot separate from the rest.

**Built.**

- 1d637ec (`engines/engine_boxes.py`): the player head is `[1, 5, 8400]` FP16. The decode
  transposed it on the device, converted all of it to float32, copied it and built 8400 boxes before
  thresholding. It now reads the head as it lies (a channel is a contiguous row), thresholds the
  scores and converts only the anchors that pass; every value is the float32 the old path computed.
  `decode_engine_boxes` also takes a host array.
- 30a1f42 (`engines/player_detector.py`): the outputs were cloned on the device (the next launch
  reuses the buffers) and later copied with a synchronous transfer on the current stream. They are
  now copied straight into pinned host memory on the engine's stream behind the launch's event; the
  finish waits for that event and decodes the host copy.

**Evidence.** Offline, 600 pomona heads: 0.148 -> 0.069 ms, identical rows on each. In the pipeline
(pomona shipped, three interleaved rounds): decode 0.75 -> 0.62 (1d637ec) -> 0.49 ms, the whole
finish 1.52 -> 1.38 -> 1.27 ms. In the timing above, `aux wait(d2h)` falls 2.48 -> 1.71 s (pomona)
and 5.06 -> 3.81 s (VolyAvw), and pass 1 by 0.9-1.4 s. Tests: the channel-major decode equals the
anchor-major one bit for bit for FP16 and FP32 heads in either layout with one and three classes,
ties across classes and scores exactly on the threshold, on the tensor and on its host copy (an
argmax that picks the last tie, or `>` for `>=`, fail it); a host copy of postprocessed rows decodes
as the tensor; on the real player engine (CUDA) two launches in a row leave each frame's pinned
outputs equal to a direct launch's.

**Left.** ByteTrack's update is 0.77 ms a frame (boxmot, in Python); the rest of `decode_engine_boxes`
(class filter, scaling, clipping on a handful of rows) is about 0.18 ms of small numpy calls.

## 4. Test hygiene

`tests/test_ball_trt.py::test_a_small_streaming_networks_engines_stream_what_its_forward_pass_answers`
failed in `tennis-analysis` (no `onnx`; the engines are built from an ONNX export) instead of
skipping. a47c753 skips it there; it runs and passes in `tennis-opt`.

## 5. Suites

At 21f88bc, all under the lock: `system/` 626 passed, 11 skipped (`tennis-analysis`) and 637 passed
(`tennis-opt`, which has CUDA, TensorRT and onnx: nothing skipped); the root suite 1359 passed, 11
skipped; `mypy` clean in `system/` (20 files; `motion.py` and `engines/engine_boxes.py` joined the
list) and at the root. The system suite went from 609 to 637 tests. The pomona samples of both models
at 21f88bc equal the base's.
