# Test D: cancel camera motion before the frame difference (stream_s4 e60, no retraining)

Question: does moving a transition's previous frame by the estimated camera translation before
`streamnet.transition` takes the difference help e60 on the moving-camera (grid) holdout clips?

## Verdict: UNCLEAR, leaning KEEP as a cheap option; it does not pass the gate as written

- Gate: grid recall up by at least 0.01 with own-camera and prof unchanged within 0.003.
- Measured (raw mode, mixed_v1, 24 clips, conf 0.42): grid recall 0.897 -> 0.903 (+0.006), own-camera
  0.963 -> 0.963 (bit-identical), prof 0.957 -> 0.957 (bit-identical). False alarms 43/518 -> 43/518.
- Why it misses the bar: only one of the three grid holdout clips actually pans (grid_match8).
  grid_match16 and grid_match22 are static to the pixel, so the pooled grid number is diluted by
  two clips the change cannot touch. On grid_match8 itself: recall 0.930 -> 0.947 (+0.017), 9 hits
  gained, 0 lost, false alarms unchanged at 0.42 (one more bridged alarm at 0.40: 5 -> 6 of 22).
- On every static clip the estimated shift is always (0, 0), so the transition is unchanged and
  the output is bit-identical: the option cannot hurt a fixed camera.
- The holdout has one panning clip with 528 visible labels; +9 hits is suggestive, not proof. The
  two other clips clips.csv tags "moving" (grid_match85, grid_match89) are training clips at
  recall 0.968 / 0.979 already: +1 / -0 at 0.42, +1 / -2 at 0.40 (neutral, as expected on
  clips the net trained on).

## Step 1: how much does the camera move between a transition's two frames? (facts)

Transition = frame t against t - stride (stride 2 at 60 FPS, 1 at 30). Shifts in 768x432 px.

OpenCV `phaseCorrelate` on 96x54 luma (`camera_motion.py`) gave medians of 0.3 to 1.4 px on
every clip including the fixed-camera controls and did not agree with itself at 192x108; on
near-identical frames its sub-pixel centroid is noise, so it was not trusted. The exhaustive
integer search of the mean absolute luma residual (`validate_shift.py` at 192x108, `fine_shift.py`
at full 768x432, radius 5, every 5th frame) is the measurement:

| clip | camera (clips.csv) | per-transition best integer shift nonzero | max per transition | drift over 29 source frames |
|---|---|---|---|---|
| grid_match8 | moving | 18.6% of transitions (1 px, rarely 2) | 2 px | nonzero in 50%, up to 5 px (search limit) |
| grid_match16 | fixed | 0% | 0 | 0 |
| grid_match22 | fixed, 30 FPS | 0% | 0 | 0 |
| video1 (control) | fixed | 0% | 0 | 0 |
| video6 (control) | fixed | 0% | 0 | 0 |

So grid_match8 pans slowly, about 0.2 to 0.5 px per transition at 768x432 (0.5 to 1.3 px at 1080p),
with a minority of transitions at an integer 1 to 2 px. The other two grid clips do not move.
The kill rule ("no shift above about 0.5 px") was therefore not triggered, but only by one clip.

## Step 2: are e60's misses where the camera moves? (facts, grid_match8, recorded e60_0.42 pipeline run)

`misses_by_shift.py`: 528 visible labels bucketed by pan speed (best integer shift over 29 source
frames divided by the transitions in the span).

| pan px per transition | n | recall |
|---|---|---|
| 0 | 258 | 0.977 |
| (0, 0.25] | 114 | 0.956 |
| (0.25, 0.5] | 48 | 0.979 |
| (0.5, 1.0] | 97 | 0.918 |

Per-transition integer shift: zero (n=439) recall 0.961, nonzero (n=89) recall 0.888. Not flat:
misses concentrate where the camera pans fastest. Confounded (the camera pans when the ball
flies far and fast), so moderate evidence.

## Step 3: the compensation (implemented, off by default)

- `Config.ball_transition_align_px: int = 0` (system/tennis_tracker/config.py). Above 0, the
  streaming prepass's `TransitionRing.push` moves each transition's previous frame before the
  encoder sees it (system/tennis_tracker/engines/ball_gpu.py: `estimate_shift`, `aligned_previous`;
  engines/ball_detectors.py passes the option). Integer shift, not sub-pixel: the exhaustive search
  of the integer shift within the radius that minimises the mean absolute luma residual over the
  frame less a margin (ties to the smaller shift, zero first); the previous frame is moved by it and
  the uncovered strip is filled from the current frame, so the difference fields are zero there
  rather than a false edge. Integer because the measured pans are 0 to 2 px and a sub-pixel warp
  would blur the 2 to 4 px ball in the previous frame, changing its own trace.
- Only the streaming path (`ball_prepass_streaming`, the default and what raw mode runs) has the
  hook; the unit-by-unit path (`forward` on 15 channels) is untouched.
- Tests (system/tests/test_ball_gpu.py): a synthetic pure translation of a random texture by
  (2,-1), (-3,3), (1,0) is estimated exactly, the aligned previous frame equals the current one so
  both polarity fields are zero, and the ring hands the encoder the aligned pair; a zero shift and
  an unaligned ring pass the previous frame through unchanged. 134 tests in test_ball_gpu,
  test_ball_detectors and test_config pass.
- Diff: 4 files, +91 / -3 lines, nothing on by default.

## Step 4: raw mode, mixed_v1, e60 (`run.py`, log in `run.log`, detections in `out/`)

Baseline reproduces the recorded streamed numbers of audit/records/opt_stream_prepass.md exactly:
0.30: 0.965 / 0.195 (101); 0.40: 0.952 / 0.005 / 0.106 (55); 0.50: 0.930 / 0.052 (27).

| group (conf 0.42) | baseline recall / wrong / FA | aligned (radius 3) recall / wrong / FA |
|---|---|---|
| custom (own camera, 10 clips) | 0.963 / 0.004 / 22 of 207 | 0.963 / 0.004 / 22 (identical) |
| grid (3 clips) | 0.897 / 0.016 / 7 of 149 | 0.903 / 0.016 / 7 |
| prof (11 clips) | 0.957 / 0.001 / 14 of 162 | 0.957 / 0.001 / 14 (identical) |
| grid_match8 | 0.930 / 0.009 / 3 of 22 | 0.947 / 0.009 / 3 |
| grid_match16 | 0.851 / 0.043 / 4 of 16 | 0.851 / 0.043 / 4 (identical) |
| grid_match22 | 0.902 / 0.003 / 0 of 111 | 0.902 / 0.003 / 0 (identical) |
| ALL | 0.948 / 0.005 / 43 of 518 | 0.949 / 0.005 / 43 |

At 0.40: ALL 0.952 -> 0.953, FA 55 -> 56 (grid_match8 bridged 5 -> 6). Paired over all labelled
frames: aligned gained 9 hits, lost 0 (at 0.42 and 0.40); 119 labelled frames changed confidence,
all on grid_match8.

Supporting run on the training clips tagged moving (`run_moving_train.log`): grid_match85 0.968 ->
0.970, grid_match89 0.979 -> 0.979 at 0.42; +1 / -2 at 0.40. Neutral.

## Cost (facts, this laptop's CPU, 12 torch threads)

- Exhaustive integer search, radius 3 (49 candidates) at 768x432, torch on CPU: 3.8 ms per
  transition. Estimate: on the GPU in the prepass it is 49 small reductions; the prepass ran grid
  clips at 230 to 260 source fps with it on, same as without (timings under contention, not a measurement).
- OpenCV `phaseCorrelate` at 96x54: 0.10 ms per transition plus the downscale (0.15 to 0.35 ms
  measured end to end in `camera_motion.py`), but see step 1: its sub-pixel answer is noise on
  near-static frames, so it would need a confidence gate and rounding before use.
- Estimate for the phone: a 3-level pyramid search or the phase correlation on the 192x108 luma the
  model already resizes to would be well under 1 ms per frame; not measured.

## What surprised me

- clips.csv's "moving" tag and 11.0 motion_px for grid_match8 describe a slow pan: at the
  transition's time scale it is 0 to 2 px at 768x432. grid_match16 and grid_match22 are static.
- The 0.5 px kill rule cannot be judged by phaseCorrelate on 96x54: its noise floor is about 1 px.
- With one panning clip the gate is structurally hard to pass: a change that only acts on pans can
  move the pooled grid number by at most about a third of its per-clip effect.

## Files

- `camera_motion.py`, `validate_shift.py`, `fine_shift.py`: step 1 (shift CSVs `shifts_*_s8.csv`, `residual_*.csv`).
- `misses_by_shift.py`: step 2, on a pipeline run folder.
- `run.py`: step 4 (`--align-px`, `--clips`), writes `out/<variant>/<clip>.csv` and `out/summary.txt`.
- Tracker changes (behind the option): system/tennis_tracker/config.py, engines/ball_gpu.py,
  engines/ball_detectors.py; tests in system/tests/test_ball_gpu.py.
