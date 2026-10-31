# Live tracker design (phase-live, v1)

Branch `phase-live` from `phase-ABplus-heatmap` (f43218a). Model: stream_s4
(`system/models/candidates/stream_s4_e60.pt`, conf 0.42). The measurements behind every number
below are in `opt_live.md`.

## Goal

Frames arrive one at a time; each frame's ball output comes out a small, fixed delay later and
never changes afterwards. The output is the same tracking JSON the offline tracker writes. No
stage waits for the whole clip, and the run never falls behind silently.

## Shape

```
source thread          tracking thread                                   selector process
cv2 decode  -> queue -> prepass push (TRT encoder/decoder, rings) -> pass 1 -> evidence -> fixed-lag selector
(arrival time)  (drop     (frame final 4 phase frames later)        (unchanged)   (per frame)  (rows back, stamped)
                policy)
```

- `live/source.py`: `LiveSource` decodes each frame once into pinned buffers on its own thread
  and stamps its arrival (realtime file pacing: frame i is due i/fps after the first, and
  arrives at max(due, decoded)). `FrameQueue` holds the drop policy.
- `live/pipeline.py`: `_Pump` pushes each frame into the prepass and keeps it for pass 1. Pass 1
  is the offline `_Pass1Collector`, unchanged; its frame reader and ball backend pull the pump on
  demand, so it trails the prepass by exactly the lookahead it needs.
- `engines/ball_detectors.py`: the prepass loop is now `PrepassStream` (`push(frame)`,
  `finish()`), a mechanical extraction. `_prepass` keeps its signature and contract and is a loop
  of `push` then `finish`. `open_stream(..., eager=True)` runs each unit as soon as it is cut. The
  TRT engines, `TransitionRing`, `HeatRing` and the motion kernels are the same objects, not forks.
- `live/fixed_lag.py`: `FixedLagSelector` re-runs the unchanged `select_ball_in_play` over a
  sliding window and commits frames at least L old.
- `live/evidence.py`: the selector's input, recorded per frame with a ghost log, so any lag can be
  replayed on the CPU (`tools/live_lag_sweep.py`); `EvidenceFeed` is the one feeding loop that the
  live run (in line or in its process) and the replay share.
- `live/selector_process.py`: the selector in a child process, so its runs never stall pass 1.

## How each stage becomes causal

| stage | offline | live | lookahead |
|---|---|---|---|
| decode | NVDEC for the prepass, cv2 for pass 1 | cv2 once, shared | 0 |
| ball model (stream_s4, step 1) | whole-clip batched prepass | the same units, each run when cut | a frame is final when the last unit covering it is cut: 4 phase frames (8 source frames at 60 FPS, 4 at 30 FPS; 133 ms) |
| motion / background | causal | unchanged | 1 frame (the temporal gate reads the next frame) |
| ROIs | from detections up to t-2 | unchanged | none beyond the prepass |
| players (every frame), court (every 400 frames, cached) | causal | unchanged | 0 |
| ghost ROIs | found when an ROI track times out; erased from every frame after pass 1 | erased from the frames a selector run reads, as they are found | none (a frame committed before its ghost was found keeps that motion) |
| selector: timeline DP, fills, refit | whole clip | the same code over [first uncommitted - H, newest], every K frames, committing frames at least L old | L |
| bounces, contacts, legs | whole committed trajectory | computed at stream end over the committed rows | not live yet |

The live prepass is bit-identical to the batched prepass on the same frames: on pomona its
detections equal `detect_clip` with OpenCV decoding on all 2022 frames (0 differences). Offline
decodes the prepass's frames with NVDEC, whose pixels differ from OpenCV's by a little; that moves
949 of 2022 detections by at most 0.007 px and 0.001 confidence. The held-out gate in live mode
includes this effect.

## The fixed-lag selector

- The window is `[max(0, first_uncommitted - H), newest]`. A frame is judged with up to H frames
  of committed context behind it and at least L frames ahead.
- A run happens once K frames are ready to commit. So a frame waits L to L+K-1 frames after pass 1
  commits it.
- Committed rows never change. `delay=None` commits nothing until the stream ends, then runs once
  over the whole clip: the offline answer exactly (verified on the 24 held-out clips: 0/13137
  frames differ at 1e-6 px).
- Track ids: each run numbers its tracks from 0. A run's track that shares an observation (frame,
  cx, cy) with a track seen before keeps that track's stream id; otherwise it keeps its run id if
  unused, else takes the next free id.
- What the window changes: the selector normalises some scores by the clip's length (track span
  over total frames, timeline coverage and gap terms, overlap tolerance). On a window those are
  the window's length. This and the limited lookahead are the whole cost of the lag; the selector
  code is not touched.

## Choosing L, H and K

The selector's input does not depend on L (every upstream stage is causal), so each held-out
clip's evidence is recorded once in a live run and the fixed-lag selector is replayed on the CPU
over a grid of L, H and K (the replay reproduces a live run exactly: 0/13137 frames differ).
`opt_live.md` has the curve and why it bends where it does:

- the smallest L within the gate is 0.33 s;
- the smallest L that matches offline exactly is 1 s (at H of at least 1 s);
- the choice is L = 0.5 s, H = 1 s, K = 0.2 s: recall 0.951, FA 95 (79/16), .967/.913/.952,
  against offline 0.953, 95 (79/16), .971/.913/.952.

A real live gate run at the chosen point scored what the replay predicted.

## Per-frame budget

At 60 FPS a frame has 16.7 ms; at 30 FPS, 33.3 ms.

- **Tracking thread.** Its work per frame is the prepass push (the encoder on one frame, the
  decoder on a unit when one is cut), pass 1 (motion kernels, players, ROI crops) and handing the
  evidence on. Measured on pomona at 60 FPS: a median of 7 to 8.5 ms a frame. The heaviest-motion
  stretches cost 26 to 30 ms a frame for about 30 frames. There, pass 1's connected components on
  a large boost mask take 8 ms a call. Those stretches build up to 0.47 s of backlog, which the
  0.5 s budget absorbs.
- **Selector.** It runs in its own process every K = 0.2 s, over an H + L + K = 1.7 s window: 20
  to 40 ms a run on average and 150 ms at most, against a 200 ms period. It never falls behind.
- **Start-up.** Building pass 1's frame window and its CUDA graphs takes about 0.25 s. A file's
  source holds its frames until this is done. The collector's full collections (80 to 170 ms each)
  are kept off the tracking thread by freezing the start-up objects (`gc.freeze`).

`opt_live.md` has the numbers per clip.

## Frame drops

- `DropPolicy.BACKLOG` (default for realtime files and cameras): once more than `max_backlog_s`
  (0.5 s) of frames wait, the oldest are dropped in groups of `stride` consecutive frames (one per phase,
  so each phase keeps its order and its frame spacing). Every dropped index is recorded and
  reported; a dropped frame is a row with no ball in the JSON.
- `DropPolicy.NEVER`: nothing is dropped; the queue's maximum wait is reported.
- `fast` pacing (files as fast as possible) blocks the source instead of dropping.
- The console prints frames in, frames out, latency and how far behind the queue is every 5 s;
  the `.live.json` report holds every measurement.

## What cannot be live (and what is not live yet)

- Cannot: the offline answer itself. With finite L the selector cannot use evidence that has not
  arrived; a frame committed at L may differ from the whole-clip answer (`opt_live.md` counts how
  often).
- Not live yet: court events (bounces, contacts, legs) are computed once at stream end; a
  fixed-lag version needs the committed trajectory plus a bounce window of two detector steps
  each side and the merge window.
- Not live yet: a provisional output (see the two-output assessment in `opt_live.md`).
- Not live yet: the tracking video. Live writes no video.
- Not live yet: on-device (iOS, Core ML, Swift). v1 is the desktop CUDA/TensorRT tracker.
- A camera starts capturing before pass 1 is built, so its first frames can be dropped (and are
  reported). A realtime file is held instead.

## Phase C seam

`_Pump._push_ball(frame)` in `live/pipeline.py` is the one place a frame reaches the ball model,
before anything runs on it, with the frame index and the committed and pending ball state at
hand. A region gate (Phase C) chooses where the model looks there and passes the region with the
frame. v1 always looks at the whole frame. `opt_live.md` ("Phase C hook") has the contract.
