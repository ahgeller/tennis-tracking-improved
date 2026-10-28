# Pass 1 optimization (motion and players), 2026-10

Phase A item 5 of `optimization_plan_2026-10.md`. Branch `worktree-agent-ab3cd5c5541546be6`.

## How it was measured

- **Parity bench** (`output/p1/bench_pass1.py`, not committed): runs `collect_pass1` on
  `sample/pomona.mp4`. The ball detector replays the detections a full baseline run recorded,
  so pass 1's outputs depend only on pass 1's code and have no prepass nondeterminism in them.
  It pickles every pass-1 output (raw and boost masks, ROIs, ghost ROIs, player boxes, court
  keypoints). Each step was compared bit for bit against the baseline replay.
- **Noise floor**: two full baseline runs (`clean_tracker --info`) on the same code differ by up
  to 0.001 px and in raw detections on 634 frames (prepass nondeterminism), with ROIs differing on
  5 frames. End-to-end JSON comparisons are judged against that floor.
- **Timing**: A/B runs interleaved with a copy of the baseline tree (`git archive` of
  `optimization`), all under the shared lock
  (`optimization_test/tools/gpu_lock.py`). Timings taken before the coordinator's lock fix
  (each worktree had its own lock) were discarded and re-taken; see "Results".

## Decisions

| # | Decision | Evidence | Result |
|---|----------|----------|--------|
| 1 | Upload the frame as uint8 into a pinned slot (3 slots, each guarded by an event) and convert to float CHW on the GPU | uint8 is a quarter of the bytes. `out.copy_(u8.permute(2,0,1)); out.div_(255)` gives exactly the host path's floats (test `test_upload_and_planes_are_bit_identical_*`) | Outputs identical (CPU share measured only in unlocked cProfile runs, so not reported as a timing) |
| 2 | Reorder channels in `engine_session` with `flip(1)` and not `t[:, [2,1,0]]` | Indexing with a Python list builds an index tensor on the host and copies it with a pageable H2D copy, which synchronizes the stream ([PyTorch: pinned memory and non_blocking](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)). In the profile this was the implicit sync in `preprocess_cuda_chw_frame` (2.7 s) | Sync gone; identical values |
| 3 | Gray, V and S computed once per frame into one `(3,H,W)` tensor; the background model holds V and S as one `(2,H,W)` tensor | The old code recomputed them per probe and per ROI, and ran every op twice. Stacked arithmetic is bit-identical (test `test_stacked_background_update_is_bit_identical_*`) | Fewer launches; identical |
| 4 | Foreground and probe deltas once over the union box of the ROIs; each ROI blurs its own sub-crop | `avg_pool2d` with padding excludes padded cells, so a crop equals the zero-padded per-ROI blur (test `test_rois_in_one_union_box_*`) | Identical masks |
| 5 | Raw-mask closing and opening on the GPU (`max_pool2d` with cv2's anchor `k//2`; erode = 1 - dilate(1 - x)) | OpenCV morphology semantics; test `test_gpu_cleaning_matches_opencv_*` checks 8 kernel pairs × 3 densities, border pixels included | Removes the per-frame full-frame D2H and the CPU morphology |
| 6 | Only the ROI crops of the cleaned masks are copied to the host, asynchronously into a pinned buffer, and resolved (CC filter, pack) at commit, one frame later | The crop is all the packed mask stores. An event and a deferred wait let the copy overlap the next frame | `pre_cuda_raw_d2h` 0.04 ms/frame |
| 7 | Connected components scan only the bounding box of the set pixels (`cv2.boundingRect`) | Motion is sparse, and CC costs in proportion to the area scanned. The centroid shifted by the box offset rounds to the same pixel (argument in code; test `test_filter_boost_mask_in_the_motion_bounding_box_*`) | Locked end to end, `pass1 pre(cuda cc)`: 0.50 to 0.33 ms/frame, and `pre(cuda d2h)` 0.30 to 0.01; identical masks |
| 8 | Player inference finished one frame later, with outputs cloned on the engine stream and a per-launch event | TensorRT reuses the output buffers, so a second launch before the finish would overwrite them. `session.wait(event)` waits for that launch only | Player boxes identical; the wait overlaps a frame of work |
| 9 | Half-resolution motion: **rejected** | After 4-7, the motion GPU work is no longer the bottleneck (see results), and half resolution changes the masks for 2-3 px balls, so it needs a full held-out re-score for at most a fraction of a ms | Not built |
| 10 | GPU connected components (CuPy/cucim): **rejected** | CuPy 14 needs numpy 2 (the env pins 1.26.4) and an NVRTC DLL that is not installed. Uninstalled again. After 7 the CPU CC costs 0.5 ms/frame | Not built |
| 11 | Fusing the elementwise ops with `torch.cuda.jiterator`: **rejected** | It works, but measured about 0.5 s of NVRTC compile per kernel per process, with no disk cache on Windows. That is more than it saves on a 2000-frame clip | Not built |
| 12 | CUDA Graphs for the motion ops: **deferred, untested in pass 1** | Measured on a 31-op sequence: 46 µs replay vs 3.5 ms eager launch. But `engine_session` notes that a capture is invalidated by the prepass worker thread's CUDA work. Pass 1 would need its capture before the prepass starts, and the ROI shapes change every frame (one graph per shape) | Open issue |
| 13 | Player detection every Nth frame | See "Players" | See "Players" |

## Players

boxmot 16.0.10 `ByteTrack.update` with no detections marks every track lost and returns no
boxes, so the old `player_detect_interval > 1` path (an update with empty detections in
between, described in `config.py` as "coasting") gave **empty** player boxes on the frames in
between. Now:

- the engine and the tracker run only on every Nth frame;
- `track_buffer` is `60 / N` updates, so a lost track is still kept for 60 frames;
- the frames in between are filled by `interpolate_player_boxes`. A track seen on both sides moves
  linearly; one seen on one side keeps the nearer side's box. Every consumer of player boxes is
  offline (selector, court events, pass 2, export), so interpolation is valid there.

Box error against every-frame tracking on Pomona (replay bench, 3785 reference boxes, each
matched by best IoU; "unmatched" means IoU < 0.3):

| N | mean corner err (px) | p95 | max | unmatched | extra |
|---|---|---|---|---|---|
| 2 | 1.13 | 4.33 | 24.4 | 5 | 0 |
| 3 | 1.81 | 7.01 | 24.8 | 15 | 1 |
| 4 | 2.35 | 8.81 | 44.1 | 21 | 4 |
| 6 | 3.11 | 11.75 | 45.9 | 43 | 2 |

Player boxes are used through `_is_near_player`, which dilates them by a margin plus twice
the box height, and by court events (striker, bounce-inside-player). Both tolerate a few
pixels. N=3 measured locked at 4.80 ms/frame against 6.02 for N=1 (see Results).

Downstream, N=3 against N=1 (same code, `player_detect_interval` the only change; outputs only,
not timed):

- Pomona, end to end with `--court`:
  - the filled frames and tracks are identical, and the same 9 bounces and 3 contacts are found;
  - contact positions move by up to 4.3 px;
  - the player ids differ only in numbering.
- Held-out selector score (`evaluate_archive.py --mode pipeline --holdout mixed_v1`):
  - recall is the same (0.946);
  - false alarms go from 95/518 to 97/518 (video23 +1).
- Labelled landings (`score_court_events.py` on the 7 clips with landing labels):
  - hits fall from 27 to 26 (video6 loses one);
  - recall falls from 0.435 to 0.419, and median px error rises from 1.4 to 1.7.

**Decision: N stays 1.** The interval path is now correct (before, it produced empty boxes) and
available. But on these measurements N=3 costs a labelled landing for about 1.2 ms/frame of
pass 1, and end to end pass 1 is not the limit while the ball prepass runs alongside it.

## Results

All timings below were taken under the shared lock
(`optimization_test/tools/gpu_lock.py`, after the coordinator's fix), on 2026-10-27 22:00-22:22.
Every timing taken earlier, under the per-worktree lock, was discarded. The lock serializes
GPU benchmarks only. Other agents' CPU work (training data loaders, tests) still ran, and the
spread between rounds shows it.

**Pass-1 replay bench** (pass 1 alone, ball detections replayed, no prepass), interleaved:

| run | base ms/frame | new ms/frame | base main-thread CPU | new main-thread CPU |
|---|---|---|---|---|
| round 1 | 12.07 | 10.47 | 11.87 | 10.14 |
| round 2 | 10.08 | 6.02 | 9.81 | 5.90 |
| new, player N=3 | | 4.80 | | 4.52 |

**End to end** (`clean_tracker.py --input sample/pomona.mp4 --no-video --info`), one run each:

| | total | init | pass 1 | pass 1 ms/frame | preprocess ms/frame |
|---|---|---|---|---|---|
| baseline | 23.3 s | 3.5 s | 19.35 s | 9.57 | 5.82 |
| new | 21.1 s | 5.7 s | 14.84 s | 7.34 | 4.42 |

End to end, pass 1 shares the GPU with the ball prepass (12.5-16 s, overlapped), so it gains
less than in the replay bench. The init difference (3.5 vs 5.7 s) is not in code this work
touched, so it is noise.

**Parity.**

- *Replay bench*: raw masks, boost masks, ROIs, ghost ROIs, court keypoints and player boxes
  are bit-identical to the baseline on all 2022 frames, after each step.
- *End to end against two baseline runs*:
  - same 1075 filled frames and the same 14 tracks;
  - positions within 0.0015 px;
  - court events (with `--court`) have the same counts, with numeric differences up to 0.001;
  - the 5 frames whose masks or ROIs differ and the 635 frames whose raw detections differ are
    the prepass nondeterminism, which two baseline runs show too (0.0011 px, 634 frames).

## Open issues

- CUDA Graphs for the motion ops (decision 12) were not built. They would need capture before the
  prepass thread starts, and one graph per ROI shape.
- Timing spread: the shared lock serializes GPU benchmarks, but other agents' CPU load still
  moves pass-1 numbers by 2x between rounds (replay base 10.1-12.1, new 6.0-10.5 ms/frame).
  Under the lock, waiting took 1.5 h. The end-to-end figures are single runs.
- The first ~200 frames of an end-to-end run go at ~26 fps (the prepass and engines warming
  up). That is outside pass 1.
- `render/summaries.py` still lists the `pass1_pre_postmask` timer row, which pass 1 no longer
  records (the row is skipped when absent).
