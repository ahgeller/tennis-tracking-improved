# Phase A items 4 and 6: GPU decode and pass 2 output (2026-10-27)

Branch `worktree-agent-aba1da3b0eef4bc0b` off `optimization`. Plan items: section 3.A.4
(decode once with NVDEC, fan frames out, fix the pass 2 seek fallback) and 3.A.6 (court
layer once, writer format) of `optimization_plan_2026-10.md`.

Every timing here was taken through the shared lock
(`C:\Users\Andrew\Desktop\clean_version\optimization_test\tools\gpu_lock.py`, lock dir
`optimization_test\.gpu_bench_lock`). The before/after numbers in sections 5 and 6 were all
taken after the coordinator's lock correction; the two benches marked "early" ran while other
agents still used per-worktree locks, and they only rank options. The lock serialises
GPU benchmarks only: other agents' CPU work (tests, data prep) still ran alongside, so the
absolute numbers are noisier than the plan's idle baseline. Compare before/after pairs, which
were run back to back in one locked script.

## 1. Decoder choice

| option | result | evidence |
|---|---|---|
| PyNvVideoCodec 2.2.3 (`ThreadedDecoder`, NV12 out) | **chosen**. Windows wheel for Python 3.10, works with the env's torch 2.10+cu128 through DLPack. pomona: 1376 fps NV12 vs OpenCV 227 fps (same run; early). | https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html ; locked `bench_decode.py` |
| PyNvVideoCodec RGB output | rejected: 1017 fps, and its conversion honours BT.709, 2-26 levels off OpenCV's pixels on the test clips | same bench; pixel diff on 4 clips |
| `SimpleDecoder` | rejected for streaming: 724 fps NV12 (no decode-ahead thread) | same bench |
| torchcodec CUDA | rejected: no Windows CUDA wheel for cu128; needs torch cu130 or conda-forge with FFmpeg 4-7 DLLs, which changes the shared env's torch | https://pypi.org/project/torchcodec/ , https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec |
| ffmpeg `-hwaccel cuda` subprocess | rejected: frames come back over a pipe to host memory (6 MB per 1080p frame) and frame identity relies on ffmpeg's own counting; the prepass wants CUDA tensors | plan section 1 (ffmpeg CPU ~2200 fps already beats it for host frames) |

## 2. Colour: bit-identical to OpenCV

OpenCV 4.9's FFmpeg backend converts with swscale, which ignores the stream's colour matrix
(SWS_CS_DEFAULT is BT.601) and uses full range only for `yuvj*` formats. The x86 path is the
fixed-point MMX/SSE code in `yuv_2_rgb.asm` (pmulhw on 3-bit-shifted samples, nearest chroma,
saturating), with coefficients from `ff_yuv2rgb_c_init_tables`.
Sources: https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1/libswscale/x86/yuv_2_rgb.asm ,
https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1/libswscale/yuv2rgb.c .

`video_decode.nv12_to_bgr` reproduces that arithmetic in int32 torch ops. Measured: equal to
OpenCV byte for byte on all four sample clips, and on H.264 and HEVC test clips in limited and
full range (`tests/test_video_decode.py`). Because the pixels are identical, detections are
unaffected by construction (no detection-difference study needed).

Range cannot be read reliably (PyNvVideoCodec's demuxer reports `UDEF` unless the container
repeats the VUI flag; OpenCV's `CAP_PROP_CODEC_PIXEL_FORMAT` says `I420` either way), so it is
calibrated on frame 0: the range under which NVDEC's frame 0 equals OpenCV's. If neither
matches, NVDEC is refused and OpenCV decodes.

## 3. Frame identity

- **pass 2 seek fallback, root cause**: pomona is variable frame rate (`r_frame_rate` 60/1,
  average 59.67). OpenCV's `CAP_PROP_POS_FRAMES` seek converts the frame number to a time
  with the average rate, so the seek to frame 689 decoded frame 683 (and 300 decoded 301);
  the cache check failed and pass 2 decoded the whole clip again. Measured by decoding pomona
  sequentially and comparing each seek's picture.
- **Fix**: pass 2 reads past the cache through `video_decode.open_decoder(start_frame=n,
  host=True)`. NVDEC seeks and checks every frame by timestamp (below). OpenCV never seeks: it
  decodes from frame 0 and drops the cached prefix with `grab()` on its first `read`, i.e. on
  pass 2's reader thread while the cached frames render. A first version seeked early and walked
  forward to the cached pictures; it found the right pictures, but after any seek OpenCV counts
  frames from its timestamp guess and ended pomona's stream early (1330 of 1332 frames from a
  seek near 620; plain seeks to 300/600/1000/1500 give 1721/1423/1023/523 frames against
  1722/1422/1022/522). `grab()` of 690 frames costs 0.30 s with no colour conversion, so the
  seek bought nothing. Verified on pomona from starts 0, 300, 690 and 1500 with both backends:
  every frame byte-identical to the sequential OpenCV decode, counts exact. Tests:
  `test_pass2.py` (an unseekable capture; the VFR index-painted clip for both backends at caches
  of 50/100/150) and `test_video_decode.py::test_opencv_starts_at_the_frame_asked_for_and_reads_to_the_end`.
- **NVDEC `start_frame`** is exact on H.264 (pomona and VFR test clips with B-frames) but lands
  1-3 frames late on an x265 open-GOP clip (start 37 gave frame 40). `NvdecDecoder` therefore
  maps each decoded frame's timestamp to its index through the packets' PTS in presentation
  order (pomona: 2022 packets demuxed in 0.06 s), drops frames before the start, and decodes
  again from frame 0 when the start lands late. `ThreadedDecoder`'s timestamps equal the
  packet PTS; `SimpleDecoder`'s sit the B-frame delay below them (-512 on the test clips),
  which is why the map is only used with `ThreadedDecoder`.
- **Clips NVDEC cannot take**: `ThreadedDecoder` aborts the process from its thread on a clip
  below the hardware minimum (seen on a 256x32 clip; H.264 min 48x64, HEVC 144x144), so the
  caps are checked first with `GetDecoderCaps`; anything unsupported falls back to OpenCV.

## 4. Shared frame source (for the prepass and pass 1 owners)

`video_decode.SharedFrameSource(decoder, capacity=32)` decodes once on its own thread into a
bounded ring; each `consumer(name)` (registered before `start()`) reads every frame in order
as `Frame(index, image)`. The producer waits while the slowest consumer is `capacity` frames
behind; a consumer that is done calls `close()` so it no longer holds the ring back. CUDA
frames carry an event recorded on the decode stream; `read()` makes the consumer's current
stream wait on it and calls `record_stream`, so a consumer on its own stream is safe.
Decode errors surface in every consumer as `RuntimeError`. Usage:

```python
decoder = open_decoder(path, device=torch.device("cuda", 0))   # NVDEC, else OpenCV host frames
source = SharedFrameSource(decoder, capacity=32)
prepass, pass1 = source.consumer("prepass"), source.consumer("pass1")
source.start()
for frame in prepass: ...      # in the prepass thread; frame.image is a CUDA HWC uint8 BGR tensor
source.close()
```

Not wired into the prepass or pass 1 here: those modules belong to other workstreams, and pass
1's ByteTrack and motion code take host frames, so wiring them is their owners' call. A host
consumer can open its own `NvdecDecoder(host=True)`, which converts a batch of 12 frames in one
pass and downloads it with one asynchronous copy to pinned memory (pomona from frame 690:
815-819 fps against OpenCV's 449-605 fps in the same locked run).

## 5. Pass 2

- **Court layer once: measured and rejected.** A cached layer (the opaque marks drawn once per
  court on a black and a white canvas, the pixels where they agree stamped per frame by index,
  the anti-aliased net line still drawn per frame) was byte-identical to the direct drawing
  (pipeline golden and noisy-frame tests passed), but the locked microbenchmark on pomona's court
  measured it slower: direct `draw_court_layer` 0.12 ms/frame, cached 0.19-0.22 ms/frame. The
  plan's and my cProfile figures (court layer ~half of the render) were cProfile's per-call
  overhead on ~45 small OpenCV calls a frame. Reverted in `baf5136`; the court layer costs
  ~0.25 s per pomona run, so there is nothing to win there.
- **Per-frame copies**: cached frames are handed to pass 2 (the cache entry is cleared) and
  drawn on directly instead of copied; the cache's memory is freed as pass 2 goes.
- **Writer format**: measured, kept bgr24. Locked `bench_writer.py` (early), 600 pomona frames to
  `h264_nvenc` p1:

  | input to ffmpeg | wall | main-thread ms/frame | PSNR vs bgr24 output |
  |---|---:|---:|---:|
  | bgr24 (current; ffmpeg converts) | 2.20 s | 3.31 | - |
  | yuv420p (cv2 BGR2YUV_I420 in Python) | 2.76 s | 4.42 | 41.6 dB |
  | bgra (NVENC converts on the GPU) | 3.65 s | 5.85 | 42.1 dB |

  Converting in the tracker costs more main-thread time than the 2x smaller pipe saves, BGRA
  pipes 33 % more bytes, and both change the output (about 42 dB), so neither is visually
  identical. Re-measured on a quieter machine as `VideoWriter` drives it (drain thread, 900
  frames): bgr24 828-850 fps, bgr24 with `-filter_threads 8` 837-839 fps (byte-identical), I420
  converted on the drain thread 627-638 fps. The writer alone outruns pass 2 (~530 fps), so
  bgr24 stays. NVDEC as an encoder input is moot: pass 2 draws with OpenCV on host frames.
- **Pass 2 decoder**: both backends give byte-identical frames, so the choice is speed. With 690
  cached frames, pass 2 alone (same pickled inputs, 3 locked runs each, medians): base 3.95 s,
  NVDEC 3.84 s (read wait 0.31, write wait 1.10), OpenCV 3.62 s (read 0.96, write 0.57). NVDEC
  cuts the read wait but its conversion and download contend with the NVENC writes, so
  `Config.gpu_decode` defaults to False (OpenCV) for pass 2.

## 6. Results (locked)

All under the shared lock, pomona with video, base = merge base `93771f8`, run back to back.

| | base | new |
|---|---:|---:|
| end to end, 3 runs (`[done] total`) | 25.9 / 25.8 / 26.2 s | 25.3 / 25.6 / 25.9 s |
| pass 2 in those runs | 3.87 / 3.96 / 4.15 s | 3.56 / 3.68 / 3.81 s |
| - read wait | 1.04 / 0.92 / 0.81 | 0.95 / 0.92 / 1.00 |
| - render | 2.22 / 2.43 / 2.72 | 1.87 / 2.12 / 2.12 |
| - write wait | 0.16 / 0.16 / 0.16 | 0.59 / 0.50 / 0.53 |
| pass 2 alone (harness, 3 runs) | 4.15 / 3.90 / 3.95 s | 3.59 / 3.62 / 3.63 s |
| `--no-video` end to end | - | 22.4 s |

Pass 2 is ~0.3 s (7%) faster: no second decode of the cached 690 frames, and no copy of each
cached frame before drawing. End to end the gain is within pass 1's run-to-run spread; pass 1
(~18.8 s) is not this workstream's. Decode alone from frame 690 (host frames, 1332 frames):
OpenCV 449-605 fps, NVDEC 815-819 fps, NVDEC open 0.25-0.34 s. Decode alone from 0 to the CUDA
side (`bench_decode.py`, NV12 not converted): `ThreadedDecoder` 1376 fps, OpenCV 227 fps (that
run overlapped other agents' CPU work).

Timings re-taken after the coordinator's lock correction: all of the above. Discarded: the
first baseline (56.7 s, per-worktree lock) and every pass 2 timing before the court-layer revert.

## 7. Parity

- Tracking JSON, `--no-video` new vs the baseline `output/base_novideo.json`: identical frames,
  tracks, bounces and positions (deep compare at 0.01 px); the only differences are the new
  `config.gpu_decode` key and the timing fields.
- Tracking JSON with video, 3 new runs vs 3 base runs: the same, plus the weight paths (the base
  ran from a scratch copy).
- Rendered video: pass 2 alone from the same inputs is byte-identical for base, NVDEC and OpenCV
  (2022 of 2022 frames). End to end, runs 2 and 3 are byte-identical to their base runs; run 1
  differs on 140 frames, starting at one frame whose HUD reads `c=0.70` in one run and `c=0.71`
  in the other (worst PSNR 48.15 dB, the rest encoder carry-over). Two base runs differ in the same
  way (161 frames, the same HUD patch): the detector's confidence varies by about 1e-3 from run to
  run, within the JSON's tolerance; it is not from this change.
- NVDEC host frames from starts 0, 300, 690 and 1500 of pomona: every frame byte-identical to the
  sequential OpenCV decode, frame counts exact.

## 8. Open issues

- `SharedFrameSource` is not yet consumed by the prepass or pass 1 (other owners). That is where
  NVDEC pays: the prepass spends ~4.3 s in `cv2` reads and ~1.4 s uploading (plan section 1).
- Pass 2's floor is now render plus a write wait from CPU contention with ffmpeg; the render
  (trails, markers, HUD) is ~1 ms/frame of OpenCV drawing.
- PyNvVideoCodec 2.2.3 is optional (`system/requirements.txt`); without it OpenCV decodes.
- NVDEC `start_frame` is exact on H.264 and late on x265 open GOPs; `NvdecDecoder` checks every
  frame against the packet timestamps and restarts from frame 0 when a start lands late, so a
  late start costs a decode of the prefix, never a wrong frame.
