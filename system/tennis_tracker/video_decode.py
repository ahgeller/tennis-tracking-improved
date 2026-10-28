"""Decoding the clip: NVDEC onto CUDA where it is available, OpenCV on the CPU otherwise, both
frame-exact from any start frame, and a shared source that fans one decode out to several readers.

``open_decoder(path, start_frame=...)`` is the one front door.  It answers an ``NvdecDecoder``
when PyNvVideoCodec is installed and the GPU decodes the clip, else an ``OpenCVDecoder``.  Both
follow ``cv2.VideoCapture``'s reading protocol (``read() -> (ok, frame)``, ``release()``), so
``video_io.ThreadedFrameReader`` prefetches either, and both hand out BGR frames in OpenCV's
layout (HWC uint8): a CUDA tensor from NVDEC (or a host array with ``host=True``), a host array
from OpenCV.

Frame identity.  OpenCV's FFmpeg backend seeks by timestamp, computed from the frame number and
the *average* frame rate; on a variable-frame-rate clip (phone recordings, ``sample/pomona.mp4``:
60/1 nominal, 59.67 average) a seek to frame 689 decodes frame 683 while reporting 690, and after
a seek it counts frames from that guess, so the stream can also end a few frames early (pomona
from a seek near 620: 1330 of 1332 frames).  ``OpenCVDecoder`` therefore never seeks: it decodes
from the first frame and skips to the start with ``grab`` (no colour conversion: 690 frames of
pomona in 0.3 s), on its first ``read``, so behind ``ThreadedFrameReader`` the skip runs on the
reader thread.  ``NvdecDecoder`` seeks with PyNvVideoCodec's ``start_frame`` and checks every
frame it hands out against the packets' timestamps (``_refill``).

Colour.  The detector was trained on OpenCV's pictures, so NVDEC's must be the same pixels.
H.264 decoding is bit-exact, so NVDEC's NV12 planes are libavcodec's; the difference is the
YUV -> BGR conversion.  OpenCV 4.9's FFmpeg backend (FFmpeg 6.1) converts with swscale and never
passes the stream's colour matrix, so swscale applies its default, BT.601, to every clip, full
range only for a ``yuvj`` pixel format (an H.264 or HEVC stream flagged full range).  PyNvVideoCodec's
own RGB output honours the stream's matrix: within one level on pomona (BT.601, full range) but
up to 26 levels off on the BT.709 sample clips.  ``nv12_to_bgr`` therefore redoes swscale's x86
fixed-point arithmetic (``libswscale/x86/yuv_2_rgb.asm`` with ``ff_yuv2rgb_c_init_tables``'
coefficients, nearest-neighbour chroma) on the GPU: it equals OpenCV's frames bit for bit on all
four sample clips (pomona, PomonaPitzer: BT.601 full range; Vereinsduell, VolyAvw: BT.709
limited range).  The range is calibrated on the first frame (``_opencv_range``), and a clip
whose first frame NVDEC cannot reproduce exactly is decoded by OpenCV instead.

``SharedFrameSource`` decodes once for several consumers (the ball prepass and pass 1 read the
same frames at different speeds): each ``FrameConsumer`` sees every frame in order, and the decode
stays at most ``capacity`` frames ahead of the slowest one.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    import PyNvVideoCodec as nvc
except (ImportError, OSError):          # optional (OSError: its DLLs without a driver): CPU decode
    nvc = None

# A decoded BGR frame: a CUDA tensor (NVDEC) or a host array (OpenCV, or NVDEC with ``host=True``).
Image = Union[np.ndarray, torch.Tensor]

BACKEND_NVDEC = "nvdec"
BACKEND_OPENCV = "opencv"
# Frames NVDEC's decode thread keeps ready ahead of the reader (PyNvVideoCodec recommends 8 to 16).
NVDEC_BUFFER_FRAMES = 12
_NO_PTS = -(1 << 63)          # FFmpeg's AV_NOPTS_VALUE


def nvdec_available() -> bool:
    """Whether PyNvVideoCodec is installed and a CUDA device is present."""
    return nvc is not None and torch.cuda.is_available()


def _swscale_coefficients(full_range: bool) -> Dict[str, int]:
    """swscale's 16-bit YUV -> RGB coefficients for BT.601 (``SWS_CS_DEFAULT``) at default
    contrast, saturation and brightness, as ``ff_yuv2rgb_c_init_tables`` derives them: the
    ``ff_yuv2rgb_coeffs`` row in 16.16 fixed point, scaled 255/219 (luma, limited range) or
    224/255 (chroma, full range) with C's truncating division, then ``roundToInt16(x * 2^13)``
    (``* 2^3`` for the luma offset)."""
    crv, cbu, cgu, cgv = 104597, 132201, -25675, -53279
    cy, oy = 1 << 16, 0
    if full_range:
        crv, cbu, cgu, cgv = (int(c * 224 / 255) for c in (crv, cbu, cgu, cgv))   # int() truncates as C does
    else:
        cy, oy = cy * 255 // 219, 16 << 16

    def round16(value: int) -> int:
        return (value + (1 << 15)) >> 16

    return {"y": round16(cy << 13), "y_offset": round16(oy << 3), "v2r": round16(crv << 13),
            "u2b": round16(cbu << 13), "u2g": round16(cgu << 13), "v2g": round16(cgv << 13)}


def _pmulhw(a: torch.Tensor, coefficient: int) -> torch.Tensor:
    """x86 ``pmulhw``: the high 16 bits of the 32-bit product (an arithmetic shift, rounding down)."""
    return (a * coefficient) >> 16


def nv12_to_bgr(nv12: torch.Tensor, full_range: bool) -> torch.Tensor:
    """NV12 pictures (``(..., H * 3 / 2, W)`` uint8: the Y plane over the interleaved UV plane,
    any leading batch dimensions) as BGR ``(..., H, W, 3)`` uint8, the pixels OpenCV's FFmpeg
    backend gives for them (swscale's x86 ``yuv2rgb``: samples shifted left 3, offsets taken off,
    ``pmulhw`` by the coefficients, chroma sums saturated to 16 bits, the result saturated to
    0..255; each chroma sample serves a 2x2 block)."""
    c = _swscale_coefficients(full_range)
    batch = nv12.shape[:-2]
    height = nv12.shape[-2] * 2 // 3
    width = nv12.shape[-1]
    y = nv12[..., :height, :].to(torch.int32)
    uv = nv12[..., height:, :].reshape(*batch, height // 2, width // 2, 2).to(torch.int32)
    u = (uv[..., 0] << 3) - 1024
    v = (uv[..., 1] << 3) - 1024
    luma = _pmulhw((y << 3) - c["y_offset"], c["y"]).view(*batch, height // 2, 2, width // 2, 2)
    blue = _pmulhw(u, c["u2b"])
    red = _pmulhw(v, c["v2r"])
    green = (_pmulhw(u, c["u2g"]) + _pmulhw(v, c["v2g"])).clamp_(-32768, 32767)
    bgr = torch.stack([luma + term[..., :, None, :, None] for term in (blue, green, red)], dim=-1)
    return bgr.clamp_(0, 255).to(torch.uint8).view(*batch, height, width, 3)


def _check_nvdec_caps(path, gpu_id: int) -> None:
    """Raise ``RuntimeError`` unless this GPU's NVDEC decodes the clip's codec and size as 8-bit
    4:2:0 (NV12, what ``nv12_to_bgr`` reads).  ``ThreadedDecoder`` creates the hardware decoder
    on its own thread, and a clip the hardware refuses (a 256x32 H.264 clip: NVDEC's minimum
    height is 64) aborts the process there instead of raising; the caps say so beforehand."""
    try:
        demuxer = nvc.CreateDemuxer(str(path))
        width, height = int(demuxer.Width()), int(demuxer.Height())
        codec, chroma, depth = demuxer.GetNvCodecId(), demuxer.ChromaFormat(), int(demuxer.BitDepth())
        caps = nvc.GetDecoderCaps(gpuid=gpu_id, codec=codec, chromaformat=chroma, bitdepth=depth)
    except Exception as error:  # noqa: BLE001 - PyNvVideoCodec raises its own types for every failure
        raise RuntimeError(f"NVDEC cannot read {path}: {error}") from error
    macroblocks = ((width + 15) // 16) * ((height + 15) // 16)
    if not (caps.get("supported") and caps["width_min"] <= width <= caps["width_max"]
            and caps["height_min"] <= height <= caps["height_max"] and macroblocks <= caps["mb_num_max"]):
        raise RuntimeError(f"NVDEC does not decode {path} ({codec}, {width}x{height})")
    if chroma != nvc.cudaVideoChromaFormat.__members__["420"] or depth != 8 or width % 2 or height % 2:
        raise RuntimeError(f"NVDEC decodes {path} ({chroma}, {depth}-bit, {width}x{height}) to a layout other than NV12")


def _opencv_range(path, gpu_id: int) -> bool:
    """Whether OpenCV converts this clip as full range (the range under which NVDEC's first
    picture converts to OpenCV's first picture bit for bit).

    swscale reads the range off the decoder's pixel format (``yuvj420p`` is full range), which
    libavcodec picks from the stream's VUI flag; neither PyNvVideoCodec's demuxer (``UDEF`` unless
    the container repeats the flag, as pomona's does) nor OpenCV (``CAP_PROP_CODEC_PIXEL_FORMAT``
    says ``I420`` either way) reports it, so the first frame is decoded both ways and compared.
    Raises ``RuntimeError`` when neither range reproduces OpenCV's picture.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        ok, reference = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"OpenCV decodes no frame of {path} to calibrate NVDEC against")
    try:
        # The decoder owns the frame's memory: it must outlive the copy.
        decoder = nvc.SimpleDecoder(str(path), gpu_id=gpu_id, use_device_memory=True,
                                    output_color_type=nvc.OutputColorType.NATIVE)
        first = decoder[0]
        nv12 = torch.from_dlpack(first).clone()
        torch.cuda.current_stream(nv12.device).synchronize()
        del first, decoder
    except Exception as error:  # noqa: BLE001 - PyNvVideoCodec raises its own types for every failure
        raise RuntimeError(f"NVDEC cannot decode {path}: {error}") from error
    expected = torch.from_numpy(reference).to(nv12.device)
    matches = [full for full in (False, True) if torch.equal(nv12_to_bgr(nv12, full), expected)]
    if not matches:
        raise RuntimeError(f"NVDEC's colour conversion does not reproduce OpenCV's pixels for {path}")
    return matches[-1] if len(matches) == 1 else False


def _frame_index_by_pts(path) -> Dict[int, int]:
    """Each frame's index by its timestamp: the packets' PTS in presentation order, which
    ``ThreadedDecoder``'s frame timestamps equal (``SimpleDecoder``'s sit the B-frame delay
    below them: -512 on the test clips).  Raises ``RuntimeError`` when a packet has no
    timestamp or two share one (the index would be ambiguous)."""
    demuxer = nvc.CreateDemuxer(str(path))
    stamps = []
    while True:
        packet = demuxer.Demux()
        if packet is None or packet.bsl == 0:
            break
        stamps.append(int(packet.pts))
    stamps.sort()
    if not stamps or stamps[0] == _NO_PTS or len(set(stamps)) != len(stamps):
        raise RuntimeError(f"{path} has no frame-exact packet timestamps")
    return {stamp: index for index, stamp in enumerate(stamps)}


class NvdecDecoder:
    """The clip decoded by NVDEC from frame ``start_frame``, as BGR CUDA tensors (HWC uint8).

    ``read`` returns ``(True, frame)`` for each frame in source order and ``(False, None)`` at the
    end.  The tensors are fresh allocations on ``device``, written on this decoder's own CUDA
    stream and safe to keep: ``read`` makes the caller's current stream wait for them.  With
    ``host=True`` each frame is downloaded instead and ``read`` hands out a host array (pass 2
    draws on it with OpenCV).  Raises ``RuntimeError`` when NVDEC cannot open the clip.
    """

    backend = BACKEND_NVDEC

    def __init__(self, path, device: Optional[torch.device] = None, start_frame: int = 0, host: bool = False,
                 buffer_frames: int = NVDEC_BUFFER_FRAMES) -> None:
        if not nvdec_available():
            raise RuntimeError("NVDEC needs PyNvVideoCodec and a CUDA device")
        self.device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        self.host = bool(host)
        self.next_index = int(start_frame)
        self._buffer = max(1, int(buffer_frames))
        self._pending: List[torch.Tensor] = []
        self._path = str(path)
        self._gpu_id = self.device.index or 0
        self._stream = torch.cuda.Stream(device=self.device)
        self._ready = torch.cuda.Event()          # the pending batch's conversion (and download) done
        _check_nvdec_caps(path, self._gpu_id)
        with torch.cuda.device(self.device):
            self.full_range = _opencv_range(path, self._gpu_id)
        # From frame 0 the decode order is the frame order (frames are counted); a later start is
        # checked frame by frame against the packets' timestamps.
        self._index_of = _frame_index_by_pts(path) if start_frame > 0 else None
        self._counted = 0                 # frames the decoder gave since it opened at frame 0
        self.restarted = False
        self._decoder = self._open(int(start_frame))

    def _open(self, start_frame: int):
        try:
            return nvc.ThreadedDecoder(self._path, buffer_size=self._buffer, gpu_id=self._gpu_id,
                                       use_device_memory=True, output_color_type=nvc.OutputColorType.NATIVE,
                                       start_frame=start_frame)
        except Exception as error:  # noqa: BLE001 - PyNvVideoCodec raises its own types for every failure
            raise RuntimeError(f"NVDEC cannot decode {self._path}: {error}") from error

    def _restart(self) -> None:
        """Decode again from frame 0, counting frames up to the start."""
        self.restarted = True
        self._index_of = None
        self._counted = 0
        self._decoder = self._open(0)

    def _refill(self) -> bool:
        """Convert the decoder's next batch out of its buffers, dropping the frames before the start.

        NVDEC's ``start_frame`` lands on the frame asked for on H.264 (pomona, the VFR test clips)
        but one to three frames late on an x265 clip with open GOPs (start 37 gave frame 40), so a
        later start is judged by each frame's timestamp, whose index the packets give; a start
        past the frame asked for, or a timestamp no packet has, decodes again from frame 0.  The
        buffers stay valid only until the next ``get_batch_frames``, so the last batch's
        conversions must have run before it."""
        self._stream.synchronize()
        while True:
            frames = self._decoder.get_batch_frames(self._buffer)
            if not frames:
                return False
            keep: List[Any] = []
            for frame in frames:
                expected = self.next_index + len(keep)
                if self._index_of is None:
                    index, self._counted = self._counted, self._counted + 1
                else:
                    index = self._index_of.get(int(frame.timestamp), -1)
                if index < 0 or (index > expected and not keep):
                    if self.restarted:
                        raise RuntimeError(f"NVDEC lost frame order in {self._path}")
                    self._restart()
                    break
                if index < self.next_index:
                    continue                          # before the start: decoded to reach it
                if index != expected:
                    raise RuntimeError(f"NVDEC skipped from frame {expected} to {index} of {self._path}")
                keep.append(frame)
            else:
                if keep:
                    self._convert(keep)
                    return True

    def _convert(self, frames: List[Any]) -> None:
        """Convert a batch in one pass (a dozen kernels for the batch, not per frame: the reader
        thread holds the GIL while it launches them) and, with ``host``, download it in one
        asynchronous copy to pinned memory; ``_ready`` marks when the batch can be read."""
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            nv12 = torch.stack([torch.from_dlpack(frame) for frame in frames])
            bgr = nv12_to_bgr(nv12, self.full_range)
            if self.host:
                pinned = torch.empty(bgr.shape, dtype=torch.uint8, pin_memory=True)
                pinned.copy_(bgr, non_blocking=True)
                bgr = pinned
            self._ready = torch.cuda.Event()
            self._ready.record(self._stream)
        self._pending = list(bgr.unbind(0))

    def read(self) -> Tuple[bool, Optional[Image]]:
        if not self._pending and not self._refill():
            return False, None
        frame = self._pending.pop(0)
        self.next_index += 1
        if self.host:
            self._ready.synchronize()
            return True, frame.numpy()
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self._ready)
        frame.record_stream(current)
        return True, frame

    def release(self) -> None:
        self._pending = []
        self._decoder = None


class OpenCVDecoder:
    """The clip decoded by OpenCV on the CPU from frame ``start_frame``, as BGR host arrays.

    The frames before the start are decoded and dropped on the first ``read`` (see the module
    docstring for why OpenCV's seek cannot be trusted).  Raises ``RuntimeError`` when OpenCV
    cannot open the clip.
    """

    backend = BACKEND_OPENCV

    def __init__(self, path, start_frame: int = 0) -> None:
        self._path = str(path)
        self.next_index = int(start_frame)
        self._skip = int(start_frame)
        self._capture = cv2.VideoCapture(self._path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open the video: {self._path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        while self._skip:
            if not self._capture.grab():
                return False, None
            self._skip -= 1
        ok, frame = self._capture.read()
        if ok:
            self.next_index += 1
        return ok, frame

    def release(self) -> None:
        self._capture.release()


Decoder = Union[NvdecDecoder, OpenCVDecoder]


def open_decoder(path, device: Optional[torch.device] = None, start_frame: int = 0, host: bool = False,
                 prefer_nvdec: bool = True) -> Decoder:
    """NVDEC when ``prefer_nvdec`` and it can open the clip, else OpenCV (see the two classes).

    ``host`` asks NVDEC for host arrays; OpenCV always gives them.  The fallback is reported on
    the console.
    """
    if prefer_nvdec and nvdec_available():
        try:
            return NvdecDecoder(path, device=device, start_frame=start_frame, host=host)
        except RuntimeError as error:
            print(f"[decode] {str(error).splitlines()[0]}; decoding on the CPU")
    return OpenCVDecoder(path, start_frame=start_frame)


# ---------------------------------------------------------------------------
# One decode, several consumers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """One decoded frame: its zero-based source index and its BGR picture."""

    index: int
    image: Image


class SharedFrameSource:
    """One decode of a clip that several consumers read, each every frame in source order.

    Register each consumer with ``consumer(name)`` before ``start()``; the decode then runs on
    its own thread, at most ``capacity`` frames ahead of the slowest consumer still reading.  A
    frame is dropped once every consumer has read past it; a consumer keeps the frames it holds
    for as long as it likes (CUDA tensors are fresh allocations, not ring slots).  A consumer that
    stops early must ``close()`` so it no longer holds the others back.  A decode failure is
    raised by each consumer's ``read`` when it reaches the frame that was not decoded.

    ``capacity`` must cover how far one consumer can be ahead of another that it waits on (the
    prepass reads a unit and a batch past the frame pass 1 waits for: a few dozen frames), or the
    two wait on each other forever.  At 1080p a frame is 6.2 MB, 32 frames 200 MB.
    """

    def __init__(self, decoder: Decoder, capacity: int = 32) -> None:
        self.decoder = decoder
        self.capacity = max(1, int(capacity))
        self._cond = threading.Condition()
        self._frames: Dict[int, Tuple[Image, Optional[torch.cuda.Event]]] = {}
        self._positions: Dict[str, int] = {}
        self._first_index = int(getattr(decoder, "next_index", 0))
        self._produced = self._first_index
        self._done = False
        self._stopped = False
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def backend(self) -> str:
        return self.decoder.backend

    def consumer(self, name: str) -> "FrameConsumer":
        with self._cond:
            if self._thread is not None:
                raise RuntimeError("register every consumer before start()")
            if name in self._positions:
                raise ValueError(f"consumer {name!r} is already registered")
            self._positions[name] = self._first_index
        return FrameConsumer(self, name)

    def start(self) -> None:
        if not self._positions:
            raise RuntimeError("SharedFrameSource has no consumers")
        self._thread = threading.Thread(target=self._decode_loop, daemon=True, name="shared-frame-source")
        self._thread.start()

    def _decode_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._stopped and self._positions and \
                            self._produced - min(self._positions.values()) >= self.capacity:
                        self._cond.wait()
                    if self._stopped or not self._positions:
                        return
                ok, image = self.decoder.read()
                if not ok or image is None:
                    return
                ready = None
                if isinstance(image, torch.Tensor) and image.is_cuda:
                    # A consumer on another stream (the prepass worker's) waits for this event.
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(image.device))
                with self._cond:
                    self._frames[self._produced] = (image, ready)
                    self._produced += 1
                    self._cond.notify_all()
        except BaseException as error:  # noqa: BLE001 - handed to every consumer's read()
            self._error = error
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _read(self, name: str) -> Optional[Frame]:
        with self._cond:
            index = self._positions[name]
            while index >= self._produced and not self._done:
                self._cond.wait()
            if index >= self._produced:
                if self._error is not None:
                    raise RuntimeError(f"decoding frame {index} failed") from self._error
                return None
            image, ready = self._frames[index]
            self._positions[name] = index + 1
            self._prune()
            self._cond.notify_all()
        if ready is not None:
            assert isinstance(image, torch.Tensor)
            current = torch.cuda.current_stream(image.device)
            current.wait_event(ready)
            image.record_stream(current)
        return Frame(index, image)

    def _close_consumer(self, name: str) -> None:
        with self._cond:
            self._positions.pop(name, None)
            self._prune()
            self._cond.notify_all()

    def _prune(self) -> None:
        """Drop the frames every remaining consumer has read past (all of them with none left)."""
        floor = min(self._positions.values(), default=self._produced)
        for index in [i for i in self._frames if i < floor]:
            del self._frames[index]

    def close(self) -> None:
        """Stop the decode, wait for its thread and release the decoder."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join()
        self._frames.clear()
        self.decoder.release()


class FrameConsumer:
    """One reader of a ``SharedFrameSource``: ``read()`` (or iteration) gives every frame in order
    and None at the end."""

    def __init__(self, source: SharedFrameSource, name: str) -> None:
        self.source = source
        self.name = name
        self._closed = False

    def read(self) -> Optional[Frame]:
        if self._closed:
            return None
        return self.source._read(self.name)

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.source._close_consumer(self.name)
