from __future__ import annotations

import contextlib
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np
import torch

from .utils import (
    ENCODER_LIBX264,
    ENCODER_NVENC,
    ENCODER_OPENCV,
    FFMPEG_ENCODER_NAMES,
    EncoderChoice,
    EncoderSettings,
    choose_encoder,
)


class WriterSettings(EncoderSettings, Protocol):
    """The slice of ``Config`` the video writer reads: the NVENC options, and
    whether frames are queued to a drain thread (and how many)."""

    nvenc_preset: str
    nvenc_bitrate: str
    use_async_writer: bool
    async_queue: int

LIBX264_PRESET = "fast"
LIBX264_CRF = "18"
_OPENCV_FOURCC = "mp4v"
# How long close() lets FFmpeg flush its last frames, and how long a kill may take.
_FFMPEG_CLOSE_TIMEOUT_S = 30.0
_FFMPEG_KILL_TIMEOUT_S = 5.0
_QUEUE_POLL_S = 0.1
# The containers an output video may use: FFmpeg picks the muxer from the suffix
# and refuses any other ("Unable to choose an output format"), and OpenCV's
# fallback writes each of these too.
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi")


def check_writable_video_path(path: Union[str, Path], label: str = "Video output") -> None:
    """Raise ``ValueError`` when ``path`` cannot become a video file.

    FFmpeg only sees the path at the first frame write, after pass 1 and the
    selector, so the run checks its outputs here before any frame is decoded:
    the path must not be an existing directory, its suffix must be one of
    ``VIDEO_SUFFIXES``, and its nearest existing ancestor must be a directory.
    A missing parent folder passes: pass 1 and pass 2 create it (``os.makedirs``)
    before they open a writer.  ``label`` names the output in the message.
    """
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        raise ValueError(f"{label} {target} is a directory; name a file in it, such as "
                         f"{target / ('tracking' + VIDEO_SUFFIXES[0])}")
    if target.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"{label} {target} needs a video container suffix FFmpeg can write "
                         f"({', '.join(VIDEO_SUFFIXES)})")
    nearest = next((ancestor for ancestor in target.parents if ancestor.exists()), None)
    if nearest is None or not nearest.is_dir():
        raise ValueError(f"{label} {target} cannot be written: {nearest or target.anchor} is not a directory")


_UPLOAD_SLOTS = 3
# OpenCV's luma weights in BGR order: the gray plane the temporal motion gate compares.
BGR_TO_GRAY = (0.114, 0.587, 0.299)


class PinnedFrameUploader:
    """Uploads decoded BGR frames through pinned host buffers.

    The frame goes to the GPU as it was decoded, HWC uint8: a quarter of the
    bytes of the float32 frame the GPU works on, which ``hwc_u8_to_chw_f32``
    then makes there.  The copy is asynchronous (``non_blocking`` from pinned
    memory), so the host only waits when it is about to refill a pinned
    buffer whose copy, ``_UPLOAD_SLOTS`` uploads ago, has not finished.
    Converting on the host first, as ``.to(device, dtype=float32)`` from a
    uint8 array does, cost 0.7 ms a 1080p frame on the CPU and turned the
    copy into a synchronous one from pageable memory (the review's profile).
    """

    def __init__(self, h: int, w: int, device: torch.device, slots: int = _UPLOAD_SLOTS):
        self.h = int(h)
        self.w = int(w)
        self.device = device
        self._slot = 0
        self._pinned_u8 = [
            torch.empty((self.h, self.w, 3), dtype=torch.uint8, pin_memory=True)
            for _ in range(max(1, int(slots)))
        ]
        self._pinned_np = [t.numpy() for t in self._pinned_u8]
        self._copy_done: List[Optional[torch.cuda.Event]] = [None] * len(self._pinned_u8)

    @torch.no_grad()
    def upload_hwc_u8(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """``frame_bgr`` as an HWC uint8 tensor on the device; the copy may still be in flight
        (it is ordered before any later work on the current stream)."""
        if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w or frame_bgr.shape[2] != 3:
            return torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device=self.device)
        t = torch.empty((self.h, self.w, 3), dtype=torch.uint8, device=self.device)
        self.upload_into(frame_bgr, t)
        return t

    @torch.no_grad()
    def upload_into(self, frame_bgr: np.ndarray, target: torch.Tensor) -> None:
        """Copy ``frame_bgr`` (this uploader's size) into the HWC uint8 device tensor ``target``
        through a pinned buffer; the copy is ordered on the current stream and may still be in flight."""
        slot = self._slot
        self._slot = (self._slot + 1) % len(self._pinned_u8)
        done = self._copy_done[slot]
        if done is not None:
            done.synchronize()
        np.copyto(self._pinned_np[slot], frame_bgr)
        target.copy_(self._pinned_u8[slot], non_blocking=True)
        done = torch.cuda.Event(blocking=False)
        done.record(torch.cuda.current_stream(device=self.device))
        self._copy_done[slot] = done

    def upload_chw_f32(self, frame_bgr: np.ndarray) -> torch.Tensor:
        return hwc_u8_to_chw_f32(self.upload_hwc_u8(frame_bgr))


@torch.no_grad()
def hwc_u8_to_chw_f32(frame_u8: torch.Tensor) -> torch.Tensor:
    """An HWC uint8 frame as a contiguous CHW float32 tensor in [0, 1] on its own device.

    One kernel converts and transposes; ``div_(255.0)`` is the same division the old
    host-converted path did (``/ 255.0`` on the device), so the values are bit-identical.
    """
    h, w, c = (int(n) for n in frame_u8.shape)
    out = torch.empty((c, h, w), dtype=torch.float32, device=frame_u8.device)
    out.copy_(frame_u8.permute(2, 0, 1))
    return out.div_(255.0)


@torch.no_grad()
def cuda_frame_to_chw_f32(frame_bgr: np.ndarray, device: torch.device,
                           uploader: Optional[PinnedFrameUploader] = None) -> torch.Tensor:
    """Upload one BGR frame once as CHW float32 in [0,1] on CUDA."""
    if uploader is not None:
        return uploader.upload_chw_f32(frame_bgr)
    return hwc_u8_to_chw_f32(torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device=device))


@torch.no_grad()
def cuda_frame_planes(chw: torch.Tensor) -> torch.Tensor:
    """The motion planes of a CHW BGR float frame in [0, 1]: one ``(3, H, W)`` tensor holding
    gray (OpenCV's luma weights), HSV value and HSV saturation, in that order.

    The arithmetic is the one the motion code always used, op for op (gray as
    ``b*wb + g*wg + r*wr``, saturation as ``(max - min) / (max + 1e-6)`` where
    ``max > 1e-6``), so every value is bit-identical; computing the gray plane
    once per frame here replaces four recomputations in the temporal gate.
    """
    planes = torch.empty((3,) + tuple(chw.shape[1:]), dtype=chw.dtype, device=chw.device)
    gray, maxc, sat = planes[0], planes[1], planes[2]
    wb, wg, wr = BGR_TO_GRAY
    torch.mul(chw[0], wb, out=gray)
    gray.add_(chw[1] * wg)
    gray.add_(chw[2] * wr)
    torch.amax(chw, dim=0, out=maxc)
    minc = torch.amin(chw, dim=0)
    torch.div(maxc - minc, maxc + 1e-6, out=sat)
    sat.masked_fill_(maxc <= 1e-6, 0.0)
    return planes


@torch.no_grad()
def cuda_vs_tensors(frame_bgr: Optional[np.ndarray], device: torch.device,
                     gpu_tensor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract V and S channels as CUDA tensors (no grad tracking).
    If gpu_tensor is provided, reuse it instead of uploading frame again."""
    if gpu_tensor is not None:
        t = gpu_tensor
    else:
        if frame_bgr is None:
            raise ValueError("frame_bgr is required when gpu_tensor is None")
        t = cuda_frame_to_chw_f32(frame_bgr, device)
    planes = cuda_frame_planes(t)
    return planes[1], planes[2]


def ffmpeg_command(ffmpeg: str, encoder: str, w: int, h: int, fps: float, path: str,
                   cfg: WriterSettings) -> List[str]:
    """The FFmpeg invocation that encodes raw BGR frames from stdin into ``path``.

    ``encoder`` is ``ENCODER_NVENC`` or ``ENCODER_LIBX264``; the codec options
    are the only part that differs.
    """
    if encoder == ENCODER_NVENC:
        codec = ["-c:v", FFMPEG_ENCODER_NAMES[ENCODER_NVENC], "-preset", str(cfg.nvenc_preset),
                 "-b:v", str(cfg.nvenc_bitrate)]
    elif encoder == ENCODER_LIBX264:
        codec = ["-c:v", FFMPEG_ENCODER_NAMES[ENCODER_LIBX264], "-preset", LIBX264_PRESET, "-crf", LIBX264_CRF]
    else:
        raise ValueError(f"not an FFmpeg encoder: {encoder!r}")
    return [
        str(ffmpeg), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{int(w)}x{int(h)}", "-r", str(fps), "-i", "-",
        *codec,
        "-pix_fmt", "yuv420p", str(path),
    ]


class VideoWriter:
    """Encode BGR frames to ``path`` through FFmpeg (NVENC, then libx264) or OpenCV.

    The encoder order comes from an ``EncoderChoice`` (see
    ``utils.resolve_runtime``); when none is passed it is derived from
    ``cfg`` in the same way.  Each FFmpeg encoder that fails to start is
    reported and the next one is tried; OpenCV is the last resort and raising
    when it cannot open the output.  With ``cfg.use_async_writer`` frames are
    queued to a drain thread; an encoder failure there surfaces on the next
    ``write`` or on ``close``.
    """

    def __init__(self, path: str, fps: float, w: int, h: int, cfg: WriterSettings,
                 encoder: Optional[EncoderChoice] = None):
        self._proc: Optional[subprocess.Popen] = None
        self._cv: Optional[cv2.VideoWriter] = None
        self._encoder = ENCODER_OPENCV
        self._thread_error: Optional[BaseException] = None
        self._path = str(path)
        choice = encoder if encoder is not None else choose_encoder(cfg)

        for name in choice.ffmpeg_encoders:
            proc = self._start_ffmpeg(name, ffmpeg_command(choice.ffmpeg, name, w, h, fps, self._path, cfg))
            if proc is not None:
                self._proc, self._encoder = proc, name
                break

        if self._proc is None:
            self._cv = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*_OPENCV_FOURCC), fps, (int(w), int(h)))
            if not self._cv.isOpened():
                self._cv.release()
                raise RuntimeError(f"Could not open video output: {path}")

        self._q: Optional[queue.Queue] = None
        if cfg.use_async_writer:
            self._q = queue.Queue(maxsize=cfg.async_queue)
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()

    @property
    def encoder(self) -> str:
        return self._encoder

    def _start_ffmpeg(self, name: str, cmd: List[str]) -> Optional[subprocess.Popen]:
        """Launch one FFmpeg encoder; None (after a message) when it cannot start."""
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        except OSError as error:
            print(f"[writer] {name} init failed; falling back ({error})")
            return None
        if proc.poll() is None:
            return proc
        err = _read_stderr(proc)
        print(f"[writer] {name} init failed; falling back ({err or 'ffmpeg exited'})")
        _close_pipes(proc)
        return None

    def _kill_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_FFMPEG_KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"[writer] ffmpeg ({self._encoder}) did not exit after kill")
        _close_pipes(proc)

    def _raise_proc_write_error(self, prefix: str) -> None:
        rc = self._proc.poll() if self._proc else None
        err = _read_stderr(self._proc) if self._proc else ""
        details = f"{prefix}. ffmpeg encoder='{self._encoder}'"
        if rc is not None:
            details += f", returncode={rc}"
        if err:
            details += f", stderr={err}"
        raise RuntimeError(details)

    def _raw_write(self, frame: np.ndarray) -> None:
        if self._proc:
            if self._proc.poll() is not None:
                self._raise_proc_write_error("FFmpeg process exited before frame write")
            try:
                # The array's own buffer goes to the pipe; tobytes() copied the 6 MB frame first.
                self._proc.stdin.write(np.ascontiguousarray(frame).data)
            except OSError:
                self._raise_proc_write_error("Failed writing frame bytes to FFmpeg")
        else:
            self._cv.write(frame)

    def _drain(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                self._raw_write(item)
        except Exception as e:  # noqa: BLE001 - any failure must reach the producer thread
            self._thread_error = e

    def _enqueue(self, item: Optional[np.ndarray]) -> None:
        while True:
            if self._thread_error is not None:
                raise RuntimeError(f"Async video writer failed: {self._thread_error}") from self._thread_error
            if not self._thread.is_alive():
                raise RuntimeError("Async video writer stopped before accepting output")
            try:
                self._q.put(item, timeout=_QUEUE_POLL_S)
                return
            except queue.Full:
                continue

    def write(self, frame: np.ndarray) -> None:
        if self._q:
            self._enqueue(frame)
        else:
            self._raw_write(frame)

    def close(self) -> None:
        try:
            if self._q:
                self._enqueue(None)
                self._thread.join()
                if self._thread_error is not None:
                    raise RuntimeError(f"Async video writer failed: {self._thread_error}") from self._thread_error
            if self._proc:
                if self._proc.stdin:
                    try:
                        self._proc.stdin.close()
                    except OSError:
                        self._raise_proc_write_error("Failed flushing the last frames to FFmpeg")
                try:
                    self._proc.wait(timeout=_FFMPEG_CLOSE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self._raise_proc_write_error(f"FFmpeg did not finish within {_FFMPEG_CLOSE_TIMEOUT_S:.0f}s")
                if self._proc.returncode not in (0, None):
                    self._raise_proc_write_error("FFmpeg exited with a non-zero status")
        finally:
            if self._proc:
                if self._proc.poll() is None:
                    self._kill_proc()
                else:
                    _close_pipes(self._proc)
                    self._proc = None
            if self._cv:
                self._cv.release()
                self._cv = None


def _read_stderr(proc: subprocess.Popen) -> str:
    if not proc.stderr:
        return ""
    try:
        data = proc.stderr.read()
    except (OSError, ValueError):  # pipe broken or already closed
        return ""
    return data.decode("utf-8", errors="replace").strip() if data else ""


def _close_pipes(proc: subprocess.Popen) -> None:
    for pipe in (proc.stdin, proc.stderr):
        if pipe is None:
            continue
        # Closing stdin flushes to a process that may already be gone.
        with contextlib.suppress(OSError):
            pipe.close()


class FrameCapture(Protocol):
    """What ``ThreadedFrameReader`` reads from: ``cv2.VideoCapture`` or a ``video_decode`` decoder."""

    def read(self) -> Tuple[bool, Any]: ...

    def release(self) -> None: ...


class ThreadedFrameReader:
    """Prefetch frames in a background thread to overlap decode with GPU work.

    ``read`` returns the next frame, None at the end of the clip, and raises
    ``RuntimeError`` from the decoder's exception if the reader thread died:
    the failure is handed over, not left as a read that never returns.
    ``release`` may be called before the clip is exhausted (pass 2 stops at
    the frame count pass 1 saw): it stops the thread and waits for it before
    releasing the capture, since releasing a capture a thread is still reading
    from aborts the process.
    """

    def __init__(self, cap: FrameCapture, prefetch: int = 4):
        self._cap = cap
        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._done = False
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _put(self, item: Optional[np.ndarray]) -> bool:
        """Queue ``item`` for the consumer; False when the reader was stopped while the queue
        stayed full (the consumer is gone and will not drain it)."""
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=_QUEUE_POLL_S)
                return True
            except queue.Full:
                continue
        return False

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                ret, frame = self._cap.read()
                if not ret or not self._put(frame):
                    break
        except BaseException as error:  # noqa: BLE001 - handed to read() on the consumer thread
            self._error = error
        finally:
            self._put(None)

    def read(self) -> Optional[np.ndarray]:
        if self._done:
            return None
        frame = self._q.get()
        if frame is None:
            self._done = True
            if self._error is not None:
                raise RuntimeError("frame reader failed") from self._error
        return frame

    def release(self) -> None:
        """Stop the reader thread, wait for it, then release the capture."""
        self._done = True
        self._stop.set()
        self._thread.join()
        self._cap.release()
