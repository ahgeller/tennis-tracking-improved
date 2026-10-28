"""tennis_tracker/video_decode.py: the shared frame source on a fake decoder, OpenCV's start-frame
search, and (with FFmpeg, PyNvVideoCodec and a GPU) NVDEC against OpenCV on real H.264 clips."""
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
import torch

from tennis_tracker import video_decode
from tennis_tracker.video_decode import (
    OpenCVDecoder, SharedFrameSource, nvdec_available, open_decoder,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to encode the clip")
needs_nvdec = pytest.mark.skipif(not nvdec_available(), reason="needs PyNvVideoCodec and a CUDA device")


class FakeDecoder:
    """A decoder whose frame ``i`` is a 2x2 picture of value ``i``; ``fail_at`` raises there."""

    backend = "fake"

    def __init__(self, count: int, fail_at: Optional[int] = None) -> None:
        self.count, self.fail_at = count, fail_at
        self.next_index = 0
        self.released = False

    def read(self):
        if self.next_index == self.fail_at:
            raise ValueError("corrupt packet")
        if self.next_index >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.next_index, np.uint8)
        self.next_index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _values(consumer) -> List[int]:
    return [int(frame.image[0, 0, 0]) for frame in consumer]


def _read_in_thread(consumer):
    """Start ``consumer.read()`` on a thread; the thread and the list its result lands in."""
    out: list = []
    thread = threading.Thread(target=lambda: out.append(consumer.read()), daemon=True)
    thread.start()
    return thread, out


# --------------------------------------------------------------------------- SharedFrameSource

def test_every_consumer_reads_every_frame_in_order():
    source = SharedFrameSource(FakeDecoder(50), capacity=4)
    consumers = [source.consumer(name) for name in ("prepass", "pass1")]
    source.start()
    results = {}
    threads = [threading.Thread(target=lambda c=c: results.setdefault(c.name, _values(c))) for c in consumers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    source.close()
    assert results == {"prepass": list(range(50)), "pass1": list(range(50))}
    assert source.decoder.released


def test_the_decode_stays_within_capacity_of_the_slowest_consumer():
    source = SharedFrameSource(FakeDecoder(50), capacity=3)
    fast, slow = source.consumer("fast"), source.consumer("slow")
    source.start()
    assert [fast.read().index for _ in range(3)] == [0, 1, 2]
    thread, out = _read_in_thread(fast)
    thread.join(timeout=0.3)
    assert thread.is_alive()                        # frame 3 waits for the slow consumer to move
    assert len(source._frames) <= 3
    assert slow.read().index == 0
    thread.join(timeout=5)
    assert out[0].index == 3
    source.close()


def test_a_consumer_that_closes_no_longer_holds_the_others_back():
    source = SharedFrameSource(FakeDecoder(20), capacity=2)
    reader, quitter = source.consumer("reader"), source.consumer("quitter")
    source.start()
    assert quitter.read().index == 0
    quitter.close()
    assert quitter.read() is None
    assert _values(reader) == list(range(20))
    assert source._frames == {}                     # with nobody left behind nothing is kept
    source.close()


def test_a_decode_failure_reaches_each_consumer_after_the_frames_that_decoded():
    source = SharedFrameSource(FakeDecoder(20, fail_at=5), capacity=8)
    first, second = source.consumer("first"), source.consumer("second")
    source.start()
    for consumer in (first, second):
        assert [consumer.read().index for _ in range(5)] == [0, 1, 2, 3, 4]
        with pytest.raises(RuntimeError, match="decoding frame 5 failed") as raised:
            consumer.read()
        assert isinstance(raised.value.__cause__, ValueError)
    source.close()


def test_consumers_are_registered_once_and_before_the_start():
    source = SharedFrameSource(FakeDecoder(5))
    with pytest.raises(RuntimeError, match="no consumers"):
        source.start()
    source.consumer("a")
    with pytest.raises(ValueError, match="already registered"):
        source.consumer("a")
    source.start()
    with pytest.raises(RuntimeError, match="before start"):
        source.consumer("b")
    source.close()


def test_close_stops_a_decode_that_waits_for_a_consumer():
    source = SharedFrameSource(FakeDecoder(10 ** 6), capacity=2)
    source.consumer("idle")
    source.start()
    time.sleep(0.05)
    source.close()                                  # returns: the waiting decode thread is woken and stops
    assert not source._thread.is_alive()


# --------------------------------------------------------------------------- OpenCV start frame

# --------------------------------------------------------------------------- real clips

def _encode(path: Path, frames: List[np.ndarray], fps: int = 60, extra=(), codec: str = "libx264",
            pix_fmt: str = "yuv420p") -> None:
    height, width = frames[0].shape[:2]
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", *extra, "-c:v", codec, "-g", "30", "-bf", "3", "-pix_fmt", pix_fmt,
            "-crf", "4", str(path)]
    encoder = subprocess.Popen(args, stdin=subprocess.PIPE)
    for frame in frames:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0


def _coloured_frames(count: int) -> List[np.ndarray]:
    """Blue, green and red thirds (BGR) over a moving grey ramp: a channel swap or a frame shift shows."""
    frames = []
    for index in range(count):
        frame = np.zeros((160, 256, 3), np.uint8)
        frame[:, :80, 0] = 220
        frame[:, 80:160, 1] = 220
        frame[:, 160:, 2] = 220
        frame[96:, :] = ((np.arange(256) * 2 + index * 5) % 256).astype(np.uint8)[None, :, None]
        frames.append(frame)
    return frames


def _opencv_frames(path: Path) -> List[np.ndarray]:
    decoder = OpenCVDecoder(path)
    frames = []
    while True:
        ok, frame = decoder.read()
        if not ok:
            break
        frames.append(frame)
    decoder.release()
    return frames


@needs_ffmpeg
def test_opencv_starts_at_the_frame_asked_for_and_reads_to_the_end(tmp_path):
    """OpenCV decodes from the first frame and drops the frames before the start, so every start
    yields exactly the sequential decode's frames from there on (a seek can end the stream early)."""
    path = tmp_path / "clip.mp4"
    _encode(path, _coloured_frames(90))
    reference = _opencv_frames(path)
    for start in (0, 1, 45, 89, 90):
        decoder = OpenCVDecoder(path, start_frame=start)
        frames = []
        while (read := decoder.read())[0]:
            frames.append(read[1])
        decoder.release()
        assert len(frames) == 90 - start and decoder.next_index == 90
        assert all(np.array_equal(frame, ref) for frame, ref in zip(frames, reference[start:]))


@needs_ffmpeg
@needs_nvdec
@pytest.mark.parametrize("codec, pix_fmt, colours", [
    ("libx264", "yuv420p", ("bt709",)),           # limited range, tagged BT.709: OpenCV still applies BT.601
    ("libx264", "yuvj420p", ("bt470bg",)),        # full range, as phone clips (pomona) are
    ("libx265", "yuv420p", ("bt709",)),
    ("libx265", "yuvj420p", ("bt470bg",)),
], ids=["h264", "h264-full", "hevc", "hevc-full"])
def test_nvdec_frames_equal_opencvs_from_any_start(tmp_path, codec, pix_fmt, colours):
    """Every frame from every start is OpenCV's picture bit for bit, on a variable-frame-rate clip."""
    if codec == "libx265" and "libx265" not in subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                                              capture_output=True, text=True).stdout:
        pytest.skip("this ffmpeg has no libx265")
    path = tmp_path / "clip.mp4"
    _encode(path, _coloured_frames(90), codec=codec, pix_fmt=pix_fmt,
            extra=("-vf", "setpts='if(lt(N,40),N,N+N/3)/60/TB'", "-fps_mode", "vfr", "-colorspace", colours[0]))
    reference = _opencv_frames(path)
    assert len(reference) == 90
    for start in (0, 37, 61):
        decoder = open_decoder(path, start_frame=start, host=True)
        assert decoder.backend == "nvdec"
        frames = []
        while True:
            ok, frame = decoder.read()
            if not ok:
                break
            frames.append(frame)
        decoder.release()
        assert len(frames) == 90 - start
        for offset, frame in enumerate(frames):
            diff = np.abs(frame.astype(np.int16) - reference[start + offset].astype(np.int16))
            assert diff.max() == 0, f"start {start}, frame {start + offset}: off by up to {diff.max()}"


@needs_ffmpeg
@needs_nvdec
def test_nvdec_hands_out_nv12_pictures_that_convert_to_opencvs(tmp_path):
    """``nv12=True`` skips the conversion: each picture, converted by ``nv12_to_bgr`` with the
    decoder's calibrated range, is OpenCV's frame, and the count is exact."""
    path = tmp_path / "clip.mp4"
    _encode(path, _coloured_frames(30), pix_fmt="yuvj420p")
    reference = _opencv_frames(path)
    decoder = video_decode.NvdecDecoder(path, nv12=True)
    pictures = []
    while (read := decoder.read())[0]:
        pictures.append(read[1])
    decoder.release()
    assert len(pictures) == len(reference) == 30
    for picture, ref in zip(pictures, reference):
        assert isinstance(picture, video_decode.Nv12Frame) and picture.full_range
        assert (picture.width, picture.height) == (256, 160) and picture.planes.is_cuda
        assert np.array_equal(video_decode.nv12_to_bgr(picture.planes, picture.full_range).cpu().numpy(), ref)
    with pytest.raises(ValueError, match="on the device only"):
        video_decode.NvdecDecoder(path, host=True, nv12=True)


@needs_ffmpeg
@needs_nvdec
def test_a_shared_nvdec_source_feeds_a_consumer_on_another_stream(tmp_path):
    path = tmp_path / "clip.mp4"
    _encode(path, _coloured_frames(40))
    reference = _opencv_frames(path)
    source = SharedFrameSource(open_decoder(path), capacity=6)
    main, side = source.consumer("main"), source.consumer("side")
    source.start()
    seen = {}

    def on_side_stream():
        with torch.cuda.stream(torch.cuda.Stream()):
            seen["side"] = [frame.image.float().sum().item() for frame in side]

    thread = threading.Thread(target=on_side_stream)
    thread.start()
    images = [frame.image for frame in main]
    thread.join(timeout=30)
    source.close()
    assert all(image.is_cuda and image.dtype == torch.uint8 for image in images)
    assert [image.float().sum().item() for image in images] == seen["side"]
    assert all(np.array_equal(image.cpu().numpy(), ref) for image, ref in zip(images, reference))


@needs_ffmpeg
@needs_nvdec
def test_a_clip_nvdec_cannot_decode_falls_back_to_opencv(tmp_path, capsys):
    """NVDEC's minimum height is 64; its threaded decoder would abort the process on a 32 px clip,
    so the caps are checked first and the clip is decoded on the CPU."""
    path = tmp_path / "short.mp4"
    _encode(path, [frame[:32] for frame in _coloured_frames(5)])
    decoder = open_decoder(path, host=True)
    assert decoder.backend == "opencv" and decoder.read()[0]
    decoder.release()
    assert "[decode] NVDEC does not decode" in capsys.readouterr().out


def test_without_pyNvVideoCodec_the_clip_decodes_on_the_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(video_decode, "nvc", None)
    assert not nvdec_available()
    with pytest.raises(RuntimeError, match="Could not open the video"):
        open_decoder(tmp_path / "missing.mp4")
