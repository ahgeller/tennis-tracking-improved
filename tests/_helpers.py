"""Synthetic fixtures shared by the regression tests. No match media or labels."""
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tennis_tracker.engines import ball_detectors
from tennis_tracker.engines.ball_detectors import GridTrackNetBallDetector
from tennis_tracker.config import Config
from tennis_tracker.gridtracknet import GridTrackNet
from tennis_tracker.motion import _PROBE_S_DIFF_SCALE, _clean_raw_motion_mask
from tennis_tracker.utils import ENCODER_OPENCV, EncoderChoice
from tennis_tracker.video_io import VideoWriter


def write_frames(path: Path, frames: Sequence[np.ndarray], fps: float = 30.0) -> Path:
    """Encode `frames` losslessly (FFV1 in AVI), so what a test measures is the synthetic picture
    itself and not an encoder's noise; skips the test when this OpenCV build cannot write FFV1."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1")
    for image in frames:
        writer.write(image)
    writer.release()
    return path


def write_clip(path: Path, count: int, size=(64, 36), fps=30.0) -> Path:
    """A tiny video of `count` frames, frame i filled with the value i (see write_frames)."""
    return write_frames(path, [np.full((size[1], size[0], 3), index, np.uint8) for index in range(count)], fps)


def ball_frames(positions: Iterable[Tuple[int, int]], size=(360, 640)) -> List[np.ndarray]:
    """Grey frames with a 3x3 bright ball whose middle pixel is at each (column, row)."""
    frames = []
    for column, row in positions:
        image = np.full(size + (3,), 90, np.uint8)
        image[row - 1:row + 2, column - 1:column + 2] = (60, 230, 230)
        frames.append(image)
    return frames


# --------------------------------------------------------------------------- motion

def motion_probe_delta_from_frames(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    """The C++ probe's frame delta on the CPU: max(gray, V, scaled-S) differences, blurred 3x3."""
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    v_diff = cv2.absdiff(curr_hsv[:, :, 2], prev_hsv[:, :, 2])
    s_diff = cv2.absdiff(curr_hsv[:, :, 1], prev_hsv[:, :, 1])
    gray_diff = cv2.absdiff(curr_gray, prev_gray)
    s_scaled = np.minimum(s_diff.astype(np.float32) * _PROBE_S_DIFF_SCALE, 255.0).astype(np.uint8)
    combined = np.maximum(gray_diff, np.maximum(v_diff, s_scaled))
    return cv2.GaussianBlur(combined, (3, 3), 0)


def refine_raw_motion_temporal_cpu(
    raw_motion_u8: Optional[np.ndarray],
    prev_bgr: Optional[np.ndarray],
    curr_bgr: Optional[np.ndarray],
    next_bgr: Optional[np.ndarray],
    cfg: Config,
) -> Optional[np.ndarray]:
    """The CPU reference for the temporal gate ``motion._raw_motion_cuda`` applies: the
    ungated foreground kept where the C++ probe's movement test passes, then the raw-mask
    morphology.  Only the tests use it; the runtime runs the CUDA path."""
    if raw_motion_u8 is None:
        return None
    refined = raw_motion_u8
    if cfg.motion_raw_temporal_gate and prev_bgr is not None and curr_bgr is not None:
        hi, lo, very_hi = cfg.motion_raw_temporal_hi, cfg.motion_raw_temporal_lo, cfg.motion_raw_temporal_very_hi
        d_prev = motion_probe_delta_from_frames(prev_bgr, curr_bgr)
        if next_bgr is not None:
            d_next = motion_probe_delta_from_frames(curr_bgr, next_bgr)
            temporal = ((d_prev >= hi) & (d_next >= lo)) | ((d_next >= hi) & (d_prev >= lo)) | (d_prev >= very_hi)
        else:
            temporal = d_prev >= hi
        refined = cv2.bitwise_and(refined, temporal.astype(np.uint8) * 255)
    return _clean_raw_motion_mask(refined, cfg)


class Capture:
    """A ``cv2.VideoCapture`` stand-in whose frame ``i`` is filled with the value ``i``."""

    def __init__(self, count: int):
        self.count, self.index = count, 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.index, np.uint8)
        self.index += 1
        return True, frame

    def release(self) -> None:
        """Nothing to release; present because callers always release captures."""


def cpu_detector(decode, conf: float = 0.5, frame_tensor=None) -> "CpuGridTrackNet":
    """A GridTrackNet backend on the CPU whose model is the identity and whose decoder is
    ``decode(output, width, height, threshold)``; each frame's tensor is its first pixel's value
    unless ``frame_tensor`` says otherwise. The model answers no grid, so the prepass runs at one
    unit per frame (window step 5), where its answer goes straight to the decoder."""
    return CpuGridTrackNet(Config(device="0", conf=conf, gridtracknet_window_step=5), model=lambda batch: batch, decode=decode,
                           frame_tensor=frame_tensor or (lambda frame, device: torch.tensor([float(frame[0, 0, 0])])))


class StandInModel:
    """A callable standing in for the ball model, carrying the contract the detector reads off a
    real one (``ball_models``: GridTrackNet's ``ARCH``, grid and channel count)."""

    ARCH, GRID_COLS, GRID_ROWS = GridTrackNet.ARCH, GridTrackNet.GRID_COLS, GridTrackNet.GRID_ROWS
    INPUT_CHANNELS, FRAMES_PER_UNIT = GridTrackNet.INPUT_CHANNELS, GridTrackNet.FRAMES_PER_UNIT

    def __init__(self, forward):
        self.forward = forward

    def __call__(self, batch):
        return self.forward(batch)


class CpuGridTrackNet(GridTrackNetBallDetector):
    """The real constructor, run on the CPU with an injected model (a callable, wrapped in
    ``StandInModel`` unless it already is one), decoder and frame encoder.

    The constructor insists on CUDA and loads weights from disk; both are
    replaced for its duration only. ``on_publish`` sees every prefix the
    prepass declares final.
    """

    def __init__(self, cfg, model, decode, frame_tensor, on_publish=None):
        if not isinstance(model, StandInModel):
            model = StandInModel(model)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch, "device", return_value=torch.device("cpu")), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(ball_detectors, "load_ball_model", return_value=model):
            super().__init__("unused.npz", cfg)
        self._decode = decode
        self._frame_tensor = frame_tensor
        self._on_publish = on_publish

    def _publish(self, upto):
        super()._publish(upto)
        if self._on_publish is not None:
            self._on_publish(upto)


class FailingWriter(VideoWriter):
    """An OpenCV-backed writer whose encoder blocks on the first frame until released, then fails."""

    def __init__(self, path, cfg):
        self.entered, self.release, self.drained = threading.Event(), threading.Event(), threading.Event()
        super().__init__(path, 30.0, 64, 32, cfg, encoder=EncoderChoice(None, (ENCODER_OPENCV,)))

    def _raw_write(self, frame):
        self.entered.set()
        self.release.wait(1)
        raise RuntimeError("encoder failed")

    def _drain(self):
        try:
            super()._drain()
        finally:
            self.drained.set()
