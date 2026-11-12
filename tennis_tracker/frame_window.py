"""The frame window pass 1 slides over the clip.

Pass 1 looks at three consecutive source frames at once: the current frame
is preprocessed, and the previous and next frames feed the temporal motion
gate.  In host memory the window keeps the current frame and the one read ahead;
with preprocessing enabled it also keeps each frame's CUDA tensors so a frame
is uploaded exactly once.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np
import torch

from .video_io import PinnedFrameUploader, cuda_frame_planes, cuda_frame_to_chw_f32, hwc_u8_to_chw_f32

# Graphed frame slots: the window holds three frames' tensors (prev, curr, next), and one more
# slot keeps the slot being written apart from every frame the GPU may still be reading.
_GRAPH_SLOTS = 4


class GpuFrame(NamedTuple):
    """One frame on CUDA: CHW float32 BGR in [0, 1] and its motion planes.

    ``planes`` is ``(3, H, W)``: gray, HSV value, HSV saturation
    (``video_io.cuda_frame_planes``); ``v``, ``s`` and ``vs`` (value and
    saturation as one ``(2, H, W)`` view) are views of it.
    """

    chw: torch.Tensor
    planes: torch.Tensor
    v: torch.Tensor
    s: torch.Tensor
    vs: torch.Tensor

    @classmethod
    def of(cls, chw: torch.Tensor) -> "GpuFrame":
        planes = cuda_frame_planes(chw)
        return cls(chw, planes, planes[1], planes[2], planes[1:3])


class _GraphSlot(NamedTuple):
    """One recorded conversion: the uint8 frame it reads and the ``GpuFrame`` it writes."""

    u8: torch.Tensor
    graph: torch.cuda.CUDAGraph
    frame: GpuFrame


class GraphedFrameConversion:
    """``GpuFrame.of(hwc_u8_to_chw_f32(frame))`` replayed from CUDA graphs: a dozen kernels launched
    for about 7 us of CPU instead of 0.3 ms (``audit/records/opt_phaseA_followup.md``), the same
    kernels and so the same floats.

    Each of ``slots`` slots owns a uint8 input buffer and a graph that converts it into the slot's
    own output tensors; ``next_slot`` hands the slots out in turn, so a frame's tensors stay valid
    until ``slots`` more frames have been converted.  All work runs on the current stream, which
    orders a slot's reuse after every earlier read of it.  Capture starts on first use; a capture
    that fails (CUDA work from another thread breaks it) raises ``RuntimeError`` and the caller
    converts eagerly from then on.
    """

    def __init__(self, height: int, width: int, device: torch.device, slots: int = _GRAPH_SLOTS) -> None:
        self.shape = (int(height), int(width), 3)
        self.device = device
        self.slot_count = max(1, int(slots))
        self._slots: List[_GraphSlot] = []
        self._next = 0

    def _capture(self) -> _GraphSlot:
        u8 = torch.zeros(self.shape, dtype=torch.uint8, device=self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            GpuFrame.of(hwc_u8_to_chw_f32(u8))          # warm-up outside the capture
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            frame = GpuFrame.of(hwc_u8_to_chw_f32(u8))
        return _GraphSlot(u8, graph, frame)

    def next_input(self) -> torch.Tensor:
        """The uint8 buffer of the slot the next ``convert`` replays; the caller fills it."""
        index = self._next
        if index >= len(self._slots):
            self._slots.append(self._capture())
        return self._slots[index].u8

    def convert(self) -> GpuFrame:
        """Replay the slot whose input ``next_input`` returned and move on to the next slot."""
        slot = self._slots[self._next]
        self._next = (self._next + 1) % self.slot_count
        slot.graph.replay()
        return slot.frame


class FrameWindow:
    """``prev`` / ``curr`` / ``next`` source frames and their CUDA tensors.

    ``curr`` and ``next`` are the decoded BGR frames (``next`` is None at the
    end of the clip).  ``prev_gpu`` / ``curr_gpu`` / ``next_gpu`` are their
    CUDA tensors when ``upload`` is on, else None; the previous frame is only
    kept on the GPU, which is where the temporal gate reads it.  ``slide``
    moves the window one frame along.  With ``cuda_graphs`` (a CUDA device
    only) each frame's conversion is replayed from a CUDA graph
    (``GraphedFrameConversion``).
    """

    def __init__(self, device: torch.device, upload: bool,
                 first: np.ndarray, second: Optional[np.ndarray], cuda_graphs: bool = False) -> None:
        self.device = device
        self.upload = bool(upload)
        self.uploader: Optional[PinnedFrameUploader] = None
        self.graphs: Optional[GraphedFrameConversion] = None
        # Pinned staging records CUDA events on the device's stream, so only a CUDA device gets it;
        # a CPU device (the tests' fake CUDA) takes the plain upload even on a box with a driver.
        if self.upload and device.type == "cuda":
            try:
                self.uploader = PinnedFrameUploader(first.shape[0], first.shape[1], device)
            except RuntimeError as error:    # no CUDA driver or no pinned memory
                print(f"[pass 1] Pinned frame uploads unavailable ({str(error).splitlines()[0]}); using plain uploads")
            if cuda_graphs and self.uploader is not None:
                self.graphs = GraphedFrameConversion(first.shape[0], first.shape[1], device)
        self.curr: np.ndarray = first
        self.next: Optional[np.ndarray] = second
        self.prev_gpu: Optional[GpuFrame] = None
        self.curr_gpu: Optional[GpuFrame] = self._to_gpu(first)
        self.next_gpu: Optional[GpuFrame] = self._to_gpu(second)

    def _to_gpu(self, frame: Optional[np.ndarray]) -> Optional[GpuFrame]:
        if frame is None or not self.upload:
            return None
        if self.graphs is not None and self.uploader is not None and tuple(frame.shape) == self.graphs.shape:
            try:
                target = self.graphs.next_input()
            except RuntimeError as error:
                print(f"[pass 1] CUDA graph capture failed; converting frames directly ({str(error).splitlines()[0]})")
                self.graphs = None
            else:
                self.uploader.upload_into(frame, target)
                return self.graphs.convert()
        return GpuFrame.of(cuda_frame_to_chw_f32(frame, self.device, uploader=self.uploader))

    def slide(self, incoming: Optional[np.ndarray]) -> None:
        """Make ``next`` the current frame and ``incoming`` the next one (None at the end)."""
        self.prev_gpu = self.curr_gpu
        self.curr_gpu = self.next_gpu
        self.curr = self.next
        self.next = incoming
        self.next_gpu = self._to_gpu(incoming)

    @property
    def exhausted(self) -> bool:
        """True once ``slide`` has moved past the last frame."""
        return self.curr is None
