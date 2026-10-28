"""The ball prepass's work on the GPU around the network: each decoded frame resized to the model's
input there, overlapping units' answers averaged in a ring of per-frame sums, and every frame's
peak found there, so the host sends one uint8 frame up and gets four floats per frame back.

``ModelResize`` is OpenCV's ``INTER_LINEAR`` resize (``gridtracknet.resize_to_model``, the resize the
trainer's frame cache stores) redone in integer tensor arithmetic. OpenCV resizes 8-bit images in
fixed point: each output pixel's two horizontal taps carry 11-bit weights, the horizontal pass keeps
the exact integer sums, and the vertical pass combines two rows the way its SIMD kernel
(``VResizeLinearVec_32s8u``) does, a 16-bit high multiply of each row shifted down 4 bits, then
``(a + b + 2) >> 2``. Reproducing those steps gives the very bytes ``cv2.resize`` gives: on 20
Pomona frames (1920x1080 to 768x432, 19.9 M values) no value differed, where a float bilinear
resize rounded to uint8 differed in 4.6% of them by one level (``audit/records/opt_ball_prepass.md``).
A frame is resized horizontally first, as OpenCV does; the colour order is swapped and the values
scaled to [0, 1] after, which commutes with the per-channel resize.

``HeatRing`` keeps the running sum of every frame's answers until the frame is final, in a ring of
slots indexed by source frame, and adds a batch's answers with one ``index_add_`` per position in
the unit, latest position first: within a batch that is the order the host summed them in (the
earliest unit holding a frame holds it at the latest position), so the float32 sums are the ones
the host computed. ``grid_peaks`` is ``gridtracknet.decode_grid``'s argmax on the device (the first
maximum, as NumPy's), returning the confidence, the two offsets and the flat cell index;
``peak_points`` turns them into points on the host with the same float arithmetic.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..gridtracknet import FRAMES_PER_UNIT, GridPoint

INTER_RESIZE_COEF_BITS = 11
INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS
PEAK_FIELDS = 4                     # confidence, x offset, y offset, flat cell index


def linear_taps(source: int, target: int, horizontal: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV's ``INTER_LINEAR`` taps along one axis of `source` pixels resized to `target`: per output
    pixel the first and second source pixel and their 11-bit fixed-point weights, as ``cv::resize``
    computes them (the scale in double, the source position in float, each weight rounded on its
    own). Past an edge OpenCV treats the axes differently: across, a position before the first or
    past the last pixel moves onto it with all the weight; down, the weights stay as computed and
    only the rows read are clamped into the image."""
    scale = 1.0 / (float(target) / float(source))
    position = ((np.arange(target, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    first = np.floor(position)
    fraction = (position - first).astype(np.float32)
    first = first.astype(np.int64)
    if horizontal:
        low = first < 0
        fraction[low], first[low] = 0.0, 0
        high = first >= source - 1
        fraction[high], first[high] = 0.0, source - 1
    weight0 = np.rint((np.float32(1.0) - fraction) * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int64)
    weight1 = np.rint(fraction * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int64)
    second = np.clip(first + 1, 0, source - 1)
    return np.clip(first, 0, source - 1), second, weight0, weight1


class ModelResize:
    """A BGR uint8 HWC frame of one source size -> the model's (3, height, width) RGB input in [0, 1],
    the bytes ``gridtracknet.frame_tensor`` gives (see the module docstring), on ``device``. The taps
    are built once per source size."""

    def __init__(self, source_width: int, source_height: int, width: int, height: int,
                 device: torch.device, dtype: torch.dtype = torch.float16):
        self.source_size = (int(source_width), int(source_height))
        self.size = (int(width), int(height))
        self.dtype = dtype
        self.identity = self.source_size == self.size
        x0, x1, a0, a1 = linear_taps(source_width, width, horizontal=True)
        y0, y1, b0, b1 = linear_taps(source_height, height, horizontal=False)
        self.columns = torch.from_numpy(np.concatenate([x0, x1])).to(device)
        self.rows = torch.from_numpy(np.concatenate([y0, y1])).to(device)
        self.column_weights = torch.from_numpy(np.stack([a0, a1])).to(device=device, dtype=torch.int32).view(1, 2, width, 1)
        self.row_weights = torch.from_numpy(np.concatenate([b0, b1])).to(device=device, dtype=torch.int32).view(-1, 1, 1)

    def resize_u8(self, frame: torch.Tensor) -> torch.Tensor:
        """`frame` (HWC uint8 on the device, the source size) resized to (height, width, 3) uint8."""
        if self.identity:
            return frame
        width, height = self.size
        picked = frame.index_select(0, self.rows).index_select(1, self.columns).to(torch.int32)
        across = (picked.view(2 * height, 2, width, -1) * self.column_weights).sum(dim=1, dtype=torch.int32)
        rows = ((across >> 4) * self.row_weights) >> 16
        return ((rows[:height] + rows[height:] + 2) >> 2).clamp_(0, 255).to(torch.uint8)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """The model's input for `frame`: (3, height, width) RGB in [0, 1], uint8 / 255 in float32
        rounded to ``dtype``."""
        resized = self.resize_u8(frame)
        return resized.permute(2, 0, 1).flip(0).to(torch.float32).div_(255.0).to(self.dtype)


def grid_peaks(grids: torch.Tensor) -> torch.Tensor:
    """Each frame's peak from its ``(3, rows, cols)`` grids (`grids` is ``(N, 3, rows, cols)``):
    an ``(N, 4)`` float32 tensor of the highest confidence, the x and y offsets in that cell and the
    cell's flat index (``row * cols + col``), the first maximum as ``np.argmax`` finds it."""
    flat = grids.float().flatten(2)
    best = flat[:, 0].argmax(dim=1)
    picked = flat.gather(2, best.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(2)
    return torch.cat([picked, best.to(torch.float32).unsqueeze(1)], dim=1)


def peak_points(peaks: np.ndarray, rows: int, cols: int, output_width: int, output_height: int,
                threshold: float) -> List[GridPoint]:
    """``grid_peaks`` rows (on the host) as ``decode_grid``'s answers: the cell's point in
    `output_width` x `output_height` pixels (None below `threshold`) and its confidence."""
    found: List[GridPoint] = []
    for confidence_value, x_offset, y_offset, cell in peaks.tolist():
        confidence = float(np.float32(confidence_value))
        point = None
        if confidence >= threshold:
            row, col = divmod(int(cell), cols)
            point = ((x_offset + col) * output_width / cols, (y_offset + row) * output_height / rows)
        found.append((point, confidence))
    return found


class HeatRing:
    """The running sums of the answers each frame of the clip gets, until the frame is final (see the
    module docstring): ``(slots, 3, rows, cols)`` float32 sums and a count per slot, a frame's slot
    being its source index modulo ``slots``. ``take`` returns the frames' peaks of the mean grids and
    clears their slots. The caller keeps every frame between the oldest not yet taken and the newest
    added within ``slots`` of each other (``check_span``)."""

    def __init__(self, slots: int, rows: int, cols: int, device: torch.device):
        self.slots = int(slots)
        self.device = device
        self.sums = torch.zeros((self.slots, 3, rows, cols), dtype=torch.float32, device=device)
        self.counts = torch.zeros((self.slots,), dtype=torch.float32, device=device)

    def check_span(self, oldest: int, newest: int) -> None:
        if newest - oldest >= self.slots:
            raise RuntimeError(f"HeatRing of {self.slots} slots cannot hold frames {oldest} to {newest} at once")

    def add(self, output: torch.Tensor, units: Sequence[Sequence[int]]) -> None:
        """Add the first ``len(units)`` units' answers in `output` (``(batch, 15, rows, cols)``) to the
        frames each covers (`units`: per unit its five source indices)."""
        count = len(units)
        grids = output[:count].reshape(count, FRAMES_PER_UNIT, 3, output.shape[-2], output.shape[-1])
        slots = torch.tensor([[index % self.slots for index in unit] for unit in units], dtype=torch.long)
        slots = slots.to(self.device, non_blocking=True)
        for position in reversed(range(FRAMES_PER_UNIT)):
            self.sums.index_add_(0, slots[:, position], grids[:, position].float())
        self.counts.index_add_(0, slots.flatten(), torch.ones(slots.numel(), dtype=torch.float32, device=self.device))

    def take(self, indices: Sequence[int]) -> torch.Tensor:
        """The ``grid_peaks`` of the mean grids of the frames at `indices`, whose slots are cleared.
        The mean is the sum times the count's float32 reciprocal: the host divided each CUDA sum by
        its count as a Python number, which PyTorch's CUDA division turns into exactly that."""
        slots = torch.tensor([index % self.slots for index in indices], dtype=torch.long).to(self.device, non_blocking=True)
        sums = self.sums.index_select(0, slots)
        means = sums * self.counts.index_select(0, slots).reciprocal().view(-1, 1, 1, 1)
        self.sums.index_fill_(0, slots, 0.0)
        self.counts.index_fill_(0, slots, 0.0)
        return grid_peaks(means)


class Readback:
    """Small device results copied to the host without stopping the device: ``start`` queues a copy
    into pinned memory and records an event after it; ``ready`` says whether it has landed and
    ``wait`` blocks until it has. On the CPU the copy is immediate."""

    def __init__(self, values: torch.Tensor):
        self.event: Optional[torch.cuda.Event] = None
        if values.device.type == "cuda":
            self.host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
            self.host.copy_(values, non_blocking=True)
            self.event = torch.cuda.Event()
            self.event.record()
        else:
            self.host = values.clone()

    def ready(self) -> bool:
        return self.event is None or bool(self.event.query())

    def wait(self) -> np.ndarray:
        if self.event is not None:
            self.event.synchronize()
        return self.host.numpy()
