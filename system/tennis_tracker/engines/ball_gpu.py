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

``TransitionRing`` is the prepass's side of a network that streams (``streamnet``, ``stream_s4``):
each frame's transition (the frame and the one before it in its phase) is encoded once, in batches
of ``StreamHalves.batch``, its features kept in a ring of slots on the device, and each unit is
decoded from the four slots of its transitions, gathered on the device by one ``index_select`` per
stride. A unit is always the last five frames its phase has seen (``ball_detectors._UnitStream``),
so its transitions are the phase's last four, whatever the window step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from ..cuda_kernels import Kernel, load_kernel
from ..gridtracknet import FRAMES_PER_UNIT, GridPoint
from ..video_decode import Nv12Frame, swscale_coefficients

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


# ``Nv12ModelResize``'s kernel: one thread per model-input pixel converts its four source taps from
# NV12 to BGR as ``video_decode.nv12_to_bgr`` does and combines them as ``ModelResize.resize_u8``
# does, every step in the same integer arithmetic, then writes the pixel's RGB values as the FP16
# bit patterns ``lut`` holds for each byte.
_NV12_MODEL_INPUT_SOURCE = r"""
extern "C" __global__ void nv12_model_input(
        const unsigned char* __restrict__ nv12, int pitch, int height,
        unsigned short* __restrict__ out, int out_height, int out_width,
        const long long* __restrict__ columns, const int* __restrict__ column_weights,
        const long long* __restrict__ rows, const int* __restrict__ row_weights,
        const unsigned short* __restrict__ lut,
        int cy, int y_offset, int v2r, int u2b, int u2g, int v2g) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) return;
    int sums[3] = {0, 0, 0};
    for (int tap_row = 0; tap_row < 2; ++tap_row) {
        const int row = (int)rows[tap_row * out_height + y];
        const unsigned char* luma_row = nv12 + (long long)row * pitch;
        const unsigned char* chroma_row = nv12 + (long long)(height + (row >> 1)) * pitch;
        int across[3] = {0, 0, 0};
        for (int tap_column = 0; tap_column < 2; ++tap_column) {
            const int column = (int)columns[tap_column * out_width + x];
            const int weight = column_weights[tap_column * out_width + x];
            const int luma = ((((int)luma_row[column]) << 3) - y_offset) * cy >> 16;
            const int u = (((int)chroma_row[column & ~1]) << 3) - 1024;
            const int v = (((int)chroma_row[(column & ~1) + 1]) << 3) - 1024;
            int green = ((u * u2g) >> 16) + ((v * v2g) >> 16);
            green = min(max(green, -32768), 32767);
            const int bgr[3] = {luma + ((u * u2b) >> 16), luma + green, luma + ((v * v2r) >> 16)};
            for (int channel = 0; channel < 3; ++channel) {
                across[channel] += min(max(bgr[channel], 0), 255) * weight;
            }
        }
        const int row_weight = row_weights[tap_row * out_height + y];
        for (int channel = 0; channel < 3; ++channel) {
            sums[channel] += ((across[channel] >> 4) * row_weight) >> 16;
        }
    }
    const int plane = out_height * out_width;
    for (int channel = 0; channel < 3; ++channel) {
        const int value = min(max((sums[channel] + 2) >> 2, 0), 255);
        out[(2 - channel) * plane + y * out_width + x] = lut[value];     // BGR in, RGB out
    }
}
"""
_NV12_KERNEL_NAME = "nv12_model_input"
_NV12_BLOCK = (32, 8, 1)


def load_nv12_kernel(device: torch.device, cache_dir: Optional[Path] = None) -> Kernel:
    """``Nv12ModelResize``'s kernel on `device` (``cuda_kernels.load_kernel``, cached in `cache_dir`);
    raises ``KernelUnavailable`` when it cannot be compiled or loaded."""
    return load_kernel(_NV12_MODEL_INPUT_SOURCE, _NV12_KERNEL_NAME, device, cache_dir)


class Nv12ModelResize:
    """An NV12 picture (``video_decode.Nv12Frame``) of one source size -> the model's (3, height,
    width) RGB FP16 input, in one kernel: the bytes ``ModelResize`` gives for OpenCV's BGR picture
    of it (``video_decode.nv12_to_bgr``), without the full-size BGR picture ever being made. The
    conversion is per pixel, so converting only the source pixels the resize reads changes nothing."""

    def __init__(self, source_width: int, source_height: int, width: int, height: int, full_range: bool,
                 device: torch.device, kernel: Kernel):
        self.source_size = (int(source_width), int(source_height))
        self.size = (int(width), int(height))
        self.full_range = bool(full_range)
        self.kernel = kernel
        taps = ModelResize(source_width, source_height, width, height, device)
        self.columns, self.rows = taps.columns, taps.rows                            # int64: x0 then x1, y0 then y1
        self.column_weights = taps.column_weights.reshape(-1).contiguous()          # a0 then a1, int32
        self.row_weights = taps.row_weights.reshape(-1).contiguous()                # b0 then b1, int32
        # uint8 / 255 in float32 rounded to FP16, made by the very ops ``ModelResize.__call__`` runs.
        self.lut = torch.arange(256, dtype=torch.uint8, device=device).to(torch.float32).div_(255.0).to(torch.float16)
        c = swscale_coefficients(self.full_range)
        self.coefficients = [c["y"], c["y_offset"], c["v2r"], c["u2b"], c["u2g"], c["v2g"]]
        self.grid = ((int(width) + _NV12_BLOCK[0] - 1) // _NV12_BLOCK[0],
                     (int(height) + _NV12_BLOCK[1] - 1) // _NV12_BLOCK[1], 1)

    def __call__(self, frame: Nv12Frame) -> torch.Tensor:
        planes = frame.planes
        if (frame.width, frame.height) != self.source_size or frame.full_range != self.full_range:
            raise ValueError(f"Nv12ModelResize for {self.source_size} got a {frame.width}x{frame.height} picture")
        if planes.dtype != torch.uint8 or planes.dim() != 2 or planes.stride(1) != 1:
            raise ValueError("Nv12ModelResize reads (H * 3 / 2, W) uint8 rows")
        width, height = self.size
        out = torch.empty((3, height, width), dtype=torch.float16, device=planes.device)
        self.kernel(self.grid, _NV12_BLOCK, [
            planes, planes.stride(0), self.source_size[1], out, height, width,
            self.columns, self.column_weights, self.rows, self.row_weights, self.lut, *self.coefficients])
        return out


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


@dataclass
class StreamHalves:
    """A streaming network's two halves as the prepass runs them: ``encode(frames, previous)``
    (``(batch, 3, H, W)`` each) returns the transitions' features (one ``(batch, C, h, w)`` tensor
    per stride) and ``decode(*features)`` (each ``(batch, 4 * C, h, w)``, a unit's four transitions
    concatenated per stride in temporal order) the units' 15-channel output. Both take
    exactly ``batch`` rows; ``stream`` is the CUDA stream they run on (TensorRT's), None for the
    current one. Either may return buffers the next call overwrites."""

    encode: Callable[..., Sequence[torch.Tensor]]
    decode: Callable[..., torch.Tensor]
    batch: int
    stream: Optional[Any] = None


class TransitionRing:
    """One clip's transitions for ``StreamHalves`` (see the module docstring). ``push(frame, phase)``
    takes each decoded frame's model input in order and returns the sequence number of the transition
    ending at it (None for a phase's first frame, which has no previous frame); every ``batch`` of
    them is encoded as it fills. ``decode(units)`` encodes what is left and returns the output of up
    to ``batch`` units, each given as its five frames' values from ``push`` (the first is not read),
    padded with copies of the last unit as a static batch is. Transition ``n`` lives in slot
    ``n % slots``; a unit whose transition was overwritten before it was decoded is a RuntimeError
    (``transition_slots`` sizes the ring so the prepass never does that)."""

    def __init__(self, halves: StreamHalves, slots: int, phases: int):
        self.halves = halves
        self.batch = int(halves.batch)
        self.slots = int(slots)
        if self.slots < FRAMES_PER_UNIT - 1:
            raise ValueError(f"TransitionRing needs at least {FRAMES_PER_UNIT - 1} slots, not {slots}")
        self.previous: List[Optional[torch.Tensor]] = [None] * int(phases)
        self.pushed = 0                 # transitions numbered so far
        self.encoded = 0                # transitions below this number are in the ring
        self.frames: Optional[torch.Tensor] = None              # the encoder's inputs: the frame ...
        self.before: Optional[torch.Tensor] = None              # ... and the one before it
        self.features: List[torch.Tensor] = []                  # per stride, (slots, C, h, w)
        self.gathered: List[torch.Tensor] = []                  # per stride, the decoder's (batch, 4 C, h, w) input

    def push(self, frame: torch.Tensor, phase: int) -> Optional[int]:
        previous, self.previous[phase] = self.previous[phase], frame
        if previous is None:
            return None
        if self.frames is None:
            self.frames = torch.empty((self.batch, *frame.shape), dtype=frame.dtype, device=frame.device)
            self.before = torch.empty_like(self.frames)
        assert self.before is not None
        row = self.pushed - self.encoded
        self.frames[row].copy_(frame)
        self.before[row].copy_(previous)
        self.pushed += 1
        if self.pushed - self.encoded == self.batch:
            self._encode()
        return self.pushed - 1

    def _encode(self) -> None:
        """Encode the transitions pushed since the last call (the rows past them are stale and
        their answers are dropped) into their slots, wrapping at the ring's end."""
        count = self.pushed - self.encoded
        if count == 0:
            return
        features = self.halves.encode(self.frames, self.before)
        if not self.features:
            self.features = [torch.empty((self.slots, *level.shape[1:]), dtype=level.dtype, device=level.device)
                             for level in features]
            pairs = FRAMES_PER_UNIT - 1
            self.gathered = [torch.empty((self.batch, pairs * level.shape[1], *level.shape[2:]), dtype=level.dtype,
                                         device=level.device) for level in features]
        start = self.encoded % self.slots
        first = min(count, self.slots - start)
        for ring, level in zip(self.features, features):
            ring[start:start + first].copy_(level[:first])
            if first < count:
                ring[:count - first].copy_(level[first:count])
        self.encoded = self.pushed

    def decode(self, units: Sequence[Sequence[Optional[int]]]) -> torch.Tensor:
        return self.halves.decode(*self.gather(units))

    def gather(self, units: Sequence[Sequence[Optional[int]]]) -> List[torch.Tensor]:
        """The decoder's inputs for `units` (as ``decode`` takes them): per stride, each unit's four
        cached transitions concatenated in temporal order, ``(batch, 4 C, h, w)``, padded as ``decode``
        pads. A decoder over a region of the frame would slice these maps; the buffers are reused by
        the next call."""
        if not 0 < len(units) <= self.batch:
            raise ValueError(f"TransitionRing decodes 1 to {self.batch} units a call, not {len(units)}")
        self._encode()
        padded = list(units) + [units[-1]] * (self.batch - len(units))
        given = [number for unit in padded for number in unit[1:]]
        if any(number is None for number in given):
            raise ValueError("a unit's frames past its first must each end a transition (a number from push)")
        numbers = cast(List[int], given)
        if self.encoded - min(numbers) > self.slots:
            raise RuntimeError(f"TransitionRing of {self.slots} slots lost transition {min(numbers)} "
                               f"before its unit was decoded ({self.encoded} encoded)")
        device = self.features[0].device
        index = torch.tensor([number % self.slots for number in numbers], dtype=torch.long)
        if device.type == "cuda":           # pinned, so the copy does not wait for the stream
            index = index.pin_memory().to(device, non_blocking=True)
        for ring, gathered in zip(self.features, self.gathered):
            torch.index_select(ring, 0, index, out=gathered.view(len(numbers), *ring.shape[1:]))
        return self.gathered


def transition_slots(batch: int, step: int, stride: int) -> int:
    """Slots enough for a ``TransitionRing`` behind the prepass with `batch` units a call, a unit
    every `step` frames of each of `stride` phases; at most one transition is pushed per source
    frame. A unit's four transitions were pushed within the ``4 * stride`` source frames it spans,
    and it is decoded at the latest when its call fills, ``batch - 1`` units later: every phase cuts
    one every ``step * stride`` source frames, so within ``ceil((batch - 1) / stride) * step * stride``
    frames. One more ``stride`` covers phases starting a frame apart, and one slot the transition
    being written. Measured over steps 1 to 5, 1 to 3 phases and batches 1 to 8, a ring two slots
    smaller still never lost a transition (``audit/records/opt_stream_prepass.md`` on the optimization branch)."""
    return (FRAMES_PER_UNIT - 1) * stride + -(-(batch - 1) // stride) * step * stride + stride + 1
