"""Pass 1's motion masks as two CUDA kernels: the same bits as ``motion.py``'s PyTorch ops.

The masks are a chain of some forty small PyTorch ops a box (the background-model foreground, the
two probe deltas, their blurs, the temporal gate, the morphology), each launched from Python; the
launches, not the GPU, were most of pass 1's motion time (``audit/records/opt_abplus.md``). Here
``motion_gate`` computes a box's foreground and temporal gate per pixel in one launch, and
``motion_morph`` runs one dilation or erosion of a box, so the default cleaning (a closing) is two
more. Every float step is the one PyTorch's kernels take, rounded the same way: ``__fsub_rn`` and
the like are never contracted into fused multiply-adds, a blur sums its taps in the order
``avg_pool2d`` does and divides by nine, and a scalar is the float32 PyTorch casts it to. The tests
compare both paths bit for bit (``tests/test_motion_gpu.py``).

``MotionKernels.load`` compiles them with ``cuda_kernels.load_kernel`` and raises
``KernelUnavailable`` when it cannot; pass 1 then keeps the PyTorch path.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .cuda_kernels import Kernel, load_kernel

Box = Tuple[int, int, int, int]

# One thread per pixel of a box. ``temporal`` is 0 (no gate: the foreground is the motion), 1 (the
# previous frame only) or 2 (the previous and the next frame). A pixel is only ever set, never
# cleared, so boxes that overlap OR their answers as the PyTorch path does.
_MOTION_GATE_SOURCE = r"""
__device__ __forceinline__ float probe_delta(const float* earlier, const float* later, long long plane,
                                             long long p) {
    // max(|later - earlier| * scale) over gray, value, saturation; the scales are 255, 255, 318.75.
    const float scale[3] = {255.0f, 255.0f, 318.75f};
    float best = 0.0f;
    for (int c = 0; c < 3; ++c) {
        const float d = __fmul_rn(fabsf(__fsub_rn(later[c * plane + p], earlier[c * plane + p])), scale[c]);
        best = c == 0 ? d : fmaxf(best, d);
    }
    return best;
}

__device__ __forceinline__ float blurred_delta(const float* earlier, const float* later, long long plane, int width,
                                               int x, int y, int x1, int y1, int x2, int y2) {
    // avg_pool2d's 3x3 mean with zero padding at the box's edges: the taps in the box, summed row by
    // row from the top left, divided by nine.
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int qy = y + dy;
        if (qy < y1 || qy >= y2) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int qx = x + dx;
            if (qx < x1 || qx >= x2) continue;
            sum = __fadd_rn(sum, probe_delta(earlier, later, plane, (long long)qy * width + qx));
        }
    }
    return __fdiv_rn(sum, 9.0f);
}

extern "C" __global__ void motion_gate(
        const float* __restrict__ curr, const float* __restrict__ prev, const float* __restrict__ next,
        const float* __restrict__ mean, const float* __restrict__ var,
        unsigned char* __restrict__ gated, unsigned char* __restrict__ ungated,
        int height, int width, int x1, int y1, int x2, int y2,
        float k_std_sq, float thresh_sq, float s_scale, float v_min, float hi, float lo, float very_hi,
        int temporal) {
    const int x = x1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= x2 || y >= y2) return;
    const long long plane = (long long)height * width;
    const long long p = (long long)y * width + x;
    // The background foreground: value (plane 1 of the frame, 0 of the model), then saturation.
    bool over = false;
    for (int c = 0; c < 2; ++c) {
        const float d = __fsub_rn(curr[(c + 1) * plane + p], mean[c * plane + p]);
        float threshold = __fadd_rn(__fmul_rn(var[c * plane + p], k_std_sq), thresh_sq);
        if (c == 1) threshold = __fmul_rn(threshold, s_scale);
        over = over || __fmul_rn(d, d) > threshold;
    }
    if (!over || !(curr[plane + p] > v_min)) return;
    if (ungated != 0) ungated[p] = 1;
    bool moving = true;
    if (temporal > 0) {
        const float dp = blurred_delta(prev, curr, plane, width, x, y, x1, y1, x2, y2);
        if (temporal == 1) {
            moving = dp >= hi;
        } else {
            const float dn = blurred_delta(curr, next, plane, width, x, y, x1, y1, x2, y2);
            moving = (dp >= hi && dn >= lo) || (dn >= hi && dp >= lo) || dp >= very_hi;
        }
    }
    if (moving) gated[p] = 1;
}
"""

# One thread per pixel of a box: a ``size`` x ``size`` dilation (``erode`` 0) or erosion of the box
# with OpenCV's anchor (``size // 2``) and border (outside the box is unset for a dilation and set
# for an erosion), written to ``dst``; a set pixel is also set in ``merge`` when it is given.
_MOTION_MORPH_SOURCE = r"""
extern "C" __global__ void motion_morph(
        const unsigned char* __restrict__ src, int src_pitch, unsigned char* __restrict__ dst, int dst_pitch,
        unsigned char* __restrict__ merge, int merge_pitch, int box_width, int box_height, int size, int erode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= box_width || y >= box_height) return;
    const int anchor = size / 2;
    bool out = erode != 0;
    for (int i = 0; i < size; ++i) {
        const int qy = y + i - anchor;
        if (qy < 0 || qy >= box_height) continue;
        for (int j = 0; j < size; ++j) {
            const int qx = x + j - anchor;
            if (qx < 0 || qx >= box_width) continue;
            const bool set = src[(long long)qy * src_pitch + qx] != 0;
            out = erode ? (out && set) : (out || set);
        }
    }
    dst[(long long)y * dst_pitch + x] = out ? 1 : 0;
    if (out && merge != 0) merge[(long long)y * merge_pitch + x] = 1;
}
"""
_BLOCK = (32, 8, 1)
_NULL = ctypes.c_void_p(None)


def _grid(width: int, height: int) -> Tuple[int, int, int]:
    return ((width + _BLOCK[0] - 1) // _BLOCK[0], (height + _BLOCK[1] - 1) // _BLOCK[1], 1)


@dataclass(frozen=True)
class GateScalars:
    """The foreground and temporal-gate thresholds as the PyTorch path applies them, each the
    float32 PyTorch casts the Python scalar to: the variance factor (``motion_k_std ** 2``), the
    squared floor (``(motion_thresh / 255) ** 2``), the saturation threshold's widening, the value
    floor (``motion_v_min / 255``) and the gate's ``hi``, ``lo`` and ``very_hi``."""

    k_std_sq: float
    thresh_sq: float
    s_scale: float
    v_min: float
    hi: float
    lo: float
    very_hi: float

    def arguments(self) -> List[ctypes.c_float]:
        return [ctypes.c_float(value) for value in (self.k_std_sq, self.thresh_sq, self.s_scale, self.v_min,
                                                    self.hi, self.lo, self.very_hi)]


def _check_planes(tensor: torch.Tensor, planes: int, height: int, width: int) -> None:
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != (planes, height, width) or not tensor.is_contiguous():
        raise ValueError(f"motion_gate reads contiguous ({planes}, {height}, {width}) float32 planes")


class MotionKernels:
    """``motion_gate`` and ``motion_morph`` loaded on one device, with the box-level launches."""

    def __init__(self, gate: Kernel, morph: Kernel) -> None:
        self.gate_kernel = gate
        self.morph_kernel = morph

    @classmethod
    def load(cls, device: torch.device, cache_dir: Optional[Path] = None) -> "MotionKernels":
        """Both kernels on `device`, compiled once and cached in `cache_dir`; raises
        ``KernelUnavailable`` when they cannot be compiled or loaded."""
        return cls(load_kernel(_MOTION_GATE_SOURCE, "motion_gate", device, cache_dir),
                   load_kernel(_MOTION_MORPH_SOURCE, "motion_morph", device, cache_dir))

    def gate(self, curr: torch.Tensor, prev: Optional[torch.Tensor], nxt: Optional[torch.Tensor],
             mean: torch.Tensor, var: torch.Tensor, box: Box, gated: torch.Tensor, ungated: Optional[torch.Tensor],
             scalars: GateScalars, temporal: bool) -> None:
        """Set the moving pixels of `box` in the frame-sized uint8 `gated` (and the foreground in
        `ungated`, when given): ``curr``, ``prev`` and ``nxt`` are frames' ``(3, H, W)`` motion planes
        (``GpuFrame.planes``), ``mean`` and ``var`` the background's ``(2, H, W)``; with `temporal`
        the gate reads ``prev`` and, when there is one, ``nxt``."""
        height, width = int(gated.shape[0]), int(gated.shape[1])
        _check_planes(curr, 3, height, width)
        _check_planes(mean, 2, height, width)
        _check_planes(var, 2, height, width)
        mode = 0
        if temporal:
            assert prev is not None
            _check_planes(prev, 3, height, width)
            mode = 1
            if nxt is not None:
                _check_planes(nxt, 3, height, width)
                mode = 2
        x1, y1, x2, y2 = box
        self.gate_kernel(_grid(x2 - x1, y2 - y1), _BLOCK, [
            curr, prev if prev is not None else _NULL, nxt if nxt is not None and mode == 2 else _NULL, mean, var,
            gated, ungated if ungated is not None else _NULL, height, width, x1, y1, x2, y2,
            *scalars.arguments(), mode])

    def morph(self, src: torch.Tensor, dst: torch.Tensor, size: int, erode: bool,
              merge: Optional[torch.Tensor] = None) -> None:
        """One dilation (or erosion) of the uint8 box `src` (a view, rows of any pitch) into `dst`,
        a contiguous tensor of its shape; the set pixels are also set in `merge` (a view of the same
        shape) when it is given. Size 1 copies the box."""
        box_height, box_width = int(src.shape[0]), int(src.shape[1])
        for tensor in (src, dst) + ((merge,) if merge is not None else ()):
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != (box_height, box_width) or tensor.stride(1) != 1:
                raise ValueError("motion_morph reads and writes uint8 boxes of one shape with unit column stride")
        self.morph_kernel(_grid(box_width, box_height), _BLOCK, [
            src, src.stride(0), dst, dst.stride(0), merge if merge is not None else _NULL,
            merge.stride(0) if merge is not None else 0, box_width, box_height, int(size), int(erode)])
