"""Motion evidence for pass 1: the background model, raw and boost motion
masks on CUDA, the court-side protect mask and the detector-input enhancement."""
from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple, Union, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ball_in_play_selector import PackedMask

from .court.court_geometry import build_court_side_guides
from .frame_window import FrameWindow, GpuFrame
from .motion_gpu import GateScalars, MotionKernels


class MotionSettings(Protocol):
    """The slice of ``Config`` this module reads: the background model, the
    raw-mask gate and morphology, the boost-mask blob filter and the
    detector-input enhancement.  ``Config`` satisfies it; a test may pass any
    object with these fields."""

    background_alpha: float
    motion_freeze_alpha: float
    motion_thresh: float
    motion_k_std: float
    motion_v_min: float
    motion_raw_temporal_gate: bool
    motion_raw_temporal_hi: float
    motion_raw_temporal_lo: float
    motion_raw_temporal_very_hi: float
    motion_raw_close_size: int
    motion_raw_open_size: int
    boost_min_blob_area: int
    boost_max_blob_area: int
    blob_shape_filter: bool
    blob_max_aspect: float
    blob_preserve_tiny: bool
    blob_tiny_max_area: int
    motion_dilate: int
    dim_static: float
    static_sat_scale: float
    pre_sat_boost: float
    pre_val_boost: float
    pre_hue_shift: float


# Boost mask component filter.
_BALL_BYPASS_AREA_DIAG_FRAC = 0.035        # ball-sized components, px per unit of frame diagonal
_BALL_BYPASS_AREA_MIN_PX, _BALL_BYPASS_AREA_MAX_PX = 15, 150
_TINY_BLOB_AREA_PX = 50                    # below this the aspect test is loosest
_TINY_BLOB_MAX_ASPECT, _SMALL_BLOB_MAX_ASPECT = 4.5, 3.5
_BOOST_DISC_MAX_RADIUS_PX = 12            # the disc drawn for a blob never balloons past the ball
# Probe-style frame delta: saturation differences count this much more than value.
_PROBE_S_DIFF_SCALE = 1.25


@functools.lru_cache(maxsize=None)
def _get_kernel(size: int) -> np.ndarray:
    return np.ones((size, size), np.uint8)


def _clean_raw_motion_mask(mask_u8: Optional[np.ndarray], cfg: MotionSettings) -> Optional[np.ndarray]:
    """Apply the active raw-mask morphology without appearance or size heuristics."""
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    if cv2.countNonZero(mask_u8) == 0:
        return mask_u8

    out = mask_u8
    close_size = cfg.motion_raw_close_size
    open_size = cfg.motion_raw_open_size
    if close_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _get_kernel(close_size))
    if open_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _get_kernel(open_size))

    return out


def _set_pixel_box(mask: np.ndarray) -> Optional[Roi]:
    """The ``(x1, y1, x2, y2)`` bounding box of the nonzero pixels of a 2D mask; None when it has none.

    Not ``cv2.boundingRect``: OpenCV 4.9's mask scan can return a box one or two columns too
    narrow when every set pixel lies in the first three columns, depending on the buffer's
    address (a misaligned crop of the download buffer hits it)."""
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    cols = np.flatnonzero(mask[y1:y2].any(axis=0))
    return int(cols[0]), y1, int(cols[-1]) + 1, y2


class BoostMaskBuffers(Protocol):
    """Where ``filter_boost_mask`` may write its result instead of allocating one."""

    boost_mask_u8: np.ndarray


def filter_boost_mask(raw_motion: np.ndarray, min_area: int, max_area: int, cfg: MotionSettings,
                      buffers: Optional[BoostMaskBuffers] = None) -> np.ndarray:
    """The boost mask: a filled disc for every ball-sized component of ``raw_motion``.

    Components outside ``[min_area, max_area]`` px are dropped; ball-sized ones
    bypass the aspect test (small balls blur into short streaks), larger ones
    must pass ``cfg.blob_max_aspect``.  Any nonzero pixel is motion.  One
    connected-components pass, over the bounding box of the set pixels only
    (motion is sparse, and the pass costs in proportion to the area it scans;
    the components, their stats and the rounded disc centres are the same);
    with ``buffers`` the result is written into ``buffers.boost_mask_u8``.
    The result is all zero when nothing survives.
    """
    img_h, img_w = raw_motion.shape[:2]
    if buffers is not None:
        result = buffers.boost_mask_u8
        result.fill(0)
    else:
        result = np.zeros((img_h, img_w), dtype=np.uint8)
    box = _set_pixel_box(raw_motion)
    if box is None:
        return result
    off_x, off_y, x2, y2 = box
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(raw_motion[off_y:y2, off_x:x2], connectivity=8)
    for cx, cy, radius in _boost_discs(stats[1:num], centroids[1:num], off_x, off_y, (img_h, img_w),
                                       min_area, max_area, cfg):
        cv2.circle(result, (int(cx), int(cy)), int(radius), 255, -1)
    return result


def _boost_discs(stats: np.ndarray, centroids: np.ndarray, off_x: Union[int, np.ndarray],
                 off_y: Union[int, np.ndarray], frame_shape: Tuple[int, int], min_area: int, max_area: int,
                 cfg: MotionSettings) -> np.ndarray:
    """The boost disc of every ball-sized component, as ``(cx, cy, radius)`` int rows: the filter
    ``filter_boost_mask`` documents, over the components' ``cv2.connectedComponentsWithStats`` rows
    (the background row left out).  The components were found in a part of a ``frame_shape``
    frame; ``off_x`` and ``off_y`` (scalars, or one per component) move them into the frame.

    A centroid in the part plus the part's offset rounds to the same pixel as the frame's centroid:
    both are an integer sum over the area, and they could only round apart within an ulp of .5,
    which a ratio with that denominator is either exactly on or at least 1 / (2 * area) away from.
    """
    img_h, img_w = frame_shape
    img_diag = math.hypot(float(img_w), float(img_h))
    ball_bypass_max_area = max(_BALL_BYPASS_AREA_MIN_PX,
                               min(_BALL_BYPASS_AREA_MAX_PX, int(round(_BALL_BYPASS_AREA_DIAG_FRAC * img_diag))))
    if cfg.blob_preserve_tiny:
        ball_bypass_max_area = max(ball_bypass_max_area, cfg.blob_tiny_max_area)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    aspect = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1)
    # Small ball blobs get the loosest aspect test (they blur into short streaks).
    small = area <= ball_bypass_max_area
    small_ok = aspect <= np.where(area < _TINY_BLOB_AREA_PX, _TINY_BLOB_MAX_ASPECT, _SMALL_BLOB_MAX_ASPECT)
    large_ok = ~(aspect > cfg.blob_max_aspect) if cfg.blob_shape_filter else np.ones(len(area), dtype=bool)
    keep = (area >= min_area) & (area <= max_area) & np.where(small, small_ok, large_ok)
    if not keep.any():
        return np.empty((0, 3), dtype=np.int64)
    # A disc at the centroid; its radius is capped so motion blur and shadows cannot balloon the
    # boost far beyond the ball.
    if isinstance(off_x, np.ndarray):
        off_x = off_x[keep]
    if isinstance(off_y, np.ndarray):
        off_y = off_y[keep]
    cx = (centroids[keep, 0] + off_x + 0.5).astype(np.int64)
    cy = (centroids[keep, 1] + off_y + 0.5).astype(np.int64)
    radius = np.minimum((np.sqrt(area[keep] / math.pi) + 0.5).astype(np.int64), _BOOST_DISC_MAX_RADIUS_PX)
    return np.stack([cx, cy, np.maximum(radius, 1)], axis=1)


def _line_side_value(a: np.ndarray, v: np.ndarray, p: np.ndarray) -> float:
    d = p - a
    return float(v[0] * d[1] - v[1] * d[0])

def _clip_polygon_to_halfplane(poly: Sequence[np.ndarray], a: np.ndarray, v: np.ndarray,
                               keep_sign: float) -> List[np.ndarray]:
    if not poly:
        return []

    def inside(pt):
        return _line_side_value(a, v, pt) * keep_sign >= -1e-6

    def intersect(p1, p2):
        d = p2 - p1
        denom = v[0] * d[1] - v[1] * d[0]
        if abs(float(denom)) < 1e-6:
            return p1
        t = ((a[0] - p1[0]) * v[1] - (a[1] - p1[1]) * v[0]) / denom
        t = float(max(0.0, min(1.0, t)))
        return p1 + t * d

    output = []
    prev = poly[-1]
    prev_in = inside(prev)
    for curr in poly:
        curr_in = inside(curr)
        if prev_in and curr_in:
            output.append(curr)
        elif prev_in and not curr_in:
            output.append(intersect(prev, curr))
        elif (not prev_in) and curr_in:
            output.append(intersect(prev, curr))
            output.append(curr)
        prev, prev_in = curr, curr_in
    return output

def _halfplane_mask_from_line(h: int, w: int, a: np.ndarray, v: np.ndarray, keep_sign: float) -> Optional[np.ndarray]:
    if float(np.hypot(v[0], v[1])) < 1e-6 or abs(float(keep_sign)) < 1e-6:
        return None
    rect = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([float(w - 1), 0.0], dtype=np.float32),
        np.array([float(w - 1), float(h - 1)], dtype=np.float32),
        np.array([0.0, float(h - 1)], dtype=np.float32),
    ]
    clipped = _clip_polygon_to_halfplane(rect, a, v, keep_sign)
    if len(clipped) < 3:
        return None
    pts = np.round(np.asarray(clipped, dtype=np.float32)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask > 0

def _x_relation_mask_from_line(h: int, w: int, anchor: np.ndarray, direction: np.ndarray,
                               relation: str) -> Optional[np.ndarray]:
    """Build mask using explicit left/right rule against line x(y).

    relation:
      "left"  -> keep pixels with x < line_x(y)
      "right" -> keep pixels with x > line_x(y)
    """
    dy = float(direction[1])
    dx = float(direction[0])
    if abs(dy) < 1e-6:
        return None

    ys = np.arange(h, dtype=np.float32)
    x_line = float(anchor[0]) + (ys - float(anchor[1])) * (dx / dy)
    xs = np.arange(w, dtype=np.float32)[None, :]

    if relation == "left":
        return xs < x_line[:, None]
    if relation == "right":
        return xs > x_line[:, None]
    return None

def build_court_side_protect_mask(h: int, w: int, court_keypoints: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Protect outside side regions using court-guided side lines.

    Left side:
      offset from kp2 to image-left by |kp5-kp2|, line parallel to kp2->kp0.
      Pixels left of this line are protected.
    Right side:
      offset from kp3 to image-right by |kp7-kp3|, line parallel to kp3->kp1.
      Pixels right of this line are protected.
    """
    guides = build_court_side_guides(court_keypoints)
    if not guides:
        return None

    protect = None
    for side in ("left", "right"):
        g = guides.get(side)
        if g is None:
            continue
        m = _side_protect_mask(h, w, side, g)
        if m is not None:
            protect = m if protect is None else (protect | m)

    return protect if protect is not None and np.any(protect) else None


def _side_protect_mask(h: int, w: int, side: str, g: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """The pixels outside one side guide: left of the left line / right of the right line, or,
    for a near-horizontal line, the half-plane away from the court."""
    # Primary rule: explicit "left of left line / right of right line".
    m = _x_relation_mask_from_line(h, w, g["anchor"], g["dir"], side)
    if m is not None:
        return m
    # Fallback for near-horizontal lines: keep the outside half-plane.
    inside_sign = _line_side_value(g["anchor"], g["dir"], g["base"])
    if abs(float(inside_sign)) < 1e-6:
        return None
    edge_sign = _outside_edge_sign(h, w, side, g)
    # Enforce "outside" side: must be opposite the court-side base point.
    # If edge sample is ambiguous/wrong, fall back to the opposite of inside_sign.
    if abs(float(edge_sign)) < 1e-6 or edge_sign * inside_sign > 0:
        keep_sign = -inside_sign
    else:
        keep_sign = edge_sign
    return _halfplane_mask_from_line(h, w, g["anchor"], g["dir"], keep_sign)


def _outside_edge_sign(h: int, w: int, side: str, g: Dict[str, np.ndarray]) -> float:
    """Which side of the guide the frame's own edge is on, sampled at that edge's midpoint and,
    when the midpoint lies on the line, at its corners."""
    x_edge = 0.0 if side == "left" else float(w - 1)
    edge_sign = _line_side_value(g["anchor"], g["dir"], np.array([x_edge, float(h) * 0.5], dtype=np.float32))
    if abs(float(edge_sign)) < 1e-6:
        for ref in (np.array([x_edge, 0.0], dtype=np.float32), np.array([x_edge, float(h - 1)], dtype=np.float32)):
            edge_sign = _line_side_value(g["anchor"], g["dir"], ref)
            if abs(float(edge_sign)) >= 1e-6:
                break
    return edge_sign

def _hsv_to_bgr_cuda(h_t: torch.Tensor, s_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    """Convert HSV tensors (all 0-1 range) to BGR tensor (0-255 uint8)."""
    h6 = h_t * 6.0
    c = v_t * s_t
    x = c * (1.0 - torch.abs((h6 % 2.0) - 1.0))
    m = v_t - c
    z = torch.zeros_like(h_t)
    sec = h6.long().clamp(0, 5)
    s0 = sec == 0
    s1 = sec == 1
    s2 = sec == 2
    s3 = sec == 3
    s4 = sec == 4
    s5 = sec == 5
    ro = torch.where(s0 | s5, c, torch.where(s1 | s4, x, z))
    go = torch.where(s1 | s2, c, torch.where(s0 | s3, x, z))
    bo = torch.where(s3 | s4, c, torch.where(s2 | s5, x, z))
    bgr = torch.stack([bo + m, go + m, ro + m], dim=0)
    return torch.clamp(bgr * 255, 0, 255).byte().permute(1, 2, 0).contiguous()

def _dilate_motion_cuda(raw_motion: torch.Tensor, dilate_k: int) -> torch.Tensor:
    """Dilate boolean motion mask on GPU via max_pool2d."""
    if dilate_k > 1:
        mm = raw_motion.float().unsqueeze(0).unsqueeze(0)
        mm = F.max_pool2d(mm, dilate_k, stride=1, padding=dilate_k // 2)
        return mm.squeeze() > 0.5
    return raw_motion

@functools.lru_cache(maxsize=None)
def _probe_scale(device: torch.device) -> torch.Tensor:
    """Per-plane factors of the probe delta: gray and value differences in 0-255 units,
    saturation differences weighted ``_PROBE_S_DIFF_SCALE`` more."""
    return torch.tensor([255.0, 255.0, 255.0 * _PROBE_S_DIFF_SCALE], dtype=torch.float32,
                        device=device).view(3, 1, 1)


def _probe_delta(earlier: torch.Tensor, later: torch.Tensor) -> torch.Tensor:
    """Probe-style delta of two frames' ``(3, h, w)`` motion planes (``GpuFrame.planes``): the
    largest of the gray, value and scaled saturation differences, in 0-255 units, unblurred.

    Per element this is the arithmetic of the original per-plane form
    (``abs(later - earlier) * 255`` and ``* 318.75``, then the maximum), so it is bit-identical.
    """
    delta = later - earlier
    delta.abs_()
    delta.mul_(_probe_scale(delta.device))
    return torch.amax(delta, dim=0)


def _blur3(delta: torch.Tensor) -> torch.Tensor:
    """3x3 box blur with zero padding at the edges of ``delta`` (counted in the mean)."""
    return F.avg_pool2d(delta[None, None], 3, stride=1, padding=1)[0, 0]


class BackgroundModel:
    """Running mean and variance of the V and S planes, on CUDA.

    ``mean`` and ``var`` are ``(2, H, W)``: value, then saturation (one
    tensor for both planes halves the kernel launches of a frame, and the
    arithmetic per element is unchanged).  A pixel counts as motion when its
    squared distance from the mean exceeds ``motion_thresh^2 +
    motion_k_std^2 * variance`` (``foreground``).  Static pixels adapt at
    ``background_alpha``; pixels that were motion on the previous frame at
    ``motion_freeze_alpha``.  Call ``update`` once per frame before
    preprocessing it; ``preprocess_frame_cuda`` then calls
    ``remember_motion`` with that frame's cleaned raw mask.
    """

    def __init__(self, cfg: MotionSettings, device: torch.device, cuda_graphs: bool = False) -> None:
        self.cfg = cfg
        self.device = device
        self.mean: Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None
        self._prev_motion: Optional[torch.Tensor] = None    # bool, last frame's cleaned raw motion
        # With CUDA graphs the steady-state update (a previous motion mask exists) is replayed from
        # a graph per input address: the frame planes come from ``GraphedFrameConversion``'s few
        # slots, and the motion mask is copied into one buffer the graphs read.
        self.cuda_graphs = bool(cuda_graphs) and device.type == "cuda"
        self._update_graphs: Dict[Tuple[int, Tuple[int, ...]], torch.cuda.CUDAGraph] = {}
        self._motion_buffer: Optional[torch.Tensor] = None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    @property
    def mean_v(self) -> torch.Tensor:
        assert self.mean is not None
        return self.mean[0]

    @property
    def var_v(self) -> torch.Tensor:
        assert self.var is not None
        return self.var[0]

    @torch.no_grad()
    def update(self, curr_vs: torch.Tensor) -> None:
        """Fold the current frame's ``(2, H, W)`` value and saturation planes into the model."""
        if not self.ready:
            self.mean = curr_vs.clone()
            thr = self.cfg.motion_thresh / 255.0
            self.var = torch.full_like(self.mean, thr ** 2)
            return
        if self.cuda_graphs and self._prev_motion is not None and self._prev_motion is self._motion_buffer:
            graph = self._update_graph(curr_vs)
            if graph is not None:
                graph.replay()
                return
        self._update(curr_vs)

    def _update_graph(self, curr_vs: torch.Tensor) -> Optional[torch.cuda.CUDAGraph]:
        """The recorded steady-state update reading `curr_vs`'s memory, captured on first use; None
        (and graphs given up, with a note) when the capture fails."""
        key = (curr_vs.data_ptr(), tuple(curr_vs.stride()))
        graph = self._update_graphs.get(key)
        if graph is None:
            try:
                graph = torch.cuda.CUDAGraph()
                # Capturing records the kernels without running them, so the model is unchanged.
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    self._update(curr_vs)
            except RuntimeError as error:
                print(f"[pass 1] CUDA graph capture failed; updating the background directly "
                      f"({str(error).splitlines()[0]})")
                self.cuda_graphs = False
                return None
            self._update_graphs[key] = graph
        return graph

    def _update(self, curr_vs: torch.Tensor) -> None:
        assert self.mean is not None and self.var is not None
        diff_sq = curr_vs - self.mean
        diff_sq.mul_(diff_sq)                       # ``x ** 2`` is computed as ``x * x``
        alpha: Union[float, torch.Tensor]
        if self._prev_motion is not None:
            alpha = torch.where(self._prev_motion, self.cfg.motion_freeze_alpha, self.cfg.background_alpha)
        else:
            alpha = self.cfg.background_alpha
        keep = 1.0 - alpha
        self.var.mul_(keep).add_(alpha * diff_sq)
        self.mean.mul_(keep).add_(curr_vs * alpha)

    @torch.no_grad()
    def foreground(self, curr: GpuFrame, box: Roi) -> torch.Tensor:
        """The boolean foreground of ``curr`` inside ``box`` (``(x1, y1, x2, y2)``), box-sized."""
        assert self.mean is not None and self.var is not None
        cfg = self.cfg
        x1, y1, x2, y2 = box
        diff_sq = curr.vs[:, y1:y2, x1:x2] - self.mean[:, y1:y2, x1:x2]
        diff_sq.mul_(diff_sq)
        thresh_sq = self.var[:, y1:y2, x1:x2] * (cfg.motion_k_std ** 2)
        thresh_sq.add_((cfg.motion_thresh / 255.0) ** 2)
        thresh_sq[1].mul_(_S_THRESHOLD_SCALE)
        over = diff_sq > thresh_sq
        fg = over[0] | over[1]
        fg &= curr.v[y1:y2, x1:x2] > (cfg.motion_v_min / 255.0)
        return fg

    def _motion_buffer_for(self, shape: Tuple[int, int]) -> torch.Tensor:
        """The one slow-adaptation buffer the recorded updates read, (re)made for `shape`; a new
        buffer drops the graphs recorded against the old one."""
        if self._motion_buffer is None or tuple(self._motion_buffer.shape) != shape:
            self._motion_buffer = torch.empty(shape, dtype=torch.bool, device=self.device)
            self._update_graphs.clear()
        return self._motion_buffer

    def motion_target(self, height: int, width: int) -> torch.Tensor:
        """A zeroed ``height`` x ``width`` boolean mask for this frame's cleaned motion to be written
        into and then handed to ``remember_motion``; with CUDA graphs it is the one buffer the
        recorded updates read, which ``remember_motion`` then takes without a copy."""
        if not self.cuda_graphs:
            return torch.zeros((height, width), dtype=torch.bool, device=self.device)
        return self._motion_buffer_for((height, width)).zero_()

    def remember_motion(self, raw_motion: Optional[torch.Tensor]) -> None:
        """Keep this frame's cleaned raw motion (a boolean frame-sized tensor, or None when there
        is none yet) for the next update's slow-adaptation mask.  The tensor is used as it is,
        not copied: the caller hands over a fresh one each frame (or ``motion_target``'s); with
        CUDA graphs any other tensor is copied into the one buffer the recorded updates read."""
        if self.cuda_graphs and raw_motion is not None and raw_motion is not self._motion_buffer:
            height, width = raw_motion.shape
            raw_motion = self._motion_buffer_for((int(height), int(width))).copy_(raw_motion)
        self._prev_motion = raw_motion


@dataclass(frozen=True)
class DetectorInputRequest:
    """What ``preprocess_frame_cuda`` should materialise beyond the motion masks.

    ``cpu`` / ``cuda``: the detector input as a host BGR array / an HWC uint8
    CUDA tensor.  ``dim_static``: when no ball-sized blob was found, still dim
    the static background (off, the source frame is handed over as it is).
    ``wide_boost``: brighten the blobs of the ungated motion, which keeps a ball
    the temporal gate suppressed visible to the detector, instead of the gated
    motion's.  ``MASKS_ONLY`` asks for nothing but the masks.
    """

    cpu: bool = False
    cuda: bool = False
    dim_static: bool = True
    wide_boost: bool = True


MASKS_ONLY = DetectorInputRequest(cpu=False, cuda=False, dim_static=False, wide_boost=False)


class PreprocessResult(NamedTuple):
    """What ``preprocess_frame_cuda`` produces for one frame.

    ``detector_frame``: the CPU BGR detector input when ``request.cpu``: the
    enhanced frame, or the source frame when nothing was applied to it; None
    otherwise.  ``detector_frame_cuda``: the enhanced frame as an HWC uint8
    CUDA tensor when ``request.cuda``, else None.  ``masks``: the frame's
    motion masks on their way to the host (``MotionMasks``); None until the
    background model is ready.  ``raw_motion`` and ``boost_mask`` wait for
    them and return them as full-frame arrays: the cleaned gated raw-motion
    mask (uint8 {0, 255}) and a disc per ball-sized blob of it (None when no
    blob survived).
    """

    detector_frame: Optional[np.ndarray]
    detector_frame_cuda: Optional[torch.Tensor]
    masks: Optional["MotionMasks"]

    @property
    def raw_motion(self) -> Optional[np.ndarray]:
        return self.masks.raw_u8() if self.masks is not None else None

    @property
    def boost_mask(self) -> Optional[np.ndarray]:
        return self.masks.selector_boost_u8() if self.masks is not None else None


Roi = Tuple[int, int, int, int]
# Saturation noise is penalised by widening its threshold this much.
_S_THRESHOLD_SCALE = 1.5
# Pixels this bright and unsaturated are white court lines: never dimmed.
_WHITE_MAX_S = 70.0 / 255.0
_WHITE_MIN_V = 170.0 / 255.0
# The boost hue (yellow-green) that the hue shift pulls boosted pixels toward, as a 0-1 hue.
_BOOST_HUE = 30.0 / 179.0


def _union_box(boxes: Sequence[Roi]) -> Roi:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _motion_boxes(curr: GpuFrame, rois: Optional[List[Roi]]) -> List[Roi]:
    """The boxes motion is computed in: the ROIs, or the whole frame as one box."""
    if rois is not None:
        return list(rois)
    h, w = int(curr.v.shape[0]), int(curr.v.shape[1])
    return [(0, 0, w, h)]


def _raw_motion_cuda(window: FrameWindow, background: BackgroundModel, cfg: MotionSettings,
                     rois: Optional[List[Roi]], want_ungated: bool = True,
                     kernels: Optional[MotionKernels] = None
                     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(gated, ungated)`` boolean raw-motion masks for the current frame.

    ``ungated`` is the background-model foreground inside the ROIs (the whole
    frame without them); ``gated`` additionally passes the frame-to-frame
    temporal test when it is enabled and a previous frame exists.  Both are
    None until the background model is ready; they are the same tensor when
    the gate did not run, and ``ungated`` is None when it is not wanted and
    the gate ran.  With ``kernels`` they come from ``motion_gpu``'s gate
    kernel, one launch a box; else PyTorch's ops build the same bits.
    """
    if not background.ready:
        return None, None
    curr = window.curr_gpu
    assert curr is not None
    boxes = _motion_boxes(curr, rois)
    prev = window.prev_gpu if cfg.motion_raw_temporal_gate else None
    nxt = window.next_gpu if prev is not None else None
    if kernels is not None:
        return _raw_motion_by_kernel(kernels, curr, prev, nxt, background, cfg, boxes, want_ungated)
    return _raw_motion_by_ops(curr, prev, nxt, background, cfg, boxes, want_ungated)


def _raw_motion_by_ops(curr: GpuFrame, prev: Optional[GpuFrame], nxt: Optional[GpuFrame],
                       background: BackgroundModel, cfg: MotionSettings, boxes: List[Roi], want_ungated: bool
                       ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``_raw_motion_cuda`` with PyTorch's ops; ``prev`` is None when the gate does not run.

    The foreground and the two probe deltas are computed once, in the box that
    holds every ROI; each ROI then blurs its own part of the deltas, with zero
    padding at the ROI's edges as when each ROI was computed on its own, so
    the masks are the same bit for bit.
    """
    h, w = int(curr.v.shape[0]), int(curr.v.shape[1])
    ux1, uy1, ux2, uy2 = _union_box(boxes)
    fg = background.foreground(curr, (ux1, uy1, ux2, uy2))
    ungated: Optional[torch.Tensor] = None
    if want_ungated or prev is None:
        ungated = torch.zeros((h, w), dtype=torch.bool, device=fg.device)
        for x1, y1, x2, y2 in boxes:
            ungated[y1:y2, x1:x2] = fg[y1 - uy1:y2 - uy1, x1 - ux1:x2 - ux1]
    if prev is None:
        return ungated, ungated
    union_planes = (slice(None), slice(uy1, uy2), slice(ux1, ux2))
    d_prev = _probe_delta(prev.planes[union_planes], curr.planes[union_planes])
    d_next = _probe_delta(curr.planes[union_planes], nxt.planes[union_planes]) if nxt is not None else None
    hi, lo, very_hi = cfg.motion_raw_temporal_hi, cfg.motion_raw_temporal_lo, cfg.motion_raw_temporal_very_hi
    gated = torch.zeros((h, w), dtype=torch.bool, device=fg.device)
    for x1, y1, x2, y2 in boxes:
        part = (slice(y1 - uy1, y2 - uy1), slice(x1 - ux1, x2 - ux1))
        dp = _blur3(d_prev[part])
        if d_next is None:
            temporal = dp >= hi
        else:
            dn = _blur3(d_next[part])
            temporal = ((dp >= hi) & (dn >= lo)) | ((dn >= hi) & (dp >= lo)) | (dp >= very_hi)
        # Ungated motion drives the detector-input boost (recall: helps the detector
        # recover misses). Gated motion drives the selector and the static dimming
        # (precision: rejects false positives).
        gated[y1:y2, x1:x2] |= fg[part] & temporal
    return gated, ungated


def _gate_scalars(cfg: MotionSettings) -> GateScalars:
    """The thresholds and scales ``BackgroundModel.foreground`` and ``_raw_motion_by_ops`` apply."""
    return GateScalars(k_std_sq=cfg.motion_k_std ** 2, thresh_sq=(cfg.motion_thresh / 255.0) ** 2,
                       s_scale=_S_THRESHOLD_SCALE, v_min=cfg.motion_v_min / 255.0,
                       probe_s_scale=255.0 * _PROBE_S_DIFF_SCALE, hi=cfg.motion_raw_temporal_hi,
                       lo=cfg.motion_raw_temporal_lo, very_hi=cfg.motion_raw_temporal_very_hi)


def _raw_motion_by_kernel(kernels: MotionKernels, curr: GpuFrame, prev: Optional[GpuFrame],
                          nxt: Optional[GpuFrame], background: BackgroundModel, cfg: MotionSettings,
                          boxes: List[Roi], want_ungated: bool
                          ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``_raw_motion_cuda`` with ``motion_gpu``'s gate kernel; ``prev`` is None when the gate does
    not run, and the gated mask is then the foreground."""
    assert background.mean is not None and background.var is not None
    h, w = int(curr.v.shape[0]), int(curr.v.shape[1])
    gated = torch.zeros((h, w), dtype=torch.uint8, device=curr.v.device)
    ungated = (torch.zeros((h, w), dtype=torch.uint8, device=curr.v.device)
               if prev is not None and want_ungated else None)
    scalars = _gate_scalars(cfg)
    prev_planes = prev.planes if prev is not None else None
    next_planes = nxt.planes if nxt is not None else None
    for box in boxes:
        kernels.gate(curr.planes, prev_planes, next_planes, background.mean, background.var, box, gated, ungated,
                     scalars)
    gated_mask = gated.view(torch.bool)
    if prev is None:
        return gated_mask, gated_mask
    return gated_mask, ungated.view(torch.bool) if ungated is not None else None


def _morph_max(mask_f: torch.Tensor, size: int) -> torch.Tensor:
    """``cv2.dilate`` of a 0/1 float mask by a ``size`` x ``size`` box with OpenCV's default
    anchor (``size // 2``) and border (outside the mask counts as unset)."""
    a = size // 2
    padded = F.pad(mask_f[None, None], (a, size - 1 - a, a, size - 1 - a))
    return F.max_pool2d(padded, size, stride=1)[0, 0]


def _clean_raw_motion_cuda(mask: torch.Tensor, cfg: MotionSettings) -> torch.Tensor:
    """``_clean_raw_motion_mask`` on the GPU: the same closing and opening, pixel for pixel.

    Erosion is ``1 - dilate(1 - mask)``, which counts the outside as set, as
    OpenCV's erode does.  Returns ``mask`` itself when neither applies.
    """
    close_size, open_size = cfg.motion_raw_close_size, cfg.motion_raw_open_size
    if close_size <= 1 and open_size <= 1:
        return mask
    out = mask.to(torch.float32)
    if close_size > 1:
        out = 1.0 - _morph_max(1.0 - _morph_max(out, close_size), close_size)
    if open_size > 1:
        out = _morph_max(1.0 - _morph_max(1.0 - out, open_size), open_size)
    return out > 0.5


class _Placed(NamedTuple):
    """A mask (any nonzero pixel is set) whose top left corner sits at ``(x, y)`` in the frame."""

    mask: np.ndarray
    x: int
    y: int


def _with_gaps(occupied: np.ndarray) -> np.ndarray:
    """The ``occupied`` indices (ascending) plus the first index of every gap between two of them,
    so each run of empty rows (or columns) between set pixels is squeezed to one."""
    gaps = np.flatnonzero(np.diff(occupied) > 1)
    return np.sort(np.concatenate([occupied, occupied[gaps] + 1]))


class _SqueezedMask(NamedTuple):
    """A sparse frame mask with every run of empty rows and of empty columns squeezed to one.

    A component (8-connected) never spans an empty row or column, so the squeezed image holds the
    frame's components whole and apart, each with its area, width and height, its pixels moved by
    one offset per axis: a full-frame mask of a few thousand set pixels spread over the frame scans
    a few percent of the area its bounding box covers.  ``rows`` and ``cols`` give the frame row and
    column of each image row and column; ``box`` is the bounding box of the set pixels.
    """

    image: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    box: Roi

    @classmethod
    def of(cls, mask: np.ndarray, row_any: np.ndarray, col_any: np.ndarray) -> Optional["_SqueezedMask"]:
        """``mask`` squeezed, given which of its rows and columns hold a set pixel; None when none
        does.  When squeezing would keep more than half the bounding box, the box is kept as it is."""
        occupied_rows, occupied_cols = np.flatnonzero(row_any), np.flatnonzero(col_any)
        if not occupied_rows.size:
            return None
        x1, y1 = int(occupied_cols[0]), int(occupied_rows[0])
        x2, y2 = int(occupied_cols[-1]) + 1, int(occupied_rows[-1]) + 1
        rows, cols = _with_gaps(occupied_rows), _with_gaps(occupied_cols)
        if 2 * rows.size * cols.size > (y2 - y1) * (x2 - x1):
            return cls(mask[y1:y2, x1:x2], np.arange(y1, y2), np.arange(x1, x2), (x1, y1, x2, y2))
        return cls(mask[rows][:, cols], rows, cols, (x1, y1, x2, y2))      # rows first: whole-row copies

    def discs(self, frame_shape: Tuple[int, int], min_area: int, max_area: int, cfg: MotionSettings) -> np.ndarray:
        """``_boost_discs`` of the frame's components, found in the squeezed image."""
        num, _labels, stats, centroids = cv2.connectedComponentsWithStats(self.image, connectivity=8)
        left, top = stats[1:num, cv2.CC_STAT_LEFT], stats[1:num, cv2.CC_STAT_TOP]
        return _boost_discs(stats[1:num], centroids[1:num], self.cols[left] - left, self.rows[top] - top,
                            frame_shape, min_area, max_area, cfg)

    def packed(self, frame: np.ndarray) -> PackedMask:
        """``PackedMask.pack(frame)`` for the frame mask this was squeezed from, its bounding box
        known without a scan of the frame.  (``np.packbits`` counts any nonzero byte as a set bit,
        as ``PackedMask.pack``'s ``> 0`` does.)"""
        x1, y1, x2, y2 = self.box
        return PackedMask(np.packbits(frame[y1:y2, x1:x2].reshape(-1)), frame.shape[0], frame.shape[1], x1, y1, x2, y2)


# A filled disc of radius r drawn at (cx, cy) stays inside cx +- (r + this), and likewise in y.
_DISC_MARGIN_PX = 2


def _disc_stamps(discs: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[List[_Placed]]:
    """Each boost disc drawn on the square of the frame it covers (clipped to the frame as on a
    full-frame mask); None when there are none."""
    if not len(discs):
        return None
    height, width = frame_shape
    stamps: List[_Placed] = []
    for cx, cy, radius in discs.tolist():
        reach = radius + _DISC_MARGIN_PX
        x1, y1 = max(0, cx - reach), max(0, cy - reach)
        stamp = np.zeros((min(height, cy + reach + 1) - y1, min(width, cx + reach + 1) - x1), np.uint8)
        cv2.circle(stamp, (cx - x1, cy - y1), radius, 255, -1)
        stamps.append(_Placed(stamp, x1, y1))
    return stamps


def _pack_stamps(stamps: List[_Placed], frame_shape: Tuple[int, int]) -> Optional[PackedMask]:
    """``PackedMask.pack`` of the frame mask that is the union of `stamps`, from their set pixels."""
    found = [(points.reshape(-1, 2), x, y) for points, (_stamp, x, y)
             in zip((cv2.findNonZero(stamp.mask) for stamp in stamps), stamps) if points is not None]
    if not found:
        return None
    xs = np.concatenate([points[:, 0] + x for points, x, _y in found])
    ys = np.concatenate([points[:, 1] + y for points, _x, y in found])
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    index = np.unique((ys - y1).astype(np.int64) * (x2 - x1) + (xs - x1))     # ascending, distinct
    byte = index >> 3
    starts = np.flatnonzero(np.diff(byte, prepend=-1))
    bits = np.zeros(((y2 - y1) * (x2 - x1) + 7) // 8, np.uint8)
    # The bits of one byte are distinct powers of two, so their sum is their OR: np.packbits' byte.
    bits[byte[starts]] = np.add.reduceat((128 >> (index & 7)).astype(np.uint8), starts)
    return PackedMask(bits, frame_shape[0], frame_shape[1], x1, y1, x2, y2)


def _pack_union(union_mask: np.ndarray, union: Roi, frame_shape: Tuple[int, int]) -> Optional[PackedMask]:
    """A union-box-sized mask (ROI mode) packed as that box; None when it has no set pixel."""
    if not cv2.countNonZero(union_mask):
        return None
    x1, y1, x2, y2 = union
    return PackedMask(np.packbits(union_mask.reshape(-1) > 0), frame_shape[0], frame_shape[1], x1, y1, x2, y2)


def _roi_boost(crops: List[np.ndarray], boxes: List[Roi], union: Roi, cfg: MotionSettings) -> Optional[_Placed]:
    """The ball-sized components of each box's crop (``filter_boost_mask``, box by box), assembled
    in the union box; None when none survived."""
    ux1, uy1, ux2, uy2 = union
    out = np.zeros((uy2 - uy1, ux2 - ux1), np.uint8)
    for crop, (x1, y1, x2, y2) in zip(crops, boxes):
        if cv2.countNonZero(crop):
            part = out[y1 - uy1:y2 - uy1, x1 - ux1:x2 - ux1]
            np.maximum(part, filter_boost_mask(crop, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg), out=part)
    return _Placed(out, ux1, uy1) if cv2.countNonZero(out) else None


def _frame_boost(squeezed: Optional[_SqueezedMask], frame_shape: Tuple[int, int],
                 cfg: MotionSettings) -> Optional[List[_Placed]]:
    """A stamp for each boost disc of a whole-frame mask's components; None when there are none."""
    if squeezed is None:
        return None
    return _disc_stamps(squeezed.discs(frame_shape, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg),
                        frame_shape)


class DeferredBoost(NamedTuple):
    """One frame's selector boost mask before its components are filtered: ``resolve`` returns
    what ``MotionMasks.packed_boost`` would have.  It holds its own copy of the raw motion (the
    ROI crops, or the squeezed whole-frame mask), not the pinned download buffer.  It runs after
    pass 1 has reported its timers, so its time is the reader's (the selector's)."""

    frame_shape: Tuple[int, int]
    union: Roi
    boxes: List[Roi]
    crops: Optional[List[np.ndarray]]          # ROI mode: each box's raw crop
    squeezed: Optional[_SqueezedMask]          # whole frame: the squeezed raw mask
    cfg: MotionSettings

    def resolve(self) -> Optional[PackedMask]:
        if self.crops is not None:
            placed = _roi_boost(self.crops, self.boxes, self.union, self.cfg)
            return _pack_union(placed.mask, self.union, self.frame_shape) if placed is not None else None
        stamps = _frame_boost(self.squeezed, self.frame_shape, self.cfg)
        return _pack_stamps(stamps, self.frame_shape) if stamps is not None else None


class BoostMasks(Sequence[Optional[PackedMask]]):
    """Pass 1's per-frame selector boost masks.  An entry stored as a ``DeferredBoost`` is
    filtered the first time it is read and kept from then on: the selector reads the boost only
    at frames with a detection or a motion search, so the other frames never filter theirs."""

    def __init__(self) -> None:
        self._items: List[Union[PackedMask, DeferredBoost, None]] = []

    def append(self, mask: Union[PackedMask, DeferredBoost, None]) -> None:
        self._items.append(mask)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> Optional[PackedMask]: ...

    @overload
    def __getitem__(self, index: slice) -> List[Optional[PackedMask]]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Optional[PackedMask], List[Optional[PackedMask]]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self._items)))]
        mask = self._items[index]
        if isinstance(mask, DeferredBoost):
            mask = self._items[index] = mask.resolve()
        return mask

    def __setitem__(self, index: int, mask: Optional[PackedMask]) -> None:
        self._items[index] = mask


class MotionMasks:
    """One frame's motion masks on their way to the host, and what the CPU makes of them.

    ``preprocess_frame_cuda`` cleans the gated mask on the GPU box by box,
    then copies each box's cleaned mask (and, for the detector boost, its
    ungated mask) into one pinned host buffer behind a CUDA event, so pass 1
    never waits for the copy while it preprocesses: the frame's evidence is
    read when the frame is committed, a frame later.  The first accessor
    waits for the copy and keeps the ball-sized components of each box
    (``filter_boost_mask``, box by box as before), assembled in the box that
    holds them all.

    ``roi_mode``: the boxes are motion ROIs, and the packed masks store their
    union box; otherwise the single box is the frame and the packed masks
    store the bounding box of their set pixels (``PackedMask.pack``).  A
    frame-sized mask also brings which of its rows and columns hold a set
    pixel, reduced on the device, so the host squeezes its empty rows and
    columns out (``_SqueezedMask``) instead of scanning the frame for the
    components and the packing; the masks are the same bits.
    """

    def __init__(self, shape: Tuple[int, int], boxes: List[Roi], roi_mode: bool, raw: List[torch.Tensor],
                 wide: Optional[List[torch.Tensor]], cfg: MotionSettings, perf: Optional[Dict[str, float]]) -> None:
        self.height, self.width = shape
        self.boxes = boxes
        self.roi_mode = roi_mode
        self.cfg = cfg
        self.perf = perf
        self.union = _union_box(boxes)
        if not roi_mode and self.union != (0, 0, self.width, self.height):
            raise ValueError(f"a whole-frame mask's box is the frame, not {self.union}")
        crops = raw + (wide or [])
        # Whole-frame masks: whether each row, then each column, holds a set pixel.
        occupancy = [] if roi_mode else [part for crop in crops for part in (crop.any(dim=1), crop.any(dim=0))]
        device = crops[0].device
        total = sum(int(c.numel()) for c in crops + occupancy)
        self._host = torch.empty(total, dtype=torch.uint8, pin_memory=device.type == "cuda")
        views: List[np.ndarray] = []
        offset = 0
        for crop in crops + occupancy:
            n = int(crop.numel())
            dst = self._host[offset:offset + n].view(tuple(crop.shape))
            dst.copy_(crop.view(torch.uint8), non_blocking=device.type == "cuda")
            views.append(dst.numpy())
            offset += n
        self._views, self._occupancy = views[:len(crops)], views[len(crops):]
        self._wide_distinct = wide is not None
        self._event: Optional[torch.cuda.Event] = None
        if device.type == "cuda":
            self._event = torch.cuda.Event(blocking=False)
            self._event.record(torch.cuda.current_stream(device=device))
        self._raw_resolved = False
        self._resolved = False
        self._raw = _Placed(np.empty((0, 0), np.uint8), 0, 0)    # 0/1
        self._raw_squeezed: Optional[_SqueezedMask] = None          # whole-frame masks only
        # 0/255 discs: the union box's (ROI mode), else one stamp a disc (``_disc_stamps``).
        self._narrow: Optional[List[_Placed]] = None
        self._wide: Optional[List[_Placed]] = None

    def _resolve_raw(self) -> None:
        """Wait for the download and assemble the raw mask: the union box's (ROI mode), else the
        frame's, squeezed."""
        if self._raw_resolved:
            return
        self._raw_resolved = True
        t_start = time.perf_counter()
        if self._event is not None:
            self._event.synchronize()
        if self.perf is not None:
            self.perf["pre_cuda_raw_d2h"] = self.perf.get("pre_cuda_raw_d2h", 0.0) + (time.perf_counter() - t_start)
        if self.roi_mode:
            ux1, uy1, ux2, uy2 = self.union
            raw = np.zeros((uy2 - uy1, ux2 - ux1), np.uint8)
            for crop, (x1, y1, x2, y2) in zip(self._views, self.boxes):
                part = raw[y1 - uy1:y2 - uy1, x1 - ux1:x2 - ux1]
                np.maximum(part, crop, out=part)
            self._raw = _Placed(raw, ux1, uy1)
        else:
            self._raw = _Placed(self._views[0], 0, 0)
            self._raw_squeezed = _SqueezedMask.of(self._views[0], self._occupancy[0], self._occupancy[1])
        if self.perf is not None:
            self.perf["pre_cuda_total"] = self.perf.get("pre_cuda_total", 0.0) + (time.perf_counter() - t_start)

    def _resolve(self) -> None:
        """The boost discs: the selector's (from the raw mask) and the detector's (from the wide
        mask when it was downloaded)."""
        if self._resolved:
            return
        self._resolved = True
        self._resolve_raw()
        t0 = time.perf_counter()
        cfg, shape, n_boxes = self.cfg, (self.height, self.width), len(self.boxes)
        if self.roi_mode:
            narrow = _roi_boost(self._views[:n_boxes], self.boxes, self.union, cfg)
            self._narrow = [narrow] if narrow is not None else None
            if self._wide_distinct:
                wide = _roi_boost(self._views[n_boxes:], self.boxes, self.union, cfg)
                self._wide = [wide] if wide is not None else None
        else:
            self._narrow = _frame_boost(self._raw_squeezed, shape, cfg)
            if self._wide_distinct:
                self._wide = _frame_boost(_SqueezedMask.of(self._views[1], self._occupancy[2], self._occupancy[3]),
                                          shape, cfg)
        if not self._wide_distinct:
            self._wide = self._narrow
        if self.perf is not None:
            seconds = time.perf_counter() - t0
            self.perf["pre_cuda_cc_filter"] = self.perf.get("pre_cuda_cc_filter", 0.0) + seconds
            self.perf["pre_cuda_total"] = self.perf.get("pre_cuda_total", 0.0) + seconds

    def _full(self, layers: List[_Placed], scale: int = 1) -> np.ndarray:
        """The frame-sized mask of `layers` (their union), times `scale`."""
        out = np.zeros((self.height, self.width), np.uint8)
        for mask, x, y in layers:
            part = out[y:y + mask.shape[0], x:x + mask.shape[1]]
            np.maximum(part, mask * scale if scale != 1 else mask, out=part)
        return out

    def _pack(self, layers: Optional[List[_Placed]]) -> Optional[PackedMask]:
        """A boost mask packed: the union box in ROI mode, else the bounding box of its discs."""
        if layers is None:
            return None
        if not self.roi_mode:
            return _pack_stamps(layers, (self.height, self.width))
        return _pack_union(layers[0].mask, self.union, (self.height, self.width))

    @property
    def has_blobs(self) -> bool:
        self._resolve()
        return self._narrow is not None or self._wide is not None

    def packed_raw(self) -> Optional[PackedMask]:
        """The cleaned gated raw-motion mask, packed (None when it has no set pixel)."""
        self._resolve_raw()
        if not self.roi_mode:
            return self._raw_squeezed.packed(self._raw.mask) if self._raw_squeezed is not None else None
        return _pack_union(self._raw.mask, self.union, (self.height, self.width))

    def packed_boost(self) -> Optional[PackedMask]:
        """The selector's boost mask (discs on the gated motion's ball-sized blobs), packed."""
        self._resolve()
        return self._pack(self._narrow)

    def deferred_boost(self) -> Union[PackedMask, DeferredBoost, None]:
        """``packed_boost``, left for its first reader (a ``DeferredBoost``) while the components
        are not filtered yet; None when there is no raw motion to filter."""
        if self._resolved:
            return self.packed_boost()
        self._resolve_raw()
        if self.roi_mode:
            if not cv2.countNonZero(self._raw.mask):
                return None
            crops: Optional[List[np.ndarray]] = [view.copy() for view in self._views[:len(self.boxes)]]
            squeezed = None
        else:
            if self._raw_squeezed is None:
                return None
            crops, squeezed = None, self._raw_squeezed
            if not squeezed.image.flags.owndata:             # a view of the download buffer
                squeezed = squeezed._replace(image=squeezed.image.copy())
        return DeferredBoost((self.height, self.width), self.union, self.boxes, crops, squeezed, self.cfg)

    def raw_u8(self) -> np.ndarray:
        """The cleaned gated raw-motion mask as a full-frame uint8 {0, 255} array."""
        self._resolve()
        return self._full([self._raw], 255)

    def selector_boost_u8(self) -> Optional[np.ndarray]:
        """The selector's boost mask as a full-frame uint8 array; None when no blob survived."""
        self._resolve()
        return self._full(self._narrow) if self._narrow is not None else None

    def detector_boost_u8(self) -> Optional[np.ndarray]:
        """The detector-input boost mask (from the ungated motion when it was downloaded, else the
        selector's), full-frame; None when no blob survived."""
        self._resolve()
        source = self._wide if self._wide is not None else self._narrow
        return self._full(source) if source is not None else None


def _cleaning_passes(cfg: MotionSettings) -> List[Tuple[int, bool]]:
    """``_clean_raw_motion_cuda``'s steps as ``(size, erode)`` passes: a closing (dilate, then erode)
    and an opening (erode, then dilate); a copy (a size-1 pass) when neither applies."""
    passes: List[Tuple[int, bool]] = []
    if cfg.motion_raw_close_size > 1:
        passes += [(cfg.motion_raw_close_size, False), (cfg.motion_raw_close_size, True)]
    if cfg.motion_raw_open_size > 1:
        passes += [(cfg.motion_raw_open_size, True), (cfg.motion_raw_open_size, False)]
    return passes or [(1, False)]


def _clean_by_kernel(kernels: MotionKernels, gated: torch.Tensor, boxes: List[Roi], remembered: torch.Tensor,
                     cfg: MotionSettings) -> List[torch.Tensor]:
    """Each box of `gated` cleaned by ``motion_gpu``'s morphology kernel, pass by pass; the last pass
    also sets the box's pixels in `remembered`."""
    gated_u8, remembered_u8 = gated.view(torch.uint8), remembered.view(torch.uint8)
    passes = _cleaning_passes(cfg)
    cleaned: List[torch.Tensor] = []
    for x1, y1, x2, y2 in boxes:
        src = gated_u8[y1:y2, x1:x2]
        for index, (size, erode) in enumerate(passes):
            dst = torch.empty((y2 - y1, x2 - x1), dtype=torch.uint8, device=gated.device)
            last = index == len(passes) - 1
            kernels.morph(src, dst, size, erode, merge=remembered_u8[y1:y2, x1:x2] if last else None)
            src = dst
        cleaned.append(src.view(torch.bool))
    return cleaned


def _queue_masks(gated: torch.Tensor, wide_source: Optional[torch.Tensor], rois: Optional[List[Roi]],
                 curr: GpuFrame, background: BackgroundModel, cfg: MotionSettings,
                 perf: Optional[Dict[str, float]], kernels: Optional[MotionKernels] = None) -> MotionMasks:
    """Clean the gated mask box by box on the GPU (with ``kernels``, by ``motion_gpu``'s morphology
    kernel, straight into ``BackgroundModel.motion_target``), keep the cleaned frame as the
    background's slow-adaptation mask and start the boxes' copies to the host."""
    boxes = _motion_boxes(curr, rois)
    h, w = int(gated.shape[0]), int(gated.shape[1])
    if kernels is not None:
        remembered = background.motion_target(h, w)
        cleaned = _clean_by_kernel(kernels, gated, boxes, remembered, cfg)
    else:
        cleaned = [_clean_raw_motion_cuda(gated[y1:y2, x1:x2], cfg) for x1, y1, x2, y2 in boxes]
        if rois is None:
            remembered = cleaned[0]                     # the one box is the frame
        else:
            remembered = torch.zeros((h, w), dtype=torch.bool, device=gated.device)
            for (x1, y1, x2, y2), mask in zip(boxes, cleaned):
                remembered[y1:y2, x1:x2] |= mask
    background.remember_motion(remembered)
    wide = [wide_source[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes] if wide_source is not None else None
    return MotionMasks((h, w), boxes, rois is not None, cleaned, wide, cfg, perf)


def _bgr_u8_hwc(t_chw: torch.Tensor) -> torch.Tensor:
    """A CHW float [0,1] frame as the HWC uint8 tensor the detector input writer takes."""
    return torch.clamp(t_chw * 255.0, 0, 255).byte().permute(1, 2, 0).contiguous()


def _dim_static_bgr(t_chw: torch.Tensor, raw_motion: torch.Tensor, protect: Optional[torch.Tensor],
                    cfg: MotionSettings) -> torch.Tensor:
    """The frame with every pixel away from motion scaled by ``dim_static``, as HWC uint8."""
    static = ~_dilate_motion_cuda(raw_motion, cfg.motion_dilate)
    if protect is not None:
        static = static & (~protect)
    t_out = t_chw * 255.0
    static_3d = static.unsqueeze(0).expand_as(t_out)
    t_out = torch.where(static_3d, t_out * cfg.dim_static, t_out)
    return torch.clamp(t_out, 0, 255).byte().permute(1, 2, 0).contiguous()


def _enhance_bgr(t_chw: torch.Tensor, curr_v: torch.Tensor, curr_s: torch.Tensor,
                 raw_motion: Optional[torch.Tensor], boost_u8: Optional[np.ndarray],
                 protect: Optional[torch.Tensor], cfg: MotionSettings) -> torch.Tensor:
    """Boost S/V (and shift hue) inside the boost mask and dim static pixels, as HWC uint8."""
    b, g, r = t_chw[0], t_chw[1], t_chw[2]
    maxc = curr_v
    minc = torch.min(t_chw, dim=0).values
    delta = maxc - minc
    h_t = torch.zeros_like(maxc)
    mask = delta > 1e-6
    h_r = ((g - b) / (delta + 1e-6)) % 6.0
    h_g = ((b - r) / (delta + 1e-6)) + 2.0
    h_b = ((r - g) / (delta + 1e-6)) + 4.0
    h_t = torch.where((maxc == r) & mask, h_r, h_t)
    h_t = torch.where((maxc == g) & mask, h_g, h_t)
    h_t = torch.where((maxc == b) & mask, h_b, h_t)
    h_t = (h_t / 6.0) % 1.0
    s_t = curr_s.clone()
    v_t = curr_v.clone()

    bm = None
    if boost_u8 is not None:
        bm = torch.from_numpy(boost_u8 > 0).to(t_chw.device)
        if protect is not None:
            bm = bm & (~protect)
        s_t = torch.where(bm, torch.clamp(s_t * cfg.pre_sat_boost, 0, 1), s_t)
        v_t = torch.where(bm, torch.clamp(v_t * cfg.pre_val_boost, 0, 1), v_t)

    if raw_motion is not None:
        static = ~_dilate_motion_cuda(raw_motion, cfg.motion_dilate)
        white = (s_t <= _WHITE_MAX_S) & (v_t >= _WHITE_MIN_V)
        static = static & (~white)
        if protect is not None:
            static = static & (~protect)
        v_t = torch.where(static, torch.clamp(v_t * cfg.dim_static, 0, 1), v_t)
        s_t = torch.where(static, torch.clamp(s_t * cfg.static_sat_scale, 0, 1), s_t)

    if cfg.pre_hue_shift > 0.0 and bm is not None:
        h_t = torch.where(bm, torch.clamp(
            h_t * (1 - cfg.pre_hue_shift) + _BOOST_HUE * cfg.pre_hue_shift, 0, 1), h_t)
    return _hsv_to_bgr_cuda(h_t, s_t, v_t)


def _enhanced_frame(curr: GpuFrame, raw_motion: Optional[torch.Tensor], masks: Optional[MotionMasks],
                    protect_mask: Optional[torch.Tensor], cfg: MotionSettings,
                    request: DetectorInputRequest) -> Optional[torch.Tensor]:
    """The detector input as HWC uint8 on CUDA: boosted where a ball-sized blob was found, else at
    most the static background dimmed; None when the source frame is to be used as it is.  Needs
    the boost masks now, so it waits for ``masks``."""
    t = curr.chw
    if masks is None or not masks.has_blobs:
        # No ball-sized blob: at most dim the static background.
        if raw_motion is not None and request.dim_static:
            return _dim_static_bgr(t, raw_motion, protect_mask, cfg)
        return _bgr_u8_hwc(t) if request.cuda else None
    return _enhance_bgr(t, curr.v, curr.s, raw_motion, masks.detector_boost_u8(), protect_mask, cfg)


@torch.no_grad()
def preprocess_frame_cuda(window: FrameWindow, background: BackgroundModel, cfg: MotionSettings, *,
                          request: DetectorInputRequest = DetectorInputRequest(cpu=True),
                          rois: Optional[List[Roi]] = None,
                          protect_mask: Optional[torch.Tensor] = None,
                          perf: Optional[Dict[str, float]] = None,
                          kernels: Optional[MotionKernels] = None) -> PreprocessResult:
    """Motion masks for the current frame of ``window`` and the detector input ``request`` asks for.

    ``rois`` limits the motion computation to those ``(x1, y1, x2, y2)``
    regions (None = the whole frame); ``protect_mask`` is a boolean CUDA mask
    of pixels that are never dimmed or boosted (the court-side mask).  The
    frame's cleaned raw mask becomes the background's slow-adaptation mask
    (``BackgroundModel.remember_motion``).  The masks reach the host
    asynchronously (``MotionMasks``) unless the detector input needs them now.
    ``perf`` accumulates ``pre_cuda_total`` (all of the motion work, the
    host side of the masks included), ``pre_cuda_raw_d2h`` (waiting for the
    masks' copy) and ``pre_cuda_cc_filter`` (their connected components)
    seconds when given.  With ``kernels`` the masks are built by ``motion_gpu``'s kernels, which
    give the same bits as the PyTorch ops in a few launches.
    """
    perf_t0 = time.perf_counter() if perf is not None else 0.0
    frame = window.curr
    curr = window.curr_gpu
    assert curr is not None

    raw_motion, raw_motion_ungated = _raw_motion_cuda(window, background, cfg, rois, want_ungated=request.wide_boost,
                                                      kernels=kernels)
    masks: Optional[MotionMasks] = None
    if raw_motion is None:
        background.remember_motion(None)
    else:
        wide_source = (raw_motion_ungated if request.wide_boost and raw_motion_ungated is not None
                       and raw_motion_ungated is not raw_motion else None)
        masks = _queue_masks(raw_motion, wide_source, rois, curr, background, cfg, perf, kernels)
    if perf is not None:
        perf["pre_cuda_total"] = perf.get("pre_cuda_total", 0.0) + (time.perf_counter() - perf_t0)

    enhanced: Optional[torch.Tensor] = None       # HWC uint8 on CUDA, None = the source frame
    if request.cpu or request.cuda:
        enhanced = _enhanced_frame(curr, raw_motion, masks, protect_mask, cfg, request)
    if enhanced is None:
        return PreprocessResult(frame if request.cpu else None, None, masks)
    return PreprocessResult(enhanced.cpu().numpy() if request.cpu else None,
                            enhanced if request.cuda else None, masks)
