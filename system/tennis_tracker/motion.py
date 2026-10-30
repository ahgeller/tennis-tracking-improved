"""Motion evidence for pass 1: the background model, raw and boost motion
masks on CUDA, the court-side protect mask and the detector-input enhancement."""
from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple, Union

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
    off_x, off_y, box_w, box_h = cv2.boundingRect(raw_motion)
    if box_w == 0 or box_h == 0:
        return result

    img_diag = math.hypot(float(img_w), float(img_h))
    ball_bypass_max_area = max(_BALL_BYPASS_AREA_MIN_PX,
                               min(_BALL_BYPASS_AREA_MAX_PX, int(round(_BALL_BYPASS_AREA_DIAG_FRAC * img_diag))))
    if cfg.blob_preserve_tiny:
        ball_bypass_max_area = max(ball_bypass_max_area, cfg.blob_tiny_max_area)

    # A centroid in the box plus the box's offset rounds to the same pixel as the frame's centroid:
    # both are an integer sum over the area, and they could only round apart within an ulp of .5,
    # which a ratio with that denominator is either exactly on or at least 1 / (2 * area) away from.
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        raw_motion[off_y:off_y + box_h, off_x:off_x + box_w], connectivity=8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if area <= ball_bypass_max_area:
            # Small ball blobs get the loosest aspect test.
            if aspect > (_TINY_BLOB_MAX_ASPECT if area < _TINY_BLOB_AREA_PX else _SMALL_BLOB_MAX_ASPECT):
                continue
        elif cfg.blob_shape_filter and aspect > cfg.blob_max_aspect:
            continue
        # A disc at the centroid; its radius is capped so motion blur and
        # shadows cannot balloon the boost far beyond the ball.
        cx = int(centroids[i][0] + off_x + 0.5)
        cy = int(centroids[i][1] + off_y + 0.5)
        radius = min(int(math.sqrt(area / math.pi) + 0.5), _BOOST_DISC_MAX_RADIUS_PX)
        cv2.circle(result, (cx, cy), max(radius, 1), 255, -1)
    return result


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

    def motion_target(self, height: int, width: int) -> torch.Tensor:
        """A zeroed ``height`` x ``width`` boolean mask for this frame's cleaned motion to be written
        into and then handed to ``remember_motion``; with CUDA graphs it is the one buffer the
        recorded updates read, which ``remember_motion`` then takes without a copy."""
        if not self.cuda_graphs:
            return torch.zeros((height, width), dtype=torch.bool, device=self.device)
        if self._motion_buffer is None or tuple(self._motion_buffer.shape) != (height, width):
            self._motion_buffer = torch.zeros((height, width), dtype=torch.bool, device=self.device)
            self._update_graphs.clear()
        else:
            self._motion_buffer.zero_()
        return self._motion_buffer

    def remember_motion(self, raw_motion: Optional[torch.Tensor]) -> None:
        """Keep this frame's cleaned raw motion (a boolean frame-sized tensor, or None when there
        is none yet) for the next update's slow-adaptation mask.  The tensor is used as it is,
        not copied: the caller hands over a fresh one each frame (or ``motion_target``'s); with
        CUDA graphs any other tensor is copied into the one buffer the recorded updates read."""
        if self.cuda_graphs and raw_motion is not None and raw_motion is not self._motion_buffer:
            if self._motion_buffer is None or self._motion_buffer.shape != raw_motion.shape:
                self._motion_buffer = torch.empty_like(raw_motion)
                self._update_graphs.clear()
            self._motion_buffer.copy_(raw_motion)
            raw_motion = self._motion_buffer
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
                     rois: Optional[List[Roi]], want_ungated: bool = True
                     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(gated, ungated)`` boolean raw-motion masks for the current frame.

    ``ungated`` is the background-model foreground inside the ROIs (the whole
    frame without them); ``gated`` additionally passes the frame-to-frame
    temporal test when it is enabled and a previous frame exists.  Both are
    None until the background model is ready; they are the same tensor when
    the gate did not run, and ``ungated`` is None when it is not wanted and
    the gate ran.

    The foreground and the two probe deltas are computed once, in the box that
    holds every ROI; each ROI then blurs its own part of the deltas, with zero
    padding at the ROI's edges as when each ROI was computed on its own, so
    the masks are the same bit for bit.
    """
    if not background.ready:
        return None, None
    curr = window.curr_gpu
    assert curr is not None
    h, w = int(curr.v.shape[0]), int(curr.v.shape[1])
    boxes = _motion_boxes(curr, rois)
    ux1, uy1, ux2, uy2 = _union_box(boxes)
    fg = background.foreground(curr, (ux1, uy1, ux2, uy2))
    prev, nxt = window.prev_gpu, window.next_gpu
    gate = cfg.motion_raw_temporal_gate and prev is not None
    ungated: Optional[torch.Tensor] = None
    if want_ungated or not gate:
        ungated = torch.zeros((h, w), dtype=torch.bool, device=fg.device)
        for x1, y1, x2, y2 in boxes:
            ungated[y1:y2, x1:x2] = fg[y1 - uy1:y2 - uy1, x1 - ux1:x2 - ux1]
    if not gate:
        return ungated, ungated
    assert prev is not None
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
    store the bounding box of their set pixels (``PackedMask.pack``).
    """

    def __init__(self, shape: Tuple[int, int], boxes: List[Roi], roi_mode: bool, raw: List[torch.Tensor],
                 wide: Optional[List[torch.Tensor]], cfg: MotionSettings, perf: Optional[Dict[str, float]]) -> None:
        self.height, self.width = shape
        self.boxes = boxes
        self.roi_mode = roi_mode
        self.cfg = cfg
        self.perf = perf
        self.union = _union_box(boxes)
        crops = raw + (wide or [])
        device = crops[0].device
        total = sum(int(c.numel()) for c in crops)
        self._host = torch.empty(total, dtype=torch.uint8, pin_memory=device.type == "cuda")
        self._views: List[np.ndarray] = []
        offset = 0
        for crop in crops:
            n = int(crop.numel())
            dst = self._host[offset:offset + n].view(tuple(crop.shape))
            dst.copy_(crop.view(torch.uint8), non_blocking=device.type == "cuda")
            self._views.append(dst.numpy())
            offset += n
        self._wide_distinct = wide is not None
        self._event: Optional[torch.cuda.Event] = None
        if device.type == "cuda":
            self._event = torch.cuda.Event(blocking=False)
            self._event.record(torch.cuda.current_stream(device=device))
        self._resolved = False
        self._raw: np.ndarray = np.empty((0, 0), np.uint8)       # 0/1, union-box sized
        self._narrow: Optional[np.ndarray] = None                  # 0/255 discs, union-box sized
        self._wide: Optional[np.ndarray] = None


    def _resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        t_start = time.perf_counter()
        if self._event is not None:
            self._event.synchronize()
        if self.perf is not None:
            self.perf["pre_cuda_raw_d2h"] = self.perf.get("pre_cuda_raw_d2h", 0.0) + (time.perf_counter() - t_start)
        ux1, uy1, ux2, uy2 = self.union
        size = (uy2 - uy1, ux2 - ux1)
        raw = np.zeros(size, np.uint8)
        narrow = np.zeros(size, np.uint8)
        wide = np.zeros(size, np.uint8) if self._wide_distinct else None
        cfg, n_boxes = self.cfg, len(self.boxes)
        min_area, max_area = cfg.boost_min_blob_area, cfg.boost_max_blob_area
        cc_seconds = 0.0
        for index, (x1, y1, x2, y2) in enumerate(self.boxes):
            part = (slice(y1 - uy1, y2 - uy1), slice(x1 - ux1, x2 - ux1))
            crop = self._views[index]
            np.maximum(raw[part], crop, out=raw[part])
            if cv2.countNonZero(crop):
                t0 = time.perf_counter()
                np.maximum(narrow[part], filter_boost_mask(crop, min_area, max_area, cfg), out=narrow[part])
                cc_seconds += time.perf_counter() - t0
            if wide is not None:
                wide_crop = self._views[n_boxes + index]
                if cv2.countNonZero(wide_crop):
                    np.maximum(wide[part], filter_boost_mask(wide_crop, min_area, max_area, cfg), out=wide[part])
        self._raw = raw
        self._narrow = narrow if cv2.countNonZero(narrow) else None
        self._wide = (wide if cv2.countNonZero(wide) else None) if wide is not None else self._narrow
        if self.perf is not None:
            self.perf["pre_cuda_cc_filter"] = self.perf.get("pre_cuda_cc_filter", 0.0) + cc_seconds
            self.perf["pre_cuda_total"] = self.perf.get("pre_cuda_total", 0.0) + (time.perf_counter() - t_start)

    def _full(self, union_mask: np.ndarray, scale: int = 1) -> np.ndarray:
        out = np.zeros((self.height, self.width), np.uint8)
        ux1, uy1, ux2, uy2 = self.union
        out[uy1:uy2, ux1:ux2] = union_mask * scale if scale != 1 else union_mask
        return out

    def _pack(self, union_mask: Optional[np.ndarray]) -> Optional[PackedMask]:
        if union_mask is None:
            return None
        if not self.roi_mode:
            return PackedMask.pack(union_mask)         # the single box is the frame
        if not cv2.countNonZero(union_mask):
            return None
        x1, y1, x2, y2 = self.union
        return PackedMask(np.packbits(union_mask.reshape(-1) > 0), self.height, self.width, x1, y1, x2, y2)

    @property
    def has_blobs(self) -> bool:
        self._resolve()
        return self._narrow is not None or self._wide is not None

    def packed_raw(self) -> Optional[PackedMask]:
        """The cleaned gated raw-motion mask, packed (None when it has no set pixel)."""
        self._resolve()
        return self._pack(self._raw)

    def packed_boost(self) -> Optional[PackedMask]:
        """The selector's boost mask (discs on the gated motion's ball-sized blobs), packed."""
        self._resolve()
        return self._pack(self._narrow)

    def raw_u8(self) -> np.ndarray:
        """The cleaned gated raw-motion mask as a full-frame uint8 {0, 255} array."""
        self._resolve()
        return self._full(self._raw, 255)

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


def _queue_masks(gated: torch.Tensor, wide_source: Optional[torch.Tensor], rois: Optional[List[Roi]],
                 curr: GpuFrame, background: BackgroundModel, cfg: MotionSettings,
                 perf: Optional[Dict[str, float]]) -> MotionMasks:
    """Clean the gated mask box by box on the GPU, keep the cleaned frame as the background's
    slow-adaptation mask and start the boxes' copies to the host."""
    boxes = _motion_boxes(curr, rois)
    cleaned = [_clean_raw_motion_cuda(gated[y1:y2, x1:x2], cfg) for x1, y1, x2, y2 in boxes]
    h, w = int(gated.shape[0]), int(gated.shape[1])
    if rois is None:
        remembered = cleaned[0]                     # the one box is the frame
    else:
        remembered = torch.zeros((h, w), dtype=torch.bool, device=gated.device)
        for (x1, y1, x2, y2), mask in zip(boxes, cleaned):
            remembered[y1:y2, x1:x2] |= mask
    background.remember_motion(remembered)
    wide = [wide_source[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes] if wide_source is not None else None
    return MotionMasks((h, w), boxes, rois is not None, cleaned, wide, cfg, perf)


def _gate_scalars(cfg: MotionSettings) -> GateScalars:
    """The thresholds ``BackgroundModel.foreground`` and ``_raw_motion_cuda`` apply, for the kernels."""
    return GateScalars(k_std_sq=cfg.motion_k_std ** 2, thresh_sq=(cfg.motion_thresh / 255.0) ** 2,
                       s_scale=_S_THRESHOLD_SCALE, v_min=cfg.motion_v_min / 255.0, hi=cfg.motion_raw_temporal_hi,
                       lo=cfg.motion_raw_temporal_lo, very_hi=cfg.motion_raw_temporal_very_hi)


def _raw_motion_kernels(kernels: MotionKernels, window: FrameWindow, background: BackgroundModel,
                        cfg: MotionSettings, rois: Optional[List[Roi]], want_ungated: bool = True
                        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``_raw_motion_cuda`` with ``motion_gpu``'s gate kernel: the same masks, one launch a box."""
    if not background.ready:
        return None, None
    assert background.mean is not None and background.var is not None
    curr = window.curr_gpu
    assert curr is not None
    h, w = int(curr.v.shape[0]), int(curr.v.shape[1])
    prev, nxt = window.prev_gpu, window.next_gpu
    gate = cfg.motion_raw_temporal_gate and prev is not None
    gated = torch.zeros((h, w), dtype=torch.uint8, device=curr.v.device)
    ungated = (torch.zeros((h, w), dtype=torch.uint8, device=curr.v.device)
               if gate and want_ungated else None)
    scalars = _gate_scalars(cfg)
    for box in _motion_boxes(curr, rois):
        kernels.gate(curr.planes, prev.planes if gate and prev is not None else None,
                     nxt.planes if gate and nxt is not None else None, background.mean, background.var, box,
                     gated, ungated, scalars, temporal=gate)
    gated_mask = gated.view(torch.bool)
    if not gate:
        return gated_mask, gated_mask
    return gated_mask, ungated.view(torch.bool) if ungated is not None else None


def _cleaning_passes(cfg: MotionSettings) -> List[Tuple[int, bool]]:
    """``_clean_raw_motion_cuda``'s steps as ``(size, erode)`` passes: a closing (dilate, then erode)
    and an opening (erode, then dilate); a copy (a size-1 pass) when neither applies."""
    passes: List[Tuple[int, bool]] = []
    if cfg.motion_raw_close_size > 1:
        passes += [(cfg.motion_raw_close_size, False), (cfg.motion_raw_close_size, True)]
    if cfg.motion_raw_open_size > 1:
        passes += [(cfg.motion_raw_open_size, True), (cfg.motion_raw_open_size, False)]
    return passes or [(1, False)]


def _queue_masks_kernels(kernels: MotionKernels, gated: torch.Tensor, wide_source: Optional[torch.Tensor],
                         rois: Optional[List[Roi]], curr: GpuFrame, background: BackgroundModel,
                         cfg: MotionSettings, perf: Optional[Dict[str, float]]) -> MotionMasks:
    """``_queue_masks`` with ``motion_gpu``'s morphology kernel: each box is cleaned pass by pass,
    the last pass writing the box's mask and setting its pixels in the background's slow-adaptation
    mask (``BackgroundModel.motion_target``)."""
    boxes = _motion_boxes(curr, rois)
    h, w = int(gated.shape[0]), int(gated.shape[1])
    gated_u8 = gated.view(torch.uint8)
    remembered = background.motion_target(h, w)
    remembered_u8 = remembered.view(torch.uint8)
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

    if kernels is not None:
        raw_motion, raw_motion_ungated = _raw_motion_kernels(kernels, window, background, cfg, rois,
                                                             want_ungated=request.wide_boost)
    else:
        raw_motion, raw_motion_ungated = _raw_motion_cuda(window, background, cfg, rois,
                                                          want_ungated=request.wide_boost)
    masks: Optional[MotionMasks] = None
    if raw_motion is None:
        background.remember_motion(None)
    else:
        wide_source = (raw_motion_ungated if request.wide_boost and raw_motion_ungated is not None
                       and raw_motion_ungated is not raw_motion else None)
        if kernels is not None:
            masks = _queue_masks_kernels(kernels, raw_motion, wide_source, rois, curr, background, cfg, perf)
        else:
            masks = _queue_masks(raw_motion, wide_source, rois, curr, background, cfg, perf)
    if perf is not None:
        perf["pre_cuda_total"] = perf.get("pre_cuda_total", 0.0) + (time.perf_counter() - perf_t0)

    enhanced: Optional[torch.Tensor] = None       # HWC uint8 on CUDA, None = the source frame
    if request.cpu or request.cuda:
        enhanced = _enhanced_frame(curr, raw_motion, masks, protect_mask, cfg, request)
    if enhanced is None:
        return PreprocessResult(frame if request.cpu else None, None, masks)
    return PreprocessResult(enhanced.cpu().numpy() if request.cpu else None,
                            enhanced if request.cuda else None, masks)
