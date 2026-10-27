"""Drawing primitives for the tracking video and its track-debug twin: the colours, the
``Canvas`` that writes a mark onto both frames at once, ``paint_blended`` (a mark at partial
opacity), the static layers (players, raw detections, ROIs), the selected result's marker in
either frame's ``MarkerStyle`` and the HUD.

The court layers are in ``court_drawing``, the trails in ``trails`` and the chosen track's
display path in ``display_path``; all three draw with what is here.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult, RawDetection, Track

from ..config import Config
from ..court.court_scene import PlayerBoxes

Color = Tuple[int, int, int]
Point = Tuple[int, int]

# Colours (BGR).
COLOR_DET: Color = (0, 255, 0)
COLOR_MOTION: Color = (0, 165, 255)
COLOR_INTERP: Color = (255, 255, 0)
COLOR_RAW: Color = (0, 180, 180)
COLOR_SEARCH: Color = (255, 0, 255)
COLOR_DISPLAY_PATH: Color = (255, 255, 255)
COLOR_DISPLAY_PATH_INTERP: Color = (180, 220, 255)
COLOR_OUTLINE: Color = (15, 15, 15)          # dark halo under markers and text
COLOR_WHITE: Color = (255, 255, 255)
COLOR_PLAYER_BOX: Color = (0, 0, 255)
COLOR_COURT_POLYGON: Color = (100, 100, 100)
COLOR_ROI: Color = (0, 255, 0)
COLOR_NET_LINE: Color = (0, 255, 0)
COLOR_COURT_LINE: Color = (0, 255, 0)
COLOR_COURT_INDEX: Color = (0, 0, 255)

# Marker colour per FrameResult.source in the motion-debug video.
MOTION_DEBUG_SOURCE_COLOR: Dict[str, Color] = {
    "det": (0, 220, 60),
    "motion": (0, 160, 255),
    "interp": (0, 220, 220),
}
HUD_LABEL: Dict[str, str] = {"det": "DET", "motion": "MOT"}
HUD_LABEL_DEFAULT = "PHY"

# Marker geometry.
_DET_MARKER_DEFAULT_RADIUS = 3
_LABEL_OFFSET: Point = (10, -5)
# The glowing marker's halo: translucent discs round the filled one, outermost first, each a
# multiple of the marker radius at an opacity; where they overlap the glow builds up toward the ball.
_GLOW_HALO: Tuple[Tuple[float, float], ...] = ((2.5, 0.15), (1.9, 0.25), (1.4, 0.40))
_ANCHOR_RADIUS_EMA_KEEP, _ANCHOR_RADIUS_EMA_NEW = 0.80, 0.20   # running ball-radius estimate weights
_ANCHOR_RADIUS_CAP_SCALE = 3               # cap = ball_marker_max_radius * this
SEARCH_MARKER_SIZE = 12
_SEARCH_CENTRE_MAX_STEP_PX = 64.0          # display smoothing of the motion-search centre
# HUD.
_HUD_FRAME_POS: Point = (10, 25)
_HUD_TEXT_POS: Point = (60, 25)
_HUD_FONT_SCALE = 0.5


# ---------------------------------------------------------------------------
# Canvas: the output frame and its optional debug twin
# ---------------------------------------------------------------------------

class Canvas:
    """The pass-2 output frame and, when the track-debug video is on, its debug twin.

    Each drawing primitive is applied to every target it is given, so a
    marker that belongs on both frames is written once.  ``both`` is the
    main frame plus the debug frame (when present); ``debug_only`` is the
    debug frame alone; ``select`` picks per flag.
    """

    def __init__(self, main: np.ndarray, debug: Optional[np.ndarray]) -> None:
        self.main = main
        self.debug = debug

    @property
    def both(self) -> Tuple[np.ndarray, ...]:
        return (self.main, self.debug) if self.debug is not None else (self.main,)

    @property
    def debug_only(self) -> Tuple[np.ndarray, ...]:
        return (self.debug,) if self.debug is not None else ()

    def select(self, main: bool, debug: bool = True) -> Tuple[np.ndarray, ...]:
        targets = []
        if main:
            targets.append(self.main)
        if debug and self.debug is not None:
            targets.append(self.debug)
        return tuple(targets)


def line(targets: Iterable[np.ndarray], p0: Point, p1: Point, color: Color, thickness: int,
         line_type: int = cv2.LINE_8) -> None:
    for img in targets:
        cv2.line(img, p0, p1, color, thickness, line_type)


def circle(targets: Iterable[np.ndarray], centre: Point, radius: int, color: Color, thickness: int,
           line_type: int = cv2.LINE_8) -> None:
    for img in targets:
        cv2.circle(img, centre, radius, color, thickness, line_type)


def rectangle(targets: Iterable[np.ndarray], p0: Point, p1: Point, color: Color, thickness: int) -> None:
    for img in targets:
        cv2.rectangle(img, p0, p1, color, thickness)


def text(targets: Iterable[np.ndarray], label: str, origin: Point, scale: float, color: Color,
         thickness: int = 1) -> None:
    for img in targets:
        cv2.putText(img, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def cross(targets: Iterable[np.ndarray], centre: Point, color: Color, size: int) -> None:
    for img in targets:
        cv2.drawMarker(img, centre, color, cv2.MARKER_CROSS, size, 1)


Painter = Callable[[np.ndarray, Point], None]


def paint_blended(target: np.ndarray, box: Tuple[int, int, int, int], alpha: float, paint: Painter) -> None:
    """Show what ``paint`` draws at opacity ``alpha`` over ``target``.

    ``paint`` gets the image to draw on and the offset to subtract from frame coordinates.  At
    full opacity it draws straight onto the frame; below it, onto a copy of the frame's ``box``
    (x0, y0, x1, y1, the ends exclusive, clipped to the frame) that one ``cv2.addWeighted``
    blends back, so a mark costs a copy of its own neighbourhood and not of the frame.
    """
    if alpha >= 1.0:
        paint(target, (0, 0))
        return
    h, w = target.shape[:2]
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(w, box[2]), min(h, box[3])
    if x0 >= x1 or y0 >= y1:
        return
    roi = target[y0:y1, x0:x1]
    overlay = roi.copy()
    paint(overlay, (x0, y0))
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


# ---------------------------------------------------------------------------
# Static layers: players, raw detections, ROIs
# ---------------------------------------------------------------------------

def draw_player_boxes(targets: Iterable[np.ndarray], player_boxes: PlayerBoxes) -> None:
    targets = tuple(targets)
    for pbox in player_boxes.values():
        if pbox is None or len(pbox) < 4:
            continue
        px1, py1, px2, py2 = map(int, pbox)
        rectangle(targets, (px1, py1), (px2, py2), COLOR_PLAYER_BOX, 2)


def draw_raw_detections(targets: Iterable[np.ndarray], dets: Sequence[RawDetection]) -> None:
    """Every detector proposal as a thin box with its bottom edge emphasised."""
    targets = tuple(targets)
    for ball_box, _conf in dets:
        ix1, iy1, ix2, iy2 = map(int, ball_box)
        rectangle(targets, (ix1, iy1), (ix2, iy2), COLOR_RAW, 1)
        line(targets, (ix1, iy2), (ix2, iy2), COLOR_PLAYER_BOX, 1)


def draw_motion_rois(targets: Iterable[np.ndarray], rois: Optional[Sequence[Sequence[int]]]) -> None:
    targets = tuple(targets)
    for rx1, ry1, rx2, ry2 in rois or ():
        rectangle(targets, (int(rx1), int(ry1)), (int(rx2), int(ry2)), COLOR_ROI, 2)
        text(targets, "ROI", (int(rx1), max(15, int(ry1) - 5)), 0.4, COLOR_ROI)


# ---------------------------------------------------------------------------
# The selected result: marker, search region, HUD
# ---------------------------------------------------------------------------

class MarkerSizer:
    """Running estimate of the ball's radius from detector boxes, for marker sizes."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.radius = float(max(cfg.ball_marker_min_radius, 2))
        self._cap = float(max(cfg.ball_marker_max_radius * _ANCHOR_RADIUS_CAP_SCALE, cfg.ball_marker_min_radius))

    def observe_box(self, box_w: float, box_h: float) -> int:
        """Fold a detector box in; returns the marker radius for that detection."""
        cfg = self.cfg
        det_radius = 0.5 * max(box_w, box_h)
        self.radius = float(np.clip(
            _ANCHOR_RADIUS_EMA_KEEP * self.radius + _ANCHOR_RADIUS_EMA_NEW * det_radius,
            cfg.ball_marker_min_radius, self._cap))
        ball_px = max(int(round(box_w)), int(round(box_h)), 1)
        return int(np.clip(round(ball_px * cfg.ball_marker_box_scale),
                           cfg.ball_marker_min_radius, cfg.ball_marker_max_radius))

    @property
    def estimate_radius(self) -> int:
        """Marker radius for a position without a detector box (motion/interp), within the same
        limits as a detection's marker so the two shrink together."""
        return int(np.clip(round(self.radius), self.cfg.ball_marker_min_radius, self.cfg.ball_marker_max_radius))


class SearchCentreSmoother:
    """Display-only smoothing of the motion-search centre (the magenta cross).

    The selector's search centre can jump between frames; the overlay caps the
    jump so the cross does not flicker.  This never feeds back into selection.
    """

    def __init__(self) -> None:
        self._last: Optional[Point] = None
        self._last_frame = -1

    def smooth(self, frame_index: int, scx: int, scy: int) -> Point:
        if self._last is not None and self._last_frame == frame_index - 1:
            pscx, pscy = self._last
            dx, dy = scx - pscx, scy - pscy
            d = float((dx * dx + dy * dy) ** 0.5)
            if d > _SEARCH_CENTRE_MAX_STEP_PX and d > 1e-6:
                s = _SEARCH_CENTRE_MAX_STEP_PX / d
                scx = int(round(pscx + dx * s))
                scy = int(round(pscy + dy * s))
        self._last = (scx, scy)
        self._last_frame = frame_index
        return scx, scy

    def reset(self) -> None:
        self._last = None
        self._last_frame = -1


class MarkerStyle(NamedTuple):
    """How ``draw_selected_result`` marks the chosen position on one frame."""
    glow: bool      # a soft halo round the filled disc, else a 1 px dark ring round a detection's disc
    box: bool       # the detection box with its emphasised bottom edge
    label: bool     # the DET/MOT/PHY text beside the marker


# The tracking video: a glowing dot and nothing else, so the ball's movement shows through.
GLOW_MARKER_STYLE = MarkerStyle(glow=True, box=False, label=False)
# The track-debug video: the flat disc, the detection box and the source label.
DEBUG_MARKER_STYLE = MarkerStyle(glow=False, box=True, label=True)

# Marker colour per FrameResult.source; a source outside it gets no marker.
MARKER_SOURCE_COLOR: Dict[str, Color] = {"det": COLOR_DET, "motion": COLOR_MOTION, "interp": COLOR_INTERP}


def draw_glowing_marker(target: np.ndarray, centre: Point, radius: int, color: Color) -> None:
    """A bright glowing dot: the ``_GLOW_HALO`` discs blended in from the outside, then the
    filled disc of ``radius`` on top."""
    cx, cy = centre
    reach = int(math.ceil(radius * _GLOW_HALO[0][0])) + 2
    box = (cx - reach, cy - reach, cx + reach + 1, cy + reach + 1)
    for scale, alpha in _GLOW_HALO:
        halo_radius = int(round(radius * scale))

        def paint(img: np.ndarray, offset: Point, halo_radius: int = halo_radius) -> None:
            cv2.circle(img, (cx - offset[0], cy - offset[1]), halo_radius, color, -1, cv2.LINE_AA)

        paint_blended(target, box, alpha, paint)
    cv2.circle(target, centre, radius, color, -1, cv2.LINE_AA)


def _draw_marker(target: np.ndarray, result: FrameResult, centre: Point, radius: int, color: Color,
                 style: MarkerStyle) -> None:
    """The chosen position on one frame: in the style's order, the detection box, the disc (glowing
    or flat under its ring) and the source label."""
    if style.box and result.source == "det" and result.bbox:
        bx1, by1, bx2, by2 = (int(float(v)) for v in result.bbox)
        # High-contrast box: dark halo + bright box + emphasised bottom edge, each one pixel
        # wide so the box does not hide the ball's movement.
        cv2.rectangle(target, (bx1, by1), (bx2, by2), COLOR_OUTLINE, 2)
        cv2.rectangle(target, (bx1, by1), (bx2, by2), color, 1)
        cv2.line(target, (bx1, by2), (bx2, by2), COLOR_WHITE, 1)
    if style.glow:
        draw_glowing_marker(target, centre, radius, color)
    else:
        if result.source == "det":
            cv2.circle(target, centre, radius + 1, COLOR_OUTLINE, 1)
        cv2.circle(target, centre, radius, color, -1)
    if style.label:
        label_at = (centre[0] + _LABEL_OFFSET[0], centre[1] + _LABEL_OFFSET[1])
        cv2.putText(target, HUD_LABEL.get(result.source, HUD_LABEL_DEFAULT), label_at, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color)


def draw_selected_result(canvas: Canvas, result: FrameResult, frame_index: int, sizer: MarkerSizer,
                         search_smoother: SearchCentreSmoother, draw_search_regions: bool) -> None:
    """The chosen position, colour-coded by source: the ``GLOW_MARKER_STYLE`` on the main frame
    and the ``DEBUG_MARKER_STYLE`` on the debug frame, with its search region on the debug frame
    (and on the main frame when ``draw_search_regions``)."""
    rcx, rcy = int(result.cx), int(result.cy)
    color = MARKER_SOURCE_COLOR.get(result.source)
    if color is not None:
        # The marker sits on the measured centre, not a bottom anchor; a detection's box sets its size.
        if result.source == "det":
            radius = _DET_MARKER_DEFAULT_RADIUS
            if result.bbox:
                bx1f, by1f, bx2f, by2f = map(float, result.bbox)
                radius = sizer.observe_box(max(bx2f - bx1f, 1.0), max(by2f - by1f, 1.0))
        else:
            radius = sizer.estimate_radius
        _draw_marker(canvas.main, result, (rcx, rcy), radius, color, GLOW_MARKER_STYLE)
        for debug in canvas.debug_only:
            _draw_marker(debug, result, (rcx, rcy), radius, color, DEBUG_MARKER_STYLE)

    if result.source == "motion":
        if result.search_radius > 0:
            scx, scy = search_smoother.smooth(frame_index, int(result.search_cx), int(result.search_cy))
            targets = canvas.select(main=draw_search_regions)
            circle(targets, (scx, scy), int(result.search_radius), COLOR_SEARCH, 1)
            line(targets, (scx, scy), (rcx, rcy), COLOR_SEARCH, 1)
            cross(targets, (scx, scy), COLOR_SEARCH, SEARCH_MARKER_SIZE)
        else:
            search_smoother.reset()
    elif result.source == "interp":
        search_smoother.reset()

    # The selector's search region for det/interp frames (court-scaled radius).
    if result.source in ("det", "interp") and float(result.search_radius) > 0.0:
        scx = int(round(float(result.search_cx)))
        scy = int(round(float(result.search_cy)))
        sr = int(round(float(result.search_radius)))
        scol = COLOR_DET if result.source == "det" else COLOR_INTERP
        targets = canvas.select(main=draw_search_regions)
        circle(targets, (scx, scy), sr, COLOR_OUTLINE, 2, cv2.LINE_AA)
        circle(targets, (scx, scy), sr, scol, 1, cv2.LINE_AA)


def draw_hud(canvas: Canvas, frame_index: int, result: Optional[FrameResult],
             chosen_track: Optional[Track]) -> None:
    """Frame number on both frames; source/confidence on both when there is a result;
    the chosen track id on the debug frame."""
    frame_label = f"F{frame_index}"
    text(canvas.both, frame_label, _HUD_FRAME_POS, _HUD_FONT_SCALE, COLOR_WHITE)
    if result is not None:
        chosen_id = int(chosen_track.track_id) if chosen_track else -1
        source_label = HUD_LABEL.get(result.source, HUD_LABEL_DEFAULT)
        text(canvas.both, f"ID:{chosen_id} {source_label} c={result.conf:.2f}", _HUD_TEXT_POS,
             _HUD_FONT_SCALE, COLOR_WHITE)
        # The debug frame's label is drawn a second time: OpenCV text overdraw
        # thickens the glyph edges, and the track-debug video has always looked so.
        text(canvas.debug_only, frame_label, _HUD_FRAME_POS, _HUD_FONT_SCALE, COLOR_WHITE)
    if chosen_track is not None:
        text(canvas.debug_only, f"track={chosen_track.track_id}", _HUD_TEXT_POS, _HUD_FONT_SCALE, COLOR_WHITE)

