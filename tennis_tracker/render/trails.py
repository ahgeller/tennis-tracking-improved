"""The trails of the tracking video and its track-debug twin: the chosen ball's recent path
(``BallTrail``: a comet on the tracking video, a thin fading line on the debug video), the
chosen track's display path (``DisplayPathTrail``) and every selector track's dots and short
trails (``CandidateTrackTrails``, at the debug video's full weight or the lighter
``LIGHT_TRACK_STYLE`` the tracking video takes with ``Config.show_all_tracks``).
Each one advances once per frame in pass 2 and draws onto the ``rendering.Canvas``.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult, Track

from ..court.court_geometry import court_axis_spans
from .rendering import (
    COLOR_DISPLAY_PATH, COLOR_DISPLAY_PATH_INTERP, COLOR_INTERP, COLOR_OUTLINE, Canvas, Color, Point,
)

# Trail colour per FrameResult.source, before the age fade.
_SOURCE_BASE_COLOR: Dict[str, Color] = {
    "det": (40, 255, 40),
    "motion": (0, 185, 255),
    "interp": COLOR_INTERP,
}
_UNKNOWN_SOURCE_COLOR: Color = (255, 220, 80)
# Trail geometry.
TRAIL_SECONDS = 0.25
TRAIL_MIN_LEN = 10
DISPLAY_PATH_TRAIL_MIN_LEN = 40
DISPLAY_PATH_TRAIL_EXTRA = 20
# The debug video's ball trail: a thin line whose colour darkens with age.
_DEBUG_TRAIL_FADE_BASE, _DEBUG_TRAIL_FADE_RANGE = 0.55, 0.45    # colour scale = base + range * position in the trail
_DEBUG_TRAIL_THICKNESS = 2
_DEBUG_TRAIL_MIN_THICKNESS = 1
# The tracking video's comet: the trail tapers and fades toward the frame from head to tail.
COMET_HEAD_THICKNESS, COMET_TAIL_THICKNESS = 4, 1
COMET_HEAD_ALPHA, COMET_TAIL_ALPHA = 1.0, 0.25
_TRAIL_HARD_MIN_X_PX, _TRAIL_HARD_MIN_Y_PX = 34.0, 26.0
_TRAIL_BRIDGE_FRAC, _TRAIL_BRIDGE_MIN_PX = 0.035, 18.0
_TRAIL_TRANSITION_FRAC, _TRAIL_TRANSITION_MIN_PX = 0.020, 14.0
_TRAIL_DIRECTION_BRIDGE_SCALE = 1.4
_MOTION_JUMP_FRAC_CAP = 0.24
_INTERP_JUMP_FRAC = 0.16
_DEFAULT_JUMP_FRAC = 0.18
_DIRECTION_BREAK_MIN_STEP_PX = 6.0
_DIRECTION_BREAK_COS = -0.32
_DIRECTION_BREAK_MIN_DY_PX = 12.0
_DISPLAY_PATH_MAX_SEGMENT_FRAC, _DISPLAY_PATH_MAX_SEGMENT_MIN_PX = 0.15, 200.0
# Candidate-track overlay.
CANDIDATE_TRAIL_LEN = 120     # the stored trail, long enough to show full rally segments
_CANDIDATE_LABEL_OFFSET: Point = (6, 6)
_TRACK_COLOR_MIN_COMPONENT, _TRACK_COLOR_TARGET = 100, 150.0


class TrailSettings(Protocol):
    """The slice of ``Config`` the ball trail reads: the court-span fractions a
    detected point may jump between frames before the trail breaks."""

    trail_hard_switch_x_frac: float
    trail_hard_switch_y_frac: float


def _trail_base_color(src: str) -> Color:
    return _SOURCE_BASE_COLOR.get(src, _UNKNOWN_SOURCE_COLOR)


def _segment_source(points: Sequence[Optional["TrailPoint"]], i: int) -> str:
    """The source that colours the segment ending on ``points[i]``: the point's own, except that a
    single motion point between two detections is a one-frame fill of the detector's track and
    takes the detector's colour; a run of two or more motion points keeps its own."""
    src = points[i][2]
    if src != "motion":
        return src
    prev = points[i - 1]
    nxt = points[i + 1] if i + 1 < len(points) else None
    if prev is not None and nxt is not None and prev[2] == "det" and nxt[2] == "det":
        return "det"
    return src


def _trail_jump_fracs(src: Optional[str], cfg: TrailSettings) -> Tuple[float, float]:
    """Court-span fractions a trail may jump between frames before it breaks, per source."""
    if src == "det":
        return cfg.trail_hard_switch_x_frac, cfg.trail_hard_switch_y_frac
    if src == "motion":
        return (min(cfg.trail_hard_switch_x_frac, _MOTION_JUMP_FRAC_CAP),
                min(cfg.trail_hard_switch_y_frac, _MOTION_JUMP_FRAC_CAP))
    if src == "interp":
        return _INTERP_JUMP_FRAC, _INTERP_JUMP_FRAC
    return _DEFAULT_JUMP_FRAC, _DEFAULT_JUMP_FRAC


def _trail_direction_break(prev2: Sequence[float], prev: Sequence[float], curr_xy: Sequence[float]) -> bool:
    """True on a strong direction reversal (a bounce or hit) between three trail points."""
    v1x = prev[0] - prev2[0]
    v1y = prev[1] - prev2[1]
    v2x = curr_xy[0] - prev[0]
    v2y = curr_xy[1] - prev[1]
    m1 = float(np.hypot(v1x, v1y))
    m2 = float(np.hypot(v2x, v2y))
    if m1 < _DIRECTION_BREAK_MIN_STEP_PX or m2 < _DIRECTION_BREAK_MIN_STEP_PX:
        return False
    cosang = (v1x * v2x + v1y * v2y) / max(m1 * m2, 1e-6)
    if cosang < _DIRECTION_BREAK_COS:
        return True
    if (np.sign(v1y) != np.sign(v2y) and abs(v1y) > _DIRECTION_BREAK_MIN_DY_PX
            and abs(v2y) > _DIRECTION_BREAK_MIN_DY_PX):
        return True
    return False


# A trail point: (x, y, source, conf); None marks a break between segments.
TrailPoint = Tuple[int, int, str, float]


class Stroke(NamedTuple):
    """One trail segment waiting to be painted by ``draw_fading_strokes``."""
    p0: Point
    p1: Point
    color: Color
    thickness: int
    newness: float      # 1.0 at the trail's head, 0.0 at the oldest point it may hold


def _paint_strokes(img: np.ndarray, offset: Point, strokes: Sequence[Stroke], outline: bool) -> None:
    """Draw the strokes on ``img`` (frame coordinates less ``offset``), in order, so a newer stroke
    lies on top where the trail crosses itself.  With ``outline`` every stroke sits on a dark line
    one pixel wider, all the outlines under all of the colour, so the outline cap of a stroke does
    not darken the end of the one it meets."""
    ox, oy = offset
    if outline:
        for stroke in strokes:
            cv2.line(img, (stroke.p0[0] - ox, stroke.p0[1] - oy), (stroke.p1[0] - ox, stroke.p1[1] - oy),
                     COLOR_OUTLINE, stroke.thickness + 1, cv2.LINE_AA)
    for stroke in strokes:
        cv2.line(img, (stroke.p0[0] - ox, stroke.p0[1] - oy), (stroke.p1[0] - ox, stroke.p1[1] - oy),
                 stroke.color, stroke.thickness, cv2.LINE_AA)


def draw_fading_strokes(target: np.ndarray, strokes: Sequence[Stroke], head_alpha: float, tail_alpha: float,
                        outline: bool) -> None:
    """Paint the strokes blended toward the frame by age: a stroke's opacity runs from
    ``tail_alpha`` at newness 0 to ``head_alpha`` at newness 1.

    At full opacity throughout, the strokes draw straight onto the frame (``_paint_strokes``).
    Otherwise they are all drawn once on a copy of the frame region they cover, each writes its
    opacity into a float mask over its whole width (the outline and the anti-aliased fringe
    included), newer over older, and the copy is blended back through the mask in one step.
    Every pixel is thus mixed with the frame exactly once, at the opacity of the newest stroke
    over it; blending the trail in bands, each over the result of the last, left a bright bead
    where a band's first stroke was blended over the end of the band before and a dark spot where
    its outline cap reached that end.
    """
    if not strokes:
        return
    if head_alpha >= 1.0 and tail_alpha >= 1.0:
        _paint_strokes(target, (0, 0), strokes, outline)
        return
    h, w = target.shape[:2]
    pad = max(stroke.thickness for stroke in strokes) + 2
    xs = [x for stroke in strokes for x in (stroke.p0[0], stroke.p1[0])]
    ys = [y for stroke in strokes for y in (stroke.p0[1], stroke.p1[1])]
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(w, max(xs) + pad + 1), min(h, max(ys) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    frame = target[y0:y1, x0:x1]
    painted = frame.copy()
    _paint_strokes(painted, (x0, y0), strokes, outline)
    alpha = np.zeros(frame.shape[:2], np.float32)
    extra = 3 if outline else 2     # the outline's pixel and the fringe anti-aliasing adds on either side
    for stroke in strokes:
        opacity = tail_alpha + (head_alpha - tail_alpha) * stroke.newness
        cv2.line(alpha, (stroke.p0[0] - x0, stroke.p0[1] - y0), (stroke.p1[0] - x0, stroke.p1[1] - y0),
                 opacity, stroke.thickness + extra, cv2.LINE_8)
    cv2.blendLinear(painted, frame, alpha, 1.0 - alpha, dst=frame)


class _SegmentLimits(NamedTuple):
    """The pixel distances beyond which a trail segment breaks (see ``BallTrail._breaks_before``)."""
    hard_x_px: float        # a jump along the court's width axis
    hard_y_px: float        # ... or its depth axis
    bridge_px: float        # how far a change of source may bridge
    transition_px: float    # ... and an interpolated-to-detected one


def _transition_too_far(dist: float, src: str, prev_src: str, limits: _SegmentLimits) -> bool:
    """A source transition connects only within a reasonable distance; too tight leaves gaps
    everywhere, too wide connects to noise."""
    if dist > limits.bridge_px:
        return True
    return prev_src == "interp" and src == "det" and dist > limits.transition_px


def _reversal_breaks(src: str, prev_src: str, dist: float, limits: _SegmentLimits) -> bool:
    """Reversals only break confident segments: not ones touching an interpolated point, nor a
    change of source within a (wider) bridging distance."""
    if src == "interp" or prev_src == "interp":
        return False
    return not (src != prev_src and dist <= limits.bridge_px * _TRAIL_DIRECTION_BRIDGE_SCALE)


class BallTrail:
    """The chosen ball's recent path, broken into segments at jumps and reversals.

    ``advance`` appends this frame's position (or a break) and ``draw`` paints the segments,
    newest brightest and thickest: a comet on the main frame, a thin line on the debug frame.
    """

    def __init__(self, cfg: TrailSettings, fps: float, frame_w: int, frame_h: int) -> None:
        self.cfg = cfg
        self.frame_w, self.frame_h = frame_w, frame_h
        self.max_len = max(TRAIL_MIN_LEN, int(round(fps * TRAIL_SECONDS)))
        self.points: List[Optional[TrailPoint]] = []
        self._pending_lost_gap = False

    def _last_two(self) -> Tuple[Optional[TrailPoint], Optional[TrailPoint]]:
        prev = prev2 = None
        for p in reversed(self.points):
            if p is None:
                continue
            if prev is None:
                prev = p
            else:
                prev2 = p
                break
        return prev, prev2

    def _segment_limits(self, src: str, prev_src: str, court_kps: Optional[Sequence[float]]) -> _SegmentLimits:
        """The distances that break a segment between a ``prev_src`` point and a ``src`` point,
        scaled by the court's extent on screen."""
        x_span, y_span = court_axis_spans(court_kps, self.frame_w, self.frame_h)
        jx_c, jy_c = _trail_jump_fracs(src, self.cfg)
        jx_p, jy_p = _trail_jump_fracs(prev_src, self.cfg)
        pair_diag = float(np.hypot(x_span, y_span))
        return _SegmentLimits(
            hard_x_px=max(max(jx_c, jx_p) * x_span, _TRAIL_HARD_MIN_X_PX),
            hard_y_px=max(max(jy_c, jy_p) * y_span, _TRAIL_HARD_MIN_Y_PX),
            bridge_px=max(_TRAIL_BRIDGE_FRAC * pair_diag, _TRAIL_BRIDGE_MIN_PX),
            transition_px=max(_TRAIL_TRANSITION_FRAC * pair_diag, _TRAIL_TRANSITION_MIN_PX),
        )

    def _breaks_before(self, x: int, y: int, src: str, court_kps: Optional[Sequence[float]]) -> bool:
        """Whether the segment ends before (x, y): too far from the last point, or a reversal."""
        prev, prev2 = self._last_two()
        if prev is None:
            return False
        prev_src = prev[2]
        limits = self._segment_limits(src, prev_src, court_kps)
        dx, dy = abs(x - prev[0]), abs(y - prev[1])
        dist = float(np.hypot(dx, dy))
        source_transition = src != prev_src
        bridged = source_transition and dist <= limits.bridge_px

        if source_transition and _transition_too_far(dist, src, prev_src, limits):
            return True
        if dx > limits.hard_x_px or dy > limits.hard_y_px:
            # A small source-transition mismatch (e.g. a reacquire) is allowed.
            return not bridged
        if prev2 is not None and _trail_direction_break(prev2, prev, (x, y)):
            return _reversal_breaks(src, prev_src, dist, limits)
        return False

    def advance(self, result: Optional[FrameResult], court_kps: Optional[Sequence[float]]) -> None:
        if result is None:
            # Nothing selected: break the segment before the next recovered point.
            if self.points and self.points[-1] is not None:
                self._pending_lost_gap = True
            self.points.append(None)
        else:
            x, y = int(result.cx), int(result.cy)
            src = result.source
            if not self._pending_lost_gap and self._breaks_before(x, y, src, court_kps):
                self.points.append(None)
            # Recovering from a fully-lost run hard-breaks the segment.
            if self._pending_lost_gap and self._last_two()[0] is not None:
                self.points.append(None)
            self._pending_lost_gap = False
            # The selector already smooths soft points: draw its path exactly so
            # estimates meet detections instead of lagging behind them.
            self.points.append((x, y, src, float(result.conf)))
        while len(self.points) > self.max_len:
            self.points.pop(0)

    def _comet_stroke(self, i: int, p0: Point, p1: Point, base: Color) -> Stroke:
        """The main frame's segment ending on ``points[i]``: its width tapers from
        ``COMET_HEAD_THICKNESS`` at the head to ``COMET_TAIL_THICKNESS`` at the age the trail
        may hold, over which ``draw_fading_strokes`` also fades it."""
        age = len(self.points) - 1 - i
        newness = 1.0 - age / max(self.max_len - 1, 1)
        thickness = int(round(COMET_TAIL_THICKNESS + (COMET_HEAD_THICKNESS - COMET_TAIL_THICKNESS) * newness))
        return Stroke(p0, p1, base, thickness, newness)

    def draw(self, canvas: Canvas, draw_main: bool) -> None:
        """Paint the segments: the debug frame always, the main frame when ``draw_main``.  Each
        segment takes its end point's source colour (``_segment_source``).  The debug frame's line
        darkens and thins with the segment's position in the trail; the main frame's comet tapers
        and blends toward the frame with the segment's age."""
        n = len(self.points)
        strokes: List[Stroke] = []
        for i in range(1, n):
            a, b = self.points[i - 1], self.points[i]
            if a is None or b is None:
                continue
            p0, p1 = (a[0], a[1]), (b[0], b[1])
            base = _trail_base_color(_segment_source(self.points, i))
            if canvas.debug is not None:
                scale = _DEBUG_TRAIL_FADE_BASE + _DEBUG_TRAIL_FADE_RANGE * (i / n)
                color = (int(base[0] * scale), int(base[1] * scale), int(base[2] * scale))
                th = max(_DEBUG_TRAIL_MIN_THICKNESS, int(_DEBUG_TRAIL_THICKNESS * scale))
                cv2.line(canvas.debug, p0, p1, COLOR_OUTLINE, th + 1, cv2.LINE_AA)
                cv2.line(canvas.debug, p0, p1, color, th, cv2.LINE_AA)
            if draw_main:
                strokes.append(self._comet_stroke(i, p0, p1, base))
        if strokes:
            draw_fading_strokes(canvas.main, strokes, COMET_HEAD_ALPHA, COMET_TAIL_ALPHA, outline=True)


# A display-path trail point: (x, y, exact); None marks a gap.
PathTrailPoint = Tuple[int, int, bool]


class DisplayPathTrail:
    """The chosen track's display path as drawn in the track-debug video.

    The path is only drawn on frames without a selected result, so the white
    line shows where the chosen track says the ball is while the selector has
    nothing; ``exact`` points (observations) are brighter and thicker.
    """

    def __init__(self, fps: float, frame_w: int) -> None:
        trail_max = max(TRAIL_MIN_LEN, int(round(fps * TRAIL_SECONDS)))
        self.max_len = max(DISPLAY_PATH_TRAIL_MIN_LEN, trail_max + DISPLAY_PATH_TRAIL_EXTRA)
        self.max_segment_px = max(_DISPLAY_PATH_MAX_SEGMENT_MIN_PX, frame_w * _DISPLAY_PATH_MAX_SEGMENT_FRAC)
        self.points: List[Optional[PathTrailPoint]] = []

    def advance(self, path_point: Optional[Tuple[float, float, bool]]) -> None:
        if path_point is not None:
            x, y, exact = path_point
            self.points.append((int(round(x)), int(round(y)), bool(exact)))
        elif self.points and self.points[-1] is not None:
            self.points.append(None)
        while len(self.points) > self.max_len:
            self.points.pop(0)

    def draw(self, debug: np.ndarray, path_point: Optional[Tuple[float, float, bool]]) -> None:
        """The path segments and the current point; call only when no result was selected."""
        for i in range(1, len(self.points)):
            p0, p1 = self.points[i - 1], self.points[i]
            if p0 is None or p1 is None:
                continue
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > self.max_segment_px:
                continue
            seg_exact = bool(p0[2] and p1[2])
            seg_color = COLOR_DISPLAY_PATH if seg_exact else COLOR_DISPLAY_PATH_INTERP
            seg_th = 2 if seg_exact else 1
            cv2.line(debug, (p0[0], p0[1]), (p1[0], p1[1]), COLOR_OUTLINE, seg_th + 1, cv2.LINE_AA)
            cv2.line(debug, (p0[0], p0[1]), (p1[0], p1[1]), seg_color, seg_th, cv2.LINE_AA)
        if path_point is not None:
            x, y, exact = path_point
            centre = (int(round(x)), int(round(y)))
            color = COLOR_DISPLAY_PATH if exact else COLOR_DISPLAY_PATH_INTERP
            radius = 6 if exact else 4
            cv2.circle(debug, centre, radius + 2, COLOR_OUTLINE, 2)
            cv2.circle(debug, centre, radius, color, -1)


def track_color(track_id: int) -> Color:
    """A stable, distinct BGR colour from a track id (never very dark)."""
    h = hashlib.md5(str(track_id).encode(), usedforsecurity=False).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    m = max(r, g, b)
    if m < _TRACK_COLOR_MIN_COMPONENT:
        scale = _TRACK_COLOR_TARGET / max(m, 1)
        r = min(255, int(r * scale))
        g = min(255, int(g * scale))
        b = min(255, int(b * scale))
    return (b, g, r)


class TrackStyle(NamedTuple):
    """The weight ``CandidateTrackTrails`` draws a frame at.  The trail segments are 1 px in
    every style; what differs is how much trail shows and for how long, the dot, its dark halo
    and the label.  The times are seconds so the trails hold the same play at any frame rate."""
    label: bool                     # the track's id and score beside its dot
    dot_radius: int                 # the ring's radius
    ring_thickness: int             # ... and its line width
    fill_chosen: bool               # the chosen track's dot is filled and 2 px wider, else every track draws alike
    halo_filled: bool               # a dark disc under the dot, else a dark 1 px ring around it
    trail_seconds: Optional[float]  # the trail is the track's last so many seconds, else the whole stored trail
    tail_alpha: float               # the trail's opacity at its oldest point, from 1.0 at the head; 1.0 is no fade
    linger_seconds: Optional[float]  # a track unseen for longer than this is not drawn at all, else it stays


# The track-debug video: ids and scores, a filled dot on the chosen track, dark discs under the dots,
# the whole stored trail at full strength for as long as the track is stored.
DEBUG_TRACK_STYLE = TrackStyle(label=True, dot_radius=3, ring_thickness=2, fill_chosen=True, halo_filled=True,
                               trail_seconds=None, tail_alpha=1.0, linger_seconds=None)
# The tracking video with ``Config.show_all_tracks``: a 2 px ring per track, no label, under the
# chosen track's own trail and marker, which keep the chosen one recognisable; a second of fading
# trail that leaves the frame half a second after the track does.
LIGHT_TRACK_STYLE = TrackStyle(label=False, dot_radius=2, ring_thickness=1, fill_chosen=False, halo_filled=False,
                               trail_seconds=1.0, tail_alpha=0.2, linger_seconds=0.5)


def _trail_strokes(points: Sequence[Optional[Tuple[float, float]]], color: Color) -> List[Stroke]:
    """One 1 px stroke per consecutive pair of points (None marks a gap), its newness the end
    point's position along ``points``."""
    span = max(len(points) - 1, 1)
    strokes: List[Stroke] = []
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        if a is None or b is None:
            continue
        strokes.append(Stroke((int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, i / span))
    return strokes


class CandidateTrackTrails:
    """Every selector track: coloured dots and short trails, with ids and scores in the
    track-debug video's ``DEBUG_TRACK_STYLE``.

    ``advance`` moves the trails to a frame once; ``draw`` paints them on any number of targets
    at a ``TrackStyle`` each; ``advance_and_draw`` does both for a single target.  ``fps`` turns
    a style's seconds into frames.
    """

    def __init__(self, tracks: Sequence[Track], chosen_track: Optional[Track], fps: float) -> None:
        self.tracks = sorted(tracks, key=lambda t: float(t.score), reverse=True)
        self.chosen_id = int(chosen_track.track_id) if chosen_track is not None else None
        self.fps = float(fps)
        # track id -> {frame: (cx, cy)}
        self.positions: Dict[int, Dict[int, Tuple[float, float]]] = {
            int(t.track_id): {o.frame: (o.cx, o.cy) for o in t.observations} for t in self.tracks}
        self.scores = {int(t.track_id): float(t.score) for t in self.tracks}
        self.trails: Dict[int, List[Optional[Tuple[float, float]]]] = {int(t.track_id): [] for t in self.tracks}
        self.last_seen: Dict[int, int] = {}     # track id -> the latest advanced frame it was observed on

    def frames(self, seconds: float) -> int:
        """A style's time as whole frames of this clip, at least one."""
        return max(1, int(round(seconds * self.fps)))

    def _lingers(self, tid: int, frame_index: int, style: TrackStyle) -> bool:
        """Whether the style still draws the track on ``frame_index``: always, unless it has a
        ``linger_seconds`` and the track has not been seen for longer than that."""
        if style.linger_seconds is None:
            return True
        last = self.last_seen.get(tid)
        return last is not None and frame_index - last <= self.frames(style.linger_seconds)

    def _trail_points(self, tid: int, frame_index: int, style: TrackStyle) -> Sequence[Optional[Tuple[float, float]]]:
        """The points the style's trail runs through, oldest first, None where the track had no
        observation: the stored trail (which holds each gap once), or the track's positions on
        the last ``trail_seconds`` of frames ending on ``frame_index``."""
        if style.trail_seconds is None:
            return self.trails[tid]
        n = max(2, self.frames(style.trail_seconds))
        positions = self.positions[tid]
        return [positions.get(f) for f in range(frame_index - n + 1, frame_index + 1)]

    def _draw_dot(self, target: np.ndarray, tid: int, pt: Tuple[float, float], color: Color,
                  style: TrackStyle) -> None:
        """The track's position this frame: a dot over a dark halo and, when the style has it,
        the id and score."""
        cx, cy = int(pt[0]), int(pt[1])
        is_chosen = style.fill_chosen and tid == self.chosen_id
        dot_r = style.dot_radius + 2 if is_chosen else style.dot_radius
        dot_th = -1 if is_chosen else style.ring_thickness
        if style.halo_filled:
            cv2.circle(target, (cx, cy), dot_r + 2, COLOR_OUTLINE, -1)   # dark outline for legibility
        else:
            cv2.circle(target, (cx, cy), dot_r + 1, COLOR_OUTLINE, 1)
        cv2.circle(target, (cx, cy), dot_r, color, dot_th)
        if not style.label:
            return
        label = f"{tid}({self.scores[tid]:.0f})"
        origin = (cx + _CANDIDATE_LABEL_OFFSET[0], cy + _CANDIDATE_LABEL_OFFSET[1])
        cv2.putText(target, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_OUTLINE, 2)
        cv2.putText(target, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def advance(self, frame_index: int) -> None:
        """Append every track's position on ``frame_index`` to its trail (a break where it has none)."""
        for trk in self.tracks:
            tid = int(trk.track_id)
            pt = self.positions[tid].get(frame_index)
            trail = self.trails[tid]
            if pt is not None:
                trail.append((pt[0], pt[1]))
                self.last_seen[tid] = frame_index
            elif trail and trail[-1] is not None:
                trail.append(None)          # a break for the gap
            while len(trail) > CANDIDATE_TRAIL_LEN:
                trail.pop(0)

    def draw(self, target: np.ndarray, frame_index: int, style: TrackStyle) -> None:
        """Paint the trails as ``advance`` left them, as much of each as the style shows and fading
        to its ``tail_alpha``, and a dot on every track seen on ``frame_index``."""
        for trk in self.tracks:
            tid = int(trk.track_id)
            if not self._lingers(tid, frame_index, style):
                continue
            pt = self.positions[tid].get(frame_index)
            color = track_color(tid)
            strokes = _trail_strokes(self._trail_points(tid, frame_index, style), color)
            if strokes:
                draw_fading_strokes(target, strokes, 1.0, style.tail_alpha, outline=False)
            if pt is not None:
                self._draw_dot(target, tid, pt, color, style)

    def advance_and_draw(self, target: np.ndarray, frame_index: int, style: TrackStyle = DEBUG_TRACK_STYLE) -> None:
        self.advance(frame_index)
        self.draw(target, frame_index, style)
