"""The court layer drawn once per court and stamped onto every frame, pixel for pixel what
``court_drawing.draw_court_layer`` draws.

The court changes only when the court detector runs (every ``Config.court_detect_interval``
frames), yet drawing it costs about 45 OpenCV calls per frame, half of pass 2's drawing time on
pomona.  Its opaque parts (the keypoint dots, their lines and index labels, the polygon outline)
are drawn once onto two canvases, one black and one white: a pixel is part of the layer exactly
where the two agree, and its colour is theirs, so the layer is a list of pixel indices and
colours that one fancy-index assignment writes.  The anti-aliased net line blends with what is
under it, so it is still drawn on every frame (two calls, its ends computed once), in its place
between the keypoint layer and the polygon, which keeps every pixel as the direct drawing has it.
"""
from __future__ import annotations

from typing import Callable, Iterable, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..court.court_drawing import draw_court, draw_court_polygon, draw_net_line, homography_net_line
from ..court.court_geometry import GroundModel
from .rendering import Point


class _Stamp(NamedTuple):
    """Pixels to overwrite: flat indices into an H x W frame and their BGR colours."""

    indices: np.ndarray
    colors: np.ndarray

    def apply(self, frame: np.ndarray) -> None:
        if self.indices.size:
            frame.reshape(-1, 3)[self.indices] = self.colors


def _stamp_of(height: int, width: int, draw: Callable[[np.ndarray], object]) -> _Stamp:
    """What ``draw`` (opaque marks only) puts on an ``height`` x ``width`` frame, as a ``_Stamp``."""
    black = np.zeros((height, width, 3), np.uint8)
    white = np.full((height, width, 3), 255, np.uint8)
    draw(black)
    draw(white)
    drawn = np.flatnonzero((black == white).all(axis=2))
    return _Stamp(drawn, black.reshape(-1, 3)[drawn].copy())


class CourtLayer:
    """``draw_court_layer`` for one renderer, its drawing cached per court.

    ``court_polygon`` is the run's one polygon; ``model_key`` names the ground model passed to
    ``draw`` (the keypoints it was built from, ``court_keypoints_key``), so the cached net line
    is rebuilt exactly when the model is.
    """

    def __init__(self, cfg: Config, height: int, width: int, court_polygon: Optional[np.ndarray]) -> None:
        self.cfg = cfg
        self.height, self.width = int(height), int(width)
        self._polygon = (_stamp_of(self.height, self.width, lambda img: draw_court_polygon(img, court_polygon))
                         if court_polygon is not None else None)
        self._points_key: Optional[Tuple[int, ...]] = None
        self._points: Optional[_Stamp] = None
        self._model_key: object = object()        # never equal to a real key: the first call builds
        self._net: Optional[Tuple[Point, Point]] = None

    def _points_for(self, court_kps: Optional[Sequence[float]]) -> Optional[_Stamp]:
        if not (self.cfg.draw_court and court_kps):
            return None
        # draw_court reads each value through int(), so the truncated values are the whole key.
        key = tuple(int(value) for value in court_kps)
        if key != self._points_key:
            self._points = _stamp_of(self.height, self.width, lambda img: draw_court(img, court_kps, self.cfg))
            self._points_key = key
        return self._points

    def _net_for(self, ground_model: Optional[GroundModel], model_key: object) -> Optional[Tuple[Point, Point]]:
        if model_key != self._model_key:
            self._net = homography_net_line(ground_model, self.width, self.height)
            self._model_key = model_key
        return self._net

    def draw(self, targets: Iterable[np.ndarray], court_kps: Optional[Sequence[float]],
             ground_model: Optional[GroundModel], model_key: object) -> None:
        """Draw the court layer onto each of ``targets`` (frames of this layer's size)."""
        points = self._points_for(court_kps)
        net = self._net_for(ground_model, model_key) if ground_model is not None else None
        for img in targets:
            if points is not None:
                points.apply(img)
            if net is not None:
                draw_net_line(img, net)
            if self._polygon is not None:
                self._polygon.apply(img)
