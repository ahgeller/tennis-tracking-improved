"""tennis_tracker.render.court_layer: the cached court layer puts every pixel where
``court_drawing.draw_court_layer`` does, on any picture, through court and model changes."""
import numpy as np
import pytest

from tennis_tracker.config import Config
from tennis_tracker.court.court_drawing import draw_court_layer
from tennis_tracker.court.court_geometry import build_court_polygon, build_ground_projection_model, court_keypoints_key
from tennis_tracker.render.court_layer import CourtLayer
from tests.test_pipeline_run import COURT_KEYPOINTS

WIDTH, HEIGHT = 640, 360


def _noisy(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


def _shifted(dx: float, dy: float):
    return [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(COURT_KEYPOINTS)]


@pytest.mark.parametrize("cfg", [Config(), Config(court_draw_indices=True), Config(court_points_only=True),
                                 Config(draw_court=False)], ids=["default", "indices", "points-only", "no-court"])
def test_the_cached_layer_draws_what_draw_court_layer_draws(cfg):
    """Across frames whose court is missing, then found, then moved (by a fraction of a pixel, and
    by whole pixels), with and without a ground model, on both the main and the debug frame."""
    courts = [None, list(COURT_KEYPOINTS), list(COURT_KEYPOINTS), _shifted(0.3, 0.2),
              _shifted(9.0, -4.0), None, _shifted(9.0, -4.0)]
    polygon, _ = build_court_polygon([kps for kps in courts if kps])
    assert polygon is not None
    layer = CourtLayer(cfg, HEIGHT, WIDTH, polygon)
    for fi, kps in enumerate(courts):
        model = build_ground_projection_model(kps) if kps else None
        expected = (_noisy(fi), _noisy(100 + fi))
        actual = tuple(img.copy() for img in expected)
        draw_court_layer(expected, cfg, kps, model, polygon)
        layer.draw(actual, kps, model, court_keypoints_key(kps))
        for want, got in zip(expected, actual):
            assert np.array_equal(want, got), f"frame {fi} differs at {np.argwhere((want != got).any(axis=2))[:3]}"


def test_without_a_polygon_or_court_the_layer_leaves_the_frame_alone():
    layer = CourtLayer(Config(), HEIGHT, WIDTH, None)
    frame = _noisy(1)
    out = frame.copy()
    layer.draw((out,), None, None, None)
    assert np.array_equal(frame, out)
