"""ball_in_play_selector.core helpers on synthetic masks (the scenarios are in test_selector_scenarios.py)."""
import cv2
import numpy as np
import pytest

from ball_in_play_selector import SelectorConfig
from ball_in_play_selector.config import REFERENCE_DIAG_PX
from ball_in_play_selector.core import (_MIN_FILL_SPEED_PXPF, _NEAR_PLAYER_FILL_SPEED_PXPF, _find_spikes, _px,
                                        make_scene, motion_near)
from ball_in_play_selector.physics import BallKalmanFilter


def test_small_motion_component_uses_pixel_area():
    mask = np.zeros((100, 100), np.uint8)
    mask[49:52, 49:52] = 255
    result = motion_near(mask, 50, 50, 175., 20., 4.)
    assert (result.x, result.y, result.area) == (50., 50., 9.)
    assert motion_near(mask, 0, 0, 175., 20., 4.) is None


def test_motion_near_on_a_packed_mask_matches_the_unpacked_frame():
    """The packed rectangle holds every set pixel, so the components found in it are the frame's,
    offset back by its corner: the same blob, centre, bbox and area as on the unpacked mask."""
    from ball_in_play_selector import PackedMask
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    cv2.circle(mask, (900, 600), 4, 255, -1)          # the ball-sized blob
    mask[590:620, 700:760] = 255                        # a large one, outside the area limit
    mask[0:3, 0:3] = 255                                # a far one
    for roi in (None, (680, 560, 940, 640), (0, 0, 1920, 1080)):
        packed = PackedMask.pack(mask, roi=roi)
        from_packed = motion_near(packed, 905., 603., 40., 25., 4.)
        from_frame = motion_near(packed.unpack(), 905., 603., 40., 25., 4.)
        assert from_packed is not None and from_packed == from_frame
        assert from_packed.bbox == (896, 596, 905, 605)
    assert motion_near(PackedMask.pack(mask), 100., 100., 40., 25., 4.) is None


def _scene(fps, width, height):
    cfg = SelectorConfig(fps=fps, width=width, height=height)
    return make_scene(cfg, fps, 100, width, height, None, None, None)


def test_the_fill_speed_floors_follow_the_frame_rate_and_the_resolution():
    """The floors are tuned in 1080p pixels per 30 FPS frame. A 720p ball moving the same way
    covers two thirds of the pixels, and at 60 FPS half as many a frame, so the same clip reads
    the same against them at any size or rate."""
    reference = _scene(30.0, 1920, 1080)        # the reference diagonal is 2203 px, 1920x1080 a hair under
    approx = lambda value: pytest.approx(value, rel=1e-3)
    assert (reference.min_fill_speed, reference.near_player_fill_speed) == approx((_MIN_FILL_SPEED_PXPF, _NEAR_PLAYER_FILL_SPEED_PXPF))
    hd_ready = _scene(30.0, 1280, 720)
    assert hd_ready.min_fill_speed == approx(reference.min_fill_speed * (1280 / 1920))
    assert hd_ready.near_player_fill_speed == approx(reference.near_player_fill_speed * (1280 / 1920))
    fast = _scene(60.0, 1920, 1080)
    assert fast.min_fill_speed == approx(reference.min_fill_speed / 2)
    assert fast.near_player_fill_speed == approx(reference.near_player_fill_speed / 2)
    assert fast.fill_frames == 2 * reference.fill_frames == 18


HD_READY_DIAG = (1280 ** 2 + 720 ** 2) ** 0.5


def test_pixel_constants_are_read_in_the_clip_s_pixels():
    """A gate tuned in 1080p pixels is a fraction of the picture: at 720p it is two thirds the
    count of pixels. The spike finder's floor shows it: an 8 px excursion off a 1 px/frame path is
    a spike at 720p (floor 6.7 px) and not at 1080p (floor 10 px)."""
    assert _px(10.0, REFERENCE_DIAG_PX) == 10.0
    assert _px(10.0, HD_READY_DIAG) == pytest.approx(10.0 * 1280 / 1920, rel=1e-3)
    x = np.arange(12, dtype=float)
    y = np.zeros(12)
    y[6] = 8.0
    assert not _find_spikes(x, y, REFERENCE_DIAG_PX).any()
    assert list(np.flatnonzero(_find_spikes(x, y, HD_READY_DIAG))) == [6]


def test_the_bounce_heuristic_s_thresholds_follow_the_resolution():
    """A ball descending 1.5 px/frame whose measurement lands 8 px above the prediction: at 720p
    that is the 2 px/frame and 10 px of 1080p scaled down, so vy flips to the rebound; at 1080p
    neither threshold is met and vy is left to the update (which, with no position-velocity
    covariance yet, leaves it alone)."""
    for width, height, flipped in ((1280, 720, True), (1920, 1080, False)):
        kf = BallKalmanFilter(0.0, 0.0, SelectorConfig(fps=30.0, width=width, height=height))
        kf.kf.x[3, 0] = 1.5
        kf.update(0.0, -8.0, conf=0.9)
        assert (kf.get_velocity()[1] < 0.0) is flipped
