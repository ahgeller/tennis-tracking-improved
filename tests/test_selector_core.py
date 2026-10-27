"""ball_in_play_selector.core helpers on synthetic masks (the scenarios are in test_selector_scenarios.py)."""
import cv2
import numpy as np
import pytest

from ball_in_play_selector import SelectorConfig
from ball_in_play_selector.core import _MIN_FILL_SPEED_PXPF, _NEAR_PLAYER_FILL_SPEED_PXPF, make_scene, motion_near


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
