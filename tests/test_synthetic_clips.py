"""finetune/training/synthetic_clips.py: track windows, placement, velocities, the plate test, the
rendering and the label rows the trainer reads through --pseudo."""
import random

import numpy as np
import pytest

from finetune.training import copy_paste, synthetic_clips
from finetune.training.frame_cache import build_units
from finetune.training.pseudo_labels import BALL, NONE, read_pseudo_labels, write_pseudo_labels
from finetune.training.synthetic_clips import (ball_free, clip_tracks, label_rows, place_track, render_clip,
                                               track_velocities, track_windows)
from tennis_tracker.gridtracknet import HEIGHT, WIDTH


def disc_bank() -> copy_paste.Banks:
    size = 2 * copy_paste.CROP_RADIUS + 1
    ys, xs = np.mgrid[0:size, 0:size] + 0.5
    disc = (np.hypot(xs - size / 2, ys - size / 2) <= 2.0).astype(np.float32)
    path = np.zeros((1, 5, 2), np.float32)
    return copy_paste.Banks.of(np.ones((1, size, size, 3), np.float32), disc[None], np.full((1, 2), size / 2, np.float32),
                               np.array([200.0], np.float32), ["c"], path, np.zeros((1, 2), np.float32), np.ones(1, np.int32))


def test_clip_tracks_reads_a_cache_record_in_cache_pixels():
    record = {"stride": 2, "samples": [{"indices": [1, 3, 5, 7, 9], "size": [1920, 1080],
                                        "points": [[1920.0, 1080.0], None, [960.0, 540.0], None, None]}]}
    points, stride = clip_tracks(record)
    assert stride == 2 and points[1] == pytest.approx((WIDTH, HEIGHT)) and points[3] is None
    assert points[5] == pytest.approx((WIDTH / 2, HEIGHT / 2))


def test_track_windows_need_consecutive_frames_and_enough_visible():
    points = {index: (float(index), 10.0) for index in range(0, 40, 2)}
    windows = track_windows(points, 2, 8)
    assert windows and all(len(window) == 8 for window in windows)
    points[10] = None
    points[12] = None
    points[14] = None
    points[16] = None
    assert all(sum(p is not None for p in w) >= synthetic_clips.MIN_VISIBLE * 8 for w in track_windows(points, 2, 8))
    del points[20]                                    # a gap: no window may span frame 20
    for window in track_windows(points, 2, 8):
        assert (20.0, 10.0) not in window


def test_place_track_keeps_the_shape_inside_the_frame_and_absent_frames_absent():
    track = [(100.0 + 10 * i, 200.0 + i * i) if i != 3 else None for i in range(8)]
    for seed in range(50):
        placed = place_track(track, random.Random(seed))
        assert placed is not None and placed[3] is None
        seen = np.array([p for p in placed if p is not None])
        assert seen[:, 0].min() >= synthetic_clips.EDGE_PX - 1e-6 and seen[:, 0].max() <= WIDTH - synthetic_clips.EDGE_PX + 1e-6
        steps = np.abs(np.diff(seen[:, 0]))
        assert np.allclose(steps[:2], 10.0)          # moved and maybe mirrored, never scaled
        assert abs(seen[0, 1] - 200.0) <= copy_paste.PASTE_DY_PX + 1e-6
        assert min(abs(seen[0, 0] - 100.0), abs(seen[0, 0] - (WIDTH - 100.0))) <= synthetic_clips.MAX_DX_PX + 1e-6


def test_velocities_are_central_differences_per_camera_frame():
    track = [(0.0, 0.0), (4.0, 2.0), (8.0, 4.0), None, (20.0, 0.0)]
    v = track_velocities(track, 2.0)
    assert v[1] == pytest.approx((2.0, 1.0)) and v[0] == pytest.approx((2.0, 1.0))
    assert v[2] == pytest.approx((2.0, 1.0)) and v[3] == (0.0, 0.0) and v[4] == (0.0, 0.0)


def test_pick_crop_stays_in_the_pool_and_prefers_the_height():
    one = disc_bank()
    bank = copy_paste.Banks.of(np.repeat(one.crops, 3, 0), np.repeat(one.mattes, 3, 0), np.repeat(one.centres, 3, 0),
                               np.array([50.0, 200.0, 400.0], np.float32), ["a", "b", "c"], one.paths, one.middles, one.strides)
    draws = random.Random(0)
    assert {synthetic_clips.pick_crop(draws, bank, [0, 2], 210.0) for _ in range(20)} <= {0, 2}
    assert synthetic_clips.pick_crop(draws, bank, [0, 1, 2], 390.0) == 2
    assert synthetic_clips.pick_crop(draws, bank, [0, 1, 2], 205.0) == 1


def test_ball_free_is_a_strict_ceiling():
    assert ball_free([0.0, 0.29]) and not ball_free([0.1, 0.3])


def test_render_clip_draws_only_where_the_track_has_a_ball():
    plate = np.zeros((3, HEIGHT, WIDTH, 3), np.uint8)
    out = render_clip(plate, [(100.5, 200.5), None, (300.5, 210.5)], disc_bank(), 0, 0.5, 2.0)
    assert out.dtype == np.uint8 and out[1].max() == 0
    assert out[0, 200, 100].max() > 200 and out[2, 210, 300].max() > 200 and out[0, 210, 300].max() == 0


def test_label_rows_round_trip_into_training_units(tmp_path):
    track = [(10.0, 20.0)] * 4 + [None] + [(30.0, 40.0)] * 5
    path = tmp_path / "labels" / "s_pseudo.csv"
    path.parent.mkdir()
    write_pseudo_labels(path, label_rows(track))
    labels, ignored = read_pseudo_labels(path, WIDTH, HEIGHT)
    assert not ignored and labels[4] is None and labels[0] == [10.0, 20.0]
    assert label_rows(track)[4] == (NONE, None) and label_rows(track)[0][0] == BALL
    assert build_units(labels, 1)[0] == [0, 1, 2, 3, 4]
