"""finetune/training/copy_paste.py: the streak kernel, the rendering, the banks cut from a cached
clip, the paste draws, and the paste through the dataset into the frames and the target."""
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from finetune.training import copy_paste
from finetune.training.copy_paste import (Banks, Paste, apply_paste, build_banks, draw_paste, load_banks, pick_crop,
                                          render_ball, streak_kernel)
from finetune.training.dataset import Augmentation, UnitDataset
from finetune.training.frame_cache import TrainingUnit
from tennis_tracker.gridtracknet import HEIGHT, WIDTH
from tennis_tracker.heatmapnet import ARCH, GRID_COLS

COUNT = 40


def disc_bank(rows=(50.0,), clips=("c",)) -> Banks:
    """One white disc crop per row in `rows`, its matte the disc, and one straight path."""
    size = 2 * copy_paste.CROP_RADIUS + 1
    ys, xs = np.mgrid[0:size, 0:size] + 0.5
    disc = (np.hypot(xs - size / 2, ys - size / 2) <= 2.0).astype(np.float32)
    count = len(rows)
    crops = np.ones((count, size, size, 3), np.float32)
    path = np.array([[-8.0, 0.0], [-4.0, 0.0], [0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], np.float32)
    return Banks.of(crops, np.stack([disc] * count), np.full((count, 2), size / 2, np.float32), np.array(rows, np.float32),
                    list(clips), path[None], np.array([[300.0, 200.0]], np.float32), np.array([1], np.int32))


def centroid(image: np.ndarray) -> tuple:
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]] + 0.5
    return float((image * xs).sum() / image.sum()), float((image * ys).sum() / image.sum())


def test_the_streak_kernel_is_normalised_and_as_long_as_the_motion():
    still = streak_kernel((0.0, 0.0), 1.0, (0.5, 0.5))
    assert still.sum() == pytest.approx(1.0) and (still > 1e-6).sum() <= 4
    moving = streak_kernel((10.0, 0.0), 0.8, (0.0, 0.0))
    assert moving.sum() == pytest.approx(1.0)
    columns = np.nonzero(moving.sum(axis=0) > 1e-6)[0]
    assert 8 <= columns.max() - columns.min() <= 10                      # 10 px a frame for 0.8 of it
    assert np.count_nonzero(moving.sum(axis=1) > 1e-6) <= 2


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (9.0, -3.0)])
def test_a_rendered_ball_is_centred_on_its_label_still_or_streaked(velocity):
    bank = disc_bank()
    frame = np.zeros((3, 120, 200), np.float32)
    render_ball(frame, bank.crops[0], bank.mattes[0], bank.centres[0], (100.3, 60.7), velocity, 1.0)
    x, y = centroid(frame[0])
    assert x == pytest.approx(100.3, abs=0.05) and y == pytest.approx(60.7, abs=0.05)
    assert 0.0 <= frame.min() and frame.max() <= 1.0 + 1e-6
    extent = np.nonzero(frame[0].sum(axis=0) > 1e-3)[0]
    assert extent.max() - extent.min() >= (12 if velocity[0] else 4)


def test_apply_paste_draws_on_every_frame_and_nowhere_else():
    bank = disc_bank()
    points = tuple((100.0 + 10 * k, 200.0) for k in range(5))
    images = np.zeros((15, HEIGHT, WIDTH), np.float32)
    apply_paste(images, bank, Paste(0, points, ((10.0, 0.0),) * 5, 0.5))
    for k, (x, y) in enumerate(points):
        red = images[3 * k]
        assert centroid(red) == pytest.approx((x, y), abs=0.1)
        assert red[int(y) - 20:int(y) + 20, int(x) - 20:int(x) + 20].sum() == pytest.approx(red.sum())


def test_pick_crop_prefers_the_clips_crops_cut_near_the_pastes_height():
    bank = disc_bank(rows=(40.0, 300.0, 310.0, 100.0), clips=("a", "a", "a", "b"))
    draws = random.Random(1)
    assert {pick_crop(draws, bank, "a", 290.0) for _ in range(20)} == {1, 2}
    assert pick_crop(draws, bank, "a", 150.0) == 0 and pick_crop(draws, bank, "a", 260.0) == 1   # none near: the nearest
    assert pick_crop(draws, bank, "pseudo_clip", 120.0) == 3                                       # not in the bank: any clip


def test_draw_paste_keeps_inside_the_frame_and_away_from_the_units_own_ball():
    bank = disc_bank()
    draws = random.Random(3)
    for _ in range(50):
        own = (draws.uniform(0, WIDTH), draws.uniform(0, HEIGHT))
        paste = draw_paste(draws, bank, "c", 2, [own] * 5)
        if paste is None:
            continue
        for x, y in paste.points:
            assert copy_paste.EDGE_PX <= x <= WIDTH - copy_paste.EDGE_PX and copy_paste.EDGE_PX <= y <= HEIGHT - copy_paste.EDGE_PX
            assert np.hypot(x - own[0], y - own[1]) >= copy_paste.AVOID_PX
        assert abs(paste.points[2][1] - 200.0) <= copy_paste.PASTE_DY_PX
        assert (abs(paste.velocities[2][0]), paste.velocities[2][1]) == pytest.approx((2.0, 0.0), abs=1e-4)   # 4 px a step, stride 2
        assert copy_paste.EXPOSURE[0] <= paste.exposure <= copy_paste.EXPOSURE[1]


def slow_clip(data: Path) -> list:
    """A clip whose white ball crawls right 1 px a frame over a textured background: slow balls to
    crop, with the background clear of them 15 frames away."""
    (data / "frames" / "s").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    background = (rng.integers(40, 90, (HEIGHT, WIDTH, 3))).astype(np.uint8)
    for index in range(COUNT):
        image = background.copy()
        cv2.circle(image, (200 + index, 150), 2, (255, 255, 255), -1)
        cv2.imwrite(str(data / "frames" / "s" / f"{index:06d}.png"), image)
    return [TrainingUnit(clip="s", source="custom", indices=list(range(start, start + 5)),
                         frames=[f"frames/s/{index:06d}.png" for index in range(start, start + 5)],
                         points=[[(200.5 + index) * 2.5, 150.5 * 2.5] for index in range(start, start + 5)],
                         size=[1920, 1080]) for start in range(COUNT - 4)]


def test_build_banks_cuts_slow_balls_and_paths_and_keeps_a_bank_of_the_same_units(tmp_path, capsys):
    units = slow_clip(tmp_path)
    path = build_banks(units, tmp_path, tmp_path / "run" / copy_paste.BANK_NAME)
    banks = load_banks(path)
    assert len(banks.crops) > 0 and banks.by_clip == {"s": list(range(len(banks.crops)))}
    assert np.allclose(banks.crop_rows, 150.5) and len(banks.paths) == len(units)
    assert all(copy_paste.MATTE_MASS[0] <= float(matte.sum()) <= copy_paste.MATTE_MASS[1] for matte in banks.mattes)
    assert float(banks.crops[0][copy_paste.CROP_RADIUS, copy_paste.CROP_RADIUS].min()) > 0.9    # the white ball
    assert "copypaste bank" in capsys.readouterr().out
    build_banks(units, tmp_path, path)
    assert capsys.readouterr().out == ""                                   # the same units: the file is kept
    with pytest.raises(ValueError, match="slow ball crops"):
        build_banks(units[:1], tmp_path, tmp_path / "other.npz")


def test_the_dataset_pastes_into_the_frames_and_labels_the_pasted_ball(tmp_path):
    units = slow_clip(tmp_path)
    banks = build_banks(units, tmp_path, tmp_path / copy_paste.BANK_NAME)
    with pytest.raises(ValueError, match="needs the ball and path banks"):
        UnitDataset(tmp_path, units, augment=True, augmentations={"copypaste"}, arch=ARCH)
    dataset = UnitDataset(tmp_path, units, augment=True, augmentations={"copypaste"}, arch=ARCH, banks=banks)
    points = tuple((500.0 + 4 * k, 300.0) for k in range(5))
    paste = Paste(0, points, ((2.0, 0.0),) * 5, 0.5)
    plain_images, plain_target = dataset.apply(units[0], Augmentation())
    images, target = dataset.apply(units[0], Augmentation(paste=paste))
    assert float(images.sum()) > float(plain_images.sum())
    scale = GRID_COLS / WIDTH
    for k, (x, y) in enumerate(points):
        assert float(target[3 * k, int(y * scale), int(x * scale)]) == 1.0 == float(plain_target[3 * k].max())
        assert int((target[3 * k] == 1.0).sum()) == 2                      # the clip's ball and the pasted one
    _, reversed_target = dataset.apply(units[0], Augmentation(paste=paste, reverse=True))
    assert float(reversed_target[0, int(points[4][1] * scale), int(points[4][0] * scale)]) == 1.0
    _, shifted = dataset.apply(units[0], Augmentation(paste=paste, shift=(300, 0)))
    assert all(int((shifted[3 * k] == 1.0).sum()) == 1 for k in range(5))  # the pasted ball shifted off the frame
    draws = [dataset.draw(units[0]).paste for _ in range(40)]
    assert any(paste is not None for paste in draws) and any(paste is None for paste in draws)
