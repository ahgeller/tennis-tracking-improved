"""finetune/training/dataset.py on its own: the grid targets, the sampler and the augmentation,
on synthetic inputs. UnitDataset's reads and draws run on a cached unit in test_train_gridtracknet.py."""
from pathlib import Path

import numpy as np
import pytest
import torch

from finetune.training.dataset import (AUGMENTATIONS, HEAT_SIGMA_CELLS, Augmentation, Augmenter, Photometric, UnitDataset,
                                       gaussian_heat, make_sampler, make_target, photometric_augment, unit_weights,
                                       weighted_sampler)
from tennis_tracker import heatmapnet
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS

SIZE = (64, 36)


# --------------------------------------------------------------------------- units and targets

def test_target_marks_the_cell_and_its_offsets_and_nothing_for_an_absent_ball():
    target = make_target([[960.0, 540.0], None, [0.0, 0.0], [1919.0, 1079.0], None], 1920, 1080)
    assert target.shape == (FRAMES_PER_UNIT * 3, GRID_ROWS, GRID_COLS)
    assert float(target[0, 13, 24]) == 1.0 and float(target[0].sum()) == 1.0
    assert float(target[3].sum()) == 0.0                                            # frame 1: no ball
    assert float(target[6, 0, 0]) == 1.0 and float(target[7, 0, 0]) == 0.0 == float(target[8, 0, 0])
    assert float(target[9, GRID_ROWS - 1, GRID_COLS - 1]) == 1.0                    # clamped into the last cell
    assert 0.0 <= float(target[10, GRID_ROWS - 1, GRID_COLS - 1]) < 1.0


def test_the_heatmap_target_is_a_gaussian_peaking_at_one_with_offsets_at_the_peak_only():
    """heatmap_s4: the heat is exactly 1 at the ball's cell on the 192x108 grid and falls off as a
    Gaussian of one cell (cut at three), the offsets are written at that cell alone, an absent ball
    is all zeros, and the gridtracknet target is untouched by the switch."""
    points = [[965.0, 545.0], None, [0.0, 0.0], [1919.0, 1079.0], None]      # grid (96.5, 54.5): cell (54, 96), offsets 0.5
    target = make_target(points, 1920, 1080, heatmapnet.ARCH)
    rows, cols = heatmapnet.GRID_ROWS, heatmapnet.GRID_COLS
    assert target.shape == (FRAMES_PER_UNIT * 3, rows, cols) == (15, 108, 192)
    heat = target[0]
    assert float(heat[54, 96]) == 1.0 and float(heat.max()) == 1.0 and int((heat == 1.0).sum()) == 1
    assert float(heat[54, 97]) == pytest.approx(np.exp(-0.5 / HEAT_SIGMA_CELLS ** 2))
    assert float(heat[53, 95]) == pytest.approx(np.exp(-1.0 / HEAT_SIGMA_CELLS ** 2))
    assert float(heat[54, 99]) == pytest.approx(np.exp(-4.5)) and float(heat[54, 100]) == 0.0       # three sigmas, then cut
    assert 0.0 < float(heat[57, 96]) and float(heat[58, 96]) == 0.0
    assert float(target[1].sum()) == float(target[1, 54, 96]) == pytest.approx(0.5)              # offsets at the peak only
    assert float(target[2].sum()) == float(target[2, 54, 96]) == pytest.approx(0.5)
    assert float(target[3:6].sum()) == 0.0                                                          # frame 1: no ball
    assert float(target[6, 0, 0]) == 1.0 and float(target[6].sum()) > 1.0                            # a corner peak, clipped
    assert float(target[9, rows - 1, cols - 1]) == 1.0
    assert torch.equal(make_target(points, 1920, 1080), make_target(points, 1920, 1080, "gridtracknet"))
    assert make_target(points, 1920, 1080).shape == (15, GRID_ROWS, GRID_COLS)
    with pytest.raises(ValueError, match="unknown ball-detector architecture"):
        make_target(points, 1920, 1080, "yolo")


def test_gaussian_heat_takes_the_maximum_where_two_peaks_overlap():
    grid = np.zeros((9, 9), np.float32)
    gaussian_heat(grid, 4, 4)
    gaussian_heat(grid, 4, 5)
    assert float(grid[4, 4]) == 1.0 == float(grid[4, 5]) and float(grid[4, 3]) == pytest.approx(np.exp(-0.5))


# --------------------------------------------------------------------------- the sampler

def test_sampler_weights_by_source_and_is_absent_without_weights():
    units = [{"source": "a", "points": [[1.0, 1.0]]}, {"source": "b", "points": [[1.0, 1.0]]}, {"source": "a", "points": [[1.0, 1.0]]}]
    assert make_sampler(units, {}, None) is None
    only_a = make_sampler(units, {"b": 0.0}, None)
    assert only_a is not None and len(only_a) == 2 and set(iter(only_a)) <= {0, 2}   # drawn with replacement
    assert len(make_sampler(units, {}, 7)) == 7                                     # --epoch-units draws that many
    with pytest.raises(ValueError, match="weight 0"):
        make_sampler(units, {"a": 0.0, "b": 0.0}, None)


def test_unit_weights_raise_absent_ball_units_and_the_flagged_ones():
    units = [{"source": "a", "points": [[1.0, 1.0], None, [2.0, 2.0]]},      # holds a no-ball frame
             {"source": "b", "points": [[1.0, 1.0], [2.0, 2.0]]},
             {"source": "a", "points": [[1.0, 1.0]]}]
    assert unit_weights(units, {}) == [1.0, 1.0, 1.0]
    assert unit_weights(units, {"a": 2.0}, absent=3.0) == [6.0, 1.0, 2.0]
    assert unit_weights(units, {}, false_alarms=4.0, flagged={1, 2}) == [1.0, 4.0, 4.0]
    assert unit_weights(units, {"b": 0.5}, absent=3.0, false_alarms=4.0, flagged={0}) == [12.0, 0.5, 1.0]
    sampler = make_sampler(units, {}, None, absent=3.0)
    assert sampler is not None and list(sampler.weights) == [3.0, 1.0, 1.0] and len(sampler) == 3
    assert make_sampler(units, {}, None, absent=1.0, false_alarms=1.0, flagged={0}) is None




# --------------------------------------------------------------------------- augmentation

def test_augmenter_is_seeded_and_photometric_augment_keeps_shape_and_range():
    first, second = Augmenter(3), Augmenter(3)
    assert first.draws.random() == second.draws.random()
    assert float(first.noise.normal()) == float(second.noise.normal())
    images = np.full((15, 8, 8), 0.5, dtype=np.float32)
    augmented = photometric_augment(images, Augmenter(1))
    assert augmented.shape == images.shape and augmented.dtype == np.float32
    assert 0.0 <= float(augmented.min()) <= float(augmented.max()) <= 1.0
    assert not np.array_equal(augmented, images)
    assert np.array_equal(photometric_augment(images, Augmenter(1)), augmented)   # the same seed, the same draw
    assert isinstance(make_target([None] * 5, *SIZE), torch.Tensor)


def test_the_draws_of_a_seed_are_the_ones_every_run_has_made():
    """The draw sequence frozen: these are the values the dataset drew for seed 5 before the
    draws were split from their application (checked equal to that code on every augmentation
    set), so a seed still reproduces an old run's augmentation. The shift is drawn within the
    unit's own limits, so the unit is fixed too."""
    unit = {"clip": "c", "source": "custom", "indices": list(range(5)), "frames": [], "size": [1920, 1080],
            "points": [[850.0, 300.0], None, [1900.0, 1070.0], [10.0, 5.0], [960.0, 540.0]]}
    basic = UnitDataset(Path("unused"), [unit], augment=True, augmentations={"basic"})
    basic.rng = Augmenter(5)
    draws = [basic.draw(unit) for _ in range(4)]
    assert draws[:3] == [Augmentation()] * 3 and draws[3].flip and draws[3].gain == pytest.approx(1.133007015099494)
    assert basic.rng.draws.random() == pytest.approx(0.6489745531369242)        # the stream is where it always was
    # Every augmentation the frozen sequence knew; copypaste draws after them and is pinned below.
    everything = UnitDataset(Path("unused"), [unit], augment=True, augmentations=set(AUGMENTATIONS) - {"copypaste"})
    everything.rng = Augmenter(5)
    everything.draw(unit)
    drawn = everything.draw(unit)
    assert (drawn.flip, drawn.gain, drawn.shift, drawn.reverse, drawn.compress, drawn.blur) == (True, 1.0, (-8, -2), True, 45, None)
    assert drawn.photometric == pytest.approx(Photometric(1.1188673887090184, 0.6104913516711218, -0.06798484788867644,
                                                          (0.933844709803333, 1.1249036115425655, 1.0797176354887426),
                                                          0.024928674785780112, 0.7704715122796699))
    assert everything.draw(unit) == Augmentation(flip=True, compress=79, blur=pytest.approx(0.9790578345674649))
    assert UnitDataset(Path("unused"), [unit], augment=False).draw(unit) == Augmentation()


def test_weighted_sampler_draws_by_weight_and_refuses_all_zero():
    sampler = weighted_sampler([2.0, 2.5], 5)
    assert len(sampler) == 5 and list(sampler.weights) == [2.0, 2.5]
    assert weighted_sampler([1.0, 1.0], None) is None and len(weighted_sampler([1.0, 0.0], None)) == 1
    with pytest.raises(ValueError, match="weight 0"):
        weighted_sampler([0.0, 0.0], None)
