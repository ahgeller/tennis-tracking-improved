"""finetune/training/dataset.py on its own: the grid targets, the sampler and the augmentation,
on synthetic inputs. UnitDataset's reads and draws run on a cached unit in test_train_gridtracknet.py."""
import numpy as np
import pytest
import torch

from finetune.training.dataset import (HEAT_SIGMA_CELLS, Augmenter, build_chains, chain_units, chain_weights,
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


# --------------------------------------------------------------------------- chains

def covering_units(clip: str, starts, stride: int, source: str = "custom", absent=()) -> list:
    """Units like the cache's: each `start` begins a five-frame window at `stride`, a ball at (1, 1)
    on every frame but those in `absent`."""
    return [{"clip": clip, "source": source, "indices": [start + k * stride for k in range(5)],
             "frames": [f"frames/{clip}/{start + k * stride:06d}.png" for k in range(5)],
             "points": [None if start + k * stride in absent else [1.0, 1.0] for k in range(5)], "size": [64, 36]}
            for start in starts]


def test_build_chains_takes_consecutive_units_five_frames_apart_from_the_clips_first_frame():
    """The cache's units advance two label rows at a time; a chain's units advance five, from the
    clip's first labelled frame, so they are the prepass's own unit boundaries. Every frame of a
    chain unit must be covered by a cached unit (so it is on disk, with its label); a gap breaks
    the chains, and the clips never mix."""
    odd = covering_units("o", range(1, 30, 4), 2)                      # 60 FPS, odd phase: frames 1, 3, ..., 37
    thirty = covering_units("t", range(0, 13, 2), 1, "grid")           # 30 FPS: frames 0 to 16
    chains = build_chains(odd + thirty, 2)
    assert [(chain["clip"], chain["source"], [unit["indices"] for unit in chain["units"]]) for chain in chains] == [
        ("o", "custom", [[1, 3, 5, 7, 9], [11, 13, 15, 17, 19]]), ("o", "custom", [[11, 13, 15, 17, 19], [21, 23, 25, 27, 29]]),
        ("t", "grid", [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), ("t", "grid", [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]),
    ]                                                                  # odd: 31..39 needs frame 39, uncovered; 30 FPS: 15..19 needs 17
    assert chains[0]["units"][1]["frames"] == [f"frames/o/{index:06d}.png" for index in (11, 13, 15, 17, 19)]
    assert [len(build_chains(odd, length)) for length in (1, 2, 3, 4)] == [3, 2, 1, 0]
    gapped = covering_units("g", [0, 2, 4, 10, 12, 14, 16], 1)         # frames 0 to 8 and 10 to 20: frame 9 missing
    assert [[unit["indices"][0] for unit in chain["units"]] for chain in build_chains(gapped, 2)] == [[10, 15]]
    assert [unit["indices"][0] for chain in build_chains(gapped, 1) for unit in chain["units"]] == [0, 10, 15]
    absent = covering_units("a", [0, 2, 4], 1, absent={3})
    assert build_chains(absent, 1)[0]["units"][0]["points"] == [[1.0, 1.0]] * 3 + [None, [1.0, 1.0]]
    assert build_chains([], 3) == []
    with pytest.raises(ValueError, match="at least one unit"):
        build_chains(odd, 0)


def test_chain_units_flatten_and_chain_weights_fold_back_as_the_mean():
    chains = build_chains(covering_units("c", range(0, 13, 2), 1, absent={7}), 2)
    units = chain_units(chains)
    assert [unit["indices"][0] for unit in units] == [0, 5, 5, 10] and units[1] is chains[0]["units"][1]
    weights = unit_weights(units, {}, absent=3.0, false_alarms=2.0, flagged={3})
    assert weights == [1.0, 3.0, 3.0, 2.0] and chain_weights(chains, weights) == [2.0, 2.5]
    assert chain_weights([], []) == []
    with pytest.raises(ValueError, match="3 unit weights for 4 chain units"):
        chain_weights(chains, [1.0, 1.0, 1.0])
    sampler = weighted_sampler(chain_weights(chains, weights), 5)
    assert len(sampler) == 5 and list(sampler.weights) == [2.0, 2.5]
    assert weighted_sampler([1.0, 1.0], None) is None and len(weighted_sampler([1.0, 0.0], None)) == 1
    with pytest.raises(ValueError, match="weight 0"):
        weighted_sampler([0.0, 0.0], None)
