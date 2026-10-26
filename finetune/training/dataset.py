"""The trainer's torch side of the frame cache: the grid targets of each architecture, the
augmentations, the ``Dataset`` that reads cached units and the sampler that weights them by
source and by what they hold (absent-ball frames, the frames the starting weights fired on).

The target follows the architecture (``make_target``): GridTrackNet's one-hot cell on its 48x27
grid, exactly as the upstream trainer has it; HeatmapNet's Gaussian heat map on its 192x108
grid (``HEAT_SIGMA_CELLS``), 1 at the ball's cell. Both carry the offsets at the ball's cell
only, and an absent ball is all zeros. The frames are the same five for both, so every
augmentation's 15-channel reshape serves either.

Each augmentation is behind a name in ``AUGMENTATIONS`` (``--augment``), so a run can switch one
on and be measured against the same run without it:

- ``basic``: a small gain jitter (with the mirror flip, the trainer's original augmentation; the
  flip is drawn whenever the dataset augments, whatever names are on);
- ``strong``: ``photometric_augment`` in place of the gain jitter;
- ``translate``: one integer shift of up to ``TRANSLATE_PX`` cache pixels, the same for all
  five frames, border replicated (not reflected, which would copy a ball near the edge), held
  so that no label leaves the frame; the labels move by the shift in native pixels;
- ``reverse``: the five frames and their labels in reverse time order;
- ``compress``: a JPEG re-encode at a drawn quality, and sometimes a slight blur.

The draws for the newer augmentations are made after the older ones, so a run that does not
ask for them draws exactly what it always did.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from tennis_tracker.ball_models import GRIDTRACKNET, HEATMAP_S4, model_class
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT
from finetune.training.frame_cache import Target, TrainingUnit

AUGMENTATIONS = ("basic", "strong", "translate", "reverse", "compress")
HEAT_SIGMA_CELLS = 1.0         # the Gaussian heat target's sigma, in cells of the model's grid
HEAT_RADIUS_SIGMAS = 3         # ... and how far it reaches before it is cut to zero
TRANSLATE_PX = 20              # the largest shift, in cache pixels (at 768x432), either way on each axis
TRANSLATE_P = 0.5
REVERSE_P = 0.5
COMPRESS_P = 0.3
COMPRESS_QUALITY = (35, 85)    # the JPEG quality is drawn between these
BLUR_P = 0.2                   # ... and softened at this rate, as photometric_augment does
BLUR_SIGMA = (0.4, 1.0)


def gaussian_heat(target: np.ndarray, row: int, col: int, sigma: float = HEAT_SIGMA_CELLS,
                  radius_sigmas: int = HEAT_RADIUS_SIGMAS) -> None:
    """Write a Gaussian of `sigma` cells, exactly 1 at (`row`, `col`) and cut to zero beyond
    `radius_sigmas` sigmas, into the 2-D `target` by maximum, as CenterNet draws its heat peaks."""
    radius = int(math.ceil(radius_sigmas * sigma))
    rows, cols = target.shape
    top, bottom = max(0, row - radius), min(rows, row + radius + 1)
    left, right = max(0, col - radius), min(cols, col + radius + 1)
    dy = np.arange(top, bottom, dtype=np.float32) - row
    dx = np.arange(left, right, dtype=np.float32) - col
    heat = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2 * sigma * sigma))
    heat[np.hypot(dy[:, None], dx[None, :]) > radius] = 0.0
    np.maximum(target[top:bottom, left:right], heat, out=target[top:bottom, left:right])


def make_target(points, width: int, height: int, arch: str = GRIDTRACKNET) -> torch.Tensor:
    """The (15, rows, cols) target of a unit's five labels on `arch`'s grid: per frame the heat
    grid (GridTrackNet: 1 at the ball's cell; HeatmapNet: ``gaussian_heat`` around it, 1 at the
    cell) and the x and y offsets, the ball's fractional position in [0, 1) within its cell,
    written at that cell only. A frame without a ball (None) is all zeros."""
    grid_cols, grid_rows = model_class(arch).GRID_COLS, model_class(arch).GRID_ROWS
    target = np.zeros((FRAMES_PER_UNIT * 3, grid_rows, grid_cols), dtype=np.float32)
    for index, point in enumerate(points):
        if point is None:
            continue
        grid_x = point[0] * grid_cols / width
        grid_y = point[1] * grid_rows / height
        col = min(grid_cols - 1, int(grid_x))
        row = min(grid_rows - 1, int(grid_y))
        if arch == HEATMAP_S4:
            gaussian_heat(target[index * 3], row, col)
        target[index * 3, row, col] = 1
        target[index * 3 + 1, row, col] = grid_x - col
        target[index * 3 + 2, row, col] = grid_y - row
    return torch.from_numpy(target)


class Augmenter:
    """The random draws one data-loading process makes, seeded from torch's seed for that process.

    In a DataLoader worker ``torch.initial_seed()`` is ``base_seed + worker_id``, the very seed torch
    gives that worker's global ``random`` module, and in the main process it is the seed
    ``train`` set; so ``draws`` reproduces the sequence the module-level ``random.*`` calls used to
    make, in both cases. The numpy generator (sensor noise) was never seeded before, so it starts
    from the same seed now and is reproducible for the first time.
    """

    def __init__(self, seed: int) -> None:
        self.draws = random.Random(seed)
        self.noise = np.random.default_rng(seed)


def blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Every plane of a (15, H, W) array softened by a Gaussian of `sigma` pixels."""
    return np.stack([cv2.GaussianBlur(plane, (0, 0), sigma) for plane in images])


def photometric_augment(images: np.ndarray, rng: Augmenter) -> np.ndarray:
    """Strong lighting/colour augmentation applied identically to all five frames of a window
    (so motion cues survive): gamma (dark night courts and washed-out daylight), contrast, brightness,
    per-channel colour cast, sensor noise and occasional softness. images: (15, H, W) float in [0, 1]."""
    draws = rng.draws
    gamma = math.exp(draws.uniform(math.log(0.45), math.log(2.2)))       # >1 darkens: night-court look
    contrast = draws.uniform(0.6, 1.4)
    brightness = draws.uniform(-0.12, 0.12)
    cast = np.array([draws.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
    out = np.power(np.clip(images, 0.0, 1.0), gamma)
    out = (out - 0.5) * contrast + 0.5 + brightness
    out = out.reshape(FRAMES_PER_UNIT, 3, *out.shape[1:]) * cast[None, :, None, None]
    out = out.reshape(-1, *out.shape[2:])
    if draws.random() < 0.5:
        # Drawn as float32 outright: a float64 draw over 15 x 432 x 768 values cost 141 ms per unit.
        out = out + rng.noise.standard_normal(out.shape, dtype=np.float32) * np.float32(draws.uniform(0.005, 0.03))
    if draws.random() < BLUR_P:
        out = blur(out, draws.uniform(*BLUR_SIGMA))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- geometry and time

def translate_limits(points: Sequence[Target], width: int, height: int, cache_width: int, cache_height: int,
                     reach: int = TRANSLATE_PX) -> Tuple[int, int, int, int]:
    """(dx_min, dx_max, dy_min, dy_max): the integer shifts in cache pixels, within `reach`, that
    keep every label of the unit on the image (labels sit in [0, width - 1] x [0, height - 1]; a
    mirrored label can sit at ``width``, and the limits then shift it back onto the image)."""
    dx_min, dx_max, dy_min, dy_max = -reach, reach, -reach, reach
    for point in points:
        if point is None:
            continue
        x, y = point[0] * cache_width / width, point[1] * cache_height / height
        dx_min, dx_max = max(dx_min, -math.floor(x)), min(dx_max, math.floor((width - 1 - point[0]) * cache_width / width))
        dy_min, dy_max = max(dy_min, -math.floor(y)), min(dy_max, math.floor((height - 1 - point[1]) * cache_height / height))
    return dx_min, dx_max, dy_min, dy_max


def translate(images: np.ndarray, points: Sequence[Target], dx: int, dy: int, width: int, height: int
              ) -> Tuple[np.ndarray, List[Target]]:
    """The (15, H, W) images shifted right by `dx` and down by `dy` cache pixels with the border
    replicated, and the labels moved by the same shift in the clip's native pixels."""
    cache_height, cache_width = images.shape[1:]
    padded = np.pad(images, ((0, 0), (max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))), mode="edge")
    shifted = padded[:, max(-dy, 0):max(-dy, 0) + cache_height, max(-dx, 0):max(-dx, 0) + cache_width]
    moved = [None if point is None else [point[0] + dx * width / cache_width, point[1] + dy * height / cache_height]
             for point in points]
    return np.ascontiguousarray(shifted), moved


def reverse(images: np.ndarray, points: Sequence[Target]) -> Tuple[np.ndarray, List[Target]]:
    """The five frames and their labels in reverse time order (a rally runs just as well backwards)."""
    frames = images.reshape(FRAMES_PER_UNIT, 3, *images.shape[1:])[::-1]
    return np.ascontiguousarray(frames.reshape(images.shape)), list(points)[::-1]


def compress(images: np.ndarray, quality: int) -> np.ndarray:
    """Every frame of the (15, H, W) RGB float array re-encoded as a JPEG of `quality`: the block
    artefacts and chroma smear of footage that was uploaded, downloaded or screen-recorded."""
    frames = []
    for frame in (images * 255.0).round().astype(np.uint8).reshape(FRAMES_PER_UNIT, 3, *images.shape[1:]):
        bgr = cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode could not write a JPEG")
        decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frames.append(np.moveaxis(decoded, -1, 0))
    return np.concatenate(frames).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- the dataset

class UnitDataset(Dataset):
    """The cached units as (images, target) pairs: the five frames as one (15, H, W) float tensor
    and ``make_target`` on `arch`'s grid, with the augmentations in `augmentations` drawn when
    `augment` is set."""

    def __init__(self, data_dir: Path, samples: Sequence[TrainingUnit], augment: bool,
                 augmentations: Collection[str] = ("basic",), arch: str = GRIDTRACKNET) -> None:
        unknown = set(augmentations) - set(AUGMENTATIONS)
        if unknown:
            raise ValueError(f"unknown augmentation(s) {sorted(unknown)}; choose from {AUGMENTATIONS}")
        model_class(arch)                        # an unknown architecture is refused here, not per unit
        self.data_dir, self.samples, self.augment, self.arch = data_dir, samples, augment, arch
        self.augmentations: Set[str] = set(augmentations) if augment else set()
        self.rng: Optional[Augmenter] = None     # made in the process that reads, see Augmenter

    def __len__(self) -> int:
        return len(self.samples)

    def augmenter(self) -> Augmenter:
        if self.rng is None:
            self.rng = Augmenter(torch.initial_seed())
        return self.rng

    def read_frames(self, sample: TrainingUnit, flip: bool) -> np.ndarray:
        """The unit's five cached frames as one (15, H, W) float array in [0, 1], RGB, mirrored when `flip`."""
        frames = []
        for relative in sample["frames"]:
            image = cv2.imread(str(self.data_dir / relative), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self.data_dir / relative)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip:
                image = cv2.flip(image, 1)
            frames.append(np.moveaxis(image, -1, 0))
        return np.concatenate(frames).astype(np.float32) / 255.0

    def __getitem__(self, index: int):
        sample = self.samples[index]
        width, height = sample["size"]
        rng = self.augmenter()
        on = self.augmentations
        # The draws are made in this order on purpose: it is the sequence every run has made, the
        # newer augmentations drawing after the older ones.
        flip = self.augment and rng.draws.random() < 0.5
        jitter = "basic" in on and "strong" not in on and rng.draws.random() < 0.5
        gain = rng.draws.uniform(0.85, 1.15) if jitter else 1.0
        images = self.read_frames(sample, flip)
        points = sample["points"]
        if flip:
            # Labels are frame coordinates with pixel k's centre at k + 0.5 (the label tool's
            # convention, which the grid encode and decode share); cv2.flip sends pixel k to
            # width - 1 - k, so a point x lands at width - x. "width - 1 - x" mirrored pixel
            # indices and taught the target one pixel left of the mirrored ball.
            points = [None if point is None else [width - point[0], point[1]] for point in points]
        if jitter:
            images = np.clip(images * gain, 0.0, 1.0)
        elif "strong" in on and rng.draws.random() < 0.8:
            images = photometric_augment(images, rng)
        if "translate" in on and rng.draws.random() < TRANSLATE_P:
            dx_min, dx_max, dy_min, dy_max = translate_limits(points, width, height, images.shape[2], images.shape[1])
            images, points = translate(images, points, rng.draws.randint(dx_min, dx_max), rng.draws.randint(dy_min, dy_max),
                                       width, height)
        if "reverse" in on and rng.draws.random() < REVERSE_P:
            images, points = reverse(images, points)
        if "compress" in on and rng.draws.random() < COMPRESS_P:
            images = compress(images, rng.draws.randint(*COMPRESS_QUALITY))
            if rng.draws.random() < BLUR_P:
                images = blur(images, rng.draws.uniform(*BLUR_SIGMA))
        return torch.from_numpy(np.ascontiguousarray(images)), make_target(points, width, height, self.arch)


# --------------------------------------------------------------------------- the sampler

def unit_weights(samples: Sequence[TrainingUnit], oversample: Dict[str, float], absent: float = 1.0,
                 false_alarms: float = 1.0, flagged: Collection[int] = ()) -> List[float]:
    """One sampling weight per unit: its source's (``--oversample``, 1 by default), times `absent`
    when the unit holds an absent-ball frame, times `false_alarms` when its index is in `flagged`
    (the units the starting weights fired on with no ball there, ``false_alarm_units``)."""
    weights = []
    for index, sample in enumerate(samples):
        weight = float(oversample.get(sample["source"], 1.0))
        if any(point is None for point in sample["points"]):
            weight *= absent
        if index in flagged:
            weight *= false_alarms
        weights.append(weight)
    return weights


def make_sampler(samples: Sequence[TrainingUnit], oversample: Dict[str, float], epoch_units: Optional[int],
                 absent: float = 1.0, false_alarms: float = 1.0, flagged: Collection[int] = ()
                 ) -> Optional[WeightedRandomSampler]:
    """Weighted sampling by source and by what a unit holds (``unit_weights``; weight 0 drops a
    unit from training); None = plain shuffle."""
    weights = unit_weights(samples, oversample, absent, false_alarms, set(flagged))
    if epoch_units is None and all(weight == 1.0 for weight in weights):
        return None
    drawable = sum(1 for weight in weights if weight > 0)
    if drawable == 0:
        raise ValueError("Every training unit has sampling weight 0")
    return WeightedRandomSampler(weights, num_samples=epoch_units or drawable, replacement=True)
