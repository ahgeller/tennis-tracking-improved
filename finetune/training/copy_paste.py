"""Copy-paste ball augmentation (``--augment copypaste``): a second, real ball is pasted into a
unit along a real ball's path, streak-blurred by its speed, the same ball on all five frames.

Kisantal et al. (arXiv 1902.07296) raised small-object detection 7.1% (relative, MS COCO) by
copy-pasting small objects many times into images that already hold them; the ball here is
2 to 4 px at the model's 768x432. Two banks are cut from the training units (``build_banks``,
once per run, stored beside the frame cache):

- the ball bank: crops of the labelled ball where it moved slowly (``SLOW_PX`` per source frame
  or less, so the crop is nearly unblurred), each with a matte, the colour difference from the
  same patch in other frames of the clip where the ball was elsewhere (the background without
  the ball, ``background_frames``), feathered to the ball's disc;
- the path bank: the five labelled points of every unit whose ball is visible throughout and
  moves, relative to the middle point, with the unit's source-frame stride. A pasted path is a
  real one (mirrored half the time) moved to a new place at about the same height in the frame
  (``PASTE_DY_PX``), so it is a physically plausible parabola by construction and the ball keeps
  about the size it had there.

The blur follows BlurBall (arXiv 2509.18387): a ball moving during the exposure is a straight
streak along its velocity, its length the speed times the exposure time, and the label sits at
the streak's centre (the workspace labels' convention, ``finetune/DATASETS.md``). The velocity is
the path's central difference per source frame; the exposure, drawn once per unit (one camera),
is a share ``EXPOSURE`` of the source frame interval. The crop and its matte are convolved with
the streak (``streak_kernel``) and composited over the frame, and the pasted points join the
labels as extra balls (``dataset.make_target``'s `extra`). A paste keeps ``AVOID_PX`` from the
unit's own ball in every frame. Which units, how often: ``PASTE_P``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH
from finetune.data.fileio import write_atomic
from finetune.training.frame_cache import TrainingUnit

BANK_NAME = "copypaste_bank.npz"
BANK_VERSION = 2                 # 2: the crops' rows
CROP_RADIUS = 7                  # crops are (2r + 1) square, in cache pixels
SLOW_PX = 1.5                    # a ball crop's speed limit, cache px per source frame
CLEAR_PX = 10.0                  # a background frame's ball sits at least this far away
BACKGROUND_FRAMES = 3            # the background is the median of this many frames of the clip ...
BACKGROUND_GAP = 15              # ... each at least this many source frames from the crop's
MATTE_LOW, MATTE_HIGH = 0.04, 0.15   # colour difference (0..1) mapped to alpha 0..1
DISC_PX = (3.5, 5.5)             # the matte is kept within the first radius and fades out by the second
MATTE_MASS = (1.5, 40.0)         # a crop whose matte holds less or more is not a lone ball
MAX_CROPS_PER_CLIP = 60
MIN_PATH_PX = 3.0                # a path must move this far over its five frames (cache px)
MAX_STEP_PX = 120.0              # ... and no step longer than this (a label error or a cut)
PASTE_P = 0.5
PASTE_DY_PX = 40.0               # the pasted path's middle stays this close to the original's height
CROP_DY_PX = 40.0                # ... and the crop is one cut this close to the paste's height, when there is one
AVOID_PX = 12.0
EDGE_PX = 8.0
EXPOSURE = (0.25, 1.0)           # the exposure's share of the source frame interval
TRIES = 8


@dataclass(frozen=True)
class Banks:
    """The ball crops (N, S, S, 3 RGB float32 in [0, 1]) with their mattes (N, S, S) and the ball's
    centre in each crop (N, 2, continuous: pixel k's centre at k + 0.5) and clip; and the paths
    (M, 5, 2) relative to their middle point, the middles (M, 2) and their source-frame strides.
    ``crop_rows`` is the height (cache px) each crop was cut at: a ball's size goes with its depth,
    which the height stands in for on a fixed camera."""
    crops: np.ndarray
    mattes: np.ndarray
    centres: np.ndarray
    crop_rows: np.ndarray
    crop_clips: List[str]
    paths: np.ndarray
    middles: np.ndarray
    strides: np.ndarray
    by_clip: Dict[str, List[int]]          # the crops of each clip

    @staticmethod
    def of(crops: np.ndarray, mattes: np.ndarray, centres: np.ndarray, crop_rows: np.ndarray, crop_clips: List[str],
           paths: np.ndarray, middles: np.ndarray, strides: np.ndarray) -> "Banks":
        by_clip: Dict[str, List[int]] = {}
        for index, clip in enumerate(crop_clips):
            by_clip.setdefault(clip, []).append(index)
        return Banks(crops, mattes, centres, crop_rows, crop_clips, paths, middles, strides, by_clip)


@dataclass(frozen=True)
class Paste:
    """One unit's paste, drawn by ``draw_paste``: the crop, the five points (cache pixels), each
    frame's velocity in cache pixels per source frame, and the exposure share."""
    crop: int
    points: Tuple[Tuple[float, float], ...]
    velocities: Tuple[Tuple[float, float], ...]
    exposure: float


def cache_point(point: Sequence[float], size: Sequence[int]) -> Tuple[float, float]:
    """A label in the clip's native pixels at the cache's 768x432."""
    return point[0] * WIDTH / size[0], point[1] * HEIGHT / size[1]


def unit_stride(unit: TrainingUnit) -> int:
    return unit["indices"][1] - unit["indices"][0]


# --------------------------------------------------------------------------- the banks

def matte(crop: np.ndarray, background: np.ndarray, centre: Tuple[float, float]) -> np.ndarray:
    """The ball's alpha in `crop` (S, S, 3 float in [0, 1]): its largest colour difference from
    `background` mapped from ``MATTE_LOW``..``MATTE_HIGH`` to 0..1, kept within ``DISC_PX`` of the
    ball's `centre`."""
    difference = np.abs(crop - background).max(axis=2)
    alpha = np.clip((difference - MATTE_LOW) / (MATTE_HIGH - MATTE_LOW), 0.0, 1.0)
    size = crop.shape[0]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    distance = np.hypot(xs - centre[0], ys - centre[1])
    inner, outer = DISC_PX
    return (alpha * np.clip((outer - distance) / (outer - inner), 0.0, 1.0)).astype(np.float32)


def patch(image: np.ndarray, column: int, row: int, radius: int = CROP_RADIUS) -> Optional[np.ndarray]:
    """The (2r + 1)-square patch of an HWC image centred on pixel (column, row), None at the border."""
    if row - radius < 0 or column - radius < 0 or row + radius >= image.shape[0] or column + radius >= image.shape[1]:
        return None
    return image[row - radius:row + radius + 1, column - radius:column + radius + 1]


def read_rgb(data_dir: Path, relative: str) -> np.ndarray:
    image = cv2.imread(str(data_dir / relative), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(data_dir / relative)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def unit_paths(units: Sequence[TrainingUnit]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The path bank: every unit with a ball on all five frames that moves at least
    ``MIN_PATH_PX`` with no step over ``MAX_STEP_PX``, as (paths relative to the middle point,
    middles, strides)."""
    paths, middles, strides = [], [], []
    for unit in units:
        if any(point is None for point in unit["points"]):
            continue
        points = np.array([cache_point(point, unit["size"]) for point in unit["points"] if point is not None], dtype=np.float32)
        steps = np.hypot(*np.diff(points, axis=0).T)
        if steps.sum() < MIN_PATH_PX or steps.max() > MAX_STEP_PX:
            continue
        paths.append(points - points[2])
        middles.append(points[2])
        strides.append(unit_stride(unit))
    if not paths:
        return np.zeros((0, FRAMES_PER_UNIT, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.int32)
    return np.stack(paths), np.stack(middles), np.array(strides, dtype=np.int32)


def background_frames(frames: Dict[int, Tuple[str, Optional[Tuple[float, float]]]], index: int,
                      point: Tuple[float, float]) -> List[str]:
    """The ``BACKGROUND_FRAMES`` cached frames of a clip nearest in time to `index` but at least
    ``BACKGROUND_GAP`` source frames from it, whose ball (if labelled) is ``CLEAR_PX`` from `point`:
    the patch's background with the ball elsewhere."""
    usable = [other for other, (_, ball) in frames.items()
              if abs(other - index) >= BACKGROUND_GAP
              and (ball is None or math.hypot(ball[0] - point[0], ball[1] - point[1]) >= CLEAR_PX)]
    usable.sort(key=lambda other: abs(other - index))
    return [frames[other][0] for other in usable[:BACKGROUND_FRAMES]]


def ball_crops(units: Sequence[TrainingUnit], data_dir: Path, seed: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """The ball bank: up to ``MAX_CROPS_PER_CLIP`` slow balls per clip (see the module docstring),
    as (crops, mattes, centres, rows, clips). The background of a crop is the per-pixel median of the
    same patch in ``background_frames``; on a moving camera it does not match, the matte fills the
    patch and ``MATTE_MASS`` turns the crop away."""
    draws = random.Random(seed)
    candidates: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {}
    frames: Dict[str, Dict[int, Tuple[str, Optional[Tuple[float, float]]]]] = {}
    for unit in units:
        clip_frames = frames.setdefault(unit["clip"], {})
        for index, relative, label in zip(unit["indices"], unit["frames"], unit["points"]):
            clip_frames[index] = (relative, None if label is None else cache_point(label, unit["size"]))
        if any(point is None for point in unit["points"]):
            continue
        points = np.array([cache_point(point, unit["size"]) for point in unit["points"] if point is not None],
                          dtype=np.float32)
        for frame in range(1, FRAMES_PER_UNIT - 1):
            speed = max(np.hypot(*(points[frame] - points[frame - 1])), np.hypot(*(points[frame + 1] - points[frame])))
            if speed / unit_stride(unit) <= SLOW_PX:
                candidates.setdefault(unit["clip"], []).append((unit["indices"][frame], tuple(points[frame])))
    crops, mattes, centres, rows, clips = [], [], [], [], []
    for clip, found in sorted(candidates.items()):
        found = sorted(set(found))
        draws.shuffle(found)
        kept = 0
        for index, (x, y) in found:
            if kept >= MAX_CROPS_PER_CLIP:
                break
            behind_paths = background_frames(frames[clip], index, (x, y))
            if len(behind_paths) < BACKGROUND_FRAMES:
                continue
            column, row = int(math.floor(x)), int(math.floor(y))
            ball = patch(read_rgb(data_dir, frames[clip][index][0]), column, row)
            found_behinds = [patch(read_rgb(data_dir, relative), column, row) for relative in behind_paths]
            behinds = [behind for behind in found_behinds if behind is not None]
            if ball is None or len(behinds) < len(found_behinds):
                continue
            behind = np.median(np.stack(behinds), axis=0)
            centre = (x - (column - CROP_RADIUS), y - (row - CROP_RADIUS))
            alpha = matte(ball, behind, centre)
            if not MATTE_MASS[0] <= float(alpha.sum()) <= MATTE_MASS[1]:
                continue
            crops.append(ball.copy())
            mattes.append(alpha)
            centres.append(centre)
            rows.append(y)
            clips.append(clip)
            kept += 1
    size = 2 * CROP_RADIUS + 1
    if not crops:
        return (np.zeros((0, size, size, 3), np.float32), np.zeros((0, size, size), np.float32),
                np.zeros((0, 2), np.float32), np.zeros(0, np.float32), [])
    return np.stack(crops), np.stack(mattes), np.array(centres, dtype=np.float32), np.array(rows, dtype=np.float32), clips


def bank_signature(units: Sequence[TrainingUnit]) -> List[object]:
    """What a bank was cut from: the format version and the clips and unit count it saw."""
    return [BANK_VERSION, sorted({unit["clip"] for unit in units}), len(units)]


def build_banks(units: Sequence[TrainingUnit], data_dir: Path, path: Path) -> Path:
    """Cut both banks from `units` (the labelled training units, their frames in the cache at
    `data_dir`) into `path` (the trainer's is ``<run dir>/copypaste_bank.npz``) unless the file
    there was cut from the same units, and return it. No crop or no path is a ValueError: the
    augmentation would have nothing to paste."""
    signature = bank_signature(units)
    if path.is_file():
        with np.load(path) as stored:
            if json.loads(str(stored["signature"])) == signature:
                return path
    crops, mattes, centres, rows, clips = ball_crops(units, data_dir)
    paths, middles, strides = unit_paths(units)
    if not len(crops) or not len(paths):
        raise ValueError(f"--augment copypaste found {len(crops)} slow ball crops and {len(paths)} moving paths in the "
                         "training units: it needs at least one of each")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, crops=crops, mattes=mattes, centres=centres, crop_rows=rows, crop_clips=np.array(clips), paths=paths,
             middles=middles, strides=strides, signature=np.array(json.dumps(signature)))
    temporary.replace(path)
    write_atomic(path.with_suffix(".json"), json.dumps({"crops": len(crops), "paths": len(paths),
                                                       "clips": len(set(clips))}))
    print(f"[augment] copypaste bank: {len(crops)} ball crops from {len(set(clips))} clips, {len(paths)} paths -> {path}")
    return path


def load_banks(path: Path) -> Banks:
    with np.load(path) as stored:
        return Banks.of(stored["crops"], stored["mattes"], stored["centres"], stored["crop_rows"], [str(clip) for clip in stored["crop_clips"]],
                        stored["paths"], stored["middles"], stored["strides"])


# --------------------------------------------------------------------------- drawing and rendering

def velocities(points: np.ndarray, stride: int) -> np.ndarray:
    """Each point's velocity in cache pixels per source frame: the central difference of the path
    (one-sided at the ends) divided by the source frames between two points."""
    return np.gradient(points, axis=0) / max(stride, 1)


def draw_paste(draws: random.Random, banks: Banks, clip: str, stride: int,
               avoid: Sequence[Optional[Tuple[float, float]]]) -> Optional[Paste]:
    """One paste for a unit of `clip` with `stride` (or None when no place was found in ``TRIES``):
    a random bank path, mirrored half the time, its middle at a random column and within
    ``PASTE_DY_PX`` of its own height, every point ``EDGE_PX`` inside the frame and ``AVOID_PX``
    from the unit's own ball in `avoid` (cache pixels, None: no ball); the crop (``pick_crop``);
    and the exposure."""
    for _ in range(TRIES):
        index = draws.randrange(len(banks.paths))
        path = banks.paths[index].copy()
        if draws.random() < 0.5:
            path[:, 0] = -path[:, 0]
        low_x, high_x = EDGE_PX - path[:, 0].min(), WIDTH - EDGE_PX - path[:, 0].max()
        low_y = max(EDGE_PX - path[:, 1].min(), banks.middles[index][1] - PASTE_DY_PX)
        high_y = min(HEIGHT - EDGE_PX - path[:, 1].max(), banks.middles[index][1] + PASTE_DY_PX)
        if low_x > high_x or low_y > high_y:
            continue
        points = path + np.array([draws.uniform(low_x, high_x), draws.uniform(low_y, high_y)], dtype=np.float32)
        if any(own is not None and math.hypot(point[0] - own[0], point[1] - own[1]) < AVOID_PX
               for point, own in zip(points, avoid)):
            continue
        crop = pick_crop(draws, banks, clip, float(points[2][1]))
        moving = velocities(points, stride)
        return Paste(crop, tuple((float(x), float(y)) for x, y in points),
                     tuple((float(x), float(y)) for x, y in moving), draws.uniform(*EXPOSURE))
    return None


def pick_crop(draws: random.Random, banks: Banks, clip: str, row: float) -> int:
    """A crop for a paste whose middle is at height `row`: one of the clip's own (the same ball,
    light and camera) when the bank has any, else any clip's; among those, one cut within
    ``CROP_DY_PX`` of `row` (about the size a ball has there), else the one cut nearest to it."""
    pool = banks.by_clip.get(clip) or range(len(banks.crops))
    near = [index for index in pool if abs(float(banks.crop_rows[index]) - row) <= CROP_DY_PX]
    if near:
        return draws.choice(near)
    return min(pool, key=lambda index: abs(float(banks.crop_rows[index]) - row))


def streak_kernel(velocity: Tuple[float, float], exposure: float, shift: Tuple[float, float]) -> np.ndarray:
    """The (K, K) kernel that smears a crop along `velocity` (cache px per source frame) for
    `exposure` of a frame and moves it by the sub-pixel `shift`: a line of length speed x
    exposure, centred (the label is the streak's centre), splatted bilinearly and normalised to
    one. A point at kernel offset (K // 2 + shift) is where the crop's own position lands."""
    dx, dy = velocity[0] * exposure, velocity[1] * exposure
    length = math.hypot(dx, dy)
    size = 2 * int(math.ceil(length / 2 + 2)) + 1
    t = np.linspace(-0.5, 0.5, max(2, int(math.ceil(length * 4))), dtype=np.float32)
    x, y = size // 2 + shift[0] + t * dx, size // 2 + shift[1] + t * dy
    column, row = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - column, y - row
    kernel = np.zeros((size + 1, size + 1), np.float32)          # one spare row and column for the splat's far side
    for oy, wy in ((0, 1 - fy), (1, fy)):
        for ox, wx in ((0, 1 - fx), (1, fx)):
            np.add.at(kernel, (row + oy, column + ox), wx * wy)
    kernel = kernel[:size, :size]
    return kernel / kernel.sum()


def full_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The full 2-D convolution of an (S, S[, C]) float32 array with a (K, K) kernel: (S + K - 1)
    square (OpenCV's correlation with the flipped kernel over the zero-padded image)."""
    k = kernel.shape[0]
    padded = cv2.copyMakeBorder(image, k - 1, k - 1, k - 1, k - 1, cv2.BORDER_CONSTANT, value=0)
    out = cv2.filter2D(padded, -1, cv2.flip(kernel, -1), anchor=(k - 1, k - 1), borderType=cv2.BORDER_CONSTANT)
    return out[k - 1:k - 1 + image.shape[0] + k - 1, k - 1:k - 1 + image.shape[1] + k - 1]


def render_ball(frame: np.ndarray, crop: np.ndarray, alpha: np.ndarray, centre: Sequence[float],
                point: Sequence[float], velocity: Tuple[float, float], exposure: float) -> None:
    """Composite the streak-blurred ball onto one (3, H, W) RGB float frame in place, its centre at
    `point` (cache pixels, pixel k's centre at k + 0.5)."""
    left, top = point[0] - centre[0], point[1] - centre[1]
    column, row = int(math.floor(left)), int(math.floor(top))
    kernel = streak_kernel(velocity, exposure, (left - column, top - row))
    premultiplied = full_convolution(crop * alpha[..., None], kernel)
    coverage = full_convolution(alpha, kernel)
    origin_x, origin_y = column - kernel.shape[0] // 2, row - kernel.shape[0] // 2
    height, width = frame.shape[1:]
    x0, y0 = max(origin_x, 0), max(origin_y, 0)
    x1, y1 = min(origin_x + coverage.shape[1], width), min(origin_y + coverage.shape[0], height)
    if x0 >= x1 or y0 >= y1:
        return
    # OpenCV's float filter leaves rounding of either sign around zero: clamped, so the frame stays in [0, 1].
    a = np.clip(coverage[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x], 0.0, 1.0)
    colour = np.maximum(premultiplied[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x], 0.0)
    region = frame[:, y0:y1, x0:x1]
    frame[:, y0:y1, x0:x1] = region * (1 - a)[None] + np.moveaxis(colour, -1, 0)


def apply_paste(images: np.ndarray, banks: Banks, paste: Paste) -> np.ndarray:
    """The (15, H, W) contiguous unit with the paste rendered on each of its five frames, in place
    (and returned). The composite stays in [0, 1]: the blurred colour never exceeds the blurred
    matte, since the crop's colours are at most one."""
    frames = images.reshape(FRAMES_PER_UNIT, 3, *images.shape[1:])
    for frame, point, velocity in zip(frames, paste.points, paste.velocities):
        render_ball(frame, banks.crops[paste.crop], banks.mattes[paste.crop], banks.centres[paste.crop],
                    point, velocity, paste.exposure)
    return images
