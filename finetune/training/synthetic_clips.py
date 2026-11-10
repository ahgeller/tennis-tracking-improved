"""Synthetic training clips: a real ball crop flown along a real labelled rally track over a
ball-free stretch of unlabelled match video, written as a short H.264 clip with a label file the
trainer reads through ``--pseudo`` (``audit/records/opt_synthetic_data.md``).

    python finetune/training/synthetic_clips.py --bank RUN/copypaste_bank.npz --out D:/synth \\
        --videos "match_a.mp4" "match_b.mp4" --clips 150

What one clip is made of:

- the plate: ``--length`` frames at 30 FPS cadence (every second frame of a 60 FPS video) from a
  random place in one of ``--videos``, resized as the tracker resizes, accepted only when the
  teacher (``--teacher``, the shipped detector) sees nothing above ``PLATE_MAX_HEAT`` on any
  frame (``ball_free``), so the only ball in the clip is the one drawn into it;
- the track: ``--length`` consecutive label frames of a fixed-camera ``TRACK_SOURCES`` training
  clip (never one of ``--holdout``'s), mirrored about the frame's centre half the time and moved
  within ``MAX_DX_PX`` sideways and ``copy_paste.PASTE_DY_PX`` up or down (``place_track``), so it
  stays over the court of a behind-baseline view: a real rally path with its hits, bounces and absent frames, which
  stay absent (the label is then "none");
- the ball: one crop of the copy-paste bank cut from a ``CROP_SOURCES`` clip near the track's
  height (``pick_crop``; a crop keeps a rim of its own court in its soft matte, so a crop from a
  clay or grass court showed as a red or green ball on these blue hard courts), streak-blurred by the track's speed for one exposure per clip, as ``copy_paste``
  renders a paste;
- the encoding: libx264 at ``CRF`` (yuv444p, so the 768x432 chroma is not halved below what the
  real cache frames carry), so the pasted ball carries codec artefacts like a real one.

The output folder has the pseudo-label layout (``videos/<clip>.mp4``, ``labels/<clip>_pseudo.csv``
with ``ball`` and ``none`` rows), so ``train_gridtracknet.py --pseudo DIR --pseudo-share S`` mixes
the clips into training without any trainer change.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import finetune  # noqa: F401,E402  (which puts system/ on sys.path)

import cv2
import numpy as np

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH, resize_to_model
from finetune.data.holdouts import holdout_clips
from finetune.training import copy_paste
from finetune.training.pseudo_labels import BALL, NONE, pseudo_csv, write_pseudo_labels

FPS = 30
CRF = 18
PLATE_MAX_HEAT = 0.3             # a plate whose teacher heat reaches this on any frame may hold a ball
EDGE_PX = copy_paste.EDGE_PX
MIN_VISIBLE = 0.6                # a track window must show the ball on this share of its frames
TEACHER_BATCH = 16
MAX_DX_PX = 80.0                 # a track moves at most this far sideways: behind-baseline views centre the court
TRACK_SOURCES = ("custom", "prof")   # behind-baseline fixed cameras, the plates' geometry
CROP_SOURCES = ("custom",)           # the plates' own camera type: the ball and court colours match

Track = List[Optional[Tuple[float, float]]]      # cache pixels per label frame, None: no ball


# --------------------------------------------------------------------------- tracks

def clip_tracks(record: dict) -> Tuple[Dict[int, Optional[Tuple[float, float]]], int]:
    """A cache record's labels as source frame -> cache point (None: absent), and its stride."""
    points: Dict[int, Optional[Tuple[float, float]]] = {}
    for unit in record["samples"]:
        for index, point in zip(unit["indices"], unit["points"]):
            points[index] = None if point is None else copy_paste.cache_point(point, unit["size"])
    return points, int(record["stride"])


def track_windows(points: Dict[int, Optional[Tuple[float, float]]], stride: int, length: int) -> List[Track]:
    """Every run of `length` consecutive label frames (``stride`` apart) of one clip, advancing half
    a window, with the ball on at least ``MIN_VISIBLE`` of them."""
    windows: List[Track] = []
    indices = sorted(points)
    present = set(indices)
    for start in indices[::max(1, length // 2)]:
        frames = [start + offset * stride for offset in range(length)]
        if not all(frame in present for frame in frames):
            continue
        track = [points[frame] for frame in frames]
        if sum(point is not None for point in track) >= MIN_VISIBLE * length:
            windows.append(track)
    return windows


def place_track(track: Track, draws: random.Random) -> Optional[Track]:
    """`track` mirrored about the frame's centre half the time and moved within ``MAX_DX_PX``
    sideways and ``PASTE_DY_PX`` up or down, every point ``EDGE_PX`` inside the frame; None when
    it cannot fit."""
    seen = np.array([point for point in track if point is not None], dtype=np.float64)
    if not len(seen):
        return None
    mirror = draws.random() < 0.5
    xs = WIDTH - seen[:, 0] if mirror else seen[:, 0]
    low_x, high_x = max(EDGE_PX - xs.min(), -MAX_DX_PX), min(WIDTH - EDGE_PX - xs.max(), MAX_DX_PX)
    low_y = max(EDGE_PX - seen[:, 1].min(), -copy_paste.PASTE_DY_PX)
    high_y = min(HEIGHT - EDGE_PX - seen[:, 1].max(), copy_paste.PASTE_DY_PX)
    if low_x > high_x or low_y > high_y:
        return None
    dx, dy = draws.uniform(low_x, high_x), draws.uniform(low_y, high_y)
    return [None if point is None else ((WIDTH - point[0] if mirror else point[0]) + dx, point[1] + dy)
            for point in track]


def track_velocities(track: Track, frames_per_step: float) -> List[Tuple[float, float]]:
    """Each point's velocity in cache px per camera frame: the central difference over its
    visible neighbours (one-sided at a gap or an end, zero for a lone point) divided by the
    camera frames between two label frames."""
    out: List[Tuple[float, float]] = []
    for index, point in enumerate(track):
        if point is None:
            out.append((0.0, 0.0))
            continue
        before = track[index - 1] if index > 0 else None
        after = track[index + 1] if index + 1 < len(track) else None
        if before is not None and after is not None:
            v = ((after[0] - before[0]) / 2, (after[1] - before[1]) / 2)
        elif after is not None:
            v = (after[0] - point[0], after[1] - point[1])
        elif before is not None:
            v = (point[0] - before[0], point[1] - before[1])
        else:
            v = (0.0, 0.0)
        out.append((v[0] / frames_per_step, v[1] / frames_per_step))
    return out


# --------------------------------------------------------------------------- plates and rendering

def pick_crop(draws: random.Random, banks: copy_paste.Banks, pool: Sequence[int], row: float) -> int:
    """One crop of `pool` cut within ``copy_paste.CROP_DY_PX`` of `row`, else the one cut nearest it."""
    near = [index for index in pool if abs(float(banks.crop_rows[index]) - row) <= copy_paste.CROP_DY_PX]
    if near:
        return draws.choice(near)
    return min(pool, key=lambda index: abs(float(banks.crop_rows[index]) - row))


def ball_free(frame_heat_max: Sequence[float], threshold: float = PLATE_MAX_HEAT) -> bool:
    """A plate is used only when the teacher's heat stays below `threshold` on every frame."""
    return max(frame_heat_max) < threshold


def render_clip(plate: np.ndarray, track: Track, banks: copy_paste.Banks, crop: int, exposure: float,
                frames_per_step: float) -> np.ndarray:
    """The (T, H, W, 3) uint8 RGB `plate` with the crop drawn along `track` (absent frames left as
    they are), each frame's streak from ``track_velocities``."""
    frames = np.ascontiguousarray(np.moveaxis(plate.astype(np.float32) / 255.0, -1, 1))     # (T, 3, H, W)
    for frame, point, velocity in zip(frames, track, track_velocities(track, frames_per_step)):
        if point is not None:
            copy_paste.render_ball(frame, banks.crops[crop], banks.mattes[crop], banks.centres[crop], point, velocity, exposure)
    return np.clip(np.moveaxis(frames, 1, -1) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def label_rows(track: Track) -> Dict[int, Tuple[str, Optional[Tuple[float, float]]]]:
    """The pseudo-label rows of a synthetic clip: every frame is a label frame (30 FPS, stride 1)."""
    return {index: (NONE, None) if point is None else (BALL, point) for index, point in enumerate(track)}


def write_video(path: Path, frames: np.ndarray) -> None:
    """(T, H, W, 3) RGB uint8 frames as an H.264 MP4 at ``FPS`` through ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv444p", str(path)]
    subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), check=True)


def read_plate(capture: cv2.VideoCapture, start: int, length: int, step: int) -> Optional[np.ndarray]:
    """`length` frames `step` apart from source frame `start`, at the model size, RGB; None at the end."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for index in range(length * step):
        ok, frame = capture.read()
        if not ok:
            return None
        if index % step == 0:
            out.append(cv2.cvtColor(resize_to_model(frame), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def teacher_heat_max(teacher, plate: np.ndarray, device) -> List[float]:
    """The teacher's highest heat on each frame of `plate` over every unit covering it."""
    import torch
    frames = torch.from_numpy(np.moveaxis(plate, -1, 1).copy()).to(device).float().div_(255.0)
    best = torch.zeros(len(plate), device=device)
    starts = list(range(len(plate) - FRAMES_PER_UNIT + 1))
    parameter = next(teacher.parameters())
    with torch.no_grad():
        for first in range(0, len(starts), TEACHER_BATCH):
            chunk = starts[first:first + TEACHER_BATCH]
            batch = torch.stack([frames[s:s + FRAMES_PER_UNIT].reshape(-1, HEIGHT, WIDTH) for s in chunk])
            out = teacher(batch.to(parameter.dtype)).float()
            heat = out.reshape(len(chunk), FRAMES_PER_UNIT, 3, out.shape[-2], out.shape[-1])[:, :, 0]
            peaks = heat.flatten(2).max(dim=2).values
            for row, start in enumerate(chunk):
                best[start:start + FRAMES_PER_UNIT] = torch.maximum(best[start:start + FRAMES_PER_UNIT], peaks[row])
    return [float(value) for value in best.cpu()]


# --------------------------------------------------------------------------- the command

def training_tracks(cache: Path, clips_csv: Path, holdouts: Path, holdout: str, length: int) -> List[Tuple[str, Track]]:
    """(clip, track window) for every fixed-camera ``TRACK_SOURCES`` clip in the cache outside the
    held-out set."""
    import csv
    with clips_csv.open(newline="", encoding="utf-8") as handle:
        manifest = {row["clip"]: row for row in csv.DictReader(handle)}
    names = sorted(path.stem for path in (cache / "clips").glob("*.json"))
    held = set(holdout_clips(holdout, [name for name in names if name in manifest], manifest, holdouts))
    out: List[Tuple[str, Track]] = []
    for name in names:
        row = manifest.get(name)
        if row is None or name in held or row.get("camera") != "fixed" or row.get("source") not in TRACK_SOURCES:
            continue
        record = json.loads((cache / "clips" / f"{name}.json").read_text(encoding="utf-8"))
        points, stride = clip_tracks(record)
        out.extend((name, track) for track in track_windows(points, stride, length))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch
    from tennis_tracker.ball_models import load_ball_model
    from finetune.data import workspace as ws

    layout = ws.current()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", type=Path, required=True, help="60 FPS fixed-camera match videos")
    parser.add_argument("--bank", type=Path, required=True, help="a run's copypaste_bank.npz (copy_paste.build_banks)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=150)
    parser.add_argument("--length", type=int, default=60, help="label frames per clip (30 FPS)")
    parser.add_argument("--teacher", type=Path, default=layout.repo_weights)
    parser.add_argument("--cache", type=Path, default=layout.cache / "data")
    parser.add_argument("--holdout", default="mixed_v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    draws = random.Random(args.seed)
    banks = copy_paste.load_banks(args.bank)
    tracks = training_tracks(args.cache, layout.root / "clips.csv", layout.root / "holdouts.json", args.holdout, args.length)
    if not tracks:
        raise SystemExit("no track window in the training clips")
    import csv
    with (layout.root / "clips.csv").open(newline="", encoding="utf-8") as handle:
        crop_clips = {row["clip"] for row in csv.DictReader(handle) if row["source"] in CROP_SOURCES}
    pool = [index for index, clip in enumerate(banks.crop_clips) if clip in crop_clips]
    if not pool:
        raise SystemExit(f"the bank holds no crop of a {CROP_SOURCES} clip")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_ball_model(args.teacher, device).eval()
    if device.type != "cuda":
        teacher = teacher.float()
    captures = [(video, cv2.VideoCapture(str(video))) for video in args.videos]
    made, tried, started = 0, 0, time.perf_counter()
    stats: Dict[str, float] = {"tracks": len(tracks), "crops": len(pool), "plates_rejected": 0, "clips": 0}
    while made < args.clips:
        tried += 1
        if tried > 20 * args.clips:
            break
        video, capture = draws.choice(captures)
        fps = capture.get(cv2.CAP_PROP_FPS)
        step = max(1, int(round(fps / FPS)))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start = draws.randrange(0, max(1, total - args.length * step - 1))
        plate = read_plate(capture, start, args.length, step)
        if plate is None:
            continue
        if not ball_free(teacher_heat_max(teacher, plate, device)):
            stats["plates_rejected"] += 1
            continue
        _, track = draws.choice(tracks)
        placed = place_track(track, draws)
        if placed is None:
            continue
        rows = [point[1] for point in placed if point is not None]
        crop = pick_crop(draws, banks, pool, float(np.median(rows)))
        exposure = draws.uniform(*copy_paste.EXPOSURE)
        frames = render_clip(plate, placed, banks, crop, exposure, frames_per_step=step)
        name = f"synth_{video.stem[:12].replace(' ', '_')}_{start:07d}"
        write_video(args.out / "videos" / f"{name}.mp4", frames)
        write_pseudo_labels(pseudo_csv(args.out, name), label_rows(placed))
        made += 1
        if made % 25 == 0:
            print(f"[synth] {made}/{args.clips} clips, {stats['plates_rejected']} plates rejected, "
                  f"{time.perf_counter() - started:.0f}s")
    stats["clips"] = made
    stats["seconds"] = round(time.perf_counter() - started, 1)
    (args.out / "synthetic.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[synth] {stats}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
