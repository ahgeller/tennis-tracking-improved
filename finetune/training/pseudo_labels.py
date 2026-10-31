"""Pseudo-labels for unlabelled tennis video (Noisy Student, Xie et al., arXiv 1911.04252): the
teacher pipeline (the ball detector with the tracker's selector, ``pretrack.py --source pipeline``'s
path) answers every frame, and only the frames it is sure of become training labels.

    python finetune/ft.py pseudo add D:\\caltennis\\court5\\*.mp4     copy videos into finetune/pseudo/videos
    python finetune/ft.py pseudo label                             run the teacher on every video without labels
    python finetune/ft.py pseudo label --clips court5_N_01 --force
    python finetune/ft.py pseudo status                            what is there, and how much of it is kept

Where the data lives: ``<workspace>/pseudo/videos/<clip>.mp4`` and ``<workspace>/pseudo/labels/
<clip>_pseudo.csv`` (``frame,ball_x,ball_y,state``, one row per label frame at the trainer's cadence,
every second frame of a 60 FPS clip; ``state`` is ``ball``, ``none`` or ``ignore``), plus the
pipeline's tracking JSON beside it (``<clip>_pipeline.json``, which ``--reuse`` filters again
without the GPU). The trainer reads the folder with ``--pseudo``; ``ft.py colab-zip`` packs it.

The filter (``frame_state``) was measured, not guessed: the shipped teacher's pipeline run over
the 24 held-out clips of ``mixed_v1`` (which it never trained on; 9,033 labels, 518 absent),
scored against their labels (``audit/records/opt_training.md``, "Pseudo-label filter"):

- ``ball``: the pipeline chose a ball that is the detector's own measurement (not a recovered or
  interpolated fill), its selected position within ``MAX_DRIFT_PX`` (1080p) of that measurement,
  on a track the timeline chain used with ``MIN_TRACK_OBS`` detections or more, at a detector
  confidence of ``MIN_CONF`` or more: 99.89% of
  such frames hold the labelled ball within 10 px (74% of the visible balls are kept; median
  error 0.83 px). Taking every pipeline answer instead would be 97.5% correct.
- ``none``: no detection at or above the run's threshold within ``QUIET_STEPS`` label frames
  either side: 99.1% of such frames have no ball. A frame the pipeline merely left empty has a
  ball in half the cases (the detector missed it), so it is never a negative.
- ``ignore``: everything else. The loss leaves these frames out (``loop.heatmap_loss``'s `valid`)
  rather than calling them negatives, the way detection self-training keeps uncertain pseudo
  boxes out of the background (STAC, arXiv 2005.04757; Soft Teacher, arXiv 2106.09018); the
  distillation loss still reads them, since the teacher's soft answer needs no label.

Unlabelled sources (``finetune/DATASETS.md`` section 2.5): Mendeley 75m8vz7jr2 (GoPro, a camera
behind the baseline, CC BY 4.0) and CalTennis (Hugging Face ``demalenk/caltennis``, CC BY-NC 4.0);
``finetune/COLAB_QUICKSTART.md`` has the download commands. A clip must be 30 or 60 FPS; ``add``
says how to re-encode one that is not.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import finetune  # noqa: F401,E402  (which puts system/ on sys.path)

from finetune.data import workspace as ws
from finetune.data.fileio import atomic_csv
from finetune.data.workspace import video_meta, videos_in
from tennis_tracker.ball_weights import operating_conf
from tennis_tracker.frame_rate import phase_stride

PSEUDO_DIR = "pseudo"            # under the workspace root
PSEUDO_SUFFIX = "_pseudo.csv"
PSEUDO_SOURCE = "pseudo"         # the source tag its units carry in the trainer's per-source lines
COLUMNS = ("frame", "ball_x", "ball_y", "state")
BALL, NONE, IGNORE = "ball", "none", "ignore"
STATES = (BALL, NONE, IGNORE)
MAX_DRIFT_PX = 3.0               # the selected position against the detector's own (1080p px)
MIN_CONF = 0.7
MIN_TRACK_OBS = 10               # detections on the chosen track: a short track is the likeliest wrong object
QUIET_STEPS = 15                 # label frames either side without any detection for a "none"
REFERENCE_WIDTH = 1920.0

PseudoTarget = Optional[List[float]]


def pseudo_root(root: Optional[Path] = None) -> Path:
    return (root or ws.current().root) / PSEUDO_DIR


def pseudo_csv(root: Path, clip: str) -> Path:
    return root / "labels" / f"{clip}{PSEUDO_SUFFIX}"


def is_pseudo_csv(path: Path) -> bool:
    return path.name.endswith(PSEUDO_SUFFIX)


# --------------------------------------------------------------------------- the file format

def write_pseudo_labels(path: Path, rows: Mapping[int, Tuple[str, Optional[Tuple[float, float]]]]) -> None:
    """One row per frame of `rows` (frame -> (state, point or None)), sorted, written atomically."""
    with atomic_csv(path) as writer:
        writer.writerow(COLUMNS)
        for frame in sorted(rows):
            state, point = rows[frame]
            if state not in STATES:
                raise ValueError(f"unknown pseudo-label state {state!r}")
            writer.writerow([frame, "" if point is None else f"{point[0]:.3f}", "" if point is None else f"{point[1]:.3f}", state])


def read_pseudo_labels(path: Path, width: int, height: int) -> Tuple[Dict[int, PseudoTarget], Set[int]]:
    """A pseudo-label file as (frame -> [x, y] clamped onto the image or None, the ignored frames);
    the ignored frames are in the first map too, as None. A row with an unknown state or a ball
    without coordinates is a ValueError naming the file."""
    targets: Dict[int, PseudoTarget] = {}
    ignored: Set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame, state = int(row["frame"]), row["state"]
            if state == BALL:
                if not row["ball_x"] or not row["ball_y"]:
                    raise ValueError(f"{path}: frame {frame} is a ball without coordinates")
                targets[frame] = [min(max(float(row["ball_x"]), 0.0), width - 1.0),
                                  min(max(float(row["ball_y"]), 0.0), height - 1.0)]
            elif state in (NONE, IGNORE):
                targets[frame] = None
                if state == IGNORE:
                    ignored.add(frame)
            else:
                raise ValueError(f"{path}: frame {frame} has the unknown state {state!r}")
    return targets, ignored


# --------------------------------------------------------------------------- the filter

def frame_states(payload: Mapping, stride: int) -> Dict[int, Tuple[str, Optional[Tuple[float, float]]]]:
    """Every label frame of a pipeline run's tracking JSON (`payload`; every `stride`-th source
    frame from 0) as (state, point): the measured filter of the module docstring."""
    video = payload["video"]
    scale = REFERENCE_WIDTH / float(video["width"])
    tracks = {track["track_id"]: track for track in payload.get("tracks") or ()}
    detected = sorted({int(item["frame"]) for track in tracks.values() for item in track["observations"]})
    rows = {int(row["frame"]): row for row in payload["frames"]}
    states: Dict[int, Tuple[str, Optional[Tuple[float, float]]]] = {}
    for frame in range(0, int(video["total_frames"]), stride):
        states[frame] = frame_state(frame, rows.get(frame, {"present": False}), tracks, detected, stride, scale)
    return states


def frame_state(frame: int, row: Mapping, tracks: Mapping[int, Mapping], detected: Sequence[int], stride: int,
                scale: float) -> Tuple[str, Optional[Tuple[float, float]]]:
    """Frame `frame`'s (state, point) from its pipeline `row`: ``ball`` at the pipeline's position,
    ``none`` or ``ignore`` (`detected`: every frame with a detection, sorted)."""
    if row.get("present"):
        measurement = row.get("measurement")
        track_id = row.get("track_id")
        track = None if track_id is None else tracks.get(int(track_id))
        if (measurement and row.get("position_kind") != "recovered" and track is not None and track.get("in_timeline")
                and int(track.get("num_obs", 0)) >= MIN_TRACK_OBS
                and float(measurement["conf"]) >= MIN_CONF
                and math.hypot(row["x"] - measurement["x"], row["y"] - measurement["y"]) * scale <= MAX_DRIFT_PX):
            return BALL, (float(row["x"]), float(row["y"]))
        return IGNORE, None
    return (NONE if quiet(detected, frame, QUIET_STEPS * stride) else IGNORE), None


def quiet(detected: Sequence[int], frame: int, reach: int) -> bool:
    """No detected frame within `reach` source frames of `frame` (`detected` sorted)."""
    position = bisect.bisect_left(detected, frame - reach)
    return position == len(detected) or detected[position] > frame + reach


# --------------------------------------------------------------------------- the commands

def clip_name(source: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)


def cmd_add(args: argparse.Namespace) -> int:
    """Copy videos into ``pseudo/videos``; one that is not 30 or 60 FPS is skipped with the
    ffmpeg line that re-encodes it."""
    root = pseudo_root()
    (root / "videos").mkdir(parents=True, exist_ok=True)
    status = 0
    for source in map(Path, args.videos):
        if not source.is_file():
            print(f"[skip] not a file: {source}")
            status = 1
            continue
        destination = root / "videos" / f"{clip_name(source)}{source.suffix.lower()}"
        try:
            meta = video_meta(source)
        except RuntimeError as error:
            print(f"[skip] {error}")
            status = 1
            continue
        if phase_stride(meta.fps) is None:
            print(f"[skip] {source.name}: {meta.fps:.2f} FPS is neither 30 nor 60. Re-encode first, e.g.\n"
                  f"       ffmpeg -i \"{source}\" -vf fps=60 -c:v libx264 -crf 14 \"{destination.with_suffix('.mp4')}\"")
            status = 1
            continue
        if destination.exists() and not args.force:
            print(f"[keep] {destination.name} already exists (use --force to overwrite)")
            continue
        shutil.copyfile(source, destination)
        print(f"[add ] {destination.name}: {meta.width}x{meta.height} @{meta.fps:.2f} FPS, {meta.total} frames")
    return status


def videos(root: Path) -> Dict[str, Path]:
    return videos_in(root / "videos")


def summary(states: Mapping[int, Tuple[str, Optional[Tuple[float, float]]]]) -> str:
    counts = {state: sum(1 for value, _ in states.values() if value == state) for state in STATES}
    total = max(len(states), 1)
    return " ".join(f"{state} {counts[state]} ({counts[state] / total:.0%})" for state in STATES)


def cmd_label(args: argparse.Namespace) -> int:
    """Run the teacher pipeline over every video without a pseudo-label file (or `--clips`) and
    write the filtered labels; ``--reuse`` filters an existing ``<clip>_pipeline.json`` again."""
    root = pseudo_root()
    (root / "labels").mkdir(parents=True, exist_ok=True)
    found = videos(root)
    chosen = [clip for clip in found if not args.clips or clip in set(args.clips)]
    if not chosen:
        print(f"no videos to pseudo-label in {root / 'videos'} (ft.py pseudo add <videos> first)")
        return 1
    weights = Path(args.weights).resolve()
    for clip in chosen:
        target = pseudo_csv(root, clip)
        if target.is_file() and not args.force and not args.reuse:
            print(f"[skip] {clip}: {target.name} exists (--force to redo)")
            continue
        json_path = root / "labels" / f"{clip}_pipeline.json"
        meta = video_meta(found[clip])
        stride = phase_stride(meta.fps)
        if stride is None:
            print(f"[skip] {clip}: {meta.fps:.2f} FPS is neither 30 nor 60")
            continue
        if not (args.reuse and json_path.is_file()):
            from tennis_tracker.headless import run_headless     # the pipeline: CUDA and TensorRT, only here

            print(f"[pipe] {clip}: {meta.width}x{meta.height} @ {meta.fps:.2f} FPS, {meta.total} frames, teacher {weights.name}")
            # The teacher runs at the threshold the tracker runs its weights at (it was 0.50, a copy of
            # the tracker's old default, until 2026-10-30); MIN_CONF filters what is kept.
            run_headless(found[clip], model_path=weights, conf=operating_conf(weights), device=args.device,
                         tracking_json=json_path)
        states = frame_states(json.loads(json_path.read_text(encoding="utf-8")), stride)
        write_pseudo_labels(target, states)
        print(f"[pseudo] {target.name}: {summary(states)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = pseudo_root()
    found = videos(root)
    if not found:
        print(f"no pseudo-label videos in {root / 'videos'}")
        return 0
    for clip, video in found.items():
        target = pseudo_csv(root, clip)
        if not target.is_file():
            print(f"{clip:40s} {video.stat().st_size / 2**20:8.0f} MB  not labelled yet")
            continue
        meta = video_meta(video)
        targets, ignored = read_pseudo_labels(target, meta.width, meta.height)
        balls = sum(point is not None for point in targets.values())
        print(f"{clip:40s} {video.stat().st_size / 2**20:8.0f} MB  {len(targets)} label frames: ball {balls}, "
              f"none {len(targets) - balls - len(ignored)}, ignore {len(ignored)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("add", help="copy unlabelled videos into the workspace's pseudo/videos")
    p.add_argument("videos", nargs="+")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_add)
    p = commands.add_parser("label", help="run the teacher pipeline and write the filtered pseudo-labels")
    p.add_argument("--clips", nargs="*")
    p.add_argument("--weights", default=str(ws.current().repo_weights), help="the teacher (default: the workspace's model, models/ball_detector.pt)")
    p.add_argument("--device", default="0")
    p.add_argument("--force", action="store_true", help="redo clips that already have pseudo-labels")
    p.add_argument("--reuse", action="store_true", help="filter the saved pipeline JSON again instead of rerunning it")
    p.set_defaults(func=cmd_label)
    commands.add_parser("status", help="the pseudo-label videos and what their labels keep").set_defaults(func=cmd_status)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
