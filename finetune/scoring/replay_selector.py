"""Replay the ball selector on the CPU over the detections a pipeline run recorded, and score it.

    python -m finetune.scoring.replay_selector output/ctrl_eval/pipeline --archive finetune
    python -m finetune.scoring.replay_selector output/ctrl_eval/pipeline --variant chain union raw control

The tracking JSON of a ``evaluate_archive.py --mode pipeline`` run holds, in its ``tracks``, every
detection the selector saw (each at or above the run's detector threshold, with the motion-mask
flag the pipeline gave it) and, in its ``frames``, the court and the player boxes of every frame.
Feeding those back into ``select_ball_in_play`` runs the whole selector again without a GPU, a
video or the motion masks: a selector change is measured against the labels in seconds, before
the owner's GPU run confirms it. Without the masks the fills cannot find motion blobs, so the few
``motion`` rows of the real run are missing here. Measured on the 24 held-out clips of the
2026-10-26 control run (output/ctrl_eval): the control hit 7946 labels, the replay 7923; 23 of
the 24 hits lost are ``motion`` rows, and apart from those the two differ on 7 frames (3 hits
gained, 2 lost, the rest fills that moved) out of 8515 labelled visible. The replay cannot see
detections the run's threshold rejected either: a change that admits weaker detections is only
measurable by a GPU run at the lower threshold.

Variants, each a row per clip, per clips.csv source and over all clips (the evaluator's metric
line, with the false alarms split into bridged and false, then the hits and tracks):

``control``  the run's own ``frames`` as written: what the GPU produced, for the fidelity check
``chain``    the selector as it is: tracks, scoring, the timeline chain, fills and refit
``union``    every track's observations output, no scoring and no chain, the fills and the refit
             run over the union: what the chooser is worth against taking every track
``raw``      every recorded detection as the output, no selector at all: the detector's answer
             at the run's threshold

The replay patches ``ball_in_play_selector.core.build_detections`` for the duration of a call so
the recorded rows, motion flags included, stand in for the ones the masks would have produced.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np

from ball_in_play_selector import Detection, FrameResult, SelectorConfig, Track, select_ball_in_play
from ball_in_play_selector import core
from ball_in_play_selector.association import build_tracks
from ball_in_play_selector.core import frame_result
from finetune.data.data_policy import clip_source
from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.labels import Labels, read_labels
from finetune.data.workspace import MANIFEST_NAME, natural_key, read_manifest
from finetune.scoring.archive import archive_folders, default_archive, fmt
from finetune.scoring.pipeline_score import ClipScore, score_pipeline_clip
from tennis_tracker.court.court_geometry import build_court_polygon
from tennis_tracker.export import frame_result_to_json, track_to_json
from tennis_tracker.tracking_schema import FrameRow, TrackingJson

VARIANTS = ("control", "chain", "union", "raw")
DEFAULT_VARIANTS = ("control", "chain")


def recorded_detections(payload: TrackingJson) -> List[List[Detection]]:
    """The detections the run's selector saw, per source frame, rebuilt from every track's
    observations (the detector gives one candidate per frame, so no frame holds two)."""
    rows: List[List[Detection]] = [[] for _ in range(int(payload["video"]["total_frames"]))]
    for track in payload.get("tracks") or ():
        for item in track["observations"]:
            x1, y1, x2, y2 = (float(value) for value in item["bbox"])
            rows[int(item["frame"])].append(Detection(
                frame=int(item["frame"]), cx=float(item["x"]), cy=float(item["y"]), x1=x1, y1=y1, x2=x2, y2=y2,
                conf=float(item["conf"]), area=float(item["area"]), on_motion=bool(item["on_motion"]),
            ))
    for frame_rows in rows:
        frame_rows.sort(key=lambda det: -det.conf)
    return rows


def _raw_rows(detections: List[List[Detection]]) -> List[List[Tuple[Sequence[float], float]]]:
    """The ``(bbox, conf)`` rows ``select_ball_in_play`` is handed; only their count per frame is
    read, since ``build_detections`` is patched to return the recorded ``Detection`` rows."""
    return [[((det.x1, det.y1, det.x2, det.y2), det.conf) for det in frame_rows] for frame_rows in detections]


class ReplayScene(NamedTuple):
    """What a clip's JSON gives the selector besides the detections."""
    fps: float
    width: int
    height: int
    total: int
    court_polygon: Optional[np.ndarray]
    player_boxes: List[Optional[Dict[str, Optional[List[float]]]]]

    @classmethod
    def of(cls, payload: TrackingJson) -> "ReplayScene":
        video = payload["video"]
        frames = payload["frames"]
        with contextlib.redirect_stdout(io.StringIO()):          # build_court_polygon reports what it built
            court_polygon, _ = build_court_polygon([row.get("court_keypoints") for row in frames])
        return cls(float(video["fps"]), int(video["width"]), int(video["height"]), int(video["total_frames"]),
                   court_polygon, [row.get("player_boxes") for row in frames])


class Replay(NamedTuple):
    """One variant's output on one clip: the per-frame results and the tracks it built."""
    per_frame: List[Optional[FrameResult]]
    tracks: List[Track]
    chosen: Optional[Track]


@contextlib.contextmanager
def recorded_rows(detections: List[List[Detection]]):
    """Make ``core.build_detections`` hand back these rows, motion flags and all, instead of
    reading motion masks the replay does not have."""
    with patch.object(core, "build_detections", lambda detections_by_frame, motion_masks=None: detections):
        yield


def replay_chain(detections: List[List[Detection]], scene: ReplayScene) -> Replay:
    """The selector as the pipeline runs it, over the recorded detections."""
    with recorded_rows(detections):
        per_frame, chosen, tracks = select_ball_in_play(
            _raw_rows(detections), scene.fps, scene.width, scene.height, court_polygon=scene.court_polygon,
            player_boxes_by_frame=scene.player_boxes)
    return Replay(per_frame, tracks, chosen)


def replay_union(detections: List[List[Detection]], scene: ReplayScene) -> Replay:
    """Every track output: association, then the fills and the refit over the union of the tracks,
    with no scoring and no timeline chain between them."""
    cfg = SelectorConfig(fps=scene.fps, width=scene.width, height=scene.height)
    tracks = build_tracks(detections, cfg)
    output_scene = core.make_scene(cfg, scene.fps, scene.total, scene.width, scene.height, None, None,
                                   scene.player_boxes)
    for track in tracks:
        track.in_timeline = True
    return Replay(core.output_trajectory(output_scene, tracks), tracks, None)


def replay_raw(detections: List[List[Detection]], scene: ReplayScene) -> Replay:
    """The detector's answer alone: the most confident recorded detection of each frame."""
    per_frame: List[Optional[FrameResult]] = [None] * scene.total
    for frame, frame_rows in enumerate(detections):
        if frame_rows:
            per_frame[frame] = frame_result(frame_rows[0].cx, frame_rows[0].cy, "det", detection=frame_rows[0])
    return Replay(per_frame, [], None)


def replay_payload(replay: Replay, payload: TrackingJson) -> TrackingJson:
    """A tracking payload of the replay's output, in the shape of the run's own JSON: the frames
    table (the run's court and player columns kept), the tracks and the chosen track."""
    frames: List[FrameRow] = []
    for row in payload["frames"]:
        frame = int(row["frame"])
        replayed = frame_result_to_json(frame, replay.per_frame[frame] if frame < len(replay.per_frame) else None)
        for key in ("court_keypoints", "player_boxes"):
            if key in row:
                replayed[key] = row[key]
        frames.append(replayed)
    out = dict(payload)
    out["frames"] = frames
    out["tracks"] = [track_to_json(track) for track in replay.tracks]
    out["chosen_track"] = track_to_json(replay.chosen)
    return out      # type: ignore[return-value]


def replay_variant(variant: str, payload: TrackingJson) -> TrackingJson:
    """The tracking payload of one variant on one clip's JSON (the run's own for ``control``)."""
    if variant == "control":
        return payload
    detections = recorded_detections(payload)
    scene = ReplayScene.of(payload)
    replayed = {"chain": replay_chain, "union": replay_union, "raw": replay_raw}[variant](detections, scene)
    return replay_payload(replayed, payload)


class ClipReplay(NamedTuple):
    clip: str
    source: str
    score: ClipScore
    tracks: int


def score_payload(clip: str, source: str, payload: TrackingJson, labels: Labels) -> ClipReplay:
    scored = score_pipeline_clip(labels, payload, int(payload["video"]["width"]))
    return ClipReplay(clip, source, scored, len(payload.get("tracks") or ()))


def _hits(stats: DetectorMetrics) -> int:
    return round(stats.recall * stats.visible)


def _wrong(stats: DetectorMetrics) -> int:
    return round(stats.wrong * stats.visible)


def line(name: str, stats: DetectorMetrics, tracks: int) -> str:
    """The evaluator's metric line with the counts a selector change is judged by after it."""
    return f"{fmt(name, stats)} | hits={_hits(stats):5d} wrong={_wrong(stats):3d} tracks={tracks:4d}"


def report_variant(variant: str, scored: List[ClipReplay]) -> List[str]:
    """The per-clip lines, then one per source and ``ALL``."""
    lines = [f"=== replay [{variant}] ==="]
    lines += [line(item.clip, item.score.stats, item.tracks) for item in scored]
    for source in sorted({item.source for item in scored}):
        of_source = [item for item in scored if item.source == source]
        lines.append(line(f"source={source}", DetectorMetrics.of_clips([item.score.pairs for item in of_source], 1.0),
                          sum(item.tracks for item in of_source)))
    lines.append(line("ALL", DetectorMetrics.of_clips([item.score.pairs for item in scored], 1.0),
                      sum(item.tracks for item in scored)))
    return lines


def _labelled_clips(run_dir: Path, model: str, archive: Path, names: Optional[Sequence[str]]
                    ) -> List[Tuple[str, str, TrackingJson, Labels]]:
    label_dir, _ = archive_folders(archive)
    manifest = read_manifest(archive / MANIFEST_NAME)
    suffix = f"_{model}.json"
    clips = []
    for path in sorted(run_dir.glob(f"*{suffix}"), key=lambda item: natural_key(item.name)):
        clip = path.name[:-len(suffix)]
        if names and clip not in names:
            continue
        csv_path = label_dir / f"{clip}_ball.csv"
        if not csv_path.is_file():
            print(f"[skip] {clip}: no label file in {label_dir}", file=sys.stderr)
            continue
        payload: TrackingJson = json.loads(path.read_text(encoding="utf-8"))
        labels = read_labels(csv_path, int(payload["video"]["width"]), int(payload["video"]["height"]))
        clips.append((clip, clip_source(clip, manifest), payload, labels))
    return clips


def report(run_dir: Path, model: str, archive: Path, variants: Sequence[str],
           names: Optional[Sequence[str]] = None) -> List[str]:
    """Every variant's report over the run's labelled clips."""
    clips = _labelled_clips(run_dir, model, archive, names)
    lines: List[str] = []
    if not clips:
        return lines
    for variant in variants:
        scored = [score_payload(clip, source, replay_variant(variant, payload), labels)
                  for clip, source, payload, labels in clips]
        if lines:
            lines.append("")
        lines.extend(report_variant(variant, scored))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A pipeline-mode evaluator output folder of tracking JSONs")
    parser.add_argument("--model", default="gridtracknet", help="The JSON suffix's model name")
    parser.add_argument("--archive", default=None,
                        help="Label set: a finetune-style folder with labels/ and clips.csv "
                             "(default: $TENNIS_ARCHIVE, else finetune/)")
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems to replay (default: every JSON with labels)")
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS),
                        help="Which outputs to score (see the module docstring)")
    args = parser.parse_args(argv)
    archive = Path(args.archive) if args.archive else default_archive()
    lines = report(args.run_dir, args.model, archive, args.variant, args.clips)
    if not lines:
        print(f"no {args.model} tracking JSONs with labels under {args.run_dir}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
