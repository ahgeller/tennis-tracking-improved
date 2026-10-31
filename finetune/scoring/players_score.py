"""Score a pipeline run's player tracks against the player identity labels (``labels/<clip>_players.csv``).

    python -m finetune.scoring.players_score --out-dir output/label_candidates [--holdout mixed_v1]
        [--compare before.log]

On every labelled frame the run's boxes (``player_boxes``: track id -> box) are matched one to one
to the labelled boxes, greedily by overlap, at an intersection over union of ``MATCH_IOU`` or more.
Each run track is then given at most one player, and each player at most one track, so that the
most matched box-frames agree (an assignment over the whole clip, as IDF1 makes it: a player the
tracker split over three ids is right on one of them only). Per clip and pooled:

- ``boxes``: labelled player box-frames (P1..P4; boxes labelled ``none`` are not players);
- ``coverage``: the share of them some run box matched;
- ``identity``: the share of them matched by a box whose track was given that player: the share of
  track-frames assigned to the right player;
- ``switches``: how often, between one labelled frame of a player and the next, the track
  matching it changed (identity switches; a frame it went unmatched is skipped over);
- ``non_players``: of the boxes labelled ``none``, the share some track given a player matched (a
  spectator or ball kid taken for a player).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from finetune.data import workspace as ws
from finetune.data.player_labels import NOT_PLAYER, PLAYERS, PlayerBoxLabel, read_player_labels
from finetune.scoring.label_scoring import (ALL, LabelledClip, add_run_arguments, finish, label_archive, labelled_clips,
                                            metric, metric_line, no_labels, share)
from finetune.scoring.tracker_events import player_boxes_by_frame, read_run, run_json

MATCH_IOU = 0.5
KEYS = ("coverage", "identity", "switches", "non_players")


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_frame(labels: Sequence[PlayerBoxLabel], boxes: Dict[str, List[float]]) -> List[Tuple[int, str]]:
    """(label index, track) pairs on one frame, one to one, the largest overlaps first, at MATCH_IOU or more."""
    candidates = sorted(((box_iou(label.box, box), li, track) for li, label in enumerate(labels)
                         for track, box in boxes.items()), key=lambda c: (-c[0], c[1], c[2]))
    pairs, used_labels, used_tracks = [], set(), set()
    for overlap, li, track in candidates:
        if overlap < MATCH_IOU:
            break
        if li in used_labels or track in used_tracks:
            continue
        pairs.append((li, track))
        used_labels.add(li)
        used_tracks.add(track)
    return pairs


def assign_tracks(counts: Dict[Tuple[str, str], int]) -> Dict[str, str]:
    """track -> player, one to one, the most matched box-frames in agreement (Hungarian)."""
    tracks = sorted({track for track, _ in counts})
    players = sorted({player for _, player in counts})
    if not tracks or not players:
        return {}
    gain = np.zeros((len(tracks), len(players)))
    for (track, player), count in counts.items():
        gain[tracks.index(track), players.index(player)] = count
    rows, cols = linear_sum_assignment(-gain)
    return {tracks[r]: players[c] for r, c in zip(rows, cols) if gain[r, c] > 0}


class ClipPlayerScore(NamedTuple):
    clip: str
    boxes: int
    covered: int
    right: int
    switches: int
    non_players: int
    non_players_taken: int
    tracks: int


def score_clip(clip: str, labels: Sequence[PlayerBoxLabel], run_boxes: Dict[int, Dict[str, List[float]]]) -> ClipPlayerScore:
    by_frame: Dict[int, List[PlayerBoxLabel]] = defaultdict(list)
    for label in labels:
        by_frame[label.frame].append(label)
    matched: List[Tuple[int, PlayerBoxLabel, str]] = []     # (frame, label, track)
    for frame in sorted(by_frame):
        frame_labels = by_frame[frame]
        for li, track in match_frame(frame_labels, run_boxes.get(frame, {})):
            matched.append((frame, frame_labels[li], track))
    counts = Counter((track, label.player) for _, label, track in matched if label.player in PLAYERS)
    mapping = assign_tracks(counts)
    players = [label for label in labels if label.player in PLAYERS]
    right = sum(1 for _, label, track in matched if label.player in PLAYERS and mapping.get(track) == label.player)
    switches = 0
    last_track: Dict[str, str] = {}
    for _, label, track in matched:                          # in frame order
        if label.player not in PLAYERS:
            continue
        if label.player in last_track and last_track[label.player] != track:
            switches += 1
        last_track[label.player] = track
    non_players = [label for label in labels if label.player == NOT_PLAYER]
    taken = sum(1 for _, label, track in matched if label.player == NOT_PLAYER and track in mapping)
    covered = sum(1 for _, label, _ in matched if label.player in PLAYERS)
    return ClipPlayerScore(clip, len(players), covered, right, switches, len(non_players), taken,
                           len({track for _, _, track in matched}))


def pooled(scores: Sequence[ClipPlayerScore]) -> ClipPlayerScore:
    return ClipPlayerScore(ALL, *(sum(getattr(s, field) for s in scores) for field in ClipPlayerScore._fields[1:]))


def format_line(score: ClipPlayerScore) -> str:
    return metric_line(score.clip, [
        [f"boxes={score.boxes:5d}", metric("coverage", share(score.covered, score.boxes)),
         metric("identity", share(score.right, score.boxes))],
        [f"switches={score.switches:3d}", f"tracks={score.tracks:3d}"],
        [metric("non_players", share(score.non_players_taken, score.non_players)), f"non_player_boxes={score.non_players:4d}"],
    ])


def score_run(clips: Sequence[LabelledClip], out_dir: Path, model: str) -> List[str]:
    lines = [f"=== player identity [{model}] boxes match at overlap >= {MATCH_IOU} ==="]
    scores = []
    for clip, _, players_path in clips:
        payload = read_run(run_json(out_dir, clip, model))
        score = score_clip(clip, read_player_labels(players_path), player_boxes_by_frame(payload))
        scores.append(score)
        lines.append(format_line(score))
    lines.append(format_line(pooled(scores)))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    archive = args.archive or label_archive()
    clips = labelled_clips(args.clips, archive, args.include_excluded, args.holdout, ws.players_csv)
    if not clips:
        return no_labels("player identity", archive, "labels/<clip>_players.csv")
    print(f"[archive] {archive}  ({len(clips)} clip{'' if len(clips) == 1 else 's'} with player labels: {', '.join(clip for clip, _, _ in clips)})")
    try:
        lines = [line for model in args.models for line in score_run(clips, args.out_dir, model)]
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return finish(lines, args.compare, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())

