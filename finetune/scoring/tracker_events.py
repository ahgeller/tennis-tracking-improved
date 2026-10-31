"""What a pipeline run says happened, in the terms the new label types use: its landings, its hits
(with the hitter's side), its rallies, its player tracks and its court keypoints.

A pipeline run is a folder of tracking JSONs, ``<out-dir>/<clip>_<model>.json``, as
``evaluate_archive.py --mode pipeline --court --out-dir <out-dir>`` writes them (``--court`` fills
their ``court_events`` block). The same folder serves twice: the label tool offers its events as
candidates to confirm or reject (``finetune.labeling.label_candidates``), and the scorers
(``hits_score``, ``rallies_score``, ``players_score``, ``court_score``) measure it against the labels.
So the tracker's answer is turned into events here, once, for both.

- Landings: every landing of every list, as the landing scorer's ``all`` tier keeps them
  (``landings_at_tier``, whose tiers the landing scorer offers: ``court_events_score``).
- Hits: the racket contacts (``court_events.contacts``), each with the side of the court its
  player stands on (``court_side``: the contact is anchored at the player's feet).
- Rallies: the rally legs grouped into rallies (``rallies_from_legs``): a serve leg, or a leg
  starting more than ``RALLY_GAP_S`` after the last one ended, starts a new rally; a rally starts
  where its first leg starts (the serve's hit) and ends where its last one ends.
- Player tracks: each frame's ``player_boxes`` (track id -> box).
- The court: each frame's ``court_keypoints``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

from finetune.data.event_labels import Hitter
from tennis_tracker.court.court_geometry import NET_V, build_ground_projection_model, homography_apply
from tennis_tracker.court.court_scene import EVENT_MERGE_S, BounceEvent, seconds_to_frames
from tennis_tracker.court.rally_legs import inferred_landings
from tennis_tracker.tracking_schema import BounceEventJson, CourtEventsJson, FrameRow, RallyLegJson, TrackingJson

RUN_MODEL = "gridtracknet"      # the model name evaluate_archive.py writes into the JSON names by default
RALLY_GAP_S = 2.5               # a leg starting this long after the last one ended starts a new rally
                                # (rally_legs' own pause before a leg may be a serve)

PlayerBoxesFrame = Dict[str, List[float]]


class TrackerHit(NamedTuple):
    """A racket contact the tracker found: its frame, score, how it was found, the hitter's side
    and the ball where the contact was judged (None when the stage had none)."""

    frame: int
    score: float
    source: str
    hitter: Optional[Hitter]
    ball: Optional[Tuple[float, float]] = None


TIERS = ("confirmed", "placed", "all")     # the landing tiers (landings_at_tier)


# --------------------------------------------------------------------------- landings

def bounce_from_json(row: BounceEventJson) -> BounceEvent:
    return BounceEvent(int(row["frame"]), float(row["x"]), float(row["y"]), float(row["score"]),
                       row["origin"], row["position"])


def without_repeats(events: Sequence[BounceEvent], fps: float) -> List[BounceEvent]:
    """``events`` in the order given, dropping each one within ``EVENT_MERGE_S`` of one already kept."""
    merge = seconds_to_frames(EVENT_MERGE_S, fps)
    kept: List[BounceEvent] = []
    for event in events:
        if not any(abs(event.frame - other.frame) <= merge for other in kept):
            kept.append(event)
    return kept


def landings_at_tier(block: CourtEventsJson, tier: str, fps: float) -> List[BounceEvent]:
    """The block's landings the tier admits, in frame order (the tiers: ``court_events_score``)."""
    confirmed = [bounce_from_json(row) for row in block.get("bounces", [])]
    candidates = [bounce_from_json(row) for row in block.get("bounce_candidates", [])]
    kinematic = [bounce_from_json(row) for row in block.get("kinematic_bounces", [])]
    if tier == "confirmed":
        landings = list(confirmed)
    elif tier == "placed":
        landings = confirmed + inferred_landings(confirmed, candidates, kinematic, fps)
    elif tier == "all":
        inferred = sorted(candidates + kinematic, key=lambda event: -event.score)
        landings = without_repeats(confirmed + inferred, fps)
    else:
        raise ValueError(f"unknown tier {tier!r}; expected one of {', '.join(TIERS)}")
    return sorted(landings, key=lambda event: event.frame)


def run_json(out_dir: Path, clip: str, model: str = RUN_MODEL) -> Path:
    return out_dir / f"{clip}_{model}.json"


def read_run(path: Path) -> TrackingJson:
    """A tracking JSON. Raises FileNotFoundError naming it when it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run evaluate_archive.py --mode pipeline --court "
                                f"--out-dir {path.parent} on the clip first")
    return cast(TrackingJson, json.loads(path.read_text(encoding="utf-8")))


def court_block(payload: TrackingJson, path: Path) -> CourtEventsJson:
    """The run's ``court_events`` block; a ValueError naming the file when it was written without
    ``--court``."""
    block = payload.get("court_events") or {}
    if not block:
        raise ValueError(f"{path} has no court events: it was written without --court "
                         "(evaluate_archive.py --mode pipeline --court)")
    return block


def run_fps(payload: TrackingJson) -> float:
    return float(payload["video"]["fps"])


def frame_keypoints(frames: Sequence[FrameRow], frame: int) -> Optional[List[float]]:
    """The court keypoints the run had on `frame`, None when it had none or the frame is not in it."""
    if 0 <= frame < len(frames) and frames[frame].get("frame") == frame:
        return frames[frame].get("court_keypoints")
    for row in frames:          # a table not indexed by frame: look the frame up
        if row.get("frame") == frame:
            return row.get("court_keypoints")
    return None


def court_side(keypoints: Optional[Sequence[float]], x: float, y: float) -> Optional[Hitter]:
    """"near" when the ground point (x, y) lies on the half of the court nearer the camera, "far"
    on the other half, None without a court to project it onto."""
    model = build_ground_projection_model(keypoints)
    if model is None:
        return None
    point = homography_apply(model["H_i2c"], x, y)
    if point is None:
        return None
    return "near" if point[1] > NET_V else "far"


# --------------------------------------------------------------------------- the events

def tracker_landings(block: CourtEventsJson, fps: float) -> List[BounceEvent]:
    """Every landing the run found, repeats within ``EVENT_MERGE_S`` dropped (the ``all`` tier)."""
    return landings_at_tier(block, "all", fps)


def tracker_hits(payload: TrackingJson, block: CourtEventsJson) -> List[TrackerHit]:
    """The run's racket contacts in frame order, each with its hitter's side."""
    frames = payload.get("frames", [])
    hits = []
    for contact in block.get("contacts", []):
        frame, ball = int(contact["frame"]), contact.get("ball")
        hitter = court_side(frame_keypoints(frames, frame), float(contact["x"]), float(contact["y"]))
        hits.append(TrackerHit(frame, float(contact["score"]), str(contact["source"]), hitter,
                               None if not ball else (float(ball["x"]), float(ball["y"]))))
    return sorted(hits, key=lambda hit: hit.frame)


def rallies_from_legs(legs: Sequence[RallyLegJson], fps: float) -> List[Tuple[int, int]]:
    """(start frame, end frame) of each rally the legs make, in order (see the module docstring)."""
    gap = seconds_to_frames(RALLY_GAP_S, fps)
    rallies: List[Tuple[int, int]] = []
    for leg in sorted(legs, key=lambda leg: int(leg["start_frame"])):
        start, end = int(leg["start_frame"]), int(leg["end_frame"])
        if rallies and leg.get("phase") != "serve" and start - rallies[-1][1] <= gap:
            rallies[-1] = (rallies[-1][0], max(rallies[-1][1], end))
        else:
            rallies.append((start, end))
    return rallies


def tracker_rallies(block: CourtEventsJson, fps: float) -> List[Tuple[int, int]]:
    return rallies_from_legs(block.get("rally_legs", []), fps)


def player_boxes_by_frame(payload: TrackingJson) -> Dict[int, PlayerBoxesFrame]:
    """frame -> {track id: [x1, y1, x2, y2]} for every frame the run tracked a person on."""
    out: Dict[int, PlayerBoxesFrame] = {}
    for row in payload.get("frames", []):
        boxes = {str(track): [float(v) for v in box[:4]] for track, box in (row.get("player_boxes") or {}).items()
                 if box is not None and len(box) >= 4}
        if boxes:
            out[int(row["frame"])] = boxes
    return out


def court_keypoints_by_frame(payload: TrackingJson) -> Dict[int, List[float]]:
    """frame -> the court keypoints the run had there, for every frame it had them on."""
    out: Dict[int, List[float]] = {}
    for row in payload.get("frames", []):
        keypoints = row.get("court_keypoints")
        if keypoints:
            out[int(row["frame"])] = [float(value) for value in keypoints]
    return out
