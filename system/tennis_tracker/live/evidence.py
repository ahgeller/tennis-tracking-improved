"""What a live run hands its fixed-lag selector, recorded so the selector can be replayed at another
lag on the CPU, and the one rule for the ghost ROIs that both the run and a replay follow.

Pass 1 finds a ghost ROI (the coasting region of an ROI track that timed out without the ball)
only when the track times out, some frames after the frames it covers; offline, every ghost is
cleared from the masks once pass 1 is over (``pass1.erase_ghost_rois``). Live, ``GhostEraser``
clears the ghosts known so far from the frames a selector run is about to read, so a frame
committed before its ghost was found keeps that motion, as it must: the ghost was not known yet.

Frames are source frames: a frame the live source dropped is fed as ``EMPTY_EVIDENCE`` (no
detections, no masks), so the selector's frame ``i`` is source frame ``i`` and its time between
frames is the clip's. ``save_evidence`` writes the masks as pass 1 made them (before any ghost was
cleared) with the ghost log, ``(commit frame, ghost frame, box)`` in the order pass 1 found them; ``replay`` feeds
them to a ``FixedLagSelector`` in arrival order, adding each ghost when pass 1 found it. Every
input of the selector is causal and independent of the lag, so ``replay`` commits exactly what a
live run at that lag commits (``tests/test_live_fixed_lag.py``).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from ball_in_play_selector import PackedMask

from ..pass1 import Pass1Result, _erase_ghosts_in_frame
from .fixed_lag import FixedLagSelector, LagSettings

Box = Tuple[int, int, int, int]
# A ghost ROI as pass 1 found it: the frame being committed, the earlier frame it covers, its box.
GhostEntry = Tuple[int, int, Box]
EVIDENCE_SUFFIX = ".evidence.pkl"
_FORMAT = 2              # 2: indexed by source frame, with source_index


class GhostEraser:
    """``FixedLagSelector.before_run``: clear, in the frames a run reads, the ghost ROIs that pass 1
    has added to them since the last run (``pass1._erase_ghosts_in_frame``, offline's own clearing)."""

    def __init__(self, result: Pass1Result) -> None:
        self.result = result
        self.erased: Dict[int, int] = {}       # frame -> ghosts already cleared from its masks

    def __call__(self, start: int, end: int) -> None:
        ghosts = self.result.ghost_rois
        for frame in range(start, min(end, len(ghosts))):
            known = len(ghosts[frame])
            if known > self.erased.get(frame, 0):
                _erase_ghosts_in_frame(self.result, frame, ghosts[frame])
                self.erased[frame] = known
        for frame in [frame for frame in self.erased if frame < start]:
            del self.erased[frame]


# One frame of selector input as pass 1 committed it: detections, boost and raw masks (before any
# ghost was cleared), player boxes, court keypoints.
FrameEvidence = Tuple[Any, Optional[PackedMask], Optional[PackedMask], Any, Optional[List[float]]]
# A source frame the live source dropped: nothing seen.
EMPTY_EVIDENCE: FrameEvidence = ([], None, None, {}, None)


class _EvidenceLists(Protocol):
    detections: List[Any]
    boost_masks: List[Optional[PackedMask]]
    raw_motions: List[Optional[PackedMask]]
    player_boxes: List[Any]
    court_keypoints: List[Optional[List[float]]]


def frame_evidence_of(evidence: _EvidenceLists, frame: int) -> FrameEvidence:
    """Frame `frame`'s selector input from per-frame lists (``Pass1Result`` or ``RecordedEvidence``)."""
    return (evidence.detections[frame], evidence.boost_masks[frame], evidence.raw_motions[frame],
            evidence.player_boxes[frame], evidence.court_keypoints[frame])


@dataclass
class RecordedEvidence:
    """One live run's selector input, per source frame: the detections, the masks as pass 1 made
    them, the player boxes and court keypoints (``EMPTY_EVIDENCE`` for a dropped frame), and the
    ghost log. ``source_index`` lists the source frames pass 1 saw; the others were dropped."""

    fps: float
    width: int
    height: int
    detections: List[Any] = field(default_factory=list)
    boost_masks: List[Optional[PackedMask]] = field(default_factory=list)
    raw_motions: List[Optional[PackedMask]] = field(default_factory=list)
    player_boxes: List[Any] = field(default_factory=list)
    court_keypoints: List[Optional[List[float]]] = field(default_factory=list)
    ghosts: List[GhostEntry] = field(default_factory=list)
    source_index: List[int] = field(default_factory=list)

    def add(self, evidence: FrameEvidence) -> None:
        """Record the next source frame's input as the selector was fed it."""
        detections, boost, raw, players, court = evidence
        self.detections.append(detections)
        self.boost_masks.append(boost)
        self.raw_motions.append(raw)
        self.player_boxes.append(players)
        self.court_keypoints.append(court)


def save_evidence(path: Path, evidence: RecordedEvidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"format": _FORMAT, "evidence": evidence}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_evidence(path: Path) -> RecordedEvidence:
    with path.open("rb") as handle:
        stored = pickle.load(handle)
    if stored.get("format") not in (1, _FORMAT):
        raise ValueError(f"{path} is not live evidence of format {_FORMAT}")
    evidence: RecordedEvidence = stored["evidence"]
    if stored["format"] == 1:           # recorded with nothing dropped: every frame was seen
        evidence.source_index = list(range(len(evidence.detections)))
    return evidence


class EvidenceFeed:
    """A ``FixedLagSelector`` on its own copy of the evidence, fed one committed frame at a time with
    the ghosts pass 1 found as it committed it: what ``replay`` and the selector process
    (``selector_process``) both run. With `keep` False the masks no later run reads are let go."""

    def __init__(self, fps: float, width: int, height: int, settings: LagSettings, keep: bool = True) -> None:
        self.result = Pass1Result()
        self.selector = FixedLagSelector(self.result, fps, width, height, settings,
                                         before_run=GhostEraser(self.result))
        self.keep = keep
        self._released = 0

    def add(self, evidence: FrameEvidence, ghosts: Sequence[GhostEntry] = ()) -> List[int]:
        """Frame ``len(detections)`` and the ghosts found as it was committed; the frames committed."""
        result = self.result
        detections, boost, raw, players, court = evidence
        result.boost_masks.append(boost)
        result.raw_motions.append(raw)
        result.player_boxes.append(players)
        result.court_keypoints.append(court)
        result.ghost_rois.append([])
        for _, ghost_frame, box in ghosts:
            result.ghost_rois[ghost_frame].append(box)
        result.detections.append(detections)       # last: the frame is now available
        done = self.selector.advance()
        self._release()
        return done

    def finish(self) -> List[int]:
        return self.selector.finish()

    def _release(self) -> None:
        if self.keep:
            return
        releasable = self.selector.releasable
        for frame in range(self._released, releasable):
            self.result.boost_masks[frame] = None
            self.result.raw_motions[frame] = None
        self._released = max(self._released, releasable)


def replay(evidence: RecordedEvidence, settings: LagSettings) -> FixedLagSelector:
    """A fixed-lag selector fed `evidence` frame by frame, as the live run fed it, and finished."""
    feed = EvidenceFeed(evidence.fps, evidence.width, evidence.height, settings)
    by_commit: Dict[int, List[GhostEntry]] = {}
    for entry in evidence.ghosts:
        by_commit.setdefault(entry[0], []).append(entry)
    for frame in range(len(evidence.detections)):
        feed.add(frame_evidence_of(evidence, frame), by_commit.get(frame, ()))
    feed.finish()
    return feed.selector
