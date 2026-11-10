"""The ball-in-play selector on a fixed lag: each frame's output is committed a set number of
frames after it arrived, from the frames seen so far, and never changes afterwards.

``select_ball_in_play`` judges a track and fills its gaps from the whole clip. Live, the clip is
the frames seen so far, so ``FixedLagSelector`` re-runs that same selector over a sliding window of
them (no other selector code) and commits only the window's frames that are at least ``delay``
frames old:

- the window ends at the newest frame and starts ``history`` frames before the oldest frame not
  yet committed, so a frame is judged with ``history`` frames of context behind it and at least
  ``delay`` ahead;
- a re-run happens once ``every`` frames are ready to commit (``finish`` commits what is left), so a
  frame waits ``delay`` to ``delay + every - 1`` frames;
- a ``delay`` of None commits nothing until ``finish``, which then selects over the whole clip: the
  offline answer, exactly.

Track ids are per run of the selector. A track that shares an observation with one committed
before keeps that track's id (``TrackIds``), so a committed row's ``track_id`` names one track
across the whole stream, as the offline output's does.

The selector normalises some scores by the clip's length (its timeline's coverage and gap terms,
and a track's span): on a window those are the window's length. This and the lookahead are what
the lag costs; ``audit/records/opt_live.md`` measures both.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
from ball_in_play_selector import FrameResult, PackedMask, RawDetection, Track, select_ball_in_play

from ..court.court_geometry import CourtKeypoints, build_court_polygon, last_valid_court_keypoints
from ..court.court_scene import PlayerBoxes

# An observation, as two runs of the selector both see it: its frame and its point.
_ObservationKey = Tuple[int, float, float]

# The live tracker's lag, in seconds (``audit/records/opt_live.md`` has the evidence): every entry
# point (``LiveOptions``, ``clean_tracker.py``, ``evaluate_archive.py``, ``tools/live_lag_sweep.py``)
# takes its defaults from here.
DEFAULT_DELAY_S: Optional[float] = 0.5
DEFAULT_HISTORY_S = 1.0
DEFAULT_EVERY_S = 0.2
WHOLE_CLIP = "clip"         # a delay of the whole clip: the offline answer


def parse_delay(value: str) -> Optional[float]:
    """A delay from the command line: seconds, or ``WHOLE_CLIP`` (None)."""
    return None if value == WHOLE_CLIP else float(value)


class Evidence(Protocol):
    """What the selector reads, one entry per frame in arrival order (``pass1.Pass1Result``):
    the lists only ever grow, and entries before ``FixedLagSelector.releasable`` may be dropped."""

    detections: List[List[RawDetection]]
    raw_motions: List[Optional[PackedMask]]
    player_boxes: List[PlayerBoxes]
    court_keypoints: List[Optional[List[float]]]

    # Read only here: pass 1's masks filter on first read (``motion.BoostMasks``), a record's are a list.
    @property
    def boost_masks(self) -> Sequence[Optional[PackedMask]]: ...


@dataclass(frozen=True)
class LagSettings:
    """How far behind the newest frame output is committed (``delay``, None for the whole clip), how
    many committed frames each run keeps as context (``history``) and how many frames wait for a
    run (``every``), all in frames."""

    delay: Optional[int]
    history: int
    every: int = 1

    def __post_init__(self) -> None:
        if self.delay is not None and self.delay < 0:
            raise ValueError(f"delay must be at least 0 frames, not {self.delay}")
        if self.history < 0 or self.every < 1:
            raise ValueError(f"history must be at least 0 and every at least 1, not {self.history} / {self.every}")

    @classmethod
    def at_fps(cls, fps: float, delay_s: Optional[float], history_s: float, every_s: float) -> "LagSettings":
        """The settings for a clip at `fps` from seconds; ``every`` is at least one frame."""
        frames = float(fps)
        delay = None if delay_s is None else int(round(delay_s * frames))
        return cls(delay=delay, history=int(round(history_s * frames)), every=max(1, int(round(every_s * frames))))


class TrackIds:
    """Stream-wide track ids for the tracks of successive selector runs. A run's track that shares an
    observation with a track seen before takes the id of the earliest such observation's track; any
    other keeps its run's id when no stream track has it yet, else takes the next unused id (so a
    single run over the whole clip keeps every id the selector gave)."""

    def __init__(self) -> None:
        self._by_observation: Dict[_ObservationKey, int] = {}
        self._used: Set[int] = set()
        self._next = 0                  # above every id used

    def assign(self, tracks: Sequence[Track]) -> Dict[int, int]:
        """Each track's stream id, by its run id; every observation is remembered under it."""
        ids: Dict[int, int] = {}
        for track in sorted(tracks, key=lambda item: (item.first_frame, item.track_id)):
            keys = [(obs.frame, obs.cx, obs.cy) for obs in track.observations]
            known = next((self._by_observation[key] for key in keys if key in self._by_observation), None)
            if known is None:
                known = track.track_id if track.track_id not in self._used else self._next
                self._used.add(known)
                self._next = max(self._next, known + 1)
            ids[track.track_id] = known
            for key in keys:
                self._by_observation.setdefault(key, known)
        return ids

    def forget_before(self, frame: int) -> None:
        """Drop the observations before `frame`, which no later run reads."""
        self._by_observation = {key: value for key, value in self._by_observation.items() if key[0] >= frame}


class FixedLagSelector:
    """``select_ball_in_play`` committed on a fixed lag (see the module docstring). ``advance`` is
    called whenever `evidence` has grown and returns the frames it committed; ``finish``, after the
    last frame, commits the rest. ``committed[i]`` is frame ``i``'s final row (None: no ball).
    ``before_run(start, end)`` is called as each run is cut, with the window it reads (the live
    pipeline clears the ghost ROIs pass 1 found since in it)."""

    def __init__(self, evidence: Evidence, fps: float, width: int, height: int, settings: LagSettings,
                 before_run: Optional[Callable[[int, int], None]] = None) -> None:
        self.evidence = evidence
        self.before_run = before_run
        self.fps, self.width, self.height = float(fps), int(width), int(height)
        self.settings = settings
        self.committed: List[Optional[FrameResult]] = []
        self.planned = 0                # frames below this belong to a run already made
        self.runs = 0
        self.ids = TrackIds()
        # Each stream track as the latest run that committed one of its rows judged it.
        self.tracks: Dict[int, Track] = {}
        self._court: Tuple[Optional[Tuple[float, ...]], Optional[np.ndarray]] = (None, None)

    @property
    def available(self) -> int:
        return len(self.evidence.detections)

    @property
    def releasable(self) -> int:
        """Frames below this are read by no run still to come: their evidence may be released."""
        return max(0, self.planned - self.settings.history)

    def ready(self) -> int:
        """Frames past the planned ones that the lag allows committing now."""
        if self.settings.delay is None:
            return 0
        return max(0, self.available - self.settings.delay - self.planned)

    def advance(self) -> List[int]:
        """Commit the frames the lag allows, once ``every`` of them are waiting; the frames committed."""
        if self.ready() < self.settings.every:
            return []
        return self._run(self.planned + self.ready())

    def finish(self) -> List[int]:
        """Commit every remaining frame (the end of the stream: no frame is coming to wait for)."""
        if self.planned >= self.available:
            return []
        return self._run(self.available)

    def _run(self, commit_end: int) -> List[int]:
        """Run the selector over the window and commit frames ``[planned, commit_end)``."""
        first, end = self.planned, self.available
        start = max(0, first - self.settings.history)
        if self.before_run is not None:
            self.before_run(start, end)
        ev, window = self.evidence, slice(start, end)
        rows, _, tracks = select_ball_in_play(
            ev.detections[window], self.fps, self.width, self.height,
            court_polygon=self._court_polygon(start, end), boost_masks=ev.boost_masks[window],
            raw_motions=ev.raw_motions[window],
            player_boxes_by_frame=ev.player_boxes[window],     # type: ignore[arg-type]  # read as a Mapping
            court_keypoints_by_frame=ev.court_keypoints[window])
        self.planned = commit_end
        self.runs += 1
        shifted = {track.track_id: _shifted(track, start) for track in tracks}
        stream_ids = self.ids.assign(list(shifted.values()))
        for frame in range(first, commit_end):
            row = rows[frame - start]
            if row is not None and row.track_id is not None:
                stream_id = stream_ids[row.track_id]
                self.tracks[stream_id] = _renamed(shifted[row.track_id], stream_id)
                row = dataclasses.replace(row, track_id=stream_id)
            self.committed.append(row)
        self.ids.forget_before(max(0, commit_end - self.settings.history))
        return list(range(first, commit_end))

    def _court_polygon(self, start: int, end: int) -> Optional[np.ndarray]:
        """The court polygon of the window's last valid keypoints, rebuilt (quietly) only when they change."""
        keypoints: Optional[CourtKeypoints] = last_valid_court_keypoints(self.evidence.court_keypoints[start:end])
        key = None if keypoints is None else tuple(float(value) for value in keypoints)
        if key != self._court[0]:
            polygon = None
            if keypoints is not None:
                polygon, _ = build_court_polygon([keypoints], quiet=True)
            self._court = (key, polygon)
        return self._court[1]


def _shifted(track: Track, offset: int) -> Track:
    """A copy of a window's track with its observations at their stream frames."""
    if offset == 0:
        return track
    moved = copy.copy(track)
    moved.observations = [dataclasses.replace(obs, frame=obs.frame + offset) for obs in track.observations]
    return moved


def _renamed(track: Track, track_id: int) -> Track:
    renamed = copy.copy(track)
    renamed.track_id = track_id
    return renamed
