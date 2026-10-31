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

import contextlib
import copy
import dataclasses
import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
from ball_in_play_selector import FrameResult, PackedMask, RawDetection, Track, select_ball_in_play

from ..court.court_geometry import CourtKeypoints, build_court_polygon, last_valid_court_keypoints
from ..court.court_scene import PlayerBoxes

# An observation, as two runs of the selector both see it: its frame and its point.
_ObservationKey = Tuple[int, float, float]


class Evidence(Protocol):
    """What the selector reads, one entry per frame in arrival order (``pass1.Pass1Result``):
    the lists only ever grow, and entries before ``FixedLagSelector.releasable`` may be dropped."""

    detections: List[List[RawDetection]]
    boost_masks: List[Optional[PackedMask]]
    raw_motions: List[Optional[PackedMask]]
    player_boxes: List[PlayerBoxes]
    court_keypoints: List[Optional[List[float]]]


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

    def assign(self, tracks: Sequence[Track]) -> Dict[int, int]:
        """Each track's stream id, by its run id; every observation is remembered under it."""
        ids: Dict[int, int] = {}
        for track in sorted(tracks, key=lambda item: (item.first_frame, item.track_id)):
            keys = [(obs.frame, obs.cx, obs.cy) for obs in track.observations]
            known = next((self._by_observation[key] for key in keys if key in self._by_observation), None)
            if known is None:
                known = track.track_id if track.track_id not in self._used else max(self._used) + 1
                self._used.add(known)
            ids[track.track_id] = known
            for key in keys:
                self._by_observation.setdefault(key, known)
        return ids

    def forget_before(self, frame: int) -> None:
        """Drop the observations before `frame`, which no later run reads."""
        self._by_observation = {key: value for key, value in self._by_observation.items() if key[0] >= frame}


@dataclass
class SelectorRun:
    """One run of the selector as ``FixedLagSelector.plan`` cut it: the window ``[start, end)`` it reads
    (its inputs taken then, so later changes to the evidence cannot reach it) and the frames
    ``[first, commit_end)`` it commits."""

    start: int
    end: int
    first: int
    commit_end: int
    inputs: Dict[str, Any]

    def execute(self, fps: float, width: int, height: int) -> Tuple[List[Optional[FrameResult]], List[Track]]:
        """The selector over the window: its rows and its tracks (window frames). Reads only the
        run's own inputs, so it may run on another thread."""
        rows, _, tracks = select_ball_in_play(self.inputs["detections"], fps, width, height, **{
            key: value for key, value in self.inputs.items() if key != "detections"})
        return rows, tracks


class FixedLagSelector:
    """``select_ball_in_play`` committed on a fixed lag (see the module docstring). ``advance`` is
    called whenever `evidence` has grown and returns the frames it committed; ``finish``, after the
    last frame, commits the rest. ``committed[i]`` is frame ``i``'s final row (None: no ball).
    ``before_run(start, end)`` is called as each run is cut, with the window it reads (the live
    pipeline clears the ghost ROIs pass 1 found since in it).

    ``advance`` is ``plan``, ``SelectorRun.execute`` and ``apply`` in a row; a caller may execute
    the runs elsewhere (another thread) as long as it applies them in the order they were planned.
    What a run commits depends only on the evidence when it was planned, so the rows are the same
    either way."""

    def __init__(self, evidence: Evidence, fps: float, width: int, height: int, settings: LagSettings,
                 before_run: Optional[Callable[[int, int], None]] = None) -> None:
        self.evidence = evidence
        self.before_run = before_run
        self.fps, self.width, self.height = float(fps), int(width), int(height)
        self.settings = settings
        self.committed: List[Optional[FrameResult]] = []
        self.planned = 0                # frames below this belong to a run already cut
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
        """Frames below this are read by no run still to be cut: their evidence may be released."""
        return max(0, self.planned - self.settings.history)

    def ready(self) -> int:
        """Frames past the planned ones that the lag allows committing now."""
        if self.settings.delay is None:
            return 0
        return max(0, self.available - self.settings.delay - self.planned)

    def plan(self, final: bool = False) -> Optional[SelectorRun]:
        """The next run when one is due (``every`` frames ready to commit; with `final`, the stream
        has ended and every remaining frame is), else None."""
        if final:
            if self.planned >= self.available:
                return None
            commit_end = self.available
        elif self.ready() < self.settings.every:
            return None
        else:
            commit_end = self.planned + self.ready()
        first, end = self.planned, self.available
        start = max(0, first - self.settings.history)
        if self.before_run is not None:
            self.before_run(start, end)
        ev, window = self.evidence, slice(start, end)
        inputs: Dict[str, Any] = {
            "detections": ev.detections[window],
            "court_polygon": self._court_polygon(start, end),
            "boost_masks": ev.boost_masks[window],
            "raw_motions": ev.raw_motions[window],
            "player_boxes_by_frame": ev.player_boxes[window],
            "court_keypoints_by_frame": ev.court_keypoints[window],
        }
        self.planned = commit_end
        return SelectorRun(start, end, first, commit_end, inputs)

    def apply(self, run: SelectorRun, rows: Sequence[Optional[FrameResult]], tracks: Sequence[Track]) -> List[int]:
        """Commit a run's frames from its answer (runs in the order they were planned); the frames."""
        if run.first != len(self.committed):
            raise RuntimeError(f"selector run for frames {run.first}-{run.commit_end} applied out of order "
                               f"({len(self.committed)} committed)")
        self.runs += 1
        shifted = {track.track_id: _shifted(track, run.start) for track in tracks}
        stream_ids = self.ids.assign(list(shifted.values()))
        for frame in range(run.first, run.commit_end):
            row = rows[frame - run.start]
            if row is not None and row.track_id is not None:
                stream_id = stream_ids[row.track_id]
                self.tracks[stream_id] = _renamed(shifted[row.track_id], stream_id)
                row = dataclasses.replace(row, track_id=stream_id)
            self.committed.append(row)
        self.ids.forget_before(max(0, run.commit_end - self.settings.history))
        return list(range(run.first, run.commit_end))

    def advance(self) -> List[int]:
        """Commit the frames the lag allows, once ``every`` of them are waiting; the frames committed."""
        return self._run_now(self.plan())

    def finish(self) -> List[int]:
        """Commit every remaining frame (the end of the stream: no frame is coming to wait for)."""
        return self._run_now(self.plan(final=True))

    def _run_now(self, run: Optional[SelectorRun]) -> List[int]:
        if run is None:
            return []
        return self.apply(run, *run.execute(self.fps, self.width, self.height))

    def _court_polygon(self, start: int, end: int) -> Optional[np.ndarray]:
        """The court polygon of the window's last valid keypoints, rebuilt only when they change
        (quietly: ``build_court_polygon`` prints what it built, which offline reads once per clip
        and a live stream would read at every court refresh)."""
        keypoints: Optional[CourtKeypoints] = last_valid_court_keypoints(self.evidence.court_keypoints[start:end])
        key = None if keypoints is None else tuple(float(value) for value in keypoints)
        if key != self._court[0]:
            polygon = None
            if keypoints is not None:
                with contextlib.redirect_stdout(io.StringIO()):
                    polygon, _ = build_court_polygon([keypoints])
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
