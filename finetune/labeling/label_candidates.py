"""The tracker's own events, offered in the label tool as candidates: landings, hits and rally
starts and ends a pipeline run found (``finetune.scoring.tracker_events``), each waiting for the
person labelling to confirm it - by marking the event with its key, which writes a label - or to
reject it (``;``). A candidate is only ever a suggestion: nothing about it reaches a label file
until it is confirmed, and the scorers never read it.

A candidate is *answered* once a label of its kind lies within its kind's reach of it
(``ANSWER_S``: the scorer's tolerance for landings and hits; for a rally start, which the tracker
puts at the serve's hit and the label at the toss before it, a few seconds) or it has been
rejected. The rejections are kept in ``labels/<clip>_events.review.json``
(``label_store.save_event_review``), so a rejected candidate does not come round again; a
rejection within ``REJECT_REACH`` frames of a candidate covers it, so rerunning the tracker
(which moves an event by a frame or so) does not bring every rejection back.

``'`` walks the candidates nobody has answered yet (``LabelSession.next_candidate``); each label
row wears the candidates nearest it (``ClipCandidates.on_row``).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, get_args

from finetune.data.event_labels import Hitter
from finetune.scoring.court_events_score import TOLERANCE_S
from finetune.scoring.tracker_events import (RUN_MODEL, court_block, read_run, run_fps, run_json, tracker_hits,
                                             tracker_landings, tracker_rallies)
from tennis_tracker.court.court_scene import seconds_to_frames

CandidateKind = Literal["landing", "hit", "rally_start", "rally_end"]
CANDIDATE_KINDS: Tuple[CandidateKind, ...] = get_args(CandidateKind)
ANSWER_S: Dict[CandidateKind, float] = {"landing": TOLERANCE_S, "hit": TOLERANCE_S,
                                         "rally_start": 3.0, "rally_end": 3.0}
REJECT_REACH = 2              # frames: a rejection this close to a candidate of its kind covers it
CANDIDATE_TEXT: Dict[CandidateKind, str] = {"landing": "LANDING?", "hit": "HIT?", "rally_start": "RALLY START?",
                                            "rally_end": "RALLY END?"}
CANDIDATE_KEYS: Dict[CandidateKind, str] = {"landing": "/", "hit": ".", "rally_start": "[", "rally_end": "]"}   # yes


class Candidate(NamedTuple):
    """One event the tracker found: its kind, source frame, a note on how (``"model 0.91"``,
    ``"kink"``...), where in the picture the tracker put it (a landing's point, the ball at a hit;
    None for a rally mark or a hit judged without a ball) and, for a hit, the side of the court its
    hitter stands on."""

    kind: CandidateKind
    frame: int
    note: str
    hitter: Optional[Hitter] = None
    point: Optional[Tuple[float, float]] = None


@dataclass
class ClipCandidates:
    """A clip's candidates from one run, in frame order, and the label row each one sits on."""

    run: Path                                       # the tracking JSON they came from
    fps: float
    events: List[Candidate]
    row_of: List[int] = field(default_factory=list)  # per event: the clip-local row index nearest it

    def place(self, row_frames: Sequence[int]) -> None:
        """Put each candidate on the label row nearest its frame (the earlier one on a tie)."""
        self.row_of = [nearest_row(row_frames, event.frame) for event in self.events]

    def on_row(self, local: int) -> List[Candidate]:
        return [event for event, row in zip(self.events, self.row_of) if row == local]


def nearest_row(row_frames: Sequence[int], frame: float) -> int:
    """The index of the frame in sorted `row_frames` nearest `frame` (the earlier one on a tie)."""
    at = bisect.bisect_left(row_frames, frame)
    if at <= 0:
        return 0
    if at >= len(row_frames):
        return len(row_frames) - 1
    return at - 1 if frame - row_frames[at - 1] <= row_frames[at] - frame else at


def load_candidates(run_dir: Path, clip: str, model: str = RUN_MODEL) -> Optional[ClipCandidates]:
    """The clip's candidates from the run folder `run_dir` (``<clip>_<model>.json``, written with
    ``--court``); None, with a note, when the run has no JSON for it or no court events."""
    path = run_json(run_dir, clip, model)
    try:
        payload = read_run(path)
        block = court_block(payload, path)
    except (FileNotFoundError, ValueError) as error:
        print(f"[cand] {clip}: no candidates ({error})")
        return None
    except OSError as error:
        print(f"[cand] {clip}: {path.name} could not be read ({error}); no candidates")
        return None
    fps = run_fps(payload)
    events: List[Candidate] = []
    for landing in tracker_landings(block, fps):
        note = f"model {landing.score:.2f}" if landing.origin == "model" else landing.origin
        events.append(Candidate("landing", int(landing.frame), note, point=(float(landing.x), float(landing.y))))
    for hit in tracker_hits(payload, block):
        events.append(Candidate("hit", hit.frame, hit.source, hit.hitter, hit.ball))
    for start, end in tracker_rallies(block, fps):
        events.append(Candidate("rally_start", start, "first leg"))
        events.append(Candidate("rally_end", end, "last leg"))
    events.sort(key=lambda event: (event.frame, CANDIDATE_KINDS.index(event.kind)))
    return ClipCandidates(path, fps, events)


# --------------------------------------------------------------------------- answered or not

def labelled_frames(kind: CandidateKind, landings: Iterable[float], hits: Iterable[float],
                    rally_starts: Iterable[int], rally_ends: Iterable[int]) -> List[float]:
    """The frames of a clip's labels of `kind`."""
    return sorted({"landing": landings, "hit": hits, "rally_start": rally_starts, "rally_end": rally_ends}[kind])


def is_answered(candidate: Candidate, fps: float, labels: Mapping[CandidateKind, Sequence[float]],
                rejected: Mapping[str, Set[int]]) -> bool:
    """Whether a label of the candidate's kind lies within its reach, or it was rejected."""
    if any(abs(frame - candidate.frame) <= REJECT_REACH for frame in rejected.get(candidate.kind, ())):
        return True
    reach = seconds_to_frames(ANSWER_S[candidate.kind], fps)
    return any(abs(frame - candidate.frame) <= reach for frame in labels.get(candidate.kind, ()))
