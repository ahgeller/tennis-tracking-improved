"""Count a tracking JSON's output frames that belong to a neighbouring court.

    python tools/off_court.py output/run/VolyAvw_tracking.json [more.json ...]
    python tools/off_court.py run.json --window 2772-2865 --window 1107-1137

**Definition.** Every present output frame is given the selector track that supports it: a
detector row carries its track's id, and a fitted or motion row belongs to the detector row before
it in its run of consecutive present frames (or after it, when the run starts without one). The
output frames of one track form a group, wherever they are in the clip; a run with no detector row
is a group of its own. A group is *off our court* when every one of its points lies more than
``MARGIN_M`` (3 m) beyond our court's doubles sidelines, sideways on the ground; its frames are
the off-court frames.

**The lateral distance** of an image point ``(x, y)`` is measured on the image row of the point,
or on the far baseline's row when the point is above it: the two doubles sidelines (through the
far and near corner keypoints) give the court's left and right edge on that row, the court's
width there is 10.97 m, and the distance is how far ``x`` lies outside the edges in those metres
(0 between them). The court is the frame's own keypoints, else the last valid ones in the file.

**Why this shape.** A ball legitimately leaves the court lines: a lob, a wide ball, a player
behind the baseline. Seen from behind a baseline, a ball in the air stands straight above its
ground point in the image, so an airborne ball over our court can appear anywhere above the
court in the picture, and one image position alone cannot say which court a ball is on. Our
rally, though, comes down to our court at a bounce or a racket: some of a track's points are low
and close to our lines. A ball on a neighbouring court stays beside that court
the whole time, and a neighbouring court's nearest doubles sideline is at least 3.66 m from ours
(the ITF's minimum side run-off between courts), so a track whose output never comes within 3 m
of our sidelines is on, or around, another court. A track's output is judged whole: a run cut
by one missing frame can leave the wide end of our own rally on its own (PomonaPitzer frames
871-897, a labelled ball flying out to the left, 4-5 m beyond the sideline). Measuring a point above the far baseline on the far baseline's
row understates its distance (it is no nearer than that row), so the rule errs towards calling a
track ours.

**What it does not count.** A dead ball inside our own surroundings (one against our back fence,
a ball a server bounces behind the baseline) is not on another court; ``--window`` reports how
many frames any stretch outputs at all, for such cases.

It assumes the usual broadcast or club view from behind a baseline, where the baselines run
roughly along the image rows; a clip without court keypoints is reported as unmeasured.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MARGIN_M = 3.0
DOUBLES_WIDTH_M = 10.97
# Corner keypoints in the court detector's order: far left, far right, near left, near right.
_FAR_LEFT, _FAR_RIGHT, _NEAR_LEFT, _NEAR_RIGHT = 0, 1, 2, 3
_MIN_KEYPOINT_VALUES = 8

Point = Tuple[float, float]


@dataclass(frozen=True)
class Court:
    """Our court's four doubles corners in image pixels."""
    far_left: Point
    far_right: Point
    near_left: Point
    near_right: Point

    @classmethod
    def of(cls, keypoints: Optional[Sequence[float]]) -> Optional["Court"]:
        """The court from a flat keypoint list, or None when a corner is missing."""
        if not keypoints or len(keypoints) < _MIN_KEYPOINT_VALUES:
            return None
        corners = []
        for index in (_FAR_LEFT, _FAR_RIGHT, _NEAR_LEFT, _NEAR_RIGHT):
            x, y = float(keypoints[2 * index]), float(keypoints[2 * index + 1])
            if x <= 0.0 and y <= 0.0:
                return None
            corners.append((x, y))
        court = cls(*corners)
        if court.near_y - court.far_y < 1.0:
            return None
        return court

    @property
    def far_y(self) -> float:
        return 0.5 * (self.far_left[1] + self.far_right[1])

    @property
    def near_y(self) -> float:
        return 0.5 * (self.near_left[1] + self.near_right[1])

    def lateral_m(self, x: float, y: float) -> float:
        """Metres beyond the nearer doubles sideline, sideways on the ground (0 between them),
        on the point's own image row or the far baseline's row when the point is above it."""
        row = max(y, self.far_y)
        left = _x_at(self.far_left, self.near_left, row)
        right = _x_at(self.far_right, self.near_right, row)
        width = right - left
        if width <= 1.0:
            return 0.0
        outside = max(left - x, x - right, 0.0)
        return outside * DOUBLES_WIDTH_M / width


def _x_at(top: Point, bottom: Point, y: float) -> float:
    """The x of the line through two points at image row y."""
    (x1, y1), (x2, y2) = top, bottom
    if abs(y2 - y1) < 1e-6:
        return 0.5 * (x1 + x2)
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


@dataclass
class Group:
    """The output frames one track supports (or one run with no detector row)."""
    track_id: Optional[int]
    frames: List[int]
    min_lateral_m: float

    @property
    def off_court(self) -> bool:
        return self.min_lateral_m > MARGIN_M


def runs(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """The runs of consecutive present output frames."""
    found: List[List[Dict[str, Any]]] = []
    previous: Optional[int] = None
    for row in rows:
        frame = int(row["frame"])
        if row.get("present"):
            if not found or previous is None or frame != previous + 1:
                found.append([])
            found[-1].append(row)
            previous = frame
        else:
            previous = None
    return found


def groups(payload: Dict[str, Any]) -> List[Group]:
    """The output's groups, each with the smallest lateral distance of its points."""
    fallback = Court.of(payload.get("last_valid_court_keypoints"))
    by_owner: Dict[Any, List[Dict[str, Any]]] = {}
    for index, run in enumerate(runs(payload["frames"])):
        for row, owner in zip(run, _owners(run)):
            by_owner.setdefault(("track", owner) if owner is not None else ("run", index), []).append(row)
    out = []
    for key, rows in by_owner.items():
        distances = [
            court.lateral_m(float(row["x"]), float(row["y"]))
            for row in rows
            if (court := Court.of(row.get("court_keypoints")) or fallback) is not None
        ]
        out.append(Group(key[1] if key[0] == "track" else None, sorted(int(row["frame"]) for row in rows),
                         min(distances) if distances else 0.0))
    return sorted(out, key=lambda group: group.frames[0])


def _owners(run: Sequence[Dict[str, Any]]) -> List[Optional[int]]:
    """The supporting track of every row of a run: a detector row's own, else the last detector
    row's before it, else the first one's after it (None in a run with no detector row)."""
    ids: List[Optional[int]] = [
        row.get("track_id") if row.get("source") == "det" else None for row in run
    ]
    last: Optional[int] = None
    for index, value in enumerate(ids):
        if value is not None:
            last = value
        else:
            ids[index] = last
    following = next((value for value in ids if value is not None), None)
    return [following if value is None else value for value in ids]


@dataclass
class Report:
    present: int
    off: int
    off_groups: List[Group]
    measured: bool


def report(payload: Dict[str, Any]) -> Report:
    found = groups(payload)
    measured = Court.of(payload.get("last_valid_court_keypoints")) is not None or any(
        Court.of(row.get("court_keypoints")) is not None for row in payload["frames"])
    off = [group for group in found if group.off_court]
    return Report(sum(len(group.frames) for group in found),
                  sum(len(group.frames) for group in off), off, measured)


def window_counts(payload: Dict[str, Any], lo: int, hi: int) -> Tuple[int, int]:
    """(present, off-court) output frames within frames lo..hi inclusive."""
    off_frames = {frame for group in report(payload).off_groups for frame in group.frames}
    present = [int(row["frame"]) for row in payload["frames"]
               if row.get("present") and lo <= int(row["frame"]) <= hi]
    return len(present), sum(frame in off_frames for frame in present)


def _window(text: str) -> Tuple[int, int]:
    lo, _, hi = text.partition("-")
    return int(lo), int(hi or lo)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json", nargs="+", type=Path, help="Tracking JSONs (clean_tracker.py --tracking-json)")
    parser.add_argument("--window", action="append", type=_window, default=[], metavar="LO-HI",
                        help="Also count the present and off-court output frames in this frame range")
    args = parser.parse_args(argv)
    for path in args.json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = report(payload)
        if not result.measured:
            print(f"{path.name}: no court keypoints, unmeasured")
            continue
        listed = ", ".join(f"{g.frames[0]}-{g.frames[-1]} (track {g.track_id}, {len(g.frames)} frames, "
                           f"{g.min_lateral_m:.1f} m)" for g in result.off_groups)
        print(f"{path.name}: present {result.present}, off our court {result.off}"
              + (f" [{listed}]" if listed else ""))
        for lo, hi in args.window:
            present, off = window_counts(payload, lo, hi)
            print(f"    frames {lo}-{hi}: present {present}, off our court {off}, on our court {present - off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
