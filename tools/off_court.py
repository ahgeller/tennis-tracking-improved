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

**The lateral distance** of an image point is ``court_geometry.sideline_distance_m``, the one the
selector's neighbouring-court veto uses: how far the point lies beyond our doubles sidelines, in
metres of the court's 10.97 m width on the point's image row, or on a sideline's far-corner row
when the point is above it. The court is the four doubles corners, keypoints 0-3, of the frame's
own keypoints, else of the file's last valid ones; a frame with neither is not measured.

**Why this shape.** A ball legitimately leaves the court lines: a lob, a wide ball, a player
behind the baseline. Seen from behind a baseline, a ball in the air stands straight above its
ground point in the image, so an airborne ball over our court can appear anywhere above the
court in the picture, and one image position alone cannot say which court a ball is on. Our
rally, though, comes down to our court at a bounce or a racket: some of a track's points are low
and close to our lines. A ball on a neighbouring court stays beside that court the whole time,
and a neighbouring court's nearest doubles sideline is at least 3.66 m from ours (the ITF's
minimum side run-off between courts), so a track whose output never comes within 3 m of our
sidelines is on, or around, another court. A track's output is judged whole: a run cut
by one missing frame can leave the wide end of our own rally on its own (PomonaPitzer frames
871-897, a labelled ball flying out to the left, 4-5 m beyond the sideline). A point in the
air reads farther out than its ground point, so a track is judged by its smallest distance,
which its bounces and contacts give.

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

from tennis_tracker.court.court_geometry import (  # noqa: E402
    Corners, doubles_corners, sideline_distance_m)

MARGIN_M = 3.0
Row = Dict[str, Any]


@dataclass
class Group:
    """The output frames one track supports (or one run with no detector row)."""
    track_id: Optional[int]
    frames: List[int]
    min_lateral_m: float

    @property
    def off_court(self) -> bool:
        return self.min_lateral_m > MARGIN_M


def runs(rows: Sequence[Row]) -> List[List[Row]]:
    """The runs of consecutive present output frames."""
    found: List[List[Row]] = []
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
    fallback = doubles_corners(payload.get("last_valid_court_keypoints"))
    by_owner: Dict[Any, List[Row]] = {}
    for index, run in enumerate(runs(payload["frames"])):
        for row, owner in zip(run, _owners(run)):
            key = ("track", owner) if owner is not None else ("run", index)
            by_owner.setdefault(key, []).append(row)
    out = []
    for key, rows in by_owner.items():
        distances = [sideline_distance_m(corners, float(row["x"]), float(row["y"]))
                     for row in rows if (corners := _corners(row, fallback)) is not None]
        frames = sorted(int(row["frame"]) for row in rows)
        out.append(Group(key[1] if key[0] == "track" else None, frames, min(distances, default=0.0)))
    return sorted(out, key=lambda group: group.frames[0])


def _corners(row: Row, fallback: Optional[Corners]) -> Optional[Corners]:
    """A frame's own doubles corners, else the file's last valid ones."""
    own = doubles_corners(row.get("court_keypoints"))
    return own if own is not None else fallback


def _owners(run: Sequence[Row]) -> List[Optional[int]]:
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
    measured = doubles_corners(payload.get("last_valid_court_keypoints")) is not None or any(
        doubles_corners(row.get("court_keypoints")) is not None for row in payload["frames"])
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json", nargs="+", type=Path,
                        help="Tracking JSONs (clean_tracker.py --tracking-json)")
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
            print(f"    frames {lo}-{hi}: present {present}, off our court {off}, "
                  f"on our court {present - off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
