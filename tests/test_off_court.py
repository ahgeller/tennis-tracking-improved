"""tools/off_court.py: the off-our-court count over a tracking JSON."""
import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "tools" / "off_court.py"
_SPEC = importlib.util.spec_from_file_location("off_court", _PATH)
off_court = importlib.util.module_from_spec(_SPEC)
sys.modules["off_court"] = off_court
_SPEC.loader.exec_module(off_court)

# The VolyAvw court's corner keypoints (far left, far right, near left, near right).
KEYPOINTS = [787.5, 199.3, 1182.0, 197.9, 201.0, 632.8, 1827.0, 631.5]


def _row(frame, x=None, y=None, source="det", track=None):
    if x is None:
        return {"frame": frame, "present": False}
    return {"frame": frame, "present": True, "x": x, "y": y, "source": source,
            "track_id": track if source == "det" else None}


def _payload(rows, keypoints=KEYPOINTS):
    return {"last_valid_court_keypoints": keypoints, "frames": rows}


def test_a_track_beside_the_next_court_is_off_and_ours_is_not():
    rows = [_row(f, 1000.0, 400.0, track=1) for f in range(5)]
    rows += [_row(5)]
    rows += [_row(f, 1600.0, 150.0, track=2) for f in range(6, 12)]
    rows += [_row(12, 1590.0, 160.0, source="interp")]
    result = off_court.report(_payload(rows))
    assert (result.present, result.off) == (12, 7)
    [group] = result.off_groups
    assert group.track_id == 2 and group.frames == list(range(6, 13))
    assert off_court.window_counts(_payload(rows), 0, 7) == (7, 2)


def test_our_wide_ball_cut_by_a_missing_frame_is_judged_with_the_rest_of_its_track():
    # PomonaPitzer 860-897: the ball flies out to the left, 4-5 m beyond the sideline, and one
    # missing frame cuts the run; on its own the wide end would read as another court's.
    rows = [_row(f, 600.0 - 40.0 * f, 340.0, track=1) for f in range(10)]
    rows += [_row(10)]
    rows += [_row(f, 30.0, 360.0 + f, track=1) for f in range(11, 20)]
    assert off_court.report(_payload(rows)).off == 0


def test_a_point_above_the_far_baseline_is_measured_on_the_far_baseline_row():
    court = off_court.Court.of(KEYPOINTS)
    assert court.lateral_m(1182.0 + 36.0, 60.0) == pytest.approx(court.lateral_m(1182.0 + 36.0, court.far_y))
    assert court.lateral_m(1000.0, 60.0) == 0.0


def test_a_clip_without_court_corners_is_unmeasured():
    result = off_court.report(_payload([_row(0, 1600.0, 150.0, track=1)], keypoints=None))
    assert not result.measured and result.off == 0
