"""finetune.scoring.tracker_events and finetune.labeling.label_candidates: a pipeline run's JSON read
as events, and those events offered as candidates and answered. The JSONs are small ones written
under tmp_path."""
import json

import pytest

from finetune.labeling.label_candidates import (ClipCandidates, Candidate, is_answered, load_candidates,
                                                nearest_row)
from finetune.scoring.tracker_events import (court_block, court_keypoints_by_frame, court_side, player_boxes_by_frame,
                                             rallies_from_legs, read_run, tracker_hits)

# A court seen square on: the far baseline at y=100, the near one at y=500, the net at y=300.
COURT = [100, 100, 500, 100, 100, 500, 500, 500,
         150, 100, 150, 500, 450, 100, 450, 500,
         150, 200, 450, 200, 150, 400, 450, 400,
         300, 200, 300, 400]


def payload(**block):
    frames = [{"frame": f, "court_keypoints": COURT if f != 3 else None,
               "player_boxes": {"7": [280, 420, 320, 480]} if f < 5 else {"7": [280, 420, 320, 480], "9": [1, 2, 3, 4]}}
              for f in range(10)]
    events = {"bounces": [], "bounce_candidates": [], "kinematic_bounces": [], "contacts": [], "rally_legs": []}
    events.update(block)
    return {"video": {"fps": 30.0, "width": 1920, "height": 1080}, "frames": frames, "court_events": events}


def write(tmp_path, clip, data):
    path = tmp_path / f"{clip}_gridtracknet.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_missing_run_or_one_without_court_events_is_an_error_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="clip_gridtracknet.json"):
        read_run(tmp_path / "clip_gridtracknet.json")
    path = write(tmp_path, "clip", {"video": {"fps": 30.0}, "frames": []})
    with pytest.raises(ValueError, match="without --court"):
        court_block(read_run(path), path)


def test_a_ground_point_is_on_the_near_or_far_half_of_the_court():
    assert court_side(COURT, 300, 450) == "near"
    assert court_side(COURT, 300, 150) == "far"
    assert court_side(None, 300, 150) is None


def test_hits_carry_the_hitters_side_and_the_ball_where_it_was_judged():
    data = payload(contacts=[
        {"frame": 6, "x": 300, "y": 150, "player_id": 9, "score": 2.0, "source": "gap"},
        {"frame": 2, "x": 300, "y": 470, "player_id": 7, "score": 9.0, "source": "kink",
         "ball": {"x": 310.0, "y": 420.0, "position": "measurement"}},
        {"frame": 3, "x": 300, "y": 470, "player_id": 7, "score": 1.0, "source": "kink"},   # no court on frame 3
    ])
    hits = tracker_hits(data, data["court_events"])
    assert [(h.frame, h.hitter, h.ball) for h in hits] == [(2, "near", (310.0, 420.0)), (3, None, None), (6, "far", None)]


def test_rally_legs_group_into_rallies_at_a_serve_or_a_long_gap():
    legs = [{"start_frame": 10, "end_frame": 40, "phase": "serve"},
            {"start_frame": 45, "end_frame": 80, "phase": "rally"},
            {"start_frame": 90, "end_frame": 120, "phase": "rally"},
            {"start_frame": 300, "end_frame": 330, "phase": "rally"},     # 6 s after: a new point
            {"start_frame": 335, "end_frame": 350, "phase": "serve"}]     # a serve: a new point at once
    assert rallies_from_legs(legs, 30.0) == [(10, 120), (300, 330), (335, 350)]


def test_player_boxes_and_court_keypoints_by_frame():
    data = payload()
    boxes = player_boxes_by_frame(data)
    assert boxes[0] == {"7": [280.0, 420.0, 320.0, 480.0]} and set(boxes[6]) == {"7", "9"}
    court = court_keypoints_by_frame(data)
    assert 3 not in court and court[0][:2] == [100.0, 100.0]


# --------------------------------------------------------------------------- candidates

def test_candidates_come_in_frame_order_with_their_notes(tmp_path):
    write(tmp_path, "clip", payload(
        bounces=[{"frame": 50, "x": 1.0, "y": 2.0, "score": 0.9, "origin": "model", "position": "measurement"}],
        kinematic_bounces=[{"frame": 80, "x": 3.0, "y": 4.0, "score": 0.0, "origin": "kink", "position": "measurement"}],
        contacts=[{"frame": 6, "x": 300, "y": 150, "player_id": 9, "score": 2.0, "source": "gap"}],
        rally_legs=[{"start_frame": 4, "end_frame": 90, "phase": "serve"}]))
    found = load_candidates(tmp_path, "clip")
    assert found is not None and found.fps == 30.0
    assert [(c.kind, c.frame, c.note, c.hitter, c.point) for c in found.events] == [
        ("rally_start", 4, "first leg", None, None), ("hit", 6, "gap", "far", None),
        ("landing", 50, "model 0.90", None, (1.0, 2.0)), ("landing", 80, "kink", None, (3.0, 4.0)),
        ("rally_end", 90, "last leg", None, None)]


def test_no_run_for_a_clip_means_no_candidates_not_an_error(tmp_path, capsys):
    assert load_candidates(tmp_path, "clip") is None
    write(tmp_path, "bare", {"video": {"fps": 30.0}, "frames": []})
    assert load_candidates(tmp_path, "bare") is None
    assert capsys.readouterr().out.count("[cand]") == 2


def test_each_candidate_sits_on_the_label_row_nearest_it():
    assert [nearest_row([0, 2, 4, 6], frame) for frame in (-5, 0, 1, 3, 5, 6, 99)] == [0, 0, 0, 1, 2, 3, 3]
    clip = ClipCandidates(None, 30.0, [Candidate("hit", 3, ""), Candidate("landing", 7, "")])
    clip.place([0, 2, 4, 6, 8])
    assert clip.row_of == [1, 3] and clip.on_row(3) == [clip.events[1]]


def test_a_candidate_is_answered_by_a_label_of_its_kind_in_reach_or_a_rejection_near_it():
    hit = Candidate("hit", 100, "kink")
    labels = {"hit": [102.5], "landing": [100]}
    assert is_answered(hit, 30.0, labels, {})                            # 2.5 frames: within 0.1 s (3 frames) at 30 FPS
    assert not is_answered(hit, 30.0, {"hit": [104.0], "landing": [100]}, {})   # a landing does not answer a hit
    assert is_answered(hit, 30.0, {}, {"hit": {102}})                    # a rejection within REJECT_REACH frames
    assert not is_answered(hit, 30.0, {}, {"hit": {103}, "landing": {100}})
    start = Candidate("rally_start", 100, "first leg")
    assert is_answered(start, 30.0, {"rally_start": [20]}, {})           # the toss, seconds before the serve's hit
