"""The label tool's court-event keys, headless: hits (.) on the landing's model, their hitter (,),
rally starts and ends ([ and ]), the tracker's candidates (; to the next, a backslash to reject,
the click that confirms one between two frames), the timeline, and the files they are saved to.
Every file is under tmp_path."""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from finetune.data.event_labels import HitLabel, RallyLabel, read_hit_labels, read_rally_labels
from finetune.labeling import label_window as win
from finetune.labeling.label_candidates import Candidate, ClipCandidates
from finetune.labeling.label_draw import HIT, LANDING
from finetune.labeling.label_store import (HitMark, load_event_review, load_hits, load_rallies, save_event_review, save_hits,
                                           save_rallies)
from finetune.labeling.label_tool import Tool
from _workspace_helpers import fake_source, fake_tool


def clip(name="clip", count=10, step=1, tmp_path=None, candidates=()):
    """A 720p clip of `count` rows `step` source frames apart, its files under tmp_path, and the
    tracker's `candidates` placed on its rows."""
    source = fake_source(name, count, 1280, 720, origin=(30.25, 24.75), step=(11.0, 3.0))
    for index, row in enumerate(source.rows):
        row.frame = index * step
    if tmp_path is not None:
        for attribute, suffix in (("out_csv", "_ball.csv"), ("review", "_ball.review.json"),
                                  ("suspects", "_ball.suspect.json"), ("bounces", "_bounces.csv"),
                                  ("hits_path", "_hits.csv"), ("rallies_path", "_rallies.csv"),
                                  ("events_review", "_events.review.json"), ("players_path", "_players.csv"),
                                  ("court_path", "_court.csv")):
            setattr(source, attribute, tmp_path / f"{name}{suffix}")
    if candidates:
        source.candidates = ClipCandidates(Path("run.json"), 30.0, list(candidates))
        source.candidates.place([row.frame for row in source.rows])
    return source


def press_key(tool, character, shift=False):
    with patch.object(Tool, "view_ready", return_value=True), patch.object(win, "shift_held", return_value=shift), \
            patch.object(win, "non_character_key", return_value=False):
        tool.keys.handle(ord(character))


def click_picture(tool, fx, fy):
    vx, vy = tool.session.to_view(fx, fy)
    with patch.object(Tool, "view_ready", return_value=True):
        tool.pointer.on_mouse(cv2.EVENT_LBUTTONDOWN, vx, vy, 0, None)


# --------------------------------------------------------------------------- hits

def test_a_hit_is_marked_and_unmarked_with_the_dot_on_its_own_frame():
    tool = fake_tool(clip())
    session = tool.session
    session.cursor = 4
    press_key(tool, ".")
    assert session.source.hits == {4: HitMark()} and session.awaiting_hit is None   # the ball label is its point
    assert session.hit_for(4) == 4 and session.hit_for(5) is None
    assert "  HIT  " in tool.frame_view.hud_lines(session.rows[4], None, None)[0]
    press_key(tool, ".")
    assert session.source.hits == {}


def test_a_dot_on_two_consecutive_frames_makes_one_hit_between_them_and_the_click_is_its_point():
    tool = fake_tool(clip())
    session = tool.session
    session.cursor = 3
    press_key(tool, ".")
    tool.move(1)
    press_key(tool, ".")
    assert session.source.hits == {3.5: HitMark()} and session.awaiting_hit == 3.5
    assert session.hit_for(3) == session.hit_for(4) == 3.5
    assert "CLICK WHERE THE RACKET MET THE BALL" in tool.frame_view.hud_lines(session.rows[4], None, None)[1]
    click_picture(tool, 200.0, 150.0)
    assert session.awaiting_hit is None and session.cursor == 4          # the frame stays up
    assert session.source.hits[3.5].point == (pytest.approx(200.0, abs=1.0), pytest.approx(150.0, abs=1.0))
    assert not session.rows[4].edited                                    # the ball label was not touched
    press_key(tool, ".")                                                 # on either frame: unmarked
    assert session.source.hits == {}


def test_any_other_key_leaves_a_hit_without_a_point_and_the_landing_key_is_untouched():
    tool = fake_tool(clip())
    session = tool.session
    session.cursor = 3
    session.rows[3].visible = False
    press_key(tool, ".")
    assert session.awaiting_hit == 3
    press_key(tool, "t")
    assert session.awaiting_hit is None and session.source.hits == {3: HitMark()}
    press_key(tool, "/")                                                 # a landing on the same frame: its own mark
    assert session.source.landings == {3: None} and session.awaiting_landing == 3 and session.source.hits == {3: HitMark()}


def test_the_hitter_is_only_ever_said_by_the_comma_never_taken_from_the_tracker():
    """The hit scorer scores the tracker's hitter against the labelled one: a label pre-filled
    from the tracker would score it against itself."""
    tool = fake_tool(clip(candidates=[Candidate("hit", 4, "kink", "far")]))
    session = tool.session
    session.cursor = 4
    press_key(tool, ",")                                                 # no hit yet: nothing to name
    assert session.source.hits == {}
    press_key(tool, ".")
    assert session.source.hits == {4: HitMark(None, None)}               # the tracker's "far" is not taken
    press_key(tool, ",")
    assert session.source.hits == {4: HitMark(None, "near")}
    assert "HIT NEAR" in tool.frame_view.hud_lines(session.rows[4], None, None)[0]
    press_key(tool, ",")
    assert session.source.hits == {4: HitMark(None, "far")}
    press_key(tool, ",")
    assert session.source.hits == {4: HitMark(None, None)}               # far -> not said
    press_key(tool, ".")                                                 # unmarked: its hitter goes with it
    assert session.source.hits == {}


def test_a_hit_moved_between_two_frames_keeps_its_hitter():
    tool = fake_tool(clip())
    session = tool.session
    session.cursor = 3
    press_key(tool, ".")
    press_key(tool, ",")
    press_key(tool, ",")                                                 # far
    tool.move(1)
    press_key(tool, ".")                                                 # it was between 3 and 4
    assert session.source.hits == {3.5: HitMark(None, "far")}


def test_hits_are_saved_beside_the_landings_and_read_back(tmp_path):
    source = clip(tmp_path=tmp_path)
    tool = fake_tool(source)
    session = tool.session
    session.cursor = 2
    press_key(tool, ".")
    press_key(tool, ",")                                                 # near
    session.cursor = 6
    session.rows[6].visible = False
    press_key(tool, ".")
    click_picture(tool, 300.0, 200.0)
    tool.save(quiet=True)
    labels = read_hit_labels(source.hits_path)
    assert [(label.frame, label.hitter) for label in labels] == [(2, "near"), (6, None)]
    assert labels[0].point is None and labels[1].point == (pytest.approx(300.0, abs=1.0), pytest.approx(200.0, abs=1.0))
    assert load_hits(source.hits_path) == {2: HitMark(None, "near"), 6: HitMark(labels[1].point, None)}
    session.cursor = 2
    press_key(tool, ".")
    session.cursor = 6
    press_key(tool, ".")
    tool.save(quiet=True)
    assert not source.hits_path.exists()                                 # nothing marked: no file


# --------------------------------------------------------------------------- rallies


def test_which_rally_a_frame_is_in_follows_the_pairing_the_file_is_written_with():
    """An end with no start before it is a rally without a start (event_labels.pair_rally_marks):
    the frames before it are in no rally, the same as the file and the timeline say."""
    tool = fake_tool(clip())
    session = tool.session
    session.source.rally_starts, session.source.rally_ends = {3, 6}, {2, 8}
    assert [session.rally_at(k) for k in range(10)] == [None, None, None, (1, 2), (1, 2), (1, 2), (2, 2), (2, 2),
                                                        (2, 2), None]
    session.source.rally_ends = {2}                                      # 6 open: it reaches the clip's end
    assert session.rally_at(5) == (1, 2) and session.rally_at(9) == (2, 2)


def test_brackets_mark_a_rallys_start_and_end_and_a_frame_is_never_both():
    tool = fake_tool(clip())
    session = tool.session
    session.cursor = 2
    press_key(tool, "[")
    assert session.source.rally_starts == {2} and session.rally_mark(2) == "start"
    assert session.rally_at(5) == (1, 1) and session.rally_at(1) is None
    session.cursor = 7
    press_key(tool, "]")
    assert session.source.rally_ends == {7} and session.rally_at(7) == (1, 1) and session.rally_at(8) is None
    press_key(tool, "[")                                                 # the end becomes a start
    assert session.source.rally_ends == set() and session.source.rally_starts == {2, 7}
    assert session.rally_at(8) == (2, 2)
    press_key(tool, "[")
    assert session.source.rally_starts == {2}
    assert "rally 1/1" in tool.frame_view.hud_lines(session.rows[4], None, None)[0]


def test_rallies_are_saved_paired_and_read_back(tmp_path):
    source = clip(tmp_path=tmp_path)
    tool = fake_tool(source)
    source.rally_starts.update({1, 6})
    source.rally_ends.update({4})
    tool.session.touch("rallies", source=source)
    tool.save(quiet=True)
    assert read_rally_labels(source.rallies_path) == [RallyLabel(1, 4), RallyLabel(6, None)]
    assert load_rallies(source.rallies_path) == ({1, 6}, {4})


# --------------------------------------------------------------------------- candidates

def candidates_clip(**extra):
    return clip(count=10, step=2, candidates=[Candidate("landing", 6, "model 0.90", point=(100.0, 80.0)),
                                              Candidate("hit", 9, "kink", "near", (120.0, 60.0)),
                                              Candidate("rally_start", 14, "first leg")], **extra)


def test_semicolon_walks_the_candidates_nobody_has_answered():
    tool = fake_tool(candidates_clip())
    session = tool.session
    assert session.open_candidate_count() == 3
    press_key(tool, ";")
    assert session.rows[session.cursor].frame == 6
    assert "LANDING? /=yes" in tool.frame_view.hud_lines(session.rows[session.cursor], None, None)[1]
    press_key(tool, "/")                                                 # a landing here answers it
    assert session.open_candidate_count() == 2
    press_key(tool, ";")
    assert session.rows[session.cursor].frame in (8, 10)                 # the hit at 9, on the row nearest it
    press_key(tool, ";", shift=True)                                     # SHIFT: back the other way: none left
    assert session.rows[session.cursor].frame in (8, 10)


def test_a_candidate_between_two_frames_waits_for_the_click_that_confirms_it_there():
    tool = fake_tool(candidates_clip())
    session = tool.session
    session.cursor = 0
    press_key(tool, ";")                                                 # the landing at 6: on a label frame
    assert session.awaiting_candidate is None
    press_key(tool, ";")                                                 # the hit at 9: between frames 8 and 10
    event = session.awaiting_candidate
    assert event is not None and event.kind == "hit"
    assert "HIT? BETWEEN FRAMES" in tool.frame_view.hud_lines(session.rows[session.cursor], None, None)[1]
    click_picture(tool, 120.0, 60.0)
    assert session.source.hits[9].point == (pytest.approx(120.0, abs=1.0), pytest.approx(60.0, abs=1.0))
    assert session.source.hits[9].hitter is None and session.awaiting_candidate is None   # said by , alone
    assert not any(row.edited for row in session.rows)                   # the click placed no ball label



def test_a_backslash_never_leaves_a_between_candidate_waiting_for_a_habitual_click():
    """The backslash moves on to the next candidate, but only ; (asked for) makes a click there
    answer it: after a backslash a click is the ball label's, as everywhere else."""
    tool = fake_tool(candidates_clip())
    session = tool.session
    press_key(tool, ";")                                                 # the landing at 6
    press_key(tool, "\\")                                               # not one: on to the hit at 9
    here = session.cursor
    assert session.rows[here].frame in (8, 10) and session.awaiting_candidate is None
    ask = tool.frame_view.hud_lines(session.rows[here], None, None)[1]
    assert "CLICK WHERE" not in ask and ";=answer it here" in ask
    press_key(tool, ";")                                                 # asked: now a click answers it, here
    assert session.cursor == here and session.awaiting_candidate is not None
    picture = tool.render()[:session.view_h]
    banner = picture[:int(30 * session.ui)]
    assert (banner == HIT).all(axis=2).mean() > 0.3                      # and the picture says so, unmissably
    press_key(tool, "k")                                                 # any other key: not answered
    assert session.awaiting_candidate is None
    assert (tool.render()[:session.view_h][:int(30 * session.ui)] == HIT).all(axis=2).mean() < 0.05   # the badge alone


def test_the_slash_on_a_frame_wearing_a_between_candidate_is_the_plain_landing_key():
    tool = fake_tool(clip(count=10, step=2, candidates=[Candidate("landing", 7, "kink", point=(1.0, 2.0))]))
    session = tool.session
    session.cursor = 3                                                   # frame 6; the candidate is at 7
    press_key(tool, "/")
    assert session.source.landings == {6: None}                          # its own frame, exactly as before


def test_a_backslash_rejects_the_candidates_on_the_frame_and_pressed_again_reopens_them(tmp_path):
    source = candidates_clip(tmp_path=tmp_path)
    tool = fake_tool(source)
    session = tool.session
    press_key(tool, ";")
    landing_row = session.cursor
    press_key(tool, "\\")
    assert source.rejected == {"landing": {6}} and session.cursor != landing_row   # and on to the next one
    assert session.open_candidate_count() == 2
    tool.save(quiet=True)
    assert load_event_review(source.events_review) == {"landing": {6}}
    session.cursor = landing_row
    press_key(tool, "\\")
    assert source.rejected == {"landing": set()} and session.open_candidate_count() == 3
    tool.save(quiet=True)
    assert not source.events_review.exists()


def test_candidates_are_drawn_as_outlined_badges_and_dashed_rings_and_never_saved(tmp_path):
    source = candidates_clip(tmp_path=tmp_path)
    tool = fake_tool(source)
    session = tool.session
    press_key(tool, ";")
    picture = tool.render()[:session.view_h]
    assert (picture == LANDING).all(axis=2).any()                       # the LANDING? outline and the ring
    press_key(tool, "\\")
    assert not (tool.render()[:session.view_h] == LANDING).all(axis=2).any()   # rejected: gone from the picture
    session.touch("landings", "hits", "rallies", source=source)
    tool.save(quiet=True)
    assert not source.bounces.exists() and not source.hits_path.exists() and not source.rallies_path.exists()


# --------------------------------------------------------------------------- the timeline and the redraw

def test_the_timeline_maps_frames_to_its_strip_and_a_click_on_it_goes_there():
    tool = fake_tool(clip(count=50, step=2))
    session = tool.session
    session.fit(1280, 720)
    tool.render()
    timeline = tool.frame_view.timeline
    left, top, right, bottom = timeline.box
    assert top >= session.view_h and bottom < session.view_h + session.timeline_h
    assert timeline.frame_at(left) == 0 and timeline.frame_at(right) == 98
    x = timeline.to_x(60)
    with patch.object(Tool, "view_ready", return_value=True):
        tool.pointer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, (top + bottom) // 2, 0, None)
    assert session.rows[session.cursor].frame == 60



def test_the_timeline_is_tall_enough_to_read_with_solid_labels_hollow_candidates_and_a_caret():
    """At ui 1 the strip is at least 18 px; a label is a solid tick the strip's full height, an open
    candidate a hollow tick 3 px wide in its kind's own colour, and the frame on screen a caret
    above the strip, where nothing else is drawn."""
    from finetune.labeling.label_draw import HIT, LANDING
    from finetune.labeling.label_session import TIMELINE
    assert TIMELINE >= 18
    tool = fake_tool(clip(count=50, step=2, candidates=[Candidate("landing", 40, "model")]))
    session = tool.session
    session.fit(1280, 720)
    session.source.hits[80] = HitMark()
    session.cursor = 5
    canvas = tool.render()
    timeline = tool.frame_view.timeline
    left, top, right, bottom = timeline.box
    strip = canvas[top:bottom + 1]
    hit_x = timeline.to_x(80)
    assert (strip[:, hit_x] == HIT).all(axis=1).all()                           # solid, full height
    land_x = timeline.to_x(40)
    columns = [(strip[:, x] == LANDING).all(axis=1) for x in range(land_x - 1, land_x + 2)]
    assert columns[0].all() and columns[2].all() and not columns[1][1:-1].any()   # hollow, 3 wide, full colour
    caret = canvas[session.view_h:top, timeline.to_x(session.rows[5].frame)]
    assert (caret == (255, 255, 255)).all(axis=1).any()                          # the caret, above the strip


def test_marks_redraw_the_frame_view():
    tool = fake_tool(clip())
    session = tool.session
    session.cursor = 4
    first = tool.render()
    assert tool.render() is first                                        # nothing changed: the cached picture
    press_key(tool, "[")
    after_rally = tool.render()
    assert after_rally is not first
    press_key(tool, ".")
    assert tool.render() is not after_rally
    assert not np.array_equal(after_rally, first)


def test_the_sidecars_roundtrip_through_the_store(tmp_path):
    hits, rallies, review = tmp_path / "a_hits.csv", tmp_path / "a_rallies.csv", tmp_path / "a_events.review.json"
    save_hits(hits, {4.5: HitMark((1.0, 2.0)), 9: HitMark(None, "far")})
    assert read_hit_labels(hits) == [HitLabel(4.5, (1.0, 2.0), None), HitLabel(9, None, "far")]
    save_rallies(rallies, {3}, set())
    assert load_rallies(rallies) == ({3}, set())
    save_event_review(review, {"hit": {5, 2}, "landing": set()})
    assert load_event_review(review) == {"hit": {2, 5}}
    review.write_text(json.dumps({"rejected": {"hit": [5]}, "note": "kept"}), encoding="utf-8")
    save_hits(hits, {})                              # nothing: the header alone (label_saving removes a file
    save_rallies(rallies, set(), set())              # the tool emptied, not the store)
    save_event_review(review, {})
    assert hits.read_text(encoding="utf-8").splitlines() == ["frame,ball_x,ball_y,hitter"]
    assert rallies.read_text(encoding="utf-8").splitlines() == ["start,end"]
    assert json.loads(review.read_text(encoding="utf-8")) == {"rejected": {}, "note": "kept"}
