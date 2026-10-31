"""The label tool's player identity view (--players) and court view (--court), headless: segments
cut from a small pipeline run, who they are said to be, the court's points pre-filled from the run,
moved and confirmed, and the files both are saved to. Every file is under tmp_path; no video is
decoded (the crop reader is not started)."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from finetune.data.court_labels import GROUND_POINTS, read_court_labels
from finetune.data.player_labels import PlayerBoxLabel, read_player_labels, write_player_labels
from finetune.labeling import label_players
from finetune.labeling import label_window as win
from finetune.labeling.label_players import Segment, appearance, cut_segments, split_at_labels
from finetune.labeling.label_tool import Tool
from _workspace_helpers import fake_source, fake_tool

WIDTH, HEIGHT = 1280, 720


def run_json(path: Path, frames: int = 40, court=None, tracks=None) -> Path:
    """A pipeline run's JSON at 30 FPS: track 1 on every frame, track 2 from frame 0 to 9 and again
    from 30 (a gap over 0.5 s), the court on every frame."""
    rows = []
    for frame in range(frames):
        boxes = {"1": [100.0, 300.0, 160.0, 460.0]}
        if frame < 10 or frame >= 30:
            boxes["2"] = [700.0, 100.0, 740.0, 200.0]
        if tracks is not None:
            boxes = tracks(frame)
        rows.append({"frame": frame, "player_boxes": boxes, "court_keypoints": court})
    path.write_text(json.dumps({"video": {"fps": 30.0, "width": WIDTH, "height": HEIGHT}, "frames": rows,
                                "court_events": {"contacts": [], "rally_legs": []}}), encoding="utf-8")
    return path


def clip(tmp_path, frames=40, court=None):
    source = fake_source("clip", frames, WIDTH, HEIGHT)
    source.run = run_json(tmp_path / "clip_gridtracknet.json", frames, court)
    source.players_path, source.court_path = tmp_path / "clip_players.csv", tmp_path / "clip_court.csv"
    for attribute, suffix in (("out_csv", "_ball.csv"), ("review", "_ball.review.json"), ("suspects", "_ball.suspect.json"),
                              ("bounces", "_bounces.csv"), ("hits_path", "_hits.csv"), ("rallies_path", "_rallies.csv"),
                              ("events_review", "_events.review.json")):
        setattr(source, attribute, tmp_path / f"clip{suffix}")
    source.video = tmp_path / "clip.mp4"
    return source


def press(tool, key, shift=False):
    code = key if isinstance(key, int) else ord(key)
    with patch.object(Tool, "view_ready", return_value=True), patch.object(win, "shift_held", return_value=shift), \
            patch.object(win, "non_character_key", return_value=False):
        tool.keys.handle(code)


def mouse(tool, event, x, y, flags=0):
    tool.pointer.on_mouse(event, x, y, flags, None)


# --------------------------------------------------------------------------- segments

def test_tracks_are_cut_into_segments_at_every_gap_over_half_a_second():
    boxes = {frame: {"1": [0, 0, 1, 1]} for frame in range(20)}
    for frame in (0, 1, 2, 30, 31):
        boxes.setdefault(frame, {})["2"] = [5, 5, 6, 6]
    segments = cut_segments(boxes, 30.0)
    assert [(s.track, s.first, s.last) for s in segments] == [("1", 0, 19), ("2", 0, 2), ("2", 30, 31)]
    assert segments[0].samples()[0] == 0 and segments[0].samples()[-1] == 19 and len(segments[0].samples()) == 12


def test_a_segment_is_split_where_its_labels_change_identity():
    segment = Segment("7", [0, 1, 2, 3, 4], {f: (0.0, 0.0, 1.0, 1.0) for f in range(5)})
    labels = {(f, "7"): PlayerBoxLabel(f, "P1" if f < 2 else "P2", (0.0, 0.0, 1.0, 1.0), "7") for f in range(4)}
    parts = split_at_labels([segment], labels)
    assert [(p.first, p.last) for p in parts] == [(0, 1), (2, 3), (4, 4)]


def test_appearance_tells_two_shirts_apart():
    red = np.zeros((176, 80, 3), np.uint8)
    red[:] = (40, 40, 220)
    blue = np.zeros_like(red)
    blue[:] = (220, 60, 40)
    same = cv2.compareHist(appearance([red]), appearance([red, red]), cv2.HISTCMP_CORREL)
    other = cv2.compareHist(appearance([red]), appearance([blue]), cv2.HISTCMP_CORREL)
    assert same > 0.99 and other < 0.2 and appearance([]) is None


# --------------------------------------------------------------------------- the player view

def open_players(tmp_path):
    source = clip(tmp_path)
    tool = fake_tool(source)
    with patch.object(label_players.CropReader, "start"):
        tool.player_view.open()
    return tool, source


def test_the_player_view_shows_the_runs_segments_and_the_keys_say_who_they_are(tmp_path):
    tool, source = open_players(tmp_path)
    view = tool.player_view
    assert tool.overview and [(s.track, s.first, s.last) for s in view.segments] == [("1", 0, 39), ("2", 0, 9), ("2", 30, 39)]
    press(tool, "1")
    assert view.identity(view.segments[0]) == "P1" and view.selected == 1        # and on to the next
    assert len(source.players) == 40 and source.changed == {"players"}
    assert view.suggestion(1) == "P2"                                            # P1 is on court with it
    press(tool, " ")
    assert view.identity(view.segments[1]) == "P2"
    press(tool, "0")
    assert view.identity(view.segments[2]) == "none" and view.selected == 2      # the last one stays up
    press(tool, "u")
    assert view.identity(view.segments[2]) is None
    canvas = tool.render()
    assert canvas.shape == (tool.session.view_h + tool.session.hud_h, tool.session.view_w, 3)


def test_player_labels_are_saved_per_box_and_a_split_comes_back_when_reopened(tmp_path):
    tool, source = open_players(tmp_path)
    view = tool.player_view
    tool.render()
    index, frame, x0, y0, x1, y1 = next(box for box in view.crop_boxes if box[0] == 0 and box[1] > 0)
    mouse(tool, cv2.EVENT_RBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2)            # split track 1 at that crop
    assert [(s.first, s.last) for s in view.segments[:2]] == [(0, frame - 1), (frame, 39)]
    view.select(0)
    press(tool, "1")
    press(tool, "2")
    tool.save(quiet=True)
    saved = read_player_labels(source.players_path)
    assert {label.player for label in saved if label.frame < frame} == {"P1"}
    assert {label.player for label in saved if label.frame >= frame and label.track == "1"} == {"P2"}
    assert saved[0].box == (100.0, 300.0, 160.0, 460.0) and saved[0].track == "1"
    source.players = None
    with patch.object(label_players.CropReader, "start"):
        view.open()
    assert [(s.track, s.first, s.last) for s in view.segments][:2] == [("1", 0, frame - 1), ("1", frame, 39)]
    assert view.selected == 2                                                     # the first one not yet said


def test_without_a_run_the_player_view_says_how_to_get_one(tmp_path):
    source = clip(tmp_path)
    source.run = None
    tool = fake_tool(source)
    tool.player_view.open()
    assert tool.player_view.message and "--candidates" in tool.player_view.message
    tool.render()
    press(tool, "1")                                                              # nothing to assign: no harm
    assert source.players is None



def test_stored_labels_follow_their_boxes_when_a_rerun_has_new_track_ids(tmp_path):
    """ByteTrack numbers its tracks afresh on every run: a label belongs to its box, found by
    overlap, not to the track id it was saved with. A fix to one person never touches the other's
    rows, and a label whose box the run no longer has is kept as it was."""
    left, right = (100.0, 300.0, 160.0, 460.0), (700.0, 100.0, 740.0, 200.0)
    stored = ([PlayerBoxLabel(f, "P1", left, "1") for f in range(40)] + [PlayerBoxLabel(f, "P2", right, "2") for f in range(40)]
              + [PlayerBoxLabel(5, "P3", (1000.0, 500.0, 1040.0, 600.0), "9"),            # nobody there in this run
                 PlayerBoxLabel(6, "P3", (1000.0, 500.0, 1040.0, 600.0), "3")])
    source = clip(tmp_path)
    write_player_labels(source.players_path, stored)
    run_json(source.run, tracks=lambda f: {"1": list(right), "3": list(left)})           # renumbered, 1 now the other one
    tool = fake_tool(source)
    with patch.object(label_players.CropReader, "start"):
        tool.player_view.open()
    view = tool.player_view
    shown = {segment.track: view.identity(segment) for segment in view.segments}
    assert shown == {"3": "P1", "1": "P2"}                                         # by box, not by id
    assert view.next_unassigned(-1, 1) is None                                    # f has nothing left to find
    view.select([s.track for s in view.segments].index("1"))
    press(tool, "4")                                                              # the right person is P4 after all
    tool.save(quiet=True)
    saved = read_player_labels(source.players_path)                               # and it reads back
    assert {(label.player, label.track) for label in saved if label.box == left} == {("P1", "3")}
    assert len([label for label in saved if label.box == left]) == 40
    assert {(label.player, label.track) for label in saved if label.box == right} == {("P4", "1")}
    assert PlayerBoxLabel(5, "P3", (1000.0, 500.0, 1040.0, 600.0), "9") in saved   # kept, never overwritten,
    assert PlayerBoxLabel(6, "P3", (1000.0, 500.0, 1040.0, 600.0), "") in saved    # its gone run's id dropped
                                                                                  # where this run uses it


def test_space_with_no_identity_free_leaves_the_label_alone(tmp_path, capsys):
    tool, source = open_players(tmp_path)
    view = tool.player_view
    press(tool, "1")
    view.select(0)
    with patch.object(type(view), "suggestion", return_value=None):
        press(tool, " ")
    assert view.identity(view.segments[0]) == "P1"                                # SPACE never takes a label off
    assert "no identity" in capsys.readouterr().out



# --------------------------------------------------------------------------- the crops

def test_the_crop_reader_cuts_every_crop_asked_for_and_says_where_the_video_ended(tmp_path):
    from tests._helpers import write_clip
    video = write_clip(tmp_path / "clip.avi", 6, size=(320, 180))
    wanted = {1: [("1", (10.0, 10.0, 60.0, 120.0)), ("2", (200.0, 20.0, 240.0, 100.0))], 4: [("1", (12.0, 10.0, 62.0, 120.0))]}
    got = {}
    reader = label_players.CropReader(str(video), wanted, lambda track, frame, crop: got.__setitem__((track, frame), crop))
    reader.read()
    assert set(got) == {("1", 1), ("2", 1), ("1", 4)} and reader.done == reader.total == 2 and reader.error is None
    assert all(crop.shape[0] == label_players.CROP_H for crop in got.values())
    short = label_players.CropReader(str(video), {9: [("1", (10.0, 10.0, 60.0, 120.0))]}, lambda *crop: None)
    short.read()
    assert short.error and "ended at frame" in short.error


def test_a_crop_at_the_picture_edge_keeps_its_shape_with_the_part_off_the_picture_black():
    image = np.full((100, 200, 3), 255, np.uint8)
    inside = label_players.crop_box(image, (50.0, 20.0, 90.0, 80.0))
    edge = label_players.crop_box(image, (-20.0, 20.0, 20.0, 80.0))               # half off the left edge
    assert edge.shape == inside.shape
    assert edge[:, :edge.shape[1] // 3].max() == 0 and edge[:, -3:].min() == 255
    assert label_players.crop_box(image, (-80.0, 20.0, -40.0, 80.0)) is None      # wholly off it


def test_a_split_while_the_crops_load_gives_each_part_the_crops_of_its_frames(tmp_path):
    from tests._helpers import write_clip
    tool, source = open_players(tmp_path)
    write_clip(source.video, 40, size=(WIDTH, HEIGHT))
    view = tool.player_view
    first = view.segments[0]
    view.split(0, 20)                                                             # before any crop is cut
    view.reader.read()                                                            # the reader carries on
    before, after = view.segments[0], view.segments[1]
    assert (before.first, before.last, after.first, after.last) == (first.first, 19, 20, first.last)
    assert view.crop_frames(before) and all(frame < 20 for frame in view.crop_frames(before))
    assert view.crop_frames(after) and all(frame >= 20 for frame in view.crop_frames(after))
    assert all(view.crop(part, frame) is not None for part in (before, after) for frame in view.crop_frames(part))
    assert view.look(before) is not None


# --------------------------------------------------------------------------- the court view

COURT = [300, 200, 980, 200, 100, 650, 1180, 650, 380, 200, 180, 650, 900, 200, 1100, 650,
         360, 280, 920, 280, 220, 560, 1060, 560, 640, 280, 640, 560]


def open_court(tmp_path, court=COURT):
    source = clip(tmp_path, court=court)
    tool = fake_tool(source)
    tool.court_view.open()
    return tool, source


def test_the_court_view_starts_from_the_runs_points_and_space_confirms_them(tmp_path):
    tool, source = open_court(tmp_path)
    view = tool.court_view
    assert view.frames == [0, 10, 20, 29, 39] and view.frame == 0
    assert view.points()["far_left"] == (300.0, 200.0) and len(view.points()) == 14
    assert source.court == {}                                                    # shown, not labelled
    press(tool, " ")
    assert set(source.court[0]) == set(GROUND_POINTS) and view.frame == 10 and source.changed == {"court"}
    tool.save(quiet=True)
    (label,) = read_court_labels(source.court_path)
    assert label.frame == 0 and label.points["near_centre"] == (640.0, 560.0)


def test_points_are_moved_by_nudge_drag_and_click_and_undone(tmp_path):
    tool, _ = open_court(tmp_path)
    view = tool.court_view
    press(tool, "l")
    press(tool, "k", shift=True)
    assert view.points()["far_left"] == (301.0, 200.25)
    press(tool, "u", shift=True)
    assert view.points()["far_left"] == (301.0, 200.0)
    tool.render()
    sx, sy = view.to_screen(*view.points()["near_right"])
    mouse(tool, cv2.EVENT_LBUTTONDOWN, sx, sy)                                    # picks near_right up
    mouse(tool, cv2.EVENT_MOUSEMOVE, sx - 20, sy - 10, cv2.EVENT_FLAG_LBUTTON)
    mouse(tool, cv2.EVENT_LBUTTONUP, sx - 20, sy - 10)
    moved = view.points()["near_right"]
    assert view.selected == GROUND_POINTS.index("near_right") and moved[0] < 1180.0 and moved[1] < 650.0
    press(tool, "u", shift=True)                                                  # the whole drag is one step
    assert view.points()["near_right"] == (1180.0, 650.0)
    for _ in range(GROUND_POINTS.index("near_right"), 14):                       # select the left net post
        press(tool, ".")
    assert view.selected == 14
    mouse(tool, cv2.EVENT_LBUTTONDOWN, 40, 40)                                    # a click away from every point
    assert "net_post_left" in view.points()
    press(tool, "v")                                                              # not visible after all
    assert "net_post_left" not in view.points()


def test_points_outside_the_picture_are_not_written_and_frames_copy_unlabel_and_reset(tmp_path):
    court = list(COURT)
    court[4:6] = [-40.0, 650.0]                                                   # near_left off the picture
    tool, source = open_court(tmp_path, court)
    view = tool.court_view
    press(tool, " ")
    assert "near_left" not in source.court[0] and len(source.court[0]) == 13
    press(tool, "r", shift=True)                                                  # frame 10: the model's points
    press(tool, "y")                                                              # ... or the last labelled frame's
    assert view.points() == source.court[0]
    press(tool, " ")
    press(tool, "p")
    press(tool, "x", shift=True)
    assert sorted(source.court) == [0]
    tool.save(quiet=True)
    assert [label.frame for label in read_court_labels(source.court_path)] == [0]



def test_the_keys_that_throw_work_away_take_shift_and_leaving_unconfirmed_changes_warns_first(tmp_path):
    tool, source = open_court(tmp_path)
    view = tool.court_view
    press(tool, " ")                                                              # frame 0 labelled; on to 10
    press(tool, "p")
    press(tool, "x")                                                              # without SHIFT: only a warning
    assert sorted(source.court) == [0] and "SHIFT+x" in view.notice
    press(tool, "l")                                                              # a change, not confirmed
    assert view.notice is None
    press(tool, "d")
    assert view.frame == 0 and "NOT CONFIRMED" in view.notice                     # held back once
    press(tool, "d")
    assert view.frame == 10                                                       # the same key again leaves
    press(tool, "a")
    press(tool, "u", shift=True)
    assert view.points() == source.court[0]


def test_a_point_said_hidden_is_listed_apart_from_one_not_placed_and_the_loupe_stays_on_the_picture(tmp_path):
    tool, _ = open_court(tmp_path)
    view = tool.court_view
    press(tool, "v")                                                              # 0 far left: cannot be seen
    line1 = " ".join(text for _, text in view.hud_lines()[0])
    assert "hidden: 0" in line1 and "not placed: 14,15" in line1 and "point 0 far left" in line1
    tool.fit_view(1280, 720)
    tool.render()
    x0, y0, size = view.loupe_box
    left, top, right, bottom = view.picture_box(tool.court_view.clip.frame(view.frame))
    assert left <= x0 and x0 + size <= right and top <= y0 and y0 + size + int(22 * tool.session.ui) <= bottom

def test_a_clip_without_a_run_starts_from_blank_frames(tmp_path):
    source = clip(tmp_path)
    source.run = None
    tool = fake_tool(source)
    tool.court_view.open()
    view = tool.court_view
    assert view.points() == {} and len(view.frames) == 5
    press(tool, " ")                                                              # nothing to keep
    assert source.court == {}
    tool.render()
    assert np.asarray(tool.render()).ndim == 3
    with pytest.raises(StopIteration):
        next(iter(source.court))


def test_space_suggests_the_identity_that_looks_most_alike_among_those_not_on_court_at_the_time(tmp_path):
    tool, source = open_players(tmp_path)
    view = tool.player_view
    red = np.zeros((176, 80, 3), np.uint8)
    red[:] = (40, 40, 220)
    blue = np.zeros_like(red)
    blue[:] = (220, 60, 40)

    view.reader = SimpleNamespace(wanted={f: [(str(t), (1.0, 1.0, 9.0, 9.0)) for t in range(1, 5)] for f in range(30)})

    def segment(track, first, last, crop):
        frames = list(range(first, last + 1))
        for frame in frames:
            view.store_crop(track, frame, crop)
        return Segment(track, frames, {f: (1.0, 1.0, 9.0, 9.0) for f in frames})
    view.segments = [segment("1", 0, 9, red), segment("2", 0, 9, blue), segment("3", 20, 29, blue),
                     segment("4", 20, 29, red)]
    view.select(0)
    press(tool, "1")
    press(tool, "2")
    assert view.suggestion(2) == "P2" and view.suggestion(3) == "P1"
    press(tool, " ")
    assert view.identity(view.segments[2]) == "P2"
    assert view.suggestion(3) == "P1"                                             # P2 is on court with it now


@pytest.mark.parametrize("view_name", ["player_view", "court_view"])
def test_c_confirms_like_space_and_shift_n_goes_to_the_next_clip_and_s_saves(tmp_path, view_name):
    tool, _ = open_players(tmp_path) if view_name == "player_view" else open_court(tmp_path)
    view = getattr(tool, view_name)
    with patch.object(type(view), "next_clip") as next_clip, patch.object(Tool, "save") as save:
        press(tool, "n")                                                          # n alone: the next segment / frame
        assert next_clip.call_count == 0
        press(tool, "p")                                                          # and back
        press(tool, "n", shift=True)                  # Qt hands SHIFT+n back as a lower-case n, 110
        press(tool, "s")
    assert next_clip.call_count == 1 and save.call_count == 1
    press(tool, "c")                                                              # c is SPACE here too
    if view_name == "player_view":
        assert view.identity(view.segments[0]) == "P1"
    else:
        assert 0 in tool.court_view.source.court



def test_the_loupe_crosshair_and_the_screen_point_sit_on_the_pixel_the_point_names(tmp_path):
    """Frame coordinates put pixel k's centre at k + 0.5: a point at (100.5, 50.5) is the middle of
    pixel (100, 50), and both the loupe's crosshair and the point on screen must be on it."""
    tool, source = open_court(tmp_path)
    view = tool.court_view
    picture = source.frame(0)
    picture[50, 100] = (0, 0, 255)                                                # one bright pixel
    view.selected = 0
    view.change({**view.points(), "far_left": (100.5, 50.5)})
    canvas = tool.render()
    x0, y0, size = view.loupe_box
    loupe = canvas[y0:y0 + size, x0:x0 + size]
    peak = np.argwhere((loupe == (0, 0, 255)).all(axis=2))
    assert [tuple(p) for p in peak] == [(size // 2, size // 2)]                   # under the crosshair's middle
    sx, sy = view.to_screen(100.5, 50.5)
    scale, off_x, off_y = view.mapping
    assert (math.floor(100 * scale + off_x) <= sx < math.floor(101 * scale + off_x) + 1
            and math.floor(50 * scale + off_y) <= sy < math.floor(51 * scale + off_y) + 1)
    assert view.to_screen(*view.to_frame(sx, sy)) == (sx, sy)                    # and back


def test_the_loupe_of_a_point_not_placed_follows_the_mouse(tmp_path):
    tool, _ = open_court(tmp_path)
    view = tool.court_view
    view.selected = 14                                                            # the left net post: not placed
    tool.render()
    x, y = view.to_screen(200.0, 400.0)
    mouse(tool, cv2.EVENT_MOUSEMOVE, x, y)
    assert view.hover is not None and abs(view.hover[0] - 200.0) < 2 and abs(view.hover[1] - 400.0) < 2
    revision = view.revision
    view.selected = 0                                                             # a placed point: the mouse is not followed
    mouse(tool, cv2.EVENT_MOUSEMOVE, x + 5, y)
    assert view.revision == revision


def test_question_mark_or_f1_lists_the_views_keys_and_any_key_closes_the_list(tmp_path):
    tool, source = open_court(tmp_path)
    plain = tool.render()
    press(tool, "?")
    assert tool.help_shown and tool.render() is not plain and tool.render() is tool.render()   # drawn once, then kept
    press(tool, "l")                                                              # closes the list, nothing else
    assert not tool.help_shown and tool.court_view.points()["far_left"] == (300.0, 200.0)
    with patch.object(win, "non_character_key", return_value=True):
        tool.keys.handle(win.VK_F1)                                               # F1 (p's code, the F1 key down)
    assert tool.help_shown
    for view in (tool.frame_view, tool.grid_view, tool.sheet_view, tool.player_view, tool.court_view):
        assert view.help_title and ("? / F1", "these keys") in view.help


def test_tab_takes_a_clip_view_back_to_the_frames_and_m_goes_full_screen_there(tmp_path):
    tool, _ = open_players(tmp_path)
    with patch.object(Tool, "toggle_fullscreen") as fullscreen:
        press(tool, "m")
    fullscreen.assert_called_once()
    press(tool, "	")
    assert tool.view is tool.frame_view and tool.player_view.reader is None      # left: no crops cut for nobody
    tool.court_view.open()
    press(tool, "	")
    assert tool.view is tool.frame_view
