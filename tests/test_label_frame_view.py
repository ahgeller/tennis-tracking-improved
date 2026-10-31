"""finetune/labeling/label_frame_view.py: the single-frame picture and HUD over a synthetic tool -
what the text lines say, what a render records for the mouse, and what a click on a chip or the
inset does. Frames are blank arrays."""
import numpy as np

from finetune.labeling.label_draw import SEPARATOR, UNSAVED_COLOUR, fit_items, middle_ellipsis, text_width

from finetune.labeling.label_frame_view import INSET_SHARE, THUMBNAILS_KEPT
from _workspace_helpers import fake_source, fake_tool

WIDTH, HEIGHT = 640, 360


def source(clip, count=6):
    return fake_source(clip, count, WIDTH, HEIGHT, conf=0.8)


def tool_over(*sources):
    return fake_tool(*(sources or [source("a")]))


def test_hud_place_names_the_clip_the_run_and_the_walk():
    tool = tool_over(source("a"), source("b"))
    view = tool.frame_view
    tool.session.cursor = 7
    assert view.hud_place() == "b [2/6]  clip 2/2  run 8/12"
    tool.session.walk, tool.session.walk_pos = [1, 7, 9], 1
    assert view.hud_place().endswith("UNCERTAIN ONLY 2/3 (3 left)")
    assert tool_over(source("a")).frame_view.hud_place() == "a [1/6]"


def test_hud_state_reads_the_label_and_the_suggestion():
    tool = tool_over()
    view = tool.frame_view
    row = tool.session.rows[0]
    assert view.hud_state(row, None, None) == "(100.0, 80.0)"
    offer = {"x": 103.0, "y": 84.0, "trusted": True}
    assert view.hud_state(row, offer, offer) == "HINT (103.0, 84.0) sure  grey draft 15.0px off"
    row.visible = False
    assert view.hud_state(row, offer, {**offer, "trusted": False}) == "HINT (103.0, 84.0) UNSURE, look closely  draft: no ball"
    assert view.hud_state(row, offer, None) == "NOT VISIBLE  hint: the video shows a ball here (y takes it)"
    row.visible = True
    assert view.hud_state(row, {"x": 100.0, "y": 80.0, "trusted": False}, None) == "(100.0, 80.0)  hint: here (unsure)"


def test_hud_lines_and_counts_follow_the_rows():
    tool = tool_over()
    view = tool.frame_view
    tool.session.rows[0].reviewed = True
    tool.session.rows[1].suspect = True
    line1, line2 = view.hud_lines(tool.session.rows[0], None, None)
    assert line1 == "a [1/6]  frame 0  (100.0, 80.0)  ok  landings 0  zoom x1.0"     # reviewed: where it came from is moot
    assert line2.startswith("SPACE=good  y=take hint  v=no ball")
    assert "reviewed 1/6  uncertain left 1  flagged left 1" in line2                  # a suspect row is uncertain
    assert "frame 2  (110.0, 84.0)  from csv conf ?  unreviewed" in view.hud_lines(tool.session.rows[2], None, None)[0]
    tool.session.rows[2].source, tool.session.rows[2].conf = "det", 0.83
    assert "from det 0.83  unreviewed" in view.hud_lines(tool.session.rows[2], None, None)[0]
    tool.session.source.landings[0] = None
    assert "  ok  LANDING  landings 1  zoom x1.0" in view.hud_lines(tool.session.rows[0], None, None)[0]
    assert "SPACE=take hint  x=keep draft" in view.hud_lines(tool.session.rows[0], None, {"x": 1.0, "y": 1.0, "trusted": True})[1]


def test_a_hud_line_drops_whole_items_least_important_first():
    items = [(1, "video6 [86/478]"), (0, "frame 170"), (5, "landings 3"), (6, "zoom x2.0"), (2, "EDITED")]
    full = SEPARATOR.join(text for _, text in items)
    width = text_width(full, 0.55, 1)
    assert fit_items(items, width, 0.55, 1) == full
    assert fit_items(items, width - 1, 0.55, 1) == "video6 [86/478]  frame 170  landings 3  EDITED"
    assert fit_items(items, text_width("video6 [86/478]  frame 170", 0.55, 1), 0.55, 1) == "video6 [86/478]  frame 170"
    assert fit_items(items, text_width("frame 1", 0.55, 1), 0.55, 1) == "frame"                 # the last one is cut
    assert middle_ellipsis("a" * 10 + "b" * 30 + "c" * 10, 20) == "a" * 9 + "..." + "c" * 8
    assert middle_ellipsis("video6") == "video6"


def test_unsaved_has_a_slot_of_its_own_so_line_one_does_not_move():
    tool = tool_over()
    tool.fit_view(1280, 720)
    session = tool.session
    hud_top = session.view_h + session.timeline_h
    before = tool.render()[hud_top:hud_top + int(30 * session.ui)].copy()
    session.touch("ball")
    after = tool.render()[hud_top:hud_top + int(30 * session.ui)]
    changed = np.argwhere((before != after).any(axis=2))
    assert len(changed) and changed[:, 1].min() > session.view_w * 0.85               # only the slot at the right end
    assert (after[changed[:, 0], changed[:, 1]] == UNSAVED_COLOUR).all(axis=1).any()


def test_the_chips_sit_on_line_two_so_line_one_has_the_whole_width():
    tool = tool_over()
    tool.fit_view(1280, 720)
    tool.render()
    session, view = tool.session, tool.frame_view
    line1_bottom = session.view_h + session.timeline_h + int(28 * session.ui)
    assert all(box[1] >= line1_bottom for box, _ in view.step_buttons)
    assert view.centre_button[1] >= line1_bottom


def test_render_records_the_chips_and_the_inset_for_the_mouse():
    tool = tool_over()
    tool.fit_view(1280, 720)
    tool.session.zoom = tool.session.zoom_target = 1.0
    tool.follow_ball()
    view = tool.frame_view
    canvas = tool.render()
    assert canvas.shape == (tool.session.view_h + tool.session.hud_h, tool.session.view_w, 3)
    assert view.centre_button and view.filter_button and view.inset_button and len(view.step_buttons) == 5
    assert all(box[1] >= tool.session.view_h for box, _ in view.step_buttons)        # on the bar, under the picture
    assert view.inset_box is None                                            # no inset at x1
    tool.session.zoom = tool.session.zoom_target = 4.0
    tool.follow_ball()
    tool.render()
    left, top, right, bottom = view.inset_box
    assert right - left + 1 == int(tool.session.view_w * INSET_SHARE) and bottom < tool.session.view_h


def test_clicks_on_the_chips_and_the_inset_do_what_they_say():
    tool = tool_over()
    tool.fit_view(1280, 720)
    tool.session.zoom = tool.session.zoom_target = 4.0
    tool.follow_ball()
    tool.render()
    view = tool.frame_view
    box, jump = view.step_buttons[-1]                                       # x1, the leftmost chip
    view.click_hud(box[0], box[1])
    assert tool.session.step == jump
    view.click_hud(view.centre_button[0], view.centre_button[1])
    assert not tool.session.follow
    view.click_hud(view.inset_button[0], view.inset_button[1])
    assert not tool.session.inset
    tool.session.inset = True
    left, top, right, bottom = view.inset_box
    assert view.inset_hit(left, top) and not view.inset_hit(left - 1, top)
    view.click_inset(right, bottom)
    assert tool.session.center[0] > WIDTH * 0.9 and tool.session.center[1] > HEIGHT * 0.9   # the inset's far corner is the frame's


def test_thumbnails_are_cached_by_frame_and_width_and_bounded():
    tool = tool_over()
    view = tool.frame_view
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    first = view.thumbnail(0, image, 64, 36)
    assert first.shape == (36, 64, 3) and view.thumbnail(0, image, 64, 36) is first
    for frame in range(1, THUMBNAILS_KEPT + 1):
        view.thumbnail(frame, image, 64, 36)
    assert len(view.thumbs) == THUMBNAILS_KEPT and ("a", 0, 64) not in view.thumbs   # the oldest went first


def test_an_ask_comes_first_on_the_second_line_so_a_narrow_bar_cuts_the_counts_not_the_ask():
    tool = tool_over()
    tool.session.awaiting_hit = 0.0
    line2 = tool.frame_view.hud_lines(tool.session.rows[0], None, None)[1]
    assert line2.startswith("CLICK WHERE THE RACKET MET THE BALL") and "reviewed 0/6" in line2
