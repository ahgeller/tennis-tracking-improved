"""finetune.data.event_labels: the hit and rally label files, read and written on temporary files,
the hits' blank points filled from the ball labels and the rally marks paired up. No workspace
file is read."""
import pytest

from finetune.data import workspace
from finetune.data.event_labels import (HitLabel, RallyLabel, check_rallies, hitter_of, pair_rally_marks, read_hit_labels,
                                        read_rally_labels, resolve_hit_points, write_hit_labels, write_rally_labels)
from finetune.data.labels import LabelFormatError


def test_the_new_label_files_sit_beside_the_ball_labels(tmp_path):
    assert workspace.hits_csv("video6", tmp_path) == tmp_path / "video6_hits.csv"
    assert workspace.rallies_csv("video6", tmp_path) == tmp_path / "video6_rallies.csv"
    assert workspace.players_csv("video6", tmp_path) == tmp_path / "video6_players.csv"
    assert workspace.court_csv("video6", tmp_path) == tmp_path / "video6_court.csv"
    assert workspace.events_review_json("video6", tmp_path) == tmp_path / "video6_events.review.json"


# --------------------------------------------------------------------------- hits

def test_hits_are_written_in_frame_order_with_their_points_and_hitters_and_read_back(tmp_path):
    path = tmp_path / "video6_hits.csv"
    labels = [HitLabel(120, None, "far"), HitLabel(35.5, (640.5, 300.25), "near"), HitLabel(80)]
    assert write_hit_labels(path, labels) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "frame,ball_x,ball_y,hitter", "frame_035.5,640.50,300.25,near", "frame_080,,,", "frame_120,,,far"]
    assert read_hit_labels(path) == [HitLabel(35.5, (640.5, 300.25), "near"), HitLabel(80, None, None),
                                     HitLabel(120, None, "far")]


def test_a_hit_frame_given_twice_is_refused_before_the_file_is_touched(tmp_path):
    path = tmp_path / "video6_hits.csv"
    write_hit_labels(path, [HitLabel(5)])
    with pytest.raises(ValueError, match="given twice"):
        write_hit_labels(path, [HitLabel(80, None, "near"), HitLabel(80, (1.0, 2.0), "far")])
    assert read_hit_labels(path) == [HitLabel(5)]


@pytest.mark.parametrize("row, message", [
    ("frame_012,,,left", "bad hitter"),
    ("12,,,near", "frame"),
    ("frame_012,5,,near", "one coordinate without the other"),
])
def test_a_damaged_hit_row_is_an_error_naming_the_file(tmp_path, row, message):
    path = tmp_path / "video6_hits.csv"
    path.write_text("frame,ball_x,ball_y,hitter\n" + row + "\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="video6_hits.csv") as raised:
        read_hit_labels(path)
    assert message in str(raised.value)


def test_a_hit_file_listing_a_frame_twice_is_refused(tmp_path):
    path = tmp_path / "video6_hits.csv"
    path.write_text("frame,ball_x,ball_y,hitter\nframe_012,,,near\nframe_012,,,far\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="listed twice"):
        read_hit_labels(path)


def test_hitter_cells_are_near_far_or_blank_in_any_case():
    assert (hitter_of("NEAR"), hitter_of(" far "), hitter_of("")) == ("near", "far", None)


def test_a_hit_without_a_point_takes_the_ball_labels_its_own_point_is_kept():
    ball = {10: (100.0, 50.0), 11: (110.0, 60.0), 20: None}
    resolved = resolve_hit_points([HitLabel(10), HitLabel(10.5), HitLabel(11, (1.0, 2.0)), HitLabel(20), HitLabel(30)], ball)
    assert [label.point for label in resolved] == [(100.0, 50.0), (105.0, 55.0), (1.0, 2.0), None, None]


# --------------------------------------------------------------------------- rallies

def test_rallies_are_written_and_read_back_with_a_blank_cell_for_an_unmarked_end(tmp_path):
    path = tmp_path / "video6_rallies.csv"
    rallies = [RallyLabel(10, 200), RallyLabel(300, None), RallyLabel(None, 900)]
    assert write_rally_labels(path, rallies) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "start,end", "frame_010,frame_200", "frame_300,", ",frame_900"]
    assert read_rally_labels(path) == rallies
    assert [rally.complete for rally in rallies] == [True, False, False]


@pytest.mark.parametrize("rallies, message", [
    ([RallyLabel(None, None)], "start or its end"),
    ([RallyLabel(50, 50)], "does not start before"),
    ([RallyLabel(10, 100), RallyLabel(90, 200)], "overlap"),
    ([RallyLabel(300, 400), RallyLabel(10, 100)], "overlap"),
])
def test_rallies_out_of_order_overlapping_or_inside_out_are_refused(tmp_path, rallies, message):
    with pytest.raises(ValueError, match=message):
        check_rallies(rallies)
    path = tmp_path / "video6_rallies.csv"
    with pytest.raises(ValueError):
        write_rally_labels(path, rallies)
    assert not path.exists()


def test_a_damaged_rally_file_is_an_error_naming_it(tmp_path):
    path = tmp_path / "video6_rallies.csv"
    path.write_text("start,end\nframe_100,frame_050\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="video6_rallies.csv"):
        read_rally_labels(path)


def test_marks_pair_into_rallies_each_start_with_the_next_end():
    assert pair_rally_marks([10, 300], [200, 500]) == [RallyLabel(10, 200), RallyLabel(300, 500)]
    # A start with another start after it before any end: its end is not marked yet.
    assert pair_rally_marks([10, 50], [200]) == [RallyLabel(10, None), RallyLabel(50, 200)]
    # An end with no start since the last end: its start is not marked yet.
    assert pair_rally_marks([300], [100, 400]) == [RallyLabel(None, 100), RallyLabel(300, 400)]
    assert pair_rally_marks([700], []) == [RallyLabel(700, None)]
    assert pair_rally_marks((frame for frame in [5]), (frame for frame in [5, 9])) == [RallyLabel(None, 5), RallyLabel(None, 9)]
    check_rallies(pair_rally_marks([10, 50, 300], [200, 100, 600]))   # whatever the marks, the pairs are valid
