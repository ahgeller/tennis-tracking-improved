"""finetune.data.player_labels and court_labels: the player identity and court keypoint label files,
read and written on temporary files. No workspace file is read."""
import math

import pytest

from finetune.data.court_labels import (COURT_METRES, COURT_POINTS, GROUND_POINTS, CourtLabel, engine_points, labels_by_frame,
                                        read_court_labels, write_court_labels)
from finetune.data.labels import LabelFormatError
from finetune.data.player_labels import PlayerBoxLabel, identity_of, read_player_labels, write_player_labels


# --------------------------------------------------------------------------- players

def test_player_boxes_are_written_in_frame_order_and_read_back(tmp_path):
    path = tmp_path / "video6_players.csv"
    labels = [PlayerBoxLabel(12, "P2", (10.0, 20.0, 50.0, 120.5), "79"),
              PlayerBoxLabel(3, "none", (400.0, 20.0, 420.0, 60.0), "81"),
              PlayerBoxLabel(3, "P1", (100.0, 200.0, 150.0, 330.0), "78")]
    assert write_player_labels(path, labels) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "frame,player,x1,y1,x2,y2,track", "frame_003,P1,100.00,200.00,150.00,330.00,78",
        "frame_003,none,400.00,20.00,420.00,60.00,81", "frame_012,P2,10.00,20.00,50.00,120.50,79"]
    assert read_player_labels(path) == sorted(labels, key=lambda label: (label.frame, label.track))


@pytest.mark.parametrize("row, message", [
    ("frame_003,P5,1,2,3,4,7", "bad player"),
    ("frame_003,P1,10,2,3,4,7", "no area"),
    ("frame_003,P1,a,2,3,4,7", "could not convert"),
])
def test_a_damaged_player_row_is_an_error_naming_the_file(tmp_path, row, message):
    path = tmp_path / "video6_players.csv"
    path.write_text("frame,player,x1,y1,x2,y2,track\n" + row + "\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="video6_players.csv") as raised:
        read_player_labels(path)
    assert message in str(raised.value)


def test_one_track_with_two_boxes_on_a_frame_is_refused(tmp_path):
    path = tmp_path / "video6_players.csv"
    path.write_text("frame,player,x1,y1,x2,y2,track\nframe_003,P1,1,2,3,4,7\nframe_003,P2,5,6,7,8,7\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="two boxes"):
        read_player_labels(path)


def test_identities_are_read_in_any_case():
    assert (identity_of("p3"), identity_of("NONE")) == ("P3", "none")


# --------------------------------------------------------------------------- the court

def test_court_points_are_written_in_frame_and_point_order_and_read_back(tmp_path):
    path = tmp_path / "video6_court.csv"
    labels = [CourtLabel(90, {"net_post_left": (5.0, 6.0), "far_left": (1.0, 2.0)}),
              CourtLabel(10, {"near_centre": (3.25, 4.5)})]
    assert write_court_labels(path, labels) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "frame,point,x,y", "frame_010,near_centre,3.25,4.50", "frame_090,far_left,1.00,2.00",
        "frame_090,net_post_left,5.00,6.00"]
    read = read_court_labels(path)
    assert [label.frame for label in read] == [10, 90]
    assert list(read[1].points) == ["far_left", "net_post_left"]
    assert labels_by_frame(read)[90]["net_post_left"] == (5.0, 6.0)


@pytest.mark.parametrize("labels, message", [
    ([CourtLabel(1, {"far_left": (1.0, 2.0)}), CourtLabel(1, {"far_right": (1.0, 2.0)})], "twice"),
    ([CourtLabel(1, {})], "no court points"),
    ([CourtLabel(1, {"baseline": (1.0, 2.0)})], "no name for: baseline"),
])
def test_bad_court_labels_are_refused_before_the_file_is_touched(tmp_path, labels, message):
    path = tmp_path / "video6_court.csv"
    with pytest.raises((ValueError, LabelFormatError), match=message):
        write_court_labels(path, labels)
    assert not path.exists()


def test_a_point_given_twice_on_a_frame_in_a_file_is_refused(tmp_path):
    path = tmp_path / "video6_court.csv"
    path.write_text("frame,point,x,y\nframe_001,far_left,1,2\nframe_001,far_left,3,4\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="video6_court.csv"):
        read_court_labels(path)


def test_the_engine_keypoints_become_named_points_without_the_undetected_ones():
    flat = []
    for index in range(len(GROUND_POINTS)):
        flat += [100.0 + index, 50.0 + index]
    flat[2:4] = [0.0, 0.0]                  # far_right: not detected
    flat[4:6] = [float("nan"), 3.0]         # near_left: not finite
    flat[6:8] = [-20.0, 400.0]              # near_right: off the picture, but a point the engine gave
    points = engine_points(flat)
    assert "far_right" not in points and "near_left" not in points
    assert points["near_right"] == (-20.0, 400.0) and points["far_left"] == (100.0, 50.0)
    assert engine_points(None) == {} and engine_points([1.0, 2.0])["far_left"] == (1.0, 2.0)


def test_the_court_in_metres_is_an_itf_court():
    assert COURT_POINTS[:len(GROUND_POINTS)] == GROUND_POINTS and len(COURT_POINTS) == 16
    x0, y0, _ = COURT_METRES["far_left"]
    x1, y1, _ = COURT_METRES["near_right"]
    assert math.isclose(x1 - x0, 10.97) and math.isclose(y1 - y0, 23.77)
    assert math.isclose(COURT_METRES["near_service_left"][1] - COURT_METRES["far_service_left"][1], 12.8)
    assert math.isclose(COURT_METRES["far_right_singles"][0] - COURT_METRES["far_left_singles"][0], 8.23)
    assert COURT_METRES["net_post_left"][2] == pytest.approx(1.07)
