"""The scorers of the new label types (hits_score, rallies_score, players_score, court_score), what
they share (label_scoring) and the scorecard that runs them all, on small archives and pipeline
runs written under tmp_path."""
import json
from pathlib import Path

import pytest

from finetune.data import player_labels
from finetune.data.court_labels import COURT_METRES, GROUND_POINTS, CourtLabel
from finetune.data.event_labels import HitLabel
from finetune.data.player_labels import PlayerBoxLabel
from finetune.scoring import court_score, hits_score, label_scoring, players_score, rallies_score, scorecard
from finetune.scoring.tracker_events import TrackerHit

ALL_NAME = label_scoring.ALL


# --------------------------------------------------------------------------- hits

def test_hits_match_one_to_one_within_the_tolerance_and_judge_the_hitter_and_the_point():
    labels = [HitLabel(100, (500.0, 300.0), "near"), HitLabel(200.5, None, "far"), HitLabel(400, None, None)]
    hits = [TrackerHit(101, 9.0, "kink", "near", (503.0, 304.0)), TrackerHit(200, 3.0, "gap", "near"),
            TrackerHit(300, 1.0, "kink", "far")]
    score = hits_score.score_clip("clip", labels, hits, 30.0, 0.1, 1.0)
    assert (score.labels, score.found, score.matched) == (3, 3, 2)
    assert score.timing_frames == [1, 0.5]
    assert (score.hitter_right, score.hitter_judged) == (1, 2)                # near right, far called near
    assert score.px_errors == [5.0]
    assert [label.frame for label in score.unmatched_labels] == [400] and [h.frame for h in score.unmatched_hits] == [300]
    line = hits_score.format_line(score)
    assert line.startswith("clip ") and "precision=0.667 recall=0.667" in line and "dt_med= 0.8fr" in line
    assert "hitter=0.500 hitter_n=  2" in line and "px_med=  5.0px" in line


# --------------------------------------------------------------------------- rallies

def test_rally_overlap_and_pairing_take_the_largest_overlaps_first():
    assert rallies_score.iou((0, 9), (0, 9)) == 1.0
    assert rallies_score.iou((0, 9), (5, 14)) == pytest.approx(5 / 15)
    assert rallies_score.iou((0, 9), (10, 19)) == 0.0
    assert rallies_score.pair_rallies([(0, 100), (200, 300)], [(150, 310), (10, 90), (0, 40)]) == [
        (0, 1, pytest.approx(81 / 101)), (1, 0, pytest.approx(101 / 161))]


def test_rallies_score_their_overlap_and_their_start_and_end_errors():
    score = rallies_score.score_clip("clip", [(0, 300), (600, 900), (1200, 1300)], [(30, 300), (600, 960)], 30.0)
    assert score.overlaps[2] == 0.0 and score.good == 2
    assert score.start_errors_s == [1.0, 0.0] and score.end_errors_s == [0.0, 2.0]
    line = rallies_score.format_line(score)
    assert "rallies=  3 found=  2" in line and "recall@.5=0.667 precision@.5=1.000" in line
    assert "start_med= 0.50s" in line and "end_med= 1.00s" in line



def test_a_found_rally_over_a_labelled_rally_still_without_an_end_is_not_counted_against_precision():
    """A labelled rally with a blank cell is left out of the scoring, but what the tracker found over
    it is not a false rally: nobody has said yet where that rally ends."""
    from finetune.data.event_labels import RallyLabel
    labels = [RallyLabel(10, 50), RallyLabel(100, None), RallyLabel(200, 250)]
    assert rallies_score.open_spans(labels) == [(100, 199)]
    found = [(12, 48), (105, 140), (202, 248), (300, 340)]
    score = rallies_score.score_clip("clip", rallies_score.complete_spans(labels), found, 30.0,
                                     rallies_score.open_spans(labels))
    assert (score.labels, score.found, score.good) == (2, 3, 2)          # (300, 340) is a false one; (105, 140) unjudged
    assert rallies_score.open_spans([RallyLabel(None, 80), RallyLabel(100, None)]) == [(0, 80), (100, None)]


# --------------------------------------------------------------------------- players

def box(x):
    return (x, 100.0, x + 40.0, 200.0)


def test_identity_is_the_share_of_labelled_boxes_on_the_track_given_that_player():
    labels = []
    run = {}
    for frame in range(10):
        labels += [PlayerBoxLabel(frame, "P1", box(100.0), ""), PlayerBoxLabel(frame, "P2", box(500.0), ""),
                   PlayerBoxLabel(frame, "none", box(900.0), "")]
        # P1 is track 1 then, from frame 6, track 3 (a switch); P2 is track 2 throughout; the
        # spectator is track 4; frame 9 misses P2.
        run[frame] = {"1" if frame < 6 else "3": list(box(102.0)), "4": list(box(900.0))}
        if frame != 9:
            run[frame]["2"] = list(box(498.0))
    score = players_score.score_clip("clip", labels, run)
    assert (score.boxes, score.covered) == (20, 19)
    assert score.right == 6 + 9                                               # track 1 is P1, track 3 is nobody
    assert score.switches == 1 and score.tracks == 4
    assert (score.non_players, score.non_players_taken) == (10, 0)
    line = players_score.format_line(score)
    assert "coverage=0.950 identity=0.750" in line and "switches=  1" in line and "non_players=0.000" in line


def test_a_spectator_taken_for_a_player_counts_against_the_track():
    labels = [PlayerBoxLabel(0, "none", box(0.0), ""), PlayerBoxLabel(1, "none", box(0.0), ""),
              PlayerBoxLabel(2, "P1", box(0.0), "")]
    run = {frame: {"5": list(box(0.0))} for frame in range(3)}
    score = players_score.score_clip("clip", labels, run)
    assert score.right == 1 and score.non_players_taken == 2


def test_boxes_below_the_match_overlap_are_not_matched():
    pairs = player_labels.match_frame([PlayerBoxLabel(0, "P1", (0.0, 0.0, 10.0, 10.0), "")],
                                      {"1": [5.0, 5.0, 15.0, 15.0], "2": [0.0, 0.0, 10.0, 9.0]})
    assert pairs == [(0, "2")]
    # One to one, the most box-frames in agreement: 5 + 6 beats 7 alone.
    assert players_score.assign_tracks({("1", "P1"): 5, ("2", "P1"): 7, ("2", "P2"): 6}) == {"1": "P1", "2": "P2"}


# --------------------------------------------------------------------------- the court

def court_image_points(scale=40.0, off=(100.0, 50.0)):
    """Every ground point of a court seen straight from above, `scale` pixels to the metre."""
    return {name: (off[0] + COURT_METRES[name][0] * scale, off[1] + COURT_METRES[name][1] * scale)
            for name in GROUND_POINTS}


def flat(points):
    out = []
    for name in GROUND_POINTS:
        out += list(points.get(name, (0.0, 0.0)))
    return out


def test_a_perfect_court_scores_zero_and_a_point_off_by_a_metre_scores_a_metre():
    truth = court_image_points()
    labels = [CourtLabel(5, {**truth, "net_post_left": (60.0, 525.0)})]
    predicted = dict(truth)
    predicted["near_centre"] = (truth["near_centre"][0] + 40.0, truth["near_centre"][1])   # a metre to the right
    del predicted["far_left"]
    score = court_score.score_clip("clip", labels, {5: flat(predicted)}, 1.0)
    assert (score.frames, score.points, score.found, score.posts) == (1, 14, 13, 1)
    assert max(score.px_errors) == pytest.approx(40.0) and sorted(score.px_errors)[0] == pytest.approx(0.0, abs=1e-6)
    assert max(score.m_errors) == pytest.approx(1.0, abs=1e-3)
    assert court_score.plane_homography({name: truth[name] for name in GROUND_POINTS[:3]}) is None
    missing = court_score.score_clip("clip", labels, {}, 1.0)
    assert missing.frames_without_court == 1 and missing.found == 0
    line = court_score.format_line(score)
    assert "frames=  1 no_court=  0 points=  14 found=0.929" in line and "m_med= 0.00m" in line and "posts=  1" in line



def test_a_frame_whose_labelled_points_cannot_pin_the_court_plane_gives_pixels_only():
    """Four points along one baseline fit some homography exactly, and it puts the rest of the court
    anywhere: metres from it would be noise. Nor is a fit that misses its own points trusted."""
    truth = court_image_points()
    baseline = {name: truth[name] for name in ("far_left", "far_left_singles", "far_right_singles", "far_right")}
    matrix, reason = court_score.fit_plane(baseline)
    assert matrix is None and "span" in reason
    bad = dict(truth)
    bad["near_centre"] = (truth["near_centre"][0] + 200.0, truth["near_centre"][1])   # five metres off: mislabelled
    matrix, reason = court_score.fit_plane(bad)
    assert matrix is None and "miss" in reason
    assert court_score.fit_plane(truth) == (pytest.approx(court_score.fit_plane(truth)[0]), None)
    labels = [CourtLabel(5, baseline), CourtLabel(6, dict(truth))]
    score = court_score.score_clip("clip", labels, {5: flat(truth), 6: flat(truth)}, 1.0)
    assert score.frames_off_plane == 1 and len(score.m_errors) == 14 and len(score.px_errors) == 18
    assert "no_plane=  1" in court_score.format_line(score)
    assert any("frame 5" in note and "span" in note for note in score.notes)


# --------------------------------------------------------------------------- shared

def test_metric_lines_read_back_and_compare():
    before = "\n".join(["=== hits [x] ===", "clip           labels=   3 found=   2 matched=   2 precision=1.000 "
                        "recall=0.667 | dt_med= 1.0fr dt_p90= 2.0fr | hitter=  n/a hitter_n=  0 | px_med=  n/a",
                        "    unmatched labels: 4", "ALL            labels=   3 found=   2 matched=   2 precision=1.000 "
                        "recall=0.667 | dt_med= 1.0fr dt_p90= 2.0fr | hitter=  n/a hitter_n=  0 | px_med=  n/a"])
    read = label_scoring.read_metric_lines(before)
    assert set(read) == {"clip", ALL_NAME} and read["clip"]["recall"] == 0.667 and read["clip"]["hitter"] is None
    assert read["clip"]["dt_med"] == 1.0
    after = {"clip": {**read["clip"], "recall": 1.0}, ALL_NAME: read[ALL_NAME]}
    table = label_scoring.compare(read, after, ("recall", "hitter"))
    assert "1.000 (+0.333)" in table and "n/a" in table and table.rstrip().splitlines()[-1].startswith("ALL")



# --------------------------------------------------------------------------- the commands

def archive_with(tmp_path: Path, **files) -> Path:
    archive = tmp_path / "archive"
    (archive / "labels").mkdir(parents=True)
    (archive / "videos").mkdir()
    (archive / "videos" / "video1.mp4").write_bytes(b"")
    (archive / "labels" / "video1_ball.csv").write_text("frame,ball_x,ball_y\nframe_100,100,200\n", encoding="utf-8")
    for suffix, text in files.items():
        (archive / "labels" / f"video1_{suffix}.csv").write_text(text, encoding="utf-8")
    return archive


def write_run(tmp_path: Path, frames=None, **block) -> Path:
    out = tmp_path / "run"
    out.mkdir(exist_ok=True)
    events = {"bounces": [], "bounce_candidates": [], "kinematic_bounces": [], "contacts": [], "rally_legs": []}
    events.update(block)
    payload = {"video": {"fps": 30.0, "width": 1920, "height": 1080}, "frames": frames or [], "court_events": events}
    (out / "video1_gridtracknet.json").write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_each_scorer_scores_a_run_and_says_so_when_nothing_is_labelled(tmp_path, capsys):
    archive = archive_with(tmp_path, hits="frame,ball_x,ball_y,hitter\nframe_100,,,near\n",
                           rallies="start,end\nframe_090,frame_400\n",
                           players="frame,player,x1,y1,x2,y2,track\nframe_000,P1,0,0,10,10,1\n",
                           court="frame,point,x,y\nframe_000,far_left,10,10\n")
    out = write_run(tmp_path, frames=[{"frame": 0, "player_boxes": {"9": [0, 0, 10, 10]}, "court_keypoints": [11, 10]}],
                    contacts=[{"frame": 101, "x": 1, "y": 2, "player_id": 9, "score": 5.0, "source": "kink"}],
                    rally_legs=[{"start_frame": 100, "end_frame": 400, "phase": "serve"}])
    common = ["--archive", str(archive), "--out-dir", str(out)]
    assert hits_score.main(common) == 0
    assert "precision=1.000 recall=1.000" in capsys.readouterr().out
    assert rallies_score.main(common) == 0
    assert "rallies=  1 found=  1" in capsys.readouterr().out
    assert players_score.main(common) == 0
    assert "coverage=1.000 identity=1.000" in capsys.readouterr().out
    assert court_score.main(common) == 0
    assert "points=   1 found=1.000" in capsys.readouterr().out
    empty = archive_with(tmp_path / "empty")
    assert hits_score.main(["--archive", str(empty), "--out-dir", str(out)]) == 0   # nothing labelled: not an error
    assert "has hit labels (labels/<clip>_hits.csv) yet: nothing to score" in capsys.readouterr().out


def test_a_scorer_compares_against_an_earlier_log_of_itself(tmp_path, capsys):
    archive = archive_with(tmp_path, hits="frame,ball_x,ball_y,hitter\nframe_100,,,\nframe_200,,,\n")
    out = write_run(tmp_path, contacts=[{"frame": 101, "x": 1, "y": 2, "player_id": 9, "score": 5.0, "source": "kink"}])
    assert hits_score.main(["--archive", str(archive), "--out-dir", str(out)]) == 0
    log = tmp_path / "before.log"
    log.write_text(capsys.readouterr().out, encoding="utf-8")
    write_run(tmp_path, contacts=[{"frame": 101, "x": 1, "y": 2, "player_id": 9, "score": 5.0, "source": "kink"},
                                  {"frame": 199, "x": 1, "y": 2, "player_id": 9, "score": 5.0, "source": "kink"}])
    assert hits_score.main(["--archive", str(archive), "--out-dir", str(out), "--compare", str(log)]) == 0
    text = capsys.readouterr().out
    assert "=== against" in text and "1.000 (+0.500)" in text


def test_a_missing_run_is_an_error_naming_it(tmp_path, capsys):
    archive = archive_with(tmp_path, rallies="start,end\nframe_090,frame_400\n")
    assert rallies_score.main(["--archive", str(archive), "--out-dir", str(tmp_path / "nowhere")]) == 1
    assert "video1_gridtracknet.json is missing" in capsys.readouterr().err


def test_the_scorecard_runs_every_section_and_names_the_ones_that_failed(tmp_path, capsys):
    archive = archive_with(tmp_path, hits="frame,ball_x,ball_y,hitter\nframe_100,,,near\n")
    (archive / "labels" / "video2_rallies.csv").write_text("start,end\nframe_090,frame_400\n", encoding="utf-8")
    out = write_run(tmp_path, contacts=[{"frame": 100, "x": 1, "y": 2, "player_id": 9, "score": 5.0, "source": "kink"}])
    assert scorecard.main(["--archive", str(archive), "--out-dir", str(out)]) == 1       # video2 was not run
    text = capsys.readouterr().out
    for name in ("landings", "hits", "rallies", "players", "court"):
        assert f"######## {name} ########" in text
    assert "precision=1.000 recall=1.000" in text and "has landing labels" in text and "has court labels" in text
    assert "[scorecard] failed: rallies" in text                                    # nothing labelled is not a failure
    assert scorecard.main(["--archive", str(archive), "--out-dir", str(out), "--only", "hits", "court"]) == 0


def test_the_new_scorers_read_the_label_home_by_default(monkeypatch, tmp_path):
    from finetune.data import workspace
    monkeypatch.setattr(workspace, "label_home", lambda: tmp_path / "main" / "finetune" / "labels")
    assert label_scoring.label_archive() == tmp_path / "main" / "finetune"



def test_one_label_source_rule_the_ball_review_gates_only_the_points_borrowed_from_the_ball_labels(tmp_path, capsys):
    """Every scorer, the landing scorer's included, reads a clip's labels of its type unless the clip
    is in exclude.txt. Whether its ball labels were reviewed holds back only the points a landing or
    a hit borrows from them, and a clip needs no ball labels at all for its other labels to count."""
    from finetune.data import workspace as ws
    from finetune.scoring import court_events_score
    archive = archive_with(tmp_path, hits="frame,ball_x,ball_y,hitter\nframe_100,,,near\n",
                           bounces="frame,ball_x,ball_y\nframe_100,,\n")
    labels = archive / "labels"
    (archive / "review_status.json").write_text(json.dumps({"verified_clips": []}), encoding="utf-8")
    (labels / "video2_court.csv").write_text("frame,point,x,y\nframe_000,far_left,10,10\n", encoding="utf-8")
    assert label_scoring.labelled_clips(None, archive, False, None, ws.hits_csv) == [
        ("video1", None, labels / "video1_hits.csv")]                         # scored; its ball labels lend no point
    assert "video1: its ball labels are still under review" in capsys.readouterr().out
    assert [clip for clip, _, _ in label_scoring.labelled_clips(None, archive, False, None, ws.court_csv)] == ["video2"]
    assert label_scoring.labelled_clips(None, archive, True, None, ws.hits_csv) == [
        ("video1", labels / "video1_ball.csv", labels / "video1_hits.csv")]
    (archive / "exclude.txt").write_text("video1\n", encoding="utf-8")
    assert label_scoring.labelled_clips(None, archive, False, None, ws.bounce_csv) == []
    assert "[skip] video1: listed in exclude.txt" in capsys.readouterr().out
    (archive / "exclude.txt").unlink()
    out = write_run(tmp_path, contacts=[{"frame": 100, "x": 1, "y": 2, "player_id": 9, "score": 5.0, "source": "kink"}],
                    bounces=[{"frame": 100, "x": 100.0, "y": 200.0, "score": 0.9, "origin": "model", "position": "filtered"}])
    assert hits_score.main(["--archive", str(archive), "--out-dir", str(out)]) == 0
    assert "precision=1.000 recall=1.000" in capsys.readouterr().out
    assert court_events_score.main(["--archive", str(archive), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "recall=1.000" in text and "px med=  n/a" in text                   # its point was not borrowed


def test_the_landing_scorer_reads_the_label_home_and_says_so_when_nothing_is_labelled(monkeypatch, tmp_path, capsys):
    from finetune.data import workspace
    from finetune.scoring import court_events_score
    archive = archive_with(tmp_path)
    monkeypatch.setattr(workspace, "label_home", lambda: archive / "labels")
    assert court_events_score.main(["--out-dir", str(tmp_path / "run")]) == 0
    assert f"no scorable clip in {archive} has landing labels (labels/<clip>_bounces.csv)" in capsys.readouterr().out
