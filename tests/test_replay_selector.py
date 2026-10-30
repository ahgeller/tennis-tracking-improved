"""finetune/scoring/replay_selector.py: the selector replayed over a tracking JSON's recorded
detections, on a synthetic run, and scored against labels."""
import json

import pytest

from ball_in_play_selector import core
from finetune.scoring.replay_selector import (DEFAULT_VARIANTS, VARIANTS, ReplayScene, main, recorded_detections,
                                        recorded_rows, replay_variant, report)
from finetune.data.labels import write_labels

WIDTH, HEIGHT, FRAMES = 640, 360, 40


def _track(track_id, frames, speed=10.0, conf=0.9, y=100.0):
    return {"track_id": track_id, "score": 1.0, "num_obs": len(frames), "span": frames[-1] - frames[0] + 1,
            "first_frame": frames[0], "last_obs_frame": frames[-1], "start": {"reason": "first", "rival_track": None},
            "end_reason": "clip_end", "selection": {}, "in_timeline": True, "score_breakdown": {},
            "observations": [{"frame": f, "x": 20.0 + speed * f, "y": y, "conf": conf, "area": 16.0, "on_motion": True,
                              "bbox": [18.0 + speed * f, y - 2, 22.0 + speed * f, y + 2]} for f in frames]}


def _payload(tracks, present_frames):
    """A run whose ``frames`` table shows the ball on ``present_frames`` at the first track's positions."""
    by_frame = {obs["frame"]: obs for obs in tracks[0]["observations"]}
    frames = []
    for frame in range(FRAMES):
        row = {"frame": frame, "present": False, "court_keypoints": None, "player_boxes": {}}
        if frame in present_frames and frame in by_frame:
            row.update(present=True, x=by_frame[frame]["x"], y=by_frame[frame]["y"], conf=0.9, source="det",
                       interpolated=False)
        frames.append(row)
    return {"schema_version": 4, "video": {"fps": 30.0, "width": WIDTH, "height": HEIGHT, "total_frames": FRAMES},
            "summary": {}, "config": {}, "timing": {}, "last_valid_court_keypoints": None,
            "chosen_track": tracks[0], "tracks": tracks, "motion_diagnostics": {}, "court_events": {}, "frames": frames}


BALL = list(range(0, 12)) + list(range(13, 30))          # a one-frame hole at 12 (a fill bridges one detector step)
CLUTTER = [34, 35, 36]                                   # a short slow track far from the ball


def test_recorded_detections_rebuild_the_rows_with_their_motion_flags():
    payload = _payload([_track(0, BALL), _track(1, CLUTTER, speed=0.5, y=300.0)], BALL)
    rows = recorded_detections(payload)
    assert len(rows) == FRAMES and [len(frame_rows) for frame_rows in rows[:3]] == [1, 1, 1] and rows[12] == []
    [det] = rows[5]
    assert (det.frame, det.cx, det.cy, det.conf, det.on_motion) == (5, 70.0, 100.0, 0.9, True)
    assert (det.x1, det.y1, det.x2, det.y2) == (68.0, 98.0, 72.0, 102.0) and det.area == 16.0
    assert ReplayScene.of(payload).court_polygon is None and len(ReplayScene.of(payload).player_boxes) == FRAMES


def test_build_detections_is_patched_only_for_the_duration_of_the_replay():
    original = core.build_detections
    rows = recorded_detections(_payload([_track(0, BALL)], BALL))
    with recorded_rows(rows):
        assert core.build_detections([[]] * FRAMES, None) is rows
    assert core.build_detections is original


def test_the_chain_fills_the_hole_the_raw_variant_leaves_and_the_union_keeps_every_track():
    payload = _payload([_track(0, BALL), _track(1, CLUTTER, speed=0.5, y=300.0)], BALL)
    present = {variant: {row["frame"] for row in replay_variant(variant, payload)["frames"] if row["present"]}
               for variant in VARIANTS}
    assert present["control"] == set(BALL)
    assert present["raw"] == set(BALL) | set(CLUTTER)                     # every detection, nothing filled
    assert present["chain"] == set(BALL) | {12}                            # the chosen ball, its hole filled
    assert present["union"] == set(BALL) | {12} | set(CLUTTER)             # every track, the fills over them
    chain = replay_variant("chain", payload)
    assert [track["track_id"] for track in chain["tracks"]] == [0, 1] and chain["chosen_track"]["track_id"] == 0
    assert chain["frames"][5]["player_boxes"] == {} and chain["frames"][12]["source"] == "interp"
    assert replay_variant("raw", payload)["tracks"] == [] and replay_variant("union", payload)["chosen_track"] is None


def test_report_scores_every_variant_against_the_labels_per_source_and_pooled(tmp_path, capsys):
    run_dir = tmp_path / "pipeline"
    run_dir.mkdir()
    (tmp_path / "labels").mkdir()
    payload = _payload([_track(0, BALL), _track(1, CLUTTER, speed=0.5, y=300.0)], BALL)
    (run_dir / "video13_gridtracknet.json").write_text(json.dumps(payload), encoding="utf-8")
    labels = {f: (20.0 + 10.0 * f, 100.0) for f in range(30)}
    labels.update({f: None for f in CLUTTER})
    write_labels(tmp_path / "labels" / "video13_ball.csv", labels, WIDTH, HEIGHT)
    (tmp_path / "clips.csv").write_text("clip,source\nvideo13,prof\n", encoding="utf-8")
    lines = report(run_dir, "gridtracknet", tmp_path, ["control", "chain", "raw"])
    by_variant = {}
    for text in lines:
        if text.startswith("=== replay ["):
            variant = text.split("[")[1].split("]")[0]
            by_variant[variant] = {}
        elif text:
            by_variant[variant][text.split()[0]] = text
    assert set(by_variant) == {"control", "chain", "raw"} and list(by_variant["chain"]) == ["video13", "source=prof", "ALL"]
    assert "hits=   30 wrong=  0 tracks=   2" in by_variant["chain"]["ALL"]        # the hole filled
    assert "hits=   30" in by_variant["chain"]["source=prof"]
    assert "hits=   29 wrong=  0 tracks=   2" in by_variant["control"]["ALL"]      # the run as written: the hole open
    assert "hits=   29 wrong=  0 tracks=   0" in by_variant["raw"]["ALL"]
    assert "falseAlarm=1.000 (3/3: bridged=0 false=3)" in by_variant["raw"]["video13"]      # the clutter, no chooser
    assert "falseAlarm=0.000 (0/3: bridged=0 false=0)" in by_variant["chain"]["video13"]
    assert main([str(run_dir), "--archive", str(tmp_path), "--variant", *DEFAULT_VARIANTS]) == 0
    assert "=== replay [control] ===" in capsys.readouterr().out
    assert main([str(tmp_path / "nothing"), "--archive", str(tmp_path)]) == 1
    with pytest.raises(SystemExit):
        main([str(run_dir), "--variant", "nope"])
