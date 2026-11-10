"""What a label tool save writes, end to end on real clips opened the way the tool opens them
(``build_source`` over a tiny video in tmp_path): only the kinds of label changed this session,
never a file another session changed since it was read, each kind on its own, and a rescue copy
of whatever could not be written. Every file is under tmp_path."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from finetune.data.bounce_labels import BounceLabel, write_bounce_labels
from finetune.data.event_labels import HitLabel, read_hit_labels, write_hit_labels
from finetune.data.labels import write_label_cells
from finetune.labeling import label_store
from finetune.labeling.label_source import build_source
from finetune.labeling.label_store import load_rows
from finetune.labeling.label_tool import Tool
from tests._helpers import write_clip

FRAMES = 8
SIZE = (320, 180)
COURT = [60, 40, 260, 40, 20, 160, 300, 160, 80, 40, 40, 160, 240, 40, 280, 160,
         70, 60, 250, 60, 50, 140, 270, 140, 160, 60, 160, 140]


def labelled_clip(folder: Path, draft: bool = False) -> Path:
    """A clip with ball labels on every frame (a pretrack draft instead with `draft`), a review
    file resuming on frame 5 at zoom 3, and one landing; the video and a pipeline run beside them."""
    write_clip(folder / "clip.avi", FRAMES, size=SIZE)
    labels = folder / ("clip_ball.csv.draft" if draft else "clip_ball.csv")
    write_label_cells(labels, [(frame, f"{40 + 10 * frame:.2f}", f"{50 + frame:.2f}") for frame in range(FRAMES)])
    (folder / "clip_ball.review.json").write_text(json.dumps({"cursor": 5, "frame": 5, "zoom": 3.0, "reviewed": [0, 1],
                                                              "edited": [], "evidence": {}}), encoding="utf-8")
    write_bounce_labels(folder / "clip_bounces.csv", [BounceLabel(3)])
    run_json(folder)
    return folder


def run_json(folder: Path) -> Path:
    """A pipeline run's tracking JSON for the clip: one player on every frame, the court on every frame."""
    path = folder / "clip_gridtracknet.json"
    rows = [{"frame": frame, "player_boxes": {"1": [100.0, 60.0, 130.0, 150.0]}, "court_keypoints": COURT}
            for frame in range(FRAMES)]
    path.write_text(json.dumps({"video": {"fps": 30.0, "width": SIZE[0], "height": SIZE[1]}, "frames": rows,
                                "court_events": {"contacts": [], "rally_legs": []}}), encoding="utf-8")
    return path


def open_tool(folder: Path) -> Tool:
    source = build_source("clip", folder, folder)
    assert source is not None
    source.run = folder / "clip_gridtracknet.json"
    tool = Tool([source])
    tool.session.cursor = source.cursor
    return tool


def digests(folder: Path):
    return {path.name: hashlib.md5(path.read_bytes()).hexdigest() for path in sorted(folder.iterdir()) if path.is_file()}


def ball_x(folder: Path, frame: int) -> float:
    return next(row.x for row in load_rows(folder / "clip_ball.csv", *SIZE) if row.frame == frame)


# --------------------------------------------------------------------------- only what changed

def test_a_court_session_writes_the_court_file_and_nothing_else(tmp_path):
    labelled_clip(tmp_path)
    before = digests(tmp_path)
    tool = open_tool(tmp_path)
    tool.court_view.open()
    tool.court_view.confirm()                        # labels frame 0 and moves on: the cursor moves too
    tool.save(quiet=True)
    tool.rescue_unsaved()
    after = digests(tmp_path)
    changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
    assert changed == {"clip_court.csv"}             # not the ball labels, the review (cursor, zoom), the landings


def test_a_court_session_on_a_drafted_clip_does_not_turn_the_draft_into_labels(tmp_path):
    labelled_clip(tmp_path, draft=True)
    tool = open_tool(tmp_path)
    tool.court_view.open()
    tool.court_view.confirm()
    tool.save(quiet=True)
    assert not (tmp_path / "clip_ball.csv").exists() and (tmp_path / "clip_court.csv").is_file()


def test_a_ball_edit_saved_meanwhile_survives_a_court_session_saving_later(tmp_path):
    labelled_clip(tmp_path)
    ball, court = open_tool(tmp_path), open_tool(tmp_path)    # two sessions on one clip
    court.court_view.open()
    ball.session.set_cursor(2)
    assert ball.set_point(140.0, 60.0)
    ball.save(quiet=True)
    court.court_view.confirm()
    court.save(quiet=True)
    assert ball_x(tmp_path, 2) == 140.0
    review = json.loads((tmp_path / "clip_ball.review.json").read_text(encoding="utf-8"))
    assert 2 in review["edited"]


def test_a_ball_edit_leaves_the_event_files_exactly_as_they_were(tmp_path):
    labelled_clip(tmp_path)
    (tmp_path / "clip_hits.csv").write_text("frame,ball_x,ball_y,hitter\n", encoding="utf-8")   # "no hits", said by hand
    (tmp_path / "clip_rallies.csv").write_text("start,end,note\nframe_001,frame_006,long\n", encoding="utf-8")
    before = digests(tmp_path)
    tool = open_tool(tmp_path)
    tool.session.set_cursor(4)
    tool.set_point(100.0, 70.0)
    tool.save(quiet=True)
    after = digests(tmp_path)
    assert {name for name in after if before.get(name) != after[name]} == {"clip_ball.csv", "clip_ball.review.json"}


def test_an_event_file_is_removed_only_when_the_tool_emptied_it(tmp_path):
    labelled_clip(tmp_path)
    tool = open_tool(tmp_path)
    tool.session.set_cursor(3)
    tool.toggle_landing()                            # the only landing, taken off: the tool emptied the file
    tool.save(quiet=True)
    assert not (tmp_path / "clip_bounces.csv").exists()
    header_only = "frame,ball_x,ball_y,hitter,note\n"
    (tmp_path / "clip_hits.csv").write_text(header_only, encoding="utf-8")
    tool = open_tool(tmp_path)
    tool.session.set_cursor(2)
    tool.toggle_hit()                                # a hit marked and taken off again
    tool.toggle_hit()
    tool.save(quiet=True)
    assert (tmp_path / "clip_hits.csv").read_text(encoding="utf-8") == header_only   # "no hits" still says so


# --------------------------------------------------------------------------- another session's changes

def test_a_file_changed_on_disk_since_it_was_read_is_never_written_over(tmp_path, capsys):
    labelled_clip(tmp_path)
    first, second = open_tool(tmp_path), open_tool(tmp_path)
    first.session.set_cursor(2)
    first.set_point(140.0, 60.0)
    first.save(quiet=True)
    second.session.set_cursor(6)
    second.set_point(200.0, 61.0)
    second.save(quiet=True)
    assert ball_x(tmp_path, 2) == 140.0 and ball_x(tmp_path, 6) == 100.0     # the first session's edit stands
    rescue = tmp_path / "clip_ball.unsaved.csv"
    assert rescue.is_file() and next(row.x for row in load_rows(rescue, *SIZE) if row.frame == 6) == 200.0
    out = capsys.readouterr().out
    assert "changed on disk" in out and "clip_ball.unsaved.csv" in out
    assert second.session.dirty                       # still unsaved, and said so
    second.save(quiet=True)
    assert "changed on disk" not in capsys.readouterr().out                   # said once, not every autosave


def test_a_hit_marked_meanwhile_in_another_session_is_kept(tmp_path):
    labelled_clip(tmp_path)
    first, second = open_tool(tmp_path), open_tool(tmp_path)
    first.session.set_cursor(1)
    first.toggle_hit()
    first.save(quiet=True)
    second.session.set_cursor(5)
    second.toggle_hit()
    second.save(quiet=True)
    assert [label.frame for label in read_hit_labels(tmp_path / "clip_hits.csv")] == [1]
    assert [label.frame for label in read_hit_labels(tmp_path / "clip_hits.unsaved.csv")] == [5]


# --------------------------------------------------------------------------- one kind at a time

def test_a_locked_ball_file_does_not_stop_the_hits_and_quitting_rescues_each_kind(tmp_path):
    labelled_clip(tmp_path)
    tool = open_tool(tmp_path)
    tool.session.set_cursor(2)
    tool.set_point(140.0, 60.0)
    tool.toggle_hit()
    tool.session.set_cursor(4)
    tool.toggle_landing()
    real = label_store.write_label_cells

    def locked(path, *rest):
        if path.name == "clip_ball.csv":
            raise PermissionError(13, "in use by another program")
        return real(path, *rest)
    with patch.object(label_store, "write_label_cells", side_effect=locked):
        tool.save(quiet=True)
        assert [label.frame for label in read_hit_labels(tmp_path / "clip_hits.csv")] == [2]   # saved anyway
        assert tool.session.dirty and tool.session.source.changed == {"ball"}
        tool.rescue_unsaved()
    assert ball_x(tmp_path, 2) == 60.0                  # on disk as it was
    assert next(row.x for row in load_rows(tmp_path / "clip_ball.unsaved.csv", *SIZE) if row.frame == 2) == 140.0


def test_rescue_copies_are_kept_out_of_git():
    ignored = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "finetune/labels/*.unsaved.*" in ignored


@pytest.mark.parametrize("name, rescue", [("clip_ball.csv", "clip_ball.unsaved.csv"),
                                          ("clip_ball.review.json", "clip_ball.review.unsaved.json"),
                                          ("clip_hits.csv", "clip_hits.unsaved.csv")])
def test_a_rescue_copy_sits_beside_its_file(tmp_path, name, rescue):
    from finetune.data.workspace import unsaved_copy
    assert unsaved_copy(tmp_path / name) == tmp_path / rescue


def test_hits_written_by_hand_with_points_survive_an_unrelated_ball_save(tmp_path):
    labelled_clip(tmp_path)
    write_hit_labels(tmp_path / "clip_hits.csv", [HitLabel(2, (10.0, 20.0), "near")])
    tool = open_tool(tmp_path)
    tool.session.set_cursor(0)
    tool.confirm()
    tool.save(quiet=True)
    assert read_hit_labels(tmp_path / "clip_hits.csv") == [HitLabel(2, (10.0, 20.0), "near")]
