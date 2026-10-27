"""Pass-1 stages that run on collected evidence: the batched ghost-ROI erase, and the RAM frame
cache pass 1 fills for pass 2."""
from types import SimpleNamespace

import numpy as np
from ball_in_play_selector import PackedMask

from tennis_tracker.config import Config
from tennis_tracker.pass1 import (
    _MIB, FrameCache, Pass1Result, _clear_boxes, _erase_ghosts_in_frame, _new_frame_cache, cache_frame_capacity,
    erase_ghost_rois,
)
from tennis_tracker.run_context import VideoInfo


def _mask(h=30, w=40, boxes=()):
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def _result(raw_boxes_by_frame, ghost_rois_by_frame):
    result = Pass1Result()
    for boxes, ghosts in zip(raw_boxes_by_frame, ghost_rois_by_frame):
        result.detections.append([])
        result.raw_motions.append(PackedMask.pack(_mask(boxes=boxes), roi=(0, 0, 40, 30)))
        result.boost_masks.append(PackedMask.pack(_mask(boxes=boxes)))
        result.rois.append(None)
        result.ghost_rois.append(list(ghosts))
        result.player_boxes.append({})
        result.court_keypoints.append(None)
    return result


def test_erase_clears_motion_inside_ghost_rois_and_repacks(capsys):
    ball = (5, 5, 9, 9)
    ghost_blob = (20, 20, 26, 26)
    result = _result(
        raw_boxes_by_frame=[[ball, ghost_blob], [ball], [ball, ghost_blob]],
        ghost_rois_by_frame=[[(18, 18, 30, 30)], [], [(18, 18, 30, 30)]],
    )
    before_frame1 = result.raw_motions[1]
    assert erase_ghost_rois(result) == 2
    for fi in (0, 2):
        raw = result.raw_motions[fi].unpack()
        assert raw[22, 22] == 0 and raw[6, 6] == 255           # ghost blob gone, ball kept
        assert result.raw_motions[fi].roi == (0, 0, 40, 30)    # packed rectangle preserved
        assert result.boost_masks[fi].unpack()[22, 22] == 0
    assert result.raw_motions[1] is before_frame1              # untouched frame keeps its object
    assert "cleaned 2 frames" in capsys.readouterr().out


def test_erase_drops_a_mask_that_becomes_empty_and_skips_frames_without_change(capsys):
    result = _result(
        raw_boxes_by_frame=[[(20, 20, 26, 26)], [(5, 5, 9, 9)]],
        ghost_rois_by_frame=[[(18, 18, 30, 30)], [(18, 18, 30, 30)]],
    )
    untouched = result.raw_motions[1]
    assert erase_ghost_rois(result) == 1          # frame 1's ghost ROI held no motion: not counted
    assert result.raw_motions[0] is None and result.boost_masks[0] is None
    assert result.raw_motions[1] is untouched and untouched.unpack()[6, 6] == 255
    assert "cleaned 1 frames" in capsys.readouterr().out


def test_erase_ignores_frames_with_no_masks():
    result = Pass1Result()
    result.detections.append([])
    result.raw_motions.append(None)
    result.boost_masks.append(None)
    result.rois.append(None)
    result.ghost_rois.append([(0, 0, 10, 10)])
    assert erase_ghost_rois(result) == 0


def test_clear_boxes_reports_only_boxes_that_held_motion():
    mask = _mask(boxes=[(5, 5, 9, 9)])
    assert _clear_boxes(None, [(5, 5, 9, 9)]) is False
    assert _clear_boxes(mask, [(20, 20, 26, 26)]) is False and mask.any()
    assert _clear_boxes(mask, [(20, 20, 26, 26), (4, 4, 10, 10)]) is True and not mask.any()


def test_erase_ghosts_in_frame_repacks_only_a_frame_that_changed():
    result = _result([[(5, 5, 9, 9)], [(5, 5, 9, 9)]], [[(20, 20, 26, 26)], [(4, 4, 10, 10)]])
    before = result.raw_motions[0]
    assert _erase_ghosts_in_frame(result, 0, result.ghost_rois[0]) is False
    assert result.raw_motions[0] is before
    assert _erase_ghosts_in_frame(result, 1, result.ghost_rois[1]) is True
    assert all(packed is None or not packed.unpack().any() for packed in (result.raw_motions[1], result.boost_masks[1]))
    assert _erase_ghosts_in_frame(Pass1Result(), 0, [(0, 0, 1, 1)]) is False       # no masks at all


# --------------------------------------------------------------------------- the pass-2 frame cache

def _frame(value: int) -> np.ndarray:
    return np.full((2, 4, 3), value, np.uint8)


def test_cache_capacity_is_the_budget_in_whole_frames():
    frame_mb = 1920 * 1080 * 3 / _MIB                      # one 1080p frame, about 5.9 MiB
    assert cache_frame_capacity(4096.0, 1920, 1080) == int(4096.0 / frame_mb) == 690
    assert cache_frame_capacity(frame_mb, 1920, 1080) == 1
    assert cache_frame_capacity(0.9 * frame_mb, 1920, 1080) == 0
    assert cache_frame_capacity(0.0, 1920, 1080) == 0


def test_frame_cache_keeps_the_first_frames_up_to_its_capacity():
    cache = FrameCache(3)
    for value in range(6):
        cache.add(_frame(value))
    assert [int(frame[0, 0, 0]) for frame in cache.frames] == [0, 1, 2]
    assert cache.full


def test_frame_cache_ends_its_prefix_where_memory_runs_out(capsys):
    class Frames(list):
        def append(self, frame):
            if len(self) == 2:
                raise MemoryError
            super().append(frame)

    cache = FrameCache(10)
    cache.frames = Frames()
    for value in range(5):
        cache.add(_frame(value))
    assert [int(frame[0, 0, 0]) for frame in cache.frames] == [0, 1] and cache.full and cache.capacity == 2
    assert "RAM frame cache stopped at 2 frames (MemoryError)" in capsys.readouterr().out


def _ctx(frames: int, budget_mb: int, needs_pass2_outputs: bool = True) -> SimpleNamespace:
    return SimpleNamespace(cfg=Config(pass2_cache_max_mb=budget_mb), needs_pass2_outputs=needs_pass2_outputs,
                           video=VideoInfo(path=None, fps=30.0, width=1920, height=1080, frame_count_hint=frames))


def test_new_frame_cache_is_sized_to_the_budget(capsys):
    whole = _new_frame_cache(_ctx(frames=100, budget_mb=4096))
    assert whole.capacity == 690 and whole.frames == []
    assert "[pass 1] RAM frame cache enabled (593 MiB est, 4096 MiB budget)" in capsys.readouterr().out

    prefix = _new_frame_cache(_ctx(frames=3674, budget_mb=4096))
    assert prefix.capacity == 690
    assert "[pass 1] RAM frame cache: 690 of 3674 frames fit the 4096 MiB budget" in capsys.readouterr().out

    assert _new_frame_cache(_ctx(frames=3674, budget_mb=5)) is None
    assert "[pass 1] RAM frame cache disabled (5 MiB budget holds no frame" in capsys.readouterr().out

    assert _new_frame_cache(_ctx(frames=100, budget_mb=4096, needs_pass2_outputs=False)) is None
    assert capsys.readouterr().out == ""
