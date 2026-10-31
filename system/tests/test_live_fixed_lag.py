"""The selector on a fixed lag (``tennis_tracker.live.fixed_lag``) and the ghost-ROI rule a live run
and its replay share (``tennis_tracker.live.evidence``), on synthetic detections."""
import dataclasses

import numpy as np
import pytest
from ball_in_play_selector import PackedMask, select_ball_in_play

from tennis_tracker.live.evidence import GhostEraser, RecordedEvidence, replay
from tennis_tracker.live.fixed_lag import FixedLagSelector, LagSettings, TrackIds
from tennis_tracker.pass1 import Pass1Result

WIDTH, HEIGHT, FPS = 640, 360, 30.0


def _rally(frames=150, gap=(60, 64)):
    """Two balls' worth of rally: a ball crossing left to right, absent for a few frames, then back
    the other way lower down, each with a matching motion mask; ``gap`` frames have nothing."""
    detections, masks = [], []
    for frame in range(frames):
        if gap[0] <= frame < gap[1]:
            detections.append([])
            masks.append(None)
            continue
        if frame < gap[0]:
            x, y = 20 + 8 * frame, 80 + 0.5 * frame
        else:
            x, y = 600 - 6 * (frame - gap[1]), 200 - 0.4 * (frame - gap[1])
        detections.append([([x - 2, y - 2, x + 2, y + 2], 0.9)])
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        mask[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] = 255
        masks.append(PackedMask.pack(mask))
    return detections, masks


def _evidence(detections, masks):
    return Pass1Result(detections=[], boost_masks=list(masks), raw_motions=list(masks),
                       player_boxes=[{} for _ in detections], court_keypoints=[None for _ in detections],
                       ghost_rois=[[] for _ in detections])


def _feed(selector, evidence, detections, on_commit=None):
    """Hand the selector one frame at a time, as pass 1 commits them; the frames each call committed."""
    commits = []
    for row in detections:
        evidence.detections.append(row)
        done = selector.advance()
        commits.append(done)
        if on_commit is not None:
            on_commit(len(evidence.detections), done)
    return commits


def _same(a, b):
    """Rows equal in what the tracking JSON keeps (track ids included)."""
    if a is None or b is None:
        return a is b
    return dataclasses.astuple(a) == dataclasses.astuple(b)


def test_a_whole_clip_delay_is_the_offline_selector_exactly():
    detections, masks = _rally()
    offline, _, _ = select_ball_in_play(detections, FPS, WIDTH, HEIGHT, boost_masks=masks, raw_motions=masks,
                                        player_boxes_by_frame=[{} for _ in detections],
                                        court_keypoints_by_frame=[None for _ in detections])
    evidence = _evidence(detections, masks)
    selector = FixedLagSelector(evidence, FPS, WIDTH, HEIGHT, LagSettings(delay=None, history=30))
    assert not any(_feed(selector, evidence, detections))
    assert selector.finish() == list(range(len(detections)))
    assert selector.runs == 1
    assert all(_same(a, b) for a, b in zip(selector.committed, offline))
    assert sum(row is not None for row in selector.committed) > 120


@pytest.mark.parametrize("delay, every", [(0, 1), (9, 1), (15, 6), (40, 10)])
def test_each_frame_is_committed_once_in_order_and_only_after_its_delay(delay, every):
    detections, masks = _rally()
    evidence = _evidence(detections, masks)
    selector = FixedLagSelector(evidence, FPS, WIDTH, HEIGHT, LagSettings(delay=delay, history=45, every=every))
    seen = []

    def check(available, done):
        for frame in done:
            assert available - frame >= delay + 1          # `delay` newer frames had arrived
        assert len(done) in (0,) or len(done) >= every
        seen.extend(done)
        if available > delay + every:                       # never more than `every - 1` frames overdue
            assert len(selector.committed) >= available - delay - every + 1

    _feed(selector, evidence, detections, check)
    seen.extend(selector.finish())
    assert seen == list(range(len(detections)))


def test_a_committed_row_never_changes():
    detections, masks = _rally()
    evidence = _evidence(detections, masks)
    selector = FixedLagSelector(evidence, FPS, WIDTH, HEIGHT, LagSettings(delay=6, history=20, every=3))
    snapshots = {}
    for row in detections:
        evidence.detections.append(row)
        for frame in selector.advance():
            snapshots[frame] = dataclasses.replace(selector.committed[frame]) if selector.committed[frame] else None
    selector.finish()
    for frame, row in snapshots.items():
        assert _same(selector.committed[frame], row)


def test_a_long_enough_lag_finds_the_ball_where_the_offline_selector_does():
    """With context on both sides of every frame, the windowed answer is the offline one on a clean
    rally: the same frames present, within a pixel."""
    detections, masks = _rally()
    offline, _, _ = select_ball_in_play(detections, FPS, WIDTH, HEIGHT, boost_masks=masks, raw_motions=masks)
    evidence = _evidence(detections, masks)
    selector = FixedLagSelector(evidence, FPS, WIDTH, HEIGHT, LagSettings(delay=30, history=60, every=5))
    _feed(selector, evidence, detections)
    selector.finish()
    for live, ref in zip(selector.committed, offline):
        assert (live is None) == (ref is None)
        if live is not None:
            assert abs(live.cx - ref.cx) < 1.0 and abs(live.cy - ref.cy) < 1.0


def test_track_ids_stay_with_a_track_across_runs():
    detections, masks = _rally()
    evidence = _evidence(detections, masks)
    selector = FixedLagSelector(evidence, FPS, WIDTH, HEIGHT, LagSettings(delay=10, history=30, every=4))
    _feed(selector, evidence, detections)
    selector.finish()
    first = {row.track_id for row in selector.committed[:55] if row is not None and row.source == "det"}
    second = {row.track_id for row in selector.committed[70:] if row is not None and row.source == "det"}
    assert len(first) == 1 and len(second) == 1 and first != second
    assert selector.runs > 20
    assert set(selector.tracks) >= first | second


def test_track_ids_follow_shared_observations():
    from ball_in_play_selector import Detection, SelectorConfig, Track
    cfg = SelectorConfig(fps=FPS, width=WIDTH, height=HEIGHT)

    def track(track_id, frames, x0):
        made = Track(track_id=track_id, cfg=cfg)
        made.observations = [Detection(f, x0 + f, 50.0, 0, 0, 1, 1, 0.9, 1.0) for f in frames]
        return made

    ids = TrackIds()
    assert ids.assign([track(0, range(0, 10), 0.0), track(1, range(3, 8), 300.0)]) == {0: 0, 1: 1}
    # The next run numbers from 0 again: the track sharing frames 5-9 keeps 0, a new one gets 2.
    assert ids.assign([track(0, range(20, 25), 500.0), track(1, range(5, 15), 0.0)]) == {1: 0, 0: 2}


def test_lag_settings_from_seconds():
    settings = LagSettings.at_fps(59.94, 0.5, 4.0, 0.1)
    assert (settings.delay, settings.history, settings.every) == (30, 240, 6)
    assert LagSettings.at_fps(30.0, None, 1.0, 0.0).delay is None
    assert LagSettings.at_fps(30.0, None, 1.0, 0.0).every == 1
    with pytest.raises(ValueError):
        LagSettings(delay=-1, history=0)


def _mask_with(boxes):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return PackedMask.pack(mask)


def test_the_ghost_eraser_clears_only_ghosts_found_since_the_last_run():
    result = Pass1Result(boost_masks=[_mask_with([(10, 10, 20, 20), (100, 100, 110, 110)]) for _ in range(3)],
                         raw_motions=[_mask_with([(10, 10, 20, 20)]) for _ in range(3)],
                         ghost_rois=[[] for _ in range(3)])
    eraser = GhostEraser(result)
    result.ghost_rois[1].append((5, 5, 25, 25))
    eraser(0, 3)
    assert result.raw_motions[1] is None                     # the only motion was the ghost's
    assert result.boost_masks[1].unpack()[105, 105] == 255 and result.boost_masks[1].unpack()[15, 15] == 0
    untouched = result.boost_masks[0]
    eraser(0, 3)
    assert result.boost_masks[0] is untouched
    result.ghost_rois[2].append((95, 95, 115, 115))
    eraser(2, 3)                                             # outside the window: frame 0 is not read
    assert result.boost_masks[2].unpack()[105, 105] == 0 and result.boost_masks[2].unpack()[15, 15] == 255


def test_a_replay_commits_what_the_live_feed_committed():
    """``evidence.replay`` adds each ghost when pass 1 found it, so it commits what the selector fed
    frame by frame with those ghosts did."""
    detections, masks = _rally()
    ghosts = [(70, 66, (300, 40, 400, 140)), (100, 90, (0, 0, 640, 360))]
    settings = LagSettings(delay=8, history=30, every=4)

    evidence = _evidence(detections, masks)
    selector = FixedLagSelector(evidence, FPS, WIDTH, HEIGHT, settings, before_run=GhostEraser(evidence))
    for frame, row in enumerate(detections):
        for found_at, ghost_frame, box in ghosts:
            if found_at == frame:
                evidence.ghost_rois[ghost_frame].append(box)
        evidence.detections.append(row)
        selector.advance()
    selector.finish()

    recorded = RecordedEvidence(FPS, WIDTH, HEIGHT, detections=list(detections), boost_masks=list(masks),
                                raw_motions=list(masks), player_boxes=[{} for _ in detections],
                                court_keypoints=[None for _ in detections], ghosts=ghosts)
    replayed = replay(recorded, settings)
    assert all(_same(a, b) for a, b in zip(replayed.committed, selector.committed))
    assert replayed.runs == selector.runs
