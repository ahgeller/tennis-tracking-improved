"""tennis_tracker.render.trails: the chosen ball's trail breaks where it should and draws as a comet,
the display-path trail keeps exact and filled points apart, and the candidate overlay draws
every track. Synthetic frames and results only."""
import timeit
from types import SimpleNamespace

import numpy as np
from ball_in_play_selector import Detection, FrameResult

from tennis_tracker.config import Config
from tennis_tracker.render.rendering import COLOR_OUTLINE, Canvas
from tennis_tracker.render.trails import (
    CANDIDATE_TRAIL_LEN, COMET_HEAD_THICKNESS, COMET_TAIL_ALPHA, COMET_TAIL_THICKNESS, DEBUG_TRACK_STYLE,
    FADE_BANDS, LIGHT_TRACK_STYLE, TRAIL_MIN_LEN, TRAIL_SECONDS, _TRAIL_DIRECTION_BRIDGE_SCALE, BallTrail,
    CandidateTrackTrails, DisplayPathTrail, Stroke, _reversal_breaks, _segment_source, _SegmentLimits,
    _trail_jump_fracs, _transition_too_far, draw_fading_strokes, track_color,
)

WIDTH, HEIGHT, FPS = 640, 360, 30.0


def result(cx, cy, source="det", conf=0.9):
    return FrameResult(cx=cx, cy=cy, conf=conf, source=source)


def blank():
    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)


GREY = (90, 90, 90)


def grey():
    return np.full((HEIGHT, WIDTH, 3), GREY, np.uint8)


def _between(pixel, a, b) -> bool:
    """``pixel`` is a blend of the colours ``a`` and ``b``: every channel within their range, and
    the pixel neither colour itself."""
    pixel = tuple(int(v) for v in pixel)
    return pixel not in (tuple(a), tuple(b)) and all(min(x, y) <= v <= max(x, y) for v, x, y in zip(pixel, a, b))


def _painted_rows(frame, column, min_contrast: int = 24) -> int:
    """How many pixels of ``column`` stand out from the grey background by ``min_contrast`` in
    some channel: a line's height there, without the faint anti-aliased fringe."""
    contrast = np.abs(frame[:, column].astype(np.int16) - np.array(GREY, np.int16)).max(axis=1)
    return int((contrast >= min_contrast).sum())


# --------------------------------------------------------------------------- BallTrail

def test_trail_length_follows_the_frame_rate_with_a_floor():
    assert BallTrail(Config(), 60.0, WIDTH, HEIGHT).max_len == max(TRAIL_MIN_LEN, round(60.0 * TRAIL_SECONDS))
    assert BallTrail(Config(), 10.0, WIDTH, HEIGHT).max_len == TRAIL_MIN_LEN


def test_a_steady_path_is_one_segment_and_a_jump_breaks_it():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    for i in range(4):
        trail.advance(result(100.0 + 8 * i, 200.0), None)
    assert None not in trail.points and len(trail.points) == 4
    trail.advance(result(500.0, 60.0), None)              # across the frame in one step
    assert trail.points[-2] is None and trail.points[-1][:2] == (500, 60)


def test_losing_the_ball_hard_breaks_the_segment_and_the_trail_is_bounded():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    trail.advance(result(100.0, 100.0), None)
    trail.advance(None, None)
    trail.advance(None, None)
    trail.advance(result(104.0, 100.0), None)
    # One break per lost frame, and recovering from a lost run breaks once more before the point.
    assert [p is None for p in trail.points] == [False, True, True, True, False]
    for i in range(3 * trail.max_len):
        trail.advance(result(104.0 + i, 100.0), None)
    assert len(trail.points) == trail.max_len


def test_source_specific_jump_fractions():
    cfg = Config()
    assert _trail_jump_fracs("det", cfg) == (cfg.trail_hard_switch_x_frac, cfg.trail_hard_switch_y_frac)
    assert _trail_jump_fracs("motion", cfg) <= _trail_jump_fracs("det", cfg)
    assert _trail_jump_fracs("interp", cfg) == _trail_jump_fracs("interp", cfg)
    assert _trail_jump_fracs(None, cfg) == _trail_jump_fracs("other", cfg)


def test_drawing_paints_the_debug_frame_and_the_main_frame_only_when_asked():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    for i in range(4):
        trail.advance(result(100.0 + 8 * i, 200.0), None)
    main, debug = blank(), blank()
    trail.draw(Canvas(main, debug), draw_main=False)
    assert debug.any() and not main.any()
    trail.draw(Canvas(main, debug), draw_main=True)
    assert main.any()


def _is_green(pixel) -> bool:
    """The detector's trail colour (green) against motion's orange, whatever the age fade."""
    b, g, r = (int(v) for v in pixel)
    return g > r and g > b


def test_a_one_frame_motion_fill_between_detections_takes_the_detector_colour():
    points = [(0, 0, "det", 0.9), (8, 0, "motion", 0.9), (16, 0, "det", 0.9)]
    assert _segment_source(points, 1) == "det" and _segment_source(points, 2) == "det"
    run = [(0, 0, "det", 0.9), (8, 0, "motion", 0.9), (16, 0, "motion", 0.9), (24, 0, "det", 0.9)]
    assert [_segment_source(run, i) for i in (1, 2, 3)] == ["motion", "motion", "det"]
    assert _segment_source([(0, 0, "det", 0.9), (8, 0, "motion", 0.9)], 1) == "motion"     # the trail's last point
    assert _segment_source([None, (8, 0, "motion", 0.9), (16, 0, "det", 0.9)], 1) == "motion"  # after a break
    assert _segment_source([(0, 0, "det", 0.9), (8, 0, "interp", 0.9), (16, 0, "det", 0.9)], 1) == "interp"


def test_drawn_trail_colours_follow_the_segment_source():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    for x, source in ((100.0, "det"), (108.0, "motion"), (116.0, "det"), (124.0, "motion"), (132.0, "motion"),
                      (140.0, "det")):
        trail.advance(result(x, 200.0, source), None)
    main = blank()
    trail.draw(Canvas(main, None), draw_main=True)
    assert _is_green(main[200, 104])            # det -> one motion fill -> det: drawn as the detector's track
    assert _is_green(main[200, 112])
    assert not _is_green(main[200, 120])        # the first of two motion points: orange
    assert not _is_green(main[200, 128])
    assert _is_green(main[200, 136])            # back on a detection


def _straight_trail(fps: float, frame_w: int = WIDTH, frame_h: int = HEIGHT) -> BallTrail:
    """A full-length detector trail along one row, the head at the right."""
    trail = BallTrail(Config(), fps, frame_w, frame_h)
    for i in range(trail.max_len):
        trail.advance(result(40.0 + 8 * i, 200.0), None)
    return trail


def test_the_main_frame_comet_tapers_and_fades_from_head_to_tail():
    trail = _straight_trail(fps=60.0)                                    # 15 points: 40 to 152
    main = grey()
    trail.draw(Canvas(main, None), draw_main=True)
    head_x, tail_x = 40 + 8 * (trail.max_len - 1) - 4, 44
    assert _painted_rows(main, head_x) > _painted_rows(main, tail_x)    # thicker at the head ...
    assert _painted_rows(main, head_x) >= COMET_HEAD_THICKNESS
    assert _painted_rows(main, tail_x) <= COMET_TAIL_THICKNESS + 1       # ... a 1 px line at the tail
    green = (40, 255, 40)
    assert tuple(main[200, head_x]) == green                             # the head at full strength
    assert _between(main[200, tail_x], GREY, green)                      # the tail blended toward the frame
    assert _is_green(main[200, tail_x])
    assert not (main == np.array(COLOR_OUTLINE, np.uint8)).all(axis=2).any()   # the outline is blended too


def test_the_debug_frame_trail_is_the_thin_line_it_was():
    trail = _straight_trail(fps=60.0)
    main, debug = grey(), grey()
    trail.draw(Canvas(main, debug), draw_main=True)
    head_x = 40 + 8 * (trail.max_len - 1) - 4
    assert _painted_rows(debug, head_x) < _painted_rows(main, head_x)
    assert (debug == np.array(COLOR_OUTLINE, np.uint8)).all(axis=2).any()   # its outline is solid


def test_fading_strokes_blend_by_band_and_colour_the_joint_between_bands():
    frame = blank()
    color = (0, 0, 200)
    strokes = [Stroke((20, 50), (60, 50), color, 2, 0.0), Stroke((60, 50), (100, 50), color, 2, 1.0)]
    draw_fading_strokes(frame, strokes, head_alpha=1.0, tail_alpha=0.25, outline=True)
    assert tuple(frame[50, 80]) == color                                 # the head band, painted straight on
    assert tuple(frame[50, 40]) == (0, 0, 50)                            # the tail band at a quarter
    assert tuple(frame[50, 60]) == color                                 # the joint is colour, not the head's outline
    assert tuple(frame[50, 52]) == (0, 0, 50)                            # ... and the patch stops behind the joint
    assert FADE_BANDS >= 3 and COMET_TAIL_ALPHA == 0.25


def test_drawing_a_sixty_point_comet_on_a_1080p_frame_is_fast():
    trail = _straight_trail(fps=240.0, frame_w=1920, frame_h=1080)
    assert trail.max_len == 60
    frame = np.full((1080, 1920, 3), GREY, np.uint8)
    canvas = Canvas(frame, None)
    seconds = min(timeit.repeat(lambda: trail.draw(canvas, draw_main=True), number=20, repeat=3)) / 20
    assert seconds < 0.005, f"{seconds * 1e3:.2f} ms per draw"


# --------------------------------------------------------------------------- DisplayPathTrail

def test_display_path_trail_rounds_points_and_records_gaps_once():
    trail = DisplayPathTrail(FPS, WIDTH)
    trail.advance((10.4, 20.6, True))
    trail.advance(None)
    trail.advance(None)
    trail.advance((30.0, 40.0, False))
    assert trail.points == [(10, 21, True), None, (30, 40, False)]


def test_display_path_trail_draws_exact_segments_brighter_than_fills():
    trail = DisplayPathTrail(FPS, WIDTH)
    for x in (100.0, 110.0, 120.0):
        trail.advance((x, 100.0, True))
    exact = blank()
    trail.draw(exact, (120.0, 100.0, True))
    filled = DisplayPathTrail(FPS, WIDTH)
    for x in (100.0, 110.0, 120.0):
        filled.advance((x, 100.0, False))
    soft = blank()
    filled.draw(soft, (120.0, 100.0, False))
    assert exact.any() and soft.any()
    assert int(exact.sum()) > int(soft.sum())        # thicker, whiter line and a bigger dot


def test_display_path_trail_skips_a_segment_that_spans_the_frame():
    trail = DisplayPathTrail(FPS, WIDTH)
    trail.advance((10.0, 100.0, True))
    trail.advance((WIDTH - 10.0, 100.0, True))
    debug = blank()
    trail.draw(debug, None)
    assert not debug[100, WIDTH // 2].any()


# --------------------------------------------------------------------------- CandidateTrackTrails

def det(frame, cx, cy):
    return Detection(frame, cx, cy, cx - 4, cy - 4, cx + 4, cy + 4, 0.9, 64.0, False)


def track(track_id, score, *observations):
    return SimpleNamespace(track_id=track_id, score=score, observations=list(observations))


def test_track_colour_is_stable_and_never_dark():
    assert track_color(7) == track_color(7)
    assert all(max(track_color(i)) >= 100 for i in range(50))


def test_candidate_trails_follow_each_track_and_mark_the_chosen_one():
    a = track(1, 10.0, det(0, 50.0, 50.0), det(1, 60.0, 50.0), det(3, 80.0, 50.0))
    b = track(2, 30.0, det(0, 300.0, 200.0))
    trails = CandidateTrackTrails([a, b], chosen_track=b)
    assert [int(t.track_id) for t in trails.tracks] == [2, 1]        # best score first
    assert trails.chosen_id == 2
    debug = blank()
    for frame in range(4):
        trails.advance_and_draw(debug, frame)
    assert trails.trails[1] == [(50.0, 50.0), (60.0, 50.0), None, (80.0, 50.0)]
    assert trails.trails[2] == [(300.0, 200.0), None]
    assert debug[50, 55].any() and debug[200, 300].any()


def test_candidate_trails_are_bounded():
    long = track(1, 1.0, *(det(f, 10.0 + f, 10.0) for f in range(CANDIDATE_TRAIL_LEN + 20)))
    trails = CandidateTrackTrails([long], chosen_track=None)
    debug = blank()
    for frame in range(CANDIDATE_TRAIL_LEN + 20):
        trails.advance_and_draw(debug, frame)
    assert len(trails.trails[1]) == CANDIDATE_TRAIL_LEN


def _painted(frame, color):
    """Pixels of ``frame`` that hold exactly ``color``."""
    return (frame == np.array(color, np.uint8)).all(axis=2)


def test_advance_once_then_draw_on_two_targets_at_two_weights():
    """One ``advance`` per frame feeds any number of ``draw`` calls: the debug frame at the full
    weight and the tracking frame at the light one hold the same trails."""
    a = track(1, 10.0, det(0, 50.0, 50.0), det(1, 60.0, 50.0), det(2, 70.0, 50.0))
    b = track(2, 30.0, det(0, 300.0, 200.0), det(1, 300.0, 210.0), det(2, 300.0, 220.0))
    trails = CandidateTrackTrails([a, b], chosen_track=b)
    debug, main = blank(), blank()
    for frame in range(3):
        trails.advance(frame)
        trails.draw(debug, frame, DEBUG_TRACK_STYLE)
        trails.draw(main, frame, LIGHT_TRACK_STYLE)
    assert trails.trails[1] == [(50.0, 50.0), (60.0, 50.0), (70.0, 50.0)]      # advanced once per frame
    for tid, (x, y) in ((1, (70, 50)), (2, (300, 220))):
        for frame in (debug, main):
            assert _painted(frame, track_color(tid))[y - 3:y + 4, x - 3:x + 4].any()
    # The light layer is lighter: fewer pixels in each track's colour (no label, a 2 px ring,
    # the chosen track drawn like the others), and the debug frame's filled chosen dot is a ring.
    for tid in (1, 2):
        assert _painted(main, track_color(tid)).sum() < _painted(debug, track_color(tid)).sum()
    assert _painted(debug, track_color(2))[220, 300]
    assert not _painted(main, track_color(2))[220, 300]
    # The label's text sits right and below the dot on the debug frame only.
    label_region = (slice(222, 240), slice(306, 360))
    assert _painted(debug, track_color(2))[label_region].any()
    assert not _painted(main, track_color(2))[label_region].any()


def test_the_debug_style_is_the_default_of_advance_and_draw():
    a = track(1, 10.0, det(0, 50.0, 50.0))
    by_default, explicit = blank(), blank()
    CandidateTrackTrails([a], chosen_track=a).advance_and_draw(by_default, 0)
    CandidateTrackTrails([a], chosen_track=a).advance_and_draw(explicit, 0, DEBUG_TRACK_STYLE)
    assert np.array_equal(by_default, explicit)
    assert _painted(by_default, track_color(1))[50, 50]                        # the chosen dot is filled


# --------------------------------------------------------------------------- the break rules

LIMITS = _SegmentLimits(hard_x_px=40.0, hard_y_px=30.0, bridge_px=20.0, transition_px=10.0)


def test_segment_limits_scale_with_the_court_and_never_drop_below_their_floors():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    without_court = trail._segment_limits("det", "det", None)
    assert without_court.hard_x_px >= 34.0 and without_court.hard_y_px >= 26.0
    assert without_court.bridge_px >= 18.0 and without_court.transition_px >= 14.0
    assert without_court.bridge_px > without_court.transition_px


def test_a_source_transition_bridges_a_short_distance_only():
    assert not _transition_too_far(8.0, "det", "interp", LIMITS)        # within the interp->det distance
    assert _transition_too_far(15.0, "det", "interp", LIMITS)           # interp->det is held tighter than the bridge
    assert not _transition_too_far(15.0, "interp", "det", LIMITS)       # det->interp only minds the bridge
    assert _transition_too_far(25.0, "interp", "det", LIMITS)


def test_reversals_break_only_confident_segments():
    assert _reversal_breaks("det", "det", 5.0, LIMITS)
    assert not _reversal_breaks("interp", "det", 5.0, LIMITS) and not _reversal_breaks("det", "interp", 5.0, LIMITS)
    assert not _reversal_breaks("det", "motion", LIMITS.bridge_px * _TRAIL_DIRECTION_BRIDGE_SCALE, LIMITS)
    assert _reversal_breaks("det", "motion", LIMITS.bridge_px * _TRAIL_DIRECTION_BRIDGE_SCALE + 1.0, LIMITS)
