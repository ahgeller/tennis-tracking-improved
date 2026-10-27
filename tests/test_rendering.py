"""tennis_tracker.render.rendering: the marker sizes, the search-centre smoothing, and what each drawing
primitive puts where.  These assert structure (which pixels a marker covers, which frame gets a
call, what the HUD says) and do not depend on how this OpenCV build rasterises, unlike the
byte-exact frame hashes in tests/test_pipeline_run.py."""
from typing import List, Tuple

import cv2
import numpy as np
import pytest
from ball_in_play_selector import FrameResult

from tennis_tracker.config import Config
from tennis_tracker.render.rendering import (
    COLOR_DET,
    COLOR_INTERP,
    COLOR_MOTION,
    COLOR_OUTLINE,
    COLOR_RAW,
    COLOR_SEARCH,
    COLOR_WHITE,
    _GLOW_HALO,
    _SEARCH_CENTRE_MAX_STEP_PX,
    Canvas,
    MarkerSizer,
    SearchCentreSmoother,
    draw_glowing_marker,
    draw_hud,
    draw_raw_detections,
    draw_selected_result,
    paint_blended,
)
from tennis_tracker.render.trails import BallTrail

WIDTH, HEIGHT = 320, 180


def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)


def result(x: float, y: float, source: str, **fields) -> FrameResult:
    return FrameResult(cx=x, cy=y, conf=0.9, source=source, **fields)


@pytest.fixture
def recorder(monkeypatch):
    """Record every cv2 text and line call: (function, id of the target image, the label or
    the two end points)."""
    calls: List[Tuple[str, int, object]] = []
    real_text, real_line = cv2.putText, cv2.line

    def put_text(img, label, origin, *args, **kwargs):
        calls.append(("text", id(img), label))
        return real_text(img, label, origin, *args, **kwargs)

    def line(img, p0, p1, *args, **kwargs):
        calls.append(("line", id(img), (p0, p1)))
        return real_line(img, p0, p1, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", put_text)
    monkeypatch.setattr(cv2, "line", line)
    return calls


# --------------------------------------------------------------------------- MarkerSizer

def test_marker_sizer_follows_detector_boxes_within_its_limits():
    cfg = Config(ball_marker_min_radius=3, ball_marker_max_radius=12, ball_marker_box_scale=0.5)
    sizer = MarkerSizer(cfg)
    assert sizer.radius == 3.0 and sizer.estimate_radius == 3
    assert sizer.observe_box(10.0, 10.0) == 5                     # round(10 * 0.5), inside the limits
    assert sizer.radius == pytest.approx(0.8 * 3.0 + 0.2 * 5.0)   # the EMA of the box radius
    assert sizer.observe_box(2.0, 2.0) == 3                       # clipped to the minimum
    assert sizer.observe_box(100.0, 100.0) == 12                  # ... and the maximum
    for _ in range(100):
        sizer.observe_box(400.0, 400.0)
    assert sizer.radius == 36.0 and sizer.estimate_radius == 12   # radius capped at 3x max, estimate at max


def test_estimated_marker_shrinks_with_the_configured_maximum():
    sizer = MarkerSizer(Config(ball_marker_min_radius=3, ball_marker_max_radius=6))
    assert sizer.radius == 3.0 and sizer.estimate_radius == 3
    for _ in range(100):
        sizer.observe_box(400.0, 400.0)
    assert sizer.estimate_radius == 6                             # the motion marker follows the same ceiling


def test_a_minimum_radius_above_the_maximum_raises_the_maximum_to_it():
    """``ball_marker_min_radius`` 8 was within the maximum of 12 the markers once had; over the
    maximum of 6 ``np.clip`` answered 6 for every size, below the minimum asked for."""
    sizer = MarkerSizer(Config(ball_marker_min_radius=8, ball_marker_max_radius=6))
    assert (sizer.min_radius, sizer.max_radius) == (8, 8)
    assert sizer.radius == 8.0 and sizer.estimate_radius == 8
    assert sizer.observe_box(2.0, 2.0) == 8 and sizer.observe_box(400.0, 400.0) == 8
    for _ in range(100):
        sizer.observe_box(400.0, 400.0)
    assert sizer.radius == 24.0 and sizer.estimate_radius == 8                 # the cap follows the raised maximum


# --------------------------------------------------------------------------- SearchCentreSmoother

def test_search_centre_smoother_caps_a_jump_only_between_consecutive_frames():
    smoother = SearchCentreSmoother()
    assert smoother.smooth(10, 100, 100) == (100, 100)
    scx, scy = smoother.smooth(11, 100 + 300, 100)                 # a 300 px jump on the next frame
    assert (scx, scy) == (100 + int(_SEARCH_CENTRE_MAX_STEP_PX), 100)
    assert smoother.smooth(13, 400, 100) == (400, 100)             # a skipped frame: no cap
    smoother.reset()
    assert smoother.smooth(14, 10, 10) == (10, 10)


# --------------------------------------------------------------------------- paint_blended

def _between(pixel, a, b) -> bool:
    """``pixel`` is a blend of the colours ``a`` and ``b``: every channel within their range, and
    the pixel neither colour itself."""
    pixel = tuple(int(v) for v in pixel)
    return pixel not in (tuple(a), tuple(b)) and all(min(x, y) <= v <= max(x, y) for v, x, y in zip(pixel, a, b))


def test_paint_blended_shows_the_painting_at_its_opacity_within_the_box():
    frame = np.full((HEIGHT, WIDTH, 3), 100, np.uint8)
    seen = []

    def paint(img, offset):
        seen.append((img.shape[:2], offset))
        cv2.rectangle(img, (10 - offset[0], 10 - offset[1]), (29 - offset[0], 29 - offset[1]), COLOR_WHITE, -1)

    paint_blended(frame, (5, 5, 35, 35), 1.0, paint)
    assert seen == [((HEIGHT, WIDTH), (0, 0))] and tuple(frame[20, 20]) == COLOR_WHITE   # straight onto the frame
    frame[:] = 100
    paint_blended(frame, (5, 5, 35, 35), 0.5, paint)
    assert seen[-1] == ((30, 30), (5, 5))                          # a copy of the box, offset for the painter
    assert tuple(frame[20, 20]) == (178, 178, 178)                 # half way from 100 to 255 (rounded)
    assert tuple(frame[5, 5]) == (100, 100, 100) and tuple(frame[40, 40]) == (100, 100, 100)
    # A box over the frame's edge is clipped; one off the frame paints nothing.
    frame[:] = 100
    paint_blended(frame, (-20, -20, 15, 15), 0.5, paint)
    assert seen[-1] == ((15, 15), (0, 0)) and tuple(frame[12, 12]) == (178, 178, 178)
    paint_blended(frame, (WIDTH, 0, WIDTH + 10, 10), 0.5, paint)
    assert len(seen) == 3


# --------------------------------------------------------------------------- draw_selected_result

def test_glowing_marker_is_a_solid_disc_in_a_halo_that_fades_outward():
    frame = np.full((HEIGHT, WIDTH, 3), 60, np.uint8)
    draw_glowing_marker(frame, (100, 80), 4, COLOR_DET)
    assert tuple(frame[80, 100]) == COLOR_DET and tuple(frame[80, 103]) == COLOR_DET   # the disc, to its radius
    halo = [tuple(frame[80, 100 + dx]) for dx in (5, 7, 9)]        # inside each of the three halo discs
    assert all(_between(px, (60, 60, 60), COLOR_DET) for px in halo)
    assert halo[0][1] > halo[1][1] > halo[2][1]                   # the glow fades outward
    reach = int(round(4 * _GLOW_HALO[0][0]))
    assert tuple(frame[80, 100 + reach + 2]) == (60, 60, 60)      # nothing past 2.5 radii


def test_detection_marker_glows_on_the_main_frame_and_keeps_its_box_and_ring_on_the_debug_frame(recorder):
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    res = result(100.0, 80.0, "det", bbox=(94.0, 74.0, 106.0, 86.0))
    draw_selected_result(canvas, res, 5, MarkerSizer(Config()), SearchCentreSmoother(), draw_search_regions=False)
    for frame in (main, debug):
        assert tuple(frame[80, 100]) == COLOR_DET                  # the filled marker at the centre
        assert not frame[150:, :].any()                            # nothing far from the ball
    assert tuple(debug[80, 103]) == COLOR_DET                      # ... out to its 3 px radius
    assert tuple(debug[80, 104]) == COLOR_OUTLINE                  # the one-pixel halo ring around it
    assert tuple(debug[74, 94]) == COLOR_DET                       # the box's top-left corner, one pixel wide
    assert tuple(debug[73, 94]) == COLOR_OUTLINE                   # inside the box's halo
    assert tuple(debug[86, 100]) == COLOR_WHITE                    # the emphasised bottom edge
    assert tuple(debug[88, 100]) == (0, 0, 0)                      # ... and nothing below the halo
    # The main frame: a glow in place of the ring, and neither the box nor the label.
    assert _between(main[80, 105], (0, 0, 0), COLOR_DET)
    assert not (main == np.array(COLOR_OUTLINE, np.uint8)).all(axis=2).any()
    assert not (main == np.array(COLOR_WHITE, np.uint8)).all(axis=2).any()
    assert [target for kind, target, _ in recorder if kind == "text"] == [id(debug)]


def test_motion_marker_draws_its_search_region_on_the_debug_frame_only(recorder):
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    res = result(100.0, 80.0, "motion", search_cx=120.0, search_cy=90.0, search_radius=20.0)
    draw_selected_result(canvas, res, 5, MarkerSizer(Config()), SearchCentreSmoother(), draw_search_regions=False)
    assert tuple(main[80, 100]) == COLOR_MOTION and tuple(debug[82, 98]) == COLOR_MOTION
    assert tuple(debug[80, 100]) == COLOR_SEARCH                   # the search line ends on the marker
    assert tuple(debug[90, 140]) == COLOR_SEARCH                   # the search circle's right edge
    assert not any(tuple(main[90, x]) == COLOR_SEARCH for x in range(WIDTH))
    search_lines = [call for call in recorder if call[0] == "line" and call[2] == ((120, 90), (100, 80))]
    assert [target for _, target, _ in search_lines] == [id(debug)]
    # With draw_search_regions the main frame gets the region too.
    main, debug = blank(), blank()
    draw_selected_result(Canvas(main, debug), res, 6, MarkerSizer(Config()), SearchCentreSmoother(), True)
    assert tuple(main[90, 140]) == COLOR_SEARCH


def test_interpolated_marker_is_the_interp_colour_with_its_label_on_the_debug_frame(recorder):
    main, debug = blank(), blank()
    draw_selected_result(Canvas(main, debug), result(50.0, 50.0, "interp"), 1, MarkerSizer(Config()),
                         SearchCentreSmoother(), False)
    assert tuple(main[50, 50]) == COLOR_INTERP and tuple(debug[50, 50]) == COLOR_INTERP
    assert [(target, label) for kind, target, label in recorder if kind == "text"] == [(id(debug), "PHY")]


# --------------------------------------------------------------------------- HUD and raw detections

def test_hud_names_the_frame_source_and_confidence_and_the_track_on_the_debug_frame(recorder):
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    track = type("Track", (), {"track_id": 7})()
    draw_hud(canvas, 3, result(1.0, 1.0, "det"), track)
    texts = [(target, label) for kind, target, label in recorder if kind == "text"]
    assert (id(main), "F3") in texts and (id(main), "ID:7 DET c=0.90") in texts
    assert (id(debug), "track=7") in texts and (id(main), "track=7") not in texts
    assert main[:40, :200].any() and not main[40:, :].any()        # the text sits in the top-left HUD strip
    recorder.clear()
    draw_hud(Canvas(blank(), None), 4, None, None)
    assert [label for _, _, label in recorder] == ["F4"]           # no result: the frame number alone


def test_raw_detections_are_boxes_on_the_debug_frame_only():
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    draw_raw_detections(canvas.debug_only, [([20.0, 30.0, 40.0, 50.0], 0.7)])
    assert tuple(debug[30, 30]) == COLOR_RAW and tuple(debug[40, 20]) == COLOR_RAW
    assert not main.any()


# --------------------------------------------------------------------------- trail segments

def test_ball_trail_draws_one_outlined_segment_per_consecutive_pair(recorder):
    trail = BallTrail(Config(), 30.0, WIDTH, HEIGHT)
    for i in range(5):
        trail.advance(result(50.0 + 10 * i, 60.0, "det"), None)
    trail.advance(None, None)                                       # a break
    trail.advance(result(200.0, 60.0, "det"), None)
    trail.advance(result(210.0, 60.0, "det"), None)
    main, debug = blank(), blank()
    trail.draw(Canvas(main, debug), draw_main=True)
    lines = [call for call in recorder if call[0] == "line"]
    # Four segments before the break and one after, each an outline plus a coloured line, on the
    # debug frame; the main frame's comet is painted on overlays of it, band by band.
    assert len([call for call in lines if call[1] == id(debug)]) == 2 * 5
    assert len([call for call in lines if call[1] == id(main)]) < 2 * 5
    assert not any(p1[0] - p0[0] == 200 - 90 for _, _, (p0, p1) in lines)   # nothing bridges the break
