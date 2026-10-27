"""finetune/data/detector_metrics.py: the (label, prediction) metric and the 1080p scale factors."""
import math

from finetune.data.detector_metrics import HIT_PX, WRONG_PX, DetectorMetrics, scale_for_height, scale_for_width


def test_scale_factors_take_a_frame_to_1080p():
    assert scale_for_width(1280) == 1.5
    assert scale_for_height(720) == 1.5
    assert scale_for_width(1920) == 1.0


def test_error_exactly_at_hit_px_is_a_hit():
    stats = DetectorMetrics.of([((0.0, 0.0), (HIT_PX, 0.0))], 1.0)
    assert stats.visible == 1
    assert stats.recall == 1.0
    assert stats.wrong == 0.0
    assert stats.median_px == HIT_PX


def test_error_between_hit_and_wrong_is_neither():
    stats = DetectorMetrics.of([((0.0, 0.0), (WRONG_PX, 0.0))], 1.0)
    assert stats.visible == 1
    assert stats.recall == 0.0
    assert stats.wrong == 0.0


def test_error_just_past_wrong_px_is_a_wrong_object():
    stats = DetectorMetrics.of([((0.0, 0.0), (WRONG_PX + 0.01, 0.0))], 1.0)
    assert stats.wrong == 1.0
    assert stats.recall == 0.0


def test_scale_multiplies_the_error():
    assert DetectorMetrics.of([((0.0, 0.0), (8.0, 0.0))], 1.0).recall == 1.0      # 8 px at 1080p: a hit
    assert DetectorMetrics.of([((0.0, 0.0), (8.0, 0.0))], 1.5).recall == 0.0      # 12 px at 1080p: not


def test_prediction_on_an_invisible_ball_is_a_false_alarm():
    stats = DetectorMetrics.of([(None, (5.0, 5.0))], 1.0)
    assert stats.false_alarm == 1.0
    assert stats.invisible == 1
    assert stats.visible == 0


def test_no_prediction_on_an_invisible_ball_is_quiet():
    stats = DetectorMetrics.of([(None, None)], 1.0)
    assert stats.false_alarm == 0.0
    assert stats.invisible == 1


def test_no_prediction_on_a_visible_ball_is_a_miss():
    stats = DetectorMetrics.of([((1.0, 1.0), None)], 1.0)
    assert stats.miss == 1.0
    assert stats.visible == 1
    assert math.isnan(stats.median_px)


def test_empty_input_has_nan_distances_and_zero_counts():
    stats = DetectorMetrics.of([], 1.0)
    assert math.isnan(stats.median_px)
    assert math.isnan(stats.p90_px)
    assert stats.visible == 0 and stats.invisible == 0
    assert stats.recall == 0.0 and stats.false_alarm == 0.0


def test_percentiles_come_from_the_sorted_errors():
    pairs = [((0.0, 0.0), (float(e), 0.0)) for e in (9, 1, 5, 3, 7)]
    stats = DetectorMetrics.of(pairs, 1.0)
    assert stats.median_px == 5.0
    assert stats.p90_px == 9.0


def test_the_metric_is_a_named_record_with_one_vocabulary():
    stats = DetectorMetrics.of([((0.0, 0.0), (1.0, 0.0)), (None, None)], 1.0)
    assert isinstance(stats, DetectorMetrics)
    assert list(stats._asdict()) == ["visible", "recall", "wrong", "miss", "invisible", "false_alarm", "median_px",
                                      "p90_px", "bridged"]
    assert stats._asdict()["wrong"] == 0.0 and "wrong_obj" not in stats._asdict()
    # A record written before the metric gained ``bridged`` (an older winner.json, checkpoint or
    # results.csv row) still builds: the field defaults.
    older = {name: value for name, value in stats._asdict().items() if name != "bridged"}
    assert DetectorMetrics(**older).bridged == 0.0


VISIBLE_RUN = [((0.0, 100.0), (0.0, 100.0)), ((100.0, 100.0), (100.0, 100.0))]    # the ball moving along y=100


def _occlusion(run, flanked=True):
    """A visible label, `run` invisible ones with a prediction each, and (when `flanked`) a visible
    label 400 px further along the same line."""
    pairs = [VISIBLE_RUN[0]] + [(None, prediction) for prediction in run]
    if flanked:
        pairs.append(((400.0, 100.0), (400.0, 100.0)))
    return pairs


def test_an_output_through_a_short_occlusion_on_the_balls_line_is_bridged():
    stats = DetectorMetrics.of(_occlusion([(100.0, 104.0), (200.0, 96.0), (300.0, 109.0)]), 1.0)
    assert stats.invisible == 3 and stats.false_alarm == 1.0                         # the total keeps every alarm
    assert stats.bridged == 1.0 and stats.bridged_count == 3 and stats.false_alarm_count == 3
    # 11 px off the line at 1080p is the ball somewhere else, so is a point within 10 native px of
    # it on a 720p clip, where the scale makes that 15.
    assert DetectorMetrics.of(_occlusion([(100.0, 111.0)]), 1.0).bridged == 0.0
    assert DetectorMetrics.of(_occlusion([(100.0, 108.0)]), 1.5).bridged == 0.0
    assert DetectorMetrics.of(_occlusion([(100.0, 106.0)]), 1.5).bridged == 1.0


def test_a_long_or_unflanked_run_of_invisible_labels_is_not_an_occlusion():
    on_line = [(100.0 + 50.0 * index, 100.0) for index in range(4)]
    assert DetectorMetrics.of(_occlusion(on_line), 1.0).bridged == 0.0               # four frames: dead time, not an occlusion
    assert DetectorMetrics.of(_occlusion(on_line[:3], flanked=False), 1.0).bridged == 0.0
    assert DetectorMetrics.of(_occlusion(on_line[:3])[1:], 1.0).bridged == 0.0       # no visible label before the run
    assert DetectorMetrics.of(_occlusion([(100.0, 100.0), None]), 1.0).bridged == 0.5  # a silent frame inside the run is quiet


def test_clips_pool_without_one_clips_labels_flanking_the_next():
    first = [VISIBLE_RUN[0], (None, (100.0, 100.0))]                     # ends in an invisible label
    second = [((200.0, 100.0), (200.0, 100.0))]                          # begins with a visible one on the same line
    assert DetectorMetrics.of(first + second, 1.0).bridged == 1.0        # one sequence: the alarm reads as bridged
    pooled = DetectorMetrics.of_clips([first, second], 1.0)
    assert pooled.bridged == 0.0 and pooled.false_alarm == 1.0 and pooled.visible == 2
    assert DetectorMetrics.of_clips([], 1.0).visible == 0 and math.isnan(DetectorMetrics.of_clips([], 1.0).median_px)
