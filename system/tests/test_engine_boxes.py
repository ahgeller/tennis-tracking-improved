"""tennis_tracker/engines/engine_boxes.py: the engine-output decode shared by the ball and player backends."""
import numpy as np
import pytest
import torch

from tennis_tracker.engines.engine_boxes import (
    OUTPUT_MODE_POSTPROCESSED,
    OUTPUT_MODE_RAW,
    decode_engine_boxes,
    engine_output_mode,
    nms_xyxy,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy_converts_centre_boxes_and_keeps_extra_columns():
    boxes = np.array([[10.0, 20.0, 4.0, 6.0, 0.9]], dtype=np.float32)
    out = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(out, [[8.0, 17.0, 12.0, 23.0, 0.9]])
    np.testing.assert_allclose(boxes, [[10.0, 20.0, 4.0, 6.0, 0.9]])  # the input is not modified


def test_nms_keeps_the_best_of_overlapping_boxes_in_score_order():
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],   # overlaps the first, lower score
        [50.0, 50.0, 60.0, 60.0],
    ], dtype=np.float32)
    scores = np.array([0.8, 0.6, 0.9], dtype=np.float32)
    assert nms_xyxy(boxes, scores, 0.45).tolist() == [2, 0]
    assert nms_xyxy(boxes, scores, 0.95).tolist() == [2, 0, 1]
    assert nms_xyxy(np.empty((0, 4)), np.empty((0,)), 0.45).tolist() == []


def test_output_mode_is_decided_by_the_tensor_shape():
    assert engine_output_mode(torch.zeros((1, 3, 6))) == OUTPUT_MODE_POSTPROCESSED
    assert engine_output_mode(torch.zeros((1, 5, 8))) == OUTPUT_MODE_RAW
    assert engine_output_mode(torch.zeros((1, 8, 5))) == OUTPUT_MODE_RAW
    assert engine_output_mode(torch.zeros((3, 6))) == OUTPUT_MODE_RAW


def test_postprocessed_rows_are_thresholded_filtered_and_rescaled():
    rows = torch.tensor([[
        [10.0, 20.0, 30.0, 40.0, 0.9, 0.0],
        [1.0, 1.0, 2.0, 2.0, 0.2, 0.0],      # below threshold
        [5.0, 5.0, 6.0, 6.0, 0.8, 1.0],      # another class
    ]])
    out = decode_engine_boxes(rows, (2.0, 0.5), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED, cls_filter=[0])
    assert out.dtype == np.float64 and out.shape == (1, 6)
    assert out[0].tolist() == [20.0, 10.0, 60.0, 20.0, pytest.approx(0.9), 0.0]
    both = decode_engine_boxes(rows, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED)
    assert both[:, 5].tolist() == [0.0, 1.0]
    assert decode_engine_boxes(rows, (1.0, 1.0), 0.95, 0.45, OUTPUT_MODE_POSTPROCESSED).shape == (0, 6)
    assert decode_engine_boxes(torch.zeros((1, 0, 6)), (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED).shape == (0, 6)


def test_rows_are_clipped_onto_the_frame_when_its_size_is_given():
    rows = torch.tensor([[
        [-4.0, 2.0, 30.0, 50.0, 0.9, 0.0],     # overhangs the left edge and the bottom
        [10.0, 10.0, 12.0, 12.0, 0.8, 0.0],    # inside: untouched
    ]])
    out = decode_engine_boxes(rows, (2.0, 2.0), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED, frame_size=(40, 60))
    assert out[:, :4].tolist() == [[0.0, 4.0, 39.0, 59.0], [20.0, 20.0, 24.0, 24.0]]
    loose = decode_engine_boxes(rows, (2.0, 2.0), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED)
    assert loose[0, :4].tolist() == [-8.0, 4.0, 60.0, 100.0]     # no size, no clipping


def _raw_head(anchors, channel_major=True):
    """``[1, 4 + classes, anchors]`` like a YOLO head, or its transpose.  Real
    heads have thousands of anchors, which is how the layout is told apart,
    so the given anchors are padded with empty ones."""
    width = len(anchors[0])
    padded = list(anchors) + [[0.0] * width for _ in range(width + 1)]
    p = torch.tensor(padded, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    return p if channel_major else p.transpose(1, 2)


@pytest.mark.parametrize("channel_major", [True, False])
def test_raw_head_rows_are_thresholded_suppressed_and_rescaled(channel_major):
    raw = _raw_head([
        [10.0, 10.0, 4.0, 4.0, 0.9],
        [10.5, 10.0, 4.0, 4.0, 0.7],   # overlaps the first at IoU 0.78: suppressed
        [50.0, 50.0, 2.0, 2.0, 0.3],   # below threshold
        [80.0, 20.0, 2.0, 4.0, 0.6],
    ], channel_major)
    out = decode_engine_boxes(raw, (2.0, 3.0), 0.5, 0.45, OUTPUT_MODE_RAW)
    assert out.shape == (2, 6)
    assert out[0].tolist() == [16.0, 24.0, 24.0, 36.0, pytest.approx(0.9), 0.0]   # best score first
    assert out[1].tolist() == [158.0, 54.0, 162.0, 66.0, pytest.approx(0.6), 0.0]
    assert decode_engine_boxes(raw, (1.0, 1.0), 0.5, 0.9, OUTPUT_MODE_RAW).shape == (3, 6)  # a looser NMS keeps it


def test_raw_head_takes_the_best_class_and_honours_the_class_filter():
    raw = _raw_head([
        [10.0, 10.0, 4.0, 4.0, 0.9, 0.1],   # class 0
        [40.0, 40.0, 4.0, 4.0, 0.2, 0.8],   # class 1
    ])
    out = decode_engine_boxes(raw, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW)
    assert out[:, 4].tolist() == [pytest.approx(0.9), pytest.approx(0.8)]
    assert out[:, 5].tolist() == [0.0, 1.0]
    only_people = decode_engine_boxes(raw, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW, cls_filter=[0])
    assert only_people.shape == (1, 6) and only_people[0, 5] == 0.0
    assert decode_engine_boxes(torch.zeros((5, 8)), (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW).shape == (0, 6)
    assert decode_engine_boxes(torch.zeros((1, 4, 8)), (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW).shape == (0, 6)


def _anchor_major_rows(head: torch.Tensor, conf_threshold: float, iou_threshold: float) -> np.ndarray:
    """The raw-head decode as it read the head before: every anchor a ``[cx, cy, w, h, scores...]``
    row, all of them converted, then thresholded."""
    p = head[0]
    if p.shape[0] <= p.shape[1]:
        p = p.transpose(0, 1)
    p = p.float().numpy()
    boxes = xywh_to_xyxy(p[:, :4].astype(np.float32))
    cls_scores = p[:, 4:].astype(np.float32)
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    keep = scores >= conf_threshold
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if not len(boxes):
        return np.empty((0, 6))
    order = nms_xyxy(boxes, scores, iou_threshold)
    return np.column_stack((boxes[order].astype(np.float64), scores[order].astype(np.float64),
                            cls_ids[order].astype(np.float64)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("channel_major", [True, False])
@pytest.mark.parametrize("classes", [1, 3])
def test_raw_head_decode_matches_the_anchor_major_decode_bit_for_bit(dtype, channel_major, classes):
    """Reading the head channel by channel and converting only the anchors that pass gives the
    rows the anchor-major decode gave, on the tensor or on its host copy; scores tie across
    classes (the first class wins) and sit exactly on the threshold."""
    rng = np.random.default_rng(classes)
    anchors = 400
    head = np.empty((4 + classes, anchors), np.float32)
    head[0:2] = rng.uniform(0, 640, (2, anchors))
    head[2:4] = rng.uniform(4, 120, (2, anchors))
    head[4:] = rng.choice([0.0, 0.25, 0.5, 0.625, 0.75, 0.875], (classes, anchors))
    t = torch.from_numpy(head).to(dtype).unsqueeze(0)
    if not channel_major:
        t = t.transpose(1, 2)
    expected = _anchor_major_rows(t, 0.5, 0.45)
    assert len(expected) > 5
    for preds in (t, t.numpy()):
        out = decode_engine_boxes(preds, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, expected)


def test_a_host_copy_of_postprocessed_rows_decodes_as_the_tensor():
    rows = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 0.9, 0.0], [1.0, 1.0, 2.0, 2.0, 0.2, 0.0]]], dtype=torch.float16)
    assert engine_output_mode(rows.numpy()) == OUTPUT_MODE_POSTPROCESSED
    np.testing.assert_array_equal(decode_engine_boxes(rows.numpy(), (2.0, 0.5), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED),
                                  decode_engine_boxes(rows, (2.0, 0.5), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED))
