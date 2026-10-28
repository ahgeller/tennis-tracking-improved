"""tennis_tracker.motion on synthetic frames, on the CPU."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ball_in_play_selector import PackedMask

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import (
    MASKS_ONLY, BackgroundModel, DetectorInputRequest, MotionMasks, _clean_raw_motion_cuda, _clean_raw_motion_mask,
    _raw_motion_cuda, filter_boost_mask, preprocess_frame_cuda,
)
from tests._helpers import refine_raw_motion_temporal_cpu

CPU = torch.device("cpu")


def _zero_background(cfg, h, w):
    background = BackgroundModel(cfg, CPU)
    background.mean = torch.zeros(2, h, w)
    background.var = torch.zeros(2, h, w)
    return background


def test_empty_motion_clears_reused_boost_buffer():
    buffers = SimpleNamespace(boost_mask_u8=np.full((10, 10), 255, np.uint8))
    result = filter_boost_mask(np.zeros((10, 10), np.uint8), 0, 600, Config(), buffers=buffers)
    assert not np.any(result)


def test_motion_retains_non_yellow_evidence():
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[10:20, 10:20, 2] = 255
    cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=0,
                 motion_thresh=1., motion_v_min=0.)
    window = FrameWindow(CPU, True, frame, None)
    background = _zero_background(cfg, 32, 32)
    raw = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY,
                                protect_mask=torch.zeros(32, 32, dtype=torch.bool)).raw_motion
    assert np.count_nonzero(raw) == 100
    expected = refine_raw_motion_temporal_cpu(raw, None, frame, None, cfg)
    np.testing.assert_array_equal(expected, raw)


def test_court_side_protect_mask_covers_the_outside_of_each_sideline():
    """A trapezoid court seen from the baseline: the guides run one alley width outside each
    sideline, and the mask keeps what lies beyond them (and nothing when there is no court)."""
    from tennis_tracker.motion import build_court_side_protect_mask
    keypoints = [0.0] * 28
    for index, (x, y) in {0: (120.0, 20.0), 1: (200.0, 20.0), 2: (60.0, 170.0), 3: (260.0, 170.0),
                          5: (80.0, 170.0), 7: (240.0, 170.0)}.items():
        keypoints[2 * index], keypoints[2 * index + 1] = x, y
    mask = build_court_side_protect_mask(180, 320, keypoints)
    assert mask is not None and mask.shape == (180, 320) and mask.dtype == bool
    assert mask[170, 5] and mask[170, 315]                 # the outer edges near the baseline
    assert not mask[170, 160] and not mask[100, 160]       # the court itself
    assert not mask[170, 70] and not mask[170, 250]        # inside the guides' alley offset
    assert build_court_side_protect_mask(180, 320, None) is None
    assert build_court_side_protect_mask(180, 320, [0.0] * 28) is None


# --------------------------------------------------------------------------- the GPU morphology

@pytest.mark.parametrize("close_size,open_size", [(2, 0), (3, 0), (4, 0), (0, 2), (0, 3), (2, 3), (3, 2), (5, 4)])
def test_gpu_cleaning_matches_opencv_morphology_pixel_for_pixel(close_size, open_size):
    """The raw-mask closing and opening on the GPU reproduce OpenCV's, anchor and borders
    included: sparse noise, blobs and set pixels on every edge of the mask."""
    cfg = Config(motion_raw_close_size=close_size, motion_raw_open_size=open_size)
    rng = np.random.default_rng(close_size * 10 + open_size)
    for density in (0.05, 0.3, 0.6):
        mask = (rng.random((37, 53)) < density).astype(np.uint8) * 255
        mask[0, :] = 255
        mask[:, -1] = 255
        expected = _clean_raw_motion_mask(mask, cfg)
        got = _clean_raw_motion_cuda(torch.from_numpy(mask > 0), cfg)
        np.testing.assert_array_equal(got.numpy().astype(np.uint8) * 255, expected)


def test_gpu_cleaning_is_a_no_op_without_morphology():
    mask = torch.rand(8, 8) > 0.5
    assert _clean_raw_motion_cuda(mask, Config(motion_raw_close_size=1, motion_raw_open_size=0)) is mask


# --------------------------------------------------------------------------- the host side of the masks

def test_motion_masks_pack_the_union_box_of_rois_and_keep_ball_sized_blobs():
    """ROI mode: each box is filtered on its own, the masks are assembled in the box that holds
    them all, and the packed masks store that box, as ``PackedMask.pack(full, roi=union)`` does."""
    cfg = Config(boost_min_blob_area=1, boost_max_blob_area=600, motion_raw_close_size=0)
    full = torch.zeros(60, 80, dtype=torch.bool)
    full[10:14, 10:14] = True                     # a ball in the first ROI
    full[40:44, 60:64] = True                     # one in the second
    full[0:2, 0:2] = True                         # outside every ROI: never downloaded
    boxes = [(5, 5, 25, 25), (50, 30, 75, 50)]
    perf = {}
    masks = MotionMasks((60, 80), boxes, True, [full[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes], None, cfg, perf)
    expected_raw = np.zeros((60, 80), np.uint8)
    for x1, y1, x2, y2 in boxes:
        expected_raw[y1:y2, x1:x2] = full[y1:y2, x1:x2].numpy() * 255
    np.testing.assert_array_equal(masks.raw_u8(), expected_raw)
    packed = masks.packed_raw()
    reference = PackedMask.pack(expected_raw, roi=(5, 5, 75, 50))
    assert packed.roi == reference.roi == (5, 5, 75, 50) and np.array_equal(packed.bits, reference.bits)
    boost = masks.selector_boost_u8()
    assert boost is not None and boost[11, 11] == 255 and boost[41, 61] == 255 and boost[1, 1] == 0
    assert masks.has_blobs and masks.detector_boost_u8() is not None
    reference_boost = PackedMask.pack(boost, roi=(5, 5, 75, 50))
    assert np.array_equal(masks.packed_boost().bits, reference_boost.bits)
    assert set(perf) == {"pre_cuda_raw_d2h", "pre_cuda_cc_filter", "pre_cuda_total"}


def test_motion_masks_of_the_whole_frame_pack_the_bounding_box():
    cfg = Config(boost_min_blob_area=1, boost_max_blob_area=600, motion_raw_close_size=0)
    full = torch.zeros(30, 40, dtype=torch.bool)
    full[5:9, 20:24] = True
    masks = MotionMasks((30, 40), [(0, 0, 40, 30)], False, [full], None, cfg, None)
    assert masks.packed_raw().roi == PackedMask.pack(full.numpy().astype(np.uint8)).roi == (20, 5, 24, 9)
    empty = MotionMasks((30, 40), [(0, 0, 40, 30)], False, [torch.zeros(30, 40, dtype=torch.bool)], None, cfg, None)
    assert empty.packed_raw() is None and empty.packed_boost() is None and not empty.has_blobs
    assert empty.selector_boost_u8() is None and empty.detector_boost_u8() is None


def test_motion_masks_share_the_narrow_mask_unless_a_wide_source_differs():
    cfg = Config(boost_min_blob_area=1, boost_max_blob_area=600, motion_raw_close_size=0)
    narrow_src = torch.zeros(12, 12, dtype=torch.bool)
    narrow_src[4:8, 4:8] = True
    wide_src = narrow_src.clone()
    wide_src[0:3, 0:3] = True
    box = [(0, 0, 12, 12)]
    masks = MotionMasks((12, 12), box, False, [narrow_src], None, cfg, None)
    np.testing.assert_array_equal(masks.detector_boost_u8(), masks.selector_boost_u8())
    masks = MotionMasks((12, 12), box, False, [narrow_src], [wide_src], cfg, None)
    wide, narrow = masks.detector_boost_u8(), masks.selector_boost_u8()
    assert int(wide[1, 1]) == 255 and int(narrow[1, 1]) == 0
    # A narrow mask with no blob still boosts the detector input from the wide one.
    masks = MotionMasks((12, 12), box, False, [torch.zeros(12, 12, dtype=torch.bool)], [wide_src], cfg, None)
    assert masks.selector_boost_u8() is None and masks.has_blobs and masks.detector_boost_u8()[1, 1] == 255


# --------------------------------------------------------------------------- the raw masks

def _window_frames():
    """Three 48x64 frames: a ball that moves its own width each frame, a static blob and a blob
    that is only in the current frame (a flicker: it moves against both neighbours)."""
    frames = []
    for index in range(3):
        frame = np.zeros((48, 64, 3), np.uint8)
        x = 6 + 10 * index
        frame[10:16, x:x + 6] = (70, 235, 240)        # the moving ball
        frame[30:36, 40:46] = (200, 200, 200)          # static: foreground, never moving
        if index == 1:
            frame[30:36, 10:16] = (255, 255, 255)      # flicker: appears for one frame only
        frames.append(frame)
    return frames


def test_temporal_gate_matches_the_cpu_reference_with_the_gate_on():
    """With the gate on, the CUDA path keeps the moving ball and the one-frame flicker and drops
    the static blob, exactly as the CPU reference in tests/_helpers.py does."""
    cfg = Config(motion_raw_temporal_gate=True, motion_raw_close_size=0, motion_thresh=1., motion_v_min=0.)
    prev, curr, nxt = _window_frames()
    window = FrameWindow(CPU, True, prev, curr)
    window.slide(nxt)                                   # prev / curr / next all present
    assert window.curr is curr and window.prev_gpu is not None and window.next_gpu is not None
    background = _zero_background(cfg, 48, 64)
    gated, ungated = _raw_motion_cuda(window, background, cfg, None)
    ungated_u8 = ungated.numpy().astype(np.uint8) * 255
    assert ungated_u8[12, 18] and ungated_u8[32, 42] and ungated_u8[32, 12]   # all three are foreground
    cuda_raw = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY).raw_motion
    np.testing.assert_array_equal(cuda_raw, refine_raw_motion_temporal_cpu(ungated_u8, prev, curr, nxt, cfg))
    assert cuda_raw[12, 18] and cuda_raw[32, 12] and not cuda_raw[32, 42]
    assert np.count_nonzero(gated.numpy()) < np.count_nonzero(ungated.numpy())

    # At the end of the clip there is no next frame: the gate is the previous delta alone.
    window.slide(None)
    assert window.curr is nxt and window.next_gpu is None
    gated, ungated = _raw_motion_cuda(window, background, cfg, None)
    ungated_u8 = ungated.numpy().astype(np.uint8) * 255
    cuda_raw = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY).raw_motion
    np.testing.assert_array_equal(cuda_raw, refine_raw_motion_temporal_cpu(ungated_u8, curr, nxt, None, cfg))
    assert cuda_raw[12, 28] and not cuda_raw[32, 42]


def _old_roi_gated(window, background, cfg, rois):
    """The gated mask as it was computed before the union box: every ROI on its own crop, the
    probe delta of each frame pair recomputed per ROI and blurred with zero padding at the crop."""
    curr, prev, nxt = window.curr_gpu, window.prev_gpu, window.next_gpu

    def delta(a, b, y1, y2, x1, x2):
        gray = [f.chw[0, y1:y2, x1:x2] * 0.114 + f.chw[1, y1:y2, x1:x2] * 0.587 + f.chw[2, y1:y2, x1:x2] * 0.299
                for f in (a, b)]
        g = torch.abs(gray[1] - gray[0]) * 255.0
        v = torch.abs(b.v[y1:y2, x1:x2] - a.v[y1:y2, x1:x2]) * 255.0
        s = torch.abs(b.s[y1:y2, x1:x2] - a.s[y1:y2, x1:x2]) * (255.0 * 1.25)
        d = torch.maximum(g, torch.maximum(v, s))
        return F.avg_pool2d(d[None, None], 3, stride=1, padding=1)[0, 0]

    thr_sq, k_sq = (cfg.motion_thresh / 255.0) ** 2, cfg.motion_k_std ** 2
    raw = torch.zeros_like(curr.v, dtype=torch.bool)
    temporal = torch.zeros_like(raw)
    for x1, y1, x2, y2 in rois:
        v_d = (curr.v[y1:y2, x1:x2] - background.mean[0, y1:y2, x1:x2]) ** 2
        s_d = (curr.s[y1:y2, x1:x2] - background.mean[1, y1:y2, x1:x2]) ** 2
        v_t = thr_sq + background.var[0, y1:y2, x1:x2] * k_sq
        s_t = (thr_sq + background.var[1, y1:y2, x1:x2] * k_sq) * 1.5
        fg = ((v_d > v_t) | (s_d > s_t)) & (curr.v[y1:y2, x1:x2] > cfg.motion_v_min / 255.0)
        raw[y1:y2, x1:x2] |= fg
        dp, dn = delta(prev, curr, y1, y2, x1, x2), delta(curr, nxt, y1, y2, x1, x2)
        hi, lo, very_hi = cfg.motion_raw_temporal_hi, cfg.motion_raw_temporal_lo, cfg.motion_raw_temporal_very_hi
        temporal[y1:y2, x1:x2] |= ((dp >= hi) & (dn >= lo)) | ((dn >= hi) & (dp >= lo)) | (dp >= very_hi)
    return raw & temporal, raw


def test_rois_in_one_union_box_give_the_masks_each_roi_gave_on_its_own():
    """Overlapping and separate ROIs, ROIs on the frame edge: the foreground and probe deltas
    computed once in the union box, each ROI blurring its own part, equal the per-ROI result
    bit for bit."""
    rng = np.random.default_rng(11)
    frames = [rng.integers(0, 256, (40, 70, 3), dtype=np.uint8) for _ in range(3)]
    cfg = Config(motion_thresh=20.0, motion_v_min=10.0)
    window = FrameWindow(CPU, True, frames[0], frames[1])
    window.slide(frames[2])
    background = BackgroundModel(cfg, CPU)
    background.mean = torch.from_numpy(rng.random((2, 40, 70)).astype(np.float32))
    background.var = torch.from_numpy(rng.random((2, 40, 70)).astype(np.float32) * 0.01)
    for rois in ([(5, 3, 30, 25)], [(5, 3, 30, 25), (20, 10, 50, 38)], [(0, 0, 12, 10), (55, 30, 70, 40)]):
        gated, ungated = _raw_motion_cuda(window, background, cfg, rois)
        old_gated, old_ungated = _old_roi_gated(window, background, cfg, rois)
        assert torch.equal(gated, old_gated) and torch.equal(ungated, old_ungated)
        assert gated.any()


def test_preprocess_remembers_the_cleaned_mask_for_the_next_background_update():
    cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=2, motion_thresh=1., motion_v_min=0.)
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[10:20, 10:20, 2] = 255
    frame[3, 3, 2] = 255
    frame[3, 5, 2] = 255                                # closed into one blob by the 2x2 closing
    window = FrameWindow(CPU, True, frame, None)
    background = _zero_background(cfg, 32, 32)
    result = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY, rois=[(0, 0, 20, 20)])
    remembered = background._prev_motion
    assert remembered is not None and remembered.shape == (32, 32)
    np.testing.assert_array_equal(remembered.numpy().astype(np.uint8) * 255, result.raw_motion)
    assert not remembered[25, 25]                       # outside the ROI
    not_ready = BackgroundModel(cfg, CPU)
    assert preprocess_frame_cuda(window, not_ready, cfg, request=MASKS_ONLY).masks is None
    assert not_ready._prev_motion is None


def test_detector_input_request_decides_what_preprocessing_materialises():
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[10:20, 10:20, 2] = 255
    cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=0, motion_thresh=1., motion_v_min=0.)
    window = FrameWindow(CPU, True, frame, None)
    background = _zero_background(cfg, 32, 32)
    masks_only = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY)
    assert masks_only.detector_frame is None and masks_only.detector_frame_cuda is None
    assert np.count_nonzero(masks_only.raw_motion) == 100
    both = preprocess_frame_cuda(window, background, cfg, request=DetectorInputRequest(cpu=True, cuda=True))
    assert both.detector_frame is not None and both.detector_frame.shape == (32, 32, 3)
    assert both.detector_frame_cuda is not None and tuple(both.detector_frame_cuda.shape) == (32, 32, 3)
    np.testing.assert_array_equal(both.detector_frame, both.detector_frame_cuda.numpy())
    cpu_only = preprocess_frame_cuda(window, background, cfg, request=DetectorInputRequest(cpu=True))
    assert cpu_only.detector_frame_cuda is None and cpu_only.detector_frame is not None
    np.testing.assert_array_equal(cpu_only.raw_motion, masks_only.raw_motion)


def test_filter_boost_mask_treats_any_nonzero_pixel_as_motion():
    """The masks come to the host as 0/1; the component filter must give the discs it gives
    for 0/255."""
    cfg = Config(boost_min_blob_area=1, boost_max_blob_area=600)
    mask = np.zeros((30, 30), np.uint8)
    mask[10:15, 10:15] = 1
    np.testing.assert_array_equal(filter_boost_mask(mask, 1, 600, cfg), filter_boost_mask(mask * 255, 1, 600, cfg))
    assert cv2.countNonZero(filter_boost_mask(mask, 1, 600, cfg)) > 0


def test_filter_boost_mask_in_the_motion_bounding_box_matches_the_whole_frame_pass(monkeypatch):
    """Components found in the bounding box of the motion give the discs a pass over the whole
    frame gives (the box is forced to the full frame for the reference)."""
    import tennis_tracker.motion as motion
    cfg = Config(boost_min_blob_area=1, boost_max_blob_area=600)
    rng = np.random.default_rng(4)
    masks = []
    for density in (0.002, 0.02, 0.2):
        mask = np.zeros((90, 160), np.uint8)
        mask[20:70, 30:120] = (rng.random((50, 90)) < density).astype(np.uint8)
        masks.append(mask)
    edge = np.zeros((90, 160), np.uint8)
    edge[0:3, 157:160] = 1
    edge[86:90, 0:5] = 1
    masks.append(edge)
    cropped = [filter_boost_mask(mask, 1, 600, cfg) for mask in masks]
    monkeypatch.setattr(motion.cv2, "boundingRect", lambda m: (0, 0, m.shape[1], m.shape[0]))
    for mask, got in zip(masks, cropped):
        np.testing.assert_array_equal(got, filter_boost_mask(mask, 1, 600, cfg))
    assert any(np.count_nonzero(got) for got in cropped)