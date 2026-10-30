"""tennis_tracker.motion_gpu: the motion-mask kernels give the PyTorch path's bits (needs CUDA)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import BackgroundModel, DetectorInputRequest, MASKS_ONLY, preprocess_frame_cuda
from tennis_tracker.motion_gpu import MotionKernels

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the motion kernels run on a CUDA device")

H, W = 72, 120
# No ROI (the whole frame), one ROI, two that overlap, two on the frame's edges and corners.
ROIS = [None, [(10, 5, 60, 50)], [(10, 5, 60, 50), (40, 30, 100, 70)], [(0, 0, 25, 18), (95, 50, 120, 72)]]


@pytest.fixture(scope="module")
def kernels():
    return MotionKernels.load(torch.device("cuda", 0))


def _frames(count, seed=3):
    """Noisy frames with a few blobs that move, appear and stay put, so every test has motion."""
    rng = np.random.default_rng(seed)
    frames = []
    for index in range(count):
        frame = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
        frame[20:28, 10 + 9 * index:18 + 9 * index] = (40, 230, 240)
        frame[45:52, 70:80] = (200, 200, 200)
        if index % 2:
            frame[5:9, 100:104] = (255, 255, 255)
        frames.append(frame)
    return frames


def _background(cfg, seed=5):
    rng = np.random.default_rng(seed)
    background = BackgroundModel(cfg, torch.device("cuda", 0))
    background.mean = torch.from_numpy(rng.random((2, H, W)).astype(np.float32)).cuda()
    background.var = torch.from_numpy((rng.random((2, H, W)) * 0.002).astype(np.float32)).cuda()
    return background


def _answers(result, background):
    masks = result.masks
    return (masks.packed_raw(), masks.packed_boost(), masks.detector_boost_u8(),
            background._prev_motion.cpu().numpy())


def _assert_same(a, b):
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            np.testing.assert_array_equal(x, y)
        elif x is None or y is None:
            assert x is None and y is None
        else:
            np.testing.assert_array_equal(x.bits, y.bits)
            assert x.roi == y.roi


@needs_cuda
@pytest.mark.parametrize("rois", ROIS)
@pytest.mark.parametrize("cfg", [
    Config(motion_thresh=6.0, motion_v_min=20.0),
    Config(motion_thresh=6.0, motion_v_min=20.0, motion_raw_temporal_gate=False),
    Config(motion_thresh=4.0, motion_v_min=0.0, motion_raw_close_size=3, motion_raw_open_size=2),
    Config(motion_thresh=4.0, motion_v_min=0.0, motion_raw_close_size=0),
])
@pytest.mark.parametrize("wide", [False, True])
def test_the_kernels_build_the_pytorch_masks_bit_for_bit(kernels, rois, cfg, wide):
    """Every window position (the clip's first frame, the middle, the last with no next frame),
    every ROI layout and each gate and cleaning setting: the same gated and ungated masks, the
    same packed masks and blobs, the same slow-adaptation mask for the background."""
    frames = _frames(4)
    device = torch.device("cuda", 0)
    window = FrameWindow(device, True, frames[0], frames[1])
    request = DetectorInputRequest(wide_boost=wide) if wide else MASKS_ONLY
    for step in range(4):
        reference, fused = _background(cfg), _background(cfg)
        ref = preprocess_frame_cuda(window, reference, cfg, request=request, rois=rois)
        got = preprocess_frame_cuda(window, fused, cfg, request=request, rois=rois, kernels=kernels)
        _assert_same(_answers(ref, reference), _answers(got, fused))
        assert ref.masks.packed_raw() is not None
        window.slide(frames[step + 2] if step + 2 < len(frames) else None)
        if window.exhausted:
            break


@needs_cuda
@pytest.mark.parametrize("graphs", [False, True])
def test_the_background_model_runs_the_same_with_either_path(kernels, graphs):
    """Ten frames through the background update and both mask paths, with and without the update's
    CUDA graphs: the model's mean and variance stay equal bit for bit, frame after frame."""
    cfg = Config(motion_thresh=6.0, motion_v_min=20.0)
    device = torch.device("cuda", 0)
    frames = _frames(10, seed=8)
    states = []
    for use_kernels in (False, True):
        window = FrameWindow(device, True, frames[0], frames[1], cuda_graphs=graphs)
        background = BackgroundModel(cfg, device, cuda_graphs=graphs)
        history = []
        for index in range(len(frames)):
            background.update(window.curr_gpu.vs)
            rois = None if index % 3 == 0 else [(10, 5, 60, 50), (40, 30, 100, 70)]
            result = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY, rois=rois,
                                           kernels=kernels if use_kernels else None)
            history.append((background.mean.cpu().numpy().copy(), background.var.cpu().numpy().copy(),
                            None if result.masks is None else result.masks.raw_u8()))
            window.slide(frames[index + 2] if index + 2 < len(frames) else None)
        states.append(history)
    for (mean_a, var_a, raw_a), (mean_b, var_b, raw_b) in zip(*states):
        np.testing.assert_array_equal(mean_a, mean_b)
        np.testing.assert_array_equal(var_a, var_b)
        if raw_a is None:
            assert raw_b is None
        else:
            np.testing.assert_array_equal(raw_a, raw_b)


@needs_cuda
def test_morph_sizes_match_the_pytorch_cleaning(kernels):
    """A dilation or erosion of any size equals the PyTorch path's, at every pixel of the box."""
    from tennis_tracker.motion import _morph_max

    rng = np.random.default_rng(2)
    mask = torch.from_numpy(rng.random((37, 53)) > 0.7).cuda()
    for size in (1, 2, 3, 4, 5):
        out = torch.empty(mask.shape, dtype=torch.uint8, device=mask.device)
        kernels.morph(mask.view(torch.uint8), out, size, erode=False)
        assert torch.equal(out.bool(), _morph_max(mask.float(), size) > 0.5)
        kernels.morph(mask.view(torch.uint8), out, size, erode=True)
        assert torch.equal(out.bool(), 1.0 - _morph_max(1.0 - mask.float(), size) > 0.5)


def test_pass1_builds_the_masks_with_pytorch_ops_when_asked_or_off_the_gpu():
    from tennis_tracker.pass1 import _load_motion_kernels

    assert _load_motion_kernels(replace(Config(), motion_kernels=False), torch.device("cuda", 0)) is None
    assert _load_motion_kernels(Config(), torch.device("cpu")) is None


def _planes(values):
    """A ``GpuFrame`` built straight from ``(3, H, W)`` motion planes (gray, value, saturation)."""
    from tennis_tracker.frame_window import GpuFrame

    planes = values.contiguous()
    return GpuFrame(planes, planes, planes[1], planes[2], planes[1:3])


@needs_cuda
def test_the_foreground_rounds_as_pytorch_does_at_the_threshold(kernels):
    """Pixels whose squared difference from the mean lands exactly on PyTorch's threshold (and a
    float32 step either side of it): only arithmetic rounded step for step as PyTorch's keeps every
    one of them on the same side."""
    from tennis_tracker.motion import _gate_scalars

    cfg = Config(motion_thresh=3.0, motion_k_std=2.7, motion_v_min=0.0, motion_raw_temporal_gate=False)
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(4)
    background = BackgroundModel(cfg, device)
    background.var = torch.from_numpy((rng.random((2, H, W)) * 0.01).astype(np.float32)).to(device)
    background.mean = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    scalars = _gate_scalars(cfg)
    threshold = background.var[0] * (cfg.motion_k_std ** 2)
    threshold.add_((cfg.motion_thresh / 255.0) ** 2)
    root = torch.sqrt(threshold)
    step = torch.from_numpy(rng.integers(-1, 2, (H, W)).astype(np.float32)).to(device)
    value = torch.where(step == 0, root, torch.nextafter(root, root + step))   # a float32 step up, down, or none
    planes = torch.zeros((3, H, W), dtype=torch.float32, device=device)
    planes[1] = value
    frame = _planes(planes)
    expected = background.foreground(frame, (0, 0, W, H))
    on_threshold = (value * value) == threshold
    assert on_threshold.sum() > 100 and expected.any() and not expected.all()
    gated = torch.zeros((H, W), dtype=torch.uint8, device=device)
    kernels.gate(frame.planes, None, None, background.mean, background.var, (0, 0, W, H), gated, None,
                 scalars, temporal=False)
    assert torch.equal(gated.bool(), expected)


@needs_cuda
@pytest.mark.parametrize("with_next", [False, True])
def test_the_temporal_gate_rounds_as_pytorch_does_at_its_thresholds(kernels, with_next):
    """``hi`` (and ``lo``) set to blurred deltas PyTorch computed, so those pixels sit exactly on the
    gate: the kernel's sums, products and division must round as ``_probe_delta`` and ``avg_pool2d``
    do for the masks to agree."""
    from tennis_tracker.motion import _blur3, _gate_scalars, _probe_delta

    device = torch.device("cuda", 0)
    rng = np.random.default_rng(6)
    frames = [_planes(torch.from_numpy(rng.random((3, H, W), dtype=np.float32)).to(device)) for _ in range(3)]
    prev, curr, nxt = frames
    box = (7, 3, 101, 66)
    x1, y1, x2, y2 = box
    crop = (slice(None), slice(y1, y2), slice(x1, x2))
    dp = _blur3(_probe_delta(prev.planes[crop], curr.planes[crop]))
    dn = _blur3(_probe_delta(curr.planes[crop], nxt.planes[crop]))
    mean = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    var = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    picks = rng.choice(dp.numel(), 40, replace=False)
    # Each picked delta is the threshold itself (the pixel passes) and the next float32 above it (it
    # fails), so a kernel that rounds a blurred delta up or down at that pixel disagrees either way.
    above = [(float(dp.flatten()[pick]), float(dn.flatten()[pick])) for pick in picks]
    for hi, lo in above + [(float(np.nextafter(np.float32(a), np.float32(np.inf))),
                             float(np.nextafter(np.float32(b), np.float32(np.inf)))) for a, b in above]:
        cfg = Config(motion_thresh=0.0, motion_v_min=0.0, motion_raw_temporal_hi=hi, motion_raw_temporal_lo=lo,
                     motion_raw_temporal_very_hi=1e9)
        if with_next:
            temporal = ((dp >= hi) & (dn >= lo)) | ((dn >= hi) & (dp >= lo)) | (dp >= 1e9)
        else:
            temporal = dp >= hi
        fg = BackgroundModel(cfg, device)
        fg.mean, fg.var = mean, var
        expected = torch.zeros((H, W), dtype=torch.bool, device=device)
        expected[y1:y2, x1:x2] = fg.foreground(curr, box) & temporal
        gated = torch.zeros((H, W), dtype=torch.uint8, device=device)
        kernels.gate(curr.planes, prev.planes, nxt.planes if with_next else None, mean, var, box, gated, None,
                     _gate_scalars(cfg), temporal=True)
        assert torch.equal(gated.bool(), expected), f"hi={hi!r}"
