"""FrameWindow: the prev/curr/next window pass 1 slides over the clip, and the background model."""
import numpy as np
import pytest
import torch

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import BackgroundModel
from tennis_tracker.video_io import cuda_frame_planes, cuda_frame_to_chw_f32

CPU = torch.device("cpu")


def _frames(n, value_step=10):
    return [np.full((4, 6, 3), i * value_step, np.uint8) for i in range(n)]


def test_slide_moves_curr_to_prev_and_next_to_curr_with_uploads():
    f = _frames(3)
    window = FrameWindow(CPU, True, f[0], f[1])
    assert window.curr is f[0] and window.next is f[1]
    assert window.prev_gpu is None
    assert torch.allclose(window.curr_gpu.chw, torch.zeros(3, 4, 6))
    assert torch.allclose(window.next_gpu.v, torch.full((4, 6), 10 / 255))
    window.slide(f[2])
    assert window.curr is f[1] and window.next is f[2]
    assert torch.allclose(window.prev_gpu.chw, torch.zeros(3, 4, 6))
    assert torch.allclose(window.curr_gpu.v, torch.full((4, 6), 10 / 255))
    assert torch.allclose(window.next_gpu.v, torch.full((4, 6), 20 / 255))
    assert not window.exhausted
    window.slide(None)
    assert window.curr is f[2] and window.next is None and window.next_gpu is None
    window.slide(None)
    assert window.exhausted and window.curr is None


def _old_planes(chw):
    """Value, saturation and gray as the motion code computed them before ``cuda_frame_planes``."""
    maxc = torch.max(chw, dim=0).values
    minc = torch.min(chw, dim=0).values
    sat = torch.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), torch.zeros_like(maxc))
    gray = chw[0] * 0.114 + chw[1] * 0.587 + chw[2] * 0.299
    return gray, maxc, sat


def _random_frame(seed, h=23, w=31):
    frame = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    frame[:3, :3] = 0                                   # black pixels: the saturation's max == 0 case
    return frame


def test_upload_and_planes_are_bit_identical_to_the_float_host_path():
    """The uint8 upload converted on the device, and the motion planes computed once per frame,
    give exactly the floats of the old host-converted upload and the per-use plane formulas."""
    from tennis_tracker.video_io import cuda_frame_planes, cuda_frame_to_chw_f32
    frame = _random_frame(5)
    chw = cuda_frame_to_chw_f32(frame, CPU)
    old_chw = torch.from_numpy(frame).to(dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    assert chw.is_contiguous() and torch.equal(chw, old_chw)
    planes = cuda_frame_planes(chw)
    for new, old in zip(planes, _old_planes(old_chw)):
        assert torch.equal(new, old)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_upload_and_planes_are_bit_identical_on_cuda():
    """The same on the GPU, through the pinned uint8 uploader the runtime uses."""
    from tennis_tracker.video_io import PinnedFrameUploader, cuda_frame_planes
    device = torch.device("cuda:0")
    frame = _random_frame(6, 54, 96)
    uploader = PinnedFrameUploader(54, 96, device)
    chws = [uploader.upload_chw_f32(frame) for _ in range(4)]          # cycles through the slots
    old_chw = torch.from_numpy(frame).to(device=device, dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    for chw in chws:
        assert torch.equal(chw, old_chw)
    for new, old in zip(cuda_frame_planes(chws[-1]), _old_planes(old_chw)):
        assert torch.equal(new, old)


def test_without_upload_the_window_keeps_frames_but_no_tensors():
    f = _frames(2)
    window = FrameWindow(CPU, False, f[0], f[1])
    assert window.curr_gpu is None and window.next_gpu is None
    window.slide(None)
    assert window.curr is f[1] and window.prev_gpu is None and window.curr_gpu is None


def test_background_model_initialises_then_adapts_slower_at_motion_pixels():
    cfg = Config(background_alpha=0.5, motion_freeze_alpha=0.0, motion_thresh=25.5)
    model = BackgroundModel(cfg, CPU)
    assert not model.ready
    model.update(torch.zeros(2, 2, 2))
    assert model.ready and torch.allclose(model.var_v, torch.full((2, 2), 0.1 ** 2))
    motion = torch.zeros((2, 2), dtype=torch.bool)
    motion[0, 0] = True
    model.remember_motion(motion)
    model.update(torch.ones(2, 2, 2))
    # The motion pixel keeps the old mean (alpha 0); the rest moved halfway.
    assert model.mean_v[0, 0] == 0.0 and model.mean_v[1, 1] == 0.5
    assert torch.equal(model.mean[1], model.mean[0])          # saturation follows the same rule
    model.remember_motion(None)
    model.update(torch.ones(2, 2, 2))
    assert model.mean_v[0, 0] == 0.5


def _old_background_step(mean, var, curr, prev_motion, cfg):
    """The per-plane update as it was written before value and saturation shared one tensor."""
    diff_sq = (curr - mean) ** 2
    alpha = (torch.where(prev_motion, cfg.motion_freeze_alpha, cfg.background_alpha)
             if prev_motion is not None else cfg.background_alpha)
    var = var.clone().mul_(1.0 - alpha).add_(alpha * diff_sq)
    mean = mean.clone().mul_(1.0 - alpha).add_(curr * alpha)
    return mean, var


def test_stacked_background_update_is_bit_identical_to_the_per_plane_one():
    """Value and saturation updated as one ``(2, H, W)`` tensor give exactly the floats the two
    separate planes gave, with and without the slow-adaptation mask."""
    generator = torch.Generator().manual_seed(3)
    cfg = Config()
    frames = [torch.rand((2, 9, 13), generator=generator) for _ in range(4)]
    model = BackgroundModel(cfg, CPU)
    model.update(frames[0])
    mean = [frames[0][0].clone(), frames[0][1].clone()]
    var = [torch.full((9, 13), (cfg.motion_thresh / 255.0) ** 2) for _ in range(2)]
    prev_motion = None
    for index, frame in enumerate(frames[1:]):
        model.update(frame)
        for plane in range(2):
            mean[plane], var[plane] = _old_background_step(mean[plane], var[plane], frame[plane], prev_motion, cfg)
            assert torch.equal(model.mean[plane], mean[plane]) and torch.equal(model.var[plane], var[plane])
        prev_motion = torch.rand((9, 13), generator=generator) > 0.5
        model.remember_motion(prev_motion)


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")


@needs_cuda
def test_graphed_frames_are_the_eager_frames_bit_for_bit_while_in_the_window():
    """Every frame the graphed window holds (prev, curr and next, from slots that are reused every
    four frames) equals the eager conversion of the same frame, all along the clip."""
    device = torch.device("cuda", torch.cuda.current_device())
    frames = [_random_frame(seed, 54, 96) for seed in range(11)]
    graphed = FrameWindow(device, True, frames[0], frames[1], cuda_graphs=True)
    eager = FrameWindow(device, True, frames[0], frames[1])
    assert graphed.graphs is not None and eager.graphs is None
    for incoming in frames[2:] + [None, None]:
        for name in ("prev_gpu", "curr_gpu", "next_gpu"):
            got, expected = getattr(graphed, name), getattr(eager, name)
            assert (got is None) == (expected is None)
            if got is not None:
                assert torch.equal(got.chw, expected.chw) and torch.equal(got.planes, expected.planes)
        graphed.slide(incoming)
        eager.slide(incoming)
    assert len(graphed.graphs._slots) == 4


@needs_cuda
def test_a_failed_capture_falls_back_to_eager_conversion(monkeypatch, capsys):
    device = torch.device("cuda", torch.cuda.current_device())

    def refuse(*args, **kwargs):
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(torch.cuda, "graph", refuse)
    frames = [_random_frame(seed, 20, 30) for seed in range(3)]
    window = FrameWindow(device, True, frames[0], frames[1], cuda_graphs=True)
    assert window.graphs is None
    assert "CUDA graph capture failed" in capsys.readouterr().out
    expected = cuda_frame_planes(cuda_frame_to_chw_f32(frames[0], device))
    assert torch.equal(window.curr_gpu.planes, expected)
    model = BackgroundModel(Config(), device, cuda_graphs=True)
    model.update(expected[1:3])
    model.remember_motion(torch.zeros((20, 30), dtype=torch.bool, device=device))
    model.update(window.next_gpu.vs)
    assert not model.cuda_graphs and "updating the background directly" in capsys.readouterr().out


@needs_cuda
def test_the_graphed_background_update_is_the_eager_one_bit_for_bit():
    """Graphs per input address (four rotating plane buffers) and the motion mask copied into one
    buffer give exactly the eager model's mean and variance, through frames with and without a
    previous motion mask."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(11)
    cfg = Config()
    slots = [torch.empty((3, 9, 13), device=device) for _ in range(4)]
    graphed, eager = BackgroundModel(cfg, device, cuda_graphs=True), BackgroundModel(cfg, device)
    for index in range(14):
        slot = slots[index % 4]
        slot.copy_(torch.rand((3, 9, 13), generator=generator))
        graphed.update(slot[1:3])
        eager.update(slot[1:3].clone())
        assert torch.equal(graphed.mean, eager.mean) and torch.equal(graphed.var, eager.var)
        motion = None if index == 6 else (torch.rand((9, 13), generator=generator) > 0.5).to(device)
        graphed.remember_motion(motion)
        eager.remember_motion(motion)
    assert graphed.cuda_graphs and len(graphed._update_graphs) == 4
