"""The prepass's device-side pieces (engines/ball_gpu.py), run on the CPU: the resize is OpenCV's to
the byte, the peaks are decode_grid's, the heat ring's means are the ones the host summed."""
import cv2
import numpy as np
import pytest
import torch

from tennis_tracker.engines import ball_gpu
from tennis_tracker.cuda_kernels import KernelUnavailable
from tennis_tracker.engines.ball_gpu import (
    HeatRing, ModelResize, Nv12ModelResize, Readback, grid_peaks, peak_points,
)
from tennis_tracker.video_decode import Nv12Frame, nv12_to_bgr
from tennis_tracker.gridtracknet import HEIGHT, WIDTH, decode_grid, frame_tensor

CPU = torch.device("cpu")


def _texture(width: int, height: int, seed: int) -> np.ndarray:
    """A BGR frame with noise, hard edges and smooth ramps, so every rounding case of the resize
    turns up."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    frame[: height // 3] = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
    frame[height // 2:, : width // 4] = (255, 0, 128)
    return frame


@pytest.mark.parametrize("width, height", [(1920, 1080), (1280, 720), (3840, 2160), (1536, 864), (2560, 1440),
                                           (1000, 563), (640, 360), (100, 60), (768, 432), (800, 450)])
def test_the_resize_is_opencvs_inter_linear_to_the_byte(width, height):
    """Downscales by 2.5, 1.67, 5 and 2 (which OpenCV runs as its area fast path), odd ratios,
    upscales and the model's own size all give exactly cv2.resize's bytes, and the model input is
    frame_tensor's."""
    frame = _texture(width, height, seed=width + height)
    resize = ModelResize(width, height, WIDTH, HEIGHT, CPU)
    got = resize.resize_u8(torch.from_numpy(frame)).numpy()
    expected = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    assert got.shape == expected.shape and np.array_equal(got, expected)
    assert torch.equal(resize(torch.from_numpy(frame)), frame_tensor(frame, CPU))


def test_the_taps_clamp_at_both_edges():
    first, second, weight0, weight1 = ball_gpu.linear_taps(4, 8, horizontal=True)     # an upscale: the ends sit past the pixels
    assert first[0] == 0 and weight0[0] == 2048 and weight1[0] == 0
    assert first[-1] == 3 and second[-1] == 3 and weight1[-1] == 0
    assert np.all(weight0 + weight1 == 2048)
    first, second, weight0, weight1 = ball_gpu.linear_taps(4, 8, horizontal=False)   # down: weights kept, rows clamped
    assert first[0] == second[0] == 0 and weight0[0] == 512 and weight1[0] == 1536     # at -0.25: rows -1 and 0


def test_peaks_are_decode_grids_answers_first_maximum_included():
    rng = np.random.default_rng(3)
    grids = rng.random((6, 3, 27, 48)).astype(np.float32)
    grids[2, 0] = 0.0
    grids[2, 0, 4, 7] = grids[2, 0, 9, 1] = 0.75            # a tie: the first in row-major order wins
    grids[4, 0] *= 0.2                                     # below the threshold everywhere
    peaks = grid_peaks(torch.from_numpy(grids)).numpy()
    found = peak_points(peaks, 27, 48, 1920, 1080, 0.5)
    assert found == [decode_grid(frame, 1920, 1080, 0.5) for frame in grids]
    assert found[2][0] == pytest.approx(((grids[2, 1, 4, 7] + 7) * 40, (grids[2, 2, 4, 7] + 4) * 40))
    assert found[4][0] is None


def test_the_ring_averages_in_the_hosts_order_and_clears_what_it_hands_back():
    """Units of five frames at one frame apart (a window step of 1) are summed in the ring; the mean
    each frame gets is bit for bit the host's (the sum in unit order, divided by the count on the
    device), and a taken slot starts again from zero."""
    rng = np.random.default_rng(5)
    output = torch.from_numpy(rng.random((4, 15, 3, 4)).astype(np.float16))
    units = [[f, f + 1, f + 2, f + 3, f + 4] for f in range(4)]
    ring = HeatRing(16, 3, 4, CPU)
    ring.add(output, units)
    grids = output.float().reshape(4, 5, 3, 3, 4)
    sums = {}
    for row, unit in enumerate(units):
        for position, index in enumerate(unit):
            sums[index] = sums[index] + grids[row, position] if index in sums else grids[row, position].clone()
    counts = {index: sum(index in unit for unit in units) for index in sums}
    frames = sorted(sums)
    expected = grid_peaks(torch.stack([sums[i] * torch.tensor(1.0 / counts[i], dtype=torch.float32) for i in frames]))
    assert torch.equal(ring.take(frames), expected)
    assert not ring.sums.any() and not ring.counts.any()
    ring.add(output[:1], [[16, 17, 18, 19, 20]])                     # slots 0 to 4 again, from zero
    assert torch.equal(ring.take([16]), grid_peaks(output[:1, 0:3].float()))


def test_the_ring_refuses_a_span_it_cannot_hold():
    ring = HeatRing(8, 1, 1, CPU)
    ring.check_span(10, 17)
    with pytest.raises(RuntimeError, match="8 slots"):
        ring.check_span(10, 18)


def test_a_cpu_readback_is_ready_at_once():
    values = torch.arange(8, dtype=torch.float32).view(2, 4)
    readback = Readback(values)
    values.zero_()                                   # the copy is the host's own
    assert readback.ready() and readback.wait().tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def _nv12_kernel_or_skip(device: torch.device):
    if not torch.cuda.is_available():
        pytest.skip("the NV12 kernel runs on a CUDA device")
    try:
        return ball_gpu.load_nv12_kernel(device)
    except KernelUnavailable as error:
        pytest.skip(f"no NVRTC here: {error}")


@pytest.mark.parametrize("width, height", [(1920, 1080), (1280, 720), (640, 360), (768, 432), (98, 64), (1000, 562)])
@pytest.mark.parametrize("full_range", [False, True])
def test_the_nv12_kernel_is_the_resize_of_opencvs_bgr_to_the_byte(width, height, full_range):
    """Random NV12 pictures (every chroma and rounding case) of downscales, upscales and the model's
    own size: one kernel gives exactly ModelResize(nv12_to_bgr(...)), the path whose bytes are
    cv2.resize's of OpenCV's picture."""
    device = torch.device("cuda")
    kernel = _nv12_kernel_or_skip(device)
    generator = torch.Generator().manual_seed(width * 7 + height + int(full_range))
    planes = torch.randint(0, 256, (height * 3 // 2, width), dtype=torch.uint8, generator=generator).to(device)
    expected = ModelResize(width, height, WIDTH, HEIGHT, device)(nv12_to_bgr(planes, full_range))
    got = Nv12ModelResize(width, height, WIDTH, HEIGHT, full_range, device, kernel)(Nv12Frame(planes, full_range))
    assert got.dtype == torch.float16 and got.shape == expected.shape
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def test_the_nv12_kernel_reads_rows_with_a_pitch_and_refuses_another_picture():
    device = torch.device("cuda")
    kernel = _nv12_kernel_or_skip(device)
    padded = torch.randint(0, 256, (96, 192), dtype=torch.uint8).to(device)
    planes = padded[:, :128]                                    # a 128x64 picture in rows of 192 bytes
    resize = Nv12ModelResize(128, 64, 40, 24, False, device, kernel)
    expected = ModelResize(128, 64, 40, 24, device)(nv12_to_bgr(planes.contiguous(), False))
    assert torch.equal(resize(Nv12Frame(planes, False)).view(torch.int16), expected.view(torch.int16))
    with pytest.raises(ValueError, match="got a"):
        resize(Nv12Frame(planes, True))
    with pytest.raises(ValueError, match="got a"):
        resize(Nv12Frame(padded[:90, :128], False))
