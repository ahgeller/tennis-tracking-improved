"""The GridTrackNet input contract and NPZ weight mapping, on CPU with synthetic data."""
import numpy as np
import pytest
import torch

import cv2

from tennis_tracker import gridtracknet
from tennis_tracker.gridtracknet import (
    BN_EPSILON,
    BN_MOMENTUM,
    GRID_COLS,
    GRID_ROWS,
    HEIGHT,
    RESIZE_INTERPOLATION,
    WIDTH,
    GridTrackNet,
    WidthBatchNorm,
    decode_predictions,
    frame_tensor,
    load_model,
    resize_to_model,
)

CPU = torch.device("cpu")


def test_frame_tensor_is_rgb_chw_in_unit_range():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = 255  # OpenCV frames are BGR: this is pure blue
    frame[..., 1] = 51   # green at 0.2
    tensor = frame_tensor(frame, CPU)
    assert tensor.shape == (3, HEIGHT, WIDTH)
    assert tensor.dtype == torch.float16
    channel_means = tensor.float().mean(dim=(1, 2)).tolist()
    assert channel_means == pytest.approx([0.0, 0.2, 1.0], abs=1e-3)  # R, G, B


def test_frame_tensor_resizes_any_frame_to_model_size():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    tensor = frame_tensor(frame, CPU)
    assert tensor.shape == (3, HEIGHT, WIDTH)
    assert tensor.float().min().item() == pytest.approx(128 / 255, abs=1e-3)
    assert tensor.float().max().item() == pytest.approx(128 / 255, abs=1e-3)


def test_frame_tensor_is_resize_to_model_in_rgb_over_255():
    """The tracker's input is the one resize rule (``resize_to_model``, INTER_LINEAR) and nothing
    else: the training cache stores that very picture, so what the network trains on is what it
    is run on."""
    assert RESIZE_INTERPOLATION == cv2.INTER_LINEAR
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    resized = resize_to_model(frame)
    assert resized.shape == (HEIGHT, WIDTH, 3) and resized.dtype == np.uint8
    np.testing.assert_array_equal(resized, cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
    expected = np.moveaxis(resized[..., ::-1], -1, 0).astype(np.float32) / 255.0
    np.testing.assert_allclose(frame_tensor(frame, CPU).float().numpy(), expected, atol=1 / 2048)   # fp16 rounding
    model_sized = np.random.default_rng(1).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    np.testing.assert_array_equal(resize_to_model(model_sized), model_sized)


def _numpy_frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """The conversion the prepass ran on the host until the GPU-side one replaced it: the
    reference ``frame_tensor`` must match bit for bit."""
    rgb = cv2.cvtColor(resize_to_model(frame), cv2.COLOR_BGR2RGB)
    chw = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
    return torch.from_numpy(chw).to(device, dtype=torch.float16)


@pytest.mark.parametrize("device", [
    CPU,
    pytest.param(torch.device("cuda:0"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_frame_tensor_matches_the_numpy_conversion_bit_for_bit(device):
    """``frame_tensor`` uploads the uint8 picture and converts on the device; the result is the
    same float16 tensor the NumPy conversion produced, exactly, on every pixel of a 1080p frame."""
    frame = np.random.default_rng(2).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    produced = frame_tensor(frame, device)
    expected = _numpy_frame_tensor(frame, device)
    assert produced.device == expected.device and produced.dtype == torch.float16
    assert produced.shape == expected.shape == (3, HEIGHT, WIDTH)
    assert torch.equal(produced, expected)


def _synthetic_npz(path):
    """An NPZ with the upstream layer names, HWIO kernels and values that
    identify the kernel position, input channel and output channel."""
    model = GridTrackNet()
    arrays = {}
    for index, conv in enumerate(model.convs):
        name = "conv2d" if index == 0 else f"conv2d_{index}"
        out_ch, in_ch, kh, kw = conv.weight.shape
        ys, xs, cin, cout = np.meshgrid(np.arange(kh), np.arange(kw), np.arange(in_ch), np.arange(out_ch), indexing="ij")
        kernel = (ys * kw + xs) / 16 + (cin % 8) / 128 + (cout % 4) / 1024  # exact in fp16
        arrays[f"{name}/{name}/kernel:0"] = kernel.astype(np.float32)
        arrays[f"{name}/{name}/bias:0"] = np.full((out_ch,), index + 0.5, dtype=np.float32)
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        width = layer.gamma.shape[0]
        base = (np.arange(width) % 16).astype(np.float32) / 16  # exact in fp16
        arrays[f"{name}/{name}/gamma:0"] = base + 1.0
        arrays[f"{name}/{name}/beta:0"] = base + 2.0
        arrays[f"{name}/{name}/moving_mean:0"] = base + 3.0
        arrays[f"{name}/{name}/moving_variance:0"] = base + 4.0
    np.savez(path, **arrays)
    return arrays


def test_load_model_maps_npz_layers_and_kernel_orientation(tmp_path):
    weights = tmp_path / "weights.npz"
    arrays = _synthetic_npz(weights)

    model = load_model(weights, CPU)

    assert not model.training
    for index, conv in enumerate(model.convs):
        name = "conv2d" if index == 0 else f"conv2d_{index}"
        assert conv.weight.dtype == torch.float16
        expected = arrays[f"{name}/{name}/kernel:0"].transpose(3, 2, 0, 1)  # HWIO -> OIHW
        np.testing.assert_array_equal(conv.weight.detach().float().numpy(), expected)
        np.testing.assert_array_equal(conv.bias.detach().float().numpy(), arrays[f"{name}/{name}/bias:0"])
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        np.testing.assert_array_equal(layer.gamma.detach().float().numpy(), arrays[f"{name}/{name}/gamma:0"])
        np.testing.assert_array_equal(layer.beta.detach().float().numpy(), arrays[f"{name}/{name}/beta:0"])
        np.testing.assert_array_equal(layer.mean.float().numpy(), arrays[f"{name}/{name}/moving_mean:0"])
        np.testing.assert_array_equal(layer.variance.float().numpy(), arrays[f"{name}/{name}/moving_variance:0"])


def test_load_model_rejects_npz_missing_a_layer(tmp_path):
    weights = tmp_path / "weights.npz"
    arrays = _synthetic_npz(weights)
    del arrays["conv2d_3/conv2d_3/bias:0"]
    np.savez(weights, **arrays)
    with pytest.raises(KeyError):
        load_model(weights, CPU)


def test_width_batchnorm_normalizes_each_column_separately():
    layer = WidthBatchNorm(4).eval()
    with torch.no_grad():
        layer.mean.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
        layer.gamma.fill_(2.0)
        layer.beta.fill_(1.0)
    value = torch.ones((1, 1, 2, 4))
    out = layer(value)
    expected = (1.0 - torch.tensor([0.0, 1.0, 2.0, 3.0])) / (1.0 + BN_EPSILON) ** 0.5 * 2.0 + 1.0
    assert out.shape == value.shape
    torch.testing.assert_close(out[0, 0, 0], expected)
    torch.testing.assert_close(out[0, 0, 1], expected)


def _arbitrary_layer(width: int = 6) -> WidthBatchNorm:
    """A layer whose four arrays are all different and nowhere near the defaults."""
    layer = WidthBatchNorm(width)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        layer.gamma.copy_(torch.rand(width, generator=generator) + 0.5)
        layer.beta.copy_(torch.rand(width, generator=generator) - 0.5)
        layer.mean.copy_(torch.rand(width, generator=generator) * 2)
        layer.variance.copy_(torch.rand(width, generator=generator) + 0.1)
    return layer


def test_width_batchnorm_trains_gamma_and_beta_and_keeps_the_statistics_as_buffers():
    layer = WidthBatchNorm(6)
    assert {name for name, _ in layer.named_parameters()} == {"gamma", "beta"}
    assert {name for name, _ in layer.named_buffers()} == {"mean", "variance"}
    assert all(tensor.dtype == torch.float32 for tensor in layer.state_dict().values())
    torch.testing.assert_close(layer.gamma.detach(), torch.ones(6))       # Keras's defaults
    torch.testing.assert_close(layer.beta.detach(), torch.zeros(6))
    torch.testing.assert_close(layer.mean, torch.zeros(6))
    torch.testing.assert_close(layer.variance, torch.ones(6))


def test_width_batchnorm_in_train_mode_normalises_each_column_over_the_batch():
    """Train mode uses the batch's own statistics over N, C and H: every width column of the
    output has mean 0 and (biased) variance 1 at the default gamma and beta, whatever the
    running statistics say."""
    layer = _arbitrary_layer()
    with torch.no_grad():
        layer.gamma.fill_(1.0)
        layer.beta.fill_(0.0)
    layer.train()
    value = torch.randn((4, 3, 5, 6), generator=torch.Generator().manual_seed(1)) * 3 + 2
    out = layer(value)
    assert out.shape == value.shape and out.dtype == torch.float32
    torch.testing.assert_close(out.mean(dim=(0, 1, 2)), torch.zeros(6), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(dim=(0, 1, 2), unbiased=False), torch.ones(6), atol=2e-3, rtol=0)


def test_width_batchnorm_moves_the_running_statistics_by_the_keras_momentum_rule():
    """``moving = moving * 0.99 + batch * 0.01``, the batch mean and the unbiased batch variance
    (what the fused Keras op hands its running variance)."""
    layer = _arbitrary_layer().train()
    mean_before, variance_before = layer.mean.clone(), layer.variance.clone()
    value = torch.randn((4, 3, 5, 6), generator=torch.Generator().manual_seed(2)) * 3 + 2
    layer(value)
    batch_mean = value.mean(dim=(0, 1, 2))
    batch_variance = value.var(dim=(0, 1, 2), unbiased=True)
    torch.testing.assert_close(layer.mean, mean_before * BN_MOMENTUM + batch_mean * (1 - BN_MOMENTUM))
    torch.testing.assert_close(layer.variance, variance_before * BN_MOMENTUM + batch_variance * (1 - BN_MOMENTUM))
    assert BN_MOMENTUM == 0.99          # Keras's default, which upstream trained with


def test_width_batchnorm_in_eval_mode_applies_the_running_statistics_unchanged():
    """Eval mode is the exported inference: the stored statistics in the formula the port always
    used, and nothing moves."""
    layer = _arbitrary_layer().eval()
    state = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    value = torch.randn((2, 3, 4, 6), generator=torch.Generator().manual_seed(4)) * 3 + 2
    out = layer(value)
    shape = (1, 1, 1, -1)
    expected = ((value - state["mean"].view(shape)) * torch.rsqrt(state["variance"].view(shape) + BN_EPSILON)
                * state["gamma"].view(shape) + state["beta"].view(shape))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    for name, tensor in layer.state_dict().items():
        torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)
    half = _arbitrary_layer().eval().half()         # the tracker's FP16 path
    assert half(value.half()).dtype == torch.float16


def test_width_batchnorm_gamma_and_beta_receive_gradients():
    layer = _arbitrary_layer().train()
    value = torch.randn((2, 3, 4, 6), generator=torch.Generator().manual_seed(5))
    (layer(value) * torch.arange(6.0)).square().sum().backward()
    assert layer.gamma.grad is not None and layer.gamma.grad.abs().sum() > 0
    assert layer.beta.grad is not None and layer.beta.grad.abs().sum() > 0
    assert not layer.mean.requires_grad and not layer.variance.requires_grad


def test_decode_predictions_places_cell_offsets_in_output_pixels():
    output = torch.zeros((1, 15, GRID_ROWS, GRID_COLS))
    output[0, 0, 4, 6] = 0.9   # frame 0 confidence at row 4, col 6
    output[0, 1, 4, 6] = 0.5   # x offset within the cell
    output[0, 2, 4, 6] = 0.25  # y offset within the cell
    decoded = decode_predictions(output, 1920, 1080, 0.5)
    assert len(decoded) == 5
    assert decoded[0] == ((260.0, 170.0), pytest.approx(0.9))
    assert decoded[1] == (None, 0.0)


def test_decode_predictions_applies_threshold():
    output = torch.zeros((1, 15, GRID_ROWS, GRID_COLS))
    output[0, 0, 4, 6] = 0.4
    point, confidence = decode_predictions(output, 1920, 1080, 0.5)[0]
    assert point is None
    assert confidence == pytest.approx(0.4)


def test_self_test_passes():
    gridtracknet.self_test()


def test_decode_grid_is_what_decode_predictions_applies_per_frame():
    """decode_grid places one frame's (confidence, x offset, y offset) grids as decode_predictions
    places every frame of a unit, and answers None below the threshold with the confidence kept."""
    from tennis_tracker.gridtracknet import decode_grid
    output = torch.zeros((1, 15, 4, 6))
    output[0, 3, 1, 2], output[0, 4, 1, 2], output[0, 5, 1, 2] = 0.8, 0.5, 0.25
    grids = output[0, 3:6].numpy()
    assert decode_grid(grids, 1920, 1080, 0.5) == decode_predictions(output, 1920, 1080, 0.5)[1]
    assert decode_grid(grids, 1920, 1080, 0.5) == ((pytest.approx(2.5 * 320), pytest.approx(1.25 * 270)), pytest.approx(0.8))
    assert decode_grid(grids, 1920, 1080, 0.9) == (None, pytest.approx(0.8))
