"""PyTorch port of GridTrackNet and the loader for its exported NPZ weights.

Input: five consecutive RGB frames at 768x432, stacked into 15 channels, in
[0, 1].  Output: per frame a confidence grid and x/y offset grids at 48x27.
The width-axis normalisation is Keras BatchNormalization: its gamma and beta train
(in fine-tunes too), its running statistics follow the batches in train mode and
are what eval mode applies.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn


ARCH = "gridtracknet"
WIDTH = 768
HEIGHT = 432
FRAMES_PER_UNIT = 5
INPUT_CHANNELS = FRAMES_PER_UNIT * 3
GRID_COLS = 48
GRID_ROWS = 27
POOL_AFTER = {1, 3, 5, 8}
# Keras BatchNormalization defaults: the epsilon baked into the exported statistics and the
# momentum of the running statistics (``moving = moving * momentum + batch * (1 - momentum)``).
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99

# The one resize rule: the tracker's prepass and the trainer's frame cache both go through
# ``resize_to_model``, so the network trains on the very pixels it is run on.
RESIZE_INTERPOLATION = cv2.INTER_LINEAR

GridPoint = Tuple[Optional[Tuple[float, float]], float]


class WidthBatchNorm(nn.Module):
    """Reproduce GridTrackNet's BatchNormalization(axis=-1) in NCHW.

    The upstream model normalises along the width axis (Keras's last axis of an NCHW
    tensor), and this layer does exactly what Keras BatchNormalization does there:
    ``gamma`` and ``beta`` are trainable parameters, so they learn in scratch runs and
    fine-tunes alike, as they would in Keras; ``mean`` and ``variance`` are the running
    statistics.  In train mode the batch is normalised with its own statistics over the
    N, C and H axes (the biased variance) and the running statistics follow them with
    Keras's momentum rule, the unbiased variance feeding the running variance as the
    fused Keras op does.  In eval mode the running statistics are applied, which on the
    exported weights is the upstream inference.
    """

    def __init__(self, width: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))
        self.beta = nn.Parameter(torch.zeros(width))
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = (1, 1, 1, -1)
        if self.training:
            return self._normalise_batch(value, shape)
        return (
            (value - self.mean.view(shape))
            * torch.rsqrt(self.variance.view(shape) + BN_EPSILON)
            * self.gamma.view(shape)
            + self.beta.view(shape)
        )

    def _normalise_batch(self, value: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Train mode: normalise with the batch's statistics (in float32 under autocast) and
        move the running statistics toward them."""
        batch = value.float()
        axes = (0, 1, 2)
        mean = batch.mean(dim=axes)
        variance = batch.var(dim=axes, unbiased=False)
        with torch.no_grad():
            count = batch.numel() // batch.shape[-1]
            unbiased = variance * (count / max(count - 1, 1))
            self.mean.mul_(BN_MOMENTUM).add_(mean, alpha=1 - BN_MOMENTUM)
            self.variance.mul_(BN_MOMENTUM).add_(unbiased, alpha=1 - BN_MOMENTUM)
        return (
            (batch - mean.view(shape))
            * torch.rsqrt(variance.view(shape) + BN_EPSILON)
            * self.gamma.view(shape)
            + self.beta.view(shape)
        )


class GridTrackNet(nn.Module):
    """The upstream network: a VGG-style encoder with four max-pools and no decoder, predicting
    one cell of the 48x27 grid per frame with its offsets. ``GRID_COLS``, ``GRID_ROWS``,
    ``INPUT_CHANNELS``, ``FRAMES_PER_UNIT`` and ``ARCH`` are the contract it shares with
    ``heatmapnet.HeatmapNet`` (``ball_models``), so the detector, the decoder and the trainer
    read the model they hold instead of asking which class it is."""

    ARCH = ARCH
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS
    INPUT_CHANNELS = INPUT_CHANNELS
    FRAMES_PER_UNIT = FRAMES_PER_UNIT

    def __init__(self):
        super().__init__()
        channels = [INPUT_CHANNELS, 64, 64, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 15]
        widths = [768, 768, 384, 384, 192, 192, 96, 96, 96, 48, 48, 48]
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[index], channels[index + 1], 3, padding=1)
            for index in range(len(channels) - 1)
        )
        self.batch_norms = nn.ModuleList(WidthBatchNorm(width) for width in widths)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            value = conv(value)
            if index == len(self.convs) - 1:
                return torch.sigmoid(value)
            value = self.batch_norms[index](torch.relu(value))
            if index in POOL_AFTER:
                value = torch.nn.functional.max_pool2d(value, 2)
        raise AssertionError("unreachable")


def load_model(weights: Path, device: torch.device) -> GridTrackNet:
    """Build the network from the upstream NPZ export (Keras layer names, HWIO
    kernels) as an FP16 eval model on ``device``."""
    model = GridTrackNet()
    with np.load(weights) as source, torch.no_grad():
        for index, conv in enumerate(model.convs):
            name = "conv2d" if index == 0 else f"conv2d_{index}"
            prefix = f"{name}/{name}"
            conv.weight.copy_(torch.from_numpy(source[f"{prefix}/kernel:0"].transpose(3, 2, 0, 1)))
            conv.bias.copy_(torch.from_numpy(source[f"{prefix}/bias:0"]))
        for index, layer in enumerate(model.batch_norms):
            name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
            prefix = f"{name}/{name}"
            layer.gamma.copy_(torch.from_numpy(source[f"{prefix}/gamma:0"]))
            layer.beta.copy_(torch.from_numpy(source[f"{prefix}/beta:0"]))
            layer.mean.copy_(torch.from_numpy(source[f"{prefix}/moving_mean:0"]))
            layer.variance.copy_(torch.from_numpy(source[f"{prefix}/moving_variance:0"]))
    return model.eval().to(device).half()


def resize_to_model(frame: np.ndarray) -> np.ndarray:
    """One BGR uint8 frame at the model's WIDTH x HEIGHT, by ``RESIZE_INTERPOLATION``: the resize
    the tracker runs and the one the training cache stores. A frame already that size is
    returned as it is."""
    if frame.shape[0] == HEIGHT and frame.shape[1] == WIDTH:
        return frame
    return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=RESIZE_INTERPOLATION)


def frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """One BGR uint8 frame -> (3, HEIGHT, WIDTH) RGB float16 in [0, 1] on ``device``.

    The resize and the colour conversion stay in OpenCV on the host (they define the
    model's picture and release the interpreter lock); the layout change, the float
    conversion and the scaling happen on ``device`` after a 1 MB uint8 upload, so the
    prepass worker holds the lock for a fraction of a millisecond per frame instead of
    the two milliseconds the NumPy conversion and the 4 MB float upload took.  The
    result is bit-identical: uint8 -> float32 -> / 255 -> float16 on either side.
    """
    rgb = cv2.cvtColor(resize_to_model(frame), cv2.COLOR_BGR2RGB)
    u8 = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    return u8.permute(2, 0, 1).to(torch.float32).div_(255.0).to(torch.float16)


def decode_predictions(output: torch.Tensor, output_width: int, output_height: int,
                       threshold: float) -> List[GridPoint]:
    """Per input frame, the most confident grid cell as a point in ``output_width`` x
    ``output_height`` pixels (None below ``threshold``) and its confidence. The grid is the
    output's own (its last two axes), so GridTrackNet's 48x27 and HeatmapNet's 192x108 decode
    alike: ``x = (x_offset + col) * output_width / cols``."""
    rows, cols = int(output.shape[-2]), int(output.shape[-1])
    units = output.float().cpu().numpy().reshape(-1, FRAMES_PER_UNIT, 3, rows, cols)
    return [decode_grid(grids, output_width, output_height, threshold) for unit in units for grids in unit]


def decode_grid(grids: np.ndarray, output_width: int, output_height: int, threshold: float) -> GridPoint:
    """One frame's answer from its ``(3, rows, cols)`` grids (confidence, x offset, y offset):
    the most confident cell as a point in ``output_width`` x ``output_height`` pixels (None
    below ``threshold``) and its confidence."""
    confidence_grid, x_grid, y_grid = grids
    rows, cols = confidence_grid.shape
    row, col = np.unravel_index(np.argmax(confidence_grid), confidence_grid.shape)
    confidence = float(confidence_grid[row, col])
    point = None
    if confidence >= threshold:
        x = (float(x_grid[row, col]) + col) * output_width / cols
        y = (float(y_grid[row, col]) + row) * output_height / rows
        point = (x, y)
    return point, confidence


def self_test() -> None:
    output = torch.zeros((1, 15, GRID_ROWS, GRID_COLS))
    output[0, 0, 4, 6] = 0.9
    output[0, 1, 4, 6] = 0.5
    output[0, 2, 4, 6] = 0.25
    point = decode_predictions(output, 1920, 1080, 0.5)[0][0]
    if point != (260.0, 170.0):
        raise AssertionError(f"decode_predictions placed the cell at {point}, expected (260.0, 170.0)")
