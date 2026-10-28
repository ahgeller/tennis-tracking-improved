"""HeatmapNet-Lite (``heatmap_lite``): HeatmapNet's topology slimmed for speed, the lowest-risk
fast retrofit. Same input (the 15-channel five-frame unit at 768x432), same output (15 sigmoid
channels on the 192x108 grid), same early fusion of the five frames in the first layer, so it
answers exactly what HeatmapNet answers and only the cost changes.

Three changes, each where HeatmapNet's cost is (``audit/records/optimization_plan_2026-10.md``:
~25 GMAC per unit, the full-resolution stem ~2.8 and the decoder ~12.6):

- the full-resolution stem is a space-to-depth (pixel-unshuffle by 2) to 60 channels at stride 2,
  lossless, as YOLOv5's Focus layer and TResNet's stem (Ridnik et al. 2021, arXiv 2003.13630) do;
- the widths are about a quarter, ``STAGE_CHANNELS`` (16, 32, 64, 128) at strides 2, 4, 8, 16
  against (48, 96, 192, 256), and the decoder ``DECODER_CHANNELS`` (64, 48) against (192, 96);
- every 3x3 convolution is a RepVGG block (``repvgg``), trained with its 1x1 and identity
  branches and fused at deployment (``reparameterize``) into one convolution.

The encoder (``encode``) hands the decoder the skips at strides 4 and 8 and the bottleneck at
16; the decoder (``decode``) upsamples twice (nearest, x2), concatenating each skip, and ends in a
3x3 head with a sigmoid whose heat biases start at ``HEAT_PRIOR``. The ``.pt`` record is
HeatmapNet's (``heatmapnet.save_model``/``read_record``) and holds the trainable form.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from . import repvgg
from .gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH
from .heatmapnet import HEAT_CHANNELS, HEAT_PRIOR, read_record

ARCH = "heatmap_lite"
STRIDE = 4
TOTAL_STRIDE = 16                        # the bottleneck's stride: an input must be a multiple of it
GRID_COLS = WIDTH // STRIDE      # 192
GRID_ROWS = HEIGHT // STRIDE     # 108
STAGE_CHANNELS = (16, 32, 64, 128)       # at strides 2, 4, 8 and 16
DECODER_CHANNELS = (64, 48)              # at strides 8 and 4, after the skip is concatenated


def _stage(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Two RepVGG blocks, the first carrying `stride`: HeatmapNet's ``_block`` in RepVGG form."""
    return nn.Sequential(repvgg.RepVGGBlock(in_channels, out_channels, stride),
                         repvgg.RepVGGBlock(out_channels, out_channels))


class HeatmapLite(nn.Module):
    """The network; see the module docstring. ``ARCH``, ``GRID_COLS``, ``GRID_ROWS``,
    ``INPUT_CHANNELS`` and ``FRAMES_PER_UNIT`` are the contract it shares with ``HeatmapNet``."""

    ARCH = ARCH
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS
    INPUT_CHANNELS = INPUT_CHANNELS
    FRAMES_PER_UNIT = FRAMES_PER_UNIT

    def __init__(self) -> None:
        super().__init__()
        s2, s4, s8, s16 = STAGE_CHANNELS
        d8, d4 = DECODER_CHANNELS
        self.stem = nn.Sequential(nn.PixelUnshuffle(2), *_stage(INPUT_CHANNELS * 4, s2))
        self.down2 = _stage(s2, s4, stride=2)
        self.down3 = _stage(s4, s8, stride=2)
        self.bottleneck = _stage(s8, s16, stride=2)
        self.up1 = _stage(s16 + s8, d8)
        self.up2 = _stage(d8 + s4, d4)
        self.head = nn.Conv2d(d4, FRAMES_PER_UNIT * 3, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-normal convolutions with zero bias and the heat channels at ``HEAT_PRIOR``, as
        HeatmapNet initialises."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        bias = self.head.bias
        assert bias is not None
        with torch.no_grad():
            bias[list(HEAT_CHANNELS)] = -math.log((1 - HEAT_PRIOR) / HEAT_PRIOR)

    def encode(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The skips at strides 4 and 8 and the bottleneck at 16. Sides that are not multiples of
        ``TOTAL_STRIDE`` are a ValueError."""
        if value.shape[-2] % TOTAL_STRIDE or value.shape[-1] % TOTAL_STRIDE:
            raise ValueError(f"{type(self).__name__} takes inputs whose height and width are multiples of "
                             f"{TOTAL_STRIDE} (the skips must meet the upsampled maps), not {tuple(value.shape[-2:])}")
        quarter = self.down2(self.stem(value))
        eighth = self.down3(quarter)
        return quarter, eighth, self.bottleneck(eighth)

    def decode(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        """The 15-channel sigmoid output at stride 4."""
        up = nn.functional.interpolate(bottom, scale_factor=2, mode="nearest")
        up = self.up1(torch.cat([up, eighth], dim=1))
        up = nn.functional.interpolate(up, scale_factor=2, mode="nearest")
        return torch.sigmoid(self.head(self.up2(torch.cat([up, quarter], dim=1))))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(value))

    def reparameterize(self) -> "HeatmapLite":
        """The deployed network, in place and in eval mode (``repvgg.fuse_for_inference``)."""
        repvgg.fuse_for_inference(self)
        return self


def load_model(weights: Path, device: torch.device) -> HeatmapLite:
    """The FP16 eval network from its ``.pt`` record, in its trainable form."""
    model = HeatmapLite()
    model.load_state_dict(read_record(Path(weights), ARCH)["state_dict"])
    return model.eval().to(device).half()
