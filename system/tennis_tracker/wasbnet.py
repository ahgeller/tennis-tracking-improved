"""WASB-5f (``wasb_5f``): the small HRNet of WASB-SBDT (Tarashima et al. 2023, "Widely
Applicable Strong Baseline for Sports Ball Detection and Tracking", arXiv 2311.05237; code
https://github.com/nttcom/WASB-SBDT, MIT, after Microsoft's HRNet, MIT) on the tracker's contract.

WASB's model (``src/configs/model/wasb.yaml``) is an HRNet with a 64-channel two-convolution stem,
one Bottleneck stage of 32 (x4) channels and three multi-resolution stages of two BasicBlocks per
branch with branch widths (16, 32, 64, 128), summed fusion, and a 1x1 head on the finest branch.
Upstream it takes three frames at 512x288 and answers one heat map per frame at the input's own
resolution (stem strides 1, 1), so its finest branch runs at full resolution. Here it takes the
15-channel five-frame unit at 768x432 and answers the 15-channel layout on the 192x108 grid: the
stem strides are 2, 2 (the finest branch at stride 4, HRNet's pose-estimation setting, and the
output grid the decoder and the trainer expect), the head has 15 channels (heat and x/y offsets
per frame, sigmoid) and its heat biases start at HeatmapNet's ``HEAT_PRIOR``. Because 432 is not
a multiple of 32, the coarsest branch is 14 rows (strided convolutions round up) and fusion
upsamples to the finer branch's exact size (nearest) rather than by a fixed factor. Changing the
stride makes this a different network from the one WASB measured: its accuracy is untested, and
WASB's released weights (3 frames, stride 1) do not load into it.

Every convolution-BatchNorm pair is an ``nn.Sequential`` (``repvgg.conv_bn``), so the trainer's
``loop.hold_conv_scale`` holds them as it holds HeatmapNet's; ``reparameterize`` folds them
(``repvgg.fuse_for_inference``) for deployment. The ``.pt`` record is HeatmapNet's.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn

from . import repvgg
from .gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH
from .heatmapnet import read_record

ARCH = "wasb_5f"
STRIDE = 4
GRID_COLS = WIDTH // STRIDE      # 192
GRID_ROWS = HEIGHT // STRIDE     # 108
STEM_CHANNELS = 64
BOTTLENECK_CHANNELS = 32                  # the stage-1 Bottleneck's width; it answers 4x as many
BRANCH_CHANNELS = (16, 32, 64, 128)       # at strides 4, 8, 16 and 32
BLOCKS_PER_BRANCH = 2


class BasicBlock(nn.Module):
    """HRNet's BasicBlock: two 3x3 convolutions with BatchNorm and a residual sum, then a ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(*repvgg.conv_bn_relu(channels, channels, 3), *repvgg.conv_bn(channels, channels, 3))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(value) + value)


class Bottleneck(nn.Module):
    """HRNet's Bottleneck (1x1, 3x3, 1x1 to four times `width`) with a projected residual."""

    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.body = nn.Sequential(*repvgg.conv_bn_relu(in_channels, width, 1), *repvgg.conv_bn_relu(width, width, 3),
                                  *repvgg.conv_bn(width, width * 4, 1))
        self.project = repvgg.conv_bn(in_channels, width * 4, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(value) + self.project(value))


def _downsample(in_channels: int, out_channels: int, steps: int) -> nn.Sequential:
    """HRNet's path to a branch `steps` strides coarser: stride-2 3x3 convolutions with BatchNorm,
    a ReLU after each but the last, the last one changing the width."""
    layers: List[nn.Module] = []
    for step in range(steps):
        last = step == steps - 1
        layers.extend(repvgg.conv_bn(in_channels, out_channels if last else in_channels, 3, stride=2))
        if not last:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    """One HRNet stage: ``BLOCKS_PER_BRANCH`` BasicBlocks on every branch, then every branch's
    output the ReLU of the sum of all branches brought to its resolution and width (a finer one by
    ``_downsample``, a coarser one by a 1x1 convolution with BatchNorm and nearest upsampling to
    the exact size)."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(*(BasicBlock(width) for _ in range(BLOCKS_PER_BRANCH)))
                                      for width in channels)
        self.fuse = nn.ModuleList()
        for target, target_width in enumerate(channels):
            row = nn.ModuleList()
            for source, source_width in enumerate(channels):
                if source == target:
                    row.append(nn.Identity())
                elif source > target:
                    row.append(repvgg.conv_bn(source_width, target_width, 1))
                else:
                    row.append(_downsample(source_width, target_width, target - source))
            self.fuse.append(row)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, branches: List[torch.Tensor]) -> List[torch.Tensor]:
        branches = [branch(value) for branch, value in zip(self.branches, branches)]
        fused = []
        for target, row in enumerate(self.fuse):
            total = branches[target]
            for source, layer in enumerate(row.children()):
                if source == target:
                    continue
                value = layer(branches[source])
                if source > target:
                    value = nn.functional.interpolate(value, size=branches[target].shape[-2:], mode="nearest")
                total = total + value
            fused.append(self.activation(total))
        return fused


class WasbNet(nn.Module):
    """The network; see the module docstring. ``ARCH``, ``GRID_COLS``, ``GRID_ROWS``,
    ``INPUT_CHANNELS`` and ``FRAMES_PER_UNIT`` are the contract it shares with ``HeatmapNet``."""

    ARCH = ARCH
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS
    INPUT_CHANNELS = INPUT_CHANNELS
    FRAMES_PER_UNIT = FRAMES_PER_UNIT

    def __init__(self) -> None:
        super().__init__()
        b4, b8, b16, b32 = BRANCH_CHANNELS
        self.stem = nn.Sequential(*repvgg.conv_bn_relu(INPUT_CHANNELS, STEM_CHANNELS, 3, stride=2),
                                  *repvgg.conv_bn_relu(STEM_CHANNELS, STEM_CHANNELS, 3, stride=2))
        self.layer1 = Bottleneck(STEM_CHANNELS, BOTTLENECK_CHANNELS)
        layer1_width = BOTTLENECK_CHANNELS * 4
        # Transitions: stage 2's two branches from layer1, then each later stage's new branch
        # from the coarsest branch before it (HRNet's _make_transition_layer for these widths).
        self.transition1 = nn.ModuleList([repvgg.conv_bn_relu(layer1_width, b4, 3),
                                          repvgg.conv_bn_relu(layer1_width, b8, 3, stride=2)])
        self.stage2 = HighResolutionModule((b4, b8))
        self.transition2 = repvgg.conv_bn_relu(b8, b16, 3, stride=2)
        self.stage3 = HighResolutionModule((b4, b8, b16))
        self.transition3 = repvgg.conv_bn_relu(b16, b32, 3, stride=2)
        self.stage4 = HighResolutionModule((b4, b8, b16, b32))
        self.head = nn.Conv2d(b4, FRAMES_PER_UNIT * 3, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-normal convolutions (fan in, the norm ``loop.hold_conv_scale`` holds; upstream
        uses fan out) with zero bias and the head as ``repvgg.reset_head`` leaves it."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        repvgg.reset_head(self.head)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2] % STRIDE or value.shape[-1] % STRIDE:
            raise ValueError(f"{type(self).__name__} takes inputs whose height and width are multiples of "
                             f"{STRIDE}, not {tuple(value.shape[-2:])}")
        top = self.layer1(self.stem(value))
        branches = self.stage2([layer(top) for layer in self.transition1])
        branches = self.stage3([*branches, self.transition2(branches[-1])])
        branches = self.stage4([*branches, self.transition3(branches[-1])])
        return torch.sigmoid(self.head(branches[0]))

    def reparameterize(self) -> "WasbNet":
        """The deployed network, in place and in eval mode: every BatchNorm folded into its
        convolution (``repvgg.fuse_for_inference``)."""
        repvgg.fuse_for_inference(self)
        return self


def load_model(weights: Path, device: torch.device) -> WasbNet:
    """The FP16 eval network from its ``.pt`` record, in its trainable form."""
    model = WasbNet()
    model.load_state_dict(read_record(Path(weights), ARCH)["state_dict"])
    return model.eval().to(device).half()
