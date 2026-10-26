"""HeatmapNet, the second ball-detector architecture: a U-shaped encoder-decoder with skip
connections that predicts a fine heat map, and the loader and writer of the ``.pt`` weights.

It takes exactly what GridTrackNet takes (five consecutive RGB frames at 768x432 stacked into
15 channels, in [0, 1]; ``gridtracknet.frame_tensor`` and the prepass stacking) and answers in
the same layout, per frame a heat grid and x/y offset grids, so ``gridtracknet.decode_predictions``
reads both. The difference is the grid: 192x108 (stride 4) in place of 48x27 (stride 16). The
ball is 2 to 4 px at the input, sub-pixel from stride 4 on, and a one-hot cell at stride 16 costs
recall at cell edges; the published work (WASB 2023, TrackNetV2/V3) measured the biggest gain
from a fine heat map head with skip connections, which this is. No background channel yet.

The encoder is a stem at full resolution and four stride-2 stages (each block two 3x3
convolutions with BatchNorm and ReLU, the first convolution of a stage carrying the stride);
the decoder upsamples twice (nearest, x2), concatenating the encoder's skip at each resolution,
and the head is one 3x3 convolution to 15 channels and a sigmoid. Convolutions are Kaiming
initialised with zero bias; the head's heat-channel bias starts at the CenterNet prior
(``HEAT_PRIOR``) so a fresh network predicts a nearly empty map and early training stays stable.
The forward pass is ``encode`` (the stem, the stages and the bottleneck) then ``decode``, so
``memorynet.MemoryNet`` can work on the bottleneck features between the two.

The ``.pt`` record (``save_model``, ``read_record``, ``load_model``) is shared with MemoryNet:
``{"arch": <the model's ARCH>, "state_dict"}``. ``read_record`` checks the architecture it is
given, each loader its own, and ``ball_models.arch_of`` reads it off any record.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH

ARCH = "heatmap_s4"
STRIDE = 4
TOTAL_STRIDE = 16                        # the bottleneck's stride: an input must be a multiple of it
GRID_COLS = WIDTH // STRIDE      # 192
GRID_ROWS = HEIGHT // STRIDE     # 108
STEM_CHANNELS = 24
STAGE_CHANNELS = (48, 96, 192, 256)      # at strides 2, 4, 8 and 16
DECODER_CHANNELS = (192, 96)             # at strides 8 and 4, after the skip is concatenated
HEAT_PRIOR = 0.1                         # the heat a fresh network predicts everywhere (CenterNet's prior)
HEAT_CHANNELS = tuple(range(0, FRAMES_PER_UNIT * 3, 3))     # every frame's heat channel in the output


def _block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Two 3x3 convolutions with BatchNorm and ReLU; `stride` is carried by the first."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class HeatmapNet(nn.Module):
    """The network; see the module docstring. ``GRID_COLS``, ``GRID_ROWS``, ``INPUT_CHANNELS``,
    ``FRAMES_PER_UNIT``, ``ARCH`` and ``STATEFUL`` are the contract it shares with ``GridTrackNet``
    and ``MemoryNet`` (``ball_models``), which the detector, the decoder and the trainer read
    instead of asking which class they hold."""

    ARCH = ARCH
    STATEFUL = False
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS
    INPUT_CHANNELS = INPUT_CHANNELS
    FRAMES_PER_UNIT = FRAMES_PER_UNIT

    def __init__(self) -> None:
        super().__init__()
        s2, s4, s8, s16 = STAGE_CHANNELS
        d8, d4 = DECODER_CHANNELS
        self.stem = _block(INPUT_CHANNELS, STEM_CHANNELS)
        self.down1 = _block(STEM_CHANNELS, s2, stride=2)
        self.down2 = _block(s2, s4, stride=2)
        self.down3 = _block(s4, s8, stride=2)
        self.bottleneck = _block(s8, s16, stride=2)
        self.up1 = _block(s16 + s8, d8)
        self.up2 = _block(d8 + s4, d4)
        self.head = nn.Conv2d(d4, FRAMES_PER_UNIT * 3, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-normal convolutions with zero bias, and the head's heat channels biased to
        ``HEAT_PRIOR`` (sigmoid(bias) = HEAT_PRIOR), as CenterNet initialises its heat head."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.head.bias[list(HEAT_CHANNELS)] = -math.log((1 - HEAT_PRIOR) / HEAT_PRIOR)

    def encode(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The encoder: the skips at strides 4 and 8 and the bottleneck features at stride 16,
        what ``decode`` takes. An input whose sides are not multiples of ``TOTAL_STRIDE`` is a
        ValueError."""
        if value.shape[-2] % TOTAL_STRIDE or value.shape[-1] % TOTAL_STRIDE:
            raise ValueError(f"{type(self).__name__} takes inputs whose height and width are multiples of "
                             f"{TOTAL_STRIDE} (the skips must meet the upsampled maps), not {tuple(value.shape[-2:])}")
        full = self.stem(value)
        half = self.down1(full)
        quarter = self.down2(half)
        eighth = self.down3(quarter)
        return quarter, eighth, self.bottleneck(eighth)

    def decode(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        """The decoder and the head: the 15-channel sigmoid output at stride 4 from the bottleneck
        features and the two skips ``encode`` returned."""
        up = nn.functional.interpolate(bottom, scale_factor=2, mode="nearest")
        up = self.up1(torch.cat([up, eighth], dim=1))
        up = nn.functional.interpolate(up, scale_factor=2, mode="nearest")
        up = self.up2(torch.cat([up, quarter], dim=1))
        return torch.sigmoid(self.head(up))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(value))


def save_model(model: nn.Module, destination: Path) -> None:
    """Write `model`'s weights as the ``.pt`` record ``load_model`` reads: a dict with the model's
    architecture id (``arch``, its ``ARCH``) and the float32 ``state_dict``, through a temporary
    file beside `destination` so an interruption leaves no half-written file. A HeatmapNet and a
    MemoryNet are written alike."""
    state = {name: tensor.detach().float().cpu() for name, tensor in model.state_dict().items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.pt")
    torch.save({"arch": model.ARCH, "state_dict": state}, temporary)
    temporary.replace(destination)


def read_record(weights: Path, arch: Optional[str] = None) -> Dict[str, Any]:
    """The ``.pt`` record at `weights`: a dict with a string ``arch`` and a ``state_dict``, which
    must be `arch`'s when one is given (a loader insists on its own architecture;
    ``ball_models.arch_of`` reads any record's). A file of another shape (a bare state dict, a
    checkpoint) or of another architecture is a ValueError naming it."""
    wanted = arch or "ball-detector"
    try:
        record = torch.load(weights, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"{weights} is not a readable {wanted} weights file ({error})") from error
    found = record.get("arch") if isinstance(record, dict) else type(record).__name__
    if not isinstance(found, str) or not isinstance(record.get("state_dict"), dict) or (arch and found != arch):
        raise ValueError(f"{weights} is not a {wanted} weights file (arch {found!r})")
    return record


def load_model(weights: Path, device: torch.device) -> HeatmapNet:
    """Build the network from the ``.pt`` record ``save_model`` wrote as an FP16 eval model on
    ``device``, as ``gridtracknet.load_model`` does for the NPZ."""
    model = HeatmapNet()
    model.load_state_dict(read_record(Path(weights), ARCH)["state_dict"])
    return model.eval().to(device).half()
