"""HeatmapNetS2, the fourth ball-detector architecture: HeatmapNet with one more decoder block,
so the heat map is answered at stride 2 (384x216 for the 768x432 input) in place of stride 4.

The input, the 15-channel sigmoid output layout and the ``.pt`` record are HeatmapNet's
(``heatmapnet``); the encoder is HeatmapNet's to the last weight. After the stride-4 decoder
features (``HeatmapNet.features``) the map is upsampled once more (nearest, x2), the encoder's
stride-2 skip (``down1``'s output) is concatenated, and a third decoder block (``up3``,
``DECODER_CHANNELS_S2`` channels) feeds a head of its own. The ball is 2 to 4 px at the input,
so at stride 4 it already sits inside one cell with the offsets doing the sub-cell work; the
published heat-map trackers (TrackNetV2/V3, WASB) answer at full or half resolution, and stride 2
is the step toward them that the encoder's existing skip affords. ``dataset.gaussian_heat``'s
sigma is in cells and is left as it is, so the target Gaussian is half as wide in pixels here as
on the stride-4 grid: a known choice, not revisited now.

``from_heatmap`` builds the network from a ``heatmap_s4`` record (the parent, ``PARENT_ARCH``):
every shared layer is copied and the two the parent does not supply (``NEW_MODULES``: ``up3``,
and ``head``, whose in-channels differ from the parent head's) start fresh, Kaiming with the
heat prior, so the trainer's parent start is the heat winner's encoder and decoder with a new
last stage to learn. ``load_model`` reads only a ``heatmap_s2`` record.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from . import heatmapnet
from .gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH
from .heatmapnet import DECODER_CHANNELS, STAGE_CHANNELS, HeatmapNet, _block, read_record

ARCH = "heatmap_s2"
STRIDE = 2
PARENT_ARCH = heatmapnet.ARCH            # the architecture a HeatmapNetS2 may start from with its last stage fresh
GRID_COLS = WIDTH // STRIDE      # 384
GRID_ROWS = HEIGHT // STRIDE     # 216
DECODER_CHANNELS_S2 = 48                 # the third decoder block's channels, at stride 2
NEW_MODULES = ("up3", "head")            # the modules a HeatmapNet record does not supply


class HeatmapNetS2(HeatmapNet):
    """The network; see the module docstring. The contract attributes are HeatmapNet's but for
    the grid."""

    ARCH = ARCH
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS

    def __init__(self) -> None:
        super().__init__()
        self.up3 = _block(DECODER_CHANNELS[-1] + STAGE_CHANNELS[0], DECODER_CHANNELS_S2)
        self.head = nn.Conv2d(DECODER_CHANNELS_S2, FRAMES_PER_UNIT * 3, 3, padding=1)
        self.reset_parameters()          # again, now that the third block and the head it serves exist

    def encode(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The encoder: the skips at strides 2, 4 and 8 and the bottleneck features at stride 16,
        what ``decode`` takes. An input whose sides are not multiples of ``TOTAL_STRIDE`` is a
        ValueError (``check_input``)."""
        self.check_input(value)
        full = self.stem(value)
        half = self.down1(full)
        quarter = self.down2(half)
        eighth = self.down3(quarter)
        return half, quarter, eighth, self.bottleneck(eighth)

    def decode(self, half: torch.Tensor, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        """The decoder and the head: the 15-channel sigmoid output at stride 2 from the bottleneck
        features and the three skips ``encode`` returned (HeatmapNet's stride-4 features, upsampled
        once more and joined with the stride-2 skip, through the third block)."""
        up = nn.functional.interpolate(self.features(quarter, eighth, bottom), scale_factor=2, mode="nearest")
        up = self.up3(torch.cat([up, half], dim=1))
        return torch.sigmoid(self.head(up))


def from_heatmap(weights: Path, device: torch.device) -> HeatmapNetS2:
    """A float32 HeatmapNetS2 on ``device`` whose shared layers are the ``heatmap_s4`` record at
    `weights` and whose third decoder block and head are fresh: the trainer's starting model for a
    fine-tune from the parent architecture. The parent's head is dropped before loading (its
    in-channels are the stride-4 features', not ``DECODER_CHANNELS_S2``, so it could not be
    loaded); a record that is not ``heatmap_s4``'s, or one without every shared layer, is a
    ValueError."""
    model = HeatmapNetS2()
    state = read_record(Path(weights), PARENT_ARCH)["state_dict"]
    shared = {name: tensor for name, tensor in state.items() if name.split(".")[0] != "head"}
    missing, unexpected = model.load_state_dict(shared, strict=False)
    if unexpected or any(name.split(".")[0] not in NEW_MODULES for name in missing):
        raise ValueError(f"{weights} does not hold a HeatmapNet's layers (missing {missing}, unexpected {unexpected})")
    return model.to(device).float()


def load_model(weights: Path, device: torch.device) -> HeatmapNetS2:
    """Build the network from a ``heatmap_s2`` ``.pt`` record as an FP16 eval model on ``device``,
    as ``heatmapnet.load_model`` does for its own."""
    model = HeatmapNetS2()
    model.load_state_dict(read_record(Path(weights), ARCH)["state_dict"])
    return model.eval().to(device).half()
