"""MemoryNet, the third ball-detector architecture: HeatmapNet with a memory carried from one
five-frame unit to the next of the same phase, so the network knows where the ball was before
the five frames it is looking at.

The input, the grid and the output layout are HeatmapNet's (``heatmapnet``). At the bottleneck
(stride 16: 48x27 for the 768x432 input, ``STATE_CHANNELS`` channels) a convolutional GRU cell
with 3x3 convolutions updates the memory ``h`` from the bottleneck features ``b``:
``z = sigmoid(conv_z([b, h]))``, ``r = sigmoid(conv_r([b, h]))``, ``h~ = tanh(conv_h([b, r*h]))``,
``h' = (1 - z) h + z h~``; the decoder then takes ``b + out(h')``, where ``out`` is a 1x1
convolution. ``out`` starts at zero weights and bias, so a MemoryNet whose shared weights were
copied from a HeatmapNet (``from_heatmap``) answers exactly as that HeatmapNet does until the
memory learns something. The cell runs in float32 whatever the input's dtype (cast in, cast
out), which keeps it stable under FP16 inference and autocast training.

``forward(value, state=None)`` returns ``(output, new_state)``; the state is ``(N, STATE_CHANNELS,
H/16, W/16)`` in the input's dtype, None standing for zeros (the start of a phase). ``STATEFUL``
is the contract flag the prepass reads (``ball_models``): it carries each phase's state from unit
to unit in temporal order, and the trainer forwards through chains of consecutive units.

Weights are the ``.pt`` record ``heatmapnet.save_model`` writes, with ``"arch": "memory_s4"``;
``load_model`` reads only that, ``from_heatmap`` a ``heatmap_s4`` record (the parent).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from . import heatmapnet
from .heatmapnet import HeatmapNet, read_record

ARCH = "memory_s4"
PARENT_ARCH = heatmapnet.ARCH            # the architecture a MemoryNet may start from with its memory fresh
STATE_CHANNELS = heatmapnet.STAGE_CHANNELS[-1]      # 256, the bottleneck's
STATE_STRIDE = heatmapnet.TOTAL_STRIDE              # the state's stride over the input
MEMORY_MODULES = ("memory", "out")                  # the modules a HeatmapNet record does not carry


class ConvGRUCell(nn.Module):
    """The memory update, see the module docstring: three 3x3 convolutions on the bottleneck
    features and the state, the gates sigmoid, the candidate tanh, all in float32 (the weights
    are cast per call, so an FP16 model's cell still computes in float32). Default
    initialisation; the gates start near half open."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_z = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.conv_r = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.conv_h = nn.Conv2d(2 * channels, channels, 3, padding=1)

    @staticmethod
    def _conv(conv: nn.Conv2d, value: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(value, conv.weight.float(), conv.bias.float(), padding=1)

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """The new state from float32 `features` and `state` of the same shape."""
        joined = torch.cat([features, state], dim=1)
        update = torch.sigmoid(self._conv(self.conv_z, joined))
        reset = torch.sigmoid(self._conv(self.conv_r, joined))
        candidate = torch.tanh(self._conv(self.conv_h, torch.cat([features, reset * state], dim=1)))
        return (1 - update) * state + update * candidate


class MemoryNet(HeatmapNet):
    """The network; see the module docstring. ``forward`` takes and returns the state."""

    ARCH = ARCH
    STATEFUL = True

    def __init__(self) -> None:
        super().__init__()
        self.memory = ConvGRUCell(STATE_CHANNELS)
        self.out = nn.Conv2d(STATE_CHANNELS, STATE_CHANNELS, 1)
        self.reset_memory()

    def reset_memory(self) -> None:
        """``out`` to zero weights and bias: the memory contributes nothing until it has trained,
        so a network built from a HeatmapNet's weights starts as that HeatmapNet."""
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "out"):             # HeatmapNet's constructor resets before the memory exists
            self.reset_memory()

    def forward(self, value: torch.Tensor, state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        quarter, eighth, bottom = self.encode(value)
        with torch.autocast(device_type=bottom.device.type, enabled=False):
            features = bottom.float()
            previous = torch.zeros_like(features) if state is None else state.float()
            new_state = self.memory(features, previous)
        bottom = bottom + self.out(new_state.to(bottom.dtype))
        return self.decode(quarter, eighth, bottom), new_state.to(value.dtype)


def from_heatmap(weights: Path, device: torch.device) -> MemoryNet:
    """A float32 MemoryNet on ``device`` whose shared layers are the ``heatmap_s4`` record at
    `weights` and whose memory cell is fresh (``out`` zero): the trainer's starting model for a
    fine-tune from the parent architecture. A record that is not ``heatmap_s4``'s is a
    ValueError."""
    model = MemoryNet()
    missing, unexpected = model.load_state_dict(read_record(Path(weights), PARENT_ARCH)["state_dict"], strict=False)
    if unexpected or any(name.split(".")[0] not in MEMORY_MODULES for name in missing):
        raise ValueError(f"{weights} does not hold a HeatmapNet's layers (missing {missing}, unexpected {unexpected})")
    return model.to(device).float()


def load_model(weights: Path, device: torch.device) -> MemoryNet:
    """Build the network from a ``memory_s4`` ``.pt`` record as an FP16 eval model on ``device``,
    as ``heatmapnet.load_model`` does for its own."""
    model = MemoryNet()
    model.load_state_dict(read_record(Path(weights), ARCH)["state_dict"])
    return model.eval().to(device).half()
