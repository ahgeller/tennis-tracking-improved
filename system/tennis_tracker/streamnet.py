"""StreamNet (``stream_s4``), a ball detector built to encode each frame once: a shared-weight
encoder runs on one frame transition at a time and a light decoder fuses the cached features of
the four transitions of a five-frame unit into GridTrackNet's output layout on HeatmapNet's
192x108 grid.

The contract is HeatmapNet's: five consecutive RGB frames at 768x432 stacked into 15 channels in
[0, 1] (oldest first) in, per frame a heat grid and x/y offset grids at stride 4 out (15 sigmoid
channels), so ``gridtracknet.decode_predictions``, the trainer's heat-map loss and the selector
read it unchanged. HeatmapNet fuses the five frames in its first convolution, so every layer runs
again for every unit a frame belongs to: with ``Config.gridtracknet_window_step`` 1 a frame is in
five units and is encoded five times. Here nothing before the decoder sees more than one
transition, so its features are computed once and reused by every unit that holds it.

What the encoder sees. TrackNetV5's Motion Direction Decoupling (arXiv 2512.02789, eq. 1) feeds
the network the signed difference of consecutive frames split by polarity, ``ReLU(I_t - I_t-1)``
and ``ReLU(I_t-1 - I_t)``: a moving ball leaves a negative trace where it was and a positive one
where it arrives, which the absolute difference of TrackNetV4 loses. A per-frame encoder cannot
take a difference itself, so its input is one *transition*: the current frame's RGB and the two
polarity fields of its difference from the frame before, per colour channel (9 channels; the
paper does not say gray or colour, and per channel keeps the previous frame exactly recoverable,
``previous = frame - positive + negative``, so nothing of the pair is lost). MDD's learnable
sigmoid mapping of the fields is left out: the encoder's first 3x3 convolution with BatchNorm and
ReLU can threshold them, and whether the mapping helps here is untested.

The first frame of a unit has no previous frame inside the unit, and a difference from a frame
outside it would break the 15-channel contract; a zero difference for it would make a frame's
features depend on its position in the unit, so the cache would no longer hold. So a unit is its
four transitions, (frame 1 | frame 0) to (frame 4 | frame 3), each the same function of two
consecutive frames wherever it sits: ``encode_frame(frame, previous)`` is computed once per frame
of a phase (from the phase's second frame on) and ``decode`` takes the last four. Frame 0's
pixels reach the decoder through the first transition's difference fields.

The encoder is a pixel-unshuffle stem (space-to-depth by 2, lossless for a 2 to 4 px ball, in
place of a full-resolution convolution) and RepVGG blocks (``repvgg``) at strides 2, 4, 8 and 16
with ``ENCODER_CHANNELS``; it hands the decoder its stride 4, 8 and 16 features. The decoder
concatenates the four transitions' features at each stride (in temporal order, so each frame
keeps its own channels), reduces them with a 1x1 convolution, refines with a RepVGG block,
upsamples (nearest, x2) into the next finer stride and ends in a 3x3 head with a sigmoid whose
heat biases start at HeatmapNet's ``HEAT_PRIOR``. It holds no state between units: the ConvGRU
``memory_s4`` removed on 2026-10-26 lost to its own control, and this keeps the five-frame window.
``reparameterize`` (``repvgg.fuse_for_inference``) fuses the deployed network to plain 3x3
convolutions; the ``.pt`` record is the trainable form (``heatmapnet.save_model``/``read_record``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn

from . import repvgg
from .gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH
from .heatmapnet import read_record

ARCH = "stream_s4"
STRIDE = 4
TOTAL_STRIDE = 16                         # the deepest feature's stride: an input must be a multiple of it
GRID_COLS = WIDTH // STRIDE      # 192
GRID_ROWS = HEIGHT // STRIDE     # 108
PAIRS_PER_UNIT = FRAMES_PER_UNIT - 1      # a unit's transitions: (frame 1 | frame 0) to (frame 4 | frame 3)
TRANSITION_CHANNELS = 9                   # RGB, the positive and the negative difference fields
ENCODER_CHANNELS = (16, 32, 64, 128)      # at strides 2, 4, 8 and 16
DECODER_CHANNELS = (128, 64, 48)          # at strides 16, 8 and 4

# One transition's features as the encoder hands them on: strides 4, 8 and 16.
PairFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def transition(frame: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """The encoder's input for the step from `previous` to `frame` (both ``(N, 3, H, W)``):
    `frame`'s RGB, then ``ReLU(frame - previous)`` and ``ReLU(previous - frame)`` (TrackNetV5's
    polarity fields), 9 channels."""
    difference = frame - previous
    return torch.cat([frame, torch.relu(difference), torch.relu(-difference)], dim=1)


class StreamNet(nn.Module):
    """The network; see the module docstring. ``ARCH``, ``GRID_COLS``, ``GRID_ROWS``,
    ``INPUT_CHANNELS`` and ``FRAMES_PER_UNIT`` are the contract it shares with ``HeatmapNet``
    (``ball_models``). ``forward`` takes the 15-channel unit; ``encode_frame`` and ``decode`` are
    the same computation split so a caller can encode every transition once and decode each unit
    from the four it cached (``forward`` is exactly ``decode`` of the four ``encode_frame``)."""

    ARCH = ARCH
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS
    INPUT_CHANNELS = INPUT_CHANNELS
    FRAMES_PER_UNIT = FRAMES_PER_UNIT
    PAIRS_PER_UNIT = PAIRS_PER_UNIT

    def __init__(self) -> None:
        super().__init__()
        e2, e4, e8, e16 = ENCODER_CHANNELS
        d16, d8, d4 = DECODER_CHANNELS
        self.stem = nn.Sequential(nn.PixelUnshuffle(2), repvgg.RepVGGBlock(TRANSITION_CHANNELS * 4, e2))
        self.down4 = nn.Sequential(repvgg.RepVGGBlock(e2, e4, stride=2), repvgg.RepVGGBlock(e4, e4))
        self.down8 = nn.Sequential(repvgg.RepVGGBlock(e4, e8, stride=2), repvgg.RepVGGBlock(e8, e8))
        self.down16 = nn.Sequential(repvgg.RepVGGBlock(e8, e16, stride=2), repvgg.RepVGGBlock(e16, e16))
        pairs = PAIRS_PER_UNIT
        self.fuse16 = nn.Sequential(repvgg.conv_bn_relu(pairs * e16, d16, 1), repvgg.RepVGGBlock(d16, d16))
        self.fuse8 = nn.Sequential(repvgg.conv_bn_relu(pairs * e8 + d16, d8, 1), repvgg.RepVGGBlock(d8, d8))
        self.fuse4 = nn.Sequential(repvgg.conv_bn_relu(pairs * e4 + d8, d4, 1), repvgg.RepVGGBlock(d4, d4))
        self.head = nn.Conv2d(d4, FRAMES_PER_UNIT * 3, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-normal convolutions (fan in, the norm ``loop.hold_conv_scale`` holds) with zero
        bias, and the head as ``repvgg.reset_head`` leaves it (the heat prior, CenterNet's small weights)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        repvgg.reset_head(self.head)

    def encode_frame(self, frame: torch.Tensor, previous: torch.Tensor) -> PairFeatures:
        """The features of the step from `previous` to `frame` (``(N, 3, H, W)`` RGB in [0, 1],
        consecutive frames of one phase) at strides 4, 8 and 16: what ``decode`` takes, four at a
        time. Sides that are not multiples of ``TOTAL_STRIDE`` are a ValueError."""
        if frame.shape[-2] % TOTAL_STRIDE or frame.shape[-1] % TOTAL_STRIDE:
            raise ValueError(f"{type(self).__name__} takes inputs whose height and width are multiples of "
                             f"{TOTAL_STRIDE} (the strides must meet in the decoder), not {tuple(frame.shape[-2:])}")
        quarter = self.down4(self.stem(transition(frame, previous)))
        eighth = self.down8(quarter)
        return quarter, eighth, self.down16(eighth)

    def decode(self, pairs: Sequence[PairFeatures]) -> torch.Tensor:
        """The 15-channel sigmoid output at stride 4 of one unit from its ``PAIRS_PER_UNIT``
        transitions' features, oldest first (``encode_frame`` of frames 1 to 4 of the unit, each
        with the frame before it)."""
        if len(pairs) != PAIRS_PER_UNIT:
            raise ValueError(f"a unit is decoded from {PAIRS_PER_UNIT} transitions, not {len(pairs)}")
        quarter, eighth, bottom = (torch.cat([pair[level] for pair in pairs], dim=1) for level in range(3))
        return self._decode(quarter, eighth, bottom)

    def _decode(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        up = self.fuse16(bottom)
        up = nn.functional.interpolate(up, scale_factor=2, mode="nearest")
        up = self.fuse8(torch.cat([eighth, up], dim=1))
        up = nn.functional.interpolate(up, scale_factor=2, mode="nearest")
        up = self.fuse4(torch.cat([quarter, up], dim=1))
        return torch.sigmoid(self.head(up))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """The unit's output: its four transitions encoded as one batch of ``4N`` (the same
        weights for each), their features regrouped per unit in temporal order, decoded."""
        batch, _, height, width = value.shape
        frames = value.reshape(batch, FRAMES_PER_UNIT, 3, height, width)
        current = frames[:, 1:].reshape(batch * PAIRS_PER_UNIT, 3, height, width)
        previous = frames[:, :-1].reshape(batch * PAIRS_PER_UNIT, 3, height, width)
        features = self.encode_frame(current, previous)
        quarter, eighth, bottom = (level.reshape(batch, PAIRS_PER_UNIT * level.shape[1], *level.shape[2:])
                                   for level in features)
        return self._decode(quarter, eighth, bottom)

    def reparameterize(self) -> "StreamNet":
        """The deployed network, in place and in eval mode: RepVGG branches and BatchNorm folded
        into plain convolutions (``repvgg.fuse_for_inference``). It answers as before to rounding
        and can no longer train or be saved as a record."""
        repvgg.fuse_for_inference(self)
        return self

    def streaming_halves(self) -> Tuple["StreamEncoder", "StreamDecoder"]:
        """The network as the two modules a streaming prepass runs (and exports to TensorRT), both
        sharing this network's weights: ``StreamEncoder`` (one transition at a time) and
        ``StreamDecoder`` (one unit from its four transitions' features). A detector streams the
        model it holds when it has this method (``ball_detectors``)."""
        return StreamEncoder(self), StreamDecoder(self)


class StreamEncoder(nn.Module):
    """``StreamNet.encode_frame`` as a module: ``(frame, previous)`` (each ``(N, 3, H, W)``) to the
    transitions' features at strides 4, 8 and 16."""

    def __init__(self, model: StreamNet):
        super().__init__()
        self.model = model

    def forward(self, frame: torch.Tensor, previous: torch.Tensor) -> PairFeatures:
        return self.model.encode_frame(frame, previous)


class StreamDecoder(nn.Module):
    """``StreamNet.decode`` as a module on the four transitions' features already concatenated per
    stride in temporal order (``(N, PAIRS_PER_UNIT * C, h, w)`` at each of strides 4, 8 and 16, the
    layout ``torch.cat`` of the four gives), to the unit's 15-channel output."""

    def __init__(self, model: StreamNet):
        super().__init__()
        self.model = model

    def forward(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        return self.model._decode(quarter, eighth, bottom)


def stream_units(model: StreamNet, frames: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Every five-frame unit's output over one phase's consecutive `frames` (each ``(N, 3, H, W)``),
    a unit at every frame from the fifth on, with each frame's transition encoded once: the
    streaming schedule a prepass with ``window_step`` 1 runs, and the reference the tests compare
    with ``forward`` on the stacked units."""
    cached: List[PairFeatures] = []
    outputs = []
    for index in range(1, len(frames)):
        cached.append(model.encode_frame(frames[index], frames[index - 1]))
        if len(cached) >= PAIRS_PER_UNIT:
            outputs.append(model.decode(cached[-PAIRS_PER_UNIT:]))
    return outputs


def load_model(weights: Path, device: torch.device) -> StreamNet:
    """The FP16 eval network from the ``.pt`` record ``heatmapnet.save_model`` wrote, in its
    trainable form (the trainer fine-tunes from it); ``reparameterize`` gives the deployed one."""
    model = StreamNet()
    model.load_state_dict(read_record(Path(weights), ARCH)["state_dict"])
    return model.eval().to(device).half()
