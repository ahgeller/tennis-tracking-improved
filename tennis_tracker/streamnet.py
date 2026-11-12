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

The visibility head (optional, ``vis_head``). The selector cannot tell a frame the network looked
at and saw no ball from one it missed, and fills the gap either way (the research of 2026-10-31:
at e60 the interpolated rows earned 7 hits against 33 false alarms). A frame-level answer to "is
a ball visible here" is the signal it lacks, and the heat map does not give it: a peak's height
is a cell's confidence, not the frame's. It is an extra output, not a change to the contract:
``forward(value, return_vis=True)`` returns ``(grids, logits)``, ``VIS_LOGITS`` logits per unit
(one per frame of it), and the default call the tracker, ``decode_predictions``, the ONNX export
and the TensorRT engine make returns the 15 channels alone, so a record with the head loads and
runs wherever one without it does. The trainer trains it with ``loop.visibility_loss`` (BCE
against the labels' visible and absent frames) and ``save_model`` stores it in the record, from
which ``load_model`` rebuilds it (``has_vis_head``, ``vis_pool_of``). It comes in two variants,
``VIS_POOLS``, chosen at construction (``vis_pool``) and told apart by their keys in the record:

- ``global`` (the first head, the s4_fg1 run of 2026-11): a global average pool of the fused
  stride-16 features (``fuse16``'s output, ``DECODER_CHANNELS[0]`` channels at 48x27 for the full
  input) and one linear layer to the five logits. Measured offline on mixed_v1 it reaches AUC 0.89
  on all held-out labels and 0.96 on long absences, but 0.45 on the 183 frames the selector
  actually fills, the one-to-three-frame occlusions inside a rally (``experiments/vis_selector``):
  a mean over the whole unit at stride 16 cannot see one frame's occlusion.
- ``peak`` (``PeakHead``): a per-frame, local answer. Which features are per frame: the encoder's
  (``encode_frame``, one transition each; in the decoder's concatenations transition j of the
  unit owns channels ``[j * C, (j + 1) * C)`` at every stride). Everything after the 1x1
  reductions (``fuse16``, ``fuse8``, ``fuse4``, the ``head``) is fused across the unit, but the
  head's 15 channels are still laid out per frame: channel ``3k`` is frame k's heat map. So for
  frame k the head finds the peak of frame k's heat map (``topk`` with fixed ``VIS_TOPK`` over the
  cells, its first index the argmax) and gathers, at that cell, the fused stride-4 features
  (``fuse4``'s ``DECODER_CHANNELS[2]`` channels, the ones the heat map was read from) and the
  stride-4 features of frame k's own transition (pair ``k - 1``, ``ENCODER_CHANNELS[1]`` channels;
  frame 0 has no transition of its own inside the unit and reads pair 0, whose negative
  difference field holds where its ball was), together with the frame's ``VIS_TOPK`` highest heat
  values (the peak and its competition). A two-layer MLP shared across the frames, with a
  per-frame bias on its hidden layer (frame 0's transition means something else), gives the
  logit. Fixed shapes and no data-dependent control flow (a top-k with constant k and gathers
  with its indices), so the ONNX / Core ML export is as static as the global head's. About
  27 thousand multiply-adds a unit (``PeakHead.macs_per_unit``), against 2.64 GMAC a frame.

A bounce and a hit logit belong here too, once the trainer reads their labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

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
VIS_LOGITS = FRAMES_PER_UNIT              # the visibility head's outputs: "a ball is visible" per frame of the unit
VIS_PREFIX = "vis."                       # the head's keys in the state dict, how load_model knows a record has one
VIS_POOLS = ("global", "peak")            # the head's variants (module docstring): vis_pool at construction
VIS_TOPK = 3                              # the peak head reads a frame's k highest heat values (the first is its peak)
VIS_HIDDEN = 64                           # ... through a hidden layer of this width
VIS_GLOBAL_KEY = VIS_PREFIX + "weight"            # the global head's Linear, how vis_pool_of tells the variants apart
VIS_PEAK_KEY = VIS_PREFIX + "hidden.weight"       # the peak head's first layer

# One transition's features as the encoder hands them on: strides 4, 8 and 16.
PairFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Output = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]      # the grids, or (grids, visibility logits)


def has_vis_head(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Whether a StreamNet state dict carries the visibility head (any key under ``VIS_PREFIX``)."""
    return any(key.startswith(VIS_PREFIX) for key in state_dict)


def vis_pool_of(state_dict: Mapping[str, torch.Tensor]) -> Optional[str]:
    """Which visibility head a StreamNet state dict carries (``VIS_POOLS``), None for none: the
    record says so through the head's keys (``VIS_GLOBAL_KEY`` a Linear's weight, ``VIS_PEAK_KEY``
    the peak head's hidden layer). Keys under ``VIS_PREFIX`` of neither shape are a ValueError."""
    if not has_vis_head(state_dict):
        return None
    if VIS_GLOBAL_KEY in state_dict:
        return "global"
    if VIS_PEAK_KEY in state_dict:
        return "peak"
    keys = sorted(key for key in state_dict if key.startswith(VIS_PREFIX))
    raise ValueError(f"the record's visibility head is of no known variant ({VIS_POOLS}): keys {keys}")


def check_vis_pool(pool: str) -> str:
    if pool not in VIS_POOLS:
        raise ValueError(f"vis_pool is one of {VIS_POOLS}, not {pool!r}")
    return pool


class PeakHead(nn.Module):
    """The ``peak`` visibility head (module docstring): per frame of the unit, the fused stride-4
    features and the frame's own transition's stride-4 features at the frame's heat peak, with its
    ``VIS_TOPK`` highest heat values, through ``hidden`` (ReLU, plus a per-frame bias) and ``out``
    to one logit. ``forward(grids, up, quarter)`` takes the 15-channel sigmoid output, ``fuse4``'s
    output and the unit's concatenated stride-4 transition features, all ``(N, C, rows, cols)``,
    and answers ``(N, FRAMES_PER_UNIT)``."""

    def __init__(self, fused_channels: int = DECODER_CHANNELS[2], transition_channels: int = ENCODER_CHANNELS[1],
                 hidden: int = VIS_HIDDEN) -> None:
        super().__init__()
        self.fused_channels, self.transition_channels = fused_channels, transition_channels
        self.hidden = nn.Linear(fused_channels + transition_channels + VIS_TOPK, hidden)
        self.frame_bias = nn.Parameter(torch.zeros(FRAMES_PER_UNIT, hidden))
        self.out = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """The hidden layer at PyTorch's default, the output at CenterNet's prior (small weights,
        zero bias: a fresh head answers a logit near 0, probability 0.5, and claims nothing)."""
        self.hidden.reset_parameters()
        with torch.no_grad():
            self.frame_bias.zero_()
            nn.init.normal_(self.out.weight, std=repvgg.HEAD_STD)
            nn.init.zeros_(self.out.bias)

    def macs_per_unit(self) -> int:
        """Multiply-adds of the two linear layers over the five frames (the top-k and the gathers
        are selections): the head's cost beside the network's 2.64 GMAC a frame."""
        return FRAMES_PER_UNIT * (self.hidden.in_features * self.hidden.out_features + self.out.in_features)

    def forward(self, grids: torch.Tensor, up: torch.Tensor, quarter: torch.Tensor) -> torch.Tensor:
        batch, cells = grids.shape[0], grids.shape[-2] * grids.shape[-1]
        heat = grids.reshape(batch, FRAMES_PER_UNIT, 3, cells)[:, :, 0]                  # (N, 5, cells)
        values, index = heat.topk(VIS_TOPK, dim=2)                                         # sorted, the peak first
        fused = up.reshape(batch, self.fused_channels, cells)
        pairs = quarter.reshape(batch, PAIRS_PER_UNIT, self.transition_channels, cells)
        logits = []
        for frame in range(FRAMES_PER_UNIT):
            at = index[:, frame, :1].unsqueeze(1)                                            # (N, 1, 1): the peak cell
            fused_at = fused.gather(2, at.expand(-1, self.fused_channels, 1)).squeeze(2)
            own_at = pairs[:, max(frame - 1, 0)].gather(2, at.expand(-1, self.transition_channels, 1)).squeeze(2)
            features = torch.cat([fused_at, own_at, values[:, frame]], dim=1)
            activated = torch.relu(self.hidden(features) + self.frame_bias[frame])
            logits.append(self.out(activated))
        return torch.cat(logits, dim=1)


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
    from the four it cached (``forward`` is exactly ``decode`` of the four ``encode_frame``).
    With `vis_head` (or after ``enable_vis_head``) the network also carries the visibility head
    of the variant `vis_pool` names (``VIS_POOLS``, module docstring), answered only when a call
    asks for it with ``return_vis``."""

    ARCH = ARCH
    GRID_COLS = GRID_COLS
    GRID_ROWS = GRID_ROWS
    INPUT_CHANNELS = INPUT_CHANNELS
    FRAMES_PER_UNIT = FRAMES_PER_UNIT
    PAIRS_PER_UNIT = PAIRS_PER_UNIT
    VIS_LOGITS = VIS_LOGITS

    def __init__(self, vis_head: bool = False, vis_pool: str = "global") -> None:
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
        self.vis: Optional[nn.Module] = None            # nn.Linear (global) or PeakHead (peak)
        self.reset_parameters()
        if vis_head:
            self.enable_vis_head(vis_pool)
        else:
            check_vis_pool(vis_pool)

    @property
    def has_vis_head(self) -> bool:
        return self.vis is not None

    @property
    def vis_pool(self) -> Optional[str]:
        """The visibility head's variant (``VIS_POOLS``), None without one."""
        if self.vis is None:
            return None
        return "peak" if isinstance(self.vis, PeakHead) else "global"

    def enable_vis_head(self, pool: str = "global", replace: bool = False) -> "StreamNet":
        """Give the network the visibility head of variant `pool` (a fresh one, on the parameters'
        device and dtype) when it has none: a fine-tune of a record without one starts it here. A
        head of the same variant is kept; one of the other variant is a ValueError unless `replace`
        says to start a fresh head of the asked variant in its place. Returns `self`."""
        check_vis_pool(pool)
        if self.vis is not None:
            if self.vis_pool == pool:
                return self
            if not replace:
                raise ValueError(f"this {type(self).__name__} already has the {self.vis_pool} visibility head; "
                                 f"replace=True starts a fresh {pool} head in its place")
        reference = self.head.weight
        head: nn.Module = nn.Linear(DECODER_CHANNELS[0], VIS_LOGITS) if pool == "global" else PeakHead()
        self.vis = head.to(reference.device, reference.dtype)
        self.reset_vis_head()
        return self

    def reset_vis_head(self) -> None:
        """The visibility head at its prior: CenterNet's small weights and a bias answering 0.5
        (a logit of 0), so a fresh head neither disturbs the heat loss nor claims anything."""
        assert self.vis is not None
        if isinstance(self.vis, PeakHead):
            self.vis.reset_parameters()
            return
        assert isinstance(self.vis, nn.Linear)
        with torch.no_grad():
            nn.init.normal_(self.vis.weight, std=repvgg.HEAD_STD)
            nn.init.zeros_(self.vis.bias)

    def reset_parameters(self) -> None:
        """Kaiming-normal convolutions (fan in, the norm ``loop.hold_conv_scale`` holds) with zero
        bias, and the head as ``repvgg.reset_head`` leaves it (the heat prior, CenterNet's small weights)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        repvgg.reset_head(self.head)
        if self.vis is not None:
            self.reset_vis_head()

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

    def decode(self, pairs: Sequence[PairFeatures], return_vis: bool = False) -> Output:
        """The 15-channel sigmoid output at stride 4 of one unit from its ``PAIRS_PER_UNIT``
        transitions' features, oldest first (``encode_frame`` of frames 1 to 4 of the unit, each
        with the frame before it); with `return_vis`, ``(grids, visibility logits)``."""
        if len(pairs) != PAIRS_PER_UNIT:
            raise ValueError(f"a unit is decoded from {PAIRS_PER_UNIT} transitions, not {len(pairs)}")
        quarter, eighth, bottom = (torch.cat([pair[level] for pair in pairs], dim=1) for level in range(3))
        return self._decode(quarter, eighth, bottom, return_vis)

    def _decode(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor,
                return_vis: bool = False) -> Output:
        fused = self.fuse16(bottom)
        up = nn.functional.interpolate(fused, scale_factor=2, mode="nearest")
        up = self.fuse8(torch.cat([eighth, up], dim=1))
        up = nn.functional.interpolate(up, scale_factor=2, mode="nearest")
        up = self.fuse4(torch.cat([quarter, up], dim=1))
        grids = torch.sigmoid(self.head(up))
        if not return_vis:
            return grids
        if self.vis is None:
            raise ValueError(f"this {type(self).__name__} has no visibility head (enable_vis_head adds one)")
        if isinstance(self.vis, PeakHead):
            return grids, self.vis(grids, up, quarter)
        return grids, self.vis(fused.mean(dim=(2, 3)))

    def forward(self, value: torch.Tensor, return_vis: bool = False) -> Output:
        """The unit's output: its four transitions encoded as one batch of ``4N`` (the same
        weights for each), their features regrouped per unit in temporal order, decoded. The
        default is the 15-channel contract alone; `return_vis` adds the visibility logits,
        ``(N, VIS_LOGITS)``, as a second tensor (a network without the head is a ValueError)."""
        batch, _, height, width = value.shape
        frames = value.reshape(batch, FRAMES_PER_UNIT, 3, height, width)
        current = frames[:, 1:].reshape(batch * PAIRS_PER_UNIT, 3, height, width)
        previous = frames[:, :-1].reshape(batch * PAIRS_PER_UNIT, 3, height, width)
        features = self.encode_frame(current, previous)
        quarter, eighth, bottom = (level.reshape(batch, PAIRS_PER_UNIT * level.shape[1], *level.shape[2:])
                                   for level in features)
        return self._decode(quarter, eighth, bottom, return_vis)

    def reparameterize(self) -> "StreamNet":
        """The deployed network, in place and in eval mode: RepVGG branches and BatchNorm folded
        into plain convolutions (``repvgg.fuse_for_inference``). It answers as before to rounding
        and can no longer train or be saved as a record."""
        repvgg.fuse_for_inference(self)
        return self

    def streaming_halves(self, visibility: bool = False) -> Tuple["StreamEncoder", "StreamDecoder"]:
        """The network as the two modules a streaming prepass runs (and exports to TensorRT), both
        sharing this network's weights: ``StreamEncoder`` (one transition at a time) and
        ``StreamDecoder`` (one unit from its four transitions' features; with `visibility`, the
        decoder that also answers the visibility head's logits, a ValueError for a network without
        the head). A detector streams the model it holds when it has this method (``ball_detectors``)."""
        return StreamEncoder(self), StreamDecoder(self, visibility=visibility)


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
    layout ``torch.cat`` of the four gives), to the unit's 15-channel output; with `visibility`, to
    ``(grids, visibility logits)``, the logits ``(N, VIS_LOGITS)`` (the head needs nothing the
    decoder does not already hold: the grids, ``fuse4``'s output and the stride-4 features)."""

    def __init__(self, model: StreamNet, visibility: bool = False):
        super().__init__()
        if visibility and not model.has_vis_head:
            raise ValueError(f"this {type(model).__name__} has no visibility head to decode with")
        self.model = model
        self.visibility = bool(visibility)

    def forward(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> Output:
        return self.model._decode(quarter, eighth, bottom, self.visibility)


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
            grids = model.decode(cached[-PAIRS_PER_UNIT:])
            assert isinstance(grids, torch.Tensor)
            outputs.append(grids)
    return outputs


def build_for(state_dict: Mapping[str, torch.Tensor]) -> StreamNet:
    """A fresh network of the shape `state_dict` loads into: with the visibility head of the
    variant the record carries (``vis_pool_of``), or without one."""
    pool = vis_pool_of(state_dict)
    return StreamNet(vis_head=pool is not None, vis_pool=pool or "global")


def load_model(weights: Path, device: torch.device) -> StreamNet:
    """The FP16 eval network from the ``.pt`` record ``heatmapnet.save_model`` wrote, in its
    trainable form (the trainer fine-tunes from it), with the visibility head of the variant the
    record carries, if one (``vis_pool_of``); ``reparameterize`` gives the deployed one."""
    state_dict = read_record(Path(weights), ARCH)["state_dict"]
    model = build_for(state_dict)
    model.load_state_dict(state_dict)
    return model.eval().to(device).half()
