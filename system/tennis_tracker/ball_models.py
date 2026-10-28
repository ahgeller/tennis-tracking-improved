"""The ball-detector architectures and the one front door to their weights.

Five networks answer the five-frame ball question: ``gridtracknet.GridTrackNet`` (``.npz``
weights, the upstream export) and the heat-map networks on the 192x108 grid, each a ``.pt``
record carrying its ``arch``: ``heatmapnet.HeatmapNet`` and the fast candidates of the
optimization plan, ``heatmap_lite.HeatmapLite``, ``streamnet.StreamNet`` (whose encoder runs once
per frame) and ``wasbnet.WasbNet``. ``Config.model_path`` stays the one setting that chooses the network: ``arch_of`` reads
the architecture off the weights file (the suffix for an ``.npz``, the record's ``arch`` for a
``.pt``), ``load_ball_model`` dispatches on it and the trainer names its output files by the
architecture's suffix (``WEIGHTS_SUFFIX``). Both classes carry the same attributes (``ARCH``,
``GRID_COLS``, ``GRID_ROWS``, ``INPUT_CHANNELS``, ``FRAMES_PER_UNIT``), the duck-typed contract
``BallModel`` spells, so the detector, the decoder and the trainer read the model they hold and
never ask which class it is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import torch
from torch import nn

from . import gridtracknet, heatmap_lite, heatmapnet, streamnet, wasbnet

GRIDTRACKNET = gridtracknet.ARCH
HEATMAP_S4 = heatmapnet.ARCH
HEATMAP_LITE = heatmap_lite.ARCH
STREAM_S4 = streamnet.ARCH
WASB_5F = wasbnet.ARCH
ARCHITECTURES = (GRIDTRACKNET, HEATMAP_S4, HEATMAP_LITE, STREAM_S4, WASB_5F)
MODEL_CLASSES: Dict[str, Type[nn.Module]] = {
    GRIDTRACKNET: gridtracknet.GridTrackNet, HEATMAP_S4: heatmapnet.HeatmapNet, HEATMAP_LITE: heatmap_lite.HeatmapLite,
    STREAM_S4: streamnet.StreamNet, WASB_5F: wasbnet.WasbNet}
WEIGHTS_SUFFIX: Dict[str, str] = {GRIDTRACKNET: ".npz", HEATMAP_S4: ".pt", HEATMAP_LITE: ".pt", STREAM_S4: ".pt",
                                  WASB_5F: ".pt"}
_MODULES = {GRIDTRACKNET: gridtracknet, HEATMAP_S4: heatmapnet, HEATMAP_LITE: heatmap_lite,     # each one's load_model
            STREAM_S4: streamnet, WASB_5F: wasbnet}
_RECORD_SUFFIX = ".pt"             # the suffix of the architectures whose file names the architecture inside


def model_class(arch: str) -> Type[nn.Module]:
    """The class behind an architecture id; an unknown id is a ValueError naming the known ones."""
    if arch not in MODEL_CLASSES:
        raise ValueError(f"unknown ball-detector architecture {arch!r}; one of {ARCHITECTURES}")
    return MODEL_CLASSES[arch]


def weights_suffix(arch: str) -> str:
    """The suffix an architecture's weights files carry: ``.npz`` for GridTrackNet, ``.pt`` for
    the heat-map networks."""
    model_class(arch)
    return WEIGHTS_SUFFIX[arch]


def arch_of(weights: Path) -> str:
    """The architecture a weights path selects: GridTrackNet for an ``.npz``; for a ``.pt`` the
    ``arch`` its record names (``heatmapnet.read_record``), which must be one of the ``.pt``
    architectures. Another suffix, an unreadable record or an unknown ``arch`` is a ValueError."""
    suffix = Path(weights).suffix.lower()
    if suffix == WEIGHTS_SUFFIX[GRIDTRACKNET]:
        return GRIDTRACKNET
    if suffix != _RECORD_SUFFIX:
        raise ValueError(f"{weights}: ball-detector weights end in .npz (GridTrackNet) or .pt (HeatmapNet)")
    arch = heatmapnet.read_record(Path(weights))["arch"]
    if WEIGHTS_SUFFIX.get(arch) != _RECORD_SUFFIX:
        recorded = sorted(name for name, suffix in WEIGHTS_SUFFIX.items() if suffix == _RECORD_SUFFIX)
        raise ValueError(f"{weights} names the architecture {arch!r}, not one a {_RECORD_SUFFIX} record holds ({recorded})")
    return arch


def load_ball_model(weights: Path, device: torch.device) -> nn.Module:
    """The FP16 eval model on ``device`` for `weights`, built by its architecture's loader
    (``arch_of``): ``gridtracknet.load_model``, ``heatmapnet.load_model`` and so on. It is the
    trainable form, which the trainer fine-tunes from; ``deployed`` gives the inference form."""
    return _MODULES[arch_of(weights)].load_model(Path(weights), device)


def deployed(model: nn.Module) -> nn.Module:
    """`model` in its inference form, in place: a network with a ``reparameterize`` (the RepVGG
    and BatchNorm folds of ``repvgg.fuse_for_inference``) is fused, which answers the same to
    rounding with fewer, plainer layers (what ONNX and TensorRT should be given); any other is
    returned as it is. A fused network can no longer be trained or saved as a record."""
    reparameterize = getattr(model, "reparameterize", None)
    return reparameterize() if callable(reparameterize) else model
