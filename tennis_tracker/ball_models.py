"""The ball-detector architectures and the one front door to their weights.

Three networks answer the five-frame ball question: ``gridtracknet.GridTrackNet`` (``.npz``
weights, the upstream export), ``heatmapnet.HeatmapNet`` and ``memorynet.MemoryNet`` (both
``.pt`` records carrying their ``arch``). ``Config.model_path`` stays the one setting that
chooses the network: ``arch_of`` reads the architecture off the weights file (the suffix for an
``.npz``, the record's ``arch`` for a ``.pt``), ``load_ball_model`` dispatches on it and the
trainer names its output files by the architecture's suffix (``WEIGHTS_SUFFIX``). All three
classes carry the same attributes (``ARCH``, ``STATEFUL``, ``GRID_COLS``, ``GRID_ROWS``,
``INPUT_CHANNELS``, ``FRAMES_PER_UNIT``), the duck-typed contract ``BallModel`` spells, so the
detector, the decoder and the trainer read the model they hold and never ask which class it is.
``STATEFUL`` says whether ``forward`` takes and returns a state carried between consecutive units
of one phase (``memorynet``); the prepass and the trainer branch on it alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import torch
from torch import nn

from . import gridtracknet, heatmapnet, memorynet

GRIDTRACKNET = gridtracknet.ARCH
HEATMAP_S4 = heatmapnet.ARCH
MEMORY_S4 = memorynet.ARCH
ARCHITECTURES = (GRIDTRACKNET, HEATMAP_S4, MEMORY_S4)
MODEL_CLASSES: Dict[str, Type[nn.Module]] = {GRIDTRACKNET: gridtracknet.GridTrackNet, HEATMAP_S4: heatmapnet.HeatmapNet,
                                             MEMORY_S4: memorynet.MemoryNet}
WEIGHTS_SUFFIX: Dict[str, str] = {GRIDTRACKNET: ".npz", HEATMAP_S4: ".pt", MEMORY_S4: ".pt"}
_MODULES = {GRIDTRACKNET: gridtracknet, HEATMAP_S4: heatmapnet, MEMORY_S4: memorynet}     # each one's load_model
_RECORD_SUFFIX = ".pt"             # the suffix of the architectures whose file names the architecture inside


def model_class(arch: str) -> Type[nn.Module]:
    """The class behind an architecture id; an unknown id is a ValueError naming the known ones."""
    if arch not in MODEL_CLASSES:
        raise ValueError(f"unknown ball-detector architecture {arch!r}; one of {ARCHITECTURES}")
    return MODEL_CLASSES[arch]


def weights_suffix(arch: str) -> str:
    """The suffix an architecture's weights files carry: ``.npz`` for GridTrackNet, ``.pt`` for
    HeatmapNet and MemoryNet."""
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
        raise ValueError(f"{weights}: ball-detector weights end in .npz (GridTrackNet) or .pt (HeatmapNet, MemoryNet)")
    arch = heatmapnet.read_record(Path(weights))["arch"]
    if WEIGHTS_SUFFIX.get(arch) != _RECORD_SUFFIX:
        raise ValueError(f"{weights} names the unknown ball-detector architecture {arch!r}; one of {ARCHITECTURES}")
    return arch


def load_ball_model(weights: Path, device: torch.device) -> nn.Module:
    """The FP16 eval model on ``device`` for `weights`, built by its architecture's loader
    (``arch_of``): ``gridtracknet.load_model``, ``heatmapnet.load_model`` or
    ``memorynet.load_model``."""
    return _MODULES[arch_of(weights)].load_model(Path(weights), device)
