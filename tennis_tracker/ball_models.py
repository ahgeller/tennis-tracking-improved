"""The ball-detector architectures and the one front door to their weights.

Two networks answer the five-frame ball question: ``gridtracknet.GridTrackNet`` (``.npz``
weights, the upstream export) and ``heatmapnet.HeatmapNet`` (``.pt`` weights). The weights
file's suffix names the architecture, so ``Config.model_path`` stays the one setting that
chooses it: ``load_ball_model`` dispatches on it and the trainer names its output files by it.
Both classes carry the same attributes (``ARCH``, ``GRID_COLS``, ``GRID_ROWS``,
``INPUT_CHANNELS``, ``FRAMES_PER_UNIT``), the duck-typed contract ``BallModel`` spells, so the
detector, the decoder and the trainer read the model they hold and never ask which class it is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import torch
from torch import nn

from . import gridtracknet, heatmapnet

GRIDTRACKNET = gridtracknet.ARCH
HEATMAP_S4 = heatmapnet.ARCH
ARCHITECTURES = (GRIDTRACKNET, HEATMAP_S4)
MODEL_CLASSES: Dict[str, Type[nn.Module]] = {GRIDTRACKNET: gridtracknet.GridTrackNet, HEATMAP_S4: heatmapnet.HeatmapNet}
WEIGHTS_SUFFIX: Dict[str, str] = {GRIDTRACKNET: ".npz", HEATMAP_S4: ".pt"}
_ARCH_OF_SUFFIX = {suffix: arch for arch, suffix in WEIGHTS_SUFFIX.items()}


def model_class(arch: str) -> Type[nn.Module]:
    """The class behind an architecture id; an unknown id is a ValueError naming the known ones."""
    if arch not in MODEL_CLASSES:
        raise ValueError(f"unknown ball-detector architecture {arch!r}; one of {ARCHITECTURES}")
    return MODEL_CLASSES[arch]


def weights_suffix(arch: str) -> str:
    """The suffix an architecture's weights files carry: ``.npz`` for GridTrackNet, ``.pt`` for
    HeatmapNet."""
    model_class(arch)
    return WEIGHTS_SUFFIX[arch]


def arch_of(weights: Path) -> str:
    """The architecture a weights path selects, by its suffix; another suffix is a ValueError."""
    suffix = Path(weights).suffix.lower()
    if suffix not in _ARCH_OF_SUFFIX:
        raise ValueError(f"{weights}: ball-detector weights end in .npz (GridTrackNet) or .pt (HeatmapNet)")
    return _ARCH_OF_SUFFIX[suffix]


def load_ball_model(weights: Path, device: torch.device) -> nn.Module:
    """The FP16 eval model on ``device`` for `weights`: ``gridtracknet.load_model`` for an
    ``.npz``, ``heatmapnet.load_model`` for a ``.pt``."""
    arch = arch_of(weights)
    loader = gridtracknet.load_model if arch == GRIDTRACKNET else heatmapnet.load_model
    return loader(Path(weights), device)
