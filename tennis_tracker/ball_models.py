"""The ball-detector architectures and the one front door to their weights.

Four networks answer the five-frame ball question: ``gridtracknet.GridTrackNet`` (``.npz``
weights, the upstream export), ``heatmapnet.HeatmapNet``, ``memorynet.MemoryNet`` and
``heatmapnet_s2.HeatmapNetS2`` (``.pt`` records carrying their ``arch``). ``Config.model_path``
stays the one setting that chooses the network: ``arch_of`` reads the architecture off the
weights file (the suffix for an ``.npz``, the record's ``arch`` for a ``.pt``),
``load_ball_model`` dispatches on it and the trainer names its output files by the architecture's
suffix (``WEIGHTS_SUFFIX``). All four classes carry the same attributes (``ARCH``, ``STATEFUL``,
``GRID_COLS``, ``GRID_ROWS``, ``INPUT_CHANNELS``, ``FRAMES_PER_UNIT``), the duck-typed contract
``BallModel`` spells, so the detector, the decoder and the trainer read the model they hold and
never ask which class it is. ``STATEFUL`` says whether ``forward`` takes and returns a state
carried between consecutive units of one phase (``memorynet``); the prepass and the trainer
branch on it alone.

Two architectures are built on HeatmapNet and may start training from one of its records, the
parent (``PARENTS``): the shared layers are copied and what the parent does not carry starts
fresh (``PARENT_START_NOTE`` says what, in the trainer's words). ``from_parent`` dispatches to
each module's ``from_heatmap``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import torch
from torch import nn

from . import gridtracknet, heatmapnet, heatmapnet_s2, memorynet

GRIDTRACKNET = gridtracknet.ARCH
HEATMAP_S4 = heatmapnet.ARCH
MEMORY_S4 = memorynet.ARCH
HEATMAP_S2 = heatmapnet_s2.ARCH
ARCHITECTURES = (GRIDTRACKNET, HEATMAP_S4, MEMORY_S4, HEATMAP_S2)
MODEL_CLASSES: Dict[str, Type[nn.Module]] = {GRIDTRACKNET: gridtracknet.GridTrackNet, HEATMAP_S4: heatmapnet.HeatmapNet,
                                             MEMORY_S4: memorynet.MemoryNet, HEATMAP_S2: heatmapnet_s2.HeatmapNetS2}
WEIGHTS_SUFFIX: Dict[str, str] = {GRIDTRACKNET: ".npz", HEATMAP_S4: ".pt", MEMORY_S4: ".pt", HEATMAP_S2: ".pt"}
_MODULES = {GRIDTRACKNET: gridtracknet, HEATMAP_S4: heatmapnet, MEMORY_S4: memorynet,
            HEATMAP_S2: heatmapnet_s2}                                                    # each one's load_model
PARENTS: Dict[str, str] = {MEMORY_S4: memorynet.PARENT_ARCH, HEATMAP_S2: heatmapnet_s2.PARENT_ARCH}
PARENT_START_NOTE: Dict[str, str] = {MEMORY_S4: "the shared layers are copied, the memory starts fresh",
                                     HEATMAP_S2: "the shared layers are copied, the stride-2 decoder block and the head start fresh"}
_RECORD_SUFFIX = ".pt"             # the suffix of the architectures whose file names the architecture inside


def model_class(arch: str) -> Type[nn.Module]:
    """The class behind an architecture id; an unknown id is a ValueError naming the known ones."""
    if arch not in MODEL_CLASSES:
        raise ValueError(f"unknown ball-detector architecture {arch!r}; one of {ARCHITECTURES}")
    return MODEL_CLASSES[arch]


def weights_suffix(arch: str) -> str:
    """The suffix an architecture's weights files carry: ``.npz`` for GridTrackNet, ``.pt`` for
    the others."""
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
        raise ValueError(f"{weights}: ball-detector weights end in .npz (GridTrackNet) or .pt (the other architectures)")
    arch = heatmapnet.read_record(Path(weights))["arch"]
    if WEIGHTS_SUFFIX.get(arch) != _RECORD_SUFFIX:
        recorded = sorted(name for name, suffix in WEIGHTS_SUFFIX.items() if suffix == _RECORD_SUFFIX)
        raise ValueError(f"{weights} names the architecture {arch!r}, not one a {_RECORD_SUFFIX} record holds ({recorded})")
    return arch


def load_ball_model(weights: Path, device: torch.device) -> nn.Module:
    """The FP16 eval model on ``device`` for `weights`, built by its architecture's loader
    (``arch_of``): ``gridtracknet.load_model``, ``heatmapnet.load_model``, ``memorynet.load_model``
    or ``heatmapnet_s2.load_model``."""
    return _MODULES[arch_of(weights)].load_model(Path(weights), device)


def from_parent(arch: str, weights: Path, device: torch.device) -> nn.Module:
    """The float32 training model of `arch` started from its parent's record at `weights`
    (``PARENTS``): ``memorynet.from_heatmap`` or ``heatmapnet_s2.from_heatmap``, each of which
    refuses a record that is not the parent's. An architecture without a parent is a ValueError."""
    if arch not in PARENTS:
        raise ValueError(f"{arch} has no parent architecture to start from; one of {tuple(PARENTS)} does")
    return _MODULES[arch].from_heatmap(Path(weights), device)
