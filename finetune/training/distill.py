"""Knowledge distillation for the ball detector: a teacher network (by default the tracker's own
``system/models/ball_detector.pt``) answers every training unit, and the student learns its soft
heat maps and offsets beside the labels, and optionally its features.

The labelled loss is untouched (``loop.unit_loss``: the focal loss on the labels, the ignored
frames of a pseudo-labelled clip left out); ``Distiller.loss`` adds, on every frame, labelled or
not:

- the soft heat-map loss against the teacher's heat (``DISTILL_LOSSES``): ``qfl``, the quality
  focal loss (Li et al., Generalized Focal Loss, arXiv 2006.04388; BlurBall arXiv 2509.18387
  trains its heat maps with it), the binary cross-entropy against the teacher's probability
  scaled by ``|teacher - student| ** QFL_BETA``, the continuous form of the focal loss the
  labels drive; or ``mse``, FPD's (Zhang et al., Fast Pose Distillation, arXiv 1811.05419, Eq. 4:
  the squared L2 between the two confidence maps, per map, averaged over maps). Both are summed
  over the cells of a frame and averaged over frames, the scale of the labelled loss (a sum over
  cells normalised by the balls, about one a frame); FPD found the weight insensitive across
  0.05 to 0.99 with equal weight best (its Table 8), so ``--distill-weight`` defaults to 1;
- the teacher's offsets, as an L1 weighted by the teacher's heat, so only where the teacher sees
  a ball (``DISTILL_OFFSET_WEIGHT``, the labelled offset loss's weight);
- with ``--mgd-weight`` above 0, Masked Generative Distillation (Yang et al., arXiv 2205.01529) of
  the decoder features at strides 8 and 4 (``feature_layers``): a 1x1 adapter takes the student's
  feature to the teacher's width, a random ``MGD_MASK_RATIO`` of its pixels is zeroed, a 3x3 conv,
  ReLU, 3x3 conv block generates the teacher's full feature from what is left, and the squared
  error, summed and divided by the batch, is the loss (the paper's one-stage detection setting:
  ratio 0.65, weight 2e-5). The adapters train with the student and live in the checkpoint.

Where the teacher's answers come from (``--teacher-maps``):

- ``online`` (the default): the teacher runs on the very batch the student sees, augmentations
  included, every step. Beyer et al. (arXiv 2106.05237, "a good teacher is patient and
  consistent") found consistent views the one setting that reaches the best student; a pasted
  ball is then answered by the teacher too. It costs one teacher forward a step.
- ``cache``: the teacher's dense-window answers (every frame averaged over the up to five units
  that cover it, as the tracker's prepass does, ``ball_detectors._HeatAverage``) precomputed once
  for every cached frame (``build_teacher_maps``) and stored as float16 192x108 maps; the dataset
  carries them through the flip, the time reversal and a translation held to whole cells, and
  the photometric augmentations leave them as they are (Noisy Student, arXiv 1911.04252: the
  teacher is not noised). A pasted ball is written into them as the labels have it. Features
  are not cached, so MGD needs ``online``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from tennis_tracker.ball_models import load_ball_model
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH
from finetune.data.fileio import write_atomic
from finetune.training.frame_cache import TrainingUnit

DISTILL_LOSSES = ("qfl", "mse")
TEACHER_MAPS = ("online", "cache")
QFL_BETA = 2.0                 # Generalized Focal Loss's modulating exponent
DISTILL_OFFSET_WEIGHT = 1.0    # the teacher's offsets against the student's, as loop.HEAT_OFFSET_WEIGHT weighs the labels'
MGD_MASK_RATIO = 0.65          # MGD's one-stage detection setting
MGD_WEIGHT = 2e-5              # ... and its loss weight there
FEATURE_STRIDES = (8, 4)       # the decoder resolutions MGD distils
EPSILON = 1e-6
TEACHER_DIR = "teacher_maps"   # the default maps folder under the frame cache, see build_teacher_maps
TEACHER_BATCH = 8              # units per teacher call while building the maps


@dataclass(frozen=True)
class DistillConfig:
    """What a distilled run does, as ``config.json`` records it (``distill``): the teacher's
    weights, the heat-map loss, its weight, where the teacher's answers come from and the MGD
    weight (0: none)."""
    teacher: str
    loss: str = "qfl"
    weight: float = 1.0
    maps: str = "online"
    mgd_weight: float = 0.0

    def record(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- the losses

def split(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A 15-channel answer as (heat (B, 5, H, W), offsets (B, 5, 2, H, W)), in float32."""
    grids = output.float().reshape(output.shape[0], FRAMES_PER_UNIT, 3, output.shape[-2], output.shape[-1])
    return grids[:, :, 0], grids[:, :, 1:3]


def heat_distillation(student: torch.Tensor, teacher: torch.Tensor, kind: str) -> torch.Tensor:
    """The soft heat-map loss of `kind` (``DISTILL_LOSSES``) between two (B, 5, H, W) heat maps in
    [0, 1], summed over each frame's cells and averaged over frames; the teacher's carries no
    gradient."""
    teacher = teacher.detach().clamp(0.0, 1.0)
    if kind == "mse":
        per_cell = (student - teacher).square()
    elif kind == "qfl":
        prediction = student.clamp(EPSILON, 1 - EPSILON)
        entropy = -(teacher * prediction.log() + (1 - teacher) * (1 - prediction).log())
        per_cell = (teacher - prediction).abs().pow(QFL_BETA) * entropy
    else:
        raise ValueError(f"unknown distillation loss {kind!r}; one of {DISTILL_LOSSES}")
    return per_cell.sum(dim=(-2, -1)).mean()


def offset_distillation(student: torch.Tensor, teacher: torch.Tensor, teacher_heat: torch.Tensor) -> torch.Tensor:
    """The L1 between the two (B, 5, 2, H, W) offset maps, each cell weighted by the teacher's heat
    there, divided by the total weight (at least one): the offsets only matter where a ball is."""
    weight = teacher_heat.detach().unsqueeze(2)
    return ((student - teacher.detach()).abs() * weight).sum() / weight.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- features (MGD)

def _children_outputs(model: nn.Module, probe: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    """Every top-level child's output in the order the forward ran them, for one `probe` batch."""
    seen: List[Tuple[str, torch.Tensor]] = []
    handles = [child.register_forward_hook(lambda module, inputs, output, name=name: seen.append((name, output)))
               for name, child in model.named_children()]
    try:
        with torch.no_grad():
            model(probe)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def feature_layers(model: nn.Module, strides: Sequence[int] = FEATURE_STRIDES) -> Dict[int, Tuple[str, int]]:
    """For each stride in `strides`, the top-level child of `model` whose output is the feature
    MGD distils there, and its channel count: the last child to run whose output is a whole
    batch at that stride, the child that runs last (the head) excepted. On HeatmapNet these are
    the decoder blocks ``up1`` (stride 8) and ``up2`` (stride 4), the neck features MGD distils on
    detectors; any architecture that runs its decoder through top-level children answers the
    same way, with no per-architecture list. A stride with no such child is a ValueError."""
    was_training = model.training
    model.eval()
    try:
        parameter = next(model.parameters())
        probe = torch.zeros(2, int(getattr(model, "INPUT_CHANNELS")), HEIGHT, WIDTH, device=parameter.device, dtype=parameter.dtype)
        outputs = _children_outputs(model, probe)
    finally:
        model.train(was_training)
    layers: Dict[int, Tuple[str, int]] = {}
    for stride in strides:
        size = (HEIGHT // stride, WIDTH // stride)
        for name, output in outputs[:-1]:
            if isinstance(output, torch.Tensor) and output.dim() == 4 and output.shape[0] == 2 \
                    and tuple(output.shape[-2:]) == size:
                layers[stride] = (name, int(output.shape[1]))
        if stride not in layers:
            raise ValueError(f"{type(model).__name__} has no top-level layer answering a whole batch at stride "
                             f"{stride}: MGD cannot find the feature to distil")
    return layers


class FeatureTap:
    """The latest outputs of named top-level children of a model, kept by forward hooks."""

    def __init__(self, model: nn.Module, names: Iterable[str]) -> None:
        self.outputs: Dict[str, torch.Tensor] = {}
        children = dict(model.named_children())
        self.handles = [children[name].register_forward_hook(
            lambda module, inputs, output, name=name: self.outputs.__setitem__(name, output)) for name in names]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class Generator(nn.Module):
    """MGD's block for one level: the 1x1 adapter to the teacher's width and the 3x3 conv, ReLU,
    3x3 conv that generates the teacher's feature from the masked student feature."""

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        self.align = nn.Conv2d(student_channels, teacher_channels, 1)
        self.generate = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1))

    def forward(self, feature: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        return self.generate(self.align(feature) * keep)


# --------------------------------------------------------------------------- the distiller

class Distiller(nn.Module):
    """The teacher and the distillation losses of one run (see the module docstring). Its own
    parameters are MGD's generators (none without MGD), which the optimizer trains beside the
    student and ``state_dict`` carries into the checkpoint; the teacher is frozen and not part of
    that state."""

    def __init__(self, config: DistillConfig, student: nn.Module, device: torch.device) -> None:
        super().__init__()
        if config.loss not in DISTILL_LOSSES:
            raise ValueError(f"unknown distillation loss {config.loss!r}; one of {DISTILL_LOSSES}")
        if config.maps not in TEACHER_MAPS:
            raise ValueError(f"--teacher-maps is one of {TEACHER_MAPS}, not {config.maps!r}")
        if config.mgd_weight and config.maps != "online":
            raise ValueError("MGD distils the teacher's features, which are not cached: it needs --teacher-maps online")
        self.config = config
        teacher: Optional[nn.Module] = None
        if config.maps == "online":
            teacher = load_ball_model(Path(config.teacher), device).eval()
            if device.type != "cuda":               # float16 is the tracker's on CUDA; the CPU runs float32
                teacher = teacher.float()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            if (teacher.GRID_COLS, teacher.GRID_ROWS) != (student.GRID_COLS, student.GRID_ROWS):
                raise ValueError(f"the teacher answers on a {teacher.GRID_COLS}x{teacher.GRID_ROWS} grid and the "
                                 f"student on {student.GRID_COLS}x{student.GRID_ROWS}: distillation needs the same grid")
        self.teacher = [teacher]           # in a list: not a submodule, so neither trained nor in state_dict
        self.generators = nn.ModuleDict()
        self.taps: List[FeatureTap] = []
        self.levels: List[Tuple[str, str]] = []      # (student layer, teacher layer) per stride
        if config.mgd_weight:
            assert teacher is not None
            student_layers, teacher_layers = feature_layers(student), feature_layers(teacher)
            for stride in FEATURE_STRIDES:
                (s_name, s_channels), (t_name, t_channels) = student_layers[stride], teacher_layers[stride]
                self.generators[str(stride)] = Generator(s_channels, t_channels)
                self.levels.append((s_name, t_name))
            self.taps = [FeatureTap(student, [name for name, _ in self.levels]),
                         FeatureTap(teacher, [name for _, name in self.levels])]
        self.to(device)

    def teacher_answer(self, images: torch.Tensor) -> torch.Tensor:
        """The frozen teacher's 15-channel answer to `images` (float16 on CUDA), no gradient."""
        teacher = self.teacher[0]
        assert teacher is not None
        parameter = next(teacher.parameters())
        with torch.no_grad():
            return teacher(images.to(parameter.dtype))

    def loss(self, prediction: torch.Tensor, images: torch.Tensor, cached: Optional[torch.Tensor]
             ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """The distillation loss of one batch (the student's `prediction` for `images`; `cached`
        the dataset's teacher maps in ``cache`` mode) and its parts as floats for the log."""
        answer = cached if self.config.maps == "cache" else self.teacher_answer(images)
        if answer is None:
            raise ValueError("--teacher-maps cache needs the dataset to carry the teacher maps")
        student_heat, student_offsets = split(prediction)
        teacher_heat, teacher_offsets = split(answer)
        heat = heat_distillation(student_heat, teacher_heat, self.config.loss)
        offsets = offset_distillation(student_offsets, teacher_offsets, teacher_heat)
        total = self.config.weight * (heat + DISTILL_OFFSET_WEIGHT * offsets)
        parts = {"kd_heat": float(heat.detach()), "kd_off": float(offsets.detach())}
        if self.config.mgd_weight:
            features = self.mgd()
            total = total + self.config.mgd_weight * features
            parts["mgd"] = float(features.detach())
        return total, parts

    def mgd(self) -> torch.Tensor:
        """MGD's loss on the features the taps kept from this step's two forwards."""
        student_tap, teacher_tap = self.taps
        total = torch.zeros((), device=next(self.generators.parameters()).device)
        for stride, (s_name, t_name) in zip(FEATURE_STRIDES, self.levels):
            feature, target = student_tap.outputs[s_name].float(), teacher_tap.outputs[t_name].float().detach()
            keep = (torch.rand(feature.shape[0], 1, *feature.shape[-2:], device=feature.device) >= MGD_MASK_RATIO).float()
            generated = self.generators[str(stride)](feature, keep)
            total = total + (generated - target).square().sum() / feature.shape[0]
        return total


# --------------------------------------------------------------------------- the cached teacher maps

def maps_file(maps_dir: Path, folder: str) -> Path:
    """Where the teacher maps of one cached clip folder (``frames/<clip>``, ``pseudo/frames/<clip>``)
    are kept: ``<maps_dir>/<folder with / as __>.npy`` with an ``.json`` index beside (the
    trainer's `maps_dir` is ``<frame cache>/teacher_maps`` unless ``--teacher-maps-dir`` says)."""
    return maps_dir / (folder.replace("/", "__") + ".npy")


def unit_folder(unit: TrainingUnit) -> str:
    return unit["frames"][0].rsplit("/", 1)[0]


def phase_runs(units: Sequence[TrainingUnit]) -> List[List[Tuple[int, str]]]:
    """The cached frames of one clip folder's units as runs of consecutive frames of one phase
    ((source index, cache path) pairs): the sequences the dense windows slide over. Two frames
    are consecutive when they are one unit stride apart."""
    frames: Dict[int, str] = {}
    strides = set()
    for unit in units:
        frames.update(zip(unit["indices"], unit["frames"]))
        strides.add(unit["indices"][1] - unit["indices"][0])
    stride = min(strides)
    runs: List[List[Tuple[int, str]]] = []
    for phase in range(stride):
        indices = sorted(index for index in frames if index % stride == phase)
        current: List[Tuple[int, str]] = []
        for index in indices:
            if current and index != current[-1][0] + stride:
                runs.append(current)
                current = []
            current.append((index, frames[index]))
        if current:
            runs.append(current)
    return runs


def dense_answers(teacher: nn.Module, run: Sequence[Tuple[int, str]], data_dir: Path, device: torch.device
                  ) -> np.ndarray:
    """The teacher's answers over one run of consecutive frames, a unit starting at every frame
    and every frame's answers averaged grid by grid over the units covering it (the tracker's
    dense prepass, ``Config.gridtracknet_window_step`` 1): (frames, 3, rows, cols) float16."""
    images = []
    for _, relative in run:
        image = cv2.imread(str(data_dir / relative), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(data_dir / relative)
        images.append(torch.from_numpy(np.ascontiguousarray(np.moveaxis(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), -1, 0))))
    frames = torch.stack(images).to(device).float().div_(255.0)
    starts = list(range(0, len(run) - FRAMES_PER_UNIT + 1))
    sums: Optional[torch.Tensor] = None
    counts = torch.zeros(len(run), device=device)
    parameter = next(teacher.parameters())
    with torch.no_grad():
        for first in range(0, len(starts), TEACHER_BATCH):
            chunk = starts[first:first + TEACHER_BATCH]
            batch = torch.stack([frames[start:start + FRAMES_PER_UNIT].reshape(-1, HEIGHT, WIDTH) for start in chunk])
            output = teacher(batch.to(parameter.dtype)).float()
            grids = output.reshape(len(chunk), FRAMES_PER_UNIT, 3, output.shape[-2], output.shape[-1])
            if sums is None:
                sums = torch.zeros(len(run), 3, output.shape[-2], output.shape[-1], device=device)
            for row, start in enumerate(chunk):
                sums[start:start + FRAMES_PER_UNIT] += grids[row]
                counts[start:start + FRAMES_PER_UNIT] += 1
    assert sums is not None
    return (sums / counts.view(-1, 1, 1, 1)).cpu().numpy().astype(np.float16)


def build_teacher_maps(teacher_weights: Path, data_dir: Path, units: Sequence[TrainingUnit], device: torch.device,
                       maps_dir: Path, rebuild: bool = False) -> Dict[str, float]:
    """Precompute the teacher's dense answers for every frame `units` read (from the frame cache
    at `data_dir`) into `maps_dir`, one ``.npy`` (frames, 3, rows, cols) float16 per clip folder
    with its ``.json`` index (frame path -> row, and the teacher's file, size and time, so a
    changed teacher rebuilds), and return the size report. A folder whose index names this
    teacher and holds every frame asked for is kept unless `rebuild`."""
    teacher = load_ball_model(teacher_weights, device).eval()
    if device.type != "cuda":
        teacher = teacher.float()
    stat = Path(teacher_weights).stat()
    signature = [str(Path(teacher_weights).name), stat.st_size, stat.st_mtime_ns]
    by_folder: Dict[str, List[TrainingUnit]] = {}
    for unit in units:
        by_folder.setdefault(unit_folder(unit), []).append(unit)
    started, built, frames, stored = time.perf_counter(), 0, 0, 0
    for folder, folder_units in sorted(by_folder.items()):
        path = maps_file(maps_dir, folder)
        index_path = path.with_suffix(".json")
        if not rebuild and path.is_file() and index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            needed = {relative for unit in folder_units for relative in unit["frames"]}
            if index.get("teacher") == signature and needed <= set(index["rows"]):
                frames += len(index["rows"])
                stored += path.stat().st_size
                continue
        rows: Dict[str, int] = {}
        arrays = []
        for run in phase_runs(folder_units):
            if len(run) < FRAMES_PER_UNIT:
                continue
            arrays.append(dense_answers(teacher, run, data_dir, device))
            for _, relative in run:
                rows[relative] = len(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, np.concatenate(arrays))
        temporary.replace(path)
        write_atomic(index_path, json.dumps({"teacher": signature, "rows": rows}))
        built += 1
        frames += len(rows)
        stored += path.stat().st_size
    report = {"folders": float(len(by_folder)), "built": float(built), "frames": float(frames), "bytes": float(stored),
              "seconds": time.perf_counter() - started}
    print(f"[distill] teacher maps: {frames} frames in {len(by_folder)} clip folders ({built} built), "
          f"{stored / 2**30:.2f} GB float16 in {maps_dir}, {report['seconds']:.0f} s")
    return report


class TeacherMaps:
    """Reads the cached teacher maps of a unit's five frames (``build_teacher_maps``) from
    `maps_dir`, each clip folder's array memory-mapped once per process."""

    def __init__(self, maps_dir: Path) -> None:
        self.maps_dir = maps_dir
        self.open: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}

    def folder(self, folder: str) -> Tuple[np.ndarray, Dict[str, int]]:
        if folder not in self.open:
            path = maps_file(self.maps_dir, folder)
            index = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.open[folder] = (np.load(path, mmap_mode="r"), index["rows"])
        return self.open[folder]

    def unit(self, unit: TrainingUnit) -> np.ndarray:
        """(15, rows, cols) float32: the five frames' maps in the unit's order."""
        array, rows = self.folder(unit_folder(unit))
        return np.concatenate([np.asarray(array[rows[relative]], dtype=np.float32) for relative in unit["frames"]])


# --------------------------------------------------------------------------- the maps under the augmentations

def flip_maps(maps: np.ndarray) -> np.ndarray:
    """The (15, rows, cols) maps of a mirrored unit: every grid mirrored left to right, and the x
    offset of a cell taken to one minus itself (a point at cell c, offset o, lands at cell
    cols - 1 - c, offset 1 - o, the mirror ``dataset.flipped`` applies to the labels)."""
    out = maps[:, :, ::-1].copy()
    out[1::3] = 1.0 - out[1::3]
    return out


def shift_maps(maps: np.ndarray, cells_x: int, cells_y: int) -> np.ndarray:
    """The maps of a unit translated by whole cells (right and down), the cells uncovered at the
    border set to zero (no ball there)."""
    out = np.zeros_like(maps)
    rows, cols = maps.shape[1:]
    source_y = slice(max(0, -cells_y), rows - max(0, cells_y))
    source_x = slice(max(0, -cells_x), cols - max(0, cells_x))
    target_y = slice(max(0, cells_y), rows - max(0, -cells_y))
    target_x = slice(max(0, cells_x), cols - max(0, -cells_x))
    out[:, target_y, target_x] = maps[:, source_y, source_x]
    return out


def reverse_maps(maps: np.ndarray) -> np.ndarray:
    """The maps of a unit in reverse time order."""
    return np.ascontiguousarray(maps.reshape(FRAMES_PER_UNIT, 3, *maps.shape[1:])[::-1].reshape(maps.shape))


def paste_into_maps(maps: np.ndarray, pasted: np.ndarray) -> np.ndarray:
    """The teacher's maps with a pasted ball's target (``dataset.make_target`` of the pasted points
    alone, (15, rows, cols)) laid in: the heat by maximum and the offsets where the pasted target
    has its peak, since the cached teacher never saw the pasted ball."""
    out = maps.copy()
    out[0::3] = np.maximum(out[0::3], pasted[0::3])
    peak = pasted[0::3] == 1.0
    for channel in (1, 2):
        out[channel::3][peak] = pasted[channel::3][peak]
    return out
