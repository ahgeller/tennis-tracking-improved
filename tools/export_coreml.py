"""Export the ball and player detectors to Core ML, ready to time on an iPhone in Xcode's performance report.

    python tools/export_coreml.py                    # writes output/coreml_speed_test/
    python tools/export_coreml.py --models stream_s4 player

coremltools converts on macOS and Linux only (no Windows wheels), so on this PC it runs in WSL:

    wsl -d Ubuntu --cd <worktree> -- ~/miniforge3/envs/coreml/bin/python tools/export_coreml.py

Every package is an ML Program (FP16 compute, FP16 inputs and outputs, fixed shapes at batch 1,
deployment target iOS 17). ``EXPORTS`` lists what can be exported (``--models``), each with its
source weights, its packages and the sample cases the Mac checks them on:

- ``stream_s4`` (from the deployed network, ``ball_models.deployed``: RepVGG branches and
  BatchNorm folded from the float32 record):
  - ``stream_s4_encoder``: ``StreamNet.encode_frame``, one transition a call: ``frame`` and
    ``previous`` (the same phase's frame before it; at 60 FPS the source frame two back), each
    ``(1, 3, 432, 768)`` RGB in [0, 1], to its features ``quarter``, ``eighth`` and ``bottom``
    (strides 4, 8 and 16), the names ``ball_trt`` gives the TensorRT halves.
  - ``stream_s4_decoder``: ``StreamNet.decode`` of one unit from its four transitions' cached
    features, passed separately (``t1_quarter`` .. ``t4_bottom``, oldest first), so an app hands
    it the encoder's last four outputs without copying them together; out ``grids``.
  - ``stream_s4_unit``: the whole five-frame unit, ``frames`` ``(1, 15, 432, 768)`` to ``grids``
    ``(1, 15, 108, 192)`` (per frame heat, x offset, y offset).
- ``heatmap_s4``: ``heatmap_s4_unit``, the same unit contract.
- ``player``: ``player_detector``, the player YOLO11n (``system/models/sources/player.pt``,
  gitignored, the checkpoint the shipped ``player.engine`` was matched to; another md5 is refused
  without ``--allow-unmatched-player``), ``image`` ``(1, 3, 640, 640)``, the frame stretched as the
  tracker's TensorRT session does, to the raw head ``predictions`` ``(1, 5, 8400)``. NMS stays in
  the app: Ultralytics' Core ML NMS pipeline needs an image input and bakes its own thresholds,
  and the tracker's (score 0.21, IoU 0.10) run on the few anchors above the threshold
  (``engine_boxes``). The court model is not exported: no source weights matching the shipped
  ``courtdetection.engine`` exist (``audit/records/opt_coreml.md``).

The tracker at window step 1 runs one encoder call and one decoder call per source frame, so
stream_s4's per-frame cost is the encoder's time plus the decoder's; at 60 FPS it runs the player
detector on every other frame. A unit's grids are not the tracker's answer for a frame on their
own: it averages each frame's grids over the up to five units that contain it (``HeatRing``)
before it decodes the peak.

Beside the packages: sample units of ``system/sample/pomona.mp4`` (five consecutive same-phase
frames, the tracker's own resize, float16 as the packages take them), the player detector's input
for each unit's first frame, the PyTorch FP32 outputs of the source networks on them, the encoder
features the decoder is checked on, ``manifest.json`` (packages, cases, the PC checks) and the Mac
kit from ``tools/coreml_speed_test/`` (``MAC_SPEED_TEST.md``, ``validate.sh``,
``validate_coreml.py``, ``requirements-mac.txt``).

Core ML predicts only on macOS, so on the PC the export checks the graphs it converts: the fused
network eagerly and as the traced TorchScript coremltools reads, in FP32 and FP16, against the
trainable FP32 network, scored by the Mac validator's own functions (``manifest["pc_validation"]``).
The exit status is 1 when any of those checks fails, 2 when a source checkpoint is missing or
unmatched (the output folder is then left as it was).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import functools
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))
sys.path.append(str(ROOT / "tools"))

from bench_ball_archs import conv_gmac  # noqa: E402
from tennis_tracker import ball_models, heatmapnet, paths, streamnet  # noqa: E402
from tennis_tracker.ball_weights import operating_conf  # noqa: E402
from tennis_tracker.config import Config  # noqa: E402
from tennis_tracker.frame_rate import require_phase_stride  # noqa: E402
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH, frame_tensor  # noqa: E402

KIT = ROOT / "tools" / "coreml_speed_test"
KIT_FILES = ("MAC_SPEED_TEST.md", "validate.sh", "validate_coreml.py", "requirements-mac.txt")
DEFAULT_OUT = ROOT / "output" / "coreml_speed_test"
DEFAULT_VIDEO = paths.ROOT / "sample" / "pomona.mp4"
# Unit starts on pomona: three rallies with the ball in view (both networks' peaks >= 0.8) and one
# stretch without it (peaks < 0.05), found by a scan every 30 frames.
DEFAULT_STARTS = (240, 810, 1800, 660)
DEPLOYMENT_TARGET = "iOS17"
RECORD = "audit/records/opt_coreml.md"

# The tensors' names: the TensorRT export's (``ball_trt``), and the decoder's per-transition inputs.
FRAME_INPUTS = ("frame", "previous")
FEATURE_NAMES = ("quarter", "eighth", "bottom")
UNIT_INPUT = "frames"
GRIDS = "grids"
PAIRS = streamnet.PAIRS_PER_UNIT
RGB = 3                                  # channels per frame of a unit's input

# The player detector: the checkpoint the shipped models/player.engine was matched to (md5, and
# boxes on pomona within FP16 noise; RECORD), at the engine's input size.
DEFAULT_PLAYER_WEIGHTS = paths.MODELS / "sources" / "player.pt"
PLAYER_SOURCE_MD5 = "4de317cc3e2cad7e9196fa438b439544"
PLAYER_SIZE = 640
PLAYER_INPUT = "image"
PLAYER_OUTPUT = "predictions"

HEAT_RING_TEXT = ("The tracker averages each frame's grids over the up to five units that contain it (HeatRing) "
                  "before it decodes the peak, so one unit's peak is not the tracker's answer.")


def decoder_input(pair: int, level: str) -> str:
    """The decoder's input for transition `pair` (1 the oldest, ``PAIRS`` the newest) at `level`."""
    return f"t{pair}_{level}"


DECODER_INPUTS = tuple(decoder_input(pair, level) for pair in range(1, PAIRS + 1) for level in FEATURE_NAMES)


def transition_channels(pair: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """The channel ranges of a unit's input that transition `pair` (1 to ``PAIRS``) reads: its
    frame (unit frame `pair`) and the frame before it."""
    return (RGB * pair, RGB * pair + RGB), (RGB * pair - RGB, RGB * pair)


def frame_text(shape: Sequence[int]) -> str:
    height, width = shape[-2:]
    return f"RGB in [0, 1] (uint8 / 255), {width}x{height}, OpenCV INTER_LINEAR resize of the source frame"


def grids_text(shape: Sequence[int]) -> str:
    rows, cols = shape[-2:]
    return (f"per frame of the unit, oldest first, three {rows}x{cols} planes: the ball heat map (sigmoid), then the "
            f"x and y offsets within the cell (sigmoid, 0..1); a peak at row r, column c is the ball at "
            f"((c + x) * W / {cols}, (r + y) * H / {rows}) in a W x H source frame. {HEAT_RING_TEXT}")


@dataclass(frozen=True)
class TensorSpec:
    """One input or output of a package: its name, its fixed shape and what it holds."""
    name: str
    shape: Tuple[int, ...]
    description: str


@dataclass(frozen=True)
class PackageSpec:
    """One package to export: the fused module it is converted from, the trainable module its
    references come from (same call signature), its tensors, its words, the score its detections
    are kept at (``threshold``) and, for a detector without NMS, the IoU the app suppresses at.
    Made by ``make_package``."""
    name: str
    arch: str
    role: str
    weights: Path
    fused: nn.Module
    trainable: nn.Module
    inputs: Tuple[TensorSpec, ...]
    outputs: Tuple[TensorSpec, ...]
    description: str
    threshold: float
    fold_note: str
    gmac: float
    nms_iou: Optional[float] = None
    notes: Mapping[str, str] = field(default_factory=dict)

    def probes(self, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.zeros(spec.shape, dtype=dtype) for spec in self.inputs)


OutputText = Callable[[Tuple[int, ...]], str]


def _as_tuple(value: Any) -> Tuple[torch.Tensor, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def make_package(*, name: str, arch: str, role: str, weights: Path, fused: nn.Module, trainable: nn.Module,
                 inputs: Sequence[TensorSpec], outputs: Sequence[Tuple[str, OutputText]], description: str,
                 threshold: float, fold_note: str, nms_iou: Optional[float] = None,
                 notes: Optional[Mapping[str, str]] = None) -> PackageSpec:
    """The package, its `outputs` (name, description of the shape) shaped and its GMAC counted by
    one call of `fused` on zeros."""
    probes = tuple(torch.zeros(t.shape) for t in inputs)
    with torch.no_grad():
        values = _as_tuple(fused(*probes))
    if len(values) != len(outputs):
        raise ValueError(f"{name} returns {len(values)} tensors, not {len(outputs)}")
    shaped = tuple(TensorSpec(out_name, tuple(value.shape), text(tuple(value.shape)))
                   for (out_name, text), value in zip(outputs, values))
    return PackageSpec(name=name, arch=arch, role=role, weights=weights, fused=fused, trainable=trainable,
                       inputs=tuple(inputs), outputs=shaped, description=description, threshold=threshold,
                       fold_note=fold_note, gmac=round(conv_gmac(fused, probes), 3), nms_iou=nms_iou,
                       notes=dict(notes or {}))


class WindowDecoder(nn.Module):
    """``StreamNet.decode`` with the four transitions' features as separate inputs, oldest first,
    each transition's quarter, eighth and bottom in turn (``DECODER_INPUTS``)."""

    def __init__(self, model: streamnet.StreamNet):
        super().__init__()
        self.model = model

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        levels = len(FEATURE_NAMES)
        if len(features) != PAIRS * levels:
            raise ValueError(f"the decoder takes {PAIRS * levels} feature maps, not {len(features)}")
        pairs = [cast(streamnet.PairFeatures, tuple(features[levels * i:levels * (i + 1)])) for i in range(PAIRS)]
        return self.model.decode(pairs)


def load_trainable(weights: Path) -> nn.Module:
    """The float32 eval network a ``.pt`` record holds, in its trainable form."""
    arch = ball_models.arch_of(weights)
    model = ball_models.model_class(arch)()
    model.load_state_dict(heatmapnet.read_record(Path(weights), arch)["state_dict"])
    return model.eval()


def fused_copy(model: nn.Module) -> nn.Module:
    """`model`'s inference form (``ball_models.deployed``), leaving `model` as it is."""
    return ball_models.deployed(copy.deepcopy(model)).eval()


def ball_fold_note(arch: str) -> str:
    """What was folded before conversion, for the package metadata."""
    if callable(getattr(ball_models.model_class(arch), "reparameterize", None)):
        return "RepVGG branches and BatchNorm folded (ball_models.deployed)"
    return "no RepVGG blocks; coremltools folds BatchNorm into the convolutions"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_specs(weights: Path, height: int = HEIGHT, width: int = WIDTH) -> List[PackageSpec]:
    """stream_s4's three packages: the encoder, the decoder and the whole unit."""
    trainable = load_trainable(weights)
    if not isinstance(trainable, streamnet.StreamNet):
        raise ValueError(f"{weights} is not a {streamnet.ARCH} record")
    fused = cast(streamnet.StreamNet, fused_copy(trainable))
    package = functools.partial(make_package, arch=streamnet.ARCH, weights=weights, threshold=operating_conf(weights),
                                fold_note=ball_fold_note(streamnet.ARCH))
    frame = (1, RGB, height, width)

    def feature_text(shape: Tuple[int, ...]) -> str:
        return f"the transition's features at stride {height // shape[-2]}, {shape[1]} channels"

    encoder = package(
        name="stream_s4_encoder", role="encoder",
        fused=streamnet.StreamEncoder(fused).eval(), trainable=streamnet.StreamEncoder(trainable).eval(),
        inputs=[TensorSpec("frame", frame, f"the current frame, {frame_text(frame)}"),
                TensorSpec("previous", frame, f"the same phase's frame before it (at 60 FPS the source frame two "
                                              f"back), {frame_text(frame)}")],
        outputs=[(level, feature_text) for level in FEATURE_NAMES],
        description="stream_s4 ball detector, encoder: one frame transition (frame, previous) to its features. Run "
                    "once per frame; the decoder takes the last four transitions' outputs.")
    decoder_inputs = []
    for pair in range(1, PAIRS + 1):
        age = "oldest" if pair == 1 else "newest" if pair == PAIRS else f"number {pair}"
        for level, out in zip(FEATURE_NAMES, encoder.outputs):
            decoder_inputs.append(TensorSpec(decoder_input(pair, level), out.shape,
                                             f"the encoder's '{level}' output for the unit's transition {pair} "
                                             f"of {PAIRS} ({age}: frame {pair} | frame {pair - 1})"))
    decoder = package(
        name="stream_s4_decoder", role="decoder",
        fused=WindowDecoder(fused).eval(), trainable=WindowDecoder(trainable).eval(),
        inputs=decoder_inputs, outputs=[(GRIDS, grids_text)],
        description=f"stream_s4 ball detector, decoder: one five-frame unit's heat maps and offsets from its four "
                    f"transitions' cached encoder features. Run once per frame. {HEAT_RING_TEXT}")
    unit = package(
        name="stream_s4_unit", role="unit", fused=fused, trainable=trainable,
        inputs=[TensorSpec(UNIT_INPUT, (1, FRAMES_PER_UNIT * RGB, height, width),
                           f"five consecutive same-phase frames, oldest first, {RGB} channels each, {frame_text(frame)}")],
        outputs=[(GRIDS, grids_text)],
        description=f"stream_s4 ball detector, the whole five-frame unit in one call (encoder on four transitions "
                    f"plus the decoder); for comparison with the encoder + decoder split. {HEAT_RING_TEXT}")
    return [encoder, decoder, unit]


def heatmap_specs(weights: Path, height: int = HEIGHT, width: int = WIDTH) -> List[PackageSpec]:
    """heatmap_s4's one package: the whole five-frame unit."""
    trainable = load_trainable(weights)
    frame = (1, RGB, height, width)
    return [make_package(
        name="heatmap_s4_unit", arch=heatmapnet.ARCH, role="unit", weights=weights,
        fused=fused_copy(trainable), trainable=trainable,
        inputs=[TensorSpec(UNIT_INPUT, (1, FRAMES_PER_UNIT * RGB, height, width),
                           f"five consecutive same-phase frames, oldest first, {RGB} channels each, {frame_text(frame)}")],
        outputs=[(GRIDS, grids_text)], threshold=operating_conf(weights), fold_note=ball_fold_note(heatmapnet.ARCH),
        description=f"heatmap_s4 ball detector (the heavy U-Net), the whole five-frame unit in one call; it runs once "
                    f"per frame. {HEAT_RING_TEXT}")]


class YoloRawHead(nn.Module):
    """An Ultralytics detection model's raw head, ``(1, 4 + classes, anchors)``: per anchor the box
    centre, width and height in input pixels, then each class's sigmoid score, the layout the
    player engine emits and ``engine_boxes`` decodes. `export` puts the head in Ultralytics' own
    export mode (Conv and BatchNorm fused, ``Detect.export``, as its exporter sets it for Core ML),
    the graph that is converted; without it the model answers as it trained, the reference."""

    def __init__(self, model: nn.Module, export: bool):
        super().__init__()
        from ultralytics.nn.modules import Detect
        model = copy.deepcopy(model).float().eval()
        if export:
            model = cast(nn.Module, cast(Any, model).fuse(verbose=False))
            for module in model.modules():
                if isinstance(module, Detect):
                    head: Any = module          # Ultralytics' attributes are untyped
                    head.export, head.format, head.dynamic = True, "coreml", False
                    head.xyxy, head.shape = False, None
        self.model = model.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        return cast(torch.Tensor, output[0] if isinstance(output, (tuple, list)) else output)


def load_yolo(weights: Path) -> nn.Module:
    """The FP32 eval network of an Ultralytics checkpoint (``ultralytics.nn.tasks.load_checkpoint``)."""
    from ultralytics.nn.tasks import load_checkpoint
    model, _ = load_checkpoint(str(weights), device="cpu")
    return cast(nn.Module, model).float().eval()


def player_specs(weights: Path, size: int = PLAYER_SIZE) -> List[PackageSpec]:
    """The player detector's one package at the engine's input `size` (square)."""
    model = load_yolo(weights)
    classes = len(cast(Any, model).names)
    strides = [int(stride) for stride in cast(Any, model).stride]
    cfg = Config()

    def predictions_text(shape: Tuple[int, ...]) -> str:
        grids = ", ".join(f"{size // stride}x{size // stride}" for stride in strides)
        return (f"per anchor ({shape[-1]} anchors: the {grids} grids at {size}): box centre x, centre y, width, "
                f"height in input pixels, then the score of each of the {shape[-2] - 4} classes (sigmoid); the app "
                f"thresholds and runs NMS")

    return [make_package(
        name="player_detector", arch="yolo11n_player", role="detector", weights=weights,
        fused=YoloRawHead(model, True), trainable=YoloRawHead(model, False),
        inputs=[TensorSpec(PLAYER_INPUT, (1, RGB, size, size),
                           f"the whole source frame stretched (not letterboxed) to {size}x{size}, bilinear, RGB in "
                           f"[0, 1] (uint8 / 255), as the tracker's TensorRT session prepares it")],
        outputs=[(PLAYER_OUTPUT, predictions_text)],
        description=f"tennis tracker player detector (YOLO11n, {classes} class: player), raw head without NMS. At "
                    f"60 FPS the tracker runs it on every other frame.",
        threshold=float(cfg.player_conf), fold_note="Conv and BatchNorm fused (Ultralytics fuse)",
        nms_iou=float(cfg.player_iou),
        notes={"nms": f"not in the model: keep anchors scoring >= {cfg.player_conf:.2f}, then greedy NMS at IoU "
                      f"{cfg.player_iou:.2f} (tennis_tracker/engines/engine_boxes.py)"})]


def player_input(frame_bgr: np.ndarray, size: int = PLAYER_SIZE) -> np.ndarray:
    """The player model's ``(1, 3, size, size)`` float16 input for a BGR uint8 frame: the ops of
    ``engine_session.TensorRTRuntimeSession.preprocess_frame`` (float32 / 255, BGR to RGB, bilinear
    resize without corner alignment, then FP16), on the CPU."""
    value = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    value = value.contiguous().flip(1) / 255.0
    if value.shape[-2:] != (size, size):
        value = nn.functional.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)
    return value.half().numpy()


def package_metadata(spec: PackageSpec, weights_md5: str, commit: str) -> Dict[str, Any]:
    """What a package carries: the description, author and version fields, a description per
    input and output, and the user-defined metadata (string values, as Core ML stores them)."""
    user: Dict[str, str] = {
        "arch": spec.arch, "role": spec.role, "source_weights": spec.weights.name,
        "source_weights_md5": weights_md5, "source_commit": commit,
        "operating_threshold": f"{spec.threshold:.2f}", "fused": spec.fold_note,
        "input_layout": "NCHW float16, batch 1, RGB in [0, 1]", "conv_gmac_per_call": f"{spec.gmac:.3f}",
    }
    if spec.nms_iou is not None:
        user["nms_iou"] = f"{spec.nms_iou:.2f}"
    user.update(spec.notes)
    return {
        "short_description": spec.description,
        "author": "tennis tracker, tools/export_coreml.py",
        "version": commit,
        "inputs": {tensor.name: tensor.description for tensor in spec.inputs},
        "outputs": {tensor.name: tensor.description for tensor in spec.outputs},
        "user_defined": user,
    }


def apply_metadata(mlmodel: Any, metadata: Mapping[str, Any]) -> None:
    mlmodel.short_description = metadata["short_description"]
    mlmodel.author = metadata["author"]
    mlmodel.version = metadata["version"]
    for name, text in metadata["inputs"].items():
        mlmodel.input_description[name] = text
    for name, text in metadata["outputs"].items():
        mlmodel.output_description[name] = text
    for key, value in metadata["user_defined"].items():
        mlmodel.user_defined_metadata[key] = str(value)


def trace(module: nn.Module, probes: Tuple[torch.Tensor, ...]) -> torch.jit.ScriptModule:
    """`module` traced on `probes`, the graph coremltools converts."""
    with warnings.catch_warnings(), torch.no_grad():
        # The shape checks in encode_frame and decode are constants at a fixed shape.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        # TorchScript is chosen on purpose: it is coremltools 9.0's tested PyTorch path. PyTorch
        # 2.10+ marks torch.jit.trace deprecated on every call; that notice is known.
        warnings.filterwarnings("ignore", message=r"`torch\.jit\.trace(_method)?` is deprecated",
                                category=DeprecationWarning)
        return cast(torch.jit.ScriptModule, torch.jit.trace(module, probes))


def convert(spec: PackageSpec, traced: torch.jit.ScriptModule, metadata: Mapping[str, Any]) -> Any:
    """The ML Program for `spec` (FP16 compute and tensors, fixed shapes, iOS 17). The player's
    boxes stay FP16 too: the program computes them in FP16, so an FP32 output would only widen
    values already rounded (and the shipped TensorRT engine's output is FP16 as well)."""
    import coremltools as ct
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=t.name, shape=t.shape, dtype=np.float16) for t in spec.inputs],
        outputs=[ct.TensorType(name=t.name, dtype=np.float16) for t in spec.outputs],
        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, DEPLOYMENT_TARGET), compute_units=ct.ComputeUnit.ALL,
        skip_model_load=sys.platform != "darwin")
    apply_metadata(mlmodel, metadata)
    return mlmodel


def describe_package(spec_proto: Any) -> Dict[str, List[Dict[str, Any]]]:
    """A saved package's inputs and outputs as its spec states them (name, shape, data type name)."""
    from coremltools.proto import FeatureTypes_pb2
    data_types = FeatureTypes_pb2.ArrayFeatureType.ArrayDataType

    def tensors(features: Any) -> List[Dict[str, Any]]:
        return [{"name": feature.name, "shape": list(feature.type.multiArrayType.shape),
                 "dtype": data_types.Name(feature.type.multiArrayType.dataType)} for feature in features]
    return {"inputs": tensors(spec_proto.description.input), "outputs": tensors(spec_proto.description.output)}


def video_stride(video: Path) -> int:
    """The source frames between consecutive frames of one phase (``require_phase_stride``)."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return require_phase_stride(fps)


def read_frames(video: Path, indices: Iterable[int]) -> Dict[int, np.ndarray]:
    """The BGR uint8 pictures OpenCV decodes at the source frame `indices`, read in order."""
    import cv2
    needed = set(indices)
    capture = cv2.VideoCapture(str(video))
    pictures: Dict[int, np.ndarray] = {}
    index = 0
    while needed - set(pictures):
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"{video} ends at frame {index}, before frames {sorted(needed - set(pictures))}")
        if index in needed:
            pictures[index] = frame
        index += 1
    capture.release()
    return pictures


def unit_frames(starts: Sequence[int], stride: int) -> List[List[int]]:
    """Per start, the source frames of its five-frame unit: one phase, `stride` apart."""
    return [[start + stride * k for k in range(FRAMES_PER_UNIT)] for start in starts]


def unit_input(pictures: Mapping[int, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    """A unit's ``(1, 15, HEIGHT, WIDTH)`` float16 input: each frame the tracker's ``frame_tensor``."""
    cpu = torch.device("cpu")
    return torch.cat([frame_tensor(pictures[i], cpu) for i in indices]).unsqueeze(0).numpy().astype(np.float16)


@dataclass(frozen=True)
class Samples:
    """What the cases are cut from: per unit its source frames and its float16 input, and the
    decoded BGR picture of every source frame they use."""
    units: Sequence[Tuple[List[int], np.ndarray]]
    pictures: Mapping[int, np.ndarray]


def read_samples(video: Path, starts: Sequence[int]) -> Samples:
    frame_lists = unit_frames(starts, video_stride(video))
    pictures = read_frames(video, [index for indices in frame_lists for index in indices])
    return Samples([(indices, unit_input(pictures, indices)) for indices in frame_lists], pictures)


def _load_validator() -> ModuleType:
    location = KIT / "validate_coreml.py"
    module_spec = importlib.util.spec_from_file_location("validate_coreml", location)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load {location}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules["validate_coreml"] = module
    module_spec.loader.exec_module(module)
    return module


validator = _load_validator()


class TorchRunner:
    """The packages' PyTorch graphs behind the Mac validator's runner interface (``predict(name,
    inputs)`` by tensor name), in one `variant`: ``fused_fp32`` (eager), ``traced_fp32`` (the
    TorchScript coremltools converts) or ``traced_fp16`` (the same in half precision, the nearest
    the PC comes to Core ML's FP16)."""

    VARIANTS = ("fused_fp32", "traced_fp32", "traced_fp16")

    def __init__(self, specs: Mapping[str, PackageSpec], traced: Mapping[str, torch.jit.ScriptModule], variant: str):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, not {variant!r}")
        self.specs, self.variant = specs, variant
        self.dtype = torch.float16 if variant.endswith("fp16") else torch.float32
        self.modules: Dict[str, Callable[..., Any]] = {}
        for name, spec in specs.items():
            if variant == "fused_fp32":
                self.modules[name] = spec.fused
            elif variant == "traced_fp32":
                self.modules[name] = traced[name]
            else:
                self.modules[name] = trace(copy.deepcopy(spec.fused).half(), spec.probes(torch.float16))

    def predict(self, name: str, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        spec = self.specs[name]
        values = tuple(torch.from_numpy(np.asarray(inputs[t.name])).to(self.dtype) for t in spec.inputs)
        with torch.no_grad():
            outputs = _as_tuple(self.modules[name](*values))
        return {t.name: out.float().numpy() for t, out in zip(spec.outputs, outputs)}


def reference_outputs(spec: PackageSpec, inputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """The trainable FP32 network's outputs on float16 `inputs` (taken as float32)."""
    with torch.no_grad():
        outputs = _as_tuple(spec.trainable(*(torch.from_numpy(value).float() for value in inputs)))
    return tuple(out.numpy() for out in outputs)


Case = Dict[str, Any]


def _unit_folder(out: Path, k: int, frames: np.ndarray) -> Tuple[Path, str]:
    """Unit `k`'s sample folder (made) and its saved frames (written once)."""
    folder = Path("samples") / f"unit{k}"
    (out / folder / "reference").mkdir(parents=True, exist_ok=True)
    frames_npy = (folder / "frames.npy").as_posix()
    if not (out / frames_npy).exists():
        np.save(out / frames_npy, frames)
    return folder, frames_npy


def heatmap_cases(out: Path, specs: Mapping[str, PackageSpec], samples: Samples) -> List[Case]:
    """heatmap_s4_unit on every unit, against the trainable network's grids."""
    spec = specs["heatmap_s4_unit"]
    cases: List[Case] = []
    for k, (_, frames) in enumerate(samples.units):
        folder, frames_npy = _unit_folder(out, k, frames)
        reference = (folder / "reference" / "heatmap_s4_grids.npy").as_posix()
        np.save(out / reference, reference_outputs(spec, [frames])[0])
        cases.append({"group": spec.name, "kind": "grids", "package": spec.name, "unit": k,
                      "inputs": {UNIT_INPUT: {"npy": frames_npy}}, "output": GRIDS, "reference": reference,
                      "threshold": spec.threshold})
    return cases


def stream_cases(out: Path, specs: Mapping[str, PackageSpec], samples: Samples) -> List[Case]:
    """stream_s4 on every unit: the whole unit and the decoder against the trainable unit's grids,
    the encoder per transition against the trainable encoder's features (saved as float16, the
    decoder's inputs), and the chain of the Core ML encoder into the Core ML decoder."""
    encoder, decoder, unit = (specs[name] for name in ("stream_s4_encoder", "stream_s4_decoder", "stream_s4_unit"))
    cases: List[Case] = []
    for k, (_, frames) in enumerate(samples.units):
        folder, frames_npy = _unit_folder(out, k, frames)
        (out / folder / "stream_s4_features").mkdir(parents=True, exist_ok=True)
        grids_ref = (folder / "reference" / "stream_s4_grids.npy").as_posix()
        np.save(out / grids_ref, reference_outputs(unit, [frames])[0])
        cases.append({"group": unit.name, "kind": "grids", "package": unit.name, "unit": k,
                      "inputs": {UNIT_INPUT: {"npy": frames_npy}}, "output": GRIDS, "reference": grids_ref,
                      "threshold": unit.threshold})
        decoder_inputs: Dict[str, Dict[str, str]] = {}
        steps = []
        for pair in range(1, PAIRS + 1):
            (f0, f1), (p0, p1) = transition_channels(pair)
            encoder_inputs = {"frame": {"npy": frames_npy, "channels": [f0, f1]},
                              "previous": {"npy": frames_npy, "channels": [p0, p1]}}
            features = reference_outputs(encoder, [frames[:, f0:f1], frames[:, p0:p1]])
            references = {}
            for level, value in zip(FEATURE_NAMES, features):
                path = (folder / "stream_s4_features" / f"{decoder_input(pair, level)}.npy").as_posix()
                np.save(out / path, value.astype(np.float16))
                references[level] = path
                decoder_inputs[decoder_input(pair, level)] = {"npy": path}
            cases.append({"group": encoder.name, "kind": "features", "package": encoder.name, "unit": k,
                          "transition": pair, "inputs": encoder_inputs, "references": references})
            steps.append({"inputs": encoder_inputs,
                          "outputs_to": {level: decoder_input(pair, level) for level in FEATURE_NAMES}})
        cases.append({"group": decoder.name, "kind": "grids", "package": decoder.name, "unit": k,
                      "inputs": decoder_inputs, "output": GRIDS, "reference": grids_ref, "threshold": unit.threshold})
        cases.append({"group": "stream_s4_chain", "kind": "stream_chain", "encoder": encoder.name,
                      "decoder": decoder.name, "unit": k, "encoder_steps": steps, "output": GRIDS,
                      "reference": grids_ref, "threshold": unit.threshold})
    return cases


def player_cases(out: Path, specs: Mapping[str, PackageSpec], samples: Samples) -> List[Case]:
    """player_detector on each unit's first frame (``player_input``), against the trainable
    network's raw head; both decoded as the tracker does."""
    spec = specs["player_detector"]
    if spec.nms_iou is None:
        raise ValueError(f"{spec.name} has no NMS IoU")
    folder = Path("samples") / "players"
    (out / folder).mkdir(parents=True, exist_ok=True)
    cases: List[Case] = []
    for indices, _ in samples.units:
        source_frame = indices[0]
        image = player_input(samples.pictures[source_frame], spec.inputs[0].shape[-1])
        image_npy = (folder / f"frame{source_frame}_image.npy").as_posix()
        reference = (folder / f"frame{source_frame}_predictions.npy").as_posix()
        np.save(out / image_npy, image)
        np.save(out / reference, reference_outputs(spec, [image])[0])
        cases.append({"group": spec.name, "kind": "boxes", "package": spec.name, "frame": source_frame,
                      "inputs": {PLAYER_INPUT: {"npy": image_npy}}, "output": PLAYER_OUTPUT,
                      "reference": reference, "threshold": spec.threshold, "iou": spec.nms_iou})
    return cases


@dataclass(frozen=True)
class Export:
    """One thing ``--models`` can name: its source weights (default path, CLI flag, the md5 it
    must have where one is recorded, and where its source is recorded), the packages it builds
    from them and the cases it writes for them."""
    name: str
    default_weights: Path
    weights_flag: str
    build: Callable[[Path], List[PackageSpec]]
    write_cases: Callable[[Path, Mapping[str, PackageSpec], Samples], List[Case]]
    source_note: str
    expected_md5: Optional[str] = None
    unmatched_flag: Optional[str] = None


EXPORTS: Dict[str, Export] = {entry.name: entry for entry in (
    Export("stream_s4", paths.STREAM_WEIGHTS, "--stream-weights", stream_specs, stream_cases,
           "the tracker's stream_s4 weights (tennis_tracker.paths.STREAM_WEIGHTS)"),
    Export("heatmap_s4", paths.HEATMAP_WEIGHTS, "--heat-weights", heatmap_specs, heatmap_cases,
           "the tracker's heatmap_s4 weights (tennis_tracker.paths.HEATMAP_WEIGHTS)"),
    Export("player", DEFAULT_PLAYER_WEIGHTS, "--player-weights", player_specs, player_cases,
           f"{RECORD} records where the player checkpoint is and its md5; copy it to "
           f"{DEFAULT_PLAYER_WEIGHTS.relative_to(ROOT).as_posix()}",
           expected_md5=PLAYER_SOURCE_MD5, unmatched_flag="--allow-unmatched-player"),
)}


@dataclass(frozen=True)
class Source:
    """An export's source weights, their md5, and whether that is the recorded one (None where
    none is recorded)."""
    path: Path
    md5: str
    matched: Optional[bool] = None


class SourceError(ValueError):
    """A source checkpoint that is missing, or not the recorded one; the message is one line."""


def resolve_source(entry: Export, path: Path, allow_unmatched: bool = False) -> Source:
    """`entry`'s weights at `path`, hashed once; a SourceError when the file is missing or (unless
    `allow_unmatched`) its md5 is not the recorded one."""
    if not path.is_file():
        others = " ".join(name for name in EXPORTS if name != entry.name)
        raise SourceError(f"{entry.name}: no checkpoint at {_display_path(path)}. {entry.source_note}, or leave it "
                          f"out with --models {others}")
    md5 = file_md5(path)
    matched = None if entry.expected_md5 is None else md5 == entry.expected_md5
    if matched is False and not allow_unmatched:
        raise SourceError(f"{entry.name}: {_display_path(path)} has md5 {md5}, not {entry.expected_md5}, the "
                          f"checkpoint the shipped engine was matched to ({RECORD}); pass {entry.unmatched_flag} "
                          f"to export it anyway")
    return Source(path.resolve(), md5, matched)


def write_samples(out: Path, specs: Mapping[str, PackageSpec], samples: Samples,
                  exports: Iterable[str]) -> List[Case]:
    """The sample inputs and PyTorch references under ``out/samples``, and the manifest's cases,
    for each of `exports` (``EXPORTS`` names) in turn."""
    return [case for name in exports for case in EXPORTS[name].write_cases(out, specs, samples)]


def pc_validation(specs: Mapping[str, PackageSpec], traced: Mapping[str, torch.jit.ScriptModule],
                  manifest: Mapping[str, Any], out: Path) -> Dict[str, Dict[str, Any]]:
    """Every case scored on the PC's PyTorch graphs, per variant and group."""
    results: Dict[str, Dict[str, Any]] = {}
    for variant in TorchRunner.VARIANTS:
        runner = TorchRunner(specs, traced, variant)
        results[variant] = validator.evaluate(runner, manifest, out, validator.Tolerances())
    return results


def pc_validation_passed(manifest: Mapping[str, Any]) -> bool:
    return all(result["pass"] for groups in manifest["pc_validation"].values() for result in groups.values())


def git_commit() -> str:
    """This checkout's short commit, by git or (in WSL, where the worktree's paths are Windows')
    git.exe, with "-dirty" when the tree has changes; "unknown" when neither answers."""
    for git in ("git", "git.exe"):
        try:
            done = subprocess.run([git, "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                  timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and done.stdout.strip():
            dirty = subprocess.run([git, "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                   text=True, timeout=60).stdout.strip()
            return done.stdout.strip() + ("-dirty" if dirty else "")
    return "unknown"


def check_kit() -> None:
    missing = [name for name in KIT_FILES if not (KIT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"the Mac kit in {KIT} lacks {missing}")


def prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def export(out: Path, sources: Mapping[str, Source], video: Path, starts: Sequence[int],
           convert_packages: bool = True) -> Dict[str, Any]:
    """Everything the Mac needs for the `sources` (``EXPORTS`` name to its resolved weights),
    written to `out` (which must be empty); returns the manifest."""
    check_kit()
    commit = git_commit()
    specs: Dict[str, PackageSpec] = {}
    source_of: Dict[str, Source] = {}
    for name, source in sources.items():
        for spec in EXPORTS[name].build(source.path):
            specs[spec.name], source_of[spec.name] = spec, source
    traced = {name: trace(spec.fused, spec.probes()) for name, spec in specs.items()}
    packages: Dict[str, Any] = {}
    for name, spec in specs.items():
        source = source_of[name]
        entry: Dict[str, Any] = {
            "file": f"models/{name}.mlpackage", "arch": spec.arch, "role": spec.role, "description": spec.description,
            "inputs": [{"name": t.name, "shape": list(t.shape), "dtype": "float16"} for t in spec.inputs],
            "outputs": [{"name": t.name, "shape": list(t.shape), "dtype": "float16"} for t in spec.outputs],
            "conv_gmac_per_call": spec.gmac, "operating_threshold": spec.threshold,
            "source_weights": _display_path(spec.weights), "source_weights_md5": source.md5}
        if spec.nms_iou is not None:
            entry["nms_iou"] = spec.nms_iou
        if source.matched is not None:
            entry["source_is_the_matched_checkpoint"] = source.matched
        if convert_packages:
            print(f"[coreml] converting {name}")
            mlmodel = convert(spec, traced[name], package_metadata(spec, source.md5, commit))
            mlmodel.save(str(out / entry["file"]))
            entry["saved_spec"] = describe_package(mlmodel.get_spec())
        packages[name] = entry
    print(f"[coreml] sampling units at {list(starts)} of {video.name}")
    samples = read_samples(video, starts)
    cases = write_samples(out, specs, samples, sources)
    manifest: Dict[str, Any] = {
        "exported": datetime.datetime.now().isoformat(timespec="seconds"), "source_commit": commit,
        "coremltools": _version("coremltools"), "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "deployment_target": DEPLOYMENT_TARGET, "compute_precision": "FLOAT16", "models": list(sources),
        "video": video.name,
        "units": [{"unit": k, "source_frames": indices} for k, (indices, _) in enumerate(samples.units)],
        "packages": packages, "cases": cases,
    }
    print("[coreml] scoring the PyTorch graphs on the PC")
    manifest["pc_validation"] = pc_validation(specs, traced, manifest, out)
    for name in KIT_FILES:
        shutil.copy2(KIT / name, out / name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _display_path(path: Path) -> str:
    """`path` relative to the checkout when it is inside it, POSIX-style."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:          # the samples-only export
        return "not installed"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", choices=list(EXPORTS), default=list(EXPORTS),
                        help="what to export (default: all)")
    for entry in EXPORTS.values():
        parser.add_argument(entry.weights_flag, dest=f"{entry.name}_weights", type=Path, default=entry.default_weights,
                            help=f"{entry.name}'s source weights (default {_display_path(entry.default_weights)})")
    for entry in EXPORTS.values():
        if entry.unmatched_flag:
            parser.add_argument(entry.unmatched_flag, dest=f"{entry.name}_unmatched", action="store_true",
                                help=f"export {entry.name} from a checkpoint whose md5 is not the recorded "
                                     f"{entry.expected_md5} (the manifest still records its md5)")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--starts", type=int, nargs="+", default=list(DEFAULT_STARTS),
                        help="source frames the sample units start at")
    parser.add_argument("--overwrite", action="store_true", help="replace a non-empty --out")
    parser.add_argument("--no-convert", action="store_true",
                        help="write the samples, references and PC checks only (no coremltools needed)")
    args = parser.parse_args(argv)
    try:
        sources = {name: resolve_source(EXPORTS[name], getattr(args, f"{name}_weights"),
                                        bool(getattr(args, f"{name}_unmatched", False)))
                   for name in dict.fromkeys(args.models)}
    except SourceError as error:
        print(f"[coreml] {error}", file=sys.stderr)
        return 2
    check_kit()
    prepare_out(args.out, args.overwrite)
    manifest = export(args.out, sources, args.video, args.starts, convert_packages=not args.no_convert)
    for variant, groups in manifest["pc_validation"].items():
        for group, result in groups.items():
            print(f"[pc] {variant:<12} {'PASS' if result['pass'] else 'FAIL'} {group:<18} {json.dumps(result['summary'])}")
    print(f"[coreml] wrote {args.out}")
    if not pc_validation_passed(manifest):
        print("[coreml] some PC checks FAILED (see the FAIL lines)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
