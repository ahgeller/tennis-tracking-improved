"""Export the ball and player detectors to Core ML, ready to time on an iPhone in Xcode's performance report.

    python tools/export_coreml.py                    # writes output/coreml_speed_test/

coremltools converts on macOS and Linux only (no Windows wheels), so on this PC it runs in WSL:

    wsl -d Ubuntu --cd <worktree> -- ~/miniforge3/envs/coreml/bin/python tools/export_coreml.py

The packages (ML Program, FP16 compute, FP16 inputs and outputs, fixed shapes at batch 1,
deployment target iOS 17), each from the deployed network (``ball_models.deployed``: RepVGG
branches and BatchNorm folded from the float32 record):

- ``stream_s4_encoder``: ``StreamNet.encode_frame``, one transition a call: ``frame`` and
  ``previous`` (the same phase's frame before it; at 60 FPS the source frame two back), each
  ``(1, 3, 432, 768)`` RGB in [0, 1], to its features ``quarter``, ``eighth`` and ``bottom``
  (strides 4, 8 and 16), the names ``ball_trt`` gives the TensorRT halves.
- ``stream_s4_decoder``: ``StreamNet.decode`` of one unit from its four transitions' cached
  features, passed separately (``t1_quarter`` .. ``t4_bottom``, oldest first), so an app hands it
  the encoder's last four outputs without copying them together; out ``grids``.
- ``stream_s4_unit`` and ``heatmap_s4_unit``: the whole five-frame unit, ``frames``
  ``(1, 15, 432, 768)`` to ``grids`` ``(1, 15, 108, 192)`` (per frame heat, x offset, y offset).

- ``player_detector``: the player YOLO11n (``system/models/sources/player.pt``, gitignored, the
  checkpoint the shipped ``player.engine`` was matched to), ``image`` ``(1, 3, 640, 640)``, the
  frame stretched as the tracker's TensorRT session does, to the raw head ``predictions``
  ``(1, 5, 8400)``. NMS stays in the app: Ultralytics' Core ML NMS pipeline needs an image
  input and bakes its own thresholds, and the tracker's (score 0.21, IoU 0.10) run on the few
  anchors above the threshold (``engine_boxes``). The court model is not exported: no source
  weights matching the shipped ``courtdetection.engine`` exist (``audit/records/opt_coreml.md``).

The tracker at window step 1 runs one encoder call and one decoder call per source frame, so
stream_s4's per-frame cost is the encoder's time plus the decoder's; at 60 FPS it runs the player
detector on every other frame.

Beside them: sample units of ``system/sample/pomona.mp4`` (five consecutive same-phase frames,
the tracker's own resize, float16 as the packages take them), the player detector's input for
each unit's first frame, the PyTorch FP32 outputs of the source networks on them, the encoder
features the decoder is checked on, ``manifest.json``
(packages, cases, the PC checks) and the Mac kit from ``tools/coreml_speed_test/``
(``MAC_SPEED_TEST.md``, ``validate.sh``, ``validate_coreml.py``, ``requirements-mac.txt``).

Core ML predicts only on macOS, so on the PC the export checks the graphs it converts: the fused
network eagerly and as the traced TorchScript coremltools reads, in FP32 and FP16, against the
trainable FP32 network, scored by the Mac validator's own functions (``manifest["pc_validation"]``).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

from tennis_tracker import ball_models, heatmapnet, paths, streamnet  # noqa: E402
from tennis_tracker.ball_weights import operating_conf  # noqa: E402
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

# The tensors' names: the TensorRT export's (``ball_trt``), and the decoder's per-transition inputs.
FRAME_INPUTS = ("frame", "previous")
FEATURE_NAMES = ("quarter", "eighth", "bottom")
UNIT_INPUT = "frames"
GRIDS = "grids"
PAIRS = streamnet.PAIRS_PER_UNIT
FEATURE_STRIDES = (4, 8, 16)

# The player detector: the checkpoint the shipped models/player.engine was matched to (md5, and
# boxes on pomona within FP16 noise; audit/records/opt_coreml.md), at the engine's input size.
DEFAULT_PLAYER_WEIGHTS = paths.MODELS / "sources" / "player.pt"
PLAYER_SOURCE_MD5 = "4de317cc3e2cad7e9196fa438b439544"
PLAYER_SIZE = 640
PLAYER_INPUT = "image"
PLAYER_OUTPUT = "predictions"


def decoder_input(pair: int, level: str) -> str:
    """The decoder's input for transition `pair` (1 the oldest, ``PAIRS`` the newest) at `level`."""
    return f"t{pair}_{level}"


DECODER_INPUTS = tuple(decoder_input(pair, level) for pair in range(1, PAIRS + 1) for level in FEATURE_NAMES)

_GRIDS_TEXT = ("per frame of the unit, oldest first, three 108x192 planes at stride 4: the ball heat map "
               "(sigmoid), then the x and y offsets within the cell (sigmoid, 0..1); a peak at row r, column c "
               "is the ball at ((c + x) * W / 192, (r + y) * H / 108) in a W x H source frame")
_FRAME_TEXT = "RGB in [0, 1] (uint8 / 255), 768x432, OpenCV INTER_LINEAR resize of the source frame"


@dataclass(frozen=True)
class TensorSpec:
    """One input or output of a package: its name, its fixed shape and what it holds."""
    name: str
    shape: Tuple[int, ...]
    description: str


@dataclass
class PackageSpec:
    """One package to export: the fused module it is converted from, the trainable module its
    references come from (same call signature), its tensors and its words."""
    name: str
    arch: str
    role: str
    weights: Path
    fused: nn.Module
    trainable: nn.Module
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    description: str
    threshold: float = 0.0
    fold_note: str = ""
    notes: Dict[str, str] = field(default_factory=dict)
    gmac: float = 0.0

    def probes(self, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.zeros(spec.shape, dtype=dtype) for spec in self.inputs)


class WindowDecoder(nn.Module):
    """``StreamNet.decode`` with the four transitions' features as separate inputs, oldest first,
    each transition's quarter, eighth and bottom in turn (``DECODER_INPUTS``)."""

    def __init__(self, model: streamnet.StreamNet):
        super().__init__()
        self.model = model

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != PAIRS * len(FEATURE_NAMES):
            raise ValueError(f"the decoder takes {PAIRS * len(FEATURE_NAMES)} feature maps, not {len(features)}")
        pairs = [cast(streamnet.PairFeatures, tuple(features[3 * i:3 * i + 3])) for i in range(PAIRS)]
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


def conv_gmac(model: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> float:
    """Multiply-accumulates of every convolution in one call, in billions (``bench_ball_archs``'s count)."""
    total = [0]

    def hook(module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total[0] += output.numel() * (module.in_channels // module.groups) * kernel

    handles = [module.register_forward_hook(hook) for module in model.modules()]
    with torch.no_grad():
        model(*inputs)
    for handle in handles:
        handle.remove()
    return total[0] / 1e9


def _as_tuple(value: Any) -> Tuple[torch.Tensor, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _finish(spec: PackageSpec, outputs: Sequence[Tuple[str, str]]) -> PackageSpec:
    """`spec` with its `outputs` (name, description) shaped and its GMAC counted on zeros."""
    probes = spec.probes()
    with torch.no_grad():
        values = _as_tuple(spec.fused(*probes))
    if len(values) != len(outputs):
        raise ValueError(f"{spec.name} returns {len(values)} tensors, not {len(outputs)}")
    spec.outputs = [TensorSpec(name, tuple(value.shape), text) for (name, text), value in zip(outputs, values)]
    spec.gmac = round(conv_gmac(spec.fused, probes), 3)
    return spec


def stream_specs(weights: Path, height: int = HEIGHT, width: int = WIDTH) -> List[PackageSpec]:
    """stream_s4's three packages: the encoder, the decoder and the whole unit."""
    trainable = load_trainable(weights)
    if not isinstance(trainable, streamnet.StreamNet):
        raise ValueError(f"{weights} is not a {streamnet.ARCH} record")
    fused = cast(streamnet.StreamNet, fused_copy(trainable))
    frame = (1, 3, height, width)
    channels = dict(zip(FEATURE_NAMES, streamnet.ENCODER_CHANNELS[1:]))
    feature_text = {level: f"stride {stride}, {channels[level]} channels" for level, stride in zip(FEATURE_NAMES, FEATURE_STRIDES)}
    encoder = PackageSpec(
        "stream_s4_encoder", streamnet.ARCH, "encoder", weights,
        streamnet.StreamEncoder(fused).eval(), streamnet.StreamEncoder(trainable).eval(),
        [TensorSpec("frame", frame, f"the current frame, {_FRAME_TEXT}"),
         TensorSpec("previous", frame, f"the same phase's frame before it (at 60 FPS the source frame two back), {_FRAME_TEXT}")],
        [], "stream_s4 ball detector, encoder: one frame transition (frame, previous) to its features. Run once per "
            "frame; the decoder takes the last four transitions' outputs.")
    _finish(encoder, [(level, f"the transition's features at {feature_text[level]}") for level in FEATURE_NAMES])
    decoder_inputs = []
    for pair in range(1, PAIRS + 1):
        age = "oldest" if pair == 1 else "newest" if pair == PAIRS else f"number {pair}"
        for level, out in zip(FEATURE_NAMES, encoder.outputs):
            decoder_inputs.append(TensorSpec(decoder_input(pair, level), out.shape,
                                             f"the encoder's '{level}' output for the unit's transition {pair} "
                                             f"of {PAIRS} ({age}: frame {pair} | frame {pair - 1})"))
    decoder = PackageSpec(
        "stream_s4_decoder", streamnet.ARCH, "decoder", weights, WindowDecoder(fused).eval(), WindowDecoder(trainable).eval(),
        decoder_inputs, [], "stream_s4 ball detector, decoder: one five-frame unit's heat maps and offsets from its "
                            "four transitions' cached encoder features. Run once per frame.")
    _finish(decoder, [(GRIDS, _GRIDS_TEXT)])
    unit = PackageSpec(
        "stream_s4_unit", streamnet.ARCH, "unit", weights, fused, trainable,
        [TensorSpec(UNIT_INPUT, (1, FRAMES_PER_UNIT * 3, height, width),
                    f"five consecutive same-phase frames, oldest first, 3 channels each, {_FRAME_TEXT}")],
        [], "stream_s4 ball detector, the whole five-frame unit in one call (encoder on four transitions plus the "
            "decoder); for comparison with the encoder + decoder split.")
    _finish(unit, [(GRIDS, _GRIDS_TEXT)])
    return [_ball_words(spec) for spec in (encoder, decoder, unit)]


def heatmap_specs(weights: Path, height: int = HEIGHT, width: int = WIDTH) -> List[PackageSpec]:
    """heatmap_s4's one package: the whole five-frame unit."""
    trainable = load_trainable(weights)
    unit = PackageSpec(
        "heatmap_s4_unit", heatmapnet.ARCH, "unit", weights, fused_copy(trainable), trainable,
        [TensorSpec(UNIT_INPUT, (1, FRAMES_PER_UNIT * 3, height, width),
                    f"five consecutive same-phase frames, oldest first, 3 channels each, {_FRAME_TEXT}")],
        [], "heatmap_s4 ball detector (the heavy U-Net), the whole five-frame unit in one call; it runs once per frame.")
    return [_ball_words(_finish(unit, [(GRIDS, _GRIDS_TEXT)]))]


def _ball_words(spec: PackageSpec) -> PackageSpec:
    """A ball package's threshold (``operating_conf``) and fold note."""
    spec.threshold = operating_conf(spec.weights)
    spec.fold_note = ball_fold_note(spec.arch)
    return spec


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


def player_specs(weights: Path, size: int = PLAYER_SIZE, expected_md5: Optional[str] = PLAYER_SOURCE_MD5
                 ) -> List[PackageSpec]:
    """The player detector's one package at the engine's `size`; weights whose md5 is not
    `expected_md5` (the checkpoint the shipped ``player.engine`` matches) are a ValueError."""
    if expected_md5 is not None and file_md5(weights) != expected_md5:
        raise ValueError(f"{weights} (md5 {file_md5(weights)}) is not the checkpoint the shipped player.engine "
                         f"was matched to (md5 {expected_md5}); pass --player-md5 to export it anyway")
    model = load_yolo(weights)
    classes = len(cast(Any, model).names)
    from tennis_tracker.config import Config
    cfg = Config()
    spec = PackageSpec(
        "player_detector", "yolo11n_player", "detector", weights, YoloRawHead(model, True), YoloRawHead(model, False),
        [TensorSpec(PLAYER_INPUT, (1, 3, size, size),
                    f"the whole source frame stretched (not letterboxed) to {size}x{size}, bilinear, RGB in [0, 1] "
                    f"(uint8 / 255), as the tracker's TensorRT session prepares it")],
        [], f"tennis tracker player detector (YOLO11n, {classes} class: player), raw head without NMS. At 60 FPS the "
            f"tracker runs it on every other frame.",
        threshold=float(cfg.player_conf), fold_note="Conv and BatchNorm fused (Ultralytics fuse)",
        notes={"nms": f"not in the model: keep anchors scoring >= {cfg.player_conf:.2f}, then greedy NMS at IoU "
                      f"{cfg.player_iou:.2f} (tennis_tracker/engines/engine_boxes.py)",
               "nms_iou": f"{cfg.player_iou:.2f}"})
    anchors = "per anchor (80x80, 40x40 and 20x20 grids at 640)" if size == PLAYER_SIZE else "per anchor"
    return [_finish(spec, [(PLAYER_OUTPUT, f"{anchors}: box centre x, centre y, width, height in input pixels, "
                                           f"then the player score (sigmoid); the app thresholds and runs NMS")])]


def player_input(frame_bgr: np.ndarray, size: int = PLAYER_SIZE) -> np.ndarray:
    """The player model's ``(1, 3, size, size)`` float16 input for a BGR uint8 frame: the ops of
    ``engine_session.TensorRTRuntimeSession.preprocess_frame`` (float32 / 255, BGR to RGB, bilinear
    resize without corner alignment, then FP16), on the CPU."""
    value = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    value = value.flip(1) / 255.0
    if value.shape[-2:] != (size, size):
        value = nn.functional.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)
    return value.half().numpy()


def package_metadata(spec: PackageSpec, weights_md5: str, commit: str) -> Dict[str, Any]:
    """What a package carries: the description, author and version fields, a description per
    input and output, and the user-defined metadata (string values, as Core ML stores them)."""
    return {
        "short_description": spec.description,
        "author": "tennis tracker, tools/export_coreml.py",
        "version": commit,
        "inputs": {tensor.name: tensor.description for tensor in spec.inputs},
        "outputs": {tensor.name: tensor.description for tensor in spec.outputs},
        "user_defined": {
            "arch": spec.arch, "role": spec.role, "source_weights": spec.weights.name,
            "source_weights_md5": weights_md5, "source_commit": commit,
            "operating_threshold": f"{spec.threshold:.2f}", "fused": spec.fold_note,
            "input_layout": "NCHW float16, batch 1, RGB in [0, 1]",
            "conv_gmac_per_call": f"{spec.gmac:.3f}", **spec.notes,
        },
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
    """The ML Program for `spec` (FP16 compute and tensors, fixed shapes, iOS 17)."""
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
    """A saved package's inputs and outputs as its spec states them (name, shape, data type)."""
    def tensors(features: Any) -> List[Dict[str, Any]]:
        return [{"name": feature.name, "shape": list(feature.type.multiArrayType.shape),
                 "dtype": int(feature.type.multiArrayType.dataType)} for feature in features]
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


def write_samples(out: Path, specs: Mapping[str, PackageSpec], units: Sequence[Tuple[List[int], np.ndarray]],
                  players: Sequence[Tuple[int, np.ndarray]] = ()) -> List[Dict[str, Any]]:
    """The sample inputs and PyTorch references under ``out/samples``, and the manifest's cases:
    the ball packages' on each unit, the player detector's on each of `players` (source frame,
    input) when `specs` holds it."""
    cases: List[Dict[str, Any]] = []
    stream, heatmap = "stream_s4_unit" in specs, "heatmap_s4_unit" in specs
    stream_conf = specs["stream_s4_unit"].threshold if stream else 0.0
    for k, (indices, frames) in enumerate(units):
        folder = Path("samples") / f"unit{k}"
        (out / folder / "reference").mkdir(parents=True, exist_ok=True)
        frames_npy = (folder / "frames.npy").as_posix()
        np.save(out / frames_npy, frames)
        unit_input = {UNIT_INPUT: {"npy": frames_npy}}
        if heatmap:
            heat_ref = (folder / "reference" / "heatmap_s4_grids.npy").as_posix()
            np.save(out / heat_ref, reference_outputs(specs["heatmap_s4_unit"], [frames])[0])
            cases.append({"group": "heatmap_s4_unit", "kind": "grids", "package": "heatmap_s4_unit", "unit": k,
                          "inputs": unit_input, "output": GRIDS, "reference": heat_ref,
                          "threshold": specs["heatmap_s4_unit"].threshold})
        if not stream:
            continue
        (out / folder / "stream_s4_features").mkdir(parents=True, exist_ok=True)
        grids_ref = (folder / "reference" / "stream_s4_grids.npy").as_posix()
        np.save(out / grids_ref, reference_outputs(specs["stream_s4_unit"], [frames])[0])
        cases.append({"group": "stream_s4_unit", "kind": "grids", "package": "stream_s4_unit", "unit": k,
                      "inputs": unit_input, "output": GRIDS, "reference": grids_ref, "threshold": stream_conf})
        decoder_inputs: Dict[str, Dict[str, str]] = {}
        steps = []
        for pair in range(1, PAIRS + 1):
            frame_in = {"npy": frames_npy, "channels": [3 * pair, 3 * pair + 3]}
            previous_in = {"npy": frames_npy, "channels": [3 * pair - 3, 3 * pair]}
            features = reference_outputs(specs["stream_s4_encoder"], [frames[:, 3 * pair:3 * pair + 3],
                                                                      frames[:, 3 * pair - 3:3 * pair]])
            references = {}
            for level, value in zip(FEATURE_NAMES, features):
                path = (folder / "stream_s4_features" / f"{decoder_input(pair, level)}.npy").as_posix()
                np.save(out / path, value.astype(np.float16))
                references[level] = path
                decoder_inputs[decoder_input(pair, level)] = {"npy": path}
            encoder_inputs = {"frame": frame_in, "previous": previous_in}
            cases.append({"group": "stream_s4_encoder", "kind": "features", "package": "stream_s4_encoder", "unit": k,
                          "transition": pair, "inputs": encoder_inputs, "references": references})
            steps.append({"inputs": encoder_inputs,
                          "outputs_to": {level: decoder_input(pair, level) for level in FEATURE_NAMES}})
        cases.append({"group": "stream_s4_decoder", "kind": "grids", "package": "stream_s4_decoder", "unit": k,
                      "inputs": decoder_inputs, "output": GRIDS, "reference": grids_ref, "threshold": stream_conf})
        cases.append({"group": "stream_s4_chain", "kind": "stream_chain", "encoder": "stream_s4_encoder",
                      "decoder": "stream_s4_decoder", "unit": k, "encoder_steps": steps, "output": GRIDS,
                      "reference": grids_ref, "threshold": stream_conf})
    if "player_detector" in specs:
        player = specs["player_detector"]
        folder = Path("samples") / "players"
        folder_path = out / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for source_frame, image in players:
            image_npy = (folder / f"frame{source_frame}_image.npy").as_posix()
            reference = (folder / f"frame{source_frame}_predictions.npy").as_posix()
            np.save(out / image_npy, image)
            np.save(out / reference, reference_outputs(player, [image])[0])
            cases.append({"group": "player_detector", "kind": "boxes", "package": "player_detector",
                          "frame": source_frame, "inputs": {PLAYER_INPUT: {"npy": image_npy}}, "output": PLAYER_OUTPUT,
                          "reference": reference, "threshold": player.threshold,
                          "iou": float(player.notes["nms_iou"])})
    return cases


def pc_validation(specs: Mapping[str, PackageSpec], traced: Mapping[str, torch.jit.ScriptModule],
                  manifest: Mapping[str, Any], out: Path) -> Dict[str, Dict[str, Any]]:
    """Every case scored on the PC's PyTorch graphs, per variant and group."""
    results: Dict[str, Dict[str, Any]] = {}
    for variant in TorchRunner.VARIANTS:
        runner = TorchRunner(specs, traced, variant)
        results[variant] = validator.evaluate(runner, manifest, out, validator.DEFAULT_HEAT_TOLERANCE)
    return results


def git_commit() -> str:
    """This checkout's short commit, by git or (in WSL, where the worktree's paths are Windows')
    git.exe; "unknown" when neither answers."""
    for git in ("git", "git.exe"):
        try:
            done = subprocess.run([git, "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                  timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and done.stdout.strip():
            dirty = subprocess.run([git, "status", "--porcelain", "--", "tools"], cwd=ROOT, capture_output=True,
                                   text=True, timeout=30).stdout.strip()
            return done.stdout.strip() + ("-dirty" if dirty else "")
    return "unknown"


def prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def export(out: Path, stream_weights: Optional[Path], heat_weights: Optional[Path], video: Path,
           starts: Sequence[int], convert_packages: bool = True, player_weights: Optional[Path] = None,
           player_md5: Optional[str] = PLAYER_SOURCE_MD5) -> Dict[str, Any]:
    """Everything the Mac needs, written to `out` (which must be empty); returns the manifest."""
    missing = [name for name in KIT_FILES if not (KIT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"the Mac kit in {KIT} lacks {missing}")
    commit = git_commit()
    specs: Dict[str, PackageSpec] = {}
    if stream_weights is not None:
        specs.update({spec.name: spec for spec in stream_specs(stream_weights)})
    if heat_weights is not None:
        specs.update({spec.name: spec for spec in heatmap_specs(heat_weights)})
    if player_weights is not None:
        specs.update({spec.name: spec for spec in player_specs(player_weights, expected_md5=player_md5)})
    md5 = {path: file_md5(path) for path in {spec.weights for spec in specs.values()}}
    traced = {name: trace(spec.fused, spec.probes()) for name, spec in specs.items()}
    packages: Dict[str, Any] = {}
    for name, spec in specs.items():
        metadata = package_metadata(spec, md5[spec.weights], commit)
        entry: Dict[str, Any] = {
            "file": f"models/{name}.mlpackage", "arch": spec.arch, "role": spec.role, "description": spec.description,
            "inputs": [{"name": t.name, "shape": list(t.shape), "dtype": "float16"} for t in spec.inputs],
            "outputs": [{"name": t.name, "shape": list(t.shape), "dtype": "float16"} for t in spec.outputs],
            "conv_gmac_per_call": spec.gmac, "source_weights": _display_path(spec.weights),
            "source_weights_md5": md5[spec.weights]}
        if convert_packages:
            print(f"[coreml] converting {name}")
            mlmodel = convert(spec, traced[name], metadata)
            mlmodel.save(str(out / entry["file"]))
            entry["saved_spec"] = describe_package(mlmodel.get_spec())
        packages[name] = entry
    print(f"[coreml] sampling units at {list(starts)} of {video.name}")
    frame_lists = unit_frames(starts, video_stride(video))
    pictures = read_frames(video, [index for indices in frame_lists for index in indices])
    units = [(indices, unit_input(pictures, indices)) for indices in frame_lists]
    players = [(start, player_input(pictures[start])) for start in starts] if player_weights is not None else []
    cases = write_samples(out, specs, units, players)
    manifest: Dict[str, Any] = {
        "exported": datetime.datetime.now().isoformat(timespec="seconds"), "source_commit": commit,
        "coremltools": _version("coremltools"), "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "deployment_target": DEPLOYMENT_TARGET, "compute_precision": "FLOAT16",
        "video": video.name, "units": [{"unit": k, "source_frames": indices} for k, (indices, _) in enumerate(units)],
        "player_frames": [start for start, _ in players],
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
        from importlib.metadata import version
        return version(package)
    except Exception:          # not installed: the samples-only export
        return "not installed"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stream-weights", type=Path, default=paths.STREAM_WEIGHTS)
    parser.add_argument("--heat-weights", type=Path, default=paths.HEATMAP_WEIGHTS)
    parser.add_argument("--player-weights", type=Path, default=DEFAULT_PLAYER_WEIGHTS,
                        help=f"the player detector's Ultralytics checkpoint (default {_display_path(DEFAULT_PLAYER_WEIGHTS)})")
    parser.add_argument("--player-md5", default=PLAYER_SOURCE_MD5,
                        help="the md5 the player checkpoint must have (the one matched to the shipped player.engine)")
    parser.add_argument("--no-player", action="store_true", help="leave the player detector out")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--starts", type=int, nargs="+", default=list(DEFAULT_STARTS),
                        help="source frames the sample units start at")
    parser.add_argument("--overwrite", action="store_true", help="replace a non-empty --out")
    parser.add_argument("--no-convert", action="store_true",
                        help="write the samples, references and PC checks only (no coremltools needed)")
    args = parser.parse_args(argv)
    prepare_out(args.out, args.overwrite)
    player = None if args.no_player else args.player_weights.resolve()
    manifest = export(args.out, args.stream_weights.resolve(), args.heat_weights.resolve(), args.video,
                      args.starts, convert_packages=not args.no_convert, player_weights=player,
                      player_md5=args.player_md5)
    for variant, groups in manifest["pc_validation"].items():
        for group, result in groups.items():
            print(f"[pc] {variant:<12} {'PASS' if result['pass'] else 'FAIL'} {group:<18} {json.dumps(result['summary'])}")
    print(f"[coreml] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
