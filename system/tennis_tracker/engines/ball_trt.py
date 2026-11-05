"""The five-frame ball model as a TensorRT FP16 engine, built from whatever
``ball_models.load_ball_model`` returns and cached on disk.

The model is exported to ONNX (opset 17, the TorchScript exporter) at a static batch of units with
FP16 input and output, the dtypes the PyTorch prepass feeds and reads, and built with TensorRT's
FP16 flag. On the RTX 5080 laptop HeatmapNet ran at 1.2 to 1.6 ms a unit this way against 3.8 to 4.1 ms
FP16 eager, thermally throttled (``audit/records/opt_ball_prepass.md`` has the batch sweep that chose the default batch).

An engine is only valid for the weights, the TensorRT version and the GPU that built it, so the
cache file's name carries a key over exactly those (the weights file's SHA-256, the architecture,
the batch and input size, the TensorRT and PyTorch versions, the GPU's name and compute capability
and ``ENGINE_RECIPE``, raised whenever the export changes): a stale engine is never found, and a
fresh one is built in its place on first use. The cache keeps old engines; delete the directory to
reclaim the space. The engine runs through ``engine_session.TensorRTRuntimeSession`` on its stream.

With `visibility` (``Config.ball_visibility_gate`` on) the export also carries the stream_s4
visibility head as a second output, ``VISIBILITY_OUTPUT_NAME`` (the session returns outputs
sorted by name, so it follows ``OUTPUT_NAME``), under a key and a file name of its own; without
it the export, the key and the engine are exactly what they were.
"""
from __future__ import annotations

import hashlib
import io
import time
import warnings
from pathlib import Path
from typing import Any, Tuple, Union, cast

import torch
from torch import nn

from ..gridtracknet import HEIGHT, WIDTH
from .build_engines import _write_bytes_atomic
from .engine_session import TensorRTRuntimeSession, import_tensorrt

ENGINE_RECIPE = 1
INPUT_NAME = "frames"
OUTPUT_NAME = "grids"
VISIBILITY_OUTPUT_NAME = "visibility"      # the head's logits, (batch, FRAMES_PER_UNIT), when exported
_ONNX_OPSET = 17
_HASH_CHUNK = 1 << 20


def weights_digest(weights: Path) -> str:
    """The SHA-256 of the weights file, hex."""
    digest = hashlib.sha256()
    with open(weights, "rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_key(weights: Path, arch: str, batch: int, device: torch.device, visibility: bool = False) -> str:
    """The cache key of an engine (see the module docstring), 16 hex digits."""
    trt = import_tensorrt()
    properties = torch.cuda.get_device_properties(device)
    parts = [
        f"recipe={ENGINE_RECIPE}", f"weights={weights_digest(weights)}", f"arch={arch}", f"batch={batch}",
        f"input={HEIGHT}x{WIDTH}", f"tensorrt={trt.__version__}", f"torch={torch.__version__}",
        f"gpu={properties.name}", f"sm={properties.major}.{properties.minor}",
    ]
    if visibility:
        parts.append(f"outputs={OUTPUT_NAME},{VISIBILITY_OUTPUT_NAME}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def engine_path(weights: Path, arch: str, batch: int, cache_dir: Path, device: torch.device,
                visibility: bool = False) -> Path:
    """Where the engine for `weights` at `batch` units lives in `cache_dir`."""
    head = "vis-" if visibility else ""
    return Path(cache_dir) / (f"{Path(weights).stem}-{arch}-b{batch}-{head}"
                              f"{engine_key(weights, arch, batch, device, visibility)}.engine")


class _WithVisibility(nn.Module):
    """`model` answering ``(grids, visibility logits)``: the traced form of the two-output export."""

    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grids, logits = self.model(value, return_vis=True)
        return grids, logits


def export_onnx(model: Any, batch: int, device: torch.device, visibility: bool = False) -> bytes:
    """`model` (an FP16 eval module on `device`) as ONNX bytes at a static batch of `batch` units;
    with `visibility`, its visibility head as a second output."""
    probe = torch.zeros((batch, model.INPUT_CHANNELS, HEIGHT, WIDTH), dtype=torch.float16, device=device)
    buffer = io.BytesIO()
    traced: Union[nn.Module, Any] = _WithVisibility(model) if visibility else model
    outputs = [OUTPUT_NAME, VISIBILITY_OUTPUT_NAME] if visibility else [OUTPUT_NAME]
    with warnings.catch_warnings(), torch.inference_mode(False), torch.no_grad():
        # The trace treats the models' shape checks as constants, which they are at a static shape.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(traced, (probe,), cast(Any, buffer), input_names=[INPUT_NAME], output_names=outputs,
                          opset_version=_ONNX_OPSET, dynamo=False)
    return buffer.getvalue()


def build_serialized_engine(onnx_bytes: bytes) -> bytes:
    """An FP16 TensorRT engine from the ONNX bytes; a parse or build failure raises."""
    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the ball model's ONNX export: {errors}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the ball model engine")
    return bytes(serialized)


def ensure_ball_engine(weights: Path, model: Any, batch: int, cache_dir: Path, device: torch.device,
                       visibility: bool = False) -> Path:
    """The cached engine for `weights` (whose loaded network is `model`) at `batch` units, built
    and written first when the cache has none for this key; with `visibility`, the two-output
    engine (``export_onnx``)."""
    path = engine_path(weights, model.ARCH, batch, cache_dir, device, visibility)
    if path.is_file():
        return path
    head = ", with the visibility head" if visibility else ""
    print(f"[detector] building the TensorRT engine for {Path(weights).name} ({model.ARCH}, {batch} units a call{head}); "
          f"cached at {path}")
    started = time.perf_counter()
    _write_bytes_atomic(path, build_serialized_engine(export_onnx(model, batch, device, visibility=visibility)))
    print(f"[detector] TensorRT engine built in {time.perf_counter() - started:.1f}s")
    return path


class BallEngine:
    """The ball model's engine, called like the model on a ``(batch, 15, 432, 768)`` FP16 tensor
    of exactly ``batch`` units. It runs on ``stream`` (wait on it, or work on it, before reading the
    output) and returns the session's output buffer, which the next call overwrites. It carries the
    model's contract (``ARCH``, ``GRID_COLS``, ``GRID_ROWS``, ``INPUT_CHANNELS``,
    ``FRAMES_PER_UNIT``) and, like the model, answers ``(grids, visibility logits)`` to
    ``return_vis`` when it was built with the head (`visibility`)."""

    def __init__(self, path: Path, model: Any, device: torch.device, visibility: bool = False):
        self.session = TensorRTRuntimeSession(str(path), device=device, async_execute=True, output_slots=1)
        if not self.session.use_async or self.session.stream is None:
            raise RuntimeError("the ball engine needs TensorRT's asynchronous execution")
        self.stream: torch.cuda.Stream = self.session.stream
        self.batch = int(self.session.bindings[str(self.session.input_name)].shape[0])
        self.has_vis_head = bool(visibility)
        if visibility and sorted(self.session.output_names) != [OUTPUT_NAME, VISIBILITY_OUTPUT_NAME]:
            raise RuntimeError(f"{path} has outputs {sorted(self.session.output_names)}, not the grids and the visibility head")
        for name in ("ARCH", "GRID_COLS", "GRID_ROWS", "INPUT_CHANNELS", "FRAMES_PER_UNIT"):
            setattr(self, name, getattr(model, name))

    def __call__(self, batch: torch.Tensor, return_vis: bool = False
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.session.forward_tensor(batch)
        if not return_vis:
            return outputs[0]
        if not self.has_vis_head:
            raise ValueError("this ball engine was built without the visibility head")
        return outputs[0], outputs[1]
