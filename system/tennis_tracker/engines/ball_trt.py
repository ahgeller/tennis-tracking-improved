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

A network that streams (``streamnet.StreamNet.streaming_halves``) is built as two engines instead:
its encoder (``ENCODER``: a batch of transitions, ``(frame, previous)`` to the features at strides
4, 8 and 16) and its decoder (``DECODER``: a batch of units from their transitions' features), each
cached the same way with the half's name in the key and the file name. The prepass exports them
from the deployed network (``ball_models.deployed``: the RepVGG branches folded, which TensorRT
cannot do itself), and ``open_stream_engines`` runs them on one stream at the addresses the
prepass's ``ball_gpu.TransitionRing`` hands them.
"""
from __future__ import annotations

import hashlib
import io
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, cast

import torch

from ..gridtracknet import HEIGHT, WIDTH
from .ball_gpu import StreamHalves
from .build_engines import _write_bytes_atomic
from .engine_files import iter_engine_io
from .engine_session import TensorRTRuntimeSession, _deserialize_engine, _torch_dtype, import_tensorrt

ENGINE_RECIPE = 1
INPUT_NAME = "frames"
OUTPUT_NAME = "grids"
ENCODER, DECODER = "encoder", "decoder"             # the halves of a streaming network
ENCODER_INPUTS = ("frame", "previous")
FEATURE_NAMES = ("quarter", "eighth", "bottom")     # the encoder's outputs, the decoder's inputs
_ONNX_OPSET = 17
_HASH_CHUNK = 1 << 20


def weights_digest(weights: Path) -> str:
    """The SHA-256 of the weights file, hex."""
    digest = hashlib.sha256()
    with open(weights, "rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_key(weights: Path, arch: str, batch: int, device: torch.device, part: Optional[str] = None) -> str:
    """The cache key of an engine (see the module docstring), 16 hex digits; `part` names the half
    of a streaming network (``ENCODER``, ``DECODER``), None the whole model (whose key it leaves
    as it was before halves existed)."""
    trt = import_tensorrt()
    properties = torch.cuda.get_device_properties(device)
    parts = [
        f"recipe={ENGINE_RECIPE}", f"weights={weights_digest(weights)}", f"arch={arch}", f"batch={batch}",
        f"input={HEIGHT}x{WIDTH}", f"tensorrt={trt.__version__}", f"torch={torch.__version__}",
        f"gpu={properties.name}", f"sm={properties.major}.{properties.minor}",
    ] + ([f"part={part}"] if part is not None else [])
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def engine_path(weights: Path, arch: str, batch: int, cache_dir: Path, device: torch.device,
                part: Optional[str] = None) -> Path:
    """Where the engine for `weights` at `batch` units (transitions, for an ``ENCODER``) lives in
    `cache_dir`."""
    name = arch if part is None else f"{arch}-{part}"
    return Path(cache_dir) / f"{Path(weights).stem}-{name}-b{batch}-{engine_key(weights, arch, batch, device, part)}.engine"


def _onnx_bytes(module: Any, probes: Tuple[torch.Tensor, ...], inputs: Sequence[str], outputs: Sequence[str]) -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings(), torch.inference_mode(False), torch.no_grad():
        # The trace treats the models' shape checks as constants, which they are at a static shape.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(module, probes, cast(Any, buffer), input_names=list(inputs), output_names=list(outputs),
                          opset_version=_ONNX_OPSET, dynamo=False)
    return buffer.getvalue()


def export_onnx(model: Any, batch: int, device: torch.device) -> bytes:
    """`model` (an FP16 eval module on `device`) as ONNX bytes at a static batch of `batch` units."""
    probe = torch.zeros((batch, model.INPUT_CHANNELS, HEIGHT, WIDTH), dtype=torch.float16, device=device)
    return _onnx_bytes(model, (probe,), [INPUT_NAME], [OUTPUT_NAME])


def export_halves(encoder: Any, decoder: Any, batch: int, device: torch.device) -> Dict[str, bytes]:
    """A streaming network's halves (FP16 eval modules on `device`, ``streaming_halves``) as ONNX
    bytes by ``ENCODER`` / ``DECODER``, each at a static batch of `batch`: the encoder on
    ``(batch, 3, HEIGHT, WIDTH)`` frame pairs, the decoder on the features of `batch` units, each
    unit's ``PAIRS_PER_UNIT`` transitions concatenated per stride."""
    frame = torch.zeros((batch, 3, HEIGHT, WIDTH), dtype=torch.float16, device=device)
    with torch.no_grad():
        levels = encoder(frame, frame)
    pairs = int(encoder.model.PAIRS_PER_UNIT)
    features = tuple(torch.zeros((batch, pairs * level.shape[1], *level.shape[2:]), dtype=level.dtype, device=device)
                     for level in levels)
    return {ENCODER: _onnx_bytes(encoder, (frame, frame), ENCODER_INPUTS, FEATURE_NAMES),
            DECODER: _onnx_bytes(decoder, features, FEATURE_NAMES, [OUTPUT_NAME])}


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


def _ensure_engine(path: Path, what: str, export: Callable[[], bytes]) -> Path:
    """`path`, built from `export`'s ONNX bytes and written first when the cache does not hold it."""
    if path.is_file():
        return path
    print(f"[detector] building the TensorRT engine for {what}; cached at {path}")
    started = time.perf_counter()
    _write_bytes_atomic(path, build_serialized_engine(export()))
    print(f"[detector] TensorRT engine built in {time.perf_counter() - started:.1f}s")
    return path


def ensure_ball_engine(weights: Path, model: Any, batch: int, cache_dir: Path, device: torch.device) -> Path:
    """The cached engine for `weights` (whose loaded network is `model`) at `batch` units, built
    and written first when the cache has none for this key."""
    path = engine_path(weights, model.ARCH, batch, cache_dir, device)
    return _ensure_engine(path, f"{Path(weights).name} ({model.ARCH}, {batch} units a call)",
                          lambda: export_onnx(model, batch, device))


def ensure_stream_engines(weights: Path, arch: str, encoder: Any, decoder: Any, batch: int, cache_dir: Path,
                          device: torch.device) -> Dict[str, Path]:
    """The cached ``ENCODER`` and ``DECODER`` engines of the streaming network `weights` holds (its
    deployed halves `encoder` and `decoder`) at `batch` a call, by half, each built and written first
    when the cache has none for its key; one export serves both."""
    exported: Dict[str, bytes] = {}

    def export(part: str) -> bytes:
        if not exported:
            exported.update(export_halves(encoder, decoder, batch, device))
        return exported[part]

    paths = {}
    for part in (ENCODER, DECODER):
        paths[part] = _ensure_engine(engine_path(weights, arch, batch, cache_dir, device, part),
                                     f"{Path(weights).name} ({arch} {part}, {batch} a call)",
                                     lambda part=part: export(part))      # type: ignore[misc]
    return paths


class BallEngine:
    """The ball model's engine, called like the model on a ``(batch, 15, 432, 768)`` FP16 tensor
    of exactly ``batch`` units. It runs on ``stream`` (wait on it, or work on it, before reading the
    output) and returns the session's output buffer, which the next call overwrites. It carries the
    model's contract (``ARCH``, ``GRID_COLS``, ``GRID_ROWS``, ``INPUT_CHANNELS``,
    ``FRAMES_PER_UNIT``)."""

    def __init__(self, path: Path, model: Any, device: torch.device):
        self.session = TensorRTRuntimeSession(str(path), device=device, async_execute=True, output_slots=1)
        if not self.session.use_async or self.session.stream is None:
            raise RuntimeError("the ball engine needs TensorRT's asynchronous execution")
        self.stream: torch.cuda.Stream = self.session.stream
        self.batch = int(self.session.bindings[str(self.session.input_name)].shape[0])
        for name in ("ARCH", "GRID_COLS", "GRID_ROWS", "INPUT_CHANNELS", "FRAMES_PER_UNIT"):
            setattr(self, name, getattr(model, name))

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.session.forward_tensor(batch)[0]


class _EngineRun:
    """One engine enqueued on `stream` at the input addresses each call gives, into output buffers
    of its own (returned in `outputs`' order; the next call overwrites them)."""

    def __init__(self, path: Path, inputs: Sequence[str], outputs: Sequence[str], device: torch.device,
                 stream: torch.cuda.Stream):
        trt = import_tensorrt()
        self.engine, self.context = _deserialize_engine(trt, str(path))
        tensors = {io.name: io for io in iter_engine_io(self.engine, trt)}
        if set(tensors) != {*inputs, *outputs}:
            raise RuntimeError(f"{Path(path).name} has the tensors {sorted(tensors)}, not {[*inputs, *outputs]}")
        self.stream = stream
        self.inputs = {name: tuple(tensors[name].shape) for name in inputs}
        self.outputs = [torch.empty(tuple(tensors[name].shape), dtype=_torch_dtype(trt.nptype(tensors[name].dtype)),
                                    device=device) for name in outputs]
        for name, buffer in zip(outputs, self.outputs):
            self.context.set_tensor_address(name, int(buffer.data_ptr()))

    def __call__(self, *values: torch.Tensor) -> Sequence[torch.Tensor]:
        for (name, shape), value in zip(self.inputs.items(), values):
            if tuple(value.shape) != shape or not value.is_contiguous():
                raise RuntimeError(f"TensorRT input {name} must be a contiguous {shape}, not {tuple(value.shape)}")
            self.context.set_tensor_address(name, int(value.data_ptr()))
        current = torch.cuda.current_stream(self.stream.device)
        if current != self.stream:
            self.stream.wait_stream(current)
        if self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream)) is False:
            raise RuntimeError("TensorRT async execution failed.")
        return self.outputs


def open_stream_engines(paths: Dict[str, Path], batch: int, device: torch.device) -> StreamHalves:
    """The ``ENCODER`` and ``DECODER`` engines at `paths` (``ensure_stream_engines``) as the
    prepass's ``StreamHalves``, both on one new stream (the prepass works on it)."""
    stream = torch.cuda.Stream(device=device)
    encoder = _EngineRun(paths[ENCODER], ENCODER_INPUTS, FEATURE_NAMES, device, stream)
    decoder = _EngineRun(paths[DECODER], FEATURE_NAMES, (OUTPUT_NAME,), device, stream)
    return StreamHalves(encode=encoder, decode=lambda *features: decoder(*features)[0], batch=batch, stream=stream)
