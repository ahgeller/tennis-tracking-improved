"""Measure the ball-detector architectures the same way: parameters, GMAC per unit and per frame,
and TensorRT FP16 latency on this GPU at the batch the prepass runs (four units per call).

    python tools/gpu_lock.py -- python tools/bench_ball_archs.py --out output/bench_archs

For every architecture the deployed network (``reparameterize`` where it has one: RepVGG branches
and BatchNorm folded) is exported to ONNX (opset 17, static shapes, the TorchScript exporter),
checked against PyTorch, built into an FP16 engine with the TensorRT Python builder and timed with
CUDA events. ``stream_s4`` is also built as its two streaming halves, the encoder of one
transition and the decoder of one unit from four cached transitions, and timed as the prepass at
window step 1 would run them: per call, four new transitions encoded and four units decoded.
Weights are random (latency and compute do not depend on them). Timings are only valid under
``tools/gpu_lock.py``.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

from tennis_tracker import ball_models, streamnet  # noqa: E402
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH, frame_tensor  # noqa: E402

UNITS_PER_CALL = 4          # ball_detectors._UNITS_PER_INFERENCE


def conv_gmac(model: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> float:
    """Multiply-accumulates of every convolution in one call of `model` on `inputs`, in billions,
    for the whole batch."""
    total = [0]

    def hook(module: nn.Module, _inputs, output: torch.Tensor) -> None:
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total[0] += output.numel() * (module.in_channels // module.groups) * kernel

    handles = [module.register_forward_hook(hook) for module in model.modules()]
    with torch.no_grad():
        model(*inputs)
    for handle in handles:
        handle.remove()
    return total[0] / 1e9


def sample_units(video: Path, skip: int = 600) -> torch.Tensor:
    """``UNITS_PER_CALL`` overlapping five-frame units of one phase (every second frame) of
    `video` from frame `skip` on, as the prepass builds them (``gridtracknet.frame_tensor``), in
    float32 on the CPU."""
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, skip)
    frames = []
    while len(frames) < UNITS_PER_CALL + FRAMES_PER_UNIT - 1:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"{video} ended before {UNITS_PER_CALL} units were read")
        ok, _ = capture.read()                    # the other phase
        frames.append(frame_tensor(frame, torch.device("cpu")).float())
    capture.release()
    return torch.stack([torch.cat(frames[start:start + FRAMES_PER_UNIT]) for start in range(UNITS_PER_CALL)])


def calibrate_batch_norms(model: nn.Module, units: torch.Tensor) -> None:
    """Running statistics measured on `units` (a cumulative average over a few train-mode passes),
    as a trained network has them, so the deployed network's activations stay the size FP16 is
    built for and the fold and TensorRT checks compare like with like; then eval mode."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    with torch.no_grad():
        for start in range(0, len(units), 2):
            model(units[start:start + 2])
    model.eval()


class _Encoder(nn.Module):
    def __init__(self, model: streamnet.StreamNet):
        super().__init__()
        self.model = model

    def forward(self, frame: torch.Tensor, previous: torch.Tensor):
        return self.model.encode_frame(frame, previous)


class _Decoder(nn.Module):
    """One unit's decoder on the four transitions' features already concatenated per stride
    (what a prepass's feature ring hands it)."""

    def __init__(self, model: streamnet.StreamNet):
        super().__init__()
        self.model = model

    def forward(self, quarter: torch.Tensor, eighth: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        return self.model._decode(quarter, eighth, bottom)


def export_onnx(model: nn.Module, inputs: Tuple[torch.Tensor, ...], path: Path, outputs: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, inputs, str(path), opset_version=17, dynamo=False,
                      input_names=[f"input{index}" for index in range(len(inputs))], output_names=outputs)


def build_engine(onnx_path: Path, engine_path: Path) -> bytes:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(index)) for index in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT could not build {onnx_path}")
    data = bytes(serialized)
    engine_path.write_bytes(data)
    return data


class Engine:
    """A deserialised engine with torch buffers bound to every I/O tensor."""

    def __init__(self, data: bytes):
        import tensorrt as trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(data)
        self.context = self.engine.create_execution_context()
        self.buffers: Dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = torch.float16 if self.engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
            self.buffers[name] = torch.zeros(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def run(self, stream: torch.cuda.Stream) -> None:
        self.context.execute_async_v3(stream.cuda_stream)


def time_ms(step: Callable[[], None], stream: torch.cuda.Stream, iterations: int, repeats: int) -> float:
    """The median over `repeats` of the mean time of one `step` over `iterations`, by CUDA events."""
    with torch.cuda.stream(stream):
        for _ in range(50):
            step()
    stream.synchronize()
    means = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                step()
        end.record(stream)
        end.synchronize()
        means.append(start.elapsed_time(end) / iterations)
    return statistics.median(means)


def check_engine(engine: Engine, inputs: Tuple[torch.Tensor, ...], expected: Tuple[torch.Tensor, ...],
                 stream: torch.cuda.Stream) -> float:
    """The largest absolute difference between the engine's outputs and PyTorch's float32 ones."""
    names = list(engine.buffers)
    for name, value in zip(names, inputs):
        engine.buffers[name].copy_(value)
    engine.run(stream)
    stream.synchronize()
    return max(float((engine.buffers[name].float() - value.float()).abs().max())
               for name, value in zip(names[len(inputs):], expected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "bench_archs")
    parser.add_argument("--archs", nargs="+", default=[name for name in ball_models.ARCHITECTURES if name != "gridtracknet"])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--video", type=Path, default=ROOT / "system" / "sample" / "pomona.mp4",
                        help="the clip whose frames fill the units (the checks compare outputs on real frames)")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    units = sample_units(args.video)
    results = {}
    for arch in args.archs:
        trainable = ball_models.model_class(arch)()
        calibrate_batch_norms(trainable, units)
        params = sum(tensor.numel() for tensor in trainable.parameters())
        deployed = copy.deepcopy(trainable)
        if hasattr(deployed, "reparameterize"):
            deployed.reparameterize()
        single = units[:1]
        with torch.no_grad():
            reference = trainable(single)
            fused_gap = float((deployed(single) - reference).abs().max())
        row = {"params_m": round(params / 1e6, 3),
               "deployed_params_m": round(sum(tensor.numel() for tensor in deployed.parameters()) / 1e6, 3),
               "gmac_per_unit": round(conv_gmac(deployed, (single,)), 3),
               "fuse_max_abs_diff_fp32": fused_gap}
        onnx_path = args.out / f"{arch}_b{UNITS_PER_CALL}.onnx"
        export_onnx(deployed, (units,), onnx_path, ["output"])
        engine = Engine(build_engine(onnx_path, onnx_path.with_suffix(".engine")))
        with torch.no_grad():
            expected = deployed.to(device)(units.to(device))
        row["trt_max_abs_diff"] = check_engine(engine, (units.to(device),), (expected,), stream)
        row["trt_fp16_ms_per_call"] = round(time_ms(lambda: engine.run(stream), stream, args.iterations, args.repeats), 4)
        row["trt_fp16_ms_per_unit"] = round(row["trt_fp16_ms_per_call"] / UNITS_PER_CALL, 4)
        if arch == streamnet.ARCH:
            row.update(stream_halves(deployed.cpu(), units, args, stream, device))
        results[arch] = row
        print(arch, json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    return 0


def stream_halves(model: streamnet.StreamNet, units: torch.Tensor, args: argparse.Namespace,
                  stream: torch.cuda.Stream, device: torch.device) -> Dict[str, float]:
    """The streaming schedule's compute and latency: per call, ``UNITS_PER_CALL`` new transitions
    encoded (one per new frame) and ``UNITS_PER_CALL`` units decoded from the cached features
    (here each unit's own four transitions, as the cache would hold them)."""
    frame, previous = units[:, -3:], units[:, -6:-3]
    encoder, decoder = _Encoder(model).eval(), _Decoder(model).eval()
    with torch.no_grad():
        per_unit = [encoder(units[:, 3 * index + 3:3 * index + 6], units[:, 3 * index:3 * index + 3])
                    for index in range(streamnet.PAIRS_PER_UNIT)]
    cached = tuple(torch.cat([pair[level] for pair in per_unit], dim=1) for level in range(3))
    encode_gmac = conv_gmac(encoder, (frame[:1], previous[:1]))
    decode_gmac = conv_gmac(decoder, tuple(level[:1] for level in cached))
    export_onnx(encoder, (frame, previous), args.out / "stream_encoder.onnx", ["quarter", "eighth", "bottom"])
    export_onnx(decoder, cached, args.out / "stream_decoder.onnx", ["output"])
    encode_engine = Engine(build_engine(args.out / "stream_encoder.onnx", args.out / "stream_encoder.engine"))
    decode_engine = Engine(build_engine(args.out / "stream_decoder.onnx", args.out / "stream_decoder.engine"))
    with torch.no_grad():
        encoder.to(device), decoder.to(device)
        expected_features = encoder(frame.to(device), previous.to(device))
        expected_output = decoder(*(level.to(device) for level in cached))
    encode_gap = check_engine(encode_engine, (frame.to(device), previous.to(device)), expected_features, stream)
    decode_gap = check_engine(decode_engine, tuple(level.to(device) for level in cached), (expected_output,), stream)

    def step() -> None:
        encode_engine.run(stream)
        decode_engine.run(stream)

    encode_ms = time_ms(lambda: encode_engine.run(stream), stream, args.iterations, args.repeats)
    decode_ms = time_ms(lambda: decode_engine.run(stream), stream, args.iterations, args.repeats)
    both_ms = time_ms(step, stream, args.iterations, args.repeats)
    return {"stream_encoder_gmac_per_transition": round(encode_gmac, 3),
            "stream_decoder_gmac_per_unit": round(decode_gmac, 3),
            "stream_gmac_per_frame": round(encode_gmac + decode_gmac, 3),
            "stream_trt_max_abs_diff": max(encode_gap, decode_gap),
            "stream_encoder_ms_per_call": round(encode_ms, 4),
            "stream_decoder_ms_per_call": round(decode_ms, 4),
            "stream_ms_per_call": round(both_ms, 4),
            "stream_ms_per_frame": round(both_ms / UNITS_PER_CALL, 4)}


if __name__ == "__main__":
    raise SystemExit(main())
