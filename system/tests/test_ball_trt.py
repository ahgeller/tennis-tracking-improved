"""The prepass engine's cache (engines/ball_trt.py) and the detector's choice of runtime, with
TensorRT and the GPU stood in for: no CUDA or TensorRT needed."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tennis_tracker import streamnet
from tennis_tracker.config import Config
from tennis_tracker.engines import ball_detectors, ball_trt
from tests._helpers import CpuGridTrackNet

CUDA = torch.device("cuda:0")


@pytest.fixture
def fake_platform():
    """TensorRT 10.x on an 'RTX test' GPU, as far as the cache key can tell."""
    properties = SimpleNamespace(name="RTX test", major=12, minor=0)
    trt = SimpleNamespace(__version__="10.15.1.29")
    with patch.object(ball_trt, "import_tensorrt", return_value=trt), \
         patch.object(torch.cuda, "get_device_properties", return_value=properties):
        yield trt, properties


def test_the_cache_key_follows_the_weights_bytes_the_batch_and_the_platform(tmp_path, fake_platform):
    trt, properties = fake_platform
    weights = tmp_path / "ball.pt"
    weights.write_bytes(b"one")
    key = ball_trt.engine_key(weights, "heatmap_s4", 8, CUDA)
    assert key == ball_trt.engine_key(weights, "heatmap_s4", 8, CUDA) and len(key) == 16
    assert key != ball_trt.engine_key(weights, "heatmap_s4", 4, CUDA)
    assert key != ball_trt.engine_key(weights, "stream_s4", 8, CUDA)
    trt.__version__ = "10.16.0"
    assert key != ball_trt.engine_key(weights, "heatmap_s4", 8, CUDA)
    trt.__version__ = "10.15.1.29"
    properties.minor = 1
    assert key != ball_trt.engine_key(weights, "heatmap_s4", 8, CUDA)
    properties.minor = 0
    weights.write_bytes(b"two")                       # retrained in place: same name, new engine
    assert key != ball_trt.engine_key(weights, "heatmap_s4", 8, CUDA)
    path = ball_trt.engine_path(weights, "heatmap_s4", 8, tmp_path / "cache", CUDA)
    assert path.parent == tmp_path / "cache" and path.name.startswith("ball-heatmap_s4-b8-") and path.suffix == ".engine"


def test_an_engine_is_built_once_and_found_after(tmp_path, fake_platform, capsys):
    weights = tmp_path / "ball.pt"
    weights.write_bytes(b"weights")
    model = SimpleNamespace(ARCH="heatmap_s4")
    with patch.object(ball_trt, "export_onnx", return_value=b"onnx") as export, \
         patch.object(ball_trt, "build_serialized_engine", return_value=b"plan") as build:
        path = ball_trt.ensure_ball_engine(weights, model, 8, tmp_path / "cache", CUDA)
        again = ball_trt.ensure_ball_engine(weights, model, 8, tmp_path / "cache", CUDA)
        weights.write_bytes(b"retrained")
        rebuilt = ball_trt.ensure_ball_engine(weights, model, 8, tmp_path / "cache", CUDA)
    assert path == again and path.read_bytes() == b"plan" and rebuilt != path and rebuilt.is_file()
    assert export.call_count == build.call_count == 2
    export.assert_called_with(model, 8, CUDA)
    build.assert_called_with(b"onnx")
    assert "building the TensorRT engine for ball.pt" in capsys.readouterr().out
    assert not list((tmp_path / "cache").glob("*.tmp"))


def _detector(runtime, engine=None, error=None):
    """A CPU detector whose device the runtime choice is told is a GPU."""
    cfg = Config(device="0", ball_prepass_runtime=runtime)
    detector = CpuGridTrackNet(cfg, model=lambda batch: batch, decode=None, frame_tensor=None)
    detector.device = CUDA
    with patch.object(ball_trt, "ensure_ball_engine", return_value=Path("e.engine"), side_effect=error), \
         patch.object(ball_trt, "BallEngine", return_value=engine):
        return detector._open_engine(Path("w.pt"))


def test_the_runtime_is_tensorrt_on_a_gpu_and_torch_on_request_or_on_failure(capsys):
    assert _detector("tensorrt", engine="engine") == "engine"
    assert _detector("torch", engine="engine") is None
    assert _detector("tensorrt", error=RuntimeError("no parser for op X")) is None
    assert "TensorRT engine unavailable, running the PyTorch model (no parser for op X)" in capsys.readouterr().out
    with pytest.raises(ValueError, match="ball_prepass_runtime must be one of"):
        _detector("onnxruntime")


def test_a_detector_off_the_gpu_runs_the_pytorch_model():
    detector = CpuGridTrackNet(Config(device="0"), model=lambda batch: batch, decode=None, frame_tensor=None)
    assert detector.cfg.ball_prepass_runtime == "tensorrt" and detector.engine is None


def test_a_streaming_networks_halves_have_keys_of_their_own_and_the_whole_model_keeps_its(tmp_path, fake_platform):
    """Each half of a streaming network is cached under its own key and file name; the whole
    model's key is the one it had before halves existed (no part line in the key)."""
    weights = tmp_path / "stream.pt"
    weights.write_bytes(b"one")
    whole = ball_trt.engine_key(weights, "stream_s4", 4, CUDA)
    trt, properties = fake_platform
    lines = ["recipe=1", f"weights={ball_trt.weights_digest(weights)}", "arch=stream_s4", "batch=4", "input=432x768",
             f"tensorrt={trt.__version__}", f"torch={torch.__version__}", "gpu=RTX test", "sm=12.0"]
    assert whole == ball_trt.hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    keys = {ball_trt.engine_key(weights, "stream_s4", 4, CUDA, part) for part in (ball_trt.ENCODER, ball_trt.DECODER)}
    assert len(keys) == 2 and whole not in keys
    encoder = ball_trt.engine_path(weights, "stream_s4", 4, tmp_path, CUDA, ball_trt.ENCODER)
    assert encoder.name.startswith("stream-stream_s4-encoder-b4-")
    assert ball_trt.engine_path(weights, "stream_s4", 4, tmp_path, CUDA).name.startswith("stream-stream_s4-b4-")


def test_a_streaming_networks_engines_come_from_one_export_and_are_found_after(tmp_path, fake_platform, capsys):
    weights = tmp_path / "stream.pt"
    weights.write_bytes(b"weights")
    onnx = {ball_trt.ENCODER: b"encoder onnx", ball_trt.DECODER: b"decoder onnx"}
    with patch.object(ball_trt, "export_halves", return_value=onnx) as export,          patch.object(ball_trt, "build_serialized_engine", side_effect=lambda data: data + b" plan") as build:
        paths = ball_trt.ensure_stream_engines(weights, "stream_s4", "enc", "dec", 4, tmp_path / "cache", CUDA)
        again = ball_trt.ensure_stream_engines(weights, "stream_s4", "enc", "dec", 4, tmp_path / "cache", CUDA)
    assert paths == again and export.call_count == 1 and build.call_count == 2
    export.assert_called_with("enc", "dec", 4, CUDA)
    assert {part: path.read_bytes() for part, path in paths.items()} == {
        ball_trt.ENCODER: b"encoder onnx plan", ball_trt.DECODER: b"decoder onnx plan"}
    out = capsys.readouterr().out
    assert "stream.pt (stream_s4 encoder, 4 a call)" in out and "stream.pt (stream_s4 decoder, 4 a call)" in out


def _stream_detector(monkeypatch, **settings):
    """A CPU detector holding a small StreamNet; its PyTorch halves warm up at 32x48."""
    monkeypatch.setattr(ball_detectors, "HEIGHT", 32)
    monkeypatch.setattr(ball_detectors, "WIDTH", 48)
    return CpuGridTrackNet(Config(device="0", **settings), model=streamnet.StreamNet().eval(), decode=None,
                           frame_tensor=None)


def test_a_streaming_network_streams_as_engines_on_a_gpu_and_as_pytorch_on_request_or_on_failure(monkeypatch, capsys):
    """A network with streaming halves streams (deployed, so folded) unless streaming is off, where
    it runs unit by unit; the halves are TensorRT engines on a GPU, else (on request, off the GPU,
    or when TensorRT fails, with a message) the PyTorch modules. A network without halves never streams."""
    detector = _stream_detector(monkeypatch)
    assert detector.stream_halves is not None and detector.stream_halves.stream is None and detector.engine is None
    assert detector.stream_halves.batch == ball_detectors._UNITS_PER_INFERENCE
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in detector.model.modules())
    assert "stream_s4 FP16 streaming, 4 a call" in capsys.readouterr().out

    detector.device = CUDA
    with patch.object(ball_trt, "ensure_stream_engines", return_value={"encoder": "e", "decoder": "d"}) as ensure,          patch.object(ball_trt, "open_stream_engines", return_value="engines") as open_engines:
        assert detector._open_stream(Path("w.pt")) == "engines"
    assert ensure.call_args.args[:2] == (Path("w.pt"), "stream_s4") and ensure.call_args.args[4] == 4
    open_engines.assert_called_with({"encoder": "e", "decoder": "d"}, 4, CUDA)
    with patch.object(ball_trt, "ensure_stream_engines", side_effect=RuntimeError("no parser for op X")):
        halves = detector._open_stream(Path("w.pt"))
    assert halves.stream is None and halves.batch == ball_detectors._UNITS_PER_INFERENCE
    assert "TensorRT engines unavailable, streaming the PyTorch model (no parser for op X)" in capsys.readouterr().out
    detector.cfg = Config(device="0", ball_prepass_runtime="torch")
    with patch.object(ball_trt, "ensure_stream_engines") as ensure:
        assert detector._open_stream(Path("w.pt")).stream is None
    ensure.assert_not_called()
    detector.cfg = Config(device="0", ball_prepass_runtime="onnxruntime")
    with pytest.raises(ValueError, match="ball_prepass_runtime must be one of"):
        detector._open_stream(Path("w.pt"))
    detector.cfg, detector.model = Config(device="0", ball_prepass_streaming=False), streamnet.StreamNet().eval()
    assert detector._open_stream(Path("w.pt")) is None
    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in detector.model.modules())     # left as loaded
    plain = CpuGridTrackNet(Config(device="0"), model=lambda batch: batch, decode=None, frame_tensor=None)
    assert plain.stream_halves is None
