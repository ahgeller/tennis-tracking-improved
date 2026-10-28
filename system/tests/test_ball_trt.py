"""The prepass engine's cache (engines/ball_trt.py) and the detector's choice of runtime, with
TensorRT and the GPU stood in for: no CUDA or TensorRT needed."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

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
