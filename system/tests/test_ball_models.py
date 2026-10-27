"""tennis_tracker/ball_models.py: the architecture ids, the suffix each one's weights carry, the
architecture read off a weights file (the suffix of an .npz, the record of a .pt) and the one loader
that dispatches on it; and the pipeline builder that sends both suffixes to the five-frame prepass
detector. No CUDA: the loaders are patched or run on the CPU."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from tennis_tracker import ball_models, pipeline
from tennis_tracker.ball_models import (ARCHITECTURES, GRIDTRACKNET, HEATMAP_S4, arch_of, load_ball_model, model_class,
                                        weights_suffix)
from tennis_tracker.config import Config
from tennis_tracker.gridtracknet import GridTrackNet
from tennis_tracker.heatmapnet import HeatmapNet, save_model

CPU = torch.device("cpu")


def test_ids_classes_suffixes_and_the_contract_agree():
    assert ARCHITECTURES == (GRIDTRACKNET, HEATMAP_S4) == ("gridtracknet", "heatmap_s4")
    assert model_class(GRIDTRACKNET) is GridTrackNet and model_class(HEATMAP_S4) is HeatmapNet
    assert [weights_suffix(name) for name in ARCHITECTURES] == [".npz", ".pt"]
    for name in ARCHITECTURES:
        assert model_class(name).ARCH == name
        for attribute in ("GRID_COLS", "GRID_ROWS", "INPUT_CHANNELS", "FRAMES_PER_UNIT"):
            assert hasattr(model_class(name), attribute), (name, attribute)
    with pytest.raises(ValueError, match="unknown ball-detector architecture 'yolo'"):
        model_class("yolo")


def test_arch_of_reads_the_suffix_of_an_npz_and_the_record_of_a_pt(tmp_path):
    """A .pt names its architecture inside: arch_of reads it, and a record of an unknown
    architecture (the two removed on 2026-10-26 among them), an unreadable one or another suffix
    is refused in a line that says so. An .npz is GridTrackNet without being read."""
    assert arch_of(Path("x/best.npz")) == GRIDTRACKNET and arch_of(tmp_path / "missing.NPZ") == GRIDTRACKNET
    save_model(HeatmapNet(), tmp_path / "heat.PT")
    assert arch_of(tmp_path / "heat.PT") == HEATMAP_S4
    for gone in ("yolo", "memory_s4", "heatmap_s2"):
        torch.save({"arch": gone, "state_dict": {}}, tmp_path / "other.pt")
        with pytest.raises(ValueError, match=f"names the architecture '{gone}', not one a .pt record holds "
                                             "\\(\\['heatmap_s4'\\]\\)"):
            arch_of(tmp_path / "other.pt")
    torch.save({"arch": GRIDTRACKNET, "state_dict": {}}, tmp_path / "grid.pt")       # an .npz architecture in a .pt
    with pytest.raises(ValueError, match="names the architecture 'gridtracknet', not one a .pt record holds"):
        arch_of(tmp_path / "grid.pt")
    with pytest.raises(ValueError, match="not a readable ball-detector weights file"):
        arch_of(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="end in .npz .* or .pt"):
        arch_of(Path("ball.engine"))


def test_load_ball_model_dispatches_on_the_architecture(tmp_path):
    """A .pt loads through heatmapnet.load_model (the real thing, on the CPU: a HeatmapNet record
    gives an FP16 eval HeatmapNet); an .npz goes to gridtracknet.load_model, which is patched since
    no NPZ is built here."""
    save_model(HeatmapNet(), tmp_path / "heat.pt")
    heat = load_ball_model(tmp_path / "heat.pt", CPU)
    assert type(heat) is HeatmapNet and not heat.training and heat.head.weight.dtype == torch.float16
    with patch.object(ball_models.gridtracknet, "load_model", return_value="grid") as grid_loader:
        assert load_ball_model(tmp_path / "net.npz", CPU) == "grid"
    grid_loader.assert_called_once_with(tmp_path / "net.npz", CPU)
    with pytest.raises(ValueError, match="end in .npz"):
        load_ball_model(tmp_path / "net.engine", CPU)


@pytest.mark.parametrize("suffix", [".npz", ".pt"])
def test_the_pipeline_builder_sends_both_weights_suffixes_to_the_prepass_detector(tmp_path, suffix):
    weights = tmp_path / f"weights{suffix}"
    weights.write_bytes(b"placeholder")
    cfg = Config(model_path=str(weights))
    with patch.object(pipeline, "GridTrackNetBallDetector", return_value="prepass") as detector:
        assert pipeline._build_ball_detector(cfg) == "prepass"
    detector.assert_called_once_with(str(weights.resolve()), cfg)
    with pytest.raises(FileNotFoundError, match="Ball-detector weights not found"):
        pipeline._build_ball_detector(Config(model_path=str(tmp_path / f"missing{suffix}")))


def test_the_pipeline_builder_still_resolves_an_engine_beside_other_paths(tmp_path):
    with patch.object(pipeline, "resolve_engine_path", return_value=None):
        with pytest.raises(RuntimeError, match="Expected a .engine file"):
            pipeline._build_ball_detector(Config(model_path=str(tmp_path / "ball.onnx")))


def test_the_prepass_detector_warms_up_with_the_models_channel_count(capsys):
    """The detector reads INPUT_CHANNELS, ARCH and the grid off the model it loaded, so a
    HeatmapNet needs nothing of it changed."""
    from tennis_tracker.engines import ball_detectors
    seen = {}

    class Probe(torch.nn.Module):
        ARCH, GRID_COLS, GRID_ROWS, INPUT_CHANNELS, FRAMES_PER_UNIT = "heatmap_s4", 192, 108, 15, 5

        def forward(self, value):
            seen["shape"] = tuple(value.shape)
            return value

    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch, "device", return_value=CPU), \
         patch.object(torch.cuda, "synchronize"), \
         patch.object(ball_detectors, "load_ball_model", return_value=Probe()):
        detector = ball_detectors.GridTrackNetBallDetector("weights.pt", Config(device="0"))
    assert seen["shape"] == (1, 15, 432, 768) and isinstance(detector.model, Probe)
    assert "[detector] heatmap_s4 FP16, 192x108 grid" in capsys.readouterr().out
