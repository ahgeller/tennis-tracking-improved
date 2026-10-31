"""tools/export_coreml.py and the Mac kit's validator: the packages' modules, names, shapes and
metadata, the RepVGG fuse against the trainable network, and the cases the Mac runs, scored here
on the PyTorch graphs. The conversion itself is checked where coremltools installs (macOS, Linux)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tennis_tracker import heatmapnet, streamnet

_PATH = Path(__file__).resolve().parent.parent / "tools" / "export_coreml.py"
_SPEC = importlib.util.spec_from_file_location("export_coreml", _PATH)
export_coreml = importlib.util.module_from_spec(_SPEC)
sys.modules["export_coreml"] = export_coreml
_SPEC.loader.exec_module(export_coreml)
validator = export_coreml.validator

HEIGHT, WIDTH = 64, 128          # a small input both networks take (multiples of 16)


def _randomise_batch_norms(model: nn.Module, seed: int) -> None:
    """Statistics and affines away from the identity, so the fold has something to fold."""
    generator = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            size = module.num_features
            module.running_mean.copy_(torch.randn(size, generator=generator) * 0.1)
            module.running_var.copy_(torch.rand(size, generator=generator) + 0.5)
            module.weight.data.copy_(torch.rand(size, generator=generator) + 0.5)
            module.bias.data.copy_(torch.randn(size, generator=generator) * 0.1)


def _record(tmp_path: Path, model: nn.Module, seed: int) -> Path:
    torch.manual_seed(seed)
    _randomise_batch_norms(model, seed)
    path = tmp_path / f"{model.ARCH}.pt"
    heatmapnet.save_model(model, path)
    return path


@pytest.fixture(scope="module")
def stream_specs(tmp_path_factory):
    weights = _record(tmp_path_factory.mktemp("stream"), streamnet.StreamNet(), 3)
    return {spec.name: spec for spec in export_coreml.stream_specs(weights, HEIGHT, WIDTH)}


@pytest.fixture(scope="module")
def heatmap_spec(tmp_path_factory):
    weights = _record(tmp_path_factory.mktemp("heat"), heatmapnet.HeatmapNet(), 5)
    (spec,) = export_coreml.heatmap_specs(weights, HEIGHT, WIDTH)
    return spec


def _unit(seed: int) -> np.ndarray:
    """A unit of random 8-bit frames as the packages take it: uint8 / 255 in float16."""
    pixels = np.random.default_rng(seed).integers(0, 256, (1, 15, HEIGHT, WIDTH)).astype(np.float32)
    return (pixels / 255.0).astype(np.float16)


def test_the_packages_names_and_shapes(stream_specs, heatmap_spec):
    encoder, decoder, unit = (stream_specs[n] for n in ("stream_s4_encoder", "stream_s4_decoder", "stream_s4_unit"))
    assert [t.name for t in encoder.inputs] == ["frame", "previous"]
    assert all(t.shape == (1, 3, HEIGHT, WIDTH) for t in encoder.inputs)
    assert [(t.name, t.shape) for t in encoder.outputs] == [
        ("quarter", (1, 32, HEIGHT // 4, WIDTH // 4)), ("eighth", (1, 64, HEIGHT // 8, WIDTH // 8)),
        ("bottom", (1, 128, HEIGHT // 16, WIDTH // 16))]
    assert [t.name for t in decoder.inputs] == list(export_coreml.DECODER_INPUTS)
    assert decoder.inputs[0].name == "t1_quarter" and decoder.inputs[-1].name == "t4_bottom"
    assert [t.shape for t in decoder.inputs[:3]] == [t.shape for t in encoder.outputs]
    for spec in (decoder, unit, heatmap_spec):
        assert [(t.name, t.shape) for t in spec.outputs] == [("grids", (1, 15, HEIGHT // 4, WIDTH // 4))]
    for spec in (unit, heatmap_spec):
        assert [(t.name, t.shape) for t in spec.inputs] == [("frames", (1, 15, HEIGHT, WIDTH))]
    assert all(t.description for spec in [*stream_specs.values(), heatmap_spec] for t in [*spec.inputs, *spec.outputs])
    assert 0 < encoder.gmac < unit.gmac < heatmap_spec.gmac


def test_the_tensorrt_halves_use_the_same_names():
    ball_trt = pytest.importorskip("tennis_tracker.engines.ball_trt")
    assert export_coreml.FRAME_INPUTS == ball_trt.ENCODER_INPUTS
    assert export_coreml.FEATURE_NAMES == ball_trt.FEATURE_NAMES
    assert (export_coreml.UNIT_INPUT, export_coreml.GRIDS) == (ball_trt.INPUT_NAME, ball_trt.OUTPUT_NAME)


def test_the_fused_stream_s4_answers_as_the_trainable_one(stream_specs):
    unit = stream_specs["stream_s4_unit"]
    assert any(isinstance(m, nn.BatchNorm2d) for m in unit.trainable.modules())
    assert not any(isinstance(m, nn.BatchNorm2d) for m in unit.fused.modules())
    frames = torch.from_numpy(_unit(1)).float()
    with torch.no_grad():
        expected = unit.trainable(frames)
        assert (unit.fused(frames) - expected).abs().max() < 1e-4
        encoder, decoder = stream_specs["stream_s4_encoder"].fused, stream_specs["stream_s4_decoder"].fused
        features = []
        for pair in range(1, 5):
            features += list(encoder(frames[:, 3 * pair:3 * pair + 3], frames[:, 3 * pair - 3:3 * pair]))
        assert (decoder(*features) - expected).abs().max() < 1e-4


def test_the_heatmap_unit_traces_to_the_trainable_answer(heatmap_spec):
    frames = torch.from_numpy(_unit(2)).float()
    traced = export_coreml.trace(heatmap_spec.fused, heatmap_spec.probes())
    with torch.no_grad():
        assert (traced(frames) - heatmap_spec.trainable(frames)).abs().max() < 1e-4


def test_the_decoder_refuses_a_short_window(stream_specs):
    features = stream_specs["stream_s4_decoder"].probes()[:9]
    with pytest.raises(ValueError, match="12 feature maps"):
        stream_specs["stream_s4_decoder"].fused(*features)


def test_package_metadata_names_the_weights_and_every_tensor(stream_specs):
    spec = stream_specs["stream_s4_decoder"]
    metadata = export_coreml.package_metadata(spec, "0123abcd", "abc1234")
    assert metadata["version"] == "abc1234" and "decoder" in metadata["short_description"]
    assert set(metadata["inputs"]) == set(export_coreml.DECODER_INPUTS)
    assert set(metadata["outputs"]) == {"grids"}
    user = metadata["user_defined"]
    assert user["source_weights_md5"] == "0123abcd" and user["arch"] == "stream_s4" and user["role"] == "decoder"
    assert user["source_weights"] == spec.weights.name and float(user["operating_threshold"]) > 0
    assert all(isinstance(value, str) for value in user.values())
    assert user["fused"].startswith("RepVGG branches and BatchNorm folded")


def test_heatmap_metadata_says_coremltools_folds_its_batch_norms(heatmap_spec):
    user = export_coreml.package_metadata(heatmap_spec, "0123abcd", "abc1234")["user_defined"]
    assert user["fused"].startswith("no RepVGG blocks") and user["arch"] == "heatmap_s4"


def test_the_mac_cases_pass_on_the_pytorch_graphs(tmp_path, stream_specs, heatmap_spec):
    specs = {**stream_specs, heatmap_spec.name: heatmap_spec}
    units = [([0, 2, 4, 6, 8], _unit(3)), ([1, 3, 5, 7, 9], _unit(4))]
    cases = export_coreml.write_samples(tmp_path, specs, units)
    groups = {case["group"] for case in cases}
    assert groups == {"stream_s4_unit", "stream_s4_encoder", "stream_s4_decoder", "stream_s4_chain", "heatmap_s4_unit"}
    assert sum(case["group"] == "stream_s4_encoder" for case in cases) == 2 * 4
    assert np.load(tmp_path / "samples/unit0/stream_s4_features/t4_bottom.npy").dtype == np.float16
    manifest = {"cases": cases}
    traced = {name: export_coreml.trace(spec.fused, spec.probes()) for name, spec in specs.items()}
    for variant in ("fused_fp32", "traced_fp32"):
        scored = validator.evaluate(export_coreml.TorchRunner(specs, traced, variant), manifest, tmp_path, 1e-3)
        assert set(scored) == groups
        assert all(result["pass"] for result in scored.values()), scored
        assert scored["stream_s4_chain"]["summary"]["argmax_exact"] == "10/10"


def test_score_grids_counts_argmax_agreement():
    reference = np.zeros((1, 15, 6, 8), dtype=np.float32)
    output = reference.copy()
    for frame in range(5):
        reference[0, 3 * frame, 2, 3] = 0.9 if frame < 3 else 0.1      # frames 0-2 above a 0.5 threshold
    output[0, 0, 2, 3] = 0.9                                            # frame 0: same cell
    output[0, 3, 3, 4] = 0.95                                           # frame 1: one cell off
    output[0, 6, 5, 0] = 0.95                                           # frame 2: far off
    output[0, 1, 0, 0] = 0.25                                           # an offset difference
    score = validator.score_grids(output, reference, 0.5)
    assert score.exact[:3] == [True, False, False] and score.near[:3] == [True, True, False]
    assert score.confident == [True, True, True, False, False]
    assert score.max_offset_diff == pytest.approx(0.25)
    assert score.max_heat_diff == pytest.approx(0.95)
    assert score.summary()["confident_argmax_exact"] == "1/3"


def test_load_input_slices_a_units_frame(tmp_path):
    frames = np.arange(15 * 2 * 2, dtype=np.float32).reshape(1, 15, 2, 2)
    np.save(tmp_path / "f.npy", frames)
    picked = validator.load_input({"npy": "f.npy", "channels": [3, 6]}, tmp_path)
    assert picked.dtype == np.float16 and picked.shape == (1, 3, 2, 2) and picked.flags["C_CONTIGUOUS"]
    assert np.array_equal(picked, frames[:, 3:6].astype(np.float16))


def test_the_kit_names_every_file_the_export_copies():
    for name in export_coreml.KIT_FILES:
        assert (export_coreml.KIT / name).is_file(), name
    assert b"\r\n" not in (export_coreml.KIT / "validate.sh").read_bytes()


PLAYER_SIZE = 128                # a small player input (a multiple of the head's stride 32)


@pytest.fixture(scope="module")
def player_spec(tmp_path_factory):
    tasks = pytest.importorskip("ultralytics.nn.tasks")
    torch.manual_seed(7)
    model = tasks.DetectionModel("yolo11n.yaml", nc=1, verbose=False).eval()
    _randomise_batch_norms(model, 7)
    weights = tmp_path_factory.mktemp("player") / "player.pt"
    torch.save({"model": model, "train_args": {}}, weights)
    (spec,) = export_coreml.player_specs(weights, PLAYER_SIZE, expected_md5=None)
    return spec


def test_the_player_package_is_the_raw_head_at_a_fixed_size(player_spec):
    anchors = sum((PLAYER_SIZE // stride) ** 2 for stride in (8, 16, 32))
    assert [(t.name, t.shape) for t in player_spec.inputs] == [("image", (1, 3, PLAYER_SIZE, PLAYER_SIZE))]
    assert [(t.name, t.shape) for t in player_spec.outputs] == [("predictions", (1, 5, anchors))]
    assert not any(isinstance(m, nn.BatchNorm2d) for m in player_spec.fused.modules())
    image = torch.rand(1, 3, PLAYER_SIZE, PLAYER_SIZE)
    traced = export_coreml.trace(player_spec.fused, player_spec.probes())
    with torch.no_grad():
        reference = player_spec.trainable(image)
        assert (traced(image) - reference).abs().max() < 1e-3
    user = export_coreml.package_metadata(player_spec, "beef", "abc1234")["user_defined"]
    assert user["role"] == "detector" and user["nms"].startswith("not in the model")
    assert float(user["operating_threshold"]) == pytest.approx(0.21) and float(user["nms_iou"]) == pytest.approx(0.1)


def test_a_player_checkpoint_of_another_md5_is_refused(player_spec):
    with pytest.raises(ValueError, match="not the checkpoint the shipped player.engine"):
        export_coreml.player_specs(player_spec.weights, PLAYER_SIZE)


def test_player_input_is_the_sessions_stretched_rgb_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 0, 128, 255          # BGR
    image = export_coreml.player_input(frame)
    assert image.shape == (1, 3, 640, 640) and image.dtype == np.float16
    assert np.allclose(image[0, 0], 1.0) and np.allclose(image[0, 1], np.float16(128 / 255)) and np.allclose(image[0, 2], 0)


def test_the_player_cases_pass_on_the_pytorch_graphs(tmp_path, player_spec):
    specs = {player_spec.name: player_spec}
    rng = np.random.default_rng(9)
    players = [(5, rng.random((1, 3, PLAYER_SIZE, PLAYER_SIZE)).astype(np.float16))]
    cases = export_coreml.write_samples(tmp_path, specs, [], players)
    assert [(c["group"], c["kind"], c["frame"]) for c in cases] == [("player_detector", "boxes", 5)]
    traced = {player_spec.name: export_coreml.trace(player_spec.fused, player_spec.probes())}
    for case in cases:        # an untrained head scores little: decode everything it says
        case["threshold"] = 0.0
    scored = validator.evaluate(export_coreml.TorchRunner(specs, traced, "traced_fp32"), {"cases": cases}, tmp_path, 1e-3)
    assert scored["player_detector"]["pass"], scored


def test_decode_head_thresholds_then_suppresses():
    head = np.zeros((1, 5, 4), dtype=np.float32)
    head[0, :, 0] = [100, 100, 20, 40, 0.9]          # a player
    head[0, :, 1] = [101, 101, 20, 40, 0.8]          # the same player, suppressed at IoU 0.1
    head[0, :, 2] = [300, 100, 20, 40, 0.5]          # another player
    head[0, :, 3] = [500, 100, 20, 40, 0.1]          # below the threshold
    rows = validator.decode_head(head, 0.21, 0.1)
    assert rows.shape == (2, 5) and rows[0, 4] == pytest.approx(0.9)
    assert rows[0, :4].tolist() == [90, 80, 110, 120]
    moved = head.copy()
    moved[0, 0, 0] += 2.0
    score = validator.score_boxes(moved, head, 0.21, 0.1)
    assert score.same_count == [True] and score.max_corner_diff == pytest.approx(2.0) and score.min_iou < 1.0


def test_the_conversion_writes_an_ios17_fp16_program(tmp_path, stream_specs):
    ct = pytest.importorskip("coremltools")
    if sys.platform == "win32":
        pytest.skip("coremltools converts on macOS and Linux only")
    spec = stream_specs["stream_s4_encoder"]
    metadata = export_coreml.package_metadata(spec, "feedface", "abc1234")
    mlmodel = export_coreml.convert(spec, export_coreml.trace(spec.fused, spec.probes()), metadata)
    mlmodel.save(str(tmp_path / "encoder.mlpackage"))
    saved = ct.models.MLModel(str(tmp_path / "encoder.mlpackage"), skip_model_load=True)
    proto = saved.get_spec()
    assert proto.specificationVersion == 8                    # iOS 17
    assert proto.WhichOneof("Type") == "mlProgram"
    described = export_coreml.describe_package(proto)
    assert [(t["name"], t["shape"]) for t in described["inputs"]] == [("frame", [1, 3, HEIGHT, WIDTH]),
                                                                    ("previous", [1, 3, HEIGHT, WIDTH])]
    assert [t["name"] for t in described["outputs"]] == ["quarter", "eighth", "bottom"]
    float16 = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
    assert all(t["dtype"] == float16 for t in [*described["inputs"], *described["outputs"]])
    assert saved.user_defined_metadata["source_weights_md5"] == "feedface"
    assert saved.input_description["previous"].startswith("the same phase's frame before it")


def test_the_player_head_converts_with_fixed_fp16_tensors(player_spec):
    pytest.importorskip("coremltools")
    if sys.platform == "win32":
        pytest.skip("coremltools converts on macOS and Linux only")
    metadata = export_coreml.package_metadata(player_spec, "feedface", "abc1234")
    mlmodel = export_coreml.convert(player_spec, export_coreml.trace(player_spec.fused, player_spec.probes()), metadata)
    described = export_coreml.describe_package(mlmodel.get_spec())
    anchors = player_spec.outputs[0].shape[-1]
    assert [(t["name"], t["shape"]) for t in described["inputs"]] == [("image", [1, 3, PLAYER_SIZE, PLAYER_SIZE])]
    assert [(t["name"], t["shape"]) for t in described["outputs"]] == [("predictions", [1, 5, anchors])]
    assert mlmodel.user_defined_metadata["nms"].startswith("not in the model")
