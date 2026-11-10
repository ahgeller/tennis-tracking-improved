"""tools/export_coreml.py and the Mac kit's validator: the packages' modules, names, shapes and
metadata, the RepVGG fuse against the trainable network, the cases the Mac runs (scored here on the
PyTorch graphs), the tracker logic the kit copies (player preprocessing, box decoding, the grid
layout) against the tracker's own, and the CLI's refusals. The conversion itself is checked where
coremltools installs (macOS, Linux)."""
import dataclasses
import functools
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from tennis_tracker import gridtracknet, heatmapnet, streamnet
from tennis_tracker.config import Config

_PATH = Path(__file__).resolve().parent.parent / "tools" / "export_coreml.py"
_SPEC = importlib.util.spec_from_file_location("export_coreml", _PATH)
export_coreml = importlib.util.module_from_spec(_SPEC)
sys.modules["export_coreml"] = export_coreml
_SPEC.loader.exec_module(export_coreml)
validator = export_coreml.validator

HEIGHT, WIDTH = 64, 128          # a small input both networks take (multiples of 16)
PLAYER_SIZE = 128                # a small player input (a multiple of the head's stride 32)
CFG = Config()


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


def _record(tmp_path: Path, make: Callable[[], nn.Module], seed: int) -> Path:
    """A record of a network made and its batch norms randomised under `seed`."""
    torch.manual_seed(seed)
    model = make()
    _randomise_batch_norms(model, seed)
    path = tmp_path / f"{model.ARCH}.pt"
    heatmapnet.save_model(model, path)
    return path


@pytest.fixture(scope="module")
def stream_weights(tmp_path_factory):
    return _record(tmp_path_factory.mktemp("stream"), streamnet.StreamNet, 3)


@pytest.fixture(scope="module")
def stream_specs(stream_weights):
    return {spec.name: spec for spec in export_coreml.stream_specs(stream_weights, HEIGHT, WIDTH)}


@pytest.fixture(scope="module")
def heatmap_spec(tmp_path_factory):
    weights = _record(tmp_path_factory.mktemp("heat"), heatmapnet.HeatmapNet, 5)
    (spec,) = export_coreml.heatmap_specs(weights, HEIGHT, WIDTH)
    return spec


@pytest.fixture(scope="module")
def player_weights(tmp_path_factory):
    tasks = pytest.importorskip("ultralytics.nn.tasks")
    torch.manual_seed(7)
    model = tasks.DetectionModel("yolo11n.yaml", nc=1, verbose=False).eval()
    _randomise_batch_norms(model, 7)
    weights = tmp_path_factory.mktemp("player") / "player.pt"
    torch.save({"model": model, "train_args": {}}, weights)
    return weights


@pytest.fixture(scope="module")
def player_spec(player_weights):
    (spec,) = export_coreml.player_specs(player_weights, PLAYER_SIZE)
    return spec


def _unit(seed: int) -> np.ndarray:
    """A unit of random 8-bit frames as the packages take it: uint8 / 255 in float16."""
    pixels = np.random.default_rng(seed).integers(0, 256, (1, 15, HEIGHT, WIDTH)).astype(np.float32)
    return (pixels / 255.0).astype(np.float16)


def _picture(seed: int, height: int = 1080, width: int = 1920) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_the_packages_names_and_shapes(stream_specs, heatmap_spec):
    encoder, decoder, unit = (stream_specs[n] for n in ("stream_s4_encoder", "stream_s4_decoder", "stream_s4_unit"))
    assert [t.name for t in encoder.inputs] == ["frame", "previous"]
    assert all(t.shape == (1, 3, HEIGHT, WIDTH) for t in encoder.inputs)
    expected = [(level, (1, channels, HEIGHT // stride, WIDTH // stride)) for level, channels, stride
                in zip(export_coreml.FEATURE_NAMES, streamnet.ENCODER_CHANNELS[1:], (4, 8, 16))]
    assert [(t.name, t.shape) for t in encoder.outputs] == expected
    assert [t.name for t in decoder.inputs] == list(export_coreml.DECODER_INPUTS)
    assert decoder.inputs[0].name == "t1_quarter"
    assert decoder.inputs[-1].name == "t4_bottom"
    assert [t.shape for t in decoder.inputs[:3]] == [t.shape for t in encoder.outputs]
    for spec in (decoder, unit, heatmap_spec):
        assert [(t.name, t.shape) for t in spec.outputs] == [("grids", (1, 15, HEIGHT // 4, WIDTH // 4))]
    for spec in (unit, heatmap_spec):
        assert [(t.name, t.shape) for t in spec.inputs] == [("frames", (1, 15, HEIGHT, WIDTH))]
    assert all(t.description for spec in [*stream_specs.values(), heatmap_spec] for t in [*spec.inputs, *spec.outputs])
    assert 0 < encoder.gmac < unit.gmac < heatmap_spec.gmac


def test_the_descriptions_follow_the_shapes(stream_specs):
    encoder, decoder = stream_specs["stream_s4_encoder"], stream_specs["stream_s4_decoder"]
    assert f"{WIDTH}x{HEIGHT}" in encoder.inputs[0].description
    assert "stride 8, 64 channels" in encoder.outputs[1].description
    assert f"three {HEIGHT // 4}x{WIDTH // 4} planes" in decoder.outputs[0].description
    assert "HeatRing" in decoder.description
    assert "HeatRing" in stream_specs["stream_s4_unit"].description


def test_the_specs_are_frozen(stream_specs):
    with pytest.raises(AttributeError):
        stream_specs["stream_s4_unit"].threshold = 0.0


def test_the_tensorrt_halves_use_the_same_names():
    ball_trt = pytest.importorskip("tennis_tracker.engines.ball_trt")
    assert export_coreml.FRAME_INPUTS == ball_trt.ENCODER_INPUTS
    assert export_coreml.FEATURE_NAMES == ball_trt.FEATURE_NAMES
    assert (export_coreml.UNIT_INPUT, export_coreml.GRIDS) == (ball_trt.INPUT_NAME, ball_trt.OUTPUT_NAME)


def test_transition_channels_pick_a_frame_and_the_one_before():
    assert export_coreml.transition_channels(1) == ((3, 6), (0, 3))
    assert export_coreml.transition_channels(4) == ((12, 15), (9, 12))


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
        for pair in range(1, export_coreml.PAIRS + 1):
            (f0, f1), (p0, p1) = export_coreml.transition_channels(pair)
            features += list(encoder(frames[:, f0:f1], frames[:, p0:p1]))
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
    assert metadata["version"] == "abc1234"
    assert "decoder" in metadata["short_description"]
    assert set(metadata["inputs"]) == set(export_coreml.DECODER_INPUTS)
    assert set(metadata["outputs"]) == {"grids"}
    user = metadata["user_defined"]
    assert user["source_weights_md5"] == "0123abcd"
    assert user["arch"] == "stream_s4"
    assert user["role"] == "decoder"
    assert user["source_weights"] == spec.weights.name
    assert float(user["operating_threshold"]) > 0
    assert all(isinstance(value, str) for value in user.values())
    assert user["fused"].startswith("RepVGG branches and BatchNorm folded")
    assert "nms_iou" not in user


def test_heatmap_metadata_says_coremltools_folds_its_batch_norms(heatmap_spec):
    user = export_coreml.package_metadata(heatmap_spec, "0123abcd", "abc1234")["user_defined"]
    assert user["fused"].startswith("no RepVGG blocks")
    assert user["arch"] == "heatmap_s4"


def test_the_mac_cases_pass_on_the_pytorch_graphs(tmp_path, stream_specs, heatmap_spec):
    specs = {**stream_specs, heatmap_spec.name: heatmap_spec}
    samples = export_coreml.Samples([([0, 2, 4, 6, 8], _unit(3)), ([1, 3, 5, 7, 9], _unit(4))], {})
    cases = export_coreml.write_samples(tmp_path, specs, samples, ["stream_s4", "heatmap_s4"])
    groups = {case["group"] for case in cases}
    assert groups == {"stream_s4_unit", "stream_s4_encoder", "stream_s4_decoder", "stream_s4_chain", "heatmap_s4_unit"}
    assert sum(case["group"] == "stream_s4_encoder" for case in cases) == 2 * 4
    assert np.load(tmp_path / "samples/unit0/stream_s4_features/t4_bottom.npy").dtype == np.float16
    manifest = {"cases": cases}
    traced = {name: export_coreml.trace(spec.fused, spec.probes()) for name, spec in specs.items()}
    tolerances = validator.Tolerances(heat=1e-3)
    for variant in ("fused_fp32", "traced_fp32"):
        scored = validator.evaluate(export_coreml.TorchRunner(specs, traced, variant), manifest, tmp_path, tolerances)
        assert set(scored) == groups
        assert all(result["pass"] for result in scored.values()), scored
        assert scored["stream_s4_chain"]["summary"]["argmax_within_1"].endswith("/10")      # both units' frames


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
    assert score.exact[:3] == [True, False, False]
    assert score.near[:3] == [True, True, False]
    assert score.confident == [True, True, True, False, False]
    assert score.max_offset_diff == pytest.approx(0.25)
    assert score.max_heat_diff == pytest.approx(0.95)
    assert score.summary()["confident_argmax_exact"] == "1/3"


def test_a_wrong_offset_on_a_confident_frame_fails():
    reference = np.zeros((1, 15, 6, 8), dtype=np.float32)
    reference[0, 0, 2, 3] = 0.9                                         # frame 0 confident
    output = reference.copy()
    output[0, 13, 0, 0] = 0.5                                           # frame 4's offset: not confident
    tolerances = validator.Tolerances()
    ok, summary = validator.judge_grids([validator.score_grids(output, reference, 0.5)], tolerances)
    assert ok
    assert summary["max_offset_diff"] == pytest.approx(0.5)
    output[0, 2, 1, 1] = 0.2                                            # frame 0's y offset
    ok, summary = validator.judge_grids([validator.score_grids(output, reference, 0.5)], tolerances)
    assert not ok
    assert summary["confident_max_offset_diff"] == pytest.approx(0.2)


def test_the_feature_check_gates_each_levels_worst_relative_difference():
    small = {"quarter": {"max_diff": 0.01, "max_diff_relative": 0.2}}   # small absolute, large relative
    large = {"quarter": {"max_diff": 0.5, "max_diff_relative": 0.01}}
    ok, worst = validator.judge_features([large, small], validator.Tolerances())
    assert not ok
    assert worst["quarter"] == {"max_diff": 0.5, "max_diff_relative": 0.2}


def test_evaluate_refuses_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="case kinds"):
        validator.evaluate(None, {"cases": [{"group": "g", "kind": "nope"}]}, tmp_path)


def test_the_validators_grid_layout_is_the_trackers():
    assert validator.FRAMES_PER_UNIT == gridtracknet.FRAMES_PER_UNIT
    assert validator.FRAMES_PER_UNIT * export_coreml.RGB == gridtracknet.INPUT_CHANNELS
    try:
        from tennis_tracker.engines.ball_gpu import HeatRing
    except ImportError as error:          # pragma: no cover - the tracker's GPU module needs its kernels
        pytest.skip(f"ball_gpu does not import here: {error}")
    assert HeatRing(1, 1, 1, torch.device("cpu")).sums.shape[1] == validator.GRID_PLANES


def test_load_input_slices_a_units_frame(tmp_path):
    frames = np.arange(15 * 2 * 2, dtype=np.float32).reshape(1, 15, 2, 2)
    np.save(tmp_path / "f.npy", frames)
    picked = validator.load_input({"npy": "f.npy", "channels": [3, 6]}, tmp_path)
    assert picked.dtype == np.float16
    assert picked.shape == (1, 3, 2, 2)
    assert picked.flags["C_CONTIGUOUS"]
    assert np.array_equal(picked, frames[:, 3:6].astype(np.float16))


def test_the_kit_names_every_file_the_export_copies():
    for name in export_coreml.KIT_FILES:
        assert (export_coreml.KIT / name).is_file(), name
    assert b"\r\n" not in (export_coreml.KIT / "validate.sh").read_bytes()


def test_the_player_package_is_the_raw_head_at_a_fixed_size(player_spec):
    anchors = sum((PLAYER_SIZE // stride) ** 2 for stride in (8, 16, 32))
    assert [(t.name, t.shape) for t in player_spec.inputs] == [("image", (1, 3, PLAYER_SIZE, PLAYER_SIZE))]
    assert [(t.name, t.shape) for t in player_spec.outputs] == [("predictions", (1, 5, anchors))]
    assert f"{anchors} anchors" in player_spec.outputs[0].description
    assert not any(isinstance(m, nn.BatchNorm2d) for m in player_spec.fused.modules())
    image = torch.rand(1, 3, PLAYER_SIZE, PLAYER_SIZE)
    traced = export_coreml.trace(player_spec.fused, player_spec.probes())
    with torch.no_grad():
        assert (traced(image) - player_spec.trainable(image)).abs().max() < 1e-3
    assert player_spec.threshold == pytest.approx(CFG.player_conf)
    assert player_spec.nms_iou == pytest.approx(CFG.player_iou)
    user = export_coreml.package_metadata(player_spec, "beef", "abc1234")["user_defined"]
    assert user["role"] == "detector"
    assert user["nms"].startswith("not in the model")
    assert float(user["operating_threshold"]) == pytest.approx(CFG.player_conf)
    assert float(user["nms_iou"]) == pytest.approx(CFG.player_iou)


def test_a_player_checkpoint_of_another_md5_is_refused_unless_allowed(player_weights):
    entry = export_coreml.EXPORTS["player"]
    with pytest.raises(export_coreml.SourceError, match="--allow-unmatched-player"):
        export_coreml.resolve_source(entry, player_weights)
    source = export_coreml.resolve_source(entry, player_weights, allow_unmatched=True)
    assert source.matched is False
    assert source.md5 == export_coreml.file_md5(player_weights)


def test_player_input_is_the_sessions_stretched_rgb_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 0, 128, 255          # BGR
    image = export_coreml.player_input(frame)
    assert image.shape == (1, 3, 640, 640)
    assert image.dtype == np.float16
    assert np.allclose(image[0, 0], 1.0)
    assert np.allclose(image[0, 1], np.float16(128 / 255))
    assert np.allclose(image[0, 2], 0)


def test_player_input_is_bit_equal_to_the_sessions_preprocess_frame():
    session = pytest.importorskip("tennis_tracker.engines.engine_session").TensorRTRuntimeSession
    stand_in = types.SimpleNamespace(device=torch.device("cpu"), input_h=640, input_w=640, fp16=True)
    stand_in._finish_preprocess = functools.partial(session._finish_preprocess, stand_in)
    frame = _picture(11)
    tensor, scale = session.preprocess_frame(stand_in, frame)
    ours = export_coreml.player_input(frame)
    assert tensor.dtype == torch.float16
    assert np.array_equal(tensor.numpy().view(np.uint16), ours.view(np.uint16))
    assert scale == (1920 / 640, 1080 / 640)


def test_decode_head_matches_the_trackers_engine_boxes():
    engine_boxes = pytest.importorskip("tennis_tracker.engines.engine_boxes")
    rng = np.random.default_rng(13)
    for trial in range(20):
        anchors = 400
        head = np.empty((1, 5, anchors), dtype=np.float16 if trial % 2 else np.float32)
        centres = rng.uniform(0, 640, (2, anchors))
        sizes = rng.uniform(5, 120, (2, anchors))
        head[0, :2], head[0, 2:4] = centres, sizes
        head[0, 4] = rng.uniform(0, 0.4, anchors)                      # about half above 0.21
        ours = validator.decode_head(head, CFG.player_conf, CFG.player_iou)
        theirs = engine_boxes._raw_head_rows(head, CFG.player_conf, CFG.player_iou)
        assert ours.shape[0] == theirs.shape[0], trial
        assert np.array_equal(ours, theirs[:, :5].astype(np.float32)), trial


def test_decode_head_thresholds_then_suppresses():
    head = np.zeros((1, 5, 4), dtype=np.float32)
    head[0, :, 0] = [100, 100, 20, 40, 0.9]          # a player
    head[0, :, 1] = [101, 101, 20, 40, 0.8]          # the same player, suppressed at IoU 0.1
    head[0, :, 2] = [300, 100, 20, 40, 0.5]          # another player
    head[0, :, 3] = [500, 100, 20, 40, 0.1]          # below the threshold
    rows = validator.decode_head(head, CFG.player_conf, CFG.player_iou)
    assert rows.shape == (2, 5)
    assert rows[0, 4] == pytest.approx(0.9)
    assert rows[0, :4].tolist() == [90, 80, 110, 120]
    moved = head.copy()
    moved[0, 0, 0] += 2.0
    score = validator.score_boxes(moved, head, CFG.player_conf, CFG.player_iou)
    assert score.same_count == [True]
    assert score.max_corner_diff == pytest.approx(2.0)
    assert score.min_iou < 1.0


def test_the_player_cases_pass_on_the_pytorch_graphs(tmp_path, player_spec):
    specs = {player_spec.name: player_spec}
    samples = export_coreml.Samples([([5, 7, 9, 11, 13], np.zeros(1))], {5: _picture(9, 72, 128)})
    cases = export_coreml.write_samples(tmp_path, specs, samples, ["player"])
    assert [(c["group"], c["kind"], c["frame"]) for c in cases] == [("player_detector", "boxes", 5)]
    assert np.load(tmp_path / cases[0]["inputs"]["image"]["npy"]).shape == (1, 3, PLAYER_SIZE, PLAYER_SIZE)
    traced = {player_spec.name: export_coreml.trace(player_spec.fused, player_spec.probes())}
    for case in cases:        # an untrained head scores little: decode everything it says
        case["threshold"] = 0.0
    scored = validator.evaluate(export_coreml.TorchRunner(specs, traced, "traced_fp32"), {"cases": cases}, tmp_path)
    assert scored["player_detector"]["pass"], scored


def _video(path: Path, frames: int, fps: float = 30.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (WIDTH, HEIGHT))
    for k in range(frames):
        writer.write(_picture(100 + k, HEIGHT, WIDTH))
    writer.release()
    return path


def test_a_samples_only_export_writes_what_the_mac_validator_reads(tmp_path, stream_weights, monkeypatch):
    # At the small test size: the stream_s4 packages, and frames as RGB / 255 without the resize.
    small = dataclasses.replace(export_coreml.EXPORTS["stream_s4"],
                                build=lambda weights: export_coreml.stream_specs(weights, HEIGHT, WIDTH))
    monkeypatch.setitem(export_coreml.EXPORTS, "stream_s4", small)
    monkeypatch.setattr(export_coreml, "frame_tensor",
                        lambda frame, device: torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1).float() / 255)
    video = _video(tmp_path / "clip.avi", 6)
    out = tmp_path / "out"
    out.mkdir()
    sources = {"stream_s4": export_coreml.resolve_source(export_coreml.EXPORTS["stream_s4"], stream_weights)}
    manifest = export_coreml.export(out, sources, video, [0], convert_packages=False)
    saved = json.loads((out / "manifest.json").read_text())
    assert saved == json.loads(json.dumps(manifest))
    for key in ("exported", "source_commit", "coremltools", "packages", "cases", "pc_validation"):
        assert key in saved, key
    assert set(saved["packages"]) == {"stream_s4_encoder", "stream_s4_decoder", "stream_s4_unit"}
    for package in saved["packages"].values():
        assert package["file"].startswith("models/")
        assert package["source_weights_md5"] == sources["stream_s4"].md5
        assert all({"name", "shape"} <= set(t) for t in package["inputs"])
    assert saved["units"] == [{"unit": 0, "source_frames": [0, 1, 2, 3, 4]}]
    for case in saved["cases"]:
        assert case["kind"] in validator.KINDS
        for spec in case.get("inputs", {}).values():
            assert (out / spec["npy"]).is_file()
    for name in export_coreml.KIT_FILES:
        assert (out / name).is_file()
    assert export_coreml.pc_validation_passed(saved), saved["pc_validation"]


def test_a_missing_checkpoint_leaves_the_output_alone(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("old export")
    code = export_coreml.main(["--out", str(out), "--overwrite", "--models", "player",
                               "--player-weights", str(tmp_path / "missing.pt")])
    assert code == 2
    assert (out / "keep.txt").read_text() == "old export"
    message = capsys.readouterr().err.strip()
    assert len(message.splitlines()) == 1
    assert export_coreml.RECORD in message
    assert "--models" in message


def test_main_exits_1_when_a_pc_check_fails(tmp_path, stream_weights, monkeypatch):
    failing = {"pc_validation": {"traced_fp16": {"stream_s4_unit": {"pass": False, "summary": {}}}}}
    monkeypatch.setattr(export_coreml, "export", lambda *args, **kwargs: failing)
    code = export_coreml.main(["--out", str(tmp_path / "out"), "--models", "stream_s4",
                               "--stream-weights", str(stream_weights)])
    assert code == 1


def test_the_conversion_writes_an_ios17_fp16_program(tmp_path, stream_specs):
    ct = pytest.importorskip("coremltools")
    spec = stream_specs["stream_s4_encoder"]
    metadata = export_coreml.package_metadata(spec, "feedface", "abc1234")
    mlmodel = export_coreml.convert(spec, export_coreml.trace(spec.fused, spec.probes()), metadata)
    mlmodel.save(str(tmp_path / "encoder.mlpackage"))
    saved = ct.models.MLModel(str(tmp_path / "encoder.mlpackage"), skip_model_load=True)
    proto = saved.get_spec()
    assert proto.specificationVersion == 8                    # iOS 17
    assert proto.WhichOneof("Type") == "mlProgram"
    described = export_coreml.describe_package(proto)
    frame = [1, 3, HEIGHT, WIDTH]
    assert [(t["name"], t["shape"]) for t in described["inputs"]] == [("frame", frame), ("previous", frame)]
    assert [t["name"] for t in described["outputs"]] == ["quarter", "eighth", "bottom"]
    assert all(t["dtype"] == "FLOAT16" for t in [*described["inputs"], *described["outputs"]])
    assert saved.user_defined_metadata["source_weights_md5"] == "feedface"
    assert saved.input_description["previous"].startswith("the same phase's frame before it")


def test_the_player_head_converts_with_fixed_fp16_tensors(player_spec):
    pytest.importorskip("coremltools")
    metadata = export_coreml.package_metadata(player_spec, "feedface", "abc1234")
    mlmodel = export_coreml.convert(player_spec, export_coreml.trace(player_spec.fused, player_spec.probes()), metadata)
    described = export_coreml.describe_package(mlmodel.get_spec())
    anchors = player_spec.outputs[0].shape[-1]
    assert [(t["name"], t["shape"]) for t in described["inputs"]] == [("image", [1, 3, PLAYER_SIZE, PLAYER_SIZE])]
    assert [(t["name"], t["shape"]) for t in described["outputs"]] == [("predictions", [1, 5, anchors])]
    assert all(t["dtype"] == "FLOAT16" for t in [*described["inputs"], *described["outputs"]])
    assert mlmodel.user_defined_metadata["nms"].startswith("not in the model")
