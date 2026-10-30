"""The fast ball-detector architectures (tennis_tracker/heatmap_lite.py, streamnet.py, wasbnet.py)
and their RepVGG blocks (repvgg.py): the shared contract and output layout, the .pt round trip
through ball_models, the fused (deployed) form answering as the trainable one, StreamNet's
streaming schedule answering exactly as its unit forward pass, and the trainer's scale hold
finding every convolution-BatchNorm pair. Tiny inputs on the CPU."""
import copy
import math

import pytest
import torch
from torch import nn

from tennis_tracker import ball_models, repvgg, streamnet
from tennis_tracker.ball_models import (HEATMAP_LITE, STREAM_S4, WASB_5F, arch_of, deployed, load_ball_model,
                                         load_deployed, model_class)
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH
from tennis_tracker.heatmapnet import HEAT_PRIOR, HeatmapNet, save_model

CPU = torch.device("cpu")
SMALL = (64, 96)        # a multiple of the total stride 16 (and 32 is not needed: WASB-5f fuses by size)
FAST = (HEATMAP_LITE, STREAM_S4, WASB_5F)


def scrambled(model: nn.Module) -> nn.Module:
    """`model` with every BatchNorm's statistics and affine parameters moved off the identity and
    the head's weights grown, so a wrong fold or a wrong order would show."""
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                size = module.num_features
                module.running_mean.copy_(torch.rand(size, generator=generator) - 0.5)
                module.running_var.copy_(torch.rand(size, generator=generator) * 2 + 0.25)
                module.weight.copy_(torch.rand(size, generator=generator) + 0.5)
                module.bias.copy_(torch.rand(size, generator=generator) - 0.5)
    return model.eval()


@pytest.mark.parametrize("arch", FAST)
def test_each_answers_fifteen_sigmoid_channels_on_the_heatmap_grid(arch):
    cls = model_class(arch)
    assert cls.ARCH == arch and (cls.GRID_COLS, cls.GRID_ROWS) == (HeatmapNet.GRID_COLS, HeatmapNet.GRID_ROWS) == (192, 108)
    assert (cls.INPUT_CHANNELS, cls.FRAMES_PER_UNIT) == (INPUT_CHANNELS, FRAMES_PER_UNIT) == (15, 5)
    model = cls().eval()
    with torch.inference_mode():
        output = model(torch.rand(2, INPUT_CHANNELS, *SMALL))
        full = model(torch.zeros(1, INPUT_CHANNELS, HEIGHT, WIDTH))
    assert output.shape == (2, 15, SMALL[0] // 4, SMALL[1] // 4)
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
    assert full.shape == (1, 15, 108, 192)


@pytest.mark.parametrize("arch", FAST)
def test_a_fresh_head_predicts_the_heat_prior(arch):
    model = model_class(arch)().eval()
    with torch.no_grad():
        model.head.weight.zero_()
        output = model(torch.rand(1, INPUT_CHANNELS, *SMALL))
    for frame in range(FRAMES_PER_UNIT):
        assert float(output[0, frame * 3].mean()) == pytest.approx(HEAT_PRIOR, abs=1e-5)
        assert float(output[0, frame * 3 + 2].mean()) == pytest.approx(0.5, abs=1e-5)
    assert float(model.head.bias.detach()[0]) == pytest.approx(-math.log((1 - HEAT_PRIOR) / HEAT_PRIOR))


@pytest.mark.parametrize("arch", FAST)
def test_the_record_round_trips_through_ball_models_in_the_trainable_form(arch, tmp_path):
    model = model_class(arch)()
    model.train()
    with torch.no_grad():
        model(torch.rand(2, INPUT_CHANNELS, *SMALL))           # moves the running statistics
    save_model(model, tmp_path / "w.pt")
    assert arch_of(tmp_path / "w.pt") == arch and ball_models.weights_suffix(arch) == ".pt"
    loaded = load_ball_model(tmp_path / "w.pt", CPU)
    assert type(loaded) is model_class(arch) and not loaded.training and loaded.head.weight.dtype == torch.float16
    assert set(loaded.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            assert torch.allclose(loaded.state_dict()[name].float(), tensor.half().float()), name
    save_model(HeatmapNet(), tmp_path / "heat.pt")
    with pytest.raises(ValueError, match=f"not a {arch} weights file"):
        ball_models._MODULES[arch].load_model(tmp_path / "heat.pt", CPU)


@pytest.mark.parametrize("arch", FAST)
def test_the_deployed_form_answers_as_the_trainable_one(arch):
    """Fusing RepVGG branches and folding BatchNorm is exact algebra: in float64 the two forms
    agree to 1e-9 on scrambled statistics, and the deployed form has no BatchNorm and no branch
    left."""
    model = scrambled(model_class(arch)()).double()
    value = torch.rand(2, INPUT_CHANNELS, *SMALL, dtype=torch.float64)
    with torch.no_grad():
        expected = model(value)
        fused = deployed(copy.deepcopy(model))
        answer = fused(value)
    assert float((answer - expected).abs().max()) < 1e-9
    assert not any(isinstance(module, nn.BatchNorm2d) for module in fused.modules())
    assert all(block.fused is not None for block in fused.modules() if isinstance(block, repvgg.RepVGGBlock))
    assert deployed(fused) is fused                                        # idempotent
    heat = HeatmapNet()
    assert deployed(heat) is heat                                          # nothing to fuse: unchanged


def test_a_repvgg_block_fuses_each_branch_shape():
    for in_channels, out_channels, stride in ((4, 4, 1), (4, 6, 1), (4, 6, 2)):
        block = scrambled(repvgg.RepVGGBlock(in_channels, out_channels, stride)).double()
        assert (block.identity is not None) == (in_channels == out_channels and stride == 1)
        value = torch.randn(1, in_channels, 10, 12, dtype=torch.float64)
        with torch.no_grad():
            expected = block(value)
            block.reparameterize()
            assert float((block(value) - expected).abs().max()) < 1e-12
        assert block.dense is None and block.fused is not None and block.fused.stride == (stride, stride)


def test_streaming_answers_exactly_as_the_unit_forward_pass():
    """Encoding each transition of a phase once and decoding every unit from the last four cached
    features is the same computation as the forward pass on each stacked five-frame unit: equal
    to float32 rounding, in the trainable and the deployed form."""
    torch.manual_seed(0)
    model = scrambled(streamnet.StreamNet())
    frames = [torch.rand(2, 3, *SMALL) for _ in range(8)]
    for network in (model, deployed(copy.deepcopy(model))):
        with torch.no_grad():
            streamed = streamnet.stream_units(network, frames)
            units = [network(torch.cat(frames[start:start + FRAMES_PER_UNIT], dim=1)) for start in range(len(frames) - 4)]
        assert len(streamed) == len(units) == 4
        for stream_output, unit_output in zip(streamed, units):
            assert stream_output.shape == unit_output.shape == (2, 15, SMALL[0] // 4, SMALL[1] // 4)
            assert float((stream_output - unit_output).abs().max()) < 1e-6


def test_a_transition_keeps_the_previous_frame_recoverable_and_decode_wants_four():
    frame, previous = torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 4)
    value = streamnet.transition(frame, previous)
    assert value.shape == (1, streamnet.TRANSITION_CHANNELS, 4, 4)
    assert torch.allclose(value[:, :3] - value[:, 3:6] + value[:, 6:], previous)
    assert float(value[:, 3:].min()) >= 0.0
    model = streamnet.StreamNet().eval()
    features = model.encode_frame(torch.rand(1, 3, *SMALL), torch.rand(1, 3, *SMALL))
    with pytest.raises(ValueError, match="from 4 transitions, not 3"):
        model.decode([features] * 3)
    with pytest.raises(ValueError, match="multiples of 16"):
        model.encode_frame(torch.rand(1, 3, 40, 40), torch.rand(1, 3, 40, 40))


def test_a_unit_frame_changes_only_through_its_transitions():
    """Frame 0 of a unit reaches the network only through the first transition, so changing it
    leaves the cached features of transitions two to four untouched: what makes the cache valid."""
    model = streamnet.StreamNet().eval()
    frames = [torch.rand(1, 3, *SMALL) for _ in range(5)]
    changed = [torch.rand(1, 3, *SMALL)] + frames[1:]
    with torch.no_grad():
        before = [model.encode_frame(frames[index], frames[index - 1]) for index in range(1, 5)]
        after = [model.encode_frame(changed[index], changed[index - 1]) for index in range(1, 5)]
    assert not torch.equal(before[0][0], after[0][0])
    for first, second in zip(before[1:], after[1:]):
        assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("arch", FAST)
def test_every_batch_norm_follows_a_bias_free_convolution_in_one_sequential(arch):
    """The trainer holds the convolutions BatchNorm follows at their Kaiming norm by finding
    conv-BatchNorm pairs inside an ``nn.Sequential`` (``loop.normalised_conv_weights``); a
    BatchNorm anywhere else would escape the hold. The only exception is the RepVGG identity
    branch, a BatchNorm with no convolution."""
    model = model_class(arch)()
    paired = set()
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            for conv, norm in zip(module, list(module)[1:]):
                if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                    assert conv.bias is None
                    paired.add(id(norm))
    identities = {id(block.identity) for block in model.modules()
                  if isinstance(block, repvgg.RepVGGBlock) and block.identity is not None}
    norms = {id(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)}
    assert norms == paired | identities and paired


def test_wasb_fuses_branches_of_sizes_that_do_not_halve_evenly():
    """At 432 rows the coarsest branch is 14 rows (27 rounded up), and fusion upsamples it to the
    finer branch's exact size, so the full input works; the parameter count is WASB's ~1.5M."""
    model = model_class(WASB_5F)().eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, INPUT_CHANNELS, 432, 64)).shape == (1, 15, 108, 16)
    assert 1.4e6 < sum(parameter.numel() for parameter in model.parameters()) < 1.6e6
    with pytest.raises(ValueError, match="multiples of 4"):
        model(torch.zeros(1, INPUT_CHANNELS, 30, 64))


@pytest.mark.parametrize("arch", FAST)
def test_a_fresh_network_answers_the_heat_prior_with_its_real_features(arch):
    """``repvgg.reset_head`` gives the head CenterNet's small weights, so the prior holds on a
    fresh network's own features too, not only with the head zeroed: Kaiming weights let WASB-5f's
    summed branches push the mean heat to 0.30 (measured 2026-10-27)."""
    torch.manual_seed(0)
    model = model_class(arch)().train()
    with torch.no_grad():
        output = model(torch.rand(2, INPUT_CHANNELS, *SMALL))
    heat = output[:, 0::3]
    assert float(heat.mean()) == pytest.approx(HEAT_PRIOR, abs=0.005) and float(heat.std()) < 0.01
    assert float(model.head.weight.detach().std()) == pytest.approx(repvgg.HEAD_STD, rel=0.2)


@pytest.mark.parametrize("arch", FAST)
def test_the_deployed_load_folds_the_float32_record_and_rounds_to_fp16_once(arch, tmp_path):
    """``load_deployed`` folds the record's float32 weights and casts the result to FP16 once:
    exactly the float32 fold rounded, which folding the FP16 model ``load_ball_model`` gives
    (rounded twice) is not."""
    torch.manual_seed(0)
    weights = tmp_path / f"{arch}.pt"
    save_model(scrambled(model_class(arch)()), weights)
    loaded = load_deployed(weights, CPU)
    reference = deployed(copy.deepcopy(load_ball_model(weights, CPU)).float())
    record = model_class(arch)()
    record.load_state_dict(torch.load(weights, weights_only=True)["state_dict"])
    once = deployed(record.eval()).half()
    assert not any(isinstance(module, nn.BatchNorm2d) for module in loaded.modules())
    pairs = list(zip(loaded.state_dict().values(), once.state_dict().values()))
    assert pairs and all(a.dtype == torch.float16 and torch.equal(a, b) for a, b in pairs)
    twice = [a for a, b in zip(loaded.state_dict().values(), reference.half().state_dict().values()) if not torch.equal(a, b)]
    assert twice                                                # the FP16 model's fold rounds differently
