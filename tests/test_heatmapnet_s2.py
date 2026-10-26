"""tennis_tracker/heatmapnet_s2.py: the network's output at stride 2, the contract, the fresh
head's prior, the parent start from a heatmap_s4 record (every shared tensor copied, the new block
and the head fresh, anything else refused) and the .pt record of its own. Tiny inputs on the CPU."""
import math

import pytest
import torch

from tennis_tracker import heatmapnet, heatmapnet_s2
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH
from tennis_tracker.heatmapnet import HEAT_PRIOR, HeatmapNet, read_record, save_model
from tennis_tracker.heatmapnet_s2 import DECODER_CHANNELS_S2, NEW_MODULES, HeatmapNetS2, from_heatmap, load_model
from tennis_tracker.memorynet import MemoryNet

CPU = torch.device("cpu")
SMALL = (96, 160)      # a multiple of the total stride 16: the output is then 48x80


def perturbed_parent() -> HeatmapNet:
    """A HeatmapNet whose every parameter has moved off its initial value, so a copy is told
    from a fresh draw."""
    parent = HeatmapNet()
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.add_(torch.rand_like(parameter) * 0.01)
    return parent


def test_output_is_fifteen_sigmoid_channels_at_stride_two_and_the_contract_is_heatmapnets_but_the_grid():
    model = HeatmapNetS2().eval()
    with torch.inference_mode():
        output = model(torch.rand(2, INPUT_CHANNELS, *SMALL))
        full = model(torch.zeros(1, INPUT_CHANNELS, HEIGHT, WIDTH))
    assert output.shape == (2, FRAMES_PER_UNIT * 3, SMALL[0] // 2, SMALL[1] // 2)
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
    assert full.shape == (1, 15, HeatmapNetS2.GRID_ROWS, HeatmapNetS2.GRID_COLS) == (1, 15, 216, 384)
    assert (HeatmapNetS2.ARCH, HeatmapNetS2.STATEFUL) == ("heatmap_s2", False)
    assert (HeatmapNetS2.INPUT_CHANNELS, HeatmapNetS2.FRAMES_PER_UNIT) == (HeatmapNet.INPUT_CHANNELS, HeatmapNet.FRAMES_PER_UNIT)
    assert (HeatmapNetS2.GRID_COLS, HeatmapNetS2.GRID_ROWS) == (2 * HeatmapNet.GRID_COLS, 2 * HeatmapNet.GRID_ROWS)
    assert heatmapnet_s2.PARENT_ARCH == heatmapnet.ARCH and heatmapnet_s2.STRIDE == 2
    assert model.head.in_channels == DECODER_CHANNELS_S2 and model.up3[0].in_channels == 96 + 48
    with pytest.raises(ValueError, match="HeatmapNetS2 takes inputs whose height and width are multiples of 16"), torch.inference_mode():
        model(torch.zeros(1, INPUT_CHANNELS, 104, 160))


def test_a_fresh_head_predicts_the_heat_prior_and_every_convolution_is_kaiming_with_zero_bias():
    """The constructor resets once the third block and the new head exist: the head's heat biases
    sit at logit(HEAT_PRIOR), the offset channels at 0.5, and the new block's convolutions are
    bias-free like every other block's."""
    model = HeatmapNetS2().eval()
    bias = model.head.bias.detach()
    assert float(bias[0]) == pytest.approx(-math.log((1 - HEAT_PRIOR) / HEAT_PRIOR)) and float(bias[1]) == 0.0 == float(bias[2])
    assert model.up3[0].bias is None and model.up3[3].bias is None
    with torch.no_grad():
        model.head.weight.zero_()
        output = model(torch.zeros(1, INPUT_CHANNELS, *SMALL))
    for frame in range(FRAMES_PER_UNIT):
        assert float(output[0, frame * 3].mean()) == pytest.approx(HEAT_PRIOR, abs=1e-4)
        assert float(output[0, frame * 3 + 1].mean()) == pytest.approx(0.5, abs=1e-4)


def test_a_network_built_from_a_heatmapnet_copies_every_shared_tensor_and_starts_the_last_stage_fresh(tmp_path):
    """from_heatmap copies every tensor but the parent's head (whose in-channels are the stride-4
    features') exactly, running statistics included; up3 and the head are a fresh network's, the
    head at the heat prior; and the result is a float32 training model."""
    parent = perturbed_parent()
    parent.train()
    with torch.no_grad():
        parent(torch.rand(2, INPUT_CHANNELS, *SMALL))             # moves the running statistics
    save_model(parent, tmp_path / "heat.pt")
    model = from_heatmap(tmp_path / "heat.pt", CPU)
    assert isinstance(model, HeatmapNetS2) and model.training and model.head.weight.dtype == torch.float32
    state, parent_state = model.state_dict(), parent.state_dict()
    shared = [name for name in parent_state if not name.startswith("head.")]
    assert shared and all(torch.equal(state[name], parent_state[name]) for name in shared)
    assert state["head.weight"].shape != parent_state["head.weight"].shape
    fresh = {name for name in state if name not in parent_state or name.startswith("head.")}
    assert {name.split(".")[0] for name in fresh} == set(NEW_MODULES)
    assert float(model.head.bias.detach()[0]) == pytest.approx(-math.log((1 - HEAT_PRIOR) / HEAT_PRIOR))
    assert float(model.up3[1].running_var.sum()) == model.up3[1].num_features          # untouched BatchNorm


def test_from_heatmap_refuses_another_architecture_and_a_record_missing_a_shared_layer(tmp_path):
    save_model(MemoryNet(), tmp_path / "memory.pt")
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file \\(arch 'memory_s4'\\)"):
        from_heatmap(tmp_path / "memory.pt", CPU)
    save_model(HeatmapNetS2(), tmp_path / "own.pt")
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file \\(arch 'heatmap_s2'\\)"):
        from_heatmap(tmp_path / "own.pt", CPU)
    state = dict(HeatmapNet().state_dict(), **{"memory.conv_z.weight": torch.zeros(1)})   # heatmap_s4 by name, more inside
    torch.save({"arch": "heatmap_s4", "state_dict": state}, tmp_path / "extra.pt")
    with pytest.raises(ValueError, match="does not hold a HeatmapNet's layers .*unexpected \\['memory.conv_z.weight'"):
        from_heatmap(tmp_path / "extra.pt", CPU)
    state = HeatmapNet().state_dict()
    del state["stem.0.weight"]
    torch.save({"arch": "heatmap_s4", "state_dict": state}, tmp_path / "short.pt")
    with pytest.raises(ValueError, match="does not hold a HeatmapNet's layers \\(missing \\['stem.0.weight'"):
        from_heatmap(tmp_path / "short.pt", CPU)


def test_the_pt_record_names_the_architecture_and_each_loader_takes_only_its_own(tmp_path):
    model = HeatmapNetS2()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.rand_like(parameter) * 0.01)
    destination = tmp_path / "out" / "s2.pt"
    save_model(model, destination)
    assert destination.is_file() and not list((tmp_path / "out").glob("*.tmp.pt"))
    assert read_record(destination)["arch"] == "heatmap_s2"
    loaded = load_model(destination, CPU)
    assert isinstance(loaded, HeatmapNetS2) and not loaded.training and loaded.up3[0].weight.dtype == torch.float16
    assert set(loaded.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        back = loaded.state_dict()[name]
        if tensor.is_floating_point():
            assert torch.allclose(back.float(), tensor.half().float()), name
        else:
            assert torch.equal(back, tensor), name
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file \\(arch 'heatmap_s2'\\)"):
        heatmapnet.load_model(destination, CPU)
    save_model(HeatmapNet(), tmp_path / "heat.pt")
    with pytest.raises(ValueError, match="not a heatmap_s2 weights file \\(arch 'heatmap_s4'\\)"):
        load_model(tmp_path / "heat.pt", CPU)
    with pytest.raises(ValueError, match="not a readable heatmap_s2 weights file"):
        load_model(tmp_path / "missing.pt", CPU)
