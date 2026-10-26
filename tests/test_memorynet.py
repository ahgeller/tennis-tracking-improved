"""tennis_tracker/memorynet.py: the output and state shapes, the identity with the parent HeatmapNet
at initialisation, a memory that has learned something changing the answer, float32 inside the cell
under FP16, and the .pt record of its own and of its parent. Tiny inputs on the CPU."""
import pytest
import torch

from tennis_tracker import heatmapnet, memorynet
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, INPUT_CHANNELS
from tennis_tracker.heatmapnet import HeatmapNet, read_record, save_model
from tennis_tracker.memorynet import STATE_CHANNELS, MemoryNet, from_heatmap, load_model

CPU = torch.device("cpu")
SMALL = (96, 160)      # a multiple of the total stride 16: the state is then 6x10


def test_output_and_state_shapes_and_the_contract():
    model = MemoryNet().eval()
    with torch.inference_mode():
        output, state = model(torch.rand(2, INPUT_CHANNELS, *SMALL))
        again, next_state = model(torch.rand(2, INPUT_CHANNELS, *SMALL), state)
    assert output.shape == again.shape == (2, FRAMES_PER_UNIT * 3, SMALL[0] // 4, SMALL[1] // 4)
    assert state.shape == next_state.shape == (2, STATE_CHANNELS, SMALL[0] // 16, SMALL[1] // 16) == (2, 256, 6, 10)
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0 and torch.isfinite(next_state).all()
    assert (MemoryNet.ARCH, MemoryNet.STATEFUL, HeatmapNet.STATEFUL) == ("memory_s4", True, False)
    assert (MemoryNet.GRID_COLS, MemoryNet.GRID_ROWS, MemoryNet.INPUT_CHANNELS) == (192, 108, 15)
    assert memorynet.PARENT_ARCH == heatmapnet.ARCH
    with pytest.raises(ValueError, match="MemoryNet takes inputs whose height and width are multiples of 16"), torch.inference_mode():
        model(torch.zeros(1, INPUT_CHANNELS, 108, 192))


def test_a_network_built_from_a_heatmapnet_answers_exactly_as_it_does_until_the_memory_learns(tmp_path):
    """from_heatmap copies every shared layer and leaves the memory fresh with ``out`` at zero, so
    the decoder sees the parent's bottleneck untouched: same input, same output, with or without
    a state, whatever the memory cell computes."""
    parent = HeatmapNet()
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.add_(torch.rand_like(parameter) * 0.01)
    save_model(parent, tmp_path / "heat.pt")
    model = from_heatmap(tmp_path / "heat.pt", CPU)
    assert isinstance(model, MemoryNet) and model.training and model.head.weight.dtype == torch.float32
    assert float(model.out.weight.detach().abs().sum()) == 0.0 == float(model.out.bias.detach().abs().sum())
    for name, tensor in parent.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name
    parent.eval()
    model.eval()
    probe = torch.rand(2, INPUT_CHANNELS, *SMALL)
    with torch.inference_mode():
        expected = parent(probe)
        output, state = model(probe)
        with_state, _ = model(probe, torch.randn_like(state))
    assert torch.equal(output, expected) and torch.equal(with_state, expected)
    torch.manual_seed(1)                              # a Kaiming parent whose CPU float16 activations stay finite
    save_model(HeatmapNet(), tmp_path / "kaiming.pt")
    parent, model = heatmapnet.load_model(tmp_path / "kaiming.pt", CPU), from_heatmap(tmp_path / "kaiming.pt", CPU).eval().half()
    with torch.inference_mode():
        expected = parent(probe.half())
        output, state = model(probe.half())
        again, _ = model(probe.half(), state)
    assert torch.isfinite(expected).all() and torch.equal(output, expected) and torch.equal(again, expected)


def test_the_state_changes_the_output_once_the_memory_cell_has_taken_a_gradient_step():
    """One optimisation step on the memory cell and ``out`` alone moves ``out`` off zero; from then
    on the same five frames answer differently depending on the state carried in, and the memory
    is what the loss reaches (the gradient at ``out`` is not zero)."""
    model = MemoryNet()
    probe = torch.rand(1, INPUT_CHANNELS, *SMALL)
    optimizer = torch.optim.SGD([*model.memory.parameters(), *model.out.parameters()], lr=1.0)
    output, state = model(probe, torch.randn(1, STATE_CHANNELS, SMALL[0] // 16, SMALL[1] // 16))
    output.mean().backward()
    assert model.out.weight.grad is not None and float(model.out.weight.grad.abs().sum()) > 0.0
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        without, _ = model(probe)
        with_state, _ = model(probe, torch.randn_like(state))
    assert float(model.out.weight.detach().abs().sum()) > 0.0 and not torch.equal(without, with_state)


def test_the_cell_computes_in_float32_under_fp16_and_the_state_comes_back_in_the_inputs_dtype():
    model = MemoryNet().eval().half()
    probe = torch.rand(1, INPUT_CHANNELS, *SMALL).half()
    seen = {}
    forward = model.memory.forward

    def spy(features, state):
        seen["dtypes"] = (features.dtype, state.dtype)
        return forward(features, state)
    model.memory.forward = spy
    with torch.inference_mode():
        output, state = model(probe)
        _, next_state = model(probe, state)
    assert seen["dtypes"] == (torch.float32, torch.float32)
    assert output.dtype == state.dtype == next_state.dtype == torch.float16
    assert torch.isfinite(output).all() and torch.isfinite(next_state).all()


def test_the_pt_record_names_the_architecture_and_each_loader_takes_only_its_own(tmp_path):
    model = MemoryNet()
    save_model(model, tmp_path / "memory.pt")
    assert read_record(tmp_path / "memory.pt")["arch"] == "memory_s4"
    loaded = load_model(tmp_path / "memory.pt", CPU)
    assert isinstance(loaded, MemoryNet) and not loaded.training and loaded.out.weight.dtype == torch.float16
    assert set(loaded.state_dict()) == set(model.state_dict())
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file \\(arch 'memory_s4'\\)"):
        heatmapnet.load_model(tmp_path / "memory.pt", CPU)
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file"):
        from_heatmap(tmp_path / "memory.pt", CPU)
    save_model(HeatmapNet(), tmp_path / "heat.pt")
    with pytest.raises(ValueError, match="not a memory_s4 weights file \\(arch 'heatmap_s4'\\)"):
        load_model(tmp_path / "heat.pt", CPU)
    with pytest.raises(ValueError, match="not a readable memory_s4 weights file"):
        load_model(tmp_path / "missing.pt", CPU)
