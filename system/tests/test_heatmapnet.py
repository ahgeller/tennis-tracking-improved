"""tennis_tracker/heatmapnet.py and the decoder the two architectures share: the network's output
shape and range, the shared contract, the .pt round trip and ``decode_predictions`` on both grids.
Tiny batches on the CPU."""
import math

import pytest
import torch

from tennis_tracker import gridtracknet, heatmapnet
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, INPUT_CHANNELS, WIDTH, decode_predictions
from tennis_tracker.heatmapnet import HEAT_PRIOR, HeatmapNet, load_model, read_record, save_model

CPU = torch.device("cpu")
SMALL = (96, 160)      # a small input (a multiple of the total stride 16): the network is fully convolutional, so the tests stay quick


def test_output_is_fifteen_sigmoid_channels_at_stride_four():
    model = HeatmapNet().eval()
    with torch.inference_mode():
        output = model(torch.rand(2, INPUT_CHANNELS, *SMALL))
    assert output.shape == (2, FRAMES_PER_UNIT * 3, SMALL[0] // 4, SMALL[1] // 4)
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
    assert output.shape[1] == INPUT_CHANNELS                                   # the layout GridTrackNet answers in


def test_the_full_size_input_maps_onto_the_declared_grid_and_another_size_is_refused():
    model = HeatmapNet().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, INPUT_CHANNELS, HEIGHT, WIDTH))
    assert output.shape == (1, 15, HeatmapNet.GRID_ROWS, HeatmapNet.GRID_COLS) == (1, 15, 108, 192)
    with pytest.raises(ValueError, match="multiples of 16"), torch.inference_mode():
        model(torch.zeros(1, INPUT_CHANNELS, 108, 192))


def test_a_fresh_head_predicts_the_heat_prior_and_the_contract_matches_gridtracknet():
    """The head's heat biases start at logit(HEAT_PRIOR), so with zero features every heat cell
    sits at the prior; the offset channels start at 0.5 (zero bias)."""
    model = HeatmapNet().eval()
    with torch.no_grad():
        model.head.weight.zero_()
        output = model(torch.zeros(1, INPUT_CHANNELS, *SMALL))
    for frame in range(FRAMES_PER_UNIT):
        assert float(output[0, frame * 3].mean()) == pytest.approx(HEAT_PRIOR, abs=1e-4)
        assert float(output[0, frame * 3 + 1].mean()) == pytest.approx(0.5, abs=1e-4)
    assert float(model.head.bias.detach()[0]) == pytest.approx(-math.log((1 - HEAT_PRIOR) / HEAT_PRIOR))
    for name in ("ARCH", "GRID_COLS", "GRID_ROWS", "INPUT_CHANNELS", "FRAMES_PER_UNIT"):
        assert hasattr(HeatmapNet, name) and hasattr(gridtracknet.GridTrackNet, name)
    assert (HeatmapNet.ARCH, gridtracknet.GridTrackNet.ARCH) == ("heatmap_s4", "gridtracknet")
    assert (HeatmapNet.INPUT_CHANNELS, HeatmapNet.FRAMES_PER_UNIT) == (gridtracknet.GridTrackNet.INPUT_CHANNELS, 5)
    assert (gridtracknet.GridTrackNet.GRID_COLS, gridtracknet.GridTrackNet.GRID_ROWS) == (48, 27)


def test_pt_round_trip_restores_every_tensor_and_loads_in_eval_fp16(tmp_path):
    model = HeatmapNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.rand_like(parameter) * 0.01)
    model.train()
    with torch.no_grad():
        model(torch.rand(2, INPUT_CHANNELS, *SMALL))           # moves the BatchNorm running statistics
    destination = tmp_path / "out" / "weights.pt"
    save_model(model, destination)
    assert destination.is_file() and not list((tmp_path / "out").glob("*.tmp.pt"))
    record = read_record(destination)
    assert record["arch"] == "heatmap_s4" and set(record["state_dict"]) == set(model.state_dict())
    loaded = load_model(destination, CPU)
    assert not loaded.training and loaded.head.weight.dtype == torch.float16
    for name, tensor in model.state_dict().items():
        back = loaded.state_dict()[name]
        if tensor.is_floating_point():
            assert torch.allclose(back.float(), tensor.half().float()), name
        else:
            assert torch.equal(back, tensor), name                   # num_batches_tracked


def test_other_pt_files_are_refused_by_name(tmp_path):
    """read_record returns any record of the right shape and refuses another architecture only
    when asked for one (the loader asks; ball_models.arch_of does not); a bare state dict and an
    unreadable file are refused either way."""
    torch.save(HeatmapNet().state_dict(), tmp_path / "bare.pt")
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file"):
        load_model(tmp_path / "bare.pt", CPU)
    with pytest.raises(ValueError, match="not a ball-detector weights file \\(arch None\\)"):
        read_record(tmp_path / "bare.pt")
    torch.save({"arch": "something_else", "state_dict": {}}, tmp_path / "other.pt")
    assert read_record(tmp_path / "other.pt")["arch"] == "something_else"
    with pytest.raises(ValueError, match="not a heatmap_s4 weights file \\(arch 'something_else'\\)"):
        read_record(tmp_path / "other.pt", "heatmap_s4")
    (tmp_path / "garbage.pt").write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="not a readable heatmap_s4 weights file"):
        read_record(tmp_path / "garbage.pt", "heatmap_s4")
    with pytest.raises(ValueError, match="not a readable ball-detector weights file"):
        read_record(tmp_path / "garbage.pt")


@pytest.mark.parametrize("grid", [(gridtracknet.GRID_COLS, gridtracknet.GRID_ROWS), (heatmapnet.GRID_COLS, heatmapnet.GRID_ROWS)])
def test_decode_round_trips_a_peak_at_a_known_native_position_on_either_grid(grid):
    """A synthetic peak whose cell and offsets encode a native point decodes back to that point
    within half a pixel on the 48x27 and the 192x108 grid alike."""
    cols, rows = grid
    width, height = 1920, 1080
    point = (1234.5, 678.25)
    grid_x, grid_y = point[0] * cols / width, point[1] * rows / height
    col, row = int(grid_x), int(grid_y)
    output = torch.zeros((1, 15, rows, cols))
    output[0, 0, row, col] = 0.9
    output[0, 1, row, col] = grid_x - col
    output[0, 2, row, col] = grid_y - row
    output[0, 6, 2, 3] = 0.3                                      # frame 2: below the threshold
    decoded = decode_predictions(output, width, height, 0.5)
    assert len(decoded) == FRAMES_PER_UNIT
    (decoded_point, confidence) = decoded[0]
    assert confidence == pytest.approx(0.9) and abs(decoded_point[0] - point[0]) < 0.5 and abs(decoded_point[1] - point[1]) < 0.5
    assert decoded[2] == (None, pytest.approx(0.3)) and decoded[1] == (None, 0.0)


def test_decode_reads_the_grid_from_the_output_so_a_cell_is_a_different_size_on_each():
    """The same cell index is a different native position on the two grids: the decoder uses the
    output's own shape, not a constant."""
    coarse = torch.zeros((1, 15, gridtracknet.GRID_ROWS, gridtracknet.GRID_COLS))
    fine = torch.zeros((1, 15, heatmapnet.GRID_ROWS, heatmapnet.GRID_COLS))
    coarse[0, 0, 4, 6] = fine[0, 0, 4, 6] = 0.9
    assert decode_predictions(coarse, 1920, 1080, 0.5)[0][0] == (240.0, 160.0)
    assert decode_predictions(fine, 1920, 1080, 0.5)[0][0] == (60.0, 40.0)
