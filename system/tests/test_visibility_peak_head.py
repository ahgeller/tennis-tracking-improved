"""The stream_s4 visibility head's ``peak`` variant (``streamnet.PeakHead``): a per-frame answer
read at frame t's own heat peak from the stride-4 decoder features, the frame's transition
features and its top heat values. The contract (``forward(value, return_vis=True)`` answering
``(grids, (N, 5) logits)``, the record round trip, the two-output ONNX export) is the global
head's; which head a record carries is read off its keys."""
import copy
import io
import warnings

import pytest
import torch
from torch import nn

from tennis_tracker import streamnet
from tennis_tracker.ball_models import deployed, load_ball_model
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, INPUT_CHANNELS
from tennis_tracker.heatmapnet import save_model

CPU = torch.device("cpu")
SMALL = (64, 96)        # a small input, a multiple of TOTAL_STRIDE


def scrambled(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    return model


def test_the_peak_head_is_selected_at_construction_and_the_record_says_which_head_it_is():
    """``vis_pool`` picks the head: the default is the global one (the keys the s4_fg1 record
    carries, unchanged), ``peak`` the new one under the same ``vis.`` prefix with keys of its own,
    so ``vis_pool_of`` reads the variant off a state dict and ``has_vis_head`` is true for both.
    An unknown pool is a ValueError; so is a ``vis.`` key set of neither shape."""
    plain = streamnet.StreamNet()
    assert plain.vis_pool is None and streamnet.vis_pool_of(plain.state_dict()) is None
    global_head = streamnet.StreamNet(vis_head=True)
    assert global_head.vis_pool == "global" and isinstance(global_head.vis, nn.Linear)
    assert set(global_head.state_dict()) - set(plain.state_dict()) == {"vis.weight", "vis.bias"}
    assert streamnet.vis_pool_of(global_head.state_dict()) == "global"
    peak = streamnet.StreamNet(vis_head=True, vis_pool="peak")
    assert peak.vis_pool == "peak" and isinstance(peak.vis, streamnet.PeakHead) and peak.has_vis_head
    extra = set(peak.state_dict()) - set(plain.state_dict())
    assert extra and all(key.startswith(streamnet.VIS_PREFIX) for key in extra) and "vis.weight" not in extra
    assert streamnet.vis_pool_of(peak.state_dict()) == "peak" and streamnet.has_vis_head(peak.state_dict())
    assert streamnet.VIS_POOLS == ("global", "peak")
    with pytest.raises(ValueError, match="vis_pool"):
        streamnet.StreamNet(vis_head=True, vis_pool="mean")
    with pytest.raises(ValueError, match="visibility head"):
        streamnet.vis_pool_of({"vis.something": torch.zeros(1)})
    # enable_vis_head: idempotent for the same pool, a ValueError for another unless replace is asked
    assert peak.enable_vis_head("peak") is peak and peak.enable_vis_head("peak").vis is peak.vis
    with pytest.raises(ValueError, match="already has"):
        peak.enable_vis_head("global")
    assert peak.enable_vis_head("global", replace=True).vis_pool == "global" and isinstance(peak.vis, nn.Linear)


def test_the_peak_head_answers_one_logit_per_frame_from_that_frames_own_peak_and_leaves_the_grids_alone():
    """The default call is bit for bit the headless network's; ``return_vis`` adds ``(N, 5)``
    logits near the prior for a fresh head. The logit of frame k depends on frame k's heat map
    and not on another frame's: moving frame 2's peak elsewhere changes logit 2 and no other
    (the decoder features at the new place differ), which the global head, pooled over the whole
    unit, cannot promise. The streaming decode answers the same as the stacked unit."""
    torch.manual_seed(2)
    plain = streamnet.StreamNet().eval()
    headed = streamnet.StreamNet(vis_head=True, vis_pool="peak").eval()
    headed.load_state_dict(plain.state_dict(), strict=False)
    value = torch.rand(2, INPUT_CHANNELS, *SMALL)
    with torch.no_grad():
        grids = headed(value)
        grids_again, logits = headed(value, return_vis=True)
        expected = plain(value)
    assert torch.equal(grids, expected) and torch.equal(grids_again, grids)
    assert logits.shape == (2, streamnet.VIS_LOGITS) == (2, FRAMES_PER_UNIT) and float(logits.abs().max()) < 0.5
    frames = [torch.rand(1, 3, *SMALL) for _ in range(FRAMES_PER_UNIT)]
    with torch.no_grad():
        cached = [headed.encode_frame(frames[index], frames[index - 1]) for index in range(1, FRAMES_PER_UNIT)]
        streamed, streamed_logits = headed.decode(cached, return_vis=True)
        unit, unit_logits = headed(torch.cat(frames, dim=1), return_vis=True)
    assert torch.allclose(streamed, unit, atol=1e-6) and torch.allclose(streamed_logits, unit_logits, atol=1e-6)
    # Per frame: the head read at frame k's peak. With the heat maps and features handed to it
    # directly, moving frame 2's peak moves logit 2 alone.
    head = scrambled(streamnet.PeakHead())
    batch, rows, cols = 1, SMALL[0] // 4, SMALL[1] // 4
    heat = torch.full((batch, FRAMES_PER_UNIT, 3, rows, cols), 0.1)
    for frame in range(FRAMES_PER_UNIT):
        heat[0, frame, 0, 2 + frame, 3] = 0.9
    up = torch.randn(batch, streamnet.DECODER_CHANNELS[2], rows, cols)
    quarter = torch.randn(batch, streamnet.PAIRS_PER_UNIT * streamnet.ENCODER_CHANNELS[1], rows, cols)
    with torch.no_grad():
        before = head(heat.reshape(batch, -1, rows, cols), up, quarter)
        moved = heat.clone()
        moved[0, 2, 0, 4, 3] = 0.1
        moved[0, 2, 0, 10, 20] = 0.9
        after = head(moved.reshape(batch, -1, rows, cols), up, quarter)
    assert before.shape == (batch, FRAMES_PER_UNIT)
    assert not torch.allclose(before[0, 2], after[0, 2])
    assert torch.equal(before[0, [0, 1, 3, 4]], after[0, [0, 1, 3, 4]])


def test_the_peak_head_reads_the_frames_own_transition_and_the_top_heat_values():
    """Frame k's logit reads the transition that produced frame k (pair k - 1; frame 0 reads the
    first pair, whose negative difference field holds where its ball was) and the frame's
    ``VIS_TOPK`` highest heat values: changing pair 1's features at frame 2's peak changes logit 2
    alone, and a second peak in frame 1's heat map (below the first) changes logit 1 alone."""
    torch.manual_seed(3)
    head = scrambled(streamnet.PeakHead())
    rows, cols = SMALL[0] // 4, SMALL[1] // 4
    e4 = streamnet.ENCODER_CHANNELS[1]
    heat = torch.full((1, FRAMES_PER_UNIT, 3, rows, cols), 0.1)
    for frame in range(FRAMES_PER_UNIT):
        heat[0, frame, 0, 2 + frame, 3] = 0.9
    grids = heat.reshape(1, -1, rows, cols)
    up = torch.randn(1, streamnet.DECODER_CHANNELS[2], rows, cols)
    quarter = torch.randn(1, streamnet.PAIRS_PER_UNIT * e4, rows, cols)
    with torch.no_grad():
        before = head(grids, up, quarter)
        touched = quarter.clone()
        touched[0, e4 * 1:e4 * 2, 4, 3] += 1.0                      # pair 1 = (frame 2 | frame 1), at frame 2's peak
        after = head(grids, up, touched)
        second = heat.clone()
        second[0, 1, 0, 9, 9] = 0.5                                   # a second, lower peak in frame 1
        with_second = head(second.reshape(1, -1, rows, cols), up, quarter)
    assert not torch.allclose(before[0, 2], after[0, 2]) and torch.equal(before[0, [0, 1, 3, 4]], after[0, [0, 1, 3, 4]])
    assert not torch.allclose(before[0, 1], with_second[0, 1]) and torch.equal(before[0, [0, 2, 3, 4]], with_second[0, [0, 2, 3, 4]])
    assert streamnet.VIS_TOPK >= 2


def test_a_record_with_the_peak_head_round_trips_deploys_and_is_rebuilt_as_the_peak_variant(tmp_path):
    headed = scrambled(streamnet.StreamNet(vis_head=True, vis_pool="peak"))
    save_model(headed, tmp_path / "peak.pt")
    loaded = load_ball_model(tmp_path / "peak.pt", CPU).float()
    assert isinstance(loaded, streamnet.StreamNet) and loaded.vis_pool == "peak" and loaded.has_vis_head
    value = torch.rand(1, INPUT_CHANNELS, *SMALL)
    with torch.no_grad():
        grids, logits = headed.eval()(value, return_vis=True)
        back, back_logits = loaded(value, return_vis=True)
        fused = deployed(copy.deepcopy(loaded))
        fused_grids, fused_logits = fused(value, return_vis=True)
        assert torch.equal(fused(value), fused_grids)
    assert torch.allclose(grids, back, atol=2e-3) and torch.allclose(logits, back_logits, atol=2e-3, rtol=2e-3)
    assert torch.allclose(fused_grids, back, atol=1e-5) and torch.allclose(fused_logits, back_logits, atol=1e-4, rtol=1e-4)
    assert fused.vis_pool == "peak" and not any(isinstance(module, nn.BatchNorm2d) for module in fused.modules())
    # the global record of the earlier run still rebuilds as the global head
    save_model(streamnet.StreamNet(vis_head=True), tmp_path / "global.pt")
    assert load_ball_model(tmp_path / "global.pt", CPU).vis_pool == "global"


def test_the_two_output_export_of_the_peak_head_has_fixed_shapes_and_no_dynamic_control_flow():
    """``ball_trt.export_onnx(visibility=True)`` of the peak variant: the same two outputs, the
    visibility one ``(batch, 5)`` with static dims; the graph holds TopK and gather nodes with fixed
    k and no Loop / If, so TensorRT and Core ML take it as they take the global head's export."""
    onnx = pytest.importorskip("onnx")
    from tennis_tracker.engines import ball_trt

    model = deployed(streamnet.StreamNet(vis_head=True, vis_pool="peak")).half()
    headed = onnx.load_from_string(ball_trt.export_onnx(model, 2, CPU, visibility=True))
    assert [output.name for output in headed.graph.output] == [ball_trt.OUTPUT_NAME, ball_trt.VISIBILITY_OUTPUT_NAME]
    dims = [d.dim_value for d in headed.graph.output[1].type.tensor_type.shape.dim]
    assert dims == [2, streamnet.VIS_LOGITS]
    grid_dims = [d.dim_value for d in headed.graph.output[0].type.tensor_type.shape.dim]
    assert grid_dims[:2] == [2, 15] and all(dim > 0 for dim in grid_dims)
    ops = {node.op_type for node in headed.graph.node}
    assert "TopK" in ops and ({"GatherElements", "Gather", "GatherND"} & ops)
    assert not ({"Loop", "If", "NonZero", "NonMaxSuppression"} & ops)
    plain = onnx.load_from_string(ball_trt.export_onnx(model, 2, CPU))
    assert [output.name for output in plain.graph.output] == [ball_trt.OUTPUT_NAME]
    assert "TopK" not in {node.op_type for node in plain.graph.node}      # the default export never computes the head
    # the traced small-input export agrees with eager on the logits (float32, the head's own math)
    small = deployed(streamnet.StreamNet(vis_head=True, vis_pool="peak"))
    probe = torch.rand(1, INPUT_CHANNELS, *SMALL)
    buffer = io.BytesIO()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(ball_trt._WithVisibility(small), (probe,), buffer, input_names=["frames"],
                          output_names=["grids", "visibility"], opset_version=ball_trt._ONNX_OPSET, dynamo=False)
    exported = onnx.load_from_string(buffer.getvalue())
    assert [d.dim_value for d in exported.graph.output[1].type.tensor_type.shape.dim] == [1, FRAMES_PER_UNIT]


def test_the_peak_head_costs_a_few_tens_of_thousands_of_macs_per_unit():
    """Counted from the layer sizes: two linear layers per frame, five frames; the top-k and the
    gathers are selections, not multiplies. Under 0.1 MMAC against the network's 2.64 GMAC a frame."""
    assert streamnet.PeakHead().macs_per_unit() < 100_000
    assert streamnet.PeakHead().macs_per_unit() == FRAMES_PER_UNIT * (
        (streamnet.DECODER_CHANNELS[2] + streamnet.ENCODER_CHANNELS[1] + streamnet.VIS_TOPK) * streamnet.VIS_HIDDEN
        + streamnet.VIS_HIDDEN)
