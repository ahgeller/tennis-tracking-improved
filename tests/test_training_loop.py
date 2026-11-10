"""finetune/training/loop.py on its own: the loss of each architecture, a fresh network, an epoch of
optimisation, the validation loss and the weights written back in the tracker's layouts. Tiny
synthetic units."""
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetune.training.dataset import make_target
from finetune.training.loop import (EMA, EMA_DECAY, HEAT_OFFSET_WEIGHT, LOSSES, VIS_LOSS_WEIGHT, Validation, batch_loss,
                                    build_optimizer, check_finite, evaluate, evaluate_loss, fresh_model, gridtracknet_loss,
                                    has_vis_head, heatmap_loss, hold_conv_scale, FILTER_NORM, largest_running_variance,
                                    loss_of, normalised_conv_weights, save_weights, train_epoch, unit_loss, unpack,
                                    visibility_loss, visibility_metrics, visible_frames, with_vis_head)
from tennis_tracker import heatmapnet, streamnet
from tennis_tracker.ball_models import load_ball_model
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, load_model
from _workspace_helpers import blank_template


def one_unit_loader():
    images = torch.zeros(1, 3 * FRAMES_PER_UNIT, HEIGHT, WIDTH)
    target = torch.zeros(1, 3 * FRAMES_PER_UNIT, GRID_ROWS, GRID_COLS)
    target[0, 0, 13, 24] = 1.0
    return DataLoader(TensorDataset(images, target), batch_size=1)


def test_loss_is_finite_and_rewards_the_target_cell():
    target = torch.zeros(1, 3 * FRAMES_PER_UNIT, GRID_ROWS, GRID_COLS)
    target[0, 0, 13, 24] = 1.0
    flat = torch.full_like(target, 0.1)
    right = flat.clone()
    right[0, 0, 13, 24] = 0.9
    assert torch.isfinite(gridtracknet_loss(flat, target)) and gridtracknet_loss(right, target) < gridtracknet_loss(flat, target)
    assert torch.isfinite(gridtracknet_loss(torch.ones_like(target), target))     # clamped: no log(0)


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")   # train_epoch reads the loss as a float
def test_a_fresh_model_trains_evaluates_and_round_trips_through_the_npz_layout(tmp_path):
    device = torch.device("cpu")
    model = fresh_model(device)
    assert model(torch.zeros(1, 3 * FRAMES_PER_UNIT, HEIGHT, WIDTH)).shape == (1, 3 * FRAMES_PER_UNIT, GRID_ROWS, GRID_COLS)
    loader = one_unit_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_epoch(model, loader, optimizer, scaler, device, epoch=1, epochs=1, max_steps=1)
    assert np.isfinite(loss) and np.isfinite(evaluate_loss(model, loader, device, max_steps=1))
    template = tmp_path / "template.npz"
    save_weights(model, blank_template(model, template), tmp_path / "out" / "weights.npz")
    reloaded = load_model(tmp_path / "out" / "weights.npz", device).float()
    with torch.inference_mode():
        probe = torch.rand(1, 3 * FRAMES_PER_UNIT, HEIGHT, WIDTH)
        assert torch.allclose(reloaded(probe), model.eval()(probe), atol=5e-3)   # the tracker loads in float16


def test_heatmap_loss_is_finite_and_lower_for_the_right_answer_than_a_wrong_one():
    """CenterNet's focal loss against the Gaussian target: a prediction with its peak at the ball
    scores below one that peaks elsewhere and below a flat one; a target with no ball at all is
    finite (the positive count is clamped to one); nothing takes log(0)."""
    target = make_target([[960.0, 540.0], None, None, None, None], 1920, 1080, heatmapnet.ARCH).unsqueeze(0)
    flat = torch.full_like(target, 0.1)
    right = flat.clone()
    right[0, 0] = target[0, 0].clamp(min=0.1)                         # the heat as labelled, offsets right
    right[0, 1, 54, 96], right[0, 2, 54, 96] = target[0, 1, 54, 96], target[0, 2, 54, 96]
    wrong = flat.clone()
    wrong[0, 0, 10, 10] = 0.95
    assert torch.isfinite(heatmap_loss(flat, target))
    assert heatmap_loss(right, target) < heatmap_loss(flat, target) < heatmap_loss(wrong, target)
    assert torch.isfinite(heatmap_loss(torch.ones_like(target), target)) and torch.isfinite(heatmap_loss(torch.zeros_like(target), target))
    assert torch.isfinite(heatmap_loss(flat, torch.zeros_like(target)))
    prediction = flat.clone().requires_grad_(True)
    heatmap_loss(prediction, target).backward()
    assert torch.isfinite(prediction.grad).all()
    assert LOSSES == {"gridtracknet": gridtracknet_loss, "heatmap_s4": heatmap_loss, "heatmap_lite": heatmap_loss,
                      "stream_s4": heatmap_loss, "wasb_5f": heatmap_loss}
    model = fresh_model(torch.device("cpu"), heatmapnet.ARCH)
    assert unit_loss(model, right, target) == heatmap_loss(right, target)
    model.eval()
    unit = torch.rand(1, 15, 32, 32)
    with torch.inference_mode():
        assert torch.equal(batch_loss(model, unit, flat[:, :, :8, :8]), unit_loss(model, model(unit), flat[:, :, :8, :8]))


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")   # train_epoch reads the loss as a float
def test_a_fresh_heatmapnet_trains_evaluates_and_round_trips_through_the_pt_record(tmp_path):
    """fresh_model builds the architecture asked for, the loop runs it with its own loss, and
    save_weights writes the .pt record (no NPZ template read) that load_ball_model brings back."""
    device = torch.device("cpu")
    model = fresh_model(device, heatmapnet.ARCH)
    assert isinstance(model, heatmapnet.HeatmapNet) and model.training
    images = torch.rand(1, 3 * FRAMES_PER_UNIT, 96, 160)
    target = make_target([[100.0, 50.0], None, None, None, None], 160, 96, heatmapnet.ARCH)[:, :24, :40].unsqueeze(0)
    loader = DataLoader(TensorDataset(images, target), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_epoch(model, loader, optimizer, scaler, device, epoch=1, epochs=1, max_steps=1)
    assert np.isfinite(loss) and np.isfinite(evaluate_loss(model, loader, device, max_steps=1))
    save_weights(model, tmp_path / "no_template.npz", tmp_path / "out" / "weights.pt")
    reloaded = load_ball_model(tmp_path / "out" / "weights.pt", device)
    assert not reloaded.training and not (tmp_path / "out" / "weights.npz").exists()
    for name, tensor in model.state_dict().items():                      # exact, after the tracker's float16
        expected = tensor.half() if tensor.is_floating_point() else tensor
        assert torch.equal(reloaded.state_dict()[name], expected), name


def test_heatmap_offsets_are_weighted_as_centernet_weights_them():
    """A wrong offset at the peak costs HEAT_OFFSET_WEIGHT times its L1 error per positive (1.0,
    lambda_off), not GridTrackNet's 0.001: heat1 trained at the latter and its offsets never
    moved (3.4 px median at 1080p, a quarter of a cell at random)."""
    target = make_target([None, (100.0, 50.0), None, None, None], WIDTH, HEIGHT, arch="heatmap_s4").unsqueeze(0)
    exact = target.clone()
    exact[:, 3] = 0.5                      # the second frame's heat is right, and so are its offsets
    exact[:, 4:6] = target[:, 4:6]
    heat_true = target[0, 3]
    row, col = divmod(int(heat_true.argmax()), heat_true.shape[-1])
    off = exact.clone()
    off[0, 4, row, col] = target[0, 4, row, col] + 0.25
    off[0, 5, row, col] = target[0, 5, row, col] - 0.25
    assert HEAT_OFFSET_WEIGHT == 1.0
    assert heatmap_loss(off, target) - heatmap_loss(exact, target) == pytest.approx(HEAT_OFFSET_WEIGHT * 0.5, abs=1e-5)


def test_the_normalised_convolutions_are_exactly_the_ones_batchnorm_follows_and_the_optimizer_is_plain():
    """A HeatmapNet's fourteen block convolutions (two per block: stem, four stages, two decoder
    stages) are the normalised ones and nothing else is: not the heat head, whose scale the
    sigmoid reads; a GridTrackNet has none. The optimizer is one undecayed Adadelta group for
    either architecture (recipe 2's decay group is gone)."""
    heat = heatmapnet.HeatmapNet()
    held = normalised_conv_weights(heat)
    assert len(held) == 14 and all(weight.dim() == 4 for weight in held)
    held_ids = {id(weight) for weight in held}
    for name, parameter in heat.named_parameters():
        assert (id(parameter) in held_ids) == (parameter.dim() == 4 and not name.startswith("head")), name
    grid = fresh_model(torch.device("cpu"), "gridtracknet")
    assert normalised_conv_weights(grid) == []
    for model, lr in ((heat, 1.0), (grid, 0.3)):
        optimizer = build_optimizer(model, lr)
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group["weight_decay"] == 0 and group["lr"] == lr and group["rho"] == 0.95 and group["eps"] == 1e-7
        assert len(group["params"]) == len(list(model.parameters()))


def test_holding_the_normalised_convolutions_leaves_the_answer_alone_and_an_epoch_keeps_them_held():
    """Scaling a normalised convolution's weight by 40 (heat3's decoder by epoch 16), one of its
    output channels alone by 10, or another weight by a quarter does not change what the network
    answers in train mode (to BatchNorm's eps, which is why a weight shrunk a hundredfold would),
    and hold_conv_scale brings every filter back to FILTER_NORM (sqrt(2), a Kaiming filter's)
    without changing the answer either; after a training step every held filter sits at that
    norm, so the pre-normalisation activations stay a fresh network's size at any learning rate.
    A biased convolution before BatchNorm is refused: the hold would change its answer."""
    torch.manual_seed(3)
    model = heatmapnet.HeatmapNet().train()
    held = normalised_conv_weights(model)
    images = torch.rand(2, 3 * FRAMES_PER_UNIT, 96, 160)
    with torch.no_grad():
        before = model(images)
        held[0].mul_(40.0)
        held[5][2].mul_(10.0)
        held[13].mul_(0.25)
        scaled = model(images)
    assert torch.allclose(before, scaled, atol=1e-3)
    hold_conv_scale(held)
    assert FILTER_NORM == pytest.approx(2 ** 0.5)
    for weight in held:
        assert torch.allclose(weight.detach().flatten(1).norm(dim=1), torch.full((weight.shape[0],), FILTER_NORM), rtol=1e-5)
    with torch.no_grad():
        assert torch.allclose(before, model(images), atol=1e-3)
    biased = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4))
    with pytest.raises(ValueError, match="bias-free"):
        normalised_conv_weights(biased)
    target = make_target([[100.0, 50.0], None, None, None, None], 160, 96, heatmapnet.ARCH)[:, :24, :40].unsqueeze(0)
    loader = DataLoader(TensorDataset(images[:1], target), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=10.0)       # a step large enough to move every norm
    train_epoch(model, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), 1, 1, 1)
    for weight in held:
        assert torch.allclose(weight.detach().flatten(1).norm(dim=1), torch.full((weight.shape[0],), FILTER_NORM), rtol=1e-5)
    grid = fresh_model(torch.device("cpu"), "gridtracknet")
    hold_conv_scale(normalised_conv_weights(grid))                   # nothing to hold, nothing happens


def test_check_finite_names_a_poisoned_running_statistic_and_the_largest_variance_is_reported():
    """An infinite BatchNorm running variance (what heat1's decoder had from epoch 24, silencing
    the eval-mode network while the training loss kept falling) is a RuntimeError naming the
    buffer; a finite network passes, and largest_running_variance says which buffer is nearest
    float16's ceiling."""
    model = heatmapnet.HeatmapNet()
    check_finite(model)
    assert largest_running_variance(model)[1] == 1.0            # every running variance starts at one
    model.up2[1].running_var[3] = 25104.0
    assert largest_running_variance(model) == ("up2.1.running_var", 25104.0)
    model.up2[4].running_var[0] = float("inf")
    with pytest.raises(RuntimeError, match=r"up2\.4\.running_var holds a non-finite value"):
        check_finite(model)
    assert largest_running_variance(fresh_model(torch.device("cpu"), "gridtracknet")) == ("", 0.0)


@pytest.mark.parametrize("arch", ["gridtracknet", heatmapnet.ARCH])
def test_an_ignored_frame_leaves_the_loss_and_a_valid_mask_of_ones_changes_nothing(arch):
    """A pseudo-labelled unit's ignored frames (``valid`` 0) count neither as positives nor as
    negatives: the loss is the loss of the other frames, whatever is predicted on them."""
    loss = loss_of(arch)
    target = make_target([[960.0, 540.0], None, [300.0, 200.0], None, [10.0, 10.0]], 1920, 1080, arch).unsqueeze(0)
    torch.manual_seed(0)
    prediction = torch.rand_like(target).clamp(0.05, 0.95)
    ones = torch.ones(1, FRAMES_PER_UNIT)
    assert torch.allclose(loss(prediction, target), loss(prediction, target, ones))
    valid = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    scrambled = prediction.clone()
    scrambled[:, 6:9] = torch.rand_like(scrambled[:, 6:9]).clamp(0.05, 0.95)       # frame 2's answer, ignored
    assert torch.allclose(loss(prediction, target, valid), loss(scrambled, target, valid))
    assert not torch.allclose(loss(prediction, target), loss(scrambled, target))
    assert loss_of("heatmap_lite") is heatmap_loss and loss_of("gridtracknet") is LOSSES["gridtracknet"]


def test_unpack_reads_the_plain_and_the_longer_batches():
    images, target = torch.zeros(1, 15, 2, 2), torch.zeros(1, 15, 1, 1)
    assert unpack((images, target), torch.device("cpu"))[2:] == (None, None)
    valid, teacher = torch.ones(1, 5), torch.ones(1, 15, 1, 1)
    unpacked = unpack((images, target, valid, teacher), torch.device("cpu"))
    assert unpacked[2] is not None and unpacked[3] is not None and torch.equal(unpacked[3], teacher)


# --------------------------------------------------------------------------- the weight average (--ema)

def test_the_ema_tracks_the_live_weights_copies_the_buffers_and_holds_the_norm():
    """After one step the shadow sits decay of the way from the live weights toward where it was,
    its BatchNorm statistics are the live network's, its held convolutions are back at FILTER_NORM
    (an average of two unit filters is shorter), it stays in eval mode with no gradient, and its
    state round-trips through a fresh EMA (the checkpoint's path). A decay outside (0, 1) is refused."""
    torch.manual_seed(2)
    model = streamnet.StreamNet().train()
    ema = EMA(model, decay=0.9)
    assert ema.decay == 0.9 and not ema.model.training and ema.updates == 0 and EMA_DECAY == 0.999
    assert all(not parameter.requires_grad for parameter in ema.model.parameters())
    before = {name: tensor.clone() for name, tensor in ema.model.state_dict().items()}
    with torch.no_grad():
        model(torch.rand(2, 15, 64, 96))                      # moves the running statistics
        model.head.bias.add_(1.0)
        model.fuse16[1].dense[0].weight.mul_(3.0)             # a held filter, off its norm
    ema.update(model)
    assert ema.updates == 1
    assert torch.allclose(ema.model.head.bias, 0.9 * before["head.bias"] + 0.1 * model.head.bias)
    assert torch.equal(ema.model.fuse16[1].dense[1].running_mean, model.fuse16[1].dense[1].running_mean)
    assert not torch.equal(ema.model.fuse16[1].dense[1].running_mean, before["fuse16.1.dense.1.running_mean"])
    norms = ema.model.fuse16[1].dense[0].weight.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, FILTER_NORM), rtol=1e-5)
    assert not torch.equal(ema.model.head.bias, model.head.bias)      # a shadow, not an alias
    again = EMA(streamnet.StreamNet())
    again.load_state_dict(ema.state_dict())
    assert again.updates == 1
    for name, tensor in ema.model.state_dict().items():
        assert torch.equal(again.model.state_dict()[name], tensor), name
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            EMA(model, decay=bad)


# --------------------------------------------------------------------------- the visibility head (--vis-head)

def test_visible_frames_reads_the_peaks_and_the_visibility_loss_prefers_the_right_answer():
    target = make_target([[960.0, 540.0], None, [300.0, 200.0], None, None], 1920, 1080, streamnet.ARCH).unsqueeze(0)
    labels, weight = visible_frames(target)
    assert torch.equal(labels, torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])) and torch.equal(weight, torch.ones(1, 5))
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])
    assert torch.equal(visible_frames(target, valid)[1], valid)
    right = torch.tensor([[4.0, -4.0, 4.0, -4.0, -4.0]])
    wrong = -right
    flat = torch.zeros(1, 5)
    assert visibility_loss(right, target) < visibility_loss(flat, target) < visibility_loss(wrong, target)
    assert float(visibility_loss(flat, target)) == pytest.approx(math.log(2.0))
    scrambled = right.clone()
    scrambled[0, 3] = 9.0                                              # frame 3's answer, ignored under valid
    assert torch.allclose(visibility_loss(right, target, valid), visibility_loss(scrambled, target, valid))
    assert not torch.allclose(visibility_loss(right, target), visibility_loss(scrambled, target))
    assert float(visibility_loss(right, target, torch.zeros(1, 5))) == 0.0
    assert VIS_LOSS_WEIGHT == 0.1


def test_visibility_metrics_are_the_rank_statistic_and_the_mean_class_recall():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    assert visibility_metrics(torch.tensor([2.0, 1.0, -1.0, -2.0, -3.0]), labels) == (1.0, 1.0)
    assert visibility_metrics(torch.tensor([-3.0, -2.0, 1.0, 2.0, 3.0]), labels) == (0.0, 0.0)
    auc, balanced = visibility_metrics(torch.tensor([2.0, -1.0, 1.0, -1.0, -3.0]), labels)
    assert auc == pytest.approx((3 + 1.5) / 6) and balanced == pytest.approx((0.5 + 2 / 3) / 2)   # the tie at -1 counts half
    assert visibility_metrics(torch.zeros(4), labels[:4]) == (0.5, 0.5)                          # all tied: chance
    assert visibility_metrics(torch.rand(3), torch.ones(3)) == (None, None)


def test_fresh_model_gives_the_head_on_request_and_an_epoch_trains_it_beside_the_heat_loss():
    """fresh_model(..., vis_head=True) is a StreamNet with the head and nothing else changed (the
    15-channel call is untouched); another architecture is refused in one line. A training step
    with --vis-weight moves the head and keeps the hold on the convolutions; evaluate reports the
    head's metrics only when asked, and the plain evaluate_loss is the Validation's loss."""
    device = torch.device("cpu")
    plain = fresh_model(device, streamnet.ARCH)
    headed = fresh_model(device, streamnet.ARCH, vis_head=True)
    assert not has_vis_head(plain) and has_vis_head(headed) and headed.training
    with pytest.raises(ValueError, match="has no visibility head"):
        fresh_model(device, heatmapnet.ARCH, vis_head=True)
    with pytest.raises(ValueError, match="heatmap_s4 has no visibility head"):
        with_vis_head(heatmapnet.HeatmapNet())
    images = torch.rand(2, 15, 64, 96)
    target = torch.stack([make_target([[10.0, 8.0], None, [5.0, 3.0], None, None], 160, 96, streamnet.ARCH)[:, :16, :24],
                          make_target([None] * 5, 160, 96, streamnet.ARCH)[:, :16, :24]])
    loader = DataLoader(TensorDataset(images, target), batch_size=2)
    head_before = headed.vis.weight.detach().clone()
    optimizer = torch.optim.SGD(headed.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = EMA(headed)
    loss = train_epoch(headed, loader, optimizer, scaler, device, 1, 1, None, ema=ema, vis_weight=VIS_LOSS_WEIGHT)
    assert np.isfinite(loss) and not torch.equal(headed.vis.weight, head_before) and ema.updates == 1
    for weight in normalised_conv_weights(headed):
        assert torch.allclose(weight.detach().flatten(1).norm(dim=1), torch.full((weight.shape[0],), FILTER_NORM), rtol=1e-5)
    validation = evaluate(headed, loader, device, visibility=True)
    assert isinstance(validation, Validation) and np.isfinite(validation.loss)
    assert validation.vis_auc is not None and 0.0 <= validation.vis_auc <= 1.0
    assert validation.vis_balanced_accuracy is not None and 0.0 <= validation.vis_balanced_accuracy <= 1.0
    assert evaluate(headed, loader, device) == Validation(validation.loss) == Validation(evaluate_loss(headed, loader, device))
    assert headed.training
    # a weight of 0 leaves the head alone, the loss the heat loss: the run without the flag
    frozen = headed.vis.weight.detach().clone()
    plain_loss = train_epoch(headed, loader, optimizer, scaler, device, 1, 1, None, vis_weight=0.0)
    assert np.isfinite(plain_loss) and torch.equal(headed.vis.weight, frozen)
