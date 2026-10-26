"""finetune/training/loop.py on its own: the loss of each architecture (the chain loss of the stateful
one), a fresh network, an epoch of optimisation, the validation loss and the weights written back
in the tracker's layouts. Tiny synthetic units."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetune.training.dataset import make_target
from finetune.training.loop import (LOSSES, batch_loss, chain_loss, evaluate_loss, fresh_model, gridtracknet_loss,
                                    heatmap_loss, save_weights, train_epoch, unit_loss)
from tennis_tracker import heatmapnet
from tennis_tracker.ball_models import load_ball_model
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, load_model
from tests._helpers import blank_template


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
    assert LOSSES == {"gridtracknet": gridtracknet_loss, "heatmap_s4": heatmap_loss, "memory_s4": heatmap_loss}
    assert unit_loss(fresh_model(torch.device("cpu"), heatmapnet.ARCH), right, target) == heatmap_loss(right, target)


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


# --------------------------------------------------------------------------- chains

class Scripted(torch.nn.Module):
    """A STATEFUL stand-in answering the predictions it was given, one per step, and returning
    as its state the step it is at (so what it received is checkable)."""

    ARCH, STATEFUL = "memory_s4", True

    def __init__(self, predictions):
        super().__init__()
        self.predictions, self.states = predictions, []

    def forward(self, images, state=None):
        self.states.append(state)
        step = len(self.states) - 1
        return self.predictions[step], torch.full((images.shape[0], 1), float(step))


def test_chain_loss_is_the_mean_of_the_units_losses_through_the_carried_state_and_prefers_the_right_answer():
    target = torch.stack([make_target([[960.0, 540.0], None, None, None, None], 1920, 1080, heatmapnet.ARCH),
                          make_target([None, None, [100.0, 100.0], None, None], 1920, 1080, heatmapnet.ARCH)]).unsqueeze(0)
    flat = torch.full_like(target, 0.1)
    right = flat.clone()
    right[0, 0, 0], right[0, 1, 6] = target[0, 0, 0].clamp(min=0.1), target[0, 1, 6].clamp(min=0.1)
    images = torch.zeros(1, 2, 15, 16, 16)
    model = Scripted([right[:, 0], right[:, 1]])
    loss = chain_loss(model, images, target)
    assert loss == pytest.approx(float((heatmap_loss(right[:, 0], target[:, 0]) + heatmap_loss(right[:, 1], target[:, 1])) / 2))
    assert model.states[0] is None and torch.equal(model.states[1], torch.zeros(1, 1))     # None at the start, then step 0's
    assert chain_loss(Scripted([right[:, 0], right[:, 1]]), images, target) < chain_loss(Scripted([flat[:, 0], flat[:, 1]]), images, target)
    assert torch.isfinite(chain_loss(Scripted([torch.ones_like(flat[:, 0])] * 2), images, target))
    assert batch_loss(Scripted([right[:, 0], right[:, 1]]), images, target) == loss
    stateless = fresh_model(torch.device("cpu"), heatmapnet.ARCH).eval()
    unit = torch.rand(1, 15, 32, 32)
    with torch.inference_mode():
        assert torch.equal(batch_loss(stateless, unit, flat[:, 0, :, :8, :8]), unit_loss(stateless, stateless(unit), flat[:, 0, :, :8, :8]))
    assert LOSSES["memory_s4"] is heatmap_loss


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")   # train_epoch reads the loss as a float
def test_a_fresh_memorynet_trains_on_chains_evaluates_and_round_trips_through_the_pt_record(tmp_path):
    """The loop on a MemoryNet: a (B, K, 15, H, W) batch goes through chain_loss, the backward pass
    reaches the memory cell and the shared layers alike, evaluate_loss is finite, and save_weights
    writes a memory_s4 record that load_ball_model brings back as a MemoryNet."""
    from tennis_tracker.memorynet import MemoryNet
    device = torch.device("cpu")
    model = fresh_model(device, "memory_s4")
    assert isinstance(model, MemoryNet) and model.training
    images = torch.rand(1, 3, 3 * FRAMES_PER_UNIT, 96, 160)
    target = torch.stack([make_target([[100.0 + 20 * k, 50.0], None, None, None, None], 160, 96, heatmapnet.ARCH)[:, :24, :40]
                          for k in range(3)]).unsqueeze(0)
    loader = DataLoader(TensorDataset(images, target), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_epoch(model, loader, optimizer, scaler, device, epoch=1, epochs=1, max_steps=1)
    assert np.isfinite(loss) and np.isfinite(evaluate_loss(model, loader, device, max_steps=1))
    assert float(model.out.weight.detach().abs().sum()) > 0.0 and model.memory.conv_z.weight.grad is not None
    assert model.stem[0].weight.grad is not None
    save_weights(model, tmp_path / "no_template.npz", tmp_path / "out" / "weights.pt")
    reloaded = load_ball_model(tmp_path / "out" / "weights.pt", device)
    assert isinstance(reloaded, MemoryNet) and not reloaded.training
    for name, tensor in model.state_dict().items():                      # exact, after the tracker's float16
        expected = tensor.half() if tensor.is_floating_point() else tensor
        assert torch.equal(reloaded.state_dict()[name], expected), name
