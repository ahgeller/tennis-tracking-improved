"""The training loop proper: the loss of each architecture, a fresh network, one epoch of
optimisation, the validation loss and writing the weights back in the layout the tracker loads
(the .npz for GridTrackNet, the .pt record for HeatmapNet).

Which loss and which layout is read off the model's ``ARCH`` (the contract ``ball_models``
spells), so the loop holds no architecture switch of its own: ``LOSSES`` maps the id to the
loss, ``save_weights`` to the writer.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from tennis_tracker import heatmapnet
from tennis_tracker.ball_models import GRIDTRACKNET, HEATMAP_S4, model_class, weights_suffix
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS

PROGRESS_EVERY = 200       # training steps between progress lines
LOSS_EPSILON = 1e-6        # keeps the confidence loss's logs finite
OFFSET_LOSS_WEIGHT = 0.001     # GridTrackNet's, as upstream trains it
HEAT_OFFSET_WEIGHT = 1.0       # the heat-map loss's, CenterNet's lambda_off: at 0.001 the offsets never trained
                               # (heat1, 2026-10-26: 3.4 px median at 1080p, a quarter of a cell at random)
HALF_MAX = 65504.0               # float16's largest value; the prepass and the training forward run in it
POSITIVE_WEIGHT, NEGATIVE_WEIGHT = 0.75, 0.25    # the focal loss's class weights
HEAT_ALPHA, HEAT_BETA = 2.0, 4.0                 # CenterNet's focal exponents: the easy-example one and the penalty reduction


def gridtracknet_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """GridTrackNet's loss, as the upstream trainer has it: a class-weighted focal loss on the
    one-hot confidence grid (``POSITIVE_WEIGHT`` / ``NEGATIVE_WEIGHT``, exponent 2), averaged over
    every cell, plus ``OFFSET_LOSS_WEIGHT`` times the L1 of the offsets at the labelled cells."""
    prediction = prediction.float().reshape(-1, FRAMES_PER_UNIT, 3, GRID_ROWS, GRID_COLS)
    target = target.reshape_as(prediction)
    confidence_true = target[:, :, 0]
    confidence_pred = prediction[:, :, 0].clamp(LOSS_EPSILON, 1 - LOSS_EPSILON)
    positive = POSITIVE_WEIGHT * confidence_true * (1 - confidence_pred).square() * confidence_pred.log()
    negative = NEGATIVE_WEIGHT * (1 - confidence_true) * confidence_pred.square() * (1 - confidence_pred).log()
    confidence_loss = -(positive + negative).mean()
    offset_loss = ((prediction[:, :, 1:3] - target[:, :, 1:3]).abs() * confidence_true.unsqueeze(2)).sum(dim=(2, 3, 4)).mean()
    return confidence_loss + OFFSET_LOSS_WEIGHT * offset_loss


def heatmap_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """HeatmapNet's loss: CenterNet's penalty-reduced focal loss on the heat channel against
    the Gaussian target (``dataset.make_target``), where a cell at exactly 1 is a positive and
    every other cell a negative whose penalty is reduced by ``(1 - heat) ** HEAT_BETA`` near the
    peak, the sum normalised by the number of positives (at least one); plus ``HEAT_OFFSET_WEIGHT``
    times the L1 of the offsets at the peak cells, normalised the same way. The grid is the
    target's own, so the loss does not care which stride the network answers at."""
    rows, cols = int(target.shape[-2]), int(target.shape[-1])
    prediction = prediction.float().reshape(-1, FRAMES_PER_UNIT, 3, rows, cols)
    target = target.reshape_as(prediction)
    heat_true = target[:, :, 0]
    heat_pred = prediction[:, :, 0].clamp(LOSS_EPSILON, 1 - LOSS_EPSILON)
    peak = (heat_true == 1).float()
    positives = peak.sum().clamp(min=1.0)
    positive = peak * (1 - heat_pred).pow(HEAT_ALPHA) * heat_pred.log()
    negative = (1 - peak) * (1 - heat_true).pow(HEAT_BETA) * heat_pred.pow(HEAT_ALPHA) * (1 - heat_pred).log()
    heat_loss = -(positive + negative).sum() / positives
    offset_loss = ((prediction[:, :, 1:3] - target[:, :, 1:3]).abs() * peak.unsqueeze(2)).sum() / positives
    return heat_loss + HEAT_OFFSET_WEIGHT * offset_loss


LOSSES = {GRIDTRACKNET: gridtracknet_loss, HEATMAP_S4: heatmap_loss}


def unit_loss(model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss of `model`'s architecture (``LOSSES`` by ``model.ARCH``) on one batch of units."""
    return LOSSES[model.ARCH](prediction, target)


def batch_loss(model: torch.nn.Module, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss of one batch from the loader: ``unit_loss`` on the model's output."""
    return unit_loss(model, model(images), target)


def save_weights(model: torch.nn.Module, template: Path, destination: Path) -> None:
    """Write the model in the layout the tracker loads for its architecture: a HeatmapNet as the
    record ``heatmapnet.save_model`` writes (`template` is not read);
    a GridTrackNet in the .npz layout, where every convolution and every BatchNorm layer's gamma,
    beta and running mean and variance come from ``model`` and ``template`` supplies whatever else
    the layout carries."""
    if weights_suffix(model.ARCH) == ".pt":
        heatmapnet.save_model(model, destination)
        return
    with np.load(template) as source:
        arrays = {name: source[name] for name in source.files}
    for index, conv in enumerate(model.convs):
        name = "conv2d" if index == 0 else f"conv2d_{index}"
        prefix = f"{name}/{name}"
        arrays[f"{prefix}/kernel:0"] = conv.weight.detach().float().cpu().numpy().transpose(2, 3, 1, 0)
        arrays[f"{prefix}/bias:0"] = conv.bias.detach().float().cpu().numpy()
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        prefix = f"{name}/{name}"
        arrays[f"{prefix}/gamma:0"] = layer.gamma.detach().float().cpu().numpy()
        arrays[f"{prefix}/beta:0"] = layer.beta.detach().float().cpu().numpy()
        arrays[f"{prefix}/moving_mean:0"] = layer.mean.detach().float().cpu().numpy()
        arrays[f"{prefix}/moving_variance:0"] = layer.variance.detach().float().cpu().numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)


def normalised_conv_weights(model: torch.nn.Module) -> List[torch.Tensor]:
    """The weights of the convolutions a ``BatchNorm2d`` follows directly (the two of every
    ``heatmapnet._block``, bias-free): the normalisation divides their scale out, so the network's
    answer does not depend on it and the loss never pushes back when the optimizer lets it grow;
    left alone, the pre-normalisation activations outgrow float16 (heat1, 2026-10-26: the
    decoder's running variance reached infinity at epoch 24 and the eval-mode network went silent
    while training looked fine; heat3, with weight decay 1e-4 on them, climbed the same way at
    ``--lr 1.0``). ``hold_conv_scale`` pins them instead. GridTrackNet's convolutions and the heat
    head are not such."""
    weights = []
    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            layers = list(module)
            for conv, following in zip(layers, layers[1:]):
                if isinstance(conv, torch.nn.Conv2d) and isinstance(following, torch.nn.BatchNorm2d):
                    if conv.bias is not None:        # a bias would not be cancelled, and the hold would change the answer
                        raise ValueError("a convolution BatchNorm follows must be bias-free to be held at a fixed scale")
                    weights.append(conv.weight)
    return weights


FILTER_NORM = math.sqrt(2.0)   # a Kaiming-normal filter's expected norm: fan_in numbers of variance 2 / fan_in


@torch.no_grad()
def hold_conv_scale(weights: List[torch.Tensor]) -> None:
    """Rescale every output channel's filter in each of `weights` (``normalised_conv_weights``) to
    ``FILTER_NORM``, the size it was born with (Hoffer et al. 2018, "Norm matters", bounded weight
    normalisation). BatchNorm divides each channel's scale out again, so the network's answer in
    train mode is unchanged to its eps; what changes is that the activations it normalises stay
    the size of a fresh network's at every step, at any learning rate, and float16's ceiling is
    never approached. (With BatchNorm a filter's effective step shrinks as it grows; held, the
    rate the run asked for is the rate it trains at.) Done after every optimizer step. Weights
    loaded far from this size (heat3's decoder sat 40x above it) are brought to it at the first
    step, and the running statistics follow over the next few dozen steps; an epoch is thousands,
    so nothing scored at an epoch's end sees the gap (a smoke run of a handful of steps would)."""
    for weight in weights:
        norms = weight.flatten(1).norm(dim=1).clamp(min=1e-12)
        weight.mul_((FILTER_NORM / norms).view(-1, 1, 1, 1))


def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Adadelta:
    """The run's Adadelta (rho 0.95, eps 1e-7, upstream's), one parameter group and no weight
    decay: recipe 2's decay on the normalised convolutions held their scale at 0.3 and lost to
    the step at 1.0 (heat3); ``hold_conv_scale`` holds it exactly at either."""
    return torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.95, eps=1e-7)


def check_finite(model: torch.nn.Module) -> None:
    """A RuntimeError naming the first parameter or buffer with a non-finite value. A BatchNorm
    running statistic poisoned by one overflowing batch stays infinite for the rest of the run
    and silences the eval-mode network, which the training loss never shows; an epoch that
    leaves one is stopped here rather than scored at zero for twenty more."""
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} holds a non-finite value after this epoch: the run cannot go on "
                               f"(a BatchNorm running statistic overflowed float16; resume from an earlier epoch's "
                               f"weights with a lower --lr)")


def largest_running_variance(model: torch.nn.Module) -> Tuple[str, float]:
    """The BatchNorm running variance nearest float16's ceiling, (buffer name, value); ("", 0.0)
    for a network without one. The trainer warns when it passes a share of ``HALF_MAX``."""
    largest = ("", 0.0)
    for name, buffer in model.named_buffers():
        if name.endswith("running_var") and buffer.numel():
            value = float(buffer.max())
            if value > largest[1]:
                largest = (name, value)
    return largest


def fresh_model(device: torch.device, arch: str = GRIDTRACKNET) -> torch.nn.Module:
    """A float32 network of `arch` to train from scratch, in train mode: a GridTrackNet with
    Kaiming convolutions and BatchNorm at Keras's defaults (gamma 1, beta 0, running mean 0,
    running variance 1); a HeatmapNet as its own constructor leaves it (Kaiming convolutions,
    the heat head at its prior)."""
    model = model_class(arch)()
    if arch == GRIDTRACKNET:
        for conv in model.convs:
            torch.nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            torch.nn.init.zeros_(conv.bias)
    return model.to(device).float()


def evaluate_loss(model, loader, device, max_steps=None) -> float:
    """The mean loss over the loader in eval mode (the running BatchNorm statistics, which
    stay as they are); the model's mode is restored afterwards."""
    was_training = model.training
    model.eval()
    total, steps = 0.0, 0
    try:
        with torch.inference_mode():
            for images, target in loader:
                images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = batch_loss(model, images, target)
                total += float(loss)
                steps += 1
                if max_steps and steps >= max_steps:
                    break
    finally:
        model.train(was_training)
    return total / max(steps, 1)


def train_epoch(model, loader, optimizer, scaler, device: torch.device, epoch: int, epochs: int,
                max_steps: Optional[int]) -> float:
    """One pass over the training loader; returns the mean loss. The normalised convolutions are
    held at ``FILTER_NORM`` per filter before the first step and after every step (``hold_conv_scale``;
    nothing to hold in a GridTrackNet)."""
    model.train()
    held = normalised_conv_weights(model)
    hold_conv_scale(held)
    total, steps = 0.0, 0
    started = time.perf_counter()
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = batch_loss(model, images, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        hold_conv_scale(held)
        total += float(loss.detach())
        steps += 1
        if steps == 1 or steps % PROGRESS_EVERY == 0:
            rate = steps / max(time.perf_counter() - started, 1e-6)
            remaining = (len(loader) - steps) / max(rate, 1e-6)
            print(f"[train] epoch {epoch}/{epochs} step {steps}/{len(loader)} "
                  f"loss={total / steps:.6f} {rate:.1f} it/s eta {remaining / 60:.0f} min", flush=True)
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)
