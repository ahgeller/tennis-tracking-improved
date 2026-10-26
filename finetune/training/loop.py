"""The training loop proper: the loss of each architecture, a fresh network, one epoch of
optimisation, the validation loss and writing the weights back in the layout the tracker loads
(the .npz for GridTrackNet, the .pt record for HeatmapNet and MemoryNet).

Which loss and which layout is read off the model's ``ARCH`` (the contract ``ball_models``
spells), so the loop holds no architecture switch of its own: ``LOSSES`` maps the id to the
loss, ``save_weights`` to the writer. A ``STATEFUL`` model (MemoryNet) is fed chains: a batch is
``(B, K, 15, H, W)`` with a ``(B, K, 15, rows, cols)`` target, ``batch_loss`` forwards through the
K units carrying the state (None at the chain's start) and averages the per-unit losses, and the
backward pass runs through the whole chain (truncated at K).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tennis_tracker import heatmapnet
from tennis_tracker.ball_models import GRIDTRACKNET, HEATMAP_S4, MEMORY_S4, model_class, weights_suffix
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS

PROGRESS_EVERY = 200       # training steps between progress lines
LOSS_EPSILON = 1e-6        # keeps the confidence loss's logs finite
OFFSET_LOSS_WEIGHT = 0.001
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
    """HeatmapNet's (and MemoryNet's) loss: CenterNet's penalty-reduced focal loss on the heat
    channel against the Gaussian target (``dataset.make_target``), where a cell at exactly 1 is a
    positive and every other cell a negative whose penalty is reduced by ``(1 - heat) ** HEAT_BETA``
    near the peak, the sum normalised by the number of positives (at least one); plus
    ``OFFSET_LOSS_WEIGHT`` times the L1 of the offsets at the peak cells, normalised the same way.
    The grid is the target's own, so the loss does not care which stride the network answers at."""
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
    return heat_loss + OFFSET_LOSS_WEIGHT * offset_loss


LOSSES = {GRIDTRACKNET: gridtracknet_loss, HEATMAP_S4: heatmap_loss, MEMORY_S4: heatmap_loss}


def unit_loss(model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss of `model`'s architecture (``LOSSES`` by ``model.ARCH``) on one batch of units."""
    return LOSSES[model.ARCH](prediction, target)


def chain_loss(model: torch.nn.Module, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A ``STATEFUL`` model forwarded through a batch of chains, ``(B, K, 15, H, W)`` images and
    ``(B, K, 15, rows, cols)`` targets: the state starts at None and is carried from unit to unit,
    and the loss is the mean of the K per-unit losses, so the gradient reaches every unit's
    answer and, through the states, every unit before it."""
    state, losses = None, []
    for step in range(images.shape[1]):
        output, state = model(images[:, step], state)
        losses.append(unit_loss(model, output, target[:, step]))
    return torch.stack(losses).mean()


def batch_loss(model: torch.nn.Module, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss of one batch from the loader: ``unit_loss`` on the model's output for a stateless
    model, ``chain_loss`` for a ``STATEFUL`` one."""
    if model.STATEFUL:
        return chain_loss(model, images, target)
    return unit_loss(model, model(images), target)


def save_weights(model: torch.nn.Module, template: Path, destination: Path) -> None:
    """Write the model in the layout the tracker loads for its architecture: a ``.pt`` architecture
    (HeatmapNet, MemoryNet) as the record ``heatmapnet.save_model`` writes (`template` is not read);
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


def fresh_model(device: torch.device, arch: str = GRIDTRACKNET) -> torch.nn.Module:
    """A float32 network of `arch` to train from scratch, in train mode: a GridTrackNet with
    Kaiming convolutions and BatchNorm at Keras's defaults (gamma 1, beta 0, running mean 0,
    running variance 1); a HeatmapNet or MemoryNet as its own constructor leaves it (Kaiming
    convolutions, the heat head at its prior, the memory's ``out`` at zero)."""
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
    """One pass over the training loader; returns the mean loss."""
    model.train()
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
