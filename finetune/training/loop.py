"""The training loop proper: the loss of each architecture, a fresh network, one epoch of
optimisation, the validation loss and writing the weights back in the layout the tracker loads
(the .npz for GridTrackNet, the .pt record for HeatmapNet).

Which loss and which layout is read off the model's ``ARCH`` (the contract ``ball_models``
spells), so the loop holds no architecture switch of its own: ``loss_of`` maps the id to the
loss (GridTrackNet's own, the heat-map loss for every other architecture: they all answer on a
Gaussian target), ``save_weights`` to the writer.

A batch is ``(images, target)``, or ``(images, target, valid)`` and ``(images, target, valid,
teacher)`` when its dataset carries frames without a label (the ignored frames of pseudo-labelled
clips) or a precomputed teacher answer (``dataset.UnitDataset``): ``valid`` is one weight per
frame, 0 for a frame the labelled loss must not read, and ``teacher`` the teacher's 15-channel
answer. With a ``distill.Distiller`` an epoch adds its loss (the teacher's soft maps and
features) to the labelled one; without it the loop is what it always was.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tennis_tracker import heatmapnet
from tennis_tracker.ball_models import ARCHITECTURES, GRIDTRACKNET, model_class, weights_suffix
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS

PROGRESS_EVERY = 200       # training steps between progress lines
LOSS_EPSILON = 1e-6        # keeps the confidence loss's logs finite
OFFSET_LOSS_WEIGHT = 0.001     # GridTrackNet's, as upstream trains it
HEAT_OFFSET_WEIGHT = 1.0       # the heat-map loss's, CenterNet's lambda_off: at 0.001 the offsets never trained
                               # (heat1, 2026-10-26: 3.4 px median at 1080p, a quarter of a cell at random)
HALF_MAX = 65504.0               # float16's largest value; the prepass and the training forward run in it
POSITIVE_WEIGHT, NEGATIVE_WEIGHT = 0.75, 0.25    # the focal loss's class weights
HEAT_ALPHA, HEAT_BETA = 2.0, 4.0                 # CenterNet's focal exponents: the easy-example one and the penalty reduction


def frame_weights(valid: Optional[torch.Tensor], units: int, like: torch.Tensor) -> torch.Tensor:
    """The (units, FRAMES_PER_UNIT, 1, 1) weight of every frame's loss terms: `valid` (one weight
    per frame, 0: no label) reshaped, or ones when the batch carries none."""
    if valid is None:
        return torch.ones(units, FRAMES_PER_UNIT, 1, 1, dtype=like.dtype, device=like.device)
    return valid.to(like.device, like.dtype).reshape(units, FRAMES_PER_UNIT, 1, 1)


def gridtracknet_loss(prediction: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """GridTrackNet's loss, as the upstream trainer has it: a class-weighted focal loss on the
    one-hot confidence grid (``POSITIVE_WEIGHT`` / ``NEGATIVE_WEIGHT``, exponent 2), averaged over
    every cell, plus ``OFFSET_LOSS_WEIGHT`` times the L1 of the offsets at the labelled cells,
    averaged over frames. `valid` leaves an unlabelled frame's terms out and the confidence average
    is then over the cells of the frames that count, so a batch without one scores as before."""
    prediction = prediction.float().reshape(-1, FRAMES_PER_UNIT, 3, GRID_ROWS, GRID_COLS)
    target = target.reshape_as(prediction)
    weight = frame_weights(valid, prediction.shape[0], prediction)
    confidence_true = target[:, :, 0]
    confidence_pred = prediction[:, :, 0].clamp(LOSS_EPSILON, 1 - LOSS_EPSILON)
    positive = POSITIVE_WEIGHT * confidence_true * (1 - confidence_pred).square() * confidence_pred.log()
    negative = NEGATIVE_WEIGHT * (1 - confidence_true) * confidence_pred.square() * (1 - confidence_pred).log()
    confidence_loss = -((positive + negative) * weight).sum() / (weight.sum().clamp(min=1.0) * GRID_ROWS * GRID_COLS)
    offsets = ((prediction[:, :, 1:3] - target[:, :, 1:3]).abs() * confidence_true.unsqueeze(2)).sum(dim=(2, 3, 4))
    offset_loss = (offsets * weight.flatten(1)).sum() / offsets.numel()
    return confidence_loss + OFFSET_LOSS_WEIGHT * offset_loss


def heatmap_loss(prediction: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """HeatmapNet's loss: CenterNet's penalty-reduced focal loss on the heat channel against
    the Gaussian target (``dataset.make_target``), where a cell at exactly 1 is a positive and
    every other cell a negative whose penalty is reduced by ``(1 - heat) ** HEAT_BETA`` near the
    peak, the sum normalised by the number of positives (at least one); plus ``HEAT_OFFSET_WEIGHT``
    times the L1 of the offsets at the peak cells, normalised the same way. The grid is the
    target's own, so the loss does not care which stride the network answers at. `valid` (one
    weight per frame, 0: no label, the ignored frames of a pseudo-labelled clip) leaves a frame
    out of both sums and of the positive count: such a frame is neither a ball nor a negative."""
    rows, cols = int(target.shape[-2]), int(target.shape[-1])
    prediction = prediction.float().reshape(-1, FRAMES_PER_UNIT, 3, rows, cols)
    target = target.reshape_as(prediction)
    weight = frame_weights(valid, prediction.shape[0], prediction)
    heat_true = target[:, :, 0]
    heat_pred = prediction[:, :, 0].clamp(LOSS_EPSILON, 1 - LOSS_EPSILON)
    peak = (heat_true == 1).float() * weight
    positives = peak.sum().clamp(min=1.0)
    positive = peak * (1 - heat_pred).pow(HEAT_ALPHA) * heat_pred.log()
    negative = weight * (1 - (heat_true == 1).float()) * (1 - heat_true).pow(HEAT_BETA) * heat_pred.pow(HEAT_ALPHA) \
        * (1 - heat_pred).log()
    heat_loss = -(positive + negative).sum() / positives
    offset_loss = ((prediction[:, :, 1:3] - target[:, :, 1:3]).abs() * peak.unsqueeze(2)).sum() / positives
    return heat_loss + HEAT_OFFSET_WEIGHT * offset_loss


# Every architecture but GridTrackNet answers on a heat-map grid and trains with the heat-map loss.
LOSSES = {arch: gridtracknet_loss if arch == GRIDTRACKNET else heatmap_loss for arch in ARCHITECTURES}


def loss_of(arch: str) -> Callable[..., torch.Tensor]:
    """The labelled loss of an architecture: its own in ``LOSSES``, else the heat-map loss, which
    every architecture after GridTrackNet answers on (``dataset.make_target`` draws the Gaussian
    target for every id but GridTrackNet's)."""
    return LOSSES.get(arch, heatmap_loss)


def unit_loss(model: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor,
              valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The loss of `model`'s architecture (``loss_of`` its ``ARCH``) on one batch of units."""
    return loss_of(model.ARCH)(prediction, target, valid)


def batch_loss(model: torch.nn.Module, images: torch.Tensor, target: torch.Tensor,
               valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The loss of one batch from the loader: ``unit_loss`` on the model's output."""
    return unit_loss(model, model(images), target, valid)


def unpack(batch: Sequence[torch.Tensor], device: torch.device
           ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """One loader batch moved to `device` as (images, target, valid or None, teacher or None): the
    two items of a plain dataset's batch and the longer batches the module docstring names."""
    moved = [item.to(device, non_blocking=True) for item in batch]
    return moved[0], moved[1], moved[2] if len(moved) > 2 else None, moved[3] if len(moved) > 3 else None


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


def build_optimizer(model: torch.nn.Module, lr: float, extra: Sequence[torch.nn.Parameter] = ()) -> torch.optim.Adadelta:
    """The run's Adadelta (rho 0.95, eps 1e-7, upstream's), one parameter group and no weight
    decay: recipe 2's decay on the normalised convolutions held their scale at 0.3 and lost to
    the step at 1.0 (heat3); ``hold_conv_scale`` holds it exactly at either. `extra` parameters
    train beside the model in the same group (a ``distill.Distiller``'s feature adapters)."""
    return torch.optim.Adadelta([*model.parameters(), *extra], lr=lr, rho=0.95, eps=1e-7)


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
            for batch in loader:
                images, target, valid, _ = unpack(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = batch_loss(model, images, target, valid)
                total += float(loss)
                steps += 1
                if max_steps and steps >= max_steps:
                    break
    finally:
        model.train(was_training)
    return total / max(steps, 1)


def train_epoch(model, loader, optimizer, scaler, device: torch.device, epoch: int, epochs: int,
                max_steps: Optional[int], distiller: Any = None) -> float:
    """One pass over the training loader; returns the mean loss. The normalised convolutions are
    held at ``FILTER_NORM`` per filter before the first step and after every step (``hold_conv_scale``;
    nothing to hold in a GridTrackNet). With a `distiller` (``distill.Distiller``) every step adds
    its loss on the same output to the labelled one, and the progress lines name its parts."""
    model.train()
    held = normalised_conv_weights(model)
    hold_conv_scale(held)
    total, steps = 0.0, 0
    parts: Dict[str, float] = {}
    started = time.perf_counter()
    for batch in loader:
        images, target, valid, teacher = unpack(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(images)
            loss = unit_loss(model, prediction, target, valid)
            if distiller is not None:
                extra, terms = distiller.loss(prediction, images, teacher)
                loss = loss + extra
                for name, value in terms.items():
                    parts[name] = parts.get(name, 0.0) + value
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        hold_conv_scale(held)
        total += float(loss.detach())
        steps += 1
        if steps == 1 or steps % PROGRESS_EVERY == 0:
            rate = steps / max(time.perf_counter() - started, 1e-6)
            remaining = (len(loader) - steps) / max(rate, 1e-6)
            named = "".join(f" {name}={value / steps:.4f}" for name, value in sorted(parts.items()))
            print(f"[train] epoch {epoch}/{epochs} step {steps}/{len(loader)} "
                  f"loss={total / steps:.6f}{named} {rate:.1f} it/s eta {remaining / 60:.0f} min", flush=True)
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)
