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

Two more parts, each off unless asked for (the trainer's ``--ema`` and ``--vis-head``):

- ``EMA``, an exponential moving average of the weights (decay ``EMA_DECAY`` per optimizer step,
  BatchNorm buffers copied). Why: on the mixed_v1 held-out set the recall of one stream_s4 run
  moves by 0.013 (one standard deviation) between its late epochs, while the models compared
  differ by less, so the epoch ``best.pt`` picks is a lottery ticket as much as a measurement.
  An average over the last thousand or so steps (0.999 ** 1000 = 0.37) is what the run converged
  to rather than where its last batch left it, and is free on the phone. The trainer scores the
  averaged network beside the live one on every scored epoch, so the run itself says whether the
  average matches the best epoch with less spread (the research report's kill test for F).
- ``visibility_loss``, binary cross-entropy of a network's per-frame visibility logits
  (``streamnet.StreamNet`` with ``vis_head``) against the labels: a frame whose target holds a
  peak is visible, an absent-ball frame is the negative, and a frame without a label (``valid``
  0) is neither. ``evaluate`` reports the head's AUC and balanced accuracy on the held-out set.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

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
VIS_LOSS_WEIGHT = 0.1      # the visibility loss's default weight beside the heat loss (--vis-weight): a tenth, so the
                           # head learns without steering the features the heat map is read from
EMA_DECAY = 0.999          # the weight average's decay per optimizer step (--ema-decay): about the last 1000 steps


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


def visible_frames(target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The visibility labels of a batch's frames and their weights, both ``(units, FRAMES_PER_UNIT)``
    float: a frame is visible (1) when its heat channel in `target` holds a peak cell at exactly 1
    (``dataset.make_target`` writes one for every ball, a pasted one too) and absent (0) otherwise;
    the weight is `valid`'s (0: no label, the frame is neither) or one."""
    rows, cols = int(target.shape[-2]), int(target.shape[-1])
    heat = target.reshape(-1, FRAMES_PER_UNIT, 3, rows, cols)[:, :, 0]
    labels = (heat.flatten(2).amax(dim=2) == 1).float()
    return labels, frame_weights(valid, labels.shape[0], labels).reshape(labels.shape)


def visibility_loss(logits: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The visibility head's loss: binary cross-entropy of the per-frame `logits`
    (``(units, FRAMES_PER_UNIT)``, before the sigmoid) against ``visible_frames`` of `target`,
    averaged over the frames that carry a label (`valid`); a batch without one scores 0."""
    labels, weight = visible_frames(target, valid)
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    return (entropy * weight).sum() / weight.sum().clamp(min=1.0)


def with_vis_head(model: torch.nn.Module) -> torch.nn.Module:
    """`model` with its visibility head enabled (``StreamNet.enable_vis_head``); an architecture
    without one is a ValueError, the one ``--vis-head`` reports."""
    enable = getattr(model, "enable_vis_head", None)
    if not callable(enable):
        raise ValueError(f"--vis-head: {model.ARCH} has no visibility head (stream_s4 has)")
    enable()
    return model


def has_vis_head(model: torch.nn.Module) -> bool:
    """Whether `model` answers visibility logits (``forward(images, return_vis=True)``)."""
    return bool(getattr(model, "has_vis_head", False))


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


def fresh_model(device: torch.device, arch: str = GRIDTRACKNET, vis_head: bool = False) -> torch.nn.Module:
    """A float32 network of `arch` to train from scratch, in train mode: a GridTrackNet with
    Kaiming convolutions and BatchNorm at Keras's defaults (gamma 1, beta 0, running mean 0,
    running variance 1); a HeatmapNet as its own constructor leaves it (Kaiming convolutions,
    the heat head at its prior). With `vis_head` the network carries its visibility head
    (``with_vis_head``)."""
    model = model_class(arch)()
    if arch == GRIDTRACKNET:
        for conv in model.convs:
            torch.nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            torch.nn.init.zeros_(conv.bias)
    if vis_head:
        with_vis_head(model)
    return model.to(device).float()


class EMA:
    """An exponential moving average of a network's weights (the module docstring says why): a
    shadow copy of `model` in eval mode, moved toward the live weights by ``1 - decay`` after
    every optimizer step (``update``), its buffers (the BatchNorm running statistics) copied
    from the live network, and its normalised convolutions held at ``FILTER_NORM`` like the live
    ones (an average of filters at that norm is shorter; held, the copied running statistics
    still match its activations). ``model`` is scored and saved like any network;
    ``state_dict`` and ``load_state_dict`` carry it through the trainer's checkpoint. The copy
    is made before any forward hook is put on the live network (``distill.FeatureTap``), so
    the shadow's forwards write into nobody's taps."""

    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"--ema-decay is the weight average's decay per step, between 0 and 1, not {decay:g}")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.held = normalised_conv_weights(self.model)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        shadow: List[torch.Tensor] = [parameter.data for parameter in self.model.parameters()]
        live: List[torch.Tensor] = [parameter.detach() for parameter in model.parameters()]
        torch._foreach_mul_(shadow, self.decay)
        torch._foreach_add_(shadow, live, alpha=1.0 - self.decay)
        for mine, theirs in zip(self.model.buffers(), model.buffers()):
            mine.copy_(theirs)
        hold_conv_scale(self.held)
        self.updates += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"model": self.model.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.updates = int(state["updates"])


class Validation(NamedTuple):
    """One pass over the held-out loader (``evaluate``): the mean loss and, when the visibility
    head was asked for, its AUC and balanced accuracy over the labelled frames (None when the
    head was not evaluated, or when the frames seen were all visible or all absent, which no
    ranking can be measured on)."""
    loss: float
    vis_auc: Optional[float] = None
    vis_balanced_accuracy: Optional[float] = None


def visibility_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    """(AUC, balanced accuracy) of the visibility `logits` against the 0/1 `labels`, both flat and
    of the labelled frames only. The AUC is the Mann-Whitney statistic (the share of visible-absent
    pairs ranked the right way, ties at half), the balanced accuracy the mean of the recall of each
    class at a logit of 0 (probability 0.5). Both None without a frame of each class."""
    scores, labels = logits.detach().float().cpu(), labels.detach().float().cpu()
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if not positives or not negatives:
        return None, None
    order = scores.argsort()
    _, inverse, counts = torch.unique(scores[order], return_inverse=True, return_counts=True)
    first = (counts.cumsum(0) - counts + 1).double()
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = (first + (counts.double() - 1) / 2)[inverse]
    auc = (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
    predicted = (scores > 0).float()
    balanced = (float((predicted[labels == 1] == 1).float().mean()) + float((predicted[labels == 0] == 0).float().mean())) / 2
    return auc, balanced


def evaluate(model, loader, device, max_steps=None, visibility: bool = False) -> Validation:
    """One pass over the loader in eval mode (the running BatchNorm statistics, which stay as they
    are): the mean labelled loss and, with `visibility` (the model has the head), the head's
    metrics over every labelled frame seen (``visibility_metrics``); the model's mode is restored
    afterwards."""
    was_training = model.training
    model.eval()
    total, steps = 0.0, 0
    logits: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                images, target, valid, _ = unpack(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    if visibility:
                        prediction, vis = model(images, return_vis=True)
                        visible, weight = visible_frames(target, valid)
                        logits.append(vis.flatten()[weight.flatten() > 0])
                        labels.append(visible.flatten()[weight.flatten() > 0])
                    else:
                        prediction = model(images)
                    loss = unit_loss(model, prediction, target, valid)
                total += float(loss)
                steps += 1
                if max_steps and steps >= max_steps:
                    break
    finally:
        model.train(was_training)
    mean = total / max(steps, 1)
    if not visibility:
        return Validation(mean)
    return Validation(mean, *visibility_metrics(torch.cat(logits), torch.cat(labels)))


def evaluate_loss(model, loader, device, max_steps=None) -> float:
    """The mean loss over the loader in eval mode: ``evaluate`` without the visibility metrics."""
    return evaluate(model, loader, device, max_steps).loss


def train_epoch(model, loader, optimizer, scaler, device: torch.device, epoch: int, epochs: int,
                max_steps: Optional[int], distiller: Any = None, ema: Optional[EMA] = None,
                vis_weight: float = 0.0) -> float:
    """One pass over the training loader; returns the mean loss. The normalised convolutions are
    held at ``FILTER_NORM`` per filter before the first step and after every step (``hold_conv_scale``;
    nothing to hold in a GridTrackNet). With a `distiller` (``distill.Distiller``) every step adds
    its loss on the same output to the labelled one, and the progress lines name its parts; with
    `vis_weight` above 0 on a network with the visibility head, `vis_weight` times
    ``visibility_loss`` of its logits (part ``vis``); an `ema` is updated after every step."""
    model.train()
    held = normalised_conv_weights(model)
    hold_conv_scale(held)
    visibility = vis_weight > 0 and has_vis_head(model)
    total, steps = 0.0, 0
    parts: Dict[str, float] = {}
    started = time.perf_counter()
    for batch in loader:
        images, target, valid, teacher = unpack(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if visibility:
                prediction, logits = model(images, return_vis=True)
            else:
                prediction = model(images)
            loss = unit_loss(model, prediction, target, valid)
            if visibility:
                vis = visibility_loss(logits, target, valid)
                loss = loss + vis_weight * vis
                parts["vis"] = parts.get("vis", 0.0) + float(vis.detach())
            if distiller is not None:
                extra, terms = distiller.loss(prediction, images, teacher)
                loss = loss + extra
                for name, value in terms.items():
                    parts[name] = parts.get(name, 0.0) + value
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        hold_conv_scale(held)
        if ema is not None:
            ema.update(model)
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
