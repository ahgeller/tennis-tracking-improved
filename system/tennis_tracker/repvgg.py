"""The building blocks the fast ball detectors share: RepVGG blocks and folding BatchNorm into
the convolution before it, so a network trained with branches and BatchNorm runs as plain biased
3x3 convolutions and ReLUs.

A ``RepVGGBlock`` (Ding et al. 2021, "RepVGG: Making VGG-style ConvNets Great Again", arXiv
2101.03697) trains three parallel branches, a 3x3 convolution with BatchNorm, a 1x1 convolution
with BatchNorm and, when the shape allows, a BatchNorm on the identity, sums them and applies a
ReLU. Every branch is linear in eval mode, so ``reparameterize`` folds each into one 3x3 kernel
and bias and sums them: the deployed block is a single convolution with the same answer to
floating-point rounding (``fuse_for_inference`` does that for every block of a network, then
folds every remaining convolution-BatchNorm pair). TensorRT folds a plain convolution-BatchNorm
pair on its own but cannot merge parallel branches, so the multi-branch graph must be fused before
ONNX export. The original RepVGG is chosen over QARepVGG (Chu et al. 2022, arXiv 2212.01593):
QARepVGG exists to make the fused kernel quantise to INT8, and the tracker runs FP16 (the plan
leaves INT8 out on sm_120); RepVGG's every branch is a bias-free convolution followed by BatchNorm
inside an ``nn.Sequential``, the pattern the trainer's ``loop.hold_conv_scale`` finds and holds
at its Kaiming norm, which keeps FP16 training finite (the ledger's heat1 and heat3 runs).
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
from torch import nn


def conv_bn(in_channels: int, out_channels: int, kernel: int, stride: int = 1) -> nn.Sequential:
    """A bias-free `kernel` x `kernel` convolution ("same" padding) and BatchNorm, as one
    ``nn.Sequential``: the pair ``loop.hold_conv_scale`` holds and ``fold_batch_norms`` folds."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def conv_bn_relu(in_channels: int, out_channels: int, kernel: int, stride: int = 1) -> nn.Sequential:
    """``conv_bn`` followed by a ReLU, in the same ``nn.Sequential``."""
    return nn.Sequential(*conv_bn(in_channels, out_channels, kernel, stride), nn.ReLU(inplace=True))


def _affine(norm: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """The per-channel scale and shift `norm` applies in eval mode, in float64:
    ``gamma / sqrt(var + eps)`` and ``beta - mean * gamma / sqrt(var + eps)``."""
    mean, variance = norm.running_mean, norm.running_var
    if mean is None or variance is None:
        raise ValueError("a BatchNorm without running statistics cannot be folded")
    scale = norm.weight.double() / torch.sqrt(variance.double() + norm.eps)
    return scale, norm.bias.double() - mean.double() * scale


def folded(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """The kernel and bias of one convolution that answers as `conv` followed by `norm` in eval
    mode: ``w * gamma / sqrt(var + eps)`` and ``beta + (b - mean) * gamma / sqrt(var + eps)``,
    computed in float64 so the fold adds no rounding of its own, returned in `conv`'s dtype."""
    scale, shift = _affine(norm)
    weight = conv.weight.double() * scale.view(-1, 1, 1, 1)
    bias = shift if conv.bias is None else shift + conv.bias.double() * scale
    return weight.to(conv.weight.dtype), bias.to(conv.weight.dtype)


def _with_kernel(conv: nn.Conv2d, weight: torch.Tensor, bias: torch.Tensor) -> nn.Conv2d:
    """A convolution of `conv`'s geometry (stride, padding, dilation, groups) with `weight` and
    a `bias`."""
    replacement = copy.deepcopy(conv)
    replacement.weight = nn.Parameter(weight.detach().clone())
    replacement.bias = nn.Parameter(bias.detach().clone())
    return replacement


def _pair(branch: nn.Module) -> Tuple[nn.Conv2d, nn.BatchNorm2d]:
    """The convolution and BatchNorm of a ``conv_bn`` branch."""
    conv, norm = list(branch.children())
    assert isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d)
    return conv, norm


class RepVGGBlock(nn.Module):
    """A 3x3 convolution of `stride` from `in_channels` to `out_channels` with a ReLU, trained as
    the three RepVGG branches (``dense``: 3x3 and BatchNorm; ``point``: 1x1 and BatchNorm;
    ``identity``: BatchNorm alone, present only when the shape is kept) and deployed as the one
    convolution ``reparameterize`` folds them into (``fused``)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dense: Optional[nn.Sequential] = conv_bn(in_channels, out_channels, 3, stride)
        self.point: Optional[nn.Sequential] = conv_bn(in_channels, out_channels, 1, stride)
        self.identity: Optional[nn.BatchNorm2d] = (
            nn.BatchNorm2d(out_channels) if in_channels == out_channels and stride == 1 else None)
        self.fused: Optional[nn.Conv2d] = None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.activation(self.fused(value))
        assert self.dense is not None and self.point is not None
        total = self.dense(value) + self.point(value)
        if self.identity is not None:
            total = total + self.identity(value)
        return self.activation(total)

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Replace the branches by ``fused``, the one 3x3 convolution with bias that answers as
        their sum in eval mode: the 1x1 kernel sits at the 3x3's centre, the identity is a 3x3
        kernel with a one at the centre of each channel's own filter. Idempotent."""
        if self.fused is not None:
            return
        assert self.dense is not None and self.point is not None
        conv, norm = _pair(self.dense)
        weight, bias = folded(conv, norm)
        point_weight, point_bias = folded(*_pair(self.point))
        weight = weight + nn.functional.pad(point_weight, [1, 1, 1, 1])
        bias = bias + point_bias
        if self.identity is not None:
            scale, shift = _affine(self.identity)
            channels = torch.arange(conv.out_channels)
            eye = torch.zeros_like(weight)
            eye[channels, channels, 1, 1] = scale.to(weight.dtype)
            weight, bias = weight + eye, bias + shift.to(bias.dtype)
        self.fused = _with_kernel(conv, weight, bias)
        self.dense = self.point = self.identity = None


@torch.no_grad()
def fold_batch_norms(model: nn.Module) -> None:
    """Fold every BatchNorm that directly follows a convolution inside an ``nn.Sequential`` of
    `model` into that convolution (which gains a bias), leaving an ``nn.Identity`` in the
    BatchNorm's place, so the module paths of everything else are unchanged."""
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        for index in range(len(module) - 1):
            conv, norm = module[index], module[index + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                module[index], module[index + 1] = _with_kernel(conv, *folded(conv, norm)), nn.Identity()


def fuse_for_inference(model: nn.Module) -> nn.Module:
    """`model` in its deployed form, in place: every ``RepVGGBlock`` reparameterised and every
    remaining convolution-BatchNorm pair folded (``fold_batch_norms``), put in eval mode (the
    folds read the running statistics, so the network can no longer train). Returns `model`."""
    model.eval()
    for module in list(model.modules()):
        if isinstance(module, RepVGGBlock):
            module.reparameterize()
    fold_batch_norms(model)
    return model
