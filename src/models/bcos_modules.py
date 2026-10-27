"""
B-cos Neural Network Core Modules
Based on Böhle et al., "B-cos Networks: Alignment is All You Need for Interpretability" (CVPR 2022)
Paper: https://arxiv.org/abs/2205.10268
Reference: https://github.com/moboehle/B-cos
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormedConv2d(nn.Conv2d):
    """
    Standard 2D convolution with unit-normalized filter weights.
    W_norm = W / ||W||_2 ensures filter outputs measure cosine alignment.
    """
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        shape = self.weight.shape
        w = self.weight.view(shape[0], -1)
        w = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return F.conv2d(
            in_tensor,
            w.view(shape),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BcosConv2d(nn.Module):
    """
    B-cos 2D Convolution Layer.
    Computes: out = ||x_patch|| * ||w|| * |cos(x_patch, w)|^B * sgn(cos(x_patch, w)) / scale
    
    When explanation_mode(detach=True) is active:
    The non-linear cosine magnitude term is detached from backprop, so:
    d(out) / d(x) exactly equals the dynamic linear transformation W_{1->L}(x).
    """
    def __init__(
        self,
        inc: int,
        outc: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        max_out: int = 2,
        b: int = 2,
        scale: float = None,
        scale_fact: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        ks = kernel_size
        self.stride = stride
        self.linear = NormedConv2d(
            inc, outc * max_out, ks, stride=stride, padding=padding, bias=False
        )
        self.outc = outc * max_out
        self.b = b
        self.max_out = max_out
        self.inc = inc
        self.kernel_size = ks
        self.kssq = ks**2 if not isinstance(ks, tuple) else int(np.prod(ks))
        self.padding = padding
        self.detach = False

        if scale is None:
            ks_scale = ks if not isinstance(ks, tuple) else np.sqrt(np.prod(ks))
            self.scale = float((ks_scale * np.sqrt(self.inc)) / scale_fact)
        else:
            self.scale = float(scale)

    def explanation_mode(self, detach: bool = True):
        """Toggle explanation mode to compute exact linear attributions."""
        self.detach = detach

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        if self.b == 2:
            return self._fwd_2(in_tensor)
        return self._fwd_b(in_tensor)

    def _fwd_2(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out = self.linear(in_tensor)
        if self.max_out > 1:
            bs, _, h, w = out.shape
            out = out.view(bs, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]

        norm = (
            F.avg_pool2d(
                (in_tensor**2).sum(1, keepdim=True),
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )
            * self.kssq
            + 1e-6
        ).sqrt_()

        if self.detach:
            out = out * out.abs().detach()
            norm = norm.detach()
        else:
            out = out * out.abs()

        return out / (norm * self.scale)

    def _fwd_b(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out = self.linear(in_tensor)
        if self.max_out > 1:
            bs, _, h, w = out.shape
            out = out.view(bs, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]

        if self.b == 1:
            return out / self.scale

        norm = (
            F.avg_pool2d(
                (in_tensor**2).sum(1, keepdim=True),
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )
            * self.kssq
            + 1e-6
        ).sqrt_()

        abs_cos = (out / norm).abs() + 1e-6
        if self.detach:
            abs_cos = abs_cos.detach()

        out = out * abs_cos.pow(self.b - 1)
        return out / self.scale


class AddInverse(nn.Module):
    """
    Transforms C-channel input into 2C-channel input by appending (1 - x).
    This ensures all spatial locations have equal L1 norm and removes color/sign bias.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([in_tensor, 1.0 - in_tensor], dim=self.dim)


def set_explanation_mode(model: nn.Module, active: bool = True):
    """Recursively set explanation mode for all B-cos modules in a model."""
    for mod in model.modules():
        if isinstance(mod, BcosConv2d):
            mod.explanation_mode(active)
