"""All models compared in v2.

B-cos models use the official B-cos v2 implementation
(https://github.com/B-cos/B-cos-v2; Boehle et al., CVPR 2022 / TPAMI 2024).
Every B-cos variant here stays *dynamic linear* end to end:

    logit_1(x) - logit_0(x) = sum_{c,h,w}  e[c,h,w] * W(e)[c,h,w]

where e = [x, 1 - x] is the encoded input. There are no learnable biases; the
only constant (the logit-layer bias) is identical for both classes, so it
cancels in the logit difference we explain. `contributions()` checks this.

Variants
--------
cnn               standard ResNet-18 (black box), trained from scratch
bcos              B-cos ResNet-18 from scratch, any number of input bands
bcos_pre_rgb      ImageNet-pretrained B-cos ResNet-18, RGB only (reference)
bcos_pre_inflate  pretrained, RGB+NIR: first layer inflated with NIR filters
                  initialised from the mean of the RGB filters
bcos_pre_project  pretrained, RGB+NIR: learned 4->3 band projection whose rows
                  sum to one, so [Px, 1-Px] = [Px, P(1-x)] stays exactly linear
bcos_pre_ndvi     pretrained RGB trunk + NDVI branch added after layer1.
                  NDVI is a ratio, i.e. NOT linear in the pixels, so the model
                  takes NDVI as a separate input and its contribution is
                  reported separately from the pixel contributions.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PRETRAINED = {"bcos_pre_rgb", "bcos_pre_inflate", "bcos_pre_project", "bcos_pre_ndvi"}
VARIANT_BANDS = {
    "bcos_pre_rgb": ["red", "green", "blue"],
    "bcos_pre_inflate": ["red", "green", "blue", "nir"],
    "bcos_pre_project": ["red", "green", "blue", "nir"],
    "bcos_pre_ndvi": ["red", "green", "blue"],
}


# --------------------------------------------------------------------------- helpers
def _bcos():
    try:
        import bcos  # noqa: F401
        from bcos.common import BcosUtilMixin
        from bcos.modules import BcosConv2d, LogitLayer
        return BcosUtilMixin, BcosConv2d, LogitLayer
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the official B-cos package: "
                          "pip install git+https://github.com/B-cos/B-cos-v2.git") from e


BcosUtilMixin, BcosConv2d, LogitLayer = _bcos()


class SumToOneProjection(nn.Module):
    """Learned linear map from n_in bands to n_out bands with rows summing to one.

    Initialised as the identity on the first n_out bands (pure RGB), so training
    starts exactly where the pretrained model is, and NIR is mixed in only if it
    helps. Because each row sums to one, the projection commutes with the B-cos
    input encoding: P(1 - x) = 1 - Px.
    """

    def __init__(self, n_in: int = 4, n_out: int = 3):
        super().__init__()
        eye = torch.zeros(n_out, n_in)
        eye[:, :n_out] = torch.eye(n_out)
        self.register_buffer("eye", eye)
        self.delta = nn.Parameter(torch.zeros(n_out, n_in))

    @property
    def matrix(self) -> torch.Tensor:
        return self.eye + self.delta - self.delta.mean(dim=1, keepdim=True)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:   # enc = [x, 1-x], 2*n_in channels
        n = enc.shape[1] // 2
        P = self.matrix
        a = torch.einsum("oc,bchw->bohw", P, enc[:, :n])
        b = torch.einsum("oc,bchw->bohw", P, enc[:, n:])
        return torch.cat([a, b], 1)


class BcosNet(BcosUtilMixin, nn.Module):
    """Wraps a B-cos ResNet so every variant shares encode / forward / explain."""

    def __init__(self, trunk: nn.Module, project: Optional[nn.Module] = None,
                 ndvi_branch: Optional[nn.Module] = None):
        super().__init__()
        self.trunk, self.project, self.ndvi_branch = trunk, project, ndvi_branch
        self.ndvi_gate = nn.Parameter(torch.tensor(0.1)) if ndvi_branch is not None else None

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, 1 - x], 1)

    def forward_encoded(self, e: torch.Tensor, ne: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.project is not None:
            e = self.project(e)
        t = self.trunk
        if self.ndvi_branch is None:
            return t(e)
        h = t.bn1(t.conv1(e))
        if t.pool is not None:
            h = t.pool(h)
        h = t.layer1(h)
        n = F.adaptive_avg_pool2d(ne, h.shape[-2:])          # linear, keeps exactness
        h = h + self.ndvi_gate * self.ndvi_branch(n)
        h = t.layer2(h); h = t.layer3(h)
        if t.layer4 is not None:
            h = t.layer4(h)
        h = t.avgpool(t.fc(h)).flatten(1)
        return t.logit_layer(h)

    def forward(self, x: torch.Tensor, ndvi: Optional[torch.Tensor] = None) -> torch.Tensor:
        ne = self.encode(ndvi) if self.ndvi_branch is not None else None
        return self.forward_encoded(self.encode(x), ne)

    def contributions(self, x: torch.Tensor, ndvi: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Exact per-pixel contributions to (logit_loss - logit_noloss).

        Returns pixel map (B,H,W), per-band maps (B,C,H,W) with x and 1-x merged,
        the NDVI map if used, the explained logit difference, and the
        completeness error |sum(contributions) - logit difference|.
        """
        was_training = self.training
        self.eval()
        e = self.encode(x).detach().requires_grad_(True)
        ne = self.encode(ndvi).detach().requires_grad_(True) if self.ndvi_branch is not None else None
        with torch.enable_grad(), self.explanation_mode():
            out = self.forward_encoded(e, ne)
            diff = out[:, 1] - out[:, 0]
            inputs = [e] + ([ne] if ne is not None else [])
            grads = torch.autograd.grad(diff.sum(), inputs)
        c = (e * grads[0]).detach()
        n = c.shape[1] // 2
        per_band = c[:, :n] + c[:, n:]
        res = dict(pixels=per_band.sum(1), per_band=per_band, logit_diff=diff.detach())
        total = res["pixels"].flatten(1).sum(1)
        if ne is not None:
            nmap = (ne * grads[1]).detach().sum(1)
            res["ndvi"] = nmap
            total = total + nmap.flatten(1).sum(1)
        res["completeness_error"] = (total - res["logit_diff"]).abs()
        self.train(was_training)
        return res


# --------------------------------------------------------------------------- builders
def _new_head(trunk: nn.Module) -> None:
    trunk.fc = BcosConv2d(trunk.num_features, 2, kernel_size=1, b=2)
    trunk.logit_layer = LogitLayer(logit_bias=None)
    trunk.num_classes = 2


def _pretrained_trunk(download: bool = True) -> nn.Module:
    from bcos.models.pretrained import resnet18
    try:
        return resnet18(pretrained=download)
    except Exception as err:  # no internet: keep the architecture, warn loudly
        if download:
            print(f"WARNING: could not download B-cos ImageNet weights ({err}); using random init")
        return resnet18(pretrained=False)


def inflate_first_layer(trunk: nn.Module, alpha: float = 1.0) -> None:
    """6-channel [R,G,B,1-R,1-G,1-B] stem -> 8-channel [R,G,B,N,1-R,1-G,1-B,1-N].

    NIR filters start as alpha * mean(RGB filters) (and the same for the inverse
    channels). B-cos re-normalises every filter to unit norm, so adding channels
    shrinks the RGB part of each filter; alpha controls how much of the unit
    norm the new NIR channels take at the start of fine-tuning.
    """
    old = trunk.conv1
    w = old.linear.weight.data                              # (64, 6, k, k)
    new = BcosConv2d(8, w.shape[0], kernel_size=old.linear.kernel_size,
                     stride=old.linear.stride, padding=old.linear.padding, b=old.b)
    nw = torch.empty(w.shape[0], 8, *w.shape[2:])
    nw[:, 0:3] = w[:, 0:3]
    nw[:, 3] = alpha * w[:, 0:3].mean(1)
    nw[:, 4:7] = w[:, 3:6]
    nw[:, 7] = alpha * w[:, 3:6].mean(1)
    new.linear.weight.data.copy_(nw)
    trunk.conv1 = new


def build_model(variant: str, n_channels: int, download: bool = True, alpha: float = 1.0) -> nn.Module:
    if variant == "cnn":
        import torchvision
        m = torchvision.models.resnet18(weights=None, num_classes=2)
        m.conv1 = nn.Conv2d(n_channels, 64, 3, 1, 1, bias=False)   # small-input stem, same as bcos
        m.maxpool = nn.Identity()
        return m
    if variant == "bcos":
        from bcos.models.resnet import resnet18
        trunk = resnet18(in_chans=2 * n_channels, num_classes=2, small_inputs=True)
        trunk.logit_layer = LogitLayer(logit_bias=None)
        return BcosNet(trunk)
    if variant in PRETRAINED:
        trunk = _pretrained_trunk(download)
        _new_head(trunk)
        if variant == "bcos_pre_rgb":
            assert n_channels == 3
            return BcosNet(trunk)
        if variant == "bcos_pre_inflate":
            assert n_channels == 4
            inflate_first_layer(trunk, alpha)
            return BcosNet(trunk)
        if variant == "bcos_pre_project":
            assert n_channels == 4
            return BcosNet(trunk, project=SumToOneProjection(4, 3))
        if variant == "bcos_pre_ndvi":
            assert n_channels == 3
            return BcosNet(trunk, ndvi_branch=BcosConv2d(2, 64, kernel_size=3, padding=1, b=2))
    raise ValueError(f"unknown variant {variant}")


def is_bcos(model: nn.Module) -> bool:
    return isinstance(model, BcosNet)


# --------------------------------------------------------------------------- Grad-CAM (black-box baseline)
def gradcam(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Grad-CAM for the 'loss' class on a torchvision ResNet (layer4)."""
    feats = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    model.eval()
    with torch.enable_grad():
        out = model(x)
        a = feats["a"]
        g = torch.autograd.grad((out[:, 1] - out[:, 0]).sum(), a)[0]
    h.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    return F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].detach()
