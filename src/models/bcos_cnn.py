"""
B-cos Neural Network Architectures for Satellite Deforestation Classification.
Paper: "B-cos Networks: Alignment is All You Need for Interpretability" (CVPR 2022)
"""

from typing import List, Optional, Type, Union
import torch
import torch.nn as nn
from .bcos_modules import BcosConv2d, AddInverse, set_explanation_mode


class BcosBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, b: int = 2):
        super().__init__()
        self.conv1 = BcosConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, b=b)
        self.conv2 = BcosConv2d(planes, planes, kernel_size=3, stride=1, padding=1, b=b)
        self.stride = stride

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = BcosConv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = out + identity
        return out


class BcosDeforestationCNN(nn.Module):
    """
    Interpretable B-cos CNN designed for satellite imagery (64x64 - 256x256).
    Replaces traditional non-linearities and unconstrained weights with
    B-cos transformations and AddInverse encoding.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        b: int = 2,
        auto_add_inverse: bool = True,
    ):
        super().__init__()
        self.auto_add_inverse = auto_add_inverse
        self.add_inverse = AddInverse(dim=1)
        effective_in_c = in_channels * 2 if auto_add_inverse else in_channels

        # Stage 1: Initial feature extraction
        self.stem = nn.Sequential(
            BcosConv2d(effective_in_c, base_channels, kernel_size=5, stride=1, padding=2, b=b),
            BcosConv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, b=b),
        )

        # Stage 2: Mid-level spatial representation
        self.stage2 = nn.Sequential(
            BcosBasicBlock(base_channels, base_channels * 2, stride=2, b=b),
            BcosBasicBlock(base_channels * 2, base_channels * 2, stride=1, b=b),
        )

        # Stage 3: High-level semantic features
        self.stage3 = nn.Sequential(
            BcosBasicBlock(base_channels * 2, base_channels * 4, stride=2, b=b),
            BcosBasicBlock(base_channels * 4, base_channels * 4, stride=1, b=b),
        )

        # Stage 4: Forest/Deforestation structure
        self.stage4 = nn.Sequential(
            BcosBasicBlock(base_channels * 4, base_channels * 8, stride=2, b=b),
        )

        # Spatial aggregation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final B-cos linear classification head (no bias, preserving exact linearity)
        self.classifier = BcosConv2d(
            base_channels * 8,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            max_out=1,
            b=b,
            scale=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_add_inverse and x.shape[1] == 3:
            x = self.add_inverse(x)

        feat = self.stem(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        pooled = self.pool(feat)
        out = self.classifier(pooled)
        return out.view(out.size(0), -1)

    def explanation_mode(self, active: bool = True):
        set_explanation_mode(self, active)


class BcosResNet18(nn.Module):
    """
    Full B-cos ResNet-18 model adapted for satellite deforestation prediction.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        b: int = 2,
        auto_add_inverse: bool = True,
    ):
        super().__init__()
        self.auto_add_inverse = auto_add_inverse
        self.add_inverse = AddInverse(dim=1)
        effective_in_c = in_channels * 2 if auto_add_inverse else in_channels

        self.conv1 = BcosConv2d(effective_in_c, 64, kernel_size=7, stride=2, padding=3, b=b)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BcosBasicBlock(64, 64, stride=1, b=b),
            BcosBasicBlock(64, 64, stride=1, b=b),
        )
        self.layer2 = nn.Sequential(
            BcosBasicBlock(64, 128, stride=2, b=b),
            BcosBasicBlock(128, 128, stride=1, b=b),
        )
        self.layer3 = nn.Sequential(
            BcosBasicBlock(128, 256, stride=2, b=b),
            BcosBasicBlock(256, 256, stride=1, b=b),
        )
        self.layer4 = nn.Sequential(
            BcosBasicBlock(256, 512, stride=2, b=b),
            BcosBasicBlock(512, 512, stride=1, b=b),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BcosConv2d(512, num_classes, kernel_size=1, stride=1, padding=0, max_out=1, b=b, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_add_inverse and x.shape[1] == 3:
            x = self.add_inverse(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)

    def explanation_mode(self, active: bool = True):
        set_explanation_mode(self, active)
