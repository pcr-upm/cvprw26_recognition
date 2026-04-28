#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ivan Ferre'
__email__ = 'ivan.ferre@upm.es'

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.pool = m.avgpool
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.pool = m.avgpool
        in_dim = 2048
        self.proj = nn.Identity() if out_dim == in_dim else nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FERBaselineNet(nn.Module):
    def __init__(self, num_expr: int, pretrained_backbone: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.head_expr = MLPHead(self.backbone.out_dim, num_expr, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head_expr(feat)
