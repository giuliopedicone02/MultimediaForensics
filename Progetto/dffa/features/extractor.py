"""Estrattori di embedding basati su ResNet18 pre-addestrata su ImageNet.

Il backbone è *congelato* (feature extractor) e produce un embedding di 512-d dopo
il global average pooling. Lo stesso tipo di estrattore è usato per i due stream
(RGB e Fourier), istanziati separatamente così che il Grad-CAM possa agganciarsi
all'ultimo blocco convoluzionale (`layer4`) di ciascuno.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetEmbedder(nn.Module):
    """ResNet18 senza il classificatore finale: restituisce embedding 512-d.

    Espone `self.target_layer` (= layer4) come punto d'aggancio per Grad-CAM.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        self.target_layer = net.layer4  # riferimento per Grad-CAM
        # tutto tranne la fc finale
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
            net.avgpool,
        )
        self.out_dim = 512
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)            # (B, 512, 1, 1)
        return torch.flatten(feats, 1)      # (B, 512)


class DualStreamExtractor(nn.Module):
    """Wrapper che gestisce i due stream con due ResNet18 indipendenti."""

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.rgb = ResNetEmbedder(pretrained=pretrained, freeze=freeze)
        self.fourier = ResNetEmbedder(pretrained=pretrained, freeze=freeze)
        self.out_dim = self.rgb.out_dim + self.fourier.out_dim  # 1024

    @torch.no_grad()
    def forward(
        self, rgb: torch.Tensor, fourier: torch.Tensor
    ) -> torch.Tensor:
        e_rgb = self.rgb(rgb)
        e_fft = self.fourier(fourier)
        return torch.cat([e_rgb, e_fft], dim=1)  # (B, 1024)
