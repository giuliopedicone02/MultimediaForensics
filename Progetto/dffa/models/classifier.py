"""Classificatore multi-task sugli embedding dual-stream.

Un tronco condiviso (MLP) elabora il vettore concatenato [emb_rgb | emb_fourier]
(1024-d) e alimenta due teste:
  * detection   -> 2 logit  (real / fake)
  * attribution -> K logit  (real / stylegan / stylegan2 / ...)

Le due teste condividono la rappresentazione: la detection beneficia del segnale
più fine dell'attribution e viceversa.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MultiTaskOutput:
    detection_logits: torch.Tensor      # (B, 2)
    attribution_logits: torch.Tensor    # (B, K)


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        num_attribution_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.detection_head = nn.Linear(hidden_dim, 2)
        self.attribution_head = nn.Linear(hidden_dim, num_attribution_classes)

    def forward(self, embeddings: torch.Tensor) -> MultiTaskOutput:
        h = self.trunk(embeddings)
        return MultiTaskOutput(
            detection_logits=self.detection_head(h),
            attribution_logits=self.attribution_head(h),
        )

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> dict:
        """Probabilità e classi predette per entrambe le teste."""
        out = self.forward(embeddings)
        det_p = torch.softmax(out.detection_logits, dim=1)
        attr_p = torch.softmax(out.attribution_logits, dim=1)
        return {
            "detection_prob": det_p,
            "detection_pred": det_p.argmax(1),
            "attribution_prob": attr_p,
            "attribution_pred": attr_p.argmax(1),
        }
