"""Classificatore multi-task sugli embedding dual-stream.

Un tronco condiviso (MLP) elabora il vettore concatenato [emb_rgb | emb_fourier]
(1024-d) e alimenta due teste:
  * detection   -> 2 logit  (real / fake)            — su tutti i campioni
  * attribution -> G logit  (stylegan / stylegan3 / ...) — SOLO sui generatori

**Cascade**: a inference si decide prima la detection; l'attribution si interpreta
solo se l'immagine è classificata *fake*. L'attribution non include la classe
'real', quindi la spiegazione non può essere incoerente (mai "fake ma → real").
In training, la loss di attribution ignora i campioni reali (`ignore_index=-1`).
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
        self.eval()  # BatchNorm su running stats: sicuro anche con batch di 1
        out = self.forward(embeddings)
        det_p = torch.softmax(out.detection_logits, dim=1)
        attr_p = torch.softmax(out.attribution_logits, dim=1)
        return {
            "detection_prob": det_p,
            "detection_pred": det_p.argmax(1),
            "attribution_prob": attr_p,         # solo sui generatori
            "attribution_pred": attr_p.argmax(1),
        }


@torch.no_grad()
def cascade_predict(model: "MultiTaskClassifier", embeddings, generator_classes):
    """Decisione a cascata: detection e, solo se *fake*, attribution.

    Restituisce (pred_dict, decisions) dove `decisions` è una lista di dict:
        {'detection': 'real'|'fake', 'attribution': <generatore>|None}
    L'attribution è None quando l'immagine è classificata reale.
    """
    p = model.predict(embeddings)
    decisions = []
    for i in range(embeddings.shape[0]):
        if int(p["detection_pred"][i]) == 0:
            decisions.append({"detection": "real", "attribution": None})
        else:
            gi = int(p["attribution_pred"][i])
            decisions.append({"detection": "fake",
                              "attribution": generator_classes[gi]})
    return p, decisions
