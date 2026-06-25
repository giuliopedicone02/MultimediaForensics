"""Grad-CAM per le ResNet dei due stream.

Grad-CAM evidenzia le regioni spaziali che più contribuiscono ad una certa
classe, pesando le mappe di attivazione dell'ultimo blocco convoluzionale con il
gradiente medio della classe rispetto a quelle attivazioni. Lo usiamo sul backbone
RGB (dove guarda la rete nell'immagine) e su quello di Fourier (quali frequenze
pesano di più), fornendo evidenza visiva all'agent VLM.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM agganciato ad un `target_layer` (tipicamente resnet.layer4).

    Esempio::

        cam = GradCAM(embedder.backbone, embedder.target_layer)
        heatmap = cam(rgb_tensor, score_fn=lambda f: head(f)[:, cls])
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._h1 = target_layer.register_forward_hook(self._save_activation)
        self._h2 = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, score_fn) -> np.ndarray:
        """Restituisce la heatmap (H, W) in [0,1] per l'input `x` (1,3,H,W).

        `score_fn` mappa l'embedding (B, C) nello scalare da differenziare
        (es. il logit della classe predetta).
        """
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        # forzo i gradienti anche se il backbone è congelato
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            feats = self.model(x)
            score = score_fn(torch.flatten(feats, 1))
            score = score.sum()
            score.backward()

        # peso ogni canale per il gradiente medio (global-average-pool)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (B,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        if was_training:
            self.model.train()
        return cam


def overlay_cam(
    rgb_chw: torch.Tensor,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Sovrappone la heatmap all'immagine (de-normalizzata) -> array RGB uint8."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (rgb_chw.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    # colormap "jet" senza dipendere da matplotlib
    cam_color = _jet(cam)
    out = (1 - alpha) * img + alpha * cam_color
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _jet(x: np.ndarray) -> np.ndarray:
    """Approssimazione della colormap jet, restituisce (H, W, 3) in [0,1]."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)
