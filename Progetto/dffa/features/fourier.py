"""Spettro di Fourier delle immagini.

Le immagini sintetiche (GAN) lasciano artefatti caratteristici nel dominio della
frequenza — tipicamente picchi/griglie periodiche dovuti alle operazioni di
up-sampling del generatore. Lo spettro di magnitudine (log) rende questi artefatti
visibili e li espone alla ResNet del secondo stream.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _to_grayscale_array(image: "Image.Image | np.ndarray") -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:  # converti RGB -> luminanza
            arr = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return arr


def fourier_spectrum(
    image: "Image.Image | np.ndarray",
    log: bool = True,
    robust: bool = True,
    clip_pct: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Spettro di magnitudine 2D normalizzato in [0, 1].

    Args:
        image:    immagine PIL o array (H, W) / (H, W, 3).
        log:      se True applica log(1 + |F|) per comprimere la dinamica.
        robust:   se True clippa al percentile `clip_pct` prima di scalare. La
                  componente DC (frequenza 0) ha magnitudine enorme: con un
                  min-max globale dominerebbe la normalizzazione, comprimendo in
                  un range minimo gli artefatti periodici di media/alta frequenza
                  (il segnale forense utile). Il clipping espande quella dinamica.
        clip_pct: percentile usato come tetto quando `robust=True`.
    Returns:
        array float32 (H, W) in [0, 1], con la frequenza zero centrata.
    """
    gray = _to_grayscale_array(image)
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)                # frequenza 0 al centro
    mag = np.abs(f)
    if log:
        mag = np.log1p(mag)
    if robust:
        hi = float(np.percentile(mag, clip_pct))
        mag = np.clip(mag, mag.min(), hi)
    mag -= mag.min()
    mag /= (mag.max() + eps)
    return mag.astype(np.float32)


def fourier_spectrum_tensor(
    image: "Image.Image | np.ndarray",
    size: int = 224,
    log: bool = True,
    robust: bool = True,
    clip_pct: float = 99.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Spettro come tensore (3, size, size) pronto per la ResNet.

    Lo spettro mono-canale viene replicato su 3 canali e normalizzato con le
    statistiche di ImageNet (coerente con il backbone pre-addestrato).
    """
    mag = fourier_spectrum(image, log=log, robust=robust, clip_pct=clip_pct)
    t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = torch.nn.functional.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    )
    t = t.squeeze(0).repeat(3, 1, 1)  # (3, size, size)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
    return t
