"""Dataset e split per deepfake detection & attribution.

Layout atteso su disco (una sottocartella per classe di *attribution*)::

    data/
      real/        # es. FFHQ
        00000.png
        ...
      stylegan/
      stylegan2/
      stylegan3/

Ogni campione restituisce sia il tensore RGB sia il tensore dello spettro di
Fourier, oltre alle due etichette (detection + attribution) ed al path.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..config import Config
from ..features.fourier import fourier_spectrum_tensor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_samples(cfg: Config) -> List[Tuple[str, int]]:
    """Elenca (path, attribution_index) scandendo `data_root/<classe>`.

    Le classi assenti dal disco vengono semplicemente ignorate (utile quando si
    parte con pochi generatori e se ne aggiungono altri in seguito).
    """
    root = Path(cfg.data_root)
    samples: List[Tuple[str, int]] = []
    for idx, cls in enumerate(cfg.classes):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if cfg.max_per_class is not None:
            files = files[: cfg.max_per_class]
        samples.extend((str(p), idx) for p in files)
    return samples


def build_splits(
    cfg: Config,
) -> Dict[str, List[Tuple[str, int]]]:
    """Split stratificato per classe in train/val/test, riproducibile via seed."""
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for path, idx in list_samples(cfg):
        by_class.setdefault(idx, []).append((path, idx))

    rng = random.Random(cfg.seed)
    train, val, test = [], [], []
    for idx, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * cfg.test_split))
        n_val = int(round(n * cfg.val_split))
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    rng.shuffle(train)
    return {"train": train, "val": val, "test": test}


class ForensicsDataset(Dataset):
    """Dataset dual-stream (RGB + Fourier) con etichette detection/attribution."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        cfg: Config,
        augment: bool = False,
    ):
        self.samples = samples
        self.cfg = cfg
        self.augment = augment

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tfs = [transforms.Resize((cfg.image_size, cfg.image_size))]
        if augment:
            tfs.append(transforms.RandomHorizontalFlip())
        tfs += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.rgb_tf = transforms.Compose(tfs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | int | str]:
        path, attr_idx = self.samples[i]
        img = Image.open(path).convert("RGB")
        # Uniformazione della risoluzione (anti-confound): tutte le classi passano
        # per la stessa identica catena di resize, così la firma del resampling è
        # condivisa e non più una scorciatoia per la classe. Lo stesso `base` (a
        # risoluzione canonica) alimenta sia lo stream RGB sia lo spettro di Fourier.
        if self.cfg.canonical_size:
            cs = self.cfg.canonical_size
            base = img.resize((cs, cs), Image.BICUBIC)
        else:
            base = img
        rgb = self.rgb_tf(base)
        fft = fourier_spectrum_tensor(
            base, size=self.cfg.image_size, log=self.cfg.fourier_log
        )
        return {
            "rgb": rgb,
            "fourier": fft,
            # attribution: indice del generatore (-1 per i real, ignorato nella loss)
            "attribution": self.cfg.attribution_label(attr_idx),
            "detection": self.cfg.detection_label(attr_idx),
            "path": path,
        }
