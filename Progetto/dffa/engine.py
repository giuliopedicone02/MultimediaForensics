"""Estrazione embedding (con cache), training e valutazione multi-task.

Poiché i backbone ResNet18 sono congelati, conviene pre-calcolare gli embedding
dual-stream una sola volta e salvarli; il classificatore MLP si addestra poi in
pochi secondi anche su CPU. Questo rende il notebook veloce e riproducibile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import Config
from .data.dataset import ForensicsDataset
from .features.extractor import DualStreamExtractor
from .models.classifier import MultiTaskClassifier


# --------------------------------------------------------------------------- #
#  Embedding pre-calcolati (con cache su disco)                                 #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(
    samples: List[Tuple[str, int]],
    cfg: Config,
    extractor: DualStreamExtractor,
    device: torch.device,
    cache_path: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Calcola (o carica) embedding 1024-d + etichette per un set di campioni."""
    if cache_path and Path(cache_path).exists():
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("n") == len(samples):
            return blob

    ds = ForensicsDataset(samples, cfg, augment=False)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    extractor.eval().to(device)

    embs, det, attr = [], [], []
    for batch in tqdm(loader, desc="embedding", leave=False):
        e = extractor(batch["rgb"].to(device), batch["fourier"].to(device))
        embs.append(e.cpu())
        det.append(batch["detection"])
        attr.append(batch["attribution"])
    blob = {
        "embeddings": torch.cat(embs),
        "detection": torch.cat(det).long(),
        "attribution": torch.cat(attr).long(),
        "n": len(samples),
    }
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, cache_path)
    return blob


def _loader_from_blob(blob: Dict[str, torch.Tensor], cfg: Config, shuffle: bool):
    ds = TensorDataset(blob["embeddings"], blob["detection"], blob["attribution"])
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
#  Training                                                                     #
# --------------------------------------------------------------------------- #
def train_classifier(
    train_blob: Dict[str, torch.Tensor],
    val_blob: Dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
) -> Tuple[MultiTaskClassifier, Dict[str, list]]:
    model = MultiTaskClassifier(
        in_dim=train_blob["embeddings"].shape[1],
        hidden_dim=cfg.hidden_dim,
        num_attribution_classes=cfg.num_attribution_classes,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    train_loader = _loader_from_blob(train_blob, cfg, shuffle=True)
    val_loader = _loader_from_blob(val_blob, cfg, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_det_acc": [], "val_attr_acc": []}
    best_acc, best_state = -1.0, None

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for emb, det, attr in train_loader:
            emb, det, attr = emb.to(device), det.to(device), attr.to(device)
            out = model(emb)
            loss = (
                cfg.detection_weight * ce(out.detection_logits, det)
                + cfg.attribution_weight * ce(out.attribution_logits, attr)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * emb.size(0)
        train_loss = running / len(train_loader.dataset)

        val = evaluate(model, val_loader, cfg, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_det_acc"].append(val["detection_acc"])
        history["val_attr_acc"].append(val["attribution_acc"])

        if val["attribution_acc"] > best_acc:
            best_acc = val["attribution_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss = det_correct = attr_correct = n = 0
    det_true, det_pred, attr_true, attr_pred = [], [], [], []
    for emb, det, attr in loader:
        emb, det, attr = emb.to(device), det.to(device), attr.to(device)
        out = model(emb)
        loss = (
            cfg.detection_weight * ce(out.detection_logits, det)
            + cfg.attribution_weight * ce(out.attribution_logits, attr)
        )
        tot_loss += loss.item() * emb.size(0)
        dp = out.detection_logits.argmax(1)
        ap = out.attribution_logits.argmax(1)
        det_correct += (dp == det).sum().item()
        attr_correct += (ap == attr).sum().item()
        n += emb.size(0)
        det_true += det.cpu().tolist(); det_pred += dp.cpu().tolist()
        attr_true += attr.cpu().tolist(); attr_pred += ap.cpu().tolist()
    return {
        "loss": tot_loss / max(n, 1),
        "detection_acc": det_correct / max(n, 1),
        "attribution_acc": attr_correct / max(n, 1),
        "detection_true": det_true, "detection_pred": det_pred,
        "attribution_true": attr_true, "attribution_pred": attr_pred,
    }
