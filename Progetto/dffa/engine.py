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
    # Firma del preprocessing: se cambia (es. risoluzione canonica), la cache
    # vecchia è incoerente e va ricalcolata.
    prep = (cfg.canonical_size, cfg.image_size, cfg.fourier_log,
            cfg.fourier_robust, cfg.fourier_clip_pct)
    if cache_path and Path(cache_path).exists():
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("n") == len(samples) and blob.get("prep") == list(prep):
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
        "prep": list(prep),
    }
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, cache_path)
    return blob


def slice_stream(blob: Dict[str, torch.Tensor], which: str) -> Dict[str, torch.Tensor]:
    """Restituisce un blob con i soli embedding di uno stream (per l'ablation).

    `which`: 'rgb' (prima metà), 'fourier' (seconda metà) o 'both' (tutto).
    L'embedding dual-stream è [emb_rgb (512) | emb_fourier (512)].
    """
    e = blob["embeddings"]
    half = e.shape[1] // 2
    if which == "rgb":
        e = e[:, :half]
    elif which == "fourier":
        e = e[:, half:]
    elif which != "both":
        raise ValueError(f"stream sconosciuto: {which!r}")
    out = dict(blob)
    out["embeddings"] = e
    return out


def _filter_blob(blob: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = {
        "embeddings": blob["embeddings"][mask],
        "detection": blob["detection"][mask],
        "attribution": blob["attribution"][mask],
        "n": int(mask.sum().item()),
    }
    if "prep" in blob:
        out["prep"] = blob["prep"]
    return out


def cross_generator_detection(
    blobs: Dict[str, Dict[str, torch.Tensor]], cfg: Config, device
) -> Dict[str, Dict[str, float]]:
    """Leave-one-generator-out: generalizzazione della *detection* a un generatore mai visto.

    Per ogni generatore `g`: allena su real + (altri generatori), testa la detection
    (real vs fake) su real + `g`. Misura quanto il rilevatore generalizza a una
    sorgente sconosciuta, invece di memorizzare la firma di ciascun dataset.
    Si leggono solo metriche di *detection* (l'attribution non è applicabile a una
    classe assente dal training).
    """
    results: Dict[str, Dict[str, float]] = {}
    for gi, gname in enumerate(cfg.generator_classes):
        tr = _filter_blob(blobs["train"], blobs["train"]["attribution"] != gi)
        va = _filter_blob(blobs["val"], blobs["val"]["attribution"] != gi)
        model, _ = train_classifier(tr, va, cfg, device)

        te_mask = (blobs["test"]["attribution"] == -1) | (blobs["test"]["attribution"] == gi)
        te = _filter_blob(blobs["test"], te_mask)
        r = evaluate(model, _loader_from_blob(te, cfg, shuffle=False), cfg, device)

        # recall sui fake del generatore mai visto: quanti riconosciuti come fake
        true = torch.tensor(r["detection_true"])
        pred = torch.tensor(r["detection_pred"])
        fake = true == 1
        unseen_recall = float(((pred == 1) & fake).sum().item() / max(int(fake.sum().item()), 1))
        results[gname] = {
            "detection_acc": r["detection_acc"],
            "unseen_fake_recall": unseen_recall,
        }
    return results


def _loader_from_blob(blob: Dict[str, torch.Tensor], cfg: Config, shuffle: bool):
    ds = TensorDataset(blob["embeddings"], blob["detection"], blob["attribution"])
    # drop_last in training: evita un batch finale di dimensione 1 che farebbe
    # fallire la BatchNorm1d del classificatore.
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=shuffle)


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
    # attribution: ignora i campioni reali (label -1) -> training solo sui generatori
    ce_attr = nn.CrossEntropyLoss(ignore_index=-1)
    train_loader = _loader_from_blob(train_blob, cfg, shuffle=True)
    val_loader = _loader_from_blob(val_blob, cfg, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_det_acc": [],
               "val_attr_acc": [], "val_cascade_acc": []}
    best_acc, best_state = -1.0, None

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for emb, det, attr in train_loader:
            emb, det, attr = emb.to(device), det.to(device), attr.to(device)
            out = model(emb)
            # se il batch non contiene fake, salta la loss di attribution (evita NaN)
            attr_loss = (
                ce_attr(out.attribution_logits, attr)
                if (attr != -1).any()
                else torch.zeros((), device=device)
            )
            loss = (
                cfg.detection_weight * ce(out.detection_logits, det)
                + cfg.attribution_weight * attr_loss
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
        history["val_cascade_acc"].append(val["cascade_acc"])

        # selezione sul cascade (end-to-end), la metrica che conta davvero
        if val["cascade_acc"] > best_acc:
            best_acc = val["cascade_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Dict:
    """Metriche a cascata.

    - detection_acc:   su tutti i campioni (real/fake).
    - attribution_acc: SOLO sui fake (label != -1), tra i generatori.
    - cascade_acc:     end-to-end con la regola di cascata (la detection fa da gate):
        corretto se predetto real ed era real, oppure predetto fake con generatore giusto.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    ce_attr = nn.CrossEntropyLoss(ignore_index=-1)
    tot_loss = det_correct = attr_correct = attr_total = casc_correct = n = 0
    det_true, det_pred, attr_true, attr_pred, casc_true, casc_pred = [], [], [], [], [], []

    for emb, det, attr in loader:
        emb, det, attr = emb.to(device), det.to(device), attr.to(device)
        out = model(emb)
        attr_loss = (
            ce_attr(out.attribution_logits, attr)
            if (attr != -1).any()
            else torch.zeros((), device=device)
        )
        loss = cfg.detection_weight * ce(out.detection_logits, det) \
            + cfg.attribution_weight * attr_loss
        tot_loss += loss.item() * emb.size(0)

        dp = out.detection_logits.argmax(1)
        ap = out.attribution_logits.argmax(1)
        det_correct += (dp == det).sum().item()
        n += emb.size(0)

        fake = attr != -1
        attr_total += int(fake.sum().item())
        attr_correct += ((ap == attr) & fake).sum().item()

        # cascata: etichetta finale come stringa (-1=real, altrimenti nome generatore)
        for k in range(emb.size(0)):
            t = "real" if int(det[k]) == 0 else cfg.generator_classes[int(attr[k])]
            p = "real" if int(dp[k]) == 0 else cfg.generator_classes[int(ap[k])]
            casc_true.append(t); casc_pred.append(p)
            casc_correct += int(t == p)

        det_true += det.cpu().tolist(); det_pred += dp.cpu().tolist()
        for k in range(emb.size(0)):
            if bool(fake[k]):
                attr_true.append(int(attr[k])); attr_pred.append(int(ap[k]))

    return {
        "loss": tot_loss / max(n, 1),
        "detection_acc": det_correct / max(n, 1),
        "attribution_acc": attr_correct / max(attr_total, 1),
        "cascade_acc": casc_correct / max(n, 1),
        "detection_true": det_true, "detection_pred": det_pred,
        "attribution_true": attr_true, "attribution_pred": attr_pred,
        "cascade_true": casc_true, "cascade_pred": casc_pred,
    }
