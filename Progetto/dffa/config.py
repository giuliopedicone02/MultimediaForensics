"""Configurazione centralizzata del progetto.

La configurazione è una semplice dataclass serializzabile da/verso YAML, così da
poter essere versionata e ripresa identica su Colab. I default sono pensati per
una T4 (16 GB) ed un dataset dell'ordine di ~1000 immagini.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml è in requirements, fallback difensivo
    yaml = None


# Ordine canonico delle classi di *attribution*. L'indice 0 è sempre "real".
# La detection è derivata: real -> 0, qualsiasi generatore -> 1.
DEFAULT_CLASSES: List[str] = ["real", "stylegan", "stylegan3", "sdxl"]


@dataclass
class Config:
    # --- dati ---
    data_root: str = "data"
    classes: List[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    image_size: int = 224
    # Risoluzione canonica: ogni immagine, qualunque sia la dimensione nativa,
    # viene prima portata a (canonical_size, canonical_size) con la STESSA
    # interpolazione, *prima* sia dello stream RGB sia dello spettro di Fourier.
    # Questo neutralizza il confound da resampling: senza, classi con risoluzioni
    # native diverse (es. StyleGAN 1024 vs SDXL 256) sarebbero separabili in modo
    # banale dalla "firma" del ridimensionamento e non dai veri artefatti.
    # None = comportamento legacy (nessuna uniformazione → confound).
    canonical_size: Optional[int] = 256
    max_per_class: Optional[int] = None  # None = usa tutte le immagini disponibili

    # --- split (frazioni sul totale) ---
    val_split: float = 0.15
    test_split: float = 0.15

    # --- feature extraction ---
    backbone: str = "resnet18"          # backbone dei due stream
    embedding_dim: int = 512            # dim. embedding di resnet18 (post global-pool)
    fourier_log: bool = True            # log-magnitudine dello spettro
    # Normalizzazione robusta dello spettro: la componente DC (freq. 0) ha
    # magnitudine enorme e, con un min-max globale, schiaccia gli artefatti
    # periodici di media/alta frequenza — proprio il segnale forense utile. Con
    # `fourier_robust=True` lo spettro viene clippato al percentile `fourier_clip_pct`
    # prima di scalare, espandendo la dinamica delle frequenze informative.
    # False = min-max globale (comportamento legacy, per l'ablation).
    fourier_robust: bool = True
    fourier_clip_pct: float = 99.0      # percentile di clipping (se fourier_robust)

    # --- classificatore multi-task ---
    hidden_dim: int = 256
    dropout: float = 0.3
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    attribution_weight: float = 1.0     # peso della loss di attribution
    detection_weight: float = 1.0       # peso della loss di detection

    # --- VLM agent (explainability) ---
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_load_in_4bit: bool = True
    vlm_max_new_tokens: int = 512
    vlm_enabled: bool = True            # se False usa la spiegazione template-based

    # --- runtime ---
    seed: int = 42
    # 0 = nessun subprocess: evita i warning di shutdown dei worker del DataLoader
    # in ambiente interattivo (IPython/Colab). L'estrazione embedding su ~1000
    # immagini con backbone congelato è comunque rapida.
    num_workers: int = 0
    results_dir: str = "results"
    device: str = "auto"                # "auto" | "cuda" | "cpu"

    # ------------------------------------------------------------------ helpers
    @property
    def generator_classes(self) -> List[str]:
        """Solo i generatori (esclude 'real'): sono le classi della 2ª testa."""
        return [c for c in self.classes if c != "real"]

    @property
    def num_attribution_classes(self) -> int:
        # Cascade: l'attribution distingue SOLO tra generatori (no 'real').
        return len(self.generator_classes)

    @property
    def num_detection_classes(self) -> int:
        return 2

    def detection_label(self, class_index: int) -> int:
        """0 se la classe è 'real', 1 altrimenti."""
        return 0 if self.classes[class_index] == "real" else 1

    def attribution_label(self, class_index: int) -> int:
        """Indice del generatore (0-based tra i soli generatori).

        Restituisce -1 per le immagini reali: usato come `ignore_index` nella
        CrossEntropy così che il real non contribuisca alla loss di attribution.
        """
        name = self.classes[class_index]
        if name == "real":
            return -1
        return self.generator_classes.index(name)

    def to_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML non disponibile: `pip install pyyaml`")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if yaml is None:
            raise RuntimeError("PyYAML non disponibile: `pip install pyyaml`")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
