"""Agent VLM open per la spiegazione in linguaggio naturale.

Il classificatore decide *cosa* (real/fake e quale generatore); l'agent VLM spiega
il *perché*. All'agent vengono mostrate fino a tre immagini — RGB originale,
spettro di Fourier, overlay Grad-CAM — insieme all'evidenza numerica prodotta dal
classificatore (probabilità, classe predetta, regioni salienti). L'agent è un VLM
open eseguito *localmente* sulla T4 (default: Qwen2.5-VL-3B-Instruct, opz. 4-bit).

Se `transformers`/il modello non sono disponibili (es. esecuzione offline o CPU),
si ricade su una spiegazione template-based deterministica, così il notebook gira
comunque end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
#  Costruzione dell'evidenza testuale a partire dagli output del classificatore  #
# --------------------------------------------------------------------------- #
def build_evidence(
    detection_prob: np.ndarray,
    attribution_prob: np.ndarray,
    generator_classes: List[str],
    cam_stats: Optional[Dict[str, float]] = None,
) -> Dict:
    """Riassume gli output del modello (a cascata) in un dizionario per l'agent.

    `attribution_prob` è la distribuzione sui SOLI generatori. L'attribution è
    riportata solo se la detection dice *fake*; altrimenti è None (non applicabile).
    """
    det_pred = int(np.argmax(detection_prob))
    evidence = {
        "detection": {
            "label": "fake" if det_pred == 1 else "real",
            "confidence": round(float(detection_prob[det_pred]), 4),
            "p_real": round(float(detection_prob[0]), 4),
            "p_fake": round(float(detection_prob[1]), 4),
        },
        "attribution": None,  # cascade: valorizzata solo se fake
    }
    if det_pred == 1:
        attr_pred = int(np.argmax(attribution_prob))
        evidence["attribution"] = {
            "label": generator_classes[attr_pred],
            "confidence": round(float(attribution_prob[attr_pred]), 4),
            "distribution": {
                c: round(float(p), 4)
                for c, p in zip(generator_classes, attribution_prob)
            },
        }
    if cam_stats:
        evidence["gradcam"] = {k: round(float(v), 4) for k, v in cam_stats.items()}
    return evidence


_SYSTEM_PROMPT = (
    "Sei un assistente di multimedia forensics. Ti vengono mostrate l'immagine "
    "originale, il suo spettro di Fourier e una mappa Grad-CAM delle regioni più "
    "rilevanti per un classificatore, insieme alle predizioni numeriche del "
    "classificatore (detection real/fake e, se fake, attribution del generatore). "
    "Segui queste regole:\n"
    "1. Scrivi in italiano corretto e scorrevole, in modo chiaro e CAUTO (5-7 frasi).\n"
    "2. Descrivi solo ciò che è effettivamente osservabile. NON dare per scontata "
    "la presenza di artefatti (es. griglie periodiche nello spettro, incoerenze di "
    "texture) se non li riconosci davvero: in caso, dichiara l'incertezza.\n"
    "3. Ricorda che il classificatore può basarsi su indizi non visibili a occhio "
    "(statistiche di colore, compressione, risoluzione legate alla sorgente dei "
    "dati): segnala questa possibilità invece di inventare artefatti.\n"
    "4. Tratta le probabilità come output del modello, non come prova certa; "
    "distingui ciò che il classificatore RIPORTA da ciò che è VISIVAMENTE "
    "verificabile.\n"
    "5. Distingui la motivazione della DETECTION da quella dell'ATTRIBUTION."
)


@dataclass
class Explanation:
    text: str
    source: str  # "vlm" | "template"


class VLMExplainer:
    """Wrapper attorno ad un VLM open (Qwen2.5-VL) con fallback template-based."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model = None
        self.processor = None
        self._load_in_4bit = load_in_4bit

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """Carica il VLM. Restituisce True se caricato, False se non disponibile."""
        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )

            kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
            if self._load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            return True
        except Exception as e:  # pragma: no cover - dipende dall'ambiente
            print(f"[VLMExplainer] modello non disponibile ({e}); uso il template.")
            self.model = None
            return False

    # --------------------------------------------------------------- explain
    def explain(
        self,
        images: Dict[str, Image.Image],
        evidence: Dict,
    ) -> Explanation:
        """Genera la spiegazione. `images` può contenere 'rgb','fourier','gradcam'."""
        if self.model is None:
            return Explanation(self._template(evidence), source="template")
        try:
            return Explanation(self._vlm(images, evidence), source="vlm")
        except Exception as e:  # pragma: no cover
            print(f"[VLMExplainer] errore in inferenza ({e}); uso il template.")
            return Explanation(self._template(evidence), source="template")

    # -------------------------------------------------------------- internal
    def _vlm(self, images: Dict[str, Image.Image], evidence: Dict) -> str:
        content: List[dict] = []
        for name in ("rgb", "fourier", "gradcam"):
            if name in images and images[name] is not None:
                content.append({"type": "image", "image": images[name]})
        det = evidence["detection"]
        attr = evidence["attribution"]
        if attr is not None:  # cascade: fake -> c'è anche l'attribution
            attr_line = (
                f"- Attribution (solo se fake): {attr['label']} "
                f"(conf. {attr['confidence']}); distribuzione={attr['distribution']}\n\n"
                "Commenta la classificazione come FAKE e l'attribuzione al generatore "
                "indicato. Indica quali evidenze visive sono realmente osservabili e "
                "quali no; se non vedi artefatti chiari, dillo e ipotizza che il modello "
                "possa appoggiarsi a indizi di sorgente non visibili a occhio."
            )
        else:  # real: nessuna attribution
            attr_line = (
                "- Attribution: non applicabile (immagine classificata reale).\n\n"
                "Commenta la classificazione come REAL: indica se osservi assenza di "
                "artefatti evidenti, restando cauto sul fatto che l'assenza visiva non "
                "è una prova definitiva."
            )
        user_text = (
            "Immagini fornite (in ordine): RGB originale, spettro di Fourier, "
            "overlay Grad-CAM.\n\n"
            f"Predizioni del classificatore:\n"
            f"- Detection: {det['label']} (conf. {det['confidence']}, "
            f"p_real={det['p_real']}, p_fake={det['p_fake']})\n"
            + attr_line
        )
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pil_images = [c["image"] for c in content if c["type"] == "image"]
        inputs = self.processor(
            text=[text], images=pil_images, return_tensors="pt"
        ).to(self.model.device)
        # decoding deterministico (greedy): spiegazioni riproducibili
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _template(evidence: Dict) -> str:
        """Spiegazione di riserva, deterministica, dalle sole evidenze numeriche."""
        det = evidence["detection"]
        attr = evidence["attribution"]
        cam = evidence.get("gradcam", {})
        lines = []
        if det["label"] == "fake":
            lines.append(
                f"Il classificatore segnala l'immagine come FAKE con confidenza "
                f"{det['confidence']:.2%} (p_fake={det['p_fake']:.2%}). Le cause "
                "possono includere artefatti tipici dei generatori (griglie periodiche "
                "nello spettro, incoerenze di texture), ma anche indizi di sorgente non "
                "visibili a occhio (statistiche di colore/compressione/risoluzione): "
                "la sola confidenza non è una prova certa."
            )
        else:
            lines.append(
                f"Il classificatore segnala l'immagine come REAL con confidenza "
                f"{det['confidence']:.2%} (p_real={det['p_real']:.2%}). Non emergono "
                "evidenze forti di sintesi, ma l'assenza di artefatti visibili non "
                "garantisce l'autenticità."
            )
        if cam:
            lines.append(
                f"Il Grad-CAM concentra l'attenzione su regioni con saliency media "
                f"{cam.get('mean', 0):.2f} (max {cam.get('max', 0):.2f}), indicando "
                "le aree che hanno guidato la decisione."
            )
        if attr is not None:  # cascade: attribution solo per i fake
            dist = ", ".join(f"{k}={v:.2%}" for k, v in attr["distribution"].items())
            lines.append(
                f"Essendo classificata fake, si procede con l'attribution: il "
                f"generatore più probabile è '{attr['label']}' (conf. "
                f"{attr['confidence']:.2%}); distribuzione: {dist}. Da notare che, in "
                "questo dataset, ogni generatore coincide con una sorgente distinta, "
                "quindi l'attribuzione può riflettere la firma della sorgente più che "
                "quella dell'architettura generativa."
            )
        else:
            lines.append(
                "Essendo classificata reale, non si procede con l'attribution "
                "del generatore (non applicabile)."
            )
        return " ".join(lines)
