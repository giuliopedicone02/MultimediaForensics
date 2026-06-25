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
    "Sei un esperto di multimedia forensics. Ti vengono mostrate l'immagine "
    "originale, il suo spettro di Fourier e una mappa Grad-CAM che indica le "
    "regioni su cui il classificatore si è basato. Ricevi inoltre le predizioni "
    "numeriche di un classificatore (detection real/fake e attribution del "
    "generatore). Il tuo compito è spiegare in modo chiaro e tecnico PERCHÉ "
    "l'immagine è classificata così, citando evidenze visive concrete: artefatti "
    "GAN, incoerenze di texture/illuminazione, picchi/griglie periodiche nello "
    "spettro di Fourier, e le regioni evidenziate dal Grad-CAM. Distingui la "
    "motivazione della DETECTION da quella dell'ATTRIBUTION. Sii conciso (5-8 frasi)."
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
                "Spiega perché l'immagine è FAKE e perché è attribuita a quel "
                "generatore, citando le evidenze visive."
            )
        else:  # real: nessuna attribution
            attr_line = (
                "- Attribution: non applicabile (immagine classificata reale).\n\n"
                "Spiega perché l'immagine è REAL, citando le evidenze visive "
                "(assenza di artefatti GAN/diffusion, spettro senza griglie periodiche)."
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
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
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
                f"L'immagine è classificata FAKE con confidenza {det['confidence']:.2%}"
                f" (p_fake={det['p_fake']:.2%}). Il segnale dominante proviene "
                "tipicamente da artefatti nel dominio della frequenza: lo spettro "
                "di Fourier mostra picchi/griglie periodiche dovuti all'up-sampling "
                "del generatore, assenti nelle immagini reali."
            )
        else:
            lines.append(
                f"L'immagine è classificata REAL con confidenza {det['confidence']:.2%}"
                f" (p_real={det['p_real']:.2%}). Lo spettro di Fourier non presenta "
                "le griglie periodiche tipiche delle GAN e le texture risultano "
                "coerenti con un sensore fotografico reale."
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
                f"Essendo fake, si procede con l'attribution: la sorgente più "
                f"probabile è '{attr['label']}' (conf. {attr['confidence']:.2%}); "
                f"distribuzione tra i generatori: {dist}. La firma spettrale "
                "specifica dell'architettura giustifica questa scelta."
            )
        else:
            lines.append(
                "Essendo classificata reale, non si procede con l'attribution "
                "del generatore (non applicabile)."
            )
        return " ".join(lines)
