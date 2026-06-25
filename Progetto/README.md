# DFFA — Deepfake Forensics & Attribution

Sistema multi-stream per **deepfake detection** e **deepfake attribution** con
**explainability**, sviluppato per il corso di *Multimedia Forensics* (Laurea
Magistrale). Il sistema non si limita a dire *se* un volto è reale o sintetico e
*da quale generatore* proviene: spiega anche **il perché** della decisione,
combinando evidenza visiva (Grad-CAM, spettro di Fourier) con un agent VLM open
che genera la motivazione in linguaggio naturale.

> Progettato per girare gratuitamente su **Google Colab (GPU T4)**.

---

## Idea in breve

Ogni immagine viene analizzata su **due stream complementari** e date in pasto a
backbone ResNet18 pre-addestrate su ImageNet:

| Stream | Input | Cosa cattura |
|--------|-------|--------------|
| **RGB** | immagine originale | artefatti semantici/texture nel dominio spaziale |
| **Fourier** | spettro di magnitudine (log) della FFT 2D | griglie/picchi periodici tipici dell'up-sampling GAN |

Gli **embedding** dei due stream (512-d ciascuno) vengono concatenati (1024-d) e
passati ad un **classificatore a cascata** (tronco condiviso, due teste):

- **Detection** → `real` vs `fake` (su tutti i campioni)
- **Attribution** → *solo se fake* → `StyleGAN` / `StyleGAN3` / `SDXL` (solo generatori)

La cascata rispecchia il flusso forense ("è fake? se sì, di chi è la firma?") e
garantisce **coerenza**: l'attribution non contiene `real`, quindi non può mai
contraddire la detection.

L'**explainability** è a due livelli:

1. **Grad-CAM** sui due backbone → *dove* guarda la rete (immagine e frequenze).
2. **Agent VLM open** (Qwen2.5-VL, eseguito localmente sulla T4) che osserva
   RGB + Fourier + Grad-CAM insieme alle probabilità del classificatore e produce
   la spiegazione discorsiva del *perché* real/fake e, se fake, del *perché* di
   quella attribuzione.

```
                ┌─────────────┐   embedding
   immagine ───▶│ ResNet18 RGB │──────────────┐
       │        └─────────────┘               │
       │ FFT                                   ▼
       ▼        ┌─────────────┐   embedding  ┌────────────┐  ① detection (real/fake)
   spettro ────▶│ ResNet18 FFT │────────────▶│ MLP +      │
                └─────────────┘   concat 1024 │ 2 teste    │─▶ ② se fake: attribution
                                              └────────────┘      (StyleGAN/3/SDXL)
                Grad-CAM (RGB+FFT) ┐
                probabilità        ├──▶  Agent VLM  ──▶  spiegazione NL (perché)
                immagini           ┘
```

---

## Struttura del repository

```
Progetto/
├── README.md
├── pyproject.toml            # package installabile (pip install -e .)
├── requirements.txt
├── configs/
│   └── default.yaml          # iperparametri e percorsi
├── dffa/                     # libreria
│   ├── config.py             # dataclass di configurazione (YAML)
│   ├── data/dataset.py       # dataset dual-stream + split stratificati
│   ├── features/
│   │   ├── fourier.py        # spettro di Fourier
│   │   └── extractor.py      # ResNet18 embedder (RGB + Fourier)
│   ├── models/classifier.py  # MLP multi-task (detection + attribution)
│   ├── explain/
│   │   ├── gradcam.py        # Grad-CAM + overlay
│   │   └── vlm_agent.py      # agent VLM open (+ fallback template)
│   ├── engine.py             # estrazione embedding (cache), train, eval
│   └── utils/common.py       # seed, device, I/O
├── notebooks/
│   └── 01_deepfake_forensics_colab.ipynb   # pipeline end-to-end (Colab)
├── report/report.md          # relazione
├── docs/architecture.md      # dettaglio architetturale
├── data/                     # dataset (NON versionato)
└── results/                  # metriche, figure, spiegazioni (NON versionato)
```

---

## Dataset

Layout atteso (una sottocartella per classe di attribution):

```
data/
├── real/         # FFHQ — volti reali
├── stylegan/     # StyleGAN
├── stylegan2/    # StyleGAN2
└── stylegan3/    # StyleGAN3
```

Indicazioni:

- **Reali**: [FFHQ](https://github.com/NVlabs/ffhq-dataset) (o un subset, es.
  `thumbnails128x128`).
- **Fake**: immagini generate da StyleGAN/StyleGAN2/StyleGAN3 (es. i checkpoint
  ufficiali NVlabs o subset già pronti su HuggingFace Hub / Kaggle).
- Per un bilanciamento ~1000 immagini, usa `max_per_class` in `configs/default.yaml`
  (es. 250 per classe). Lo split è **stratificato** e riproducibile (`seed`).

> Le classi assenti su disco vengono ignorate automaticamente: puoi partire con
> `real` + `stylegan` e aggiungere gli altri generatori in un secondo momento.

---

## Quickstart (Google Colab, T4)

1. Apri `notebooks/01_deepfake_forensics_colab.ipynb` in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Esegui le celle in ordine: setup → dati → feature → training → valutazione →
   Grad-CAM → spiegazioni VLM → salvataggio risultati.

## Quickstart (locale)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vlm]"          # libreria + dipendenze VLM
# disponi i dati in data/<classe>/ e poi apri il notebook
```

> Senza GPU il classificatore e il Grad-CAM funzionano comunque (più lenti);
> il VLM in 4-bit richiede CUDA — in assenza, l'agent ricade automaticamente su
> una spiegazione template-based deterministica.

---

## Explainability: come legge l'output

- **Detection** — la spiegazione cita gli artefatti nel dominio della frequenza
  (griglie periodiche dello spettro), la coerenza di texture/illuminazione e le
  regioni evidenziate dal Grad-CAM sull'immagine RGB.
- **Attribution** — la spiegazione si appoggia alla *firma spettrale* specifica
  dell'architettura generativa e alla distribuzione di probabilità tra i
  generatori candidati.

---

## Riproducibilità

- Seed globale (Python/NumPy/PyTorch) e cuDNN deterministico (`dffa/utils`).
- Configurazione interamente serializzata in YAML.
- Embedding pre-calcolati e messi in cache: il training del classificatore è
  deterministico e dura pochi secondi.

## Licenza

MIT — vedi [LICENSE](LICENSE).
