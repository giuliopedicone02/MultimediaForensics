# Architettura del sistema

Documento di dettaglio della pipeline DFFA. Per la visione d'insieme vedi il
[README](../README.md).

## 1. Pre-processing e stream

**Risoluzione canonica (anti-confound).** Le classi hanno risoluzioni native
eterogenee (real/sdxl 256, StyleGAN 1024, StyleGAN3 244). Se si ridimensiona
direttamente a 224, il *fattore* di resampling — diverso per classe — diventa una
firma che il classificatore sfrutta al posto dei veri artefatti generativi,
gonfiando le metriche (attribution ≈ 100%). Per evitarlo, ogni immagine viene prima
portata a `canonical_size × canonical_size` (default 256) con la **stessa**
interpolazione bicubica; lo stesso tensore canonico alimenta entrambi gli stream.
Con `cfg.canonical_size = None` si ripristina il comportamento legacy, usato come
termine di paragone nell'ablation (vedi `engine.slice_stream` e la sezione 7-bis del
notebook).

Dopo l'uniformazione, ogni immagine è ridimensionata a `224×224` e normalizzata con
le statistiche di ImageNet (coerenza col backbone pre-addestrato).

- **Stream RGB**: immagine normalizzata → ResNet18.
- **Stream Fourier**: si converte l'immagine in scala di grigi, si calcola la
  FFT 2D, si centra la frequenza zero (`fftshift`), si prende la magnitudine in
  scala logaritmica `log(1 + |F|)`, si normalizza in `[0,1]` e si replica su 3
  canali → ResNet18.
  - **Normalizzazione robusta (`fourier_robust`, default on).** La componente DC
    (frequenza 0) ha magnitudine enorme: con un semplice min-max globale dominerebbe
    la scala, comprimendo in un range minimo gli artefatti periodici di media/alta
    frequenza — proprio il segnale forense utile. Si clippa quindi lo spettro al
    percentile `fourier_clip_pct` (default 99) prima di scalare, espandendo la
    dinamica delle frequenze informative. Con `fourier_robust=False` si ripristina il
    min-max globale (termine di paragone per l'ablation).

Motivazione: i generatori GAN introducono **artefatti periodici** nello spettro
(dovuti alle convoluzioni trasposte / up-sampling), spesso invisibili nel dominio
spaziale ma evidenti in frequenza. I due stream sono quindi complementari.

## 2. Feature extraction

Due ResNet18 indipendenti, pre-addestrate su ImageNet e **congelate**. Si rimuove
la `fc` finale: l'output è l'embedding 512-d dopo il global average pooling.
L'indipendenza dei due backbone consente di agganciare Grad-CAM separatamente a
`layer4` di ciascuno stream.

Gli embedding vengono concatenati: `[emb_rgb (512) | emb_fourier (512)] = 1024-d`.

> Poiché i backbone sono congelati, gli embedding sono pre-calcolati una volta e
> messi in cache (`dffa/engine.py`). Il training successivo è velocissimo.

## 3. Classificatore a cascata

MLP con tronco condiviso (2 layer `Linear+BN+ReLU+Dropout`) e due teste lineari:

- **Detection** (2 classi): `real` vs `fake` — addestrata su *tutti* i campioni.
- **Attribution** (G classi): *solo i generatori* (`stylegan` / `stylegan3` / `sdxl`,
  **niente `real`**) — addestrata solo sui fake.

**Inferenza a cascata** (come in un flusso forense reale): si decide prima la
detection; l'attribution si interpreta **solo se** l'immagine è classificata *fake*.
Poiché l'attribution non contiene la classe `real`, la decisione finale non può mai
essere incoerente (mai "fake ma attribuito a real").

Loss = `w_det · CE(detection) + w_attr · CE(attribution, ignore_index=-1)`: i campioni
reali (label di attribution `-1`) sono ignorati dalla loss di attribution. La
condivisione della rappresentazione regolarizza entrambi i task. La selezione del
modello usa la **cascade accuracy** (end-to-end) sul validation set.

Metriche riportate:
- `detection_acc` — su tutti i campioni;
- `attribution_acc` — sui soli fake, tra i generatori;
- `cascade_acc` — end-to-end con la detection che fa da *gate*.

## 4. Explainability

### 4.1 Grad-CAM
Per la classe predetta si calcolano i gradienti rispetto alle attivazioni di
`layer4`; la media spaziale dei gradienti pesa le mappe di attivazione, seguita da
ReLU e up-sampling alla risoluzione di input. Prodotto su entrambi gli stream:
- RGB → *dove* nell'immagine.
- Fourier → *quali* regioni di frequenza.

### 4.2 Agent VLM
Un VLM open (Qwen2.5-VL-3B, 4-bit su T4) riceve:
1. le immagini (RGB, spettro di Fourier, overlay Grad-CAM);
2. l'evidenza numerica del classificatore (probabilità di detection/attribution,
   statistiche del Grad-CAM).

Tramite un *system prompt* da esperto forense, l'agent produce una spiegazione che
**distingue** la motivazione della detection da quella dell'attribution, citando
evidenze concrete. Coerentemente con la cascata, l'attribution viene spiegata
**solo se** l'immagine è classificata *fake*; se è *real*, la spiegazione si limita
alla detection (assenza di artefatti GAN/diffusion). In assenza di GPU/modello, un
generatore template-based deterministico fornisce comunque una spiegazione coerente.

## 5. Valutazione

- Accuracy di detection e attribution.
- Matrici di confusione e classification report (`scikit-learn`).
- Ispezione qualitativa: per alcuni campioni di test si mostrano immagine,
  spettro, Grad-CAM e la spiegazione generata.

## 6. Riproducibilità

Seed globale, cuDNN deterministico, split stratificato per seed, configurazione in
YAML, embedding in cache. Vedi `dffa/utils/common.py` e `dffa/config.py`.

## 7. Estensioni possibili

- Aggiungere generatori (StyleGAN-XL, diffusion: SDXL, Midjourney) → attribution
  più ricca, cambiando solo `classes` in config.
- Fine-tuning leggero (LoRA) dei backbone.
- Calibrazione della confidenza (temperature scaling) per spiegazioni più affidabili.
- Metriche di robustezza (JPEG, resize, blur) — tipiche della forensics.
