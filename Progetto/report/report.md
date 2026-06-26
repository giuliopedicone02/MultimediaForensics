# Deepfake Detection & Attribution con Explainability multi-stream

**Corso:** Multimedia Forensics — Laurea Magistrale (II anno), A.A. 2025-26 (UniCT)
**Autore:** Giulio Pedicone
**Data:** _(da compilare)_

---

## Abstract

Si presenta una pipeline forense per il **rilevamento** di volti sintetici e
l'**attribuzione** del generatore che li ha prodotti. Ogni immagine è analizzata
su due stream complementari — l'immagine RGB e lo **spettro di magnitudine della
FFT 2D** — ciascuno codificato da una ResNet18 pre-addestrata su ImageNet e
congelata; gli embedding (512+512 = 1024-d) alimentano un classificatore
multi-task con inferenza **a cascata** (detection real/fake → se *fake* →
attribution tra i generatori). Il sistema integra l'**explainability**: Grad-CAM
sui due stream e un **agent VLM open** (Qwen2.5-VL) che motiva in linguaggio
naturale il *perché* di detection e attribution. L'intero esperimento è
eseguibile gratuitamente su Google Colab (GPU T4). I risultati *in-distribution*
sono elevati (detection acc = 0.967, attribution acc sui fake = 0.991, cascade acc =
0.961) ma gli esperimenti di controllo mostrano che dipendono in larga parte da un
**confound di sorgente**: in un test *leave-one-generator-out* la detection su un
generatore mai visto crolla a 0.557 di media (recall sui fake nuovi fino a 0.00 per
SDXL). Il contributo principale di questo lavoro è quindi **metodologico**:
identificare, misurare e discutere onestamente tale confound.

## 1. Introduzione

- Contesto: diffusione dei volti sintetici (StyleGAN e successori, modelli
  diffusion come SDXL) e necessità forense di **rilevarli** e **attribuirne la
  sorgente**.
- Limite degli approcci "black-box": serve **explainability** (perché real/fake,
  perché quel generatore).
- Contributo di questo lavoro: pipeline multi-stream + classificatore a cascata +
  spiegazione in linguaggio naturale tramite VLM open, interamente eseguibile
  gratis su Colab T4.

## 2. Dati

- **Reali:** FFHQ (`bitmind/ffhq-256`).
- **Fake:** StyleGAN (`34data/STYLEGAN`), StyleGAN3
  (`34data/stylegan3_T_FFHQU_processed`), SDXL — diffusion
  (`bitmind/ffhq-256___stable-diffusion-xl-base-1.0`).
- Download: subset scaricato dal *datasets-server* di HuggingFace (endpoint
  `/rows`), **300 immagini per classe disponibili in locale** (1200 totali).
- Pre-processing: **risoluzione canonica** (tutte le immagini portate a 256×256
  con la stessa interpolazione bicubica *prima* di RGB e Fourier, vedi §3.1) →
  resize a 224×224, normalizzazione ImageNet.
- Split **stratificato** train/val/test = 70% / 15% / 15% (seed = 42).

| Classe | Sorgente HuggingFace | Risoluzione nativa | # immagini |
|--------|----------------------|-------------------:|-----------:|
| `real`      | `bitmind/ffhq-256`                                | 256×256   | 300 |
| `stylegan`  | `34data/STYLEGAN`                                 | 1024×1024 | 300 |
| `stylegan3` | `34data/stylegan3_T_FFHQU_processed`              | 244×244   | 300 |
| `sdxl`      | `bitmind/ffhq-256___stable-diffusion-xl-base-1.0` | 256×256   | 300 |

Conteggi per split (run con `max_per_class = 250`, seed = 42 → 1000 immagini,
split 696 / 152 / 152):

| Classe | # immagini | Train | Val | Test |
|--------|-----------:|------:|----:|-----:|
| real      | 250 | 174 | 38 | 38 |
| stylegan  | 250 | 174 | 38 | 38 |
| stylegan3 | 250 | 174 | 38 | 38 |
| sdxl      | 250 | 174 | 38 | 38 |

> **Confound della risoluzione (centrale in questo lavoro).** Le risoluzioni
> native sono eterogenee (real/sdxl 256, StyleGAN **1024**, StyleGAN3 244). Se ogni
> classe viene semplicemente portata a 224, il *fattore* di resampling — diverso per
> classe — lascia una firma che il classificatore sfrutta al posto dei veri artefatti
> generativi, **gonfiando** le metriche fino a valori irrealistici (attribution ≈
> 100%). Per neutralizzarlo applichiamo una **risoluzione canonica** (§3.1): tutte le
> immagini passano per la stessa identica catena di resize. L'effetto del confound e
> della sua rimozione è quantificato in §5.5.

## 3. Metodo

### 3.1 Risoluzione canonica (anti-confound)

Prima di qualunque elaborazione, **ogni** immagine — indipendentemente dalla
dimensione nativa — viene portata a `canonical_size × canonical_size` (256×256) con
la **stessa** interpolazione bicubica. Lo stesso tensore canonico alimenta sia lo
stream RGB sia lo spettro di Fourier. Questo è essenziale soprattutto per lo stream
Fourier: la FFT calcolata su una griglia 1024 (StyleGAN) è un oggetto diverso da
quella su una griglia 256 (SDXL); fissando la griglia a monte si rende lo spettro
confrontabile tra le classi e si rimuove la scorciatoia del resampling. Il
parametro è `cfg.canonical_size` (`None` = comportamento legacy, usato come termine
di paragone nell'ablation §5.5).

### 3.2 Stream RGB e Stream Fourier

Lo stream RGB lavora nel dominio spaziale (texture, artefatti semantici). Lo
stream Fourier converte l'immagine in scala di grigi, ne calcola la FFT 2D, centra
la frequenza zero (`fftshift`), prende la magnitudine in scala logaritmica
`log(1+|F|)` normalizzata in `[0,1]` e la replica su 3 canali. Motivazione: le
convoluzioni trasposte / l'up-sampling dei generatori introducono **artefatti
periodici** nello spettro, spesso invisibili nel dominio spaziale ma evidenti in
frequenza. I due stream sono quindi complementari.

### 3.3 Feature extraction (ResNet18 ImageNet, congelata)

Due ResNet18 indipendenti, pre-addestrate su ImageNet e **congelate**; rimossa la
`fc`, l'output è l'embedding 512-d dopo il global average pooling. Gli embedding
dei due stream sono concatenati (1024-d). Essendo i backbone congelati, gli
embedding sono **pre-calcolati una sola volta e messi in cache** su disco: il
training del classificatore dura pochi secondi. L'indipendenza dei backbone
consente di agganciare Grad-CAM separatamente a `layer4` di ciascuno stream.

### 3.4 Classificatore a cascata

MLP con tronco condiviso (2 layer `Linear+BN+ReLU+Dropout`) e due teste lineari:

- **Detection** (2 classi): `real` vs `fake`, addestrata su *tutti* i campioni.
- **Attribution** (G classi): *solo* i generatori (`stylegan`/`stylegan3`/`sdxl`,
  **niente `real`**), addestrata sui *soli* fake.

La loss è `w_det·CE(detection) + w_attr·CE(attribution, ignore_index=-1)`: i
campioni reali (label di attribution `-1`) sono ignorati dalla loss di
attribution. L'**inferenza a cascata** decide prima la detection e interpreta
l'attribution **solo se** l'immagine è classificata *fake*; poiché l'attribution
non contiene `real`, la decisione finale non può mai essere incoerente. La
selezione del modello usa la **cascade accuracy** (end-to-end) sul validation set.

### 3.5 Explainability

**Grad-CAM** sui due stream (gradienti rispetto a `layer4`, media spaziale, ReLU,
up-sampling): RGB indica *dove* nell'immagine, Fourier *quali* regioni di
frequenza. **Agent VLM**: Qwen2.5-VL-3B (4-bit su T4) riceve immagine RGB, spettro
di Fourier e overlay Grad-CAM insieme alle probabilità del classificatore, e —
guidato da un *system prompt* da esperto forense — produce una spiegazione che
distingue la motivazione della detection da quella dell'attribution. In assenza di
GPU/modello, un generatore template-based deterministico fornisce comunque una
spiegazione coerente con le evidenze numeriche.

## 4. Setup sperimentale

- **Hardware:** Google Colab GPU T4 (16 GB).
- **Pre-processing:** risoluzione canonica `canonical_size = 256` (bicubica), poi
  224×224 + normalizzazione ImageNet.
- **Backbone:** ResNet18 (ImageNet), congelata; embedding 512+512 = 1024-d.
- **Iperparametri:** epochs = 30, batch = 32, lr = 1e-3, weight decay = 1e-4,
  dropout = 0.3, hidden_dim = 256, `attribution_weight = detection_weight = 1.0`.
- **VLM:** `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit (nf4), `max_new_tokens = 512`.
- **Software:** PyTorch, torchvision, transformers. Seed = 42, cuDNN deterministico.

## 5. Risultati

### 5.1 Detection (real vs fake)

| Metrica | Valore |
|---------|-------:|
| Accuracy | 0.967 |
| Precision (macro) | 0.95 |
| Recall (macro) | 0.97 |
| F1 (macro) | 0.96 |

Per classe (test = 38 real, 114 fake): `real` precision 0.90 / recall 0.97;
`fake` precision 0.99 / recall 0.96.

### 5.2 Attribution (multi-generatore, solo sui fake)

- Accuracy (sui soli fake): **0.991** (113/114).
- Per generatore (F1): `stylegan` 0.99, `stylegan3` 1.00, `sdxl` 0.99.
- Matrice di confusione tra generatori: _(figura dal notebook, §7)_ — quasi diagonale.
- ⚠️ Questo valore è **gonfiato dal confound di sorgente** (vedi §5.5b: il solo
  stream RGB attribuisce al 100%) e va letto alla luce del LOGO (§5.6).

### 5.3 Cascade (end-to-end)

- Cascade accuracy (detection come gate → attribution): **0.961**.

### 5.4 Curve di training

_(loss e accuracy detection/attribution/cascade per epoca — figura prodotta dal
notebook, §6.)_

### 5.5 Ablation — confound della risoluzione e contributo degli stream

**(a) RAW vs CANONICA.** Stessa pipeline con e senza uniformazione della
risoluzione. Il crollo dell'accuracy (in particolare dell'attribution) misura
quanto il modello "RAW" si appoggiasse al confound invece che agli artefatti reali.

| Pipeline | Detection acc | Attribution acc (fake) | Cascade acc |
|----------|--------------:|-----------------------:|------------:|
| RAW (`canonical_size=None`) | 0.980 | 1.000 | 0.980 |
| CANONICA (256) | 0.967 | 0.991 | 0.961 |

> Osservazione (dal run): l'uniformazione della risoluzione sposta **poco** le
> metriche (es. attribution 1.00 → 0.99). La risoluzione era quindi solo **uno** dei
> confound: ne restano altri correlati alla sorgente (compressione JPEG, color
> grading, pipeline del dataset). L'ablation (b) e soprattutto il LOGO (§5.6) lo
> mostrano in modo netto.

**(b) Contributo degli stream** (sui dati canonici): solo RGB, solo Fourier, entrambi.

| Stream | Detection acc | Attribution acc (fake) | Cascade acc |
|--------|--------------:|-----------------------:|------------:|
| RGB-only | 0.961 | 1.000 | 0.961 |
| Fourier-only | 0.763 | 0.842 | 0.671 |
| Both | 0.961 | 0.991 | 0.954 |

Il segnale è **dominato dall'RGB** (attribution perfetta da solo); lo stream Fourier
— quello motivato dagli artefatti di frequenza — è nettamente il più debole. Questo
ridimensiona l'ipotesi di partenza e indica che l'RGB cattura indizi di sorgente
(colore/compressione) più che veri artefatti generativi.

> Permette di capire se i due stream sono davvero complementari o se uno domina.
> _(da compilare dal run.)_

### 5.6 Generalizzazione — Leave-One-Generator-Out (LOGO)

Esperimento chiave per la validità forense. Per ogni generatore `g`: detection
addestrata su `real + (altri generatori)`, testata su `real + g`. Misura se il
rilevatore generalizza a una sorgente **mai vista** o se ha solo memorizzato la
firma di ciascun dataset.

| Held-out (mai visto) | Detection acc | Recall sui fake mai visti |
|----------------------|--------------:|--------------------------:|
| StyleGAN | 0.566 | 0.132 |
| StyleGAN3 | 0.618 | 0.289 |
| SDXL | 0.487 | 0.000 |
| **media** | **0.557** | — |

Riferimento in-distribution: detection acc = 0.967. La detection **crolla a 0.557**
(vicino al caso, dato che il test è ~50% real / 50% fake) e il **recall sui fake mai
visti precipita**: 0.13, 0.29 e **0.00** per SDXL. Il caso SDXL è il più indicativo:
è un modello *diffusion*, mentre l'addestramento (sui soli StyleGAN/StyleGAN3) ha
visto solo *GAN* → il rilevatore classifica **tutti** gli SDXL come reali. È la prova
diretta che le metriche in-distribution riflettono la memorizzazione della firma
delle sorgenti note, non una reale capacità di rilevare contenuto sintetico
sconosciuto.

## 6. Analisi qualitativa dell'explainability

Per alcuni campioni di test si riportano immagine RGB, spettro di Fourier, overlay
Grad-CAM e la spiegazione generata dall'agent (campo `source`: `vlm` o `template`).

- **Esempio FAKE corretto:** _(immagini + testo della spiegazione.)_
- **Esempio REAL corretto:** _(…)_
- **Caso di errore:** _(…)_ — commento su cosa ha "ingannato" il modello.

## 7. Discussione

- **Risultato principale (onesto).** Le metriche in-distribution sono alte
  (detection ≈ 0.97, attribution ≈ 0.99) ma **non vanno interpretate come reale
  capacità forense**: gli esperimenti di controllo mostrano che derivano in larga
  parte da un **confound di sorgente**. Nel nostro dataset *generatore* e *dataset di
  origine* sono perfettamente correlati (ogni classe = un dataset HF distinto, con la
  sua compressione, risoluzione e color pipeline), quindi il modello può separare le
  classi dalla "firma" della sorgente anziché dagli artefatti generativi.
- **Contributo degli stream (§5.5b).** Il segnale è dominato dallo stream **RGB**
  (attribution ≈ 1.00 da solo), mentre lo stream **Fourier** — quello motivato dagli
  artefatti di frequenza — è il più debole. Questo ridimensiona l'ipotesi di partenza
  e suggerisce che l'RGB sta catturando indizi di sorgente (colore/compressione).
- **Generalizzazione (§5.6, LOGO).** Su un generatore mai visto la detection
  **crolla** rispetto all'in-distribution: prova diretta che il rilevatore non
  generalizza alla "fakeness" in senso lato, ma memorizza le sorgenti note.
- **Confound della risoluzione (§5.5a).** Identificato e mitigato con la risoluzione
  canonica (§3.1), ma da solo sposta poco le metriche: era una delle cause, non
  l'unica.
- **Fedeltà del VLM.** La spiegazione cita con sicurezza artefatti di frequenza
  ("griglie periodiche") che però l'ablation mostra poco usati dal modello (Fourier
  debole): la spiegazione è fluente ma non sempre **fedele** al reale processo
  decisionale — limite noto degli explainer generativi.
- **Altri limiti:** dimensione contenuta del dataset; un solo dominio (volti);
  robustezza a compressione/resize non valutata sistematicamente; fedeltà delle
  spiegazioni del VLM da verificare (rischio di artefatti "plausibili" ma non reali).

## 8. Conclusioni e sviluppi futuri

Abbiamo realizzato una pipeline completa di detection + attribution a cascata con
explainability (Grad-CAM + agent VLM), eseguibile gratuitamente su Colab T4. I
risultati in-distribution sono elevati (detection 0.967, attribution 0.991), ma
l'analisi critica — risoluzione canonica, ablation degli stream e soprattutto il test
*leave-one-generator-out* — dimostra che tali valori sono in gran parte un artefatto
del **confound di sorgente**: nel dataset i generatori coincidono con dataset di
origine distinti, e il modello ne memorizza la firma invece di apprendere la
"sinteticità" in generale. Su un generatore mai visto la detection scende a 0.557
(recall 0.00 su SDXL). Il valore del lavoro è quindi metodologico: mostrare come si
smaschera un risultato troppo bello per essere vero.

**Sviluppi futuri.** (i) Un benchmark *controllato* in cui i fake siano generati
dalla **stessa** pipeline a partire dagli stessi reali (per scorporare generatore e
sorgente); (ii) normalizzazione aggressiva (ricompressione JPEG uniforme, color
matching) e augmentation che rompano le firme di sorgente; (iii) valutazione di
robustezza (JPEG/resize/blur); (iv) fine-tuning leggero (LoRA) dei backbone e
calibrazione della confidenza; (v) protocollo di valutazione *cross-source* come
metrica primaria al posto dell'accuracy in-distribution.

## Riferimenti

- Karras et al., *A Style-Based Generator Architecture for GANs* (StyleGAN), CVPR 2019.
- Karras et al., *Alias-Free GAN* (StyleGAN3), NeurIPS 2021.
- Podell et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, 2023.
- Wang et al., *CNN-generated images are surprisingly easy to spot... for now*, CVPR 2020.
- Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition*, ICML 2020.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.
- Bai et al., *Qwen2.5-VL Technical Report*, 2025.
