# Deepfake Detection & Attribution con Explainability multi-stream

**Corso:** Multimedia Forensics — Laurea Magistrale (II anno)
**Autore:** Giulio Pedicone
**Data:** _(da compilare)_

---

## Abstract

_(2–3 frasi: problema, approccio multi-stream RGB+Fourier con ResNet18, classificatore
multi-task per detection e attribution, explainability con Grad-CAM e agent VLM open,
risultati principali.)_

## 1. Introduzione

- Contesto: diffusione dei volti sintetici (StyleGAN e successori) e necessità
  forense di **rilevarli** e **attribuirne la sorgente**.
- Limite degli approcci "black-box": serve **explainability** (perché real/fake,
  perché quel generatore).
- Contributo di questo lavoro: pipeline multi-stream + spiegazione in linguaggio
  naturale tramite VLM open, interamente eseguibile gratis su Colab T4.

## 2. Dati

- **Reali:** FFHQ (n = _…_).
- **Fake:** StyleGAN / StyleGAN2 / StyleGAN3 (n per classe = _…_).
- Pre-processing: resize 224×224, normalizzazione ImageNet.
- Split stratificato train/val/test = _…/…/…_ (seed = 42).

| Classe | # immagini | Train | Val | Test |
|--------|-----------:|------:|----:|-----:|
| real      | | | | |
| stylegan  | | | | |
| stylegan2 | | | | |
| stylegan3 | | | | |

## 3. Metodo

### 3.1 Stream RGB e Stream Fourier
_(descrivere i due stream; perché la frequenza espone gli artefatti GAN.)_

### 3.2 Feature extraction (ResNet18 ImageNet, congelata)
_(embedding 512+512 = 1024-d; cache.)_

### 3.3 Classificatore a cascata
_(MLP a tronco condiviso; testa di detection su tutti i campioni; testa di
attribution sui soli generatori, addestrata solo sui fake con `ignore_index=-1`.
Inferenza a cascata: detection → se fake → attribution. Coerenza garantita.
Selezione del modello sulla cascade accuracy.)_

### 3.4 Explainability
_(Grad-CAM sui due stream; agent VLM open Qwen2.5-VL e suo ruolo.)_

## 4. Setup sperimentale

- Hardware: Google Colab GPU T4.
- Iperparametri: epochs = 30, batch = 32, lr = 1e-3, weight decay = 1e-4, dropout = 0.3.
- Software: PyTorch, torchvision, transformers. Seed = 42.

## 5. Risultati

### 5.1 Detection (real vs fake)

| Metrica | Valore |
|---------|-------:|
| Accuracy | |
| Precision | |
| Recall | |
| F1 | |

### 5.2 Attribution (multi-generatore, solo sui fake)

- Accuracy (sui soli fake): _…_
- Matrice di confusione tra generatori: _(figura)_
- Classification report per generatore: _(tabella)_

### 5.3 Cascade (end-to-end)

- Cascade accuracy (detection come gate → attribution): _…_

### 5.4 Curve di training
_(loss e accuracy detection/attribution/cascade per epoca — figura.)_

## 6. Analisi qualitativa dell'explainability

Per alcuni campioni di test si riportano immagine RGB, spettro di Fourier, overlay
Grad-CAM e la spiegazione generata dall'agent.

- **Esempio FAKE corretto:** _(immagini + testo della spiegazione.)_
- **Esempio REAL corretto:** _(…)_
- **Caso di errore:** _(…)_ — commento su cosa ha "ingannato" il modello.

## 7. Discussione

- Contributo relativo dei due stream (RGB vs Fourier).
- Qualità e fedeltà delle spiegazioni del VLM rispetto alle evidenze numeriche.
- Limiti: dimensione del dataset, un solo dominio (volti), robustezza a
  compressione/resize non valutata.

## 8. Conclusioni e sviluppi futuri

_(sintesi dei risultati; estensioni: generatori diffusion, LoRA fine-tuning,
calibrazione confidenza, test di robustezza.)_

## Riferimenti

- Karras et al., *A Style-Based Generator Architecture for GANs* (StyleGAN), CVPR 2019.
- Karras et al., *Analyzing and Improving the Image Quality of StyleGAN* (StyleGAN2), CVPR 2020.
- Karras et al., *Alias-Free GAN* (StyleGAN3), NeurIPS 2021.
- Wang et al., *CNN-generated images are surprisingly easy to spot... for now*, CVPR 2020.
- Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition*, ICML 2020.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.
- Bai et al., *Qwen2.5-VL Technical Report*, 2025.
