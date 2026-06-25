"""dffa — Deepfake Forensics & Attribution.

Pipeline multi-stream per deepfake *detection* e *attribution* con explainability:

1. Feature extraction dual-stream
   - stream RGB:     immagine -> ResNet18 (pre-addestrata su ImageNet) -> embedding
   - stream Fourier: spettro di magnitudine FFT -> ResNet18 -> embedding
2. Classificazione multi-task (MLP) sugli embedding concatenati
   - testa di *detection*   : real vs fake
   - testa di *attribution* : real / StyleGAN / StyleGAN2 / StyleGAN3 / ...
3. Explainability
   - Grad-CAM sui due stream (dove guarda la rete)
   - VLM agent open (Qwen2.5-VL) che osserva RGB + Fourier + Grad-CAM e
     produce la spiegazione in linguaggio naturale del *perché* della decisione.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
