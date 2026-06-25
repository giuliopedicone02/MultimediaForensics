"""Scarica subset di immagini per classe dal datasets-server di HuggingFace.

Usa l'endpoint `/rows` (https://datasets-server.huggingface.co/rows) che restituisce
le immagini già estratte — così non si scaricano i parquet interi (centinaia di MB).
Le immagini vengono salvate in `data/<classe>/NNNNN.png`, pronte per `dffa`.

Esempio:
    python scripts/download_data.py --per-class 300
    python scripts/download_data.py --only real stylegan --per-class 500
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

ROWS_URL = "https://datasets-server.huggingface.co/rows"

# classe -> (dataset_id, config, split)
SOURCES = {
    "real":      ("bitmind/ffhq-256", "default", "train"),
    "stylegan":  ("34data/STYLEGAN", "default", "train"),
    "stylegan3": ("34data/stylegan3_T_FFHQU_processed", "default", "train"),
    # SDXL (diffusion) su FFHQ-256: 4a classe same-domain per l'attribution.
    "sdxl":      ("bitmind/ffhq-256___stable-diffusion-xl-base-1.0", "default", "train"),
}

PAGE = 100  # max righe per richiesta del datasets-server


def _get(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "dffa-downloader/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _image_from_row(row: dict) -> Image.Image | None:
    """Estrae un'immagine PIL da un campo 'image' del datasets-server.

    Il campo può essere: dict con 'src' (URL asset), dict con 'bytes' (base64),
    oppure direttamente una stringa base64.
    """
    img = row.get("image")
    try:
        if isinstance(img, dict):
            if img.get("src"):
                return Image.open(io.BytesIO(_get(img["src"]))).convert("RGB")
            if img.get("bytes"):
                return Image.open(io.BytesIO(base64.b64decode(img["bytes"]))).convert("RGB")
        elif isinstance(img, str):
            return Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
    except Exception as e:
        print(f"    ! errore immagine: {e}")
    return None


def download_class(name: str, per_class: int, out_root: Path) -> int:
    import json

    dataset, config, split = SOURCES[name]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # riprende da dove eventualmente interrotto
    existing = sorted(out_dir.glob("*.png"))
    saved = len(existing)
    offset = saved
    print(f"[{name}] da {dataset} -> {out_dir} (già presenti: {saved})")

    while saved < per_class:
        length = min(PAGE, per_class - saved)
        url = (
            f"{ROWS_URL}?dataset={dataset}&config={config}"
            f"&split={split}&offset={offset}&length={length}"
        )
        try:
            data = json.loads(_get(url))
        except Exception as e:
            print(f"    ! richiesta fallita (offset={offset}): {e}; retry...")
            time.sleep(3)
            continue

        rows = data.get("rows", [])
        if not rows:
            print("    fine dataset raggiunta.")
            break

        for item in rows:
            im = _image_from_row(item["row"])
            if im is None:
                continue
            im.save(out_dir / f"{saved:05d}.png")
            saved += 1
            if saved >= per_class:
                break
        offset += len(rows)
        print(f"    {saved}/{per_class}", end="\r", flush=True)
        time.sleep(0.2)  # cortesia verso il server

    print(f"\n[{name}] completato: {saved} immagini.")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=300, help="immagini per classe")
    ap.add_argument("--out", default="data", help="cartella di output")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="solo queste classi")
    args = ap.parse_args()

    out_root = Path(args.out)
    classes = args.only or list(SOURCES)
    totals = {}
    for name in classes:
        totals[name] = download_class(name, args.per_class, out_root)
    print("\n== Riepilogo ==")
    for name, n in totals.items():
        print(f"  {name:12s}: {n}")


if __name__ == "__main__":
    main()
