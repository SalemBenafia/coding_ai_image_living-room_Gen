#!/usr/bin/env python
"""Step 5 — Assemble the LoRA training folder (diffusers imagefolder format).

Produces data/lora_dataset/ containing:
  * every prepared image (00001.jpg ...)
  * a matching <stem>.txt caption sidecar (for kohya-style trainers)
  * metadata.jsonl mapping {"file_name": "...", "text": "..."} — the format the
    diffusers `train_text_to_image_lora_sdxl.py` reads via --train_data_dir.

Usage:
  python 05_build_lora_dataset.py
"""
from __future__ import annotations

import json
import shutil

from common import CAPTIONS_DIR, LORA_DIR, PREPARED_DIR, ensure_dirs, find_images, load_dotenv


def main() -> int:
    load_dotenv()
    images = find_images(PREPARED_DIR)
    if not images:
        print(f"ERROR: no prepared images in {PREPARED_DIR}. Run steps 2–3 first.")
        return 1

    ensure_dirs(LORA_DIR)
    for old in LORA_DIR.iterdir():
        if old.is_file():
            old.unlink()

    rows: list[dict] = []
    missing = 0
    for img in images:
        cap_file = CAPTIONS_DIR / f"{img.stem}.txt"
        if not cap_file.exists():
            missing += 1
            continue
        caption = cap_file.read_text(encoding="utf-8").strip()
        if not caption:
            missing += 1
            continue
        dst = LORA_DIR / img.name
        shutil.copy2(img, dst)
        shutil.copy2(cap_file, LORA_DIR / cap_file.name)  # sidecar for kohya trainers
        rows.append({"file_name": img.name, "text": caption})

    meta_path = LORA_DIR / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[lora-dataset] {len(rows)} image/caption pairs -> {LORA_DIR}")
    if missing:
        print(f"[lora-dataset] skipped {missing} images without captions.")
    print(f"[lora-dataset] wrote {meta_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
