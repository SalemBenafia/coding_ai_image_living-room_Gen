#!/usr/bin/env python
"""Step 3 — Generate captions with BLIP for every prepared image.

Writes data/captions/<stem>.txt. Uses a conditional prompt so BLIP stays on-topic
for interiors, then prefixes an interior-design trigger phrase.

Usage:
  python 03_generate_captions.py [--model Salesforce/blip-image-captioning-large]
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from common import CAPTIONS_DIR, PREPARED_DIR, ensure_dirs, find_images, load_dotenv

# Prepended to every caption so the LoRA learns a consistent trigger context.
TRIGGER = "a living room interior,"
CONDITION = "a living room with"  # BLIP conditional-captioning seed


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    args = ap.parse_args()

    images = find_images(PREPARED_DIR)
    if not images:
        print(f"ERROR: no images in {PREPARED_DIR}. Run 02_prepare_dataset.py first.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[caption] loading BLIP '{args.model}' on {device}")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)

    ensure_dirs(CAPTIONS_DIR)
    for i, path in enumerate(images, 1):
        image = Image.open(path).convert("RGB")
        inputs = processor(image, CONDITION, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        # Ensure the trigger context is present and de-duplicated.
        if "living room" not in caption.lower():
            caption = f"{TRIGGER} {caption}"
        (CAPTIONS_DIR / f"{path.stem}.txt").write_text(caption, encoding="utf-8")
        if i % 25 == 0 or i == len(images):
            print(f"[caption] {i}/{len(images)}  e.g. -> {caption}")

    print(f"[caption] wrote {len(images)} captions -> {CAPTIONS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
