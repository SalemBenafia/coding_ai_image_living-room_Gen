#!/usr/bin/env python
"""Stage 06 — Assemble the LoRA training dataset.

Produces a HuggingFace `imagefolder` directory that the diffusers SDXL LoRA
trainer consumes directly:

    data/lora_dataset/
        00001_galinakg_modern.jpg
        ...
        metadata.jsonl          # {"file_name": "...", "text": "<caption>"}

Reads data/captions_final.csv + data/upscaled/. Optionally mirrors the finished
dataset to MinIO for provenance.

Usage:
  python s06_build_dataset.py [--upload]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil

from config import (
    BUCKET_DATA,
    DATA_DIR,
    LORA_DATASET_DIR,
    TRIGGER,
    UPSCALED_DIR,
    ensure_dirs,
    minio_client,
)

# Never train on a caption that describes the wrong room; for folder-labeled
# living rooms (galinakg) BLIP/Qwen occasionally mislabel — we repair those.
_WRONG_ROOM = ("bedroom", "bed ", "queen size", "king size", "crib", "nightstand",
               "bathroom", "toilet", "shower", "kitchen", "dining room")
_QUALITY = ("interior design photography, architectural photography, "
            "highly detailed, natural lighting")


def repair_caption(caption: str, style: str, room_known: bool) -> tuple[str, str]:
    """Return (caption, action). Rebuild captions that are runaway or wrong-room.

    action: 'ok' | 'trimmed' | 'rebuilt'
    """
    phrases = [p.strip() for p in caption.split(",") if p.strip()]
    # Analyse the caption WITHOUT the trigger phrase — the trigger literally
    # contains "living room", which would otherwise mask a wrong-room description.
    body = caption.lower().replace(TRIGGER.lower(), " ")
    wrong_room = (
        room_known
        and any(w in body for w in _WRONG_ROOM)
        and "living room" not in body
        and "living area" not in body
    )

    if wrong_room or len(phrases) > 30:
        # Rebuild a clean, correct caption from the known style.
        base = [f"{TRIGGER}", f"a {style} living room" if style else "a living room"]
        base += [_QUALITY]
        seen, out = set(), []
        for tok in ", ".join(base).split(","):
            t = tok.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return ", ".join(out), "rebuilt"

    if len(phrases) > 24:
        return ", ".join(phrases[:24]), "trimmed"
    return caption, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    final_csv = DATA_DIR / "captions_final.csv"
    if not final_csv.exists():
        print(f"[build] ERROR: {final_csv} missing. Run s05_enhance.py first.")
        return 1
    with final_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    ensure_dirs(LORA_DATASET_DIR)
    for old in LORA_DATASET_DIR.iterdir():
        if old.is_file():
            old.unlink()

    meta_path = LORA_DATASET_DIR / "metadata.jsonl"
    n = 0
    actions = {"ok": 0, "trimmed": 0, "rebuilt": 0}
    with meta_path.open("w", encoding="utf-8") as meta:
        for row in rows:
            src = UPSCALED_DIR / row["filename"]
            caption = (row.get("caption") or "").strip()
            if not src.exists() or not caption:
                continue
            caption, action = repair_caption(
                caption, row.get("style", ""), row.get("room_known", "1") == "1"
            )
            actions[action] += 1
            shutil.copy2(src, LORA_DATASET_DIR / row["filename"])
            meta.write(json.dumps({"file_name": row["filename"], "text": caption}) + "\n")
            n += 1

    print(f"[build] LoRA dataset ready: {n} pairs -> {LORA_DATASET_DIR}")
    print(f"[build] caption QC: {actions}")
    print(f"[build] metadata: {meta_path}")

    if args.upload:
        client = minio_client()
        for f in LORA_DATASET_DIR.iterdir():
            client.fput_object(BUCKET_DATA, f"lora_dataset/{f.name}", str(f))
        print(f"[build] mirrored dataset -> {BUCKET_DATA}/lora_dataset/")

    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
