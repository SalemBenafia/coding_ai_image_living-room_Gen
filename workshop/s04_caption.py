#!/usr/bin/env python
"""Stage 04 — Generate BLIP captions (and filter room type).

Calls the caption-service (BLIP) batch endpoint, which reads the upscaled images
straight from MinIO — the "MinIO Image -> BLIP -> Caption" arrow in the design.
For stepanyarullin images (room_known=0) BLIP doubles as a room-type filter: if
the caption does not look like a living room, the image is dropped.

Reads data/upscaled.csv, writes data/captions.csv with a `blip_caption` column,
and stores each caption to MinIO under captions/<stem>.txt.

Falls back to a local in-process BLIP model if the caption-service is unreachable.

Usage:
  python s04_caption.py [--batch 16] [--no-upload]
"""
from __future__ import annotations

import argparse
import csv
import time

import httpx

from config import (
    BUCKET_DATA,
    CAPTION_SERVICE_URL,
    DATA_DIR,
    KEY_CAPTIONS_BLIP,
    KEY_IMAGES,
    minio_client,
)

# Words that indicate a living-room-like scene (room-type filter for unlabeled data).
LIVING_ROOM_HINTS = (
    "living room", "living-room", "livingroom", "sofa", "couch", "sofas",
    "sitting room", "lounge", "coffee table", "sectional", "loveseat",
)
# Rooms we explicitly do not want to leak in from unlabeled data.
OTHER_ROOM_HINTS = ("bedroom", "bathroom", "kitchen", "bed ", "toilet", "shower", "sink")


def looks_living_room(caption: str) -> bool:
    c = caption.lower()
    if any(h in c for h in OTHER_ROOM_HINTS) and not any(h in c for h in LIVING_ROOM_HINTS):
        return False
    return any(h in c for h in LIVING_ROOM_HINTS)


def caption_via_service(keys: list[str], batch: int) -> dict[str, str]:
    """Caption UNCONDITIONALLY (condition=None) so the text honestly reflects the
    room; this is what makes the downstream room-type filter meaningful."""
    out: dict[str, str] = {}
    url = f"{CAPTION_SERVICE_URL.rstrip('/')}/v1/caption/batch"
    with httpx.Client(timeout=180.0) as client:
        for i in range(0, len(keys), batch):
            chunk = [f"{KEY_IMAGES}/{k}" for k in keys[i:i + batch]]
            resp = client.post(url, json={"bucket": BUCKET_DATA, "keys": chunk,
                                          "condition": None})
            resp.raise_for_status()
            for item in resp.json()["items"]:
                stem = item["key"].split("/", 1)[-1]
                if item.get("caption"):
                    out[stem] = item["caption"]
            print(f"[caption] {min(i + batch, len(keys))}/{len(keys)}")
    return out


def caption_locally(filenames: list[str], batch: int) -> dict[str, str]:
    """Fallback: run BLIP in-process over the local upscaled images."""
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from config import UPSCALED_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "Salesforce/blip-image-captioning-large"
    print(f"[caption] local BLIP fallback on {device}")
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)

    out: dict[str, str] = {}
    for i in range(0, len(filenames), batch):
        chunk = filenames[i:i + batch]
        images = [Image.open(UPSCALED_DIR / f).convert("RGB") for f in chunk]
        # Unconditional captioning (honest room description for filtering).
        inputs = proc(images=images, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        for f, g in zip(chunk, gen):
            out[f] = proc.decode(g, skip_special_tokens=True).strip()
        print(f"[caption] {min(i + batch, len(filenames))}/{len(filenames)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    up_csv = DATA_DIR / "upscaled.csv"
    if not up_csv.exists():
        print(f"[caption] ERROR: {up_csv} missing. Run s03_upscale.py first.")
        return 1
    with up_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    filenames = [r["filename"] for r in rows]

    started = time.time()
    try:
        captions = caption_via_service(filenames, args.batch)
        source = "caption-service"
    except Exception as exc:  # noqa: BLE001
        print(f"[caption] caption-service unavailable ({exc}); using local fallback")
        captions = caption_locally(filenames, args.batch)
        source = "local"

    client = None if args.no_upload else minio_client()
    kept: list[dict] = []
    dropped_room = 0
    for row in rows:
        fn = row["filename"]
        cap = captions.get(fn)
        if not cap:
            continue
        # Room-type filter for unlabeled (stepanyarullin) images.
        if row.get("room_known", "1") == "0" and not looks_living_room(cap):
            dropped_room += 1
            continue
        row["blip_caption"] = cap
        kept.append(row)
        if client is not None:
            stem = fn.rsplit(".", 1)[0]
            data = cap.encode("utf-8")
            import io
            client.put_object(BUCKET_DATA, f"{KEY_CAPTIONS_BLIP}/{stem}.txt",
                              io.BytesIO(data), length=len(data), content_type="text/plain")

    out_csv = DATA_DIR / "captions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "source", "room_type", "style", "room_known", "blip_caption"]
        )
        writer.writeheader()
        writer.writerows(kept)

    print(f"[caption] via {source}: captioned {len(kept)}, "
          f"dropped {dropped_room} non-living-room, in {time.time() - started:.0f}s")
    if kept:
        print("[caption] example:", kept[0]["blip_caption"])
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
