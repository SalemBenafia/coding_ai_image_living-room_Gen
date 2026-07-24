#!/usr/bin/env python
"""Stage 03 — Real-ESRGAN upscale + resize to the training resolution.

Both Kaggle interior datasets are web thumbnails (galinakg ~236px, stepanyarullin
~360px short side). Training SDXL at 1024 directly on them teaches the VAE to
reproduce upscaling artifacts. So we run Real-ESRGAN x4 to recover clean edges,
then resize to TARGET_SIZE (default 1024, center-cropped square).

Reads data/prepared/ + prepared.csv, writes data/upscaled/ + upscaled.csv, and
(optionally) uploads the 1024px images to MinIO for the caption service to read.

Usage:
  python s03_upscale.py [--size 1024] [--upload] [--limit 0]
"""
from __future__ import annotations

import argparse
import csv

from PIL import Image

from config import (
    BUCKET_DATA,
    DATA_DIR,
    KEY_IMAGES,
    PREPARED_DIR,
    TARGET_SIZE,
    UPSCALED_DIR,
    ensure_buckets,
    ensure_dirs,
    minio_client,
)
from upscaler import Upscaler, resize_to


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=TARGET_SIZE)
    ap.add_argument("--crop", choices=["cover", "contain"], default="cover")
    ap.add_argument("--upload", action="store_true", help="upload 1024px images to MinIO")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prepared_csv = DATA_DIR / "prepared.csv"
    if not prepared_csv.exists():
        print(f"[upscale] ERROR: {prepared_csv} missing. Run s02_prepare.py first.")
        return 1
    with prepared_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    ensure_dirs(UPSCALED_DIR)
    for old in UPSCALED_DIR.glob("*.jpg"):
        old.unlink()

    print(f"[upscale] loading Real-ESRGAN; processing {len(rows)} images -> {args.size}px")
    up = Upscaler()

    client = None
    if args.upload:
        ensure_buckets()
        client = minio_client()

    out_rows: list[dict] = []
    for i, row in enumerate(rows, 1):
        src = PREPARED_DIR / row["filename"]
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                # Only upscale when it actually helps (short side below target).
                if min(im.size) < args.size:
                    im = up.upscale(im)
                final = resize_to(im, args.size, mode=args.crop)
        except Exception as exc:  # noqa: BLE001
            print(f"[upscale] skip {src.name}: {exc}")
            continue

        dst = UPSCALED_DIR / row["filename"]
        final.save(dst, quality=95)
        out_rows.append({**row})
        if client is not None:
            key = f"{KEY_IMAGES}/{row['filename']}"
            client.fput_object(BUCKET_DATA, key, str(dst), content_type="image/jpeg")
        if i % 50 == 0 or i == len(rows):
            print(f"[upscale] {i}/{len(rows)}")

    out_csv = DATA_DIR / "upscaled.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "source", "room_type", "style", "room_known"]
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[upscale] wrote {len(out_rows)} images -> {UPSCALED_DIR}")
    if args.upload:
        print(f"[upscale] uploaded to {BUCKET_DATA}/{KEY_IMAGES}/")
    return 0 if out_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
