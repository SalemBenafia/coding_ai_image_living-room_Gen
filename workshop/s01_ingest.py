#!/usr/bin/env python
"""Stage 01 — Data ingestion.

Discovers living-room images from the downloaded Kaggle datasets, derives their
(room_type, style) labels, and writes a single manifest CSV. Optionally uploads
the original images + manifest to MinIO (the "Data Ingestion -> MinIO" arrow in
the architecture).

Two sources are supported (see config.py):
  * galinakg      — labeled by folder: <room_type>/<style>/*.jpg  (living rooms known)
  * stepanyarullin — labeled by style only; room type is inferred later via BLIP

For a focused living-room style LoRA we take galinakg's living_room split as the
guaranteed corpus. stepanyarullin can be folded in with --include-hires (its
images are room-type-filtered downstream by the caption stage).

Usage:
  python s01_ingest.py [--include-hires] [--upload] [--limit N]
"""
from __future__ import annotations

import argparse
import csv

from config import (
    KEY_METADATA,
    BUCKET_DATA,
    RAW_DIR,
    ensure_buckets,
    ensure_dirs,
    galinakg_root,
    minio_client,
    stepanyarullin_root,
    DATA_DIR,
)

# stepanyarullin styles that overlap our living-room presets (used when --include-hires).
HIRES_STYLES = {
    "modern", "scandinavian", "contemporary", "industrial",
    "mid-century-modern", "rustic", "coastal", "traditional",
}


def discover_galinakg() -> list[dict]:
    root = galinakg_root()
    if not root:
        print("[ingest] galinakg dataset not found under", RAW_DIR)
        return []
    lr = root / "living_room"
    if not lr.exists():
        print("[ingest] no living_room folder in", root)
        return []
    records: list[dict] = []
    for style_dir in sorted(p for p in lr.iterdir() if p.is_dir()):
        for img in sorted(style_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                records.append(
                    {
                        "source": "galinakg",
                        "path": str(img),
                        "room_type": "living_room",
                        "style": style_dir.name,
                        "room_known": "1",
                    }
                )
    print(f"[ingest] galinakg living_room: {len(records)} images")
    return records


def discover_hires(per_style: int = 120) -> list[dict]:
    root = stepanyarullin_root()
    if not root:
        print("[ingest] stepanyarullin dataset not found; skipping")
        return []
    records: list[dict] = []
    for style_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if style_dir.name not in HIRES_STYLES:
            continue
        imgs = sorted(
            p for p in style_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        # Cap per style: these are room-type-unlabeled, so BLIP filters them to
        # living rooms downstream; a bounded sample keeps upscale/caption time sane.
        if per_style > 0:
            imgs = imgs[:per_style]
        for img in imgs:
            records.append(
                {
                    "source": "stepanyarullin",
                    "path": str(img),
                    "room_type": "unknown",   # inferred by BLIP room filter later
                    "style": style_dir.name,
                    "room_known": "0",
                }
            )
    print(f"[ingest] stepanyarullin (overlapping styles, <= {per_style}/style): {len(records)} images")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-hires", action="store_true",
                    help="also ingest stepanyarullin (room-filtered downstream)")
    ap.add_argument("--hires-per-style", type=int, default=120,
                    help="cap stepanyarullin images per style (0 = all)")
    ap.add_argument("--upload", action="store_true", help="upload manifest to MinIO")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    ensure_dirs(DATA_DIR)
    records = discover_galinakg()
    if args.include_hires:
        records += discover_hires(args.hires_per_style)
    if not records:
        print("[ingest] ERROR: no images discovered. Did the datasets download?")
        return 1
    if args.limit:
        records = records[: args.limit]

    manifest = DATA_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["source", "path", "room_type", "style", "room_known"]
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"[ingest] wrote manifest {manifest} ({len(records)} rows)")

    # style distribution
    from collections import Counter
    dist = Counter(r["style"] for r in records)
    print("[ingest] style distribution:", dict(dist))

    if args.upload:
        ensure_buckets()
        client = minio_client()
        client.fput_object(BUCKET_DATA, KEY_METADATA, str(manifest))
        print(f"[ingest] uploaded manifest -> {BUCKET_DATA}/{KEY_METADATA}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
