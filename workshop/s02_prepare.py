#!/usr/bin/env python
"""Stage 02 — Clean & prepare the dataset.

  * remove exact / near-duplicate images (perceptual average-hash);
  * remove blurry / low-quality images (variance of the Laplacian);
  * remove images that are too small even to upscale meaningfully;
  * copy the survivors to data/prepared/ with deterministic names and a
    prepared.csv carrying (filename, source, room_type, style).

Upscaling and resizing happen in stage 03, so this stage keeps native pixels.

Usage:
  python s02_prepare.py [--blur-threshold 40] [--hash-distance 4]
                        [--min-short-side 200] [--limit 0]
"""
from __future__ import annotations

import argparse
import csv
import shutil

import cv2
import imagehash
from PIL import Image, ImageOps

from config import DATA_DIR, PREPARED_DIR, ensure_dirs


def laplacian_sharpness(path: str) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur-threshold", type=float, default=40.0,
                    help="min Laplacian variance to keep (lower = allow softer)")
    ap.add_argument("--hash-distance", type=int, default=4,
                    help="max Hamming distance treated as a duplicate")
    ap.add_argument("--min-short-side", type=int, default=200,
                    help="drop images whose short side is below this")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        print(f"[prepare] ERROR: {manifest} missing. Run s01_ingest.py first.")
        return 1

    with manifest.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[prepare] {len(rows)} candidate images")

    ensure_dirs(PREPARED_DIR)
    for old in PREPARED_DIR.glob("*"):
        if old.is_file():
            old.unlink()

    seen_hashes: list[imagehash.ImageHash] = []
    kept: list[dict] = []
    dropped = {"unreadable": 0, "small": 0, "blurry": 0, "duplicate": 0}

    for row in rows:
        path = row["path"]
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                phash = imagehash.average_hash(im)
        except Exception:  # noqa: BLE001
            dropped["unreadable"] += 1
            continue

        if min(w, h) < args.min_short_side:
            dropped["small"] += 1
            continue
        if laplacian_sharpness(path) < args.blur_threshold:
            dropped["blurry"] += 1
            continue
        if any(abs(phash - h2) <= args.hash_distance for h2 in seen_hashes):
            dropped["duplicate"] += 1
            continue

        seen_hashes.append(phash)
        idx = len(kept) + 1
        stem = f"{idx:05d}_{row['source']}_{row['style']}"
        dst = PREPARED_DIR / f"{stem}.jpg"
        # Re-encode to a clean JPEG (strips odd modes / broken EXIF).
        with Image.open(path) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(dst, quality=95)
        kept.append(
            {
                "filename": dst.name,
                "source": row["source"],
                "room_type": row["room_type"],
                "style": row["style"],
                "room_known": row.get("room_known", "1"),
            }
        )
        if args.limit and len(kept) >= args.limit:
            break

    out_csv = DATA_DIR / "prepared.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "source", "room_type", "style", "room_known"]
        )
        writer.writeheader()
        writer.writerows(kept)

    print(f"[prepare] kept {len(kept)} -> {PREPARED_DIR}")
    print(f"[prepare] dropped: {dropped}")
    from collections import Counter
    print("[prepare] style distribution:", dict(Counter(k["style"] for k in kept)))
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
