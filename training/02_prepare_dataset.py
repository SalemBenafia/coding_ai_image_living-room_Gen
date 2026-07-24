#!/usr/bin/env python
"""Step 2 — Clean & prepare images.

  * (optional) keep only living-room images if a metadata CSV with a room-type
    column is found in data/raw/;
  * drop exact/near duplicates (average hash);
  * drop blurry / low-quality images (Laplacian variance);
  * center-crop to square and resize to TARGET_SIZE (default 1024);
  * write cleaned JPEGs to data/prepared/ as 00001.jpg, 00002.jpg, ...

Usage:
  python 02_prepare_dataset.py [--room-type "living"] [--blur-threshold 60] [--limit 0]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

from common import (
    PREPARED_DIR,
    RAW_DIR,
    TARGET_SIZE,
    ensure_dirs,
    find_images,
    load_dotenv,
)


def find_metadata_csv(root: Path) -> Path | None:
    csvs = list(root.rglob("*.csv"))
    return csvs[0] if csvs else None


def living_room_filenames(csv_path: Path, room_type: str) -> set[str] | None:
    """Return a set of image filenames whose room-type column matches `room_type`.

    Returns None if the CSV has no recognizable room-type/filename columns, meaning
    'do not filter'.
    """
    try:
        with csv_path.open(newline="", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            headers = [h.lower() for h in (reader.fieldnames or [])]
            room_col = next((h for h in headers if "room" in h or "type" in h or "category" in h), None)
            file_col = next((h for h in headers if "image" in h or "file" in h or "name" in h or "path" in h), None)
            if not room_col or not file_col:
                return None
            wanted: set[str] = set()
            fh.seek(0)
            reader = csv.DictReader(fh)
            for row in reader:
                lower = {k.lower(): (v or "") for k, v in row.items()}
                if room_type.lower() in lower[room_col].lower():
                    wanted.add(Path(lower[file_col]).name)
            return wanted or None
    except Exception as exc:  # noqa: BLE001
        print(f"[prepare] could not parse metadata ({exc}); skipping room filter.")
        return None


def is_sharp(path: Path, threshold: float) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    return cv2.Laplacian(img, cv2.CV_64F).var() >= threshold


def square_resize(path: Path, size: int) -> Image.Image | None:
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    return ImageOps.fit(im, (size, size), Image.LANCZOS)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-type", default="living", help="substring to keep (empty = keep all)")
    ap.add_argument("--blur-threshold", type=float, default=60.0)
    ap.add_argument("--hash-distance", type=int, default=5, help="max hamming distance = duplicate")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    images = find_images(RAW_DIR)
    if not images:
        print(f"ERROR: no images under {RAW_DIR}. Run 01_download_dataset.py first.")
        return 1
    print(f"[prepare] {len(images)} raw images found.")

    # Optional living-room filter via metadata CSV.
    keep_names: set[str] | None = None
    if args.room_type:
        meta = find_metadata_csv(RAW_DIR)
        if meta:
            keep_names = living_room_filenames(meta, args.room_type)
            if keep_names:
                print(f"[prepare] metadata filter: {len(keep_names)} '{args.room_type}' entries.")
            else:
                print("[prepare] no usable room-type filter found; keeping all images.")

    ensure_dirs(PREPARED_DIR)
    for old in PREPARED_DIR.glob("*.jpg"):
        old.unlink()

    seen_hashes: list[imagehash.ImageHash] = []
    kept = 0
    dropped = {"room": 0, "blurry": 0, "duplicate": 0, "unreadable": 0}

    for path in images:
        if keep_names is not None and path.name not in keep_names:
            dropped["room"] += 1
            continue
        if not is_sharp(path, args.blur_threshold):
            dropped["blurry"] += 1
            continue
        try:
            ph = imagehash.average_hash(Image.open(path).convert("RGB"))
        except Exception:  # noqa: BLE001
            dropped["unreadable"] += 1
            continue
        if any(abs(ph - h) <= args.hash_distance for h in seen_hashes):
            dropped["duplicate"] += 1
            continue
        img = square_resize(path, TARGET_SIZE)
        if img is None:
            dropped["unreadable"] += 1
            continue

        seen_hashes.append(ph)
        kept += 1
        img.save(PREPARED_DIR / f"{kept:05d}.jpg", quality=95)
        if args.limit and kept >= args.limit:
            break

    print(f"[prepare] kept {kept} images -> {PREPARED_DIR}")
    print(f"[prepare] dropped: {dropped}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
