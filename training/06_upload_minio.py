#!/usr/bin/env python
"""Step 6 — Upload the prepared dataset to MinIO (interior-design-data bucket).

Layout created in the bucket:
  interior-design-data/
    ├── images/00001.jpg ...
    ├── captions/00001.txt ...
    └── metadata/metadata.jsonl

Requires MinIO to be running (docker compose up -d minio) and reachable at
localhost:9000 from the host.

Usage:
  python 06_upload_minio.py
"""
from __future__ import annotations

import os

from common import CAPTIONS_DIR, LORA_DIR, PREPARED_DIR, load_dotenv
from minio_utils import ensure_bucket, get_client, upload_dir, upload_file


def main() -> int:
    load_dotenv()
    bucket = os.environ.get("MINIO_BUCKET_DATASET", "interior-design-data")

    if not any(PREPARED_DIR.glob("*.jpg")):
        print(f"ERROR: no prepared images in {PREPARED_DIR}. Run steps 2–3 first.")
        return 1

    client = get_client()
    ensure_bucket(client, bucket)

    n_img = upload_dir(client, bucket, "images", PREPARED_DIR, content_type="image/jpeg")
    print(f"[upload] {n_img} images -> {bucket}/images/")

    if CAPTIONS_DIR.exists():
        n_cap = upload_dir(client, bucket, "captions", CAPTIONS_DIR, content_type="text/plain")
        print(f"[upload] {n_cap} captions -> {bucket}/captions/")

    meta = LORA_DIR / "metadata.jsonl"
    if meta.exists():
        upload_file(client, bucket, "metadata/metadata.jsonl", meta, content_type="application/json")
        print(f"[upload] metadata.jsonl -> {bucket}/metadata/")

    print("[upload] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
