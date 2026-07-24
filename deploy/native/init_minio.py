#!/usr/bin/env python
"""Create the three MinIO buckets the stack uses (idempotent).

  interior-design-data   raw images, metadata, captions (workshop)
  generated-images        user-generated PNGs (backend)
  models                  LoRA adapters (ai-service loads from here)

Reads MINIO_* from the environment (same vars as docker-compose).
"""
from __future__ import annotations

import os
import sys

from minio import Minio


def main() -> int:
    client = Minio(
        os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.environ.get("MINIO_ROOT_USER", "admin"),
        secret_key=os.environ.get("MINIO_ROOT_PASSWORD", "password123"),
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )
    buckets = [
        os.environ.get("MINIO_BUCKET_DATA", "interior-design-data"),
        os.environ.get("MINIO_BUCKET_IMAGES", "generated-images"),
        os.environ.get("MINIO_BUCKET_MODELS", "models"),
    ]
    for bucket in buckets:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"[minio] created bucket {bucket}")
        else:
            print(f"[minio] bucket {bucket} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
