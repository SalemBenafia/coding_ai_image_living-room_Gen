"""MinIO helpers for the training pipeline (dataset + model artifacts)."""
from __future__ import annotations

import os
from pathlib import Path

from minio import Minio


def get_client() -> Minio:
    # Training runs on the host, so default to localhost:9000 (mapped port),
    # not the in-compose hostname 'minio'.
    endpoint = os.environ.get("MINIO_ENDPOINT_HOST", os.environ.get("MINIO_ENDPOINT", "localhost:9000"))
    if endpoint == "minio:9000":  # in-compose name is not reachable from the host
        endpoint = "localhost:9000"
    return Minio(
        endpoint,
        access_key=os.environ.get("MINIO_ROOT_USER", "admin"),
        secret_key=os.environ.get("MINIO_ROOT_PASSWORD", "change-me-please-123"),
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"[minio] created bucket {bucket}")


def upload_file(client: Minio, bucket: str, object_name: str, path: Path,
                content_type: str = "application/octet-stream") -> None:
    client.fput_object(bucket, object_name, str(path), content_type=content_type)


def upload_dir(client: Minio, bucket: str, prefix: str, local_dir: Path,
               content_type: str = "application/octet-stream") -> int:
    count = 0
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            client.fput_object(bucket, f"{prefix}/{rel}", str(p), content_type=content_type)
            count += 1
    return count
