"""MinIO object-storage wrapper for generated images."""
from __future__ import annotations

import io
import logging

from minio import Minio
from minio.error import S3Error

from .config import get_settings

logger = logging.getLogger("storage")
settings = get_settings()


class Storage:
    def __init__(self) -> None:
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket_generated

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info("Created bucket %s", self.bucket)

    def ping(self) -> bool:
        try:
            self.client.bucket_exists(self.bucket)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MinIO ping failed: %s", exc)
            return False

    def put_image(self, object_name: str, data: bytes, content_type: str = "image/png") -> None:
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def get_image(self, object_name: str) -> bytes:
        resp = None
        try:
            resp = self.client.get_object(self.bucket, object_name)
            return resp.read()
        finally:
            if resp is not None:
                resp.close()
                resp.release_conn()

    def remove_image(self, object_name: str) -> None:
        try:
            self.client.remove_object(self.bucket, object_name)
        except S3Error as exc:
            logger.warning("Could not remove %s: %s", object_name, exc)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
