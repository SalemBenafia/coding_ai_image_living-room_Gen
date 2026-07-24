#!/usr/bin/env python
"""Stage 08 — Publish the trained LoRA and hot-swap it into the ai-service.

  1. upload workshop/output/<name>/pytorch_lora_weights.safetensors to MinIO
     under models/<name>/pytorch_lora_weights.safetensors;
  2. POST the ai-service /v1/reload-lora so the running SDXL service picks up the
     new adapter without a restart (the "Use LoRA with SDXL" step).

Usage:
  python s08_publish_lora.py [--no-reload]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from config import (
    AI_SERVICE_URL,
    BUCKET_MODELS,
    LORA_NAME,
    LORA_OBJECT_KEY,
    OUTPUT_DIR,
    ensure_buckets,
    minio_client,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reload", action="store_true")
    ap.add_argument("--weights", default=None, help="override path to the safetensors")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else (
        OUTPUT_DIR / LORA_NAME / "pytorch_lora_weights.safetensors"
    )
    if not weights.exists():
        print(f"[publish] ERROR: {weights} not found. Run s07_train_lora.sh first.")
        return 1

    ensure_buckets()
    client = minio_client()
    # LORA_OBJECT_KEY already includes the models/ prefix; strip it for the key.
    key = LORA_OBJECT_KEY
    if key.startswith("models/"):
        key = key[len("models/"):]
    client.fput_object(BUCKET_MODELS, key, str(weights),
                       content_type="application/octet-stream")
    size_mb = weights.stat().st_size / 1e6
    print(f"[publish] uploaded {size_mb:.1f} MB -> {BUCKET_MODELS}/{key}")

    if not args.no_reload:
        try:
            resp = httpx.post(f"{AI_SERVICE_URL.rstrip('/')}/v1/reload-lora", timeout=60.0)
            resp.raise_for_status()
            print(f"[publish] ai-service reloaded LoRA: {resp.json()}")
        except Exception as exc:  # noqa: BLE001
            print(f"[publish] WARNING: could not trigger ai-service reload ({exc}). "
                  f"Restart the ai-service to pick up the new LoRA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
