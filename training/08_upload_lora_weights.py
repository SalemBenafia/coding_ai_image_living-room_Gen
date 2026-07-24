#!/usr/bin/env python
"""Step 8 — Upload the trained LoRA weights to MinIO (models bucket).

The ai-service downloads this object at startup (LORA_OBJECT env var).
Default object key matches the .env.example default:
    models/living-room-style-v1/pytorch_lora_weights.safetensors

Usage:
  python 08_upload_lora_weights.py [--weights training/output/living-room-lora/pytorch_lora_weights.safetensors]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import OUTPUT_DIR, load_dotenv
from minio_utils import ensure_bucket, get_client, upload_file


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=str(OUTPUT_DIR / "living-room-lora" / "pytorch_lora_weights.safetensors"),
    )
    ap.add_argument(
        "--object",
        default=os.environ.get("LORA_OBJECT", "living-room-style-v1/pytorch_lora_weights.safetensors"),
    )
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found at {weights}. Run 07_train_lora.sh first.")
        return 1

    bucket = os.environ.get("MINIO_BUCKET_MODELS", "models")
    client = get_client()
    ensure_bucket(client, bucket)
    upload_file(client, bucket, args.object, weights, content_type="application/octet-stream")

    print(f"[upload] {weights} -> {bucket}/{args.object}")
    print("[upload] set LORA_OBJECT in .env and restart ai-service to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
