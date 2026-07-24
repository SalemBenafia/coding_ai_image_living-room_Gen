"""Workshop configuration: paths, dataset registry, MinIO client, service URLs.

The workshop is the offline training pipeline (the "Workshop / Training Pipelines"
box in the architecture). It is MinIO-centric: prepared images, captions and the
final LoRA all live in object storage, exactly as the design specifies.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- Local staging layout (scratch; the source of truth is MinIO) -----------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("WORKSHOP_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"                 # galinakg (Pinterest) download
RAW_HIRES_DIR = DATA_DIR / "raw_hires"     # stepanyarullin download
PREPARED_DIR = DATA_DIR / "prepared"       # cleaned (deduped/deblurred), pre-upscale
UPSCALED_DIR = DATA_DIR / "upscaled"       # Real-ESRGAN'd + resized to TARGET_SIZE
LORA_DATASET_DIR = DATA_DIR / "lora_dataset"   # imagefolder + metadata.jsonl
OUTPUT_DIR = REPO_ROOT / "workshop" / "output"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_SIZE = int(os.environ.get("TRAIN_RESOLUTION", "1024"))

# ---- MinIO layout -----------------------------------------------------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_USER = os.environ.get("MINIO_ROOT_USER", "admin")
MINIO_PASSWORD = os.environ.get("MINIO_ROOT_PASSWORD", "password123")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
BUCKET_DATA = os.environ.get("MINIO_BUCKET_DATA", "interior-design-data")
BUCKET_MODELS = os.environ.get("MINIO_BUCKET_MODELS", "models")

# Object-key prefixes inside BUCKET_DATA
KEY_IMAGES = "images"              # upscaled 1024 training images
KEY_CAPTIONS_BLIP = "captions"     # raw BLIP captions
KEY_CAPTIONS_FINAL = "captions_enhanced"  # Qwen-enriched captions
KEY_METADATA = "metadata/metadata.csv"

# ---- Downstream services (used by the caption/enhance stages) ---------------
CAPTION_SERVICE_URL = os.environ.get("CAPTION_SERVICE_URL", "http://localhost:8300")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:8200")

# ---- LoRA identity ----------------------------------------------------------
LORA_NAME = os.environ.get("LORA_NAME", "living-room-style-v1")
LORA_OBJECT_KEY = os.environ.get(
    "LORA_OBJECT_KEY", f"models/{LORA_NAME}/pytorch_lora_weights.safetensors"
)
AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:8100")

# A consistent trigger phrase the LoRA learns to associate with the style.
TRIGGER = os.environ.get("LORA_TRIGGER", "lvngrm living room")

# ---- Dataset registry -------------------------------------------------------
# Each dataset declares how to discover images and derive (room_type, style).
#   galinakg   : dir layout data/raw/Pinterest.../<room_type>/<style>/*.jpg (labeled)
#   stepanyarullin : data/raw_hires/dataset_train/dataset_train/<style>/*.jpg
#                    (style only; room type must be inferred via BLIP filtering)


def galinakg_root() -> Path | None:
    if not RAW_DIR.exists():
        return None
    for child in RAW_DIR.iterdir():
        if child.is_dir() and (child / "metadata.csv").exists():
            return child
    # fall back to RAW_DIR itself if it already holds room_type folders
    if (RAW_DIR / "living_room").exists():
        return RAW_DIR
    return None


def stepanyarullin_root() -> Path | None:
    cand = RAW_HIRES_DIR / "dataset_train" / "dataset_train"
    return cand if cand.exists() else None


def minio_client():
    from minio import Minio

    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASSWORD,
        secure=MINIO_SECURE,
    )


def ensure_buckets() -> None:
    client = minio_client()
    for bucket in (BUCKET_DATA, BUCKET_MODELS):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)


def find_images(root: Path) -> list[Path]:
    if not root or not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
