"""Shared helpers for the training pipeline (paths, env, image utilities)."""
from __future__ import annotations

import os
from pathlib import Path

# ---- Directory layout (relative to repo root) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # unpacked Kaggle download
PREPARED_DIR = DATA_DIR / "prepared"       # cleaned + resized images
CAPTIONS_DIR = DATA_DIR / "captions"       # <stem>.txt caption sidecars
LORA_DIR = DATA_DIR / "lora_dataset"       # images + metadata.jsonl for training
OUTPUT_DIR = REPO_ROOT / "training" / "output"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_SIZE = int(os.environ.get("TRAIN_RESOLUTION", "1024"))


def load_dotenv() -> None:
    """Load KEY=VALUE lines from the repo-root .env, if present (no dependency)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def find_images(root: Path) -> list[Path]:
    """Recursively list image files under root."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
