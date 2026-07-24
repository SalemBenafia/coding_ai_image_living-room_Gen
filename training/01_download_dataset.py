#!/usr/bin/env python
"""Step 1 — Download the Kaggle interior-design dataset into data/raw/.

Auth (any one of these works with the Kaggle client):
  * export KAGGLE_USERNAME=... KAGGLE_KEY=...
  * ~/.kaggle/kaggle.json                (chmod 600)
  * export KAGGLE_API_TOKEN=KGAT_...      (newer token) — see training/README.md

Usage:
  python 01_download_dataset.py [--dataset galinakg/interior-design-images-and-metadata]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from common import RAW_DIR, ensure_dirs, find_images, load_dotenv

DEFAULT_DATASET = "galinakg/interior-design-images-and-metadata"


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    args = ap.parse_args()

    if shutil.which("kaggle") is None:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle", file=sys.stderr)
        print("Then configure credentials (see this script's docstring).", file=sys.stderr)
        return 1

    ensure_dirs(RAW_DIR)
    print(f"[download] {args.dataset} -> {RAW_DIR}")
    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(RAW_DIR), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: kaggle download failed ({exc}).", file=sys.stderr)
        print("Check your Kaggle credentials and dataset slug.", file=sys.stderr)
        return exc.returncode or 1

    imgs = find_images(RAW_DIR)
    print(f"[download] done — {len(imgs)} image files found under {RAW_DIR}")
    if not imgs:
        print("WARNING: no images found. Inspect the extracted contents manually.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
