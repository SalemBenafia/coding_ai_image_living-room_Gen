#!/usr/bin/env bash
# Convenience: run the whole data pipeline end-to-end (steps 1–6), then train (7)
# and upload the LoRA (8). Each step is idempotent and can be run individually.
#
#   bash run_all.sh            # full pipeline including training (long!)
#   SKIP_TRAIN=1 bash run_all.sh   # data prep only (skip 7 & 8)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

echo "== 1/8 download ==";        $PY 01_download_dataset.py
echo "== 2/8 prepare ==";         $PY 02_prepare_dataset.py
echo "== 3/8 caption (BLIP) ==";  $PY 03_generate_captions.py
echo "== 4/8 enhance (LLM) ==";   $PY 04_enhance_captions.py
echo "== 5/8 build dataset ==";   $PY 05_build_lora_dataset.py
echo "== 6/8 upload to MinIO =="; $PY 06_upload_minio.py || echo "(skipped: MinIO not reachable)"

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "SKIP_TRAIN=1 — stopping before LoRA training."
  exit 0
fi

echo "== 7/8 train LoRA ==";      bash 07_train_lora.sh
echo "== 8/8 upload LoRA ==";     $PY 08_upload_lora_weights.py
echo "All done."
