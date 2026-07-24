#!/usr/bin/env bash
# =============================================================================
# End-to-end workshop run (native or in-container). Thin wrapper over the stages
# so the whole "Kaggle -> MinIO -> BLIP -> Qwen -> LoRA -> SDXL" flow is one call.
#
#   bash run_all.sh
#
# Honours the same env knobs as the Makefile (MAX_TRAIN_STEPS, LORA_RANK, ...).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYBIN:-python}"

INCLUDE_HIRES="${INCLUDE_HIRES:-1}"
HIRES_FLAG=""
[ "$INCLUDE_HIRES" = "1" ] && HIRES_FLAG="--include-hires --hires-per-style ${HIRES_PER_STYLE:-120}"

echo "==================== 01 ingest ===================="
$PY s01_ingest.py $HIRES_FLAG --upload
echo "==================== 02 prepare ==================="
$PY s02_prepare.py --blur-threshold "${BLUR_THRESHOLD:-30}"
echo "==================== 03 upscale ==================="
$PY s03_upscale.py --size "${TRAIN_RESOLUTION:-1024}" --upload
echo "==================== 04 caption ==================="
$PY s04_caption.py
echo "==================== 05 enhance ==================="
$PY s05_enhance.py
echo "==================== 06 build ====================="
$PY s06_build_dataset.py --upload
echo "==================== 07 train ====================="
bash s07_train_lora.sh
echo "==================== 08 publish ==================="
$PY s08_publish_lora.py
echo "==================== DONE ========================="
