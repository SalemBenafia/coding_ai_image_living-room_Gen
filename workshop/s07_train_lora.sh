#!/usr/bin/env bash
# =============================================================================
# Stage 07 — Fine-tune an SDXL LoRA on the prepared living-room dataset.
#
# Uses the official diffusers example train_text_to_image_lora_sdxl.py against
# data/lora_dataset/ (imagefolder + metadata.jsonl, caption column "text").
#
# Requirements: a CUDA GPU with >= ~16 GB VRAM (24 GB comfortable at 1024).
#
#   bash s07_train_lora.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# ---- Config (override via env) ----
MODEL="${SDXL_MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
VAE="${SDXL_VAE_ID:-madebyollin/sdxl-vae-fp16-fix}"
DATA_DIR="${WORKSHOP_DATA:-$REPO_ROOT/data}/lora_dataset"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/output/${LORA_NAME:-living-room-style-v1}}"
RESOLUTION="${TRAIN_RESOLUTION:-1024}"
RANK="${LORA_RANK:-16}"
MAX_STEPS="${MAX_TRAIN_STEPS:-1000}"
LR="${LEARNING_RATE:-1e-4}"
BATCH="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEED="${TRAIN_SEED:-42}"
CKPT_STEPS="${CHECKPOINTING_STEPS:-500}"

PYBIN="${PYBIN:-python}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-$REPO_ROOT/.runtime/diffusers}"
TRAIN_SCRIPT="$DIFFUSERS_DIR/examples/text_to_image/train_text_to_image_lora_sdxl.py"

# ---- Fetch the training script (pinned) ----
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[train] cloning diffusers into $DIFFUSERS_DIR"
  git clone --depth 1 --branch v0.36.0 https://github.com/huggingface/diffusers "$DIFFUSERS_DIR" \
    || git clone --depth 1 https://github.com/huggingface/diffusers "$DIFFUSERS_DIR"
fi

if [ ! -f "$DATA_DIR/metadata.jsonl" ]; then
  echo "[train] ERROR: $DATA_DIR/metadata.jsonl not found. Run s06_build_dataset.py first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
N=$(wc -l < "$DATA_DIR/metadata.jsonl")
echo "[train] model=$MODEL  images=$N  res=$RESOLUTION  steps=$MAX_STEPS  rank=$RANK  out=$OUTPUT_DIR"

# ---- single-GPU fp16 launch. SDXL uses PyTorch SDPA attention by default, so no
#      xformers is required. SNR gamma weighting speeds up convergence. ----
export ACCELERATE_MIXED_PRECISION="fp16"

# The diffusers SDXL trainer applies --variant to BOTH the base model and any
# explicit VAE. The fp16-fix VAE (madebyollin) ships no ".fp16" variant, so when
# training with --variant=fp16 we omit the explicit VAE and let the trainer use
# the base model's VAE subfolder (it keeps the VAE in fp32 internally for stable
# latents). Set SDXL_VAE_ID explicitly only if that repo has an fp16 variant.
VAE_ARG=()
if [ -n "${SDXL_VAE_ID:-}" ] && [ "${SDXL_USE_EXPLICIT_VAE:-0}" = "1" ]; then
  VAE_ARG=(--pretrained_vae_model_name_or_path="$VAE")
fi

set +e
accelerate launch --num_processes=1 --mixed_precision=fp16 "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="$MODEL" \
  "${VAE_ARG[@]}" \
  --variant="${SDXL_VARIANT:-fp16}" \
  --train_data_dir="$DATA_DIR" \
  --caption_column="text" \
  --resolution="$RESOLUTION" --center_crop --random_flip \
  --train_batch_size="$BATCH" \
  --gradient_accumulation_steps="$GRAD_ACCUM" \
  --gradient_checkpointing \
  --max_train_steps="$MAX_STEPS" \
  --learning_rate="$LR" \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --rank="$RANK" \
  --seed="$SEED" \
  --checkpointing_steps="$CKPT_STEPS" \
  --dataloader_num_workers=4 \
  --output_dir="$OUTPUT_DIR"
rc=$?
set -e

WEIGHTS="$OUTPUT_DIR/pytorch_lora_weights.safetensors"
# The trainer runs a final validation-inference pass AFTER saving the weights;
# in offline mode that pass can raise (it tries to auto-guess the weight name).
# Treat the run as successful as long as the weights file was actually written.
if [ -f "$WEIGHTS" ]; then
  echo "[train] done. LoRA weights: $WEIGHTS"
  echo "[train] next: python s08_publish_lora.py"
  exit 0
fi
echo "[train] ERROR: training exited $rc and no weights were saved." >&2
exit "${rc:-1}"
