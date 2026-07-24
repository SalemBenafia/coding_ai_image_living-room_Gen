#!/usr/bin/env bash
# Step 7 — Fine-tune an SDXL LoRA on the prepared living-room dataset.
#
# Uses the official diffusers example `train_text_to_image_lora_sdxl.py`, which
# reads data/lora_dataset/ (imagefolder + metadata.jsonl, caption column "text").
#
# Requirements: a CUDA GPU with >= ~16 GB VRAM (24 GB comfortable at 1024).
# Run from the training/ directory:  bash 07_train_lora.sh
set -euo pipefail

# ---- Config (override via env) ----
MODEL="${SDXL_MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
VAE="${SDXL_VAE:-madebyollin/sdxl-vae-fp16-fix}"   # fp16-safe VAE
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/lora_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "$0")" && pwd)/output/living-room-lora}"
RESOLUTION="${RESOLUTION:-1024}"
RANK="${RANK:-16}"
MAX_STEPS="${MAX_STEPS:-1500}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEED="${SEED:-42}"

DIFFUSERS_DIR="${DIFFUSERS_DIR:-$HOME/diffusers}"
TRAIN_SCRIPT="$DIFFUSERS_DIR/examples/text_to_image/train_text_to_image_lora_sdxl.py"

# ---- Get the training script ----
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[train] cloning diffusers into $DIFFUSERS_DIR"
  git clone --depth 1 https://github.com/huggingface/diffusers "$DIFFUSERS_DIR"
  pip install -e "$DIFFUSERS_DIR"
  pip install -r "$DIFFUSERS_DIR/examples/text_to_image/requirements_sdxl.txt"
fi

if [ ! -f "$DATA_DIR/metadata.jsonl" ]; then
  echo "ERROR: $DATA_DIR/metadata.jsonl not found. Run 05_build_lora_dataset.py first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "[train] model=$MODEL  data=$DATA_DIR  out=$OUTPUT_DIR  steps=$MAX_STEPS  rank=$RANK"

accelerate launch "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="$MODEL" \
  --pretrained_vae_model_name_or_path="$VAE" \
  --train_data_dir="$DATA_DIR" \
  --caption_column="text" \
  --resolution="$RESOLUTION" --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps="$GRAD_ACCUM" \
  --gradient_checkpointing \
  --max_train_steps="$MAX_STEPS" \
  --learning_rate="$LR" \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --rank="$RANK" \
  --seed="$SEED" \
  --checkpointing_steps=500 \
  --output_dir="$OUTPUT_DIR"

echo "[train] done. LoRA weights: $OUTPUT_DIR/pytorch_lora_weights.safetensors"
echo "[train] next: python 08_upload_lora_weights.py"
