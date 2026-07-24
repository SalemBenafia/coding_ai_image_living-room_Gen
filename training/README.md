# Training pipeline — SDXL LoRA for living-room interiors

This is the **workshop** side of the project. It turns the Kaggle interior-design
dataset into a fine-tuned LoRA adapter that the `ai-service` loads from MinIO.

```
Kaggle ──▶ download ──▶ clean/prepare ──▶ BLIP captions ──▶ (LLM enrich)
       ──▶ build LoRA dataset ──▶ upload to MinIO ──▶ train LoRA ──▶ upload weights
```

## 0. Prerequisites

- A CUDA GPU with ≥ ~16 GB VRAM (24 GB comfortable at 1024²).
- A Python env with **CUDA-enabled torch** already installed (e.g. the Vast
  PyTorch venv: `source /venv/main/bin/activate`).
- MinIO running for the upload steps: `docker compose up -d minio`.

Install the extra deps:

```bash
pip install -r requirements.txt
```

## 1. Kaggle credentials

Any one of these works:

```bash
# (a) classic kaggle.json
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# (b) username + key
export KAGGLE_USERNAME=your_user
export KAGGLE_KEY=xxxxxxxxxxxxxxxx

# (c) newer API token
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxx
```

> ⚠️ **Never commit credentials.** `kaggle.json`, `access_token`, and `.env` are
> already in `.gitignore`.

## 2. Run it

End-to-end (long — includes training):

```bash
bash run_all.sh
# or data-prep only:
SKIP_TRAIN=1 bash run_all.sh
```

Or step by step:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_download_dataset.py` | Download + unzip the Kaggle dataset to `data/raw/` |
| 2 | `02_prepare_dataset.py`  | Filter living rooms, drop dups/blurry, resize to 1024² → `data/prepared/` |
| 3 | `03_generate_captions.py`| BLIP captions → `data/captions/*.txt` |
| 4 | `04_enhance_captions.py` | (optional) LLM-enrich captions (needs `GROQ_API_KEY`) |
| 5 | `05_build_lora_dataset.py`| Assemble `data/lora_dataset/` + `metadata.jsonl` |
| 6 | `06_upload_minio.py`     | Upload images/captions/metadata to `interior-design-data` bucket |
| 7 | `07_train_lora.sh`       | Fine-tune the SDXL LoRA (diffusers) → `output/living-room-lora/` |
| 8 | `08_upload_lora_weights.py`| Upload `pytorch_lora_weights.safetensors` to `models` bucket |

## 3. Use the trained LoRA

Set in the repo-root `.env`:

```ini
LORA_OBJECT=living-room-style-v1/pytorch_lora_weights.safetensors
LORA_SCALE=0.8
```

Then restart the AI service so it downloads and applies the adapter:

```bash
docker compose restart ai-service
```

## Tuning notes

- `MAX_STEPS` (default 1500) — more steps = stronger style, risk of overfitting.
- `RANK` (default 16) — LoRA capacity; 8–32 typical.
- `LR` (default 1e-4) — lower if the style drifts too hard.
- Override any of these via env vars, e.g. `MAX_STEPS=2500 RANK=32 bash 07_train_lora.sh`.
