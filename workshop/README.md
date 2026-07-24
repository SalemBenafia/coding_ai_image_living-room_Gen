# Workshop — LoRA training pipeline

The offline pipeline that turns Kaggle interior-design photos into a custom
**SDXL LoRA** for living rooms. It is MinIO-centric: prepared images, captions,
and the final adapter all live in object storage.

```
Kaggle datasets
   │  s01_ingest        discover living-room images, derive (room_type, style)
   ▼
Clean / prepare        s02_prepare   dedupe (avg-hash) · deblur (Laplacian) · drop tiny
   ▼
Real-ESRGAN upscale    s03_upscale   x4 super-resolution → 1024px  → MinIO images/
   ▼
BLIP captions          s04_caption   caption-service (reads MinIO) + room-type filter
   ▼
Qwen enrichment        s05_enhance   llm-service expands captions, adds style + trigger
   ▼
Build LoRA dataset     s06_build     imagefolder + metadata.jsonl
   ▼
Train SDXL LoRA        s07_train     diffusers train_text_to_image_lora_sdxl
   ▼
Publish                s08_publish   upload to MinIO models/ + hot-reload ai-service
```

## Run it

With the stack running (the caption + llm services must be up):

```bash
# everything
docker compose --profile workshop run --rm workshop make all

# or stage groups
docker compose --profile workshop run --rm workshop make data      # ingest+prepare+upscale
docker compose --profile workshop run --rm workshop make captions  # BLIP + Qwen
docker compose --profile workshop run --rm workshop make dataset
docker compose --profile workshop run --rm workshop make train MAX_TRAIN_STEPS=1500 LORA_RANK=32
docker compose --profile workshop run --rm workshop make publish
```

Natively (no Docker), from `workshop/` with the venv active and the stack up:

```bash
export WORKSHOP_DATA=../data   # wherever the datasets were downloaded
python s01_ingest.py --include-hires --upload
python s02_prepare.py
python s03_upscale.py --upload
python s04_caption.py
python s05_enhance.py
python s06_build_dataset.py --upload
bash   s07_train_lora.sh
python s08_publish_lora.py
```

## The datasets (and why we upscale)

Two Kaggle datasets are combined:

| Dataset | Content | Native resolution | Room-type labels |
|---------|---------|-------------------|------------------|
| `galinakg/interior-design-images-and-metadata` | 1056 living-room images across 5 styles | ~236px (Pinterest thumbnails) | ✅ labeled |
| `stepanyarullin/interior-design-styles` | 14.8k images across 20 styles | ~360px | ❌ style only |

**Both are web thumbnails**, not native 1024px. Training SDXL at 1024 directly on
them would teach the VAE to reproduce upscaling artifacts. So `s03_upscale` runs
**Real-ESRGAN x4** (via `spandrel`, no `basicsr` dependency) to recover clean
edges, then resizes to 1024.

`galinakg` provides the guaranteed living-room corpus (folder-labeled). From
`stepanyarullin` we take a bounded sample of the overlapping styles and let **BLIP
double as a room-type filter** (`s04_caption`): images whose caption doesn't look
like a living room are dropped, so bedrooms/kitchens never leak into the LoRA.

## Get the Kaggle data

```bash
pip install kaggle
export KAGGLE_API_TOKEN=...            # or ~/.kaggle/kaggle.json
kaggle datasets download -d galinakg/interior-design-images-and-metadata --unzip -p data/raw
kaggle datasets download -d stepanyarullin/interior-design-styles --unzip -p data/raw_hires
```

## Tunable knobs (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `TRAIN_RESOLUTION` | 1024 | training / upscale target size |
| `MAX_TRAIN_STEPS` | 1000 | optimizer steps |
| `LORA_RANK` | 16 | LoRA rank (higher = more capacity, larger file) |
| `LEARNING_RATE` | 1e-4 | AdamW LR |
| `HIRES_PER_STYLE` | 120 | cap on stepanyarullin images per style |
| `LORA_TRIGGER` | `lvngrm living room` | activation phrase folded into captions |

## Output

`s07` writes `workshop/output/<LORA_NAME>/pytorch_lora_weights.safetensors`.
`s08` uploads it to `models/<LORA_NAME>/pytorch_lora_weights.safetensors` in MinIO
and calls the ai-service `/v1/reload-lora` endpoint so the running SDXL service
hot-swaps the new adapter — no restart needed.
