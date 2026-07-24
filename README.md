# 🛋️ AI Living Room Generator

Generate realistic, high-quality **living-room interior designs** from natural-language
prompts using **Stable Diffusion XL (SDXL)** enhanced with a **custom-trained LoRA**.

Type a description → the backend validates it → **Qwen 2.5 1.5B** enriches the prompt →
a negative prompt is built → **SDXL + LoRA** renders the image → it is stored in
**MinIO** → the web UI shows it with full history and downloads.

```
User ▶ Frontend ▶ FastAPI ▶ Validate (Pydantic) ▶ LLM enhance (Qwen 2.5)
     ▶ Negative-prompt builder ▶ SDXL + LoRA ▶ MinIO ▶ back to the user (+ history)
```

Everything runs **locally on your own GPU** — no third-party inference APIs
(no Groq, no OpenAI). The LLM, the captioner, and the image model are all
open-weight models served from containers on the same machine.

---

## Architecture

```
                         ┌───────────────────────── docker compose ─────────────────────────┐
   Browser ─ :8080 ──▶   │  frontend (nginx)  ── proxies /api ──▶ backend                    │
                         │                                                                   │
                         │  backend (FastAPI, :8000) ──┬── validate (Pydantic + profanity)   │
                         │      │                        ├── enhance prompt ─▶ llm-service    │
                         │      │                        ├── build negative prompt            │
                         │      │                        ├── generate ────────▶ ai-service    │
                         │      │                        ├── history (SQLAlchemy ▶ postgres)  │
                         │      │                        └── store/serve PNG ──▶ minio        │
                         │      ▼                                                             │
                         │  ai-service (:8100)   SDXL + LoRA (Diffusers, GPU)                 │
                         │  llm-service (:8200)  Qwen 2.5 1.5B Instruct (GPU)                 │
                         │  caption-service (:8300)  BLIP (GPU, workshop only)               │
                         │                                                                   │
                         │  minio (:9000/:9001)   postgres (:5432)                            │
                         └───────────────────────────────────────────────────────────────────┘

   workshop (profile) :  Kaggle ▶ clean ▶ Real-ESRGAN ▶ BLIP caption ▶ Qwen enrich
                          ▶ build dataset ▶ train SDXL LoRA ▶ publish to MinIO ▶ hot-reload
```

**Clean separation of concerns.** The *backend* owns business logic (validation,
prompt engineering, storage, history). The *ai-service* owns only model loading +
generation, so it can be restarted or moved to a bigger GPU independently. The
*llm-service* and *caption-service* keep their models resident on the GPU and are
reusable by both the app and the training pipeline.

---

## Quickstart (Docker Compose)

Requires a Linux host with an **NVIDIA GPU**, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/SalemBenafia/coding_ai_image_living-room_Gen.git
cd coding_ai_image_living-room_Gen

cp .env.example .env         # defaults work out of the box; change the MinIO password
docker compose up -d --build
```

| Service | URL | Notes |
|---------|-----|-------|
| **Web UI** | http://localhost:8080 | the app |
| Backend API docs | http://localhost:8000/docs | OpenAPI / Swagger |
| MinIO console | http://localhost:9001 | login = `.env` MinIO creds |

> ⏳ On first boot the AI services download their weights (SDXL ~7 GB, Qwen ~3 GB,
> BLIP ~1.8 GB) into a shared `hf_cache` volume — watch with
> `docker compose logs -f ai-service llm-service`. Each reports healthy once loaded.

Prompt enhancement uses the **Qwen llm-service** by default and falls back to a
deterministic rule-based enhancer if the LLM is unavailable, so generation never
breaks.

### No Docker on your box?

Unprivileged containers (e.g. some cloud GPU instances) can't run Docker-in-Docker.
A native launcher runs the **identical service code** under plain processes:

```bash
deploy/native/run_stack.sh up       # start minio, postgres, all services, frontend
deploy/native/run_stack.sh status
deploy/native/run_stack.sh down
```

---

## Using it

Enter a prompt such as:

> *A modern Scandinavian living room with a large beige sofa, wooden flooring,
> indoor plants, floor-to-ceiling windows, natural sunlight.*

Pick a **style preset**, tweak **aspect ratio / steps / guidance / seed** under
*Advanced*, and click **Generate**. Images are saved automatically and appear in the
**History** gallery (click to reopen, ✕ to delete).

### Backend API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/generate` | validate ▶ enhance ▶ generate ▶ store ▶ record |
| `GET`  | `/api/styles` | list style presets |
| `GET`  | `/api/history?limit=&offset=` | paginated history |
| `GET`  | `/api/history/{id}` | one generation's metadata |
| `DELETE` | `/api/history/{id}` | delete image + record |
| `GET`  | `/api/images/{id}` | stream the PNG from MinIO |
| `GET`  | `/api/health` | MinIO + ai-service + llm-service status, LoRA state |
| `GET`  | `/api/health/live` | liveness probe |

```jsonc
// POST /api/generate
{
  "prompt": "a cozy scandinavian living room, big windows",
  "negative_prompt": "clutter, dark",     // optional (merged with defaults)
  "style": "scandinavian",                // optional preset key
  "width": 1024, "height": 1024,
  "steps": 30, "guidance_scale": 7.5,
  "seed": null,                           // null / -1 = random (recorded for reuse)
  "enhance_prompt": true
}
```

---

## Training the LoRA (workshop)

The `workshop/` folder is the offline pipeline — the "Workshop / Training Pipelines"
box in the design. It is MinIO-centric: prepared images, captions, and the final
LoRA all live in object storage.

```
Kaggle dataset ▶ clean (dedupe, deblur) ▶ Real-ESRGAN upscale ▶ 1024px
   ▶ BLIP captions ▶ Qwen caption enrichment ▶ build LoRA dataset
   ▶ train SDXL LoRA ▶ publish to MinIO ▶ hot-reload ai-service
```

```bash
# with the stack running (needs caption-service + llm-service up)
docker compose --profile workshop run --rm workshop make all
# or stage by stage, e.g.  make data  /  make train MAX_TRAIN_STEPS=1500 LORA_RANK=32
```

See **[workshop/README.md](workshop/README.md)** for the full guide, including how
both Kaggle interior datasets are combined and why upscaling is necessary (the
source images are web thumbnails).

---

## Repository layout

```
living-room-ai/
├── docker-compose.yml         # one command starts the whole stack
├── .env.example               # copy to .env
├── frontend/                  # HTML/CSS/JS + nginx (reverse-proxies /api)
├── backend/                   # FastAPI: validation, prompt eng., storage, history
│   └── app/  main.py config.py database.py models.py schemas.py
│             storage.py prompt_enhancer.py llm_client.py negative_prompt.py
│             ai_client.py styles.py  routers/{generate,history}.py
├── ai-service/                # FastAPI: SDXL + LoRA (Diffusers, GPU)
├── llm-service/               # FastAPI: Qwen 2.5 1.5B (prompt + caption enhancement)
├── caption-service/           # FastAPI: BLIP captioning (workshop)
├── workshop/                  # LoRA training pipeline (stages s01–s08)
├── deploy/native/             # run the same services without Docker
└── scripts/                   # smoke test + local helpers
```

## Tech stack

Frontend (HTML/CSS/JS, nginx) · FastAPI · Pydantic · SQLAlchemy · PostgreSQL ·
MinIO · Stable Diffusion XL · Diffusers · LoRA · PyTorch/CUDA · BLIP ·
Qwen 2.5 1.5B · Real-ESRGAN · OpenCV · Pillow · Docker Compose.

## End-to-end smoke test

With the stack up (Docker or native), exercise the full flow — validation,
Qwen enhancement, SDXL generation, MinIO storage, history:

```bash
python scripts/smoke_test.py            # BACKEND=http://localhost:8000
```

## License

MIT — see [LICENSE](LICENSE).
