# 🛋️ AI Living Room Generator

Generate realistic, high-quality **living-room interior designs** from natural-language
prompts using **Stable Diffusion XL (SDXL)** enhanced with **LoRA fine-tuning**.

Type a description → the backend validates it, an LLM enriches it, a negative prompt
is built, **SDXL + your LoRA** renders the image, it's stored in **MinIO**, and the
web UI shows it with full history and downloads.

```
User ▶ Frontend ▶ FastAPI ▶ Validate ▶ LLM enhance ▶ Negative prompt
     ▶ SDXL + LoRA ▶ MinIO ▶ back to the user (+ history)
```

---

## Architecture

```
                         ┌────────────────────────── docker-compose ──────────────────────────┐
   Browser  ── 8080 ──▶  │  frontend (nginx)                                                   │
                         │      │  proxies /api, /health                                       │
                         │      ▼                                                              │
                         │  backend (FastAPI, :8000) ──┬── validate (Pydantic)                │
                         │      │                       ├── enhance prompt (Groq/local)        │
                         │      │                       ├── build negative prompt              │
                         │      │                       ├── SQLite (history, SQLAlchemy)       │
                         │      │                       └── MinIO (store/serve PNG)            │
                         │      ▼                                                              │
                         │  ai-service (FastAPI, :8100)  ── SDXL + LoRA (Diffusers, GPU)       │
                         │      │                                                              │
                         │      └── loads LoRA from ──▶ minio (:9000 / console :9001)          │
                         └─────────────────────────────────────────────────────────────────────┘
```

**Two AI stages, cleanly separated:** the *backend* owns business logic
(validation, prompt engineering, storage, history); the *ai-service* owns only
model loading + generation, so it can be restarted or moved to a bigger GPU
independently.

---

## Quickstart (Docker Compose)

Requires a Linux host with an **NVIDIA GPU**, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/SalemBenafia/coding_ai_image_living-room_Gen.git
cd coding_ai_image_living-room_Gen

cp .env.example .env        # edit secrets (MinIO password, optional GROQ_API_KEY)
docker compose up -d --build
```

| Service | URL | Notes |
|---------|-----|-------|
| **Web UI** | http://localhost:8080 | the app |
| Backend API docs | http://localhost:8000/docs | OpenAPI / Swagger |
| MinIO console | http://localhost:9001 | login = `.env` MinIO creds |

> ⏳ On first boot the `ai-service` downloads SDXL (~7 GB) — watch progress with
> `docker compose logs -f ai-service`. It reports healthy once the model is loaded.

Prompt enhancement works **out of the box** with a local rule-based enhancer.
To use a hosted LLM (Groq/Qwen), set `PROMPT_ENHANCER=groq` and `GROQ_API_KEY` in `.env`.

---

## Using it

Enter a prompt such as:

> *A modern Scandinavian living room with a large beige sofa, wooden flooring,
> indoor plants, floor-to-ceiling windows, natural sunlight, minimalist decoration,
> and warm ambient lighting.*

Pick a **style preset**, tweak **aspect ratio / steps / guidance / seed** under
*Advanced*, and click **Generate**. Images are saved automatically and appear in the
**History** gallery (click to reopen, ✕ to delete).

### API (backend)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/generate` | generate an image (see schema below) |
| `GET`  | `/api/styles` | list style presets |
| `GET`  | `/api/history?limit=&offset=` | paginated history |
| `GET`  | `/api/history/{id}` | one generation's metadata |
| `DELETE` | `/api/history/{id}` | delete image + record |
| `GET`  | `/api/images/{id}` | stream the PNG |
| `GET`  | `/health` | MinIO + ai-service status |

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

## Training the LoRA

The `training/` folder is a standalone pipeline: Kaggle → clean → BLIP captions →
(LLM enrich) → build dataset → **train SDXL LoRA** → upload weights to MinIO. The
`ai-service` picks them up via `LORA_OBJECT` in `.env`.

See **[training/README.md](training/README.md)** for the full guide.

---

## Repository layout

```
living-room-ai/
├── docker-compose.yml         # one command starts the whole stack
├── .env.example               # copy to .env
├── frontend/                  # HTML/CSS/JS + nginx (reverse-proxies /api)
├── backend/                   # FastAPI: validation, prompt eng., storage, history
│   └── app/
│       ├── main.py  config.py  database.py  models.py  schemas.py
│       ├── storage.py  prompt_enhancer.py  negative_prompt.py  ai_client.py  styles.py
│       └── routers/  generate.py  history.py
├── ai-service/                # FastAPI: SDXL + LoRA (Diffusers, GPU)
│   ├── app.py  pipeline.py
├── training/                  # LoRA fine-tuning workshop (steps 01–08)
└── scripts/                   # local test helpers (mock ai-service, smoke test)
```

## Tech stack

Frontend (HTML/CSS/JS, nginx) · FastAPI · Pydantic · SQLAlchemy · MinIO ·
Stable Diffusion XL · Diffusers · LoRA · PyTorch/CUDA · BLIP · Groq/Qwen 2.5 ·
OpenCV · Pillow · Docker Compose.

## Local testing without a GPU

You can exercise the full backend ↔ MinIO ↔ history flow using a **mock** AI service
(returns a generated placeholder PNG, no SDXL/GPU needed):

```bash
pip install -r backend/requirements.txt minio
bash scripts/run_local_smoke.sh     # starts MinIO(if available)+mock+backend, runs smoke_test.py
```

## Persistence & deployment note

On ephemeral GPU hosts (e.g. Vast.ai) the container filesystem is wiped on
recycle/destroy. **Code lives in Git; models + images live in MinIO** (back MinIO
with a persistent volume/host mount in production). See `docker-compose.yml` volumes.

## License

MIT — see [LICENSE](LICENSE).
