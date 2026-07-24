# Architecture

This document details the runtime topology, the request/training data flows, and
the contracts between services.

## 1. Service topology

```mermaid
flowchart TB
    subgraph client[Browser]
        UI[Web UI - HTML/CSS/JS]
    end
    subgraph edge[frontend :8080]
        NGINX[nginx - static + /api proxy]
    end
    subgraph app[backend :8000 - no GPU]
        API[FastAPI]
        VAL[Pydantic validation + profanity]
        ENH[prompt enhancer + fallback]
        NEG[negative-prompt builder]
        ORM[SQLAlchemy]
    end
    subgraph gpu[GPU services]
        AI[ai-service :8100 - SDXL + LoRA]
        LLM[llm-service :8200 - Qwen 2.5 1.5B]
        CAP[caption-service :8300 - BLIP]
    end
    subgraph data[Stateful]
        MINIO[(MinIO - images/models/data)]
        PG[(PostgreSQL - history)]
    end

    UI --> NGINX --> API
    API --> VAL --> ENH -->|/v1/enhance| LLM
    API --> NEG
    API -->|/generate| AI
    AI -->|load LoRA| MINIO
    API -->|store/serve PNG| MINIO
    API --> ORM --> PG
    CAP -->|read images| MINIO
```

## 2. Generation request flow

The core interaction flow, exactly as specified:

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant L as llm-service (Qwen)
    participant A as ai-service (SDXL+LoRA)
    participant M as MinIO
    participant D as Postgres

    U->>F: type prompt + style
    F->>B: POST /api/generate
    B->>B: Pydantic validation (empty/length/unsafe/offensive)
    B->>L: POST /v1/enhance (prompt, style)
    L-->>B: rich SDXL prompt  (fallback: rule-based on error)
    B->>B: build negative prompt (defaults + user, de-duped)
    B->>A: POST /generate (prompt, negative, wxh, steps, cfg, seed)
    A->>A: prepend LoRA trigger, run SDXL (+LoRA)
    A-->>B: image/png bytes
    B->>M: put_object generated/{id}.png
    B->>D: INSERT generation (prompts, params, provenance, timing)
    B-->>F: GenerationOut (id, image_url, meta)
    F->>B: GET /api/images/{id}
    B->>M: get_object
    B-->>F: PNG  ->  displayed + added to history
```

## 3. Workshop (LoRA training) DAG

```mermaid
flowchart LR
    K[Kaggle datasets] --> S1[s01 ingest<br/>discover + label]
    S1 --> S2[s02 prepare<br/>dedupe + deblur]
    S2 --> S3[s03 upscale<br/>Real-ESRGAN x4 -> 1024]
    S3 -->|images/| M[(MinIO)]
    S3 --> S4[s04 caption<br/>BLIP + room filter]
    S4 -->|captions/| M
    S4 --> S5[s05 enhance<br/>Qwen enrich + trigger]
    S5 -->|captions_enhanced/| M
    S5 --> S6[s06 build<br/>imagefolder + QC gate]
    S6 --> S7[s07 train<br/>SDXL LoRA]
    S7 --> S8[s08 publish<br/>upload + hot-reload]
    S8 -->|models/| M
    S8 -->|/v1/reload-lora| AI[ai-service]
```

Key data-quality decisions (see [workshop/README.md](workshop/README.md)):

- **Upscaling is mandatory** — both Kaggle datasets are web thumbnails
  (236px / 360px). Real-ESRGAN x4 recovers clean 1024px pixels so the SDXL VAE
  never trains on pixelation.
- **BLIP is a room-type filter** — the second dataset is style-labeled only and
  contains mixed rooms; unconditional BLIP captions are checked for living-room
  vs bedroom/kitchen/bathroom evidence, dropping non-living-rooms (galinakg is
  folder-labeled and kept as-is).
- **A caption QC gate** (s06) rebuilds any caption that describes the wrong room
  or runs away into repetition, guaranteeing clean training text.

## 4. Service contracts

| Service | Endpoint | Contract |
|---------|----------|----------|
| ai-service | `POST /generate` | JSON → `image/png` |
| ai-service | `POST /v1/reload-lora` | re-pull LoRA from MinIO, hot-swap |
| ai-service | `GET /health` | `{ready, lora_loaded, lora_source, ...}` |
| llm-service | `POST /v1/enhance` | short prompt → rich SDXL prompt |
| llm-service | `POST /v1/enhance-caption` | BLIP caption → training caption |
| caption-service | `POST /v1/caption/batch` | MinIO keys → captions |

Every GPU service loads its model once at startup and serializes GPU work behind
a lock, exposing `/health/live` (process up) and `/health/ready` (model loaded)
for orchestration.

## 5. Storage layout (MinIO)

```
interior-design-data/          generated-images/         models/
├── images/        (1024px)    └── generated/            └── living-room-style-v1/
├── captions/      (BLIP)          {uuid}.png                └── pytorch_lora_weights.safetensors
├── captions_enhanced/ (Qwen)
├── metadata/metadata.csv
└── lora_dataset/  (mirror)
```

## 6. Deployment modes

- **Docker Compose** (`docker-compose.yml`) — the intended production deployment;
  9 services incl. bucket init, healthchecks, and a `workshop` profile.
- **Native launcher** (`deploy/native/run_stack.sh`) — runs the *identical*
  service code as plain processes for hosts where Docker cannot run (unprivileged
  containers with no user-namespace / CAP_SYS_ADMIN). Same env, same ports.
