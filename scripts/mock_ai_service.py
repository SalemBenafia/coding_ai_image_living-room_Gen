#!/usr/bin/env python
"""A mock of the SDXL ai-service for local/CI testing WITHOUT a GPU.

Implements the same contract as ai-service/app.py:
    GET  /health   -> {status, ready, model, device, lora_loaded}
    POST /generate -> image/png bytes

It renders a deterministic placeholder image (seeded gradient + prompt text) so the
backend ↔ MinIO ↔ history flow can be exercised end-to-end without loading SDXL.

Run:  uvicorn scripts.mock_ai_service:app --port 8100
"""
from __future__ import annotations

import io
import random
import textwrap

from fastapi import FastAPI, Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

app = FastAPI(title="Mock AI Service")


class GenerateReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 0


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": True, "model": "mock", "device": "cpu", "lora_loaded": False}


@app.post("/generate")
def generate(req: GenerateReq) -> Response:
    rng = random.Random(req.seed)  # deterministic per seed
    base = tuple(rng.randint(40, 210) for _ in range(3))

    img = Image.new("RGB", (req.width, req.height), base)
    draw = ImageDraw.Draw(img)
    # simple diagonal shading so different seeds look different
    for y in range(0, req.height, 4):
        shade = int(30 * (y / req.height))
        draw.line([(0, y), (req.width, y)], fill=tuple(min(255, c + shade) for c in base))

    caption = textwrap.fill(f"[MOCK] {req.prompt}", width=40)[:400]
    draw.multiline_text((24, 24), caption, fill=(15, 15, 15))
    draw.text((24, req.height - 40), f"seed={req.seed}  {req.width}x{req.height}", fill=(15, 15, 15))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/")
def root() -> dict:
    return {"service": "mock-ai-service"}
