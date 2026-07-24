"""FastAPI wrapper around the SDXL + LoRA pipeline.

Contract with the backend:
    POST /generate   JSON GenerateReq  ->  image/png bytes
    GET  /health     -> {status, model, device, lora_loaded, ready}
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from pipeline import SDXLPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai-service")

sdxl = SDXLPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the (heavy) model at startup so requests don't pay the cost.
    await run_in_threadpool(sdxl.load)
    yield


app = FastAPI(title="SDXL Living Room AI Service", version="1.0.0", lifespan=lifespan)


class GenerateReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int = Field(default=0, ge=0)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if sdxl.is_ready() else "loading",
        "ready": sdxl.is_ready(),
        "model": sdxl.model_id,
        "device": "cuda" if (sdxl.device == "cuda" and torch.cuda.is_available()) else "cpu",
        "lora_loaded": sdxl.lora_loaded,
    }


@app.post("/generate")
async def generate(req: GenerateReq) -> Response:
    if not sdxl.is_ready():
        raise HTTPException(status_code=503, detail="Model still loading")
    try:
        png = await run_in_threadpool(
            sdxl.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        torch.cuda.empty_cache()
        logger.error("CUDA OOM: %s", exc)
        raise HTTPException(status_code=507, detail="GPU out of memory; reduce size/steps") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return Response(content=png, media_type="image/png")


@app.get("/")
def root() -> dict:
    return {"service": "ai-service", "endpoints": ["/generate", "/health"]}
