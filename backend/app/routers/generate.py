"""Generation endpoints — the core interaction flow.

  User prompt
    -> Pydantic validation (schemas.GenerateRequest)
    -> LLM prompt enhancement (Qwen llm-service, rule-based fallback)
    -> Negative prompt builder
    -> Stable Diffusion XL (+ LoRA) via ai-service
    -> Save PNG to MinIO
    -> Persist metadata (SQLAlchemy) and return it to the frontend
"""
from __future__ import annotations

import logging
import secrets
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import ai_client
from ..config import get_settings
from ..database import get_db
from ..models import Generation
from ..negative_prompt import build_negative_prompt
from ..prompt_enhancer import build_positive_prompt
from ..schemas import GenerateRequest, GenerationOut, StyleOut
from ..storage import get_storage
from ..styles import STYLE_PRESETS

logger = logging.getLogger("generate")
router = APIRouter(tags=["generate"])
settings = get_settings()

_MAX_SEED = 2_147_483_647  # fits a 32-bit signed INTEGER column


@router.get("/styles", response_model=list[StyleOut])
def list_styles() -> list[StyleOut]:
    """Style presets the frontend offers; the backend owns the prompt fragments."""
    return [
        StyleOut(key=k, label=v["label"], description=v["description"])
        for k, v in STYLE_PRESETS.items()
    ]


@router.post("/generate", response_model=GenerationOut)
async def generate(req: GenerateRequest, db: Session = Depends(get_db)) -> GenerationOut:
    gen_id = str(uuid.uuid4())
    started = time.perf_counter()

    # 1) Resolve seed (None / -1 -> random, but recorded for reproducibility).
    seed = req.seed if (req.seed is not None and req.seed >= 0) else secrets.randbelow(_MAX_SEED)

    # 2) Build the final prompts.
    positive, method = await build_positive_prompt(
        req.prompt, req.style, req.enhance_prompt, seed=seed
    )
    negative = build_negative_prompt(req.negative_prompt)

    # 3) Generate the image via the SDXL service (may apply the LoRA).
    try:
        png = await ai_client.generate_image(
            prompt=positive,
            negative_prompt=negative,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=seed,
        )
    except ai_client.AIServiceError as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 4) Persist the image to object storage (MinIO).
    object_name = f"generated/{gen_id}.png"
    try:
        get_storage().put_image(object_name, png, content_type="image/png")
    except Exception as exc:  # noqa: BLE001
        logger.error("Storage failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store image") from exc

    # 5) Record metadata.
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    row = Generation(
        id=gen_id,
        prompt=req.prompt,
        enhanced_prompt=positive,
        negative_prompt=negative,
        style=req.style,
        width=req.width,
        height=req.height,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=seed,
        enhance_method=method,
        lora_scale=settings.lora_scale,
        generation_ms=elapsed_ms,
        object_name=object_name,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    logger.info("generated %s in %dms (enhance=%s)", gen_id, elapsed_ms, method)
    return GenerationOut.model_validate(row.to_dict())
