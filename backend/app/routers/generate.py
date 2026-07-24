"""Generation endpoints: text -> enhanced prompt -> SDXL -> MinIO -> DB."""
from __future__ import annotations

import logging
import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import ai_client
from ..database import get_db
from ..models import Generation
from ..negative_prompt import build_negative_prompt
from ..prompt_enhancer import build_positive_prompt
from ..schemas import GenerateRequest, GenerationOut, StyleOut
from ..storage import get_storage
from ..styles import STYLE_PRESETS

logger = logging.getLogger("generate")
router = APIRouter(tags=["generate"])

_MAX_SEED = 2_147_483_647  # fits SQLite INTEGER; wide enough for reproducibility


@router.get("/styles", response_model=list[StyleOut])
def list_styles() -> list[StyleOut]:
    return [
        StyleOut(key=k, label=v["label"], description=v["description"])
        for k, v in STYLE_PRESETS.items()
    ]


@router.post("/generate", response_model=GenerationOut)
async def generate(req: GenerateRequest, db: Session = Depends(get_db)) -> GenerationOut:
    gen_id = str(uuid.uuid4())

    # 1) Resolve seed (None or -1 -> random, but recorded for reproducibility).
    seed = req.seed if (req.seed is not None and req.seed >= 0) else secrets.randbelow(_MAX_SEED)

    # 2) Build final prompts.
    positive = build_positive_prompt(req.prompt, req.style, req.enhance_prompt)
    negative = build_negative_prompt(req.negative_prompt)

    # 3) Generate the image via the SDXL service.
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

    # 4) Persist the image to object storage.
    object_name = f"generated/{gen_id}.png"
    try:
        storage = get_storage()
        storage.put_image(object_name, png, content_type="image/png")
    except Exception as exc:  # noqa: BLE001
        logger.error("Storage failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store image") from exc

    # 5) Record metadata.
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
        object_name=object_name,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return GenerationOut.model_validate(row.to_dict())
