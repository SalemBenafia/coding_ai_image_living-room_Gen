"""History + image-serving endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Generation
from ..schemas import GenerationOut, HistoryPage
from ..storage import get_storage

logger = logging.getLogger("history")
router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryPage)
def list_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> HistoryPage:
    total = db.scalar(select(func.count()).select_from(Generation)) or 0
    rows = db.scalars(
        select(Generation).order_by(Generation.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return HistoryPage(
        total=total,
        items=[GenerationOut.model_validate(r.to_dict()) for r in rows],
    )


@router.get("/history/{gen_id}", response_model=GenerationOut)
def get_one(gen_id: str, db: Session = Depends(get_db)) -> GenerationOut:
    row = db.get(Generation, gen_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return GenerationOut.model_validate(row.to_dict())


@router.delete("/history/{gen_id}", status_code=204)
def delete_one(gen_id: str, db: Session = Depends(get_db)) -> Response:
    row = db.get(Generation, gen_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    # Best-effort object removal, then DB row.
    get_storage().remove_image(row.object_name)
    db.delete(row)
    db.commit()
    return Response(status_code=204)


@router.get("/images/{gen_id}")
def get_image(gen_id: str, db: Session = Depends(get_db)) -> Response:
    """Stream the PNG from MinIO through the backend (no direct MinIO exposure)."""
    row = db.get(Generation, gen_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        data = get_storage().get_image(row.object_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fetch image failed for %s: %s", gen_id, exc)
        raise HTTPException(status_code=404, detail="Image object missing") from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Disposition": f'inline; filename="living_room_{gen_id[:8]}.png"',
        },
    )
