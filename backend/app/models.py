"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Generation(Base):
    """One text-to-image generation request and its result."""

    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # Prompts
    prompt: Mapped[str] = mapped_column(Text)  # original user prompt
    enhanced_prompt: Mapped[str] = mapped_column(Text)  # final positive prompt to SDXL
    negative_prompt: Mapped[str] = mapped_column(Text)
    style: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Generation parameters (what was actually used)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    steps: Mapped[int] = mapped_column(Integer)
    guidance_scale: Mapped[float] = mapped_column(Float)
    seed: Mapped[int] = mapped_column(Integer)  # resolved seed actually used

    # Storage
    object_name: Mapped[str] = mapped_column(String(255))  # key in MinIO bucket

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "negative_prompt": self.negative_prompt,
            "style": self.style,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "object_name": self.object_name,
            "image_url": f"/api/images/{self.id}",
            "created_at": self.created_at,
        }
