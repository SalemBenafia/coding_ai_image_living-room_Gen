"""Pydantic request/response schemas + input validation."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import get_settings
from .styles import STYLE_PRESETS

settings = get_settings()

# Minimal safety filter. Interior-design app: reject obviously unsafe/offensive
# requests. This is a lightweight guard, not a full moderation system.
_BLOCKED_TERMS = {
    "nude", "naked", "nsfw", "porn", "sex", "gore", "blood", "corpse",
    "weapon", "gun", "knife attack", "child", "kill", "suicide",
}


def find_unsafe_terms(text: str) -> list[str]:
    low = text.lower()
    return sorted({t for t in _BLOCKED_TERMS if t in low})


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Natural-language description of the room.")
    negative_prompt: str | None = Field(
        default=None, description="Extra things to avoid (merged with defaults)."
    )
    style: str | None = Field(default=None, description="Style preset key, e.g. 'modern'.")
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    steps: int = Field(default=30, ge=10, le=60)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, description="None or -1 → random seed.")
    enhance_prompt: bool = Field(default=True)

    @field_validator("prompt")
    @classmethod
    def _prompt_ok(cls, v: str) -> str:
        v = v.strip()
        if len(v) < settings.min_prompt_length:
            raise ValueError(
                f"Prompt must be at least {settings.min_prompt_length} characters."
            )
        if len(v) > settings.max_prompt_length:
            raise ValueError(
                f"Prompt must be at most {settings.max_prompt_length} characters."
            )
        bad = find_unsafe_terms(v)
        if bad:
            raise ValueError(f"Prompt contains disallowed content: {', '.join(bad)}")
        return v

    @field_validator("style")
    @classmethod
    def _style_ok(cls, v: str | None) -> str | None:
        if v in (None, ""):
            return None
        if v.lower() not in STYLE_PRESETS:
            raise ValueError(
                f"Unknown style '{v}'. Valid: {', '.join(STYLE_PRESETS)}"
            )
        return v.lower()

    @field_validator("negative_prompt")
    @classmethod
    def _neg_ok(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("width/height must be a multiple of 8.")
        return v

    @model_validator(mode="after")
    def _aspect_guard(self) -> "GenerateRequest":
        # SDXL is trained around ~1MP; extreme aspect ratios degrade quality.
        ratio = max(self.width, self.height) / min(self.width, self.height)
        if ratio > 2.5:
            raise ValueError("Aspect ratio too extreme (max 2.5:1).")
        return self


class GenerationOut(BaseModel):
    id: str
    prompt: str
    enhanced_prompt: str
    negative_prompt: str
    style: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    object_name: str
    image_url: str
    created_at: datetime

    model_config = {"from_attributes": True}


class StyleOut(BaseModel):
    key: str
    label: str
    description: str


class HistoryPage(BaseModel):
    total: int
    items: list[GenerationOut]


class HealthOut(BaseModel):
    status: str
    minio: bool
    ai_service: bool
    enhancer: str
