"""Central configuration, loaded from environment variables (.env / compose).

Pydantic-Settings is the single source of truth for backend configuration.
Everything here has a safe default so the app boots in development, but real
credentials (MinIO, Postgres) come from the environment in production.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- Database (SQLAlchemy) ---
    # Postgres in the full stack; falls back to a local SQLite file if unset.
    database_url: str = "postgresql+psycopg2://livingroom:livingroom_pw@localhost:5432/livingroom"

    # --- MinIO object storage ---
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "admin"
    minio_root_password: str = "password123"
    minio_secure: bool = False
    minio_bucket_images: str = "generated-images"
    minio_bucket_models: str = "models"
    minio_bucket_data: str = "interior-design-data"

    # --- Downstream AI services ---
    ai_service_url: str = "http://localhost:8100"       # SDXL + LoRA
    llm_service_url: str = "http://localhost:8200"       # Qwen 2.5 1.5B
    caption_service_url: str = "http://localhost:8300"   # BLIP
    ai_request_timeout: float = 300.0                    # SDXL generation is slow
    llm_request_timeout: float = 30.0

    # --- Prompt validation ---
    max_prompt_length: int = 1000
    min_prompt_length: int = 3

    # --- Prompt enhancement ---
    # "llm"   -> call the Qwen llm-service, with a rule-based fallback on failure
    # "local" -> rule-based enhancement only (deterministic, no network)
    prompt_enhancer: str = "llm"

    # --- Generation defaults (used when the client omits a field) ---
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    lora_scale: float = 0.8

    # --- CORS ---
    backend_cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        raw = self.backend_cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
