"""Central configuration, loaded from environment variables (.env)."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    database_url: str = "sqlite:////data/app.db"

    # --- MinIO ---
    # Credentials must be provided via environment (.env / docker-compose).
    # No working password is shipped as a source-level default.
    minio_endpoint: str = "minio:9000"
    minio_root_user: str = "admin"
    minio_root_password: str = ""
    minio_secure: bool = False
    minio_bucket_generated: str = "generated-images"

    # --- AI service ---
    ai_service_url: str = "http://ai-service:8100"
    ai_request_timeout: float = 300.0  # SDXL generation can be slow

    # --- Validation ---
    max_prompt_length: int = 1000
    min_prompt_length: int = 3

    # --- Prompt enhancement ---
    prompt_enhancer: str = "local"  # "groq" | "local"
    groq_api_key: str = ""
    groq_model: str = "qwen-2.5-32b"

    # --- CORS ---
    backend_cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
