"""FastAPI application entrypoint for the AI Living Room Generator backend."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, ai_client, llm_client
from .config import get_settings
from .database import init_db
from .prompt_enhancer import active_backend
from .routers import generate, history
from .schemas import HealthOut
from .storage import get_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables and make sure the images bucket exists before serving traffic.
    init_db()
    try:
        get_storage().ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure MinIO bucket at startup: %s", exc)
    logger.info("Backend %s ready (enhancer=%s).", __version__, active_backend())
    yield


app = FastAPI(
    title="AI Living Room Generator API",
    version=__version__,
    description="Generate realistic living-room interior designs with SDXL + LoRA.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(history.router, prefix="/api")


# ---- Health -----------------------------------------------------------------
# Liveness/readiness are namespaced under /api so the nginx frontend proxies
# them like every other API call (and so orchestrators can probe them).


@app.get("/api/health/live", tags=["health"])
def health_live() -> dict:
    """Process is up (does not check dependencies)."""
    return {"status": "alive"}


async def _full_health() -> HealthOut:
    minio_ok = get_storage().ping()
    ai = await ai_client.status()
    ai_ok = bool(ai) and ai.get("ready", False)
    llm_ok = await llm_client.ping()
    status = "ok" if (minio_ok and ai_ok) else "degraded"
    return HealthOut(
        status=status,
        minio=minio_ok,
        ai_service=ai_ok,
        llm_service=llm_ok,
        enhancer=active_backend(),
        lora_loaded=ai.get("lora_loaded"),
        lora_scale=ai.get("lora_scale"),
    )


@app.get("/api/health", response_model=HealthOut, tags=["health"])
async def health() -> HealthOut:
    """Full dependency health (MinIO, ai-service, llm-service)."""
    return await _full_health()


# Back-compat: /health (unprefixed) mirrors /api/health.
@app.get("/health", response_model=HealthOut, tags=["health"])
async def health_root() -> HealthOut:
    return await _full_health()


@app.get("/", tags=["health"])
def root() -> dict:
    return {"name": "AI Living Room Generator API", "version": __version__, "docs": "/docs"}
