"""caption-service — BLIP image captioning.

Used by the workshop pipeline to auto-caption the interior-design dataset before
LoRA training. Accepts an image three ways:

  * multipart upload            POST /v1/caption          (file=@room.jpg)
  * an object already in MinIO  POST /v1/caption/object   {"bucket":..,"key":..}
  * a batch of MinIO objects    POST /v1/caption/batch    {"keys":[...]}

The model is loaded once at startup and stays resident in GPU memory.
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, Field
from transformers import BlipForConditionalGeneration, BlipProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("caption-service")

MODEL_ID = os.environ.get("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-large")
DEVICE = os.environ.get("BLIP_DEVICE", "cuda")
DEFAULT_BUCKET = os.environ.get("MINIO_BUCKET_DATA", "interior-design-data")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Conditional-captioning seed keeps BLIP on-topic for interiors.
DEFAULT_CONDITION = "a living room with"


class BlipEngine:
    """Thread-safe BLIP wrapper (one batch at a time on the GPU)."""

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.processor: BlipProcessor | None = None
        self.model: BlipForConditionalGeneration | None = None
        self.device = "cpu"
        self.dtype = torch.float32
        self._lock = threading.Lock()
        self._ready = False
        self._load_error: str | None = None

    def load(self) -> None:
        try:
            use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            self.dtype = torch.float16 if use_cuda else torch.float32
            logger.info("Loading BLIP %s on %s", self.model_id, self.device)
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("BLIP ready on %s", self.device)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.exception("Failed to load BLIP")

    def is_ready(self) -> bool:
        return self._ready and self.model is not None

    def caption_batch(
        self,
        images: list[Image.Image],
        *,
        condition: str | None,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[str]:
        if not self.is_ready():
            raise RuntimeError("model not loaded")
        assert self.processor is not None and self.model is not None

        text = [condition] * len(images) if condition else None
        with self._lock:
            inputs = self.processor(images=images, text=text, return_tensors="pt")
            inputs = {
                k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
                )
        assert self.processor is not None
        return [
            self.processor.decode(o, skip_special_tokens=True).strip() for o in out
        ]


engine = BlipEngine()
_minio_client = None


def get_minio():
    """Lazily build a MinIO client (only needed for the object endpoints)."""
    global _minio_client
    if _minio_client is None:
        from minio import Minio

        endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
        _minio_client = Minio(
            endpoint,
            access_key=os.environ.get("MINIO_ROOT_USER", "admin"),
            secret_key=os.environ.get("MINIO_ROOT_PASSWORD", ""),
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        )
    return _minio_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(engine.load)
    yield


app = FastAPI(
    title="BLIP Caption Service",
    version="1.0.0",
    description="Automatic image captioning for the LoRA training pipeline.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- schemas ---


class ObjectCaptionRequest(BaseModel):
    key: str = Field(..., min_length=1)
    bucket: str = Field(default=DEFAULT_BUCKET)
    condition: str | None = Field(default=DEFAULT_CONDITION)
    max_new_tokens: int = Field(default=40, ge=5, le=120)
    num_beams: int = Field(default=3, ge=1, le=8)


class BatchCaptionRequest(BaseModel):
    keys: list[str] = Field(..., min_length=1, max_length=64)
    bucket: str = Field(default=DEFAULT_BUCKET)
    condition: str | None = Field(default=DEFAULT_CONDITION)
    max_new_tokens: int = Field(default=40, ge=5, le=120)
    num_beams: int = Field(default=3, ge=1, le=8)


class CaptionResponse(BaseModel):
    caption: str
    model: str
    duration_ms: int


class BatchItem(BaseModel):
    key: str
    caption: str | None = None
    error: str | None = None


class BatchCaptionResponse(BaseModel):
    items: list[BatchItem]
    model: str
    duration_ms: int


def _require_ready() -> None:
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="BLIP still loading")


def _normalise(caption: str) -> str:
    """Tidy whitespace/punctuation only.

    We deliberately do NOT inject a room type here: callers (e.g. the workshop's
    room-type filter) rely on the caption being an honest description of what BLIP
    actually saw. Room/style anchoring happens later in the enrichment step.
    """
    c = " ".join((caption or "").split()).strip(" .")
    return c or "an interior room"


def _open_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Not a readable image: {exc}") from exc


# -------------------------------------------------------------- endpoints ---


@app.get("/health/live", tags=["health"])
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def ready() -> dict:
    if not engine.is_ready():
        raise HTTPException(
            status_code=503, detail={"status": "loading", "error": engine._load_error}
        )
    return {"status": "ready", "model": engine.model_id, "device": engine.device}


@app.get("/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok" if engine.is_ready() else "loading",
        "ready": engine.is_ready(),
        "model": engine.model_id,
        "device": engine.device,
        "error": engine._load_error,
    }


@app.post("/v1/caption", response_model=CaptionResponse, tags=["caption"])
async def caption_upload(
    file: UploadFile = File(...),
    condition: str | None = DEFAULT_CONDITION,
    max_new_tokens: int = 40,
    num_beams: int = 3,
) -> CaptionResponse:
    _require_ready()
    started = time.perf_counter()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    image = _open_image(raw)
    captions = await run_in_threadpool(
        engine.caption_batch,
        [image],
        condition=condition,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return CaptionResponse(
        caption=_normalise(captions[0]),
        model=engine.model_id,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/v1/caption/object", response_model=CaptionResponse, tags=["caption"])
async def caption_object(req: ObjectCaptionRequest) -> CaptionResponse:
    _require_ready()
    started = time.perf_counter()
    try:
        resp = get_minio().get_object(req.bucket, req.key)
        try:
            raw = resp.read()
        finally:
            resp.close()
            resp.release_conn()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=404, detail=f"Cannot read {req.bucket}/{req.key}: {exc}"
        ) from exc

    image = _open_image(raw)
    captions = await run_in_threadpool(
        engine.caption_batch,
        [image],
        condition=req.condition,
        max_new_tokens=req.max_new_tokens,
        num_beams=req.num_beams,
    )
    return CaptionResponse(
        caption=_normalise(captions[0]),
        model=engine.model_id,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/v1/caption/batch", response_model=BatchCaptionResponse, tags=["caption"])
async def caption_batch(req: BatchCaptionRequest) -> BatchCaptionResponse:
    """Caption many MinIO objects in one GPU batch (much faster than one-by-one)."""
    _require_ready()
    started = time.perf_counter()
    client = get_minio()

    images: list[Image.Image] = []
    ok_keys: list[str] = []
    errors: dict[str, str] = {}

    for key in req.keys:
        try:
            resp = client.get_object(req.bucket, key)
            try:
                raw = resp.read()
            finally:
                resp.close()
                resp.release_conn()
            img = Image.open(io.BytesIO(raw))
            img.load()
            images.append(img.convert("RGB"))
            ok_keys.append(key)
        except Exception as exc:  # noqa: BLE001
            errors[key] = str(exc)[:200]

    captions: list[str] = []
    if images:
        try:
            captions = await run_in_threadpool(
                engine.caption_batch,
                images,
                condition=req.condition,
                max_new_tokens=req.max_new_tokens,
                num_beams=req.num_beams,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("batch caption failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    by_key = {k: _normalise(c) for k, c in zip(ok_keys, captions)}
    items = [
        BatchItem(key=k, caption=by_key.get(k), error=errors.get(k)) for k in req.keys
    ]
    return BatchCaptionResponse(
        items=items,
        model=engine.model_id,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": "caption-service",
        "model": engine.model_id,
        "endpoints": ["/v1/caption", "/v1/caption/object", "/v1/caption/batch", "/health"],
    }
