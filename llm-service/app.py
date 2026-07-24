"""llm-service — Qwen 2.5 1.5B Instruct.

Two jobs in this stack:

  1. /v1/enhance          turn a short user prompt ("modern living room") into a
                          rich, comma-separated SDXL prompt.
  2. /v1/enhance-caption   rewrite a terse BLIP caption into a detailed interior
                          design training caption (workshop / LoRA pipeline).

Also exposes an OpenAI-compatible /v1/chat/completions so the model is reusable.

The model is loaded once at startup and stays resident in GPU memory.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("llm-service")

MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = os.environ.get("LLM_DEVICE", "cuda")
MAX_INPUT_CHARS = int(os.environ.get("LLM_MAX_INPUT_CHARS", "2000"))


# ---------------------------------------------------------------- prompts ---

ENHANCE_SYSTEM = (
    "You are a prompt engineer for Stable Diffusion XL, specialising in interior "
    "design photography of LIVING ROOMS.\n"
    "Rewrite the user's short request into ONE single-line, comma-separated image "
    "prompt.\n"
    "Rules:\n"
    "1. Always describe a living room interior.\n"
    "2. Include concrete visual detail: furniture, materials, flooring, wall "
    "treatment, lighting, colour palette, decor, plants, windows.\n"
    "3. End with photographic quality descriptors.\n"
    "4. 40-70 words. No sentences, only comma-separated phrases.\n"
    "5. Output ONLY the prompt. No quotes, no preamble, no explanation, no newlines."
)

# Few-shot examples keep a 1.5B model reliably on-format.
ENHANCE_SHOTS: list[tuple[str, str]] = [
    (
        "Modern living room",
        "a luxurious Scandinavian modern living room, light oak wooden flooring, "
        "white textured walls, minimalist grey fabric sofa, low walnut coffee table, "
        "large indoor monstera plants, floor-to-ceiling windows, warm natural "
        "sunlight, soft linen curtains, brass accent lighting, neutral colour "
        "palette, architectural interior photography, ultra realistic, highly "
        "detailed, sharp focus, 8k, interior design magazine",
    ),
    (
        "cozy industrial loft with a fireplace",
        "a cozy industrial loft living room, exposed red brick wall, polished "
        "concrete floor, black steel window frames, worn leather chesterfield sofa, "
        "reclaimed wood coffee table, cast iron fireplace with warm glowing embers, "
        "edison bulb pendant lights, layered wool throws, moody warm ambient light, "
        "architectural photography, ultra realistic, highly detailed, sharp focus, "
        "8k, interior design magazine",
    ),
]

CAPTION_SYSTEM = (
    "You rewrite short image captions into rich interior-design training captions "
    "for a text-to-image model.\n"
    "Given a basic caption of a living room plus its known room type and style, "
    "expand it into ONE comma-separated description covering furniture, materials, "
    "flooring, wall treatment, lighting, colour palette and decor.\n"
    "Rules: keep the words 'living room'; keep the given style; 25-55 words; "
    "no sentences, only comma-separated phrases; output ONLY the caption, no "
    "quotes, no preamble, no newlines."
)

CAPTION_SHOTS: list[tuple[str, str]] = [
    (
        "caption: a living room with a couch and a table\n"
        "room type: living room\nstyle: scandinavian",
        "a scandinavian living room, pale oak flooring, white plaster walls, light "
        "grey linen sofa, round pine coffee table, woven jute rug, sheer curtains, "
        "large window with soft daylight, potted greenery, minimal decor, calm "
        "neutral palette, interior design photography",
    ),
    (
        "caption: a room with a brick wall and leather chairs\n"
        "room type: living room\nstyle: industrial",
        "an industrial living room, exposed brick wall, dark concrete floor, brown "
        "leather armchairs, black metal shelving, reclaimed wood side table, edison "
        "bulb lighting, tall factory windows, muted grey and rust palette, raw "
        "textured surfaces, interior design photography",
    ),
]


# ------------------------------------------------------------ output clean ---

_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|here(?:'s| is)|okay|ok|of course|prompt|caption|output)"
    r"\b[^,\n]{0,60}?[:\-]\s*",
    re.IGNORECASE,
)
_MD = re.compile(r"[*_`#>\[\]]+")


def clean_single_line(text: str) -> str:
    """Force model output into one clean comma-separated line."""
    text = (text or "").strip()
    # Take the first non-empty line only; small models like to add commentary.
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    text = _MD.sub("", text)
    text = _PREAMBLE.sub("", text).strip()
    text = text.strip(" \"'“”‘’.;")
    # Collapse whitespace and tidy comma spacing.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text).strip(" ,")
    # Drop duplicate comma-separated phrases, preserving order.
    seen: set[str] = set()
    parts: list[str] = []
    for part in text.split(","):
        p = part.strip()
        k = p.lower()
        if p and k not in seen:
            seen.add(k)
            parts.append(p)
    return ", ".join(parts)


# ----------------------------------------------------------------- engine ---


class QwenEngine:
    """Thread-safe wrapper: one generation at a time on the GPU."""

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._lock = threading.Lock()
        self._ready = False
        self._load_error: str | None = None

    def load(self) -> None:
        try:
            use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda else torch.float32
            logger.info("Loading %s on %s (%s)", self.model_id, self.device, dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("Qwen ready on %s", self.device)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.exception("Failed to load Qwen")

    def is_ready(self) -> bool:
        return self._ready and self.model is not None

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> str:
        if not self.is_ready():
            raise RuntimeError("model not loaded")
        assert self.tokenizer is not None and self.model is not None

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                )
        generated = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


engine = QwenEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(engine.load)
    yield


app = FastAPI(
    title="Qwen 2.5 1.5B LLM Service",
    version="1.0.0",
    description="Prompt enhancement and caption enrichment for the Living Room AI stack.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- schemas ---


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    style: str | None = Field(default=None, max_length=64)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=180, ge=16, le=512)
    seed: int | None = None

    @field_validator("prompt")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty")
        return v


class EnhanceResponse(BaseModel):
    enhanced_prompt: str
    raw_output: str
    model: str
    duration_ms: int


class CaptionEnhanceRequest(BaseModel):
    caption: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    room_type: str = Field(default="living room", max_length=64)
    style: str | None = Field(default=None, max_length=64)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=140, ge=16, le=512)
    seed: int | None = None


class CaptionEnhanceResponse(BaseModel):
    caption: str
    model: str
    duration_ms: int


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    seed: int | None = None


# --------------------------------------------------------------- endpoints --


@app.get("/health/live", tags=["health"])
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def ready() -> dict:
    if not engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail={"status": "loading", "error": engine._load_error},
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


def _require_ready() -> None:
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="LLM still loading")


@app.post("/v1/enhance", response_model=EnhanceResponse, tags=["enhance"])
async def enhance(req: EnhanceRequest) -> EnhanceResponse:
    """Short user prompt -> rich SDXL prompt."""
    _require_ready()
    started = time.perf_counter()

    user = req.prompt if not req.style else f"{req.prompt}\nPreferred style: {req.style}"
    messages: list[dict] = [{"role": "system", "content": ENHANCE_SYSTEM}]
    for shot_in, shot_out in ENHANCE_SHOTS:
        messages.append({"role": "user", "content": shot_in})
        messages.append({"role": "assistant", "content": shot_out})
    messages.append({"role": "user", "content": user})

    try:
        raw = await run_in_threadpool(
            engine.chat,
            messages,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            seed=req.seed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("enhance failed")
        raise HTTPException(status_code=500, detail=f"enhance failed: {exc}") from exc

    cleaned = clean_single_line(raw)
    if len(cleaned) < 20:
        # Model produced something unusable; surface it so the caller can fall back.
        raise HTTPException(
            status_code=422, detail=f"LLM produced unusable output: {raw[:200]!r}"
        )
    return EnhanceResponse(
        enhanced_prompt=cleaned,
        raw_output=raw,
        model=engine.model_id,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post(
    "/v1/enhance-caption", response_model=CaptionEnhanceResponse, tags=["enhance"]
)
async def enhance_caption(req: CaptionEnhanceRequest) -> CaptionEnhanceResponse:
    """Terse BLIP caption -> detailed LoRA training caption."""
    _require_ready()
    started = time.perf_counter()

    user = (
        f"caption: {req.caption.strip()}\n"
        f"room type: {req.room_type}\n"
        f"style: {req.style or 'unspecified'}"
    )
    messages: list[dict] = [{"role": "system", "content": CAPTION_SYSTEM}]
    for shot_in, shot_out in CAPTION_SHOTS:
        messages.append({"role": "user", "content": shot_in})
        messages.append({"role": "assistant", "content": shot_out})
    messages.append({"role": "user", "content": user})

    try:
        raw = await run_in_threadpool(
            engine.chat,
            messages,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            seed=req.seed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("enhance-caption failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cleaned = clean_single_line(raw)
    if len(cleaned) < 15:
        cleaned = clean_single_line(req.caption)  # keep the original rather than fail
    return CaptionEnhanceResponse(
        caption=cleaned,
        model=engine.model_id,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(req: ChatRequest) -> dict:
    """Minimal OpenAI-compatible chat endpoint."""
    _require_ready()
    messages = [m.model_dump() for m in req.messages]
    try:
        raw = await run_in_threadpool(
            engine.chat,
            messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            seed=req.seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": raw},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/models", tags=["openai"])
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": engine.model_id, "object": "model", "owned_by": "qwen"}],
    }


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": "llm-service",
        "model": engine.model_id,
        "endpoints": [
            "/v1/enhance",
            "/v1/enhance-caption",
            "/v1/chat/completions",
            "/v1/models",
            "/health",
        ],
    }
