"""HTTP client for the Qwen llm-service (prompt enhancement).

The backend never runs an LLM itself — it delegates to the dedicated llm-service
so the model stays resident on the GPU and can be scaled independently.
"""
from __future__ import annotations

import logging

import httpx

from .config import get_settings

logger = logging.getLogger("llm_client")
settings = get_settings()


class LLMServiceError(RuntimeError):
    """Raised when the llm-service is unreachable or returns an error."""


async def enhance_prompt(prompt: str, style: str | None, *, seed: int | None = None) -> str:
    """Call llm-service /v1/enhance; return the enhanced prompt text.

    Raises LLMServiceError on any failure so the caller can fall back to the
    deterministic rule-based enhancer.
    """
    url = f"{settings.llm_service_url.rstrip('/')}/v1/enhance"
    payload = {"prompt": prompt, "style": style, "seed": seed}
    try:
        async with httpx.AsyncClient(timeout=settings.llm_request_timeout) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise LLMServiceError(f"llm-service unreachable: {exc}") from exc

    if resp.status_code != 200:
        raise LLMServiceError(f"llm-service error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    enhanced = (data.get("enhanced_prompt") or "").strip()
    if not enhanced:
        raise LLMServiceError("llm-service returned an empty enhanced_prompt")
    return enhanced


async def ping() -> bool:
    url = f"{settings.llm_service_url.rstrip('/')}/health/live"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
