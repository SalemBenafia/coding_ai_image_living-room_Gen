"""HTTP client for the SDXL ai-service."""
from __future__ import annotations

import logging

import httpx

from .config import get_settings

logger = logging.getLogger("ai_client")
settings = get_settings()


class AIServiceError(RuntimeError):
    pass


async def generate_image(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> bytes:
    """Call the ai-service /generate endpoint; return PNG bytes."""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    url = f"{settings.ai_service_url.rstrip('/')}/generate"
    try:
        async with httpx.AsyncClient(timeout=settings.ai_request_timeout) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise AIServiceError(f"AI service unreachable: {exc}") from exc

    if resp.status_code != 200:
        detail = resp.text[:300]
        raise AIServiceError(f"AI service error {resp.status_code}: {detail}")

    content_type = resp.headers.get("content-type", "")
    if "image/png" not in content_type:
        raise AIServiceError(f"AI service returned unexpected type: {content_type}")
    return resp.content


async def ping() -> bool:
    url = f"{settings.ai_service_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
