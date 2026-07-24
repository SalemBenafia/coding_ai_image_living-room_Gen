"""Prompt enhancement orchestration.

Turns a short user prompt into a rich, SDXL-friendly prompt. Two strategies:

  * "llm"   — call the Qwen llm-service (Step 3 of the flow). If it is unreachable
              or returns something unusable, we transparently fall back to
              rule-based so image generation never breaks.
  * "local" — deterministic rule-based enhancement (no network, always available).

Whichever path runs, we always guarantee the style fragment and photographic
quality suffix are present and de-duplicated, so SDXL gets consistent guidance.
"""
from __future__ import annotations

import logging

from . import llm_client
from .config import get_settings
from .styles import QUALITY_SUFFIX, style_fragment

logger = logging.getLogger("prompt_enhancer")
settings = get_settings()


def _dedupe_join(*parts: str) -> str:
    """Join fragments, dropping empty and duplicate comma-separated tokens."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        for token in part.split(","):
            t = token.strip()
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                out.append(t)
    return ", ".join(out)


def enhance_local(prompt: str, style: str | None) -> str:
    """Rule-based enhancement: prompt + style fragment + quality suffix."""
    return _dedupe_join(prompt, style_fragment(style), QUALITY_SUFFIX)


async def build_positive_prompt(
    prompt: str, style: str | None, enhance: bool, *, seed: int | None = None
) -> tuple[str, str]:
    """Assemble the final positive prompt.

    Returns (positive_prompt, method) where method is one of
    "llm" | "local" | "local-fallback" so it can be recorded/reported.
    """
    if not enhance:
        return enhance_local(prompt, style), "local"

    if settings.prompt_enhancer.lower() == "llm":
        try:
            enhanced = await llm_client.enhance_prompt(prompt, style, seed=seed)
            # Guarantee style + quality descriptors even if the LLM omitted them.
            return _dedupe_join(enhanced, style_fragment(style), QUALITY_SUFFIX), "llm"
        except llm_client.LLMServiceError as exc:
            logger.warning("LLM enhancement failed (%s); using local fallback.", exc)
            return enhance_local(prompt, style), "local-fallback"

    return enhance_local(prompt, style), "local"


def active_backend() -> str:
    """Which enhancer is configured, for /health reporting."""
    return "llm" if settings.prompt_enhancer.lower() == "llm" else "local"
