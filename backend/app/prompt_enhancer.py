"""Prompt enhancement.

Turns a short user prompt into a rich, SDXL-friendly prompt. Two backends:

  * "groq"  — calls the Groq API (fast hosted Qwen/Llama). Requires GROQ_API_KEY.
  * "local" — a deterministic, rule-based enhancer (no network, always available).

If "groq" is selected but the key is missing or the call fails, we transparently
fall back to "local" so image generation never breaks.
"""
from __future__ import annotations

import logging

from .config import get_settings
from .styles import QUALITY_SUFFIX, style_fragment

logger = logging.getLogger("prompt_enhancer")
settings = get_settings()

_SYSTEM = (
    "You are a prompt engineer for an interior-design text-to-image model "
    "(Stable Diffusion XL). Rewrite the user's request into ONE vivid, comma-"
    "separated prompt describing a living-room interior. Include concrete details: "
    "furniture, materials, flooring, wall treatment, lighting, color palette, decor, "
    "and camera/photography descriptors. Keep it under 70 words. Output ONLY the "
    "prompt text — no quotes, no preamble, no explanation."
)


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


def _enhance_local(prompt: str, style: str | None) -> str:
    """Rule-based enhancement: prompt + style fragment + quality suffix."""
    return _dedupe_join(prompt, style_fragment(style), QUALITY_SUFFIX)


def _enhance_groq(prompt: str, style: str | None) -> str:
    from groq import Groq  # imported lazily so the dep is optional

    client = Groq(api_key=settings.groq_api_key)
    user = prompt if not style else f"{prompt}\nPreferred style: {style}"
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Groq returned an empty completion")
    # Always guarantee quality descriptors even if the LLM omits them.
    return _dedupe_join(text, style_fragment(style), QUALITY_SUFFIX)


def enhance_prompt(prompt: str, style: str | None = None) -> str:
    """Return an enhanced positive prompt. Never raises — falls back to local."""
    backend = settings.prompt_enhancer.lower()
    if backend == "groq" and settings.groq_api_key:
        try:
            return _enhance_groq(prompt, style)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groq enhancement failed (%s); using local fallback.", exc)
    elif backend == "groq":
        logger.info("PROMPT_ENHANCER=groq but GROQ_API_KEY is unset; using local.")
    return _enhance_local(prompt, style)


def build_positive_prompt(prompt: str, style: str | None, enhance: bool) -> str:
    """Assemble the final positive prompt.

    enhance=True uses the configured LLM (with local fallback); enhance=False uses
    only deterministic rule-based assembly (style fragment + quality suffix).
    """
    if enhance:
        return enhance_prompt(prompt, style)
    return _enhance_local(prompt, style)


def active_backend() -> str:
    """Which enhancer will actually be used, for /health reporting."""
    if settings.prompt_enhancer.lower() == "groq" and settings.groq_api_key:
        return "groq"
    return "local"
