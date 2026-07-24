"""Negative-prompt builder.

Combines a curated set of default negatives (things that ruin interior renders)
with any extra negatives the user supplies, de-duplicated.
"""
from __future__ import annotations

DEFAULT_NEGATIVES: list[str] = [
    "low quality", "blurry", "out of focus", "grainy", "jpeg artifacts",
    "distorted", "deformed", "disproportionate furniture", "duplicate furniture",
    "crooked walls", "bad perspective", "warped lines", "tilted horizon",
    "watermark", "signature", "text", "logo", "caption",
    "low resolution", "pixelated", "oversaturated", "underexposed",
    "cluttered", "messy", "dirty", "cropped", "frame border",
    "people", "person", "human", "hands", "cartoon", "cgi look", "3d render look",
]


def build_negative_prompt(user_negative: str | None = None) -> str:
    tokens: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        for part in raw.split(","):
            t = part.strip()
            k = t.lower()
            if t and k not in seen:
                seen.add(k)
                tokens.append(t)

    for neg in DEFAULT_NEGATIVES:
        add(neg)
    if user_negative:
        add(user_negative)
    return ", ".join(tokens)
