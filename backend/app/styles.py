"""Interior-design style presets.

Each preset contributes a fragment appended to the positive prompt. Keeping these
server-side means the frontend only sends a style *key* and the backend controls
the actual prompt engineering.
"""
from __future__ import annotations

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "modern": {
        "label": "Modern",
        "description": "Clean lines, neutral palette, uncluttered surfaces.",
        "fragment": "modern interior design, clean lines, neutral color palette, sleek furniture, uncluttered",
    },
    "scandinavian": {
        "label": "Scandinavian",
        "description": "Light woods, white walls, cozy minimalism (hygge).",
        "fragment": "scandinavian interior, light oak wood, white walls, cozy hygge, soft textiles, bright and airy",
    },
    "minimalist": {
        "label": "Minimalist",
        "description": "Essential furniture only, lots of negative space.",
        "fragment": "minimalist interior, essential furniture only, lots of negative space, muted tones, calm",
    },
    "industrial": {
        "label": "Industrial",
        "description": "Exposed brick, concrete, black metal, Edison bulbs.",
        "fragment": "industrial interior, exposed brick wall, concrete floor, black metal frames, edison bulb lighting",
    },
    "luxury": {
        "label": "Luxury",
        "description": "Marble, velvet, gold accents, statement lighting.",
        "fragment": "luxury interior, marble surfaces, velvet upholstery, gold accents, statement chandelier, opulent",
    },
    "japanese": {
        "label": "Japanese",
        "description": "Natural materials, low furniture, shoji screens, zen.",
        "fragment": "japanese interior, natural wood, low furniture, shoji screens, tatami, zen minimalism, wabi-sabi",
    },
    "contemporary": {
        "label": "Contemporary",
        "description": "Current trends, mixed textures, curated accents.",
        "fragment": "contemporary interior design, mixed textures, curated decor, layered lighting, current trends",
    },
    "rustic": {
        "label": "Rustic",
        "description": "Reclaimed wood, stone, warm earthy tones, cozy.",
        "fragment": "rustic interior, reclaimed wood beams, stone fireplace, warm earthy tones, cozy farmhouse",
    },
    "bohemian": {
        "label": "Bohemian",
        "description": "Layered textiles, plants, warm eclectic mix.",
        "fragment": "bohemian interior, layered textiles, macrame, abundant plants, warm eclectic mix, rattan furniture",
    },
    "coastal": {
        "label": "Coastal",
        "description": "Airy blues and whites, natural light, breezy.",
        "fragment": "coastal interior, airy blue and white palette, linen fabrics, natural light, breezy relaxed feel",
    },
}

# A quality suffix always appended for photographic realism.
QUALITY_SUFFIX = (
    "interior design photography, architectural photography, ultra realistic, "
    "highly detailed, sharp focus, professional lighting, 8k, magazine quality"
)


def style_keys() -> list[str]:
    return list(STYLE_PRESETS.keys())


def style_fragment(style: str | None) -> str:
    if not style:
        return ""
    preset = STYLE_PRESETS.get(style.lower())
    return preset["fragment"] if preset else ""
