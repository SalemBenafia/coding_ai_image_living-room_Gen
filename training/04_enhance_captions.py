#!/usr/bin/env python
"""Step 4 (optional) — Enrich BLIP captions with an LLM (Groq / Qwen 2.5).

BLIP captions are short ("a living room with a sofa"). This rewrites them into
detailed, style-aware interior-design descriptions that train a better LoRA.

Skips gracefully (leaves captions unchanged) if GROQ_API_KEY is not set.

Usage:
  python 04_enhance_captions.py [--model qwen-2.5-32b]
"""
from __future__ import annotations

import argparse
import os

from common import CAPTIONS_DIR, load_dotenv

SYSTEM = (
    "You rewrite short image captions into rich interior-design training captions "
    "for a text-to-image model. Given a basic caption of a living room, expand it "
    "into ONE comma-separated description including furniture, materials, flooring, "
    "wall treatment, lighting, color palette and decor. Keep 'living room'. Under "
    "60 words. Output ONLY the caption — no quotes, no preamble."
)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", "qwen-2.5-32b"))
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[enhance] GROQ_API_KEY not set — skipping (captions left as-is).")
        return 0

    from groq import Groq

    client = Groq(api_key=api_key)
    txts = sorted(CAPTIONS_DIR.glob("*.txt"))
    if not txts:
        print(f"ERROR: no captions in {CAPTIONS_DIR}. Run 03_generate_captions.py first.")
        return 1

    for i, txt in enumerate(txts, 1):
        base = txt.read_text(encoding="utf-8").strip()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": base},
                ],
                temperature=0.7,
                max_tokens=160,
            )
            enriched = (resp.choices[0].message.content or "").strip()
            if enriched:
                txt.write_text(enriched, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[enhance] {txt.name}: kept original ({exc})")
        if i % 25 == 0 or i == len(txts):
            print(f"[enhance] {i}/{len(txts)}")

    print(f"[enhance] done ({len(txts)} captions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
