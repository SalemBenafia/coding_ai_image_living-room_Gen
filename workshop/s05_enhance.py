#!/usr/bin/env python
"""Stage 05 — Enrich BLIP captions with Qwen 2.5 1.5B.

BLIP captions are short ("a living room with a couch and a table"). This stage
calls the llm-service (Qwen) /v1/enhance-caption endpoint to expand each into a
detailed, style-aware interior-design caption — the "Improve Caption" step —
producing far better LoRA training text.

The known `style` label is passed to Qwen and a consistent trigger phrase is
prepended so the LoRA learns a reusable activation token.

Reads data/captions.csv, writes data/captions_final.csv, and stores each final
caption to MinIO under captions_enhanced/<stem>.txt.

Falls back to a deterministic template if the llm-service is unreachable.

Usage:
  python s05_enhance.py [--no-upload] [--no-llm]
"""
from __future__ import annotations

import argparse
import csv
import io
import time

import httpx

from config import (
    BUCKET_DATA,
    DATA_DIR,
    KEY_CAPTIONS_FINAL,
    LLM_SERVICE_URL,
    TRIGGER,
    minio_client,
)

QUALITY = ("interior design photography, architectural photography, "
           "highly detailed, natural lighting")


def enhance_via_service(caption: str, style: str, client: httpx.Client) -> str:
    url = f"{LLM_SERVICE_URL.rstrip('/')}/v1/enhance-caption"
    resp = client.post(url, json={"caption": caption, "room_type": "living room",
                                  "style": style, "temperature": 0.6})
    resp.raise_for_status()
    return resp.json()["caption"]


def enhance_template(caption: str, style: str) -> str:
    """Deterministic fallback: fold the style + quality descriptors into the caption."""
    parts = [f"a {style} living room"] if style else ["a living room"]
    tail = caption.lower().replace("a living room with", "").replace("a living room", "").strip(" ,")
    if tail:
        parts.append(tail)
    parts.append(QUALITY)
    seen, out = set(), []
    for token in ", ".join(parts).split(","):
        t = token.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)


def with_trigger(caption: str) -> str:
    if TRIGGER.lower() in caption.lower():
        return caption
    return f"{TRIGGER}, {caption}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="use template fallback only")
    args = ap.parse_args()

    cap_csv = DATA_DIR / "captions.csv"
    if not cap_csv.exists():
        print(f"[enhance] ERROR: {cap_csv} missing. Run s04_caption.py first.")
        return 1
    with cap_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    use_llm = not args.no_llm
    http = httpx.Client(timeout=60.0) if use_llm else None
    if use_llm:
        try:
            http.get(f"{LLM_SERVICE_URL.rstrip('/')}/health/live", timeout=5.0).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"[enhance] llm-service unavailable ({exc}); using template fallback")
            use_llm = False

    client = None if args.no_upload else minio_client()
    started = time.time()
    out_rows: list[dict] = []
    for i, row in enumerate(rows, 1):
        style = row.get("style", "") or ""
        base = row.get("blip_caption", "") or "a living room"
        try:
            if use_llm:
                final = enhance_via_service(base, style, http)
            else:
                final = enhance_template(base, style)
        except Exception as exc:  # noqa: BLE001
            print(f"[enhance] row {i}: llm failed ({exc}); template")
            final = enhance_template(base, style)
        final = with_trigger(final)
        row["caption"] = final
        out_rows.append(row)

        if client is not None:
            stem = row["filename"].rsplit(".", 1)[0]
            data = final.encode("utf-8")
            client.put_object(BUCKET_DATA, f"{KEY_CAPTIONS_FINAL}/{stem}.txt",
                              io.BytesIO(data), length=len(data), content_type="text/plain")
        if i % 50 == 0 or i == len(rows):
            print(f"[enhance] {i}/{len(rows)}")

    out_csv = DATA_DIR / "captions_final.csv"
    fields = ["filename", "source", "room_type", "style", "room_known", "blip_caption", "caption"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[enhance] wrote {len(out_rows)} captions ({'llm' if use_llm else 'template'}) "
          f"in {time.time() - started:.0f}s")
    if out_rows:
        print("[enhance] example:", out_rows[0]["caption"])
    return 0 if out_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
