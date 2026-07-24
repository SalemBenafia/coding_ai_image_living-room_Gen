#!/usr/bin/env python
"""End-to-end smoke test against a running backend.

Exercises the full flow: styles -> generate -> fetch image -> history -> validation
errors -> delete. Exits non-zero on the first failure.

Usage:
  BACKEND=http://localhost:8000 python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time

import httpx

BACKEND = os.environ.get("BACKEND", "http://localhost:8000")
passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}  {detail}")


def wait_for_health(timeout: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{BACKEND}/health", timeout=5)
            if r.status_code == 200:
                print(f"backend health: {r.json()}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.5)
    raise SystemExit("backend did not become healthy in time")


def main() -> int:
    print(f"== smoke test against {BACKEND} ==")
    wait_for_health()

    with httpx.Client(base_url=BACKEND, timeout=120) as c:
        # styles
        r = c.get("/api/styles")
        styles = r.json()
        check("GET /api/styles returns presets", r.status_code == 200 and len(styles) >= 5,
              f"status={r.status_code}")

        # generate
        payload = {
            "prompt": "a cozy scandinavian living room with big windows and plants",
            "style": "scandinavian",
            "width": 768, "height": 768, "steps": 12, "guidance_scale": 7.0,
            "seed": 123, "enhance_prompt": True,
        }
        r = c.post("/api/generate", json=payload)
        check("POST /api/generate -> 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
        data = r.json() if r.status_code == 200 else {}
        gen_id = data.get("id")
        check("response has id", bool(gen_id))
        check("seed echoed (reproducible)", data.get("seed") == 123, f"seed={data.get('seed')}")
        check("enhanced_prompt built", "living room" in (data.get("enhanced_prompt") or "").lower())
        check("negative_prompt built", "low quality" in (data.get("negative_prompt") or "").lower())
        check("image_url points to backend proxy", (data.get("image_url") or "").startswith("/api/images/"))

        # fetch image
        if gen_id:
            r = c.get(f"/api/images/{gen_id}")
            body = r.content
            check("GET image returns PNG", r.status_code == 200 and body[:8] == b"\x89PNG\r\n\x1a\n",
                  f"status={r.status_code}")

        # history contains it
        r = c.get("/api/history?limit=5")
        page = r.json()
        check("history total >= 1", page.get("total", 0) >= 1)
        check("history contains new item", any(it["id"] == gen_id for it in page.get("items", [])))

        # validation: too-short prompt -> 422
        r = c.post("/api/generate", json={"prompt": "hi"})
        check("short prompt rejected (422)", r.status_code == 422, f"status={r.status_code}")

        # validation: unsafe content -> 422
        r = c.post("/api/generate", json={"prompt": "a living room with a gun on the table"})
        check("unsafe prompt rejected (422)", r.status_code == 422, f"status={r.status_code}")

        # validation: unknown style -> 422
        r = c.post("/api/generate", json={"prompt": "a nice living room", "style": "nonsense"})
        check("unknown style rejected (422)", r.status_code == 422, f"status={r.status_code}")

        # delete
        if gen_id:
            r = c.delete(f"/api/history/{gen_id}")
            check("DELETE -> 204", r.status_code == 204, f"status={r.status_code}")
            r = c.get(f"/api/images/{gen_id}")
            check("image gone after delete (404)", r.status_code == 404, f"status={r.status_code}")

    print(f"\n== {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
