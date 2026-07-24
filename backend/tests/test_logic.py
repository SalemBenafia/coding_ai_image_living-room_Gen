"""Unit tests for the backend's pure business logic (no services required).

Run:  cd backend && python -m pytest tests/ -q
"""
from __future__ import annotations

import os

os.environ.setdefault("MINIO_ROOT_PASSWORD", "test")

from app.negative_prompt import DEFAULT_NEGATIVES, build_negative_prompt
from app.prompt_enhancer import _dedupe_join, enhance_local
from app.schemas import find_unsafe_terms
from app.styles import STYLE_PRESETS, style_fragment


# ---- negative prompt --------------------------------------------------------

def test_negative_includes_defaults():
    neg = build_negative_prompt(None)
    assert "low quality" in neg
    assert all(d in neg for d in DEFAULT_NEGATIVES)


def test_negative_merges_user_terms():
    neg = build_negative_prompt("blurry, my-custom-thing")
    assert "my-custom-thing" in neg
    # 'blurry' is already a default -> not duplicated
    assert neg.lower().count("blurry") == 1


# ---- prompt enhancement -----------------------------------------------------

def test_dedupe_join_removes_duplicates():
    assert _dedupe_join("modern, modern", "modern, cozy") == "modern, cozy"


def test_local_enhance_folds_style_and_quality():
    out = enhance_local("cozy room", "scandinavian")
    assert "scandinavian" in out
    assert "8k" in out or "ultra realistic" in out


def test_local_enhance_without_style():
    out = enhance_local("cozy room", None)
    assert out.startswith("cozy room")


# ---- safety validation ------------------------------------------------------

def test_detects_unsafe_topics():
    assert "gun" in find_unsafe_terms("a room with a gun")


def test_detects_profanity():
    assert find_unsafe_terms("this shit room")


def test_clean_prompt_passes():
    assert find_unsafe_terms("a cozy modern living room") == []


# ---- styles -----------------------------------------------------------------

def test_ten_style_presets():
    assert len(STYLE_PRESETS) == 10
    for key, preset in STYLE_PRESETS.items():
        assert {"label", "description", "fragment"} <= preset.keys()


def test_style_fragment_lookup():
    assert "modern" in style_fragment("modern")
    assert style_fragment("does-not-exist") == ""
    assert style_fragment(None) == ""
