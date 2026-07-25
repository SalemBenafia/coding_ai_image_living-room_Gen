#!/usr/bin/env python
"""Stage 09 (analytics) — Exploratory data report for the training corpus.

Produces a single figure summarising the LoRA dataset: style distribution,
per-source contribution, caption-length distribution, and the room-type filter's
keep/drop breakdown. Useful for spotting class imbalance before/after training.

Reads the workshop CSVs (manifest/prepared/captions/captions_final) and writes
docs/dataset_report.png.

Usage:
  python s09_dataset_report.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_DIR, REPO_ROOT


def _load(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> int:
    prepared = _load("prepared.csv")
    captions = _load("captions.csv")
    final = _load("captions_final.csv")
    if final.empty:
        print("[report] captions_final.csv missing — run the pipeline first.")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Living-Room LoRA — training corpus report", fontsize=15, fontweight="bold")

    # 1) style distribution (final)
    styles = final["style"].value_counts().sort_values()
    axes[0, 0].barh(styles.index, styles.values, color="#c9a24a")
    axes[0, 0].set_title(f"Style distribution (n={len(final)})")
    axes[0, 0].set_xlabel("images")

    # 2) source contribution
    if "source" in final:
        src = final["source"].value_counts()
        axes[0, 1].pie(src.values, labels=src.index, autopct="%1.0f%%",
                       colors=["#4a86c9", "#c95a4a"])
        axes[0, 1].set_title("Dataset source contribution")

    # 3) caption length (words) histogram
    wc = final["caption"].fillna("").str.split().apply(len)
    axes[1, 0].hist(wc, bins=20, color="#5aa469", edgecolor="white")
    axes[1, 0].axvline(wc.median(), color="black", ls="--", label=f"median {int(wc.median())}")
    axes[1, 0].set_title("Caption length (words)")
    axes[1, 0].set_xlabel("words per caption")
    axes[1, 0].legend()

    # 4) room-type filter keep/drop (prepared -> captioned survivors)
    if not prepared.empty and not captions.empty:
        by_src_prep = prepared["source"].value_counts()
        by_src_kept = captions["source"].value_counts()
        labels = list(by_src_prep.index)
        kept = [int(by_src_kept.get(s, 0)) for s in labels]
        dropped = [int(by_src_prep.get(s, 0)) - k for s, k in zip(labels, kept)]
        axes[1, 1].bar(labels, kept, label="kept (living room)", color="#5aa469")
        axes[1, 1].bar(labels, dropped, bottom=kept, label="dropped (other room)", color="#c95a4a")
        axes[1, 1].set_title("BLIP room-type filter")
        axes[1, 1].legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REPO_ROOT / "docs" / "dataset_report.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[report] wrote {out}")

    # Also print a compact text summary.
    print("\n=== corpus summary ===")
    print(f"images: {len(final)}")
    if "source" in final:
        print("by source:", dict(final["source"].value_counts()))
    print("by style:", dict(final["style"].value_counts()))
    print(f"caption words: min={wc.min()} median={int(wc.median())} max={wc.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
