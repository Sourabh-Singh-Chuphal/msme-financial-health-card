"""Verify feature pipeline across all 200 synthetic MSMEs.

Produces a persona x dimension summary table showing mean scores,
then validates that the pattern makes intuitive credit sense:
  - risky_* personas should score low across all dimensions
  - healthy_established should score high
  - seasonal_business should score well (especially growth — deseasonalized)
  - healthy_ntc should score similarly to healthy_established on their covered dimensions

Run from project root:
    python notebooks/verify_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.pipeline import compute as feature_compute
from src.ingestion.validator import validate

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = ["stability", "cashflow", "compliance", "growth", "repayment"]

PERSONA_ORDER = [
    "healthy_established",
    "healthy_ntc",
    "risky_declining",
    "risky_volatile",
    "seasonal_business",
]

PERSONA_COLORS = {
    "healthy_established": "#27ae60",
    "healthy_ntc":         "#f39c12",
    "risky_declining":     "#e74c3c",
    "risky_volatile":      "#c0392b",
    "seasonal_business":   "#2980b9",
}


def main() -> None:
    if not MANIFEST_PATH.exists():
        print("ERROR: manifest.json not found. Run generate_cohort.py first.")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    msme_ids: list[str] = manifest["msme_ids"]
    total = len(msme_ids)
    print(f"Running feature pipeline across {total} MSMEs...\n")

    records: list[dict] = []

    for i, msme_id in enumerate(msme_ids, 1):
        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}] {msme_id}")

        meta_path = DATA_DIR / msme_id / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            persona_type = meta.get("persona_type", "unknown")
        except Exception:
            persona_type = "unknown"

        ingestion = validate(msme_id, data_dir=DATA_DIR)
        fv = feature_compute(ingestion)
        flat = fv.to_flat_dict()
        flat["persona_type"] = persona_type
        records.append(flat)

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------ Table
    score_cols = [f"{d}_score" for d in DIMENSIONS]
    conf_cols  = [f"{d}_confidence" for d in DIMENSIONS]

    print("\n" + "=" * 80)
    print("MEAN DIMENSION SCORES BY PERSONA TYPE  (0-100, None treated as NaN)")
    print("=" * 80)

    summary_score = (
        df.groupby("persona_type")[score_cols]
        .mean()
        .rename(columns={f"{d}_score": d for d in DIMENSIONS})
        .round(1)
    )
    # Reorder rows
    present = [p for p in PERSONA_ORDER if p in summary_score.index]
    summary_score = summary_score.loc[present]
    print(summary_score.to_string())

    print("\n" + "-" * 80)
    print("MEAN DIMENSION CONFIDENCE BY PERSONA TYPE")
    print("-" * 80)
    summary_conf = (
        df.groupby("persona_type")[conf_cols]
        .mean()
        .rename(columns={f"{d}_confidence": d for d in DIMENSIONS})
        .round(3)
    )
    summary_conf = summary_conf.loc[present]
    print(summary_conf.to_string())

    print("\n" + "-" * 80)
    print("REPAYMENT SCORE AVAILABILITY (% MSMEs with score, not None)")
    print("-" * 80)
    avail = df.groupby("persona_type")["repayment_score"].apply(
        lambda s: f"{100 * s.notna().mean():.0f}%"
    )
    print(avail.to_string())

    # ----------------------------------------------------------------- Checks
    print("\n" + "-" * 80)
    print("PATTERN SANITY CHECKS")
    print("-" * 80)

    checks = [
        (
            "risky_declining scores < healthy_established on all dims",
            lambda: all(
                summary_score.loc["risky_declining", d]
                < summary_score.loc["healthy_established", d]
                for d in DIMENSIONS
            ),
        ),
        (
            "seasonal_business growth score within 15pts of healthy_established",
            lambda: abs(
                summary_score.loc["seasonal_business", "growth"]
                - summary_score.loc["healthy_established", "growth"]
            ) < 15,
        ),
        (
            "risky_declining compliance < 50",
            lambda: summary_score.loc["risky_declining", "compliance"] < 50,
        ),
        (
            "healthy_established compliance > 80",
            lambda: summary_score.loc["healthy_established", "compliance"] > 80,
        ),
        (
            "risky_volatile growth < healthy_ntc growth",
            lambda: summary_score.loc["risky_volatile", "growth"]
            < summary_score.loc["healthy_ntc", "growth"],
        ),
    ]

    all_pass = True
    for desc, fn in checks:
        try:
            result = fn()
        except Exception as e:
            result = False
            desc = f"{desc} [ERROR: {e}]"
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {desc}")

    print()
    if all_pass:
        print("All pattern checks PASSED.")
    else:
        print("Some checks FAILED. Review scoring formulas.")

    # ------------------------------------------------------------------ Save
    csv_path = OUTPUT_DIR / "feature_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved -> {csv_path}")

    summary_csv = OUTPUT_DIR / "feature_summary.csv"
    summary_score.to_csv(summary_csv)
    print(f"Summary table saved   -> {summary_csv}")

    # ----------------------------------------------------------------- Chart
    _plot_heatmap(summary_score, OUTPUT_DIR)
    _plot_radar(summary_score, OUTPUT_DIR)

    print("\nDone.")


def _plot_heatmap(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    data = summary.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels(DIMENSIONS, fontsize=11)
    ax.set_yticks(range(len(summary.index)))
    ax.set_yticklabels(summary.index, fontsize=10)

    for i in range(len(summary.index)):
        for j in range(len(DIMENSIONS)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if val < 40 or val > 80 else "black")

    plt.colorbar(im, ax=ax, label="Score (0-100)")
    ax.set_title("MSME Feature Scores: Persona x Dimension", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out / "feature_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved         -> {path}")


def _plot_radar(summary: pd.DataFrame, out: Path) -> None:
    cats = DIMENSIONS
    n = len(cats)
    angles = [k / n * 2 * np.pi for k in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for persona in summary.index:
        vals = summary.loc[persona].tolist()
        vals += vals[:1]
        color = PERSONA_COLORS.get(persona, "gray")
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=persona)
        ax.fill(angles, vals, alpha=0.06, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("Feature Score Radar by Persona", fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    path = out / "feature_radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar chart saved     -> {path}")


if __name__ == "__main__":
    main()
