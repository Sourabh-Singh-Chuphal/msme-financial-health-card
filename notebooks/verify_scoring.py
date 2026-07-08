"""Verify MSME credit scoring engine across all 200 synthetic MSMEs.

Scores each MSME using the trained model and rule-based engine, prints a
summary table of final scores and confidence bands by persona, and plots
the score distributions.

Run from project root:
    python notebooks/verify_scoring.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.ingestion.validator import validate
from src.features.pipeline import compute as feature_compute
from src.scoring.composite_scorer import compute_score

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}. Please run train_model.py or synthetic generators first.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    msme_ids = manifest["msme_ids"]
    total = len(msme_ids)

    print(f"Running composite scoring across {total} MSMEs...\n")

    records = []
    for i, msme_id in enumerate(msme_ids, 1):
        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}] Scoring {msme_id}...")

        # Get persona
        meta_path = DATA_DIR / msme_id / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        persona = meta.get("persona_type", "unknown")

        # Score
        ingestion = validate(msme_id, data_dir=DATA_DIR)
        fv = feature_compute(ingestion)
        score_res = compute_score(fv)

        records.append({
            "msme_id": msme_id,
            "persona_type": persona,
            "completeness_score": fv.completeness_score,
            "rule_based_score": score_res.rule_score.overall_score,
            "ml_score": score_res.ml_score.overall_score,
            "final_score": score_res.overall_score,
            "confidence_score": score_res.confidence_score,
            "confidence_band": score_res.confidence_band,
            "rule_blend_weight": score_res.blend_ratio_used["rule_based"],
            "ml_blend_weight": score_res.blend_ratio_used["ml"],
            "can_score": ingestion.can_score
        })

    df = pd.DataFrame(records)

    # 1. Show distribution stats by persona
    print("\n" + "=" * 80)
    print("FINAL CREDIT SCORE STATS BY PERSONA TYPE")
    print("=" * 80)

    summary_score = (
        df.groupby("persona_type")["final_score"]
        .agg(["count", "mean", "std", "min", "max"])
        .loc[PERSONA_ORDER]
        .round(2)
    )
    print(summary_score.to_string())

    print("\n" + "-" * 80)
    print("CONFIDENCE SCORE & BAND DISTRIBUTION BY PERSONA TYPE")
    print("-" * 80)
    
    # Calculate average confidence score and counts of confidence bands
    summary_conf = (
        df.groupby("persona_type")["confidence_score"]
        .agg(["mean", "min", "max"])
        .loc[PERSONA_ORDER]
        .round(3)
    )
    
    bands_df = df.groupby(["persona_type", "confidence_band"]).size().unstack(fill_value=0)
    # Ensure all columns exist
    for band in ["High", "Medium", "Low"]:
        if band not in bands_df.columns:
            bands_df[band] = 0
    bands_df = bands_df.loc[PERSONA_ORDER]
    
    print("Mean Confidence Scores:")
    print(summary_conf.to_string())
    print("\nConfidence Band Counts:")
    print(bands_df.to_string())

    # 2. Check scoring sanity
    print("\n" + "-" * 80)
    print("SCORING SYSTEM SANITY CHECK")
    print("-" * 80)
    
    h_est_mean = summary_score.loc["healthy_established", "mean"]
    h_ntc_mean = summary_score.loc["healthy_ntc", "mean"]
    r_dec_mean = summary_score.loc["risky_declining", "mean"]
    r_vol_mean = summary_score.loc["risky_volatile", "mean"]

    checks = [
        (
            "Healthy Established scores highest on average",
            h_est_mean == max(summary_score["mean"])
        ),
        (
            "Risky Declining scores lowest on average",
            r_dec_mean == min(summary_score["mean"])
        ),
        (
            "Healthy NTC scores well (mean > 70)",
            h_ntc_mean > 70.0
        ),
        (
            "Healthy NTC has lower/wider confidence than Established",
            summary_conf.loc["healthy_ntc", "mean"] < summary_conf.loc["healthy_established", "mean"]
        ),
        (
            "Mean score difference (established - declining) is substantial (> 20 points)",
            (h_est_mean - r_dec_mean) > 20.0
        )
    ]

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")

    print()
    if all_pass:
        print("All credit scoring sanity checks PASSED.")
    else:
        print("Some checks FAILED. Review scorer configuration or persona generation.")

    # 3. Save CSV outputs
    csv_path = OUTPUT_DIR / "composite_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved -> {csv_path}")

    # 4. Plot Boxplot distribution of Final Scores by Persona
    fig, ax = plt.subplots(figsize=(10, 6))
    
    boxplot_data = [df[df["persona_type"] == p]["final_score"].values for p in PERSONA_ORDER]
    
    bp = ax.boxplot(
        boxplot_data,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2)
    )
    ax.set_xticks(range(1, len(PERSONA_ORDER) + 1))
    ax.set_xticklabels([p.replace("_", "\n") for p in PERSONA_ORDER])
    
    for patch, p_name in zip(bp["boxes"], PERSONA_ORDER):
        patch.set_facecolor(PERSONA_COLORS[p_name])
        patch.set_alpha(0.7)
        
    ax.set_ylabel("Final Credit Score (0 - 100)", fontsize=11)
    ax.set_title("MSME Credit Score Distribution by Persona", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "composite_scores.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved             -> {plot_path}")


if __name__ == "__main__":
    main()
