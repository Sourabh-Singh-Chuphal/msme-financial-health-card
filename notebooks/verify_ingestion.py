"""Verify ingestion completeness distribution across all 200 synthetic MSMEs.

Run from the project root:
    python notebooks/verify_ingestion.py

Expected result:
    - healthy_established, risky_*, seasonal_business → completeness ≈ 1.0
    - healthy_ntc → completeness ≈ 0.50–0.65 (only 2–3 of 4 sources)

This confirms the critical design property: NTC personas genuinely produce
lower completeness scores, meaning downstream confidence flags will
correctly reflect reduced data availability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.ingestion.validator import validate

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run: python data/synthetic_generators/generate_cohort.py")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    msme_ids: list[str] = manifest["msme_ids"]
    total = len(msme_ids)
    print(f"Running validator across {total} MSMEs...\n")

    records: list[dict] = []
    errors: list[str] = []

    for i, msme_id in enumerate(msme_ids, 1):
        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}] Processing {msme_id}...")

        # Load persona_type from per-MSME metadata
        meta_path = DATA_DIR / msme_id / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            persona_type = meta.get("persona_type", "unknown")
            sources_in_meta = meta.get("sources_available", [])
        except Exception as exc:
            errors.append(f"{msme_id}: cannot read metadata — {exc}")
            continue

        result = validate(msme_id, data_dir=DATA_DIR)

        # Flag discrepancies between metadata declaration and actual files found
        meta_set = set(sources_in_meta)
        found_set = set(result.sources_present)
        if meta_set != found_set:
            errors.append(
                f"{msme_id}: metadata declares {sorted(meta_set)} "
                f"but validator found {sorted(found_set)}"
            )

        records.append(
            {
                "msme_id": msme_id,
                "persona_type": persona_type,
                "completeness_score": result.completeness_score,
                "sources_present": len(result.sources_present),
                "sources_list": ",".join(sorted(result.sources_present)),
                "can_score": result.can_score,
                "n_warnings": len(result.validation_warnings),
                "n_error_warnings": sum(
                    1 for w in result.validation_warnings if w.severity == "error"
                ),
            }
        )

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------ stats
    print("\n" + "=" * 68)
    print("COMPLETENESS SCORE DISTRIBUTION BY PERSONA TYPE")
    print("=" * 68)

    summary = (
        df.groupby("persona_type")["completeness_score"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )
    print(summary.to_string())

    print("\n" + "-" * 68)
    print("SOURCES PRESENT COUNT DISTRIBUTION BY PERSONA TYPE")
    print("-" * 68)
    src_summary = (
        df.groupby("persona_type")["sources_present"]
        .agg(["mean", "min", "max"])
        .round(2)
    )
    print(src_summary.to_string())

    print("\n" + "-" * 68)
    print("CAN_SCORE = FALSE COUNT:")
    cant_score = df[~df["can_score"]]
    print(f"  {len(cant_score)} MSMEs cannot be scored (expected: 0)")

    # Verify key design invariant
    ntc_mean = df[df["persona_type"] == "healthy_ntc"]["completeness_score"].mean()
    established_mean = df[df["persona_type"] == "healthy_established"][
        "completeness_score"
    ].mean()
    print("\n" + "-" * 68)
    print("KEY DESIGN INVARIANT CHECK")
    print("-" * 68)
    print(f"  healthy_ntc mean completeness       : {ntc_mean:.4f}")
    print(f"  healthy_established mean completeness: {established_mean:.4f}")
    gap = established_mean - ntc_mean
    print(f"  Gap (established - ntc)              : {gap:.4f}")
    if gap > 0.20:
        print("  PASS -- NTC genuinely scores lower than established (gap > 0.20)")
    else:
        print("  FAIL -- Gap too small; check source generation or completeness formula")

    if errors:
        print(f"\n  {len(errors)} discrepancy / metadata warnings:")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    # ------------------------------------------------------------------ CSV
    csv_path = OUTPUT_DIR / "ingestion_completeness.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved -> {csv_path}")

    # ---------------------------------------------------------------- chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "MSME Ingestion Completeness by Persona Type",
        fontsize=14,
        fontweight="bold",
    )

    persona_order = [
        "healthy_established",
        "healthy_ntc",
        "risky_declining",
        "risky_volatile",
        "seasonal_business",
    ]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#e67e22", "#3498db"]

    # Bar chart — mean completeness ± std
    ax0 = axes[0]
    means = [
        df[df["persona_type"] == p]["completeness_score"].mean()
        for p in persona_order
    ]
    stds = [
        df[df["persona_type"] == p]["completeness_score"].std()
        for p in persona_order
    ]
    bars = ax0.bar(
        range(len(persona_order)),
        means,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )
    ax0.errorbar(
        range(len(persona_order)),
        means,
        yerr=stds,
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.5,
    )
    ax0.set_xticks(range(len(persona_order)))
    ax0.set_xticklabels(
        [p.replace("_", "\n") for p in persona_order], fontsize=9
    )
    ax0.set_ylabel("Mean Completeness Score (0–1)", fontsize=10)
    ax0.set_title("Mean Completeness Score ± Std Dev", fontsize=11)
    ax0.set_ylim(0, 1.15)
    ax0.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, mean in zip(bars, means):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Box plot — completeness distribution per persona
    ax1 = axes[1]
    box_data = [
        df[df["persona_type"] == p]["completeness_score"].values
        for p in persona_order
    ]
    bp = ax1.boxplot(
        box_data,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax1.set_xticks(range(1, len(persona_order) + 1))
    ax1.set_xticklabels(
        [p.replace("_", "\n") for p in persona_order], fontsize=9
    )
    ax1.set_ylabel("Completeness Score", fontsize=10)
    ax1.set_title("Score Distribution (Box Plot)", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "ingestion_completeness.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved               -> {chart_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
