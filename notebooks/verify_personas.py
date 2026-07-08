"""Verify synthetic persona distinguishability via summary statistics and plots.

Run after cohort generation to confirm healthy vs risky archetypes separate
on observable metrics — if they overlap, downstream scoring cannot tell the
hackathon story of alternate-data credit assessment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python notebooks/verify_personas.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.synthetic_generators.aa_generator import summarize_aa
from data.synthetic_generators.epfo_generator import summarize_epfo
from data.synthetic_generators.gst_generator import summarize_gst
from data.synthetic_generators.persona_builder import generate_cohort
from data.synthetic_generators.upi_generator import summarize_upi

RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_DIR = PROJECT_ROOT / "notebooks/output"


def _load_or_generate(raw_dir: Path) -> list[dict]:
    manifest = raw_dir / "manifest.json"
    if not manifest.exists():
        print("No cohort found — generating 200 MSMEs (seed=42)...")
        generate_cohort(seed=42, output_dir=raw_dir)

    rows: list[dict] = []
    for meta_path in sorted(raw_dir.glob("MSME_*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        msme_dir = meta_path.parent
        row = {
            "msme_id": meta["msme_id"],
            "persona_type": meta["persona_type"],
            "business_type": meta["business_type"],
            "source_count": len(meta["sources_available"]),
        }

        gst_path = msme_dir / "gst.csv"
        if gst_path.exists():
            gst = pd.read_csv(gst_path)
            row.update({f"gst_{k}": v for k, v in summarize_gst(gst).items()})

        upi_path = msme_dir / "upi.csv"
        if upi_path.exists():
            upi = pd.read_csv(upi_path)
            row.update({f"upi_{k}": v for k, v in summarize_upi(upi).items()})

        aa_path = msme_dir / "aa_daily.csv"
        if aa_path.exists():
            aa = pd.read_csv(aa_path)
            aa_monthly_path = msme_dir / "aa_monthly.csv"
            if aa_monthly_path.exists():
                aa.attrs["monthly_summary"] = pd.read_csv(aa_monthly_path)
            row.update({f"aa_{k}": v for k, v in summarize_aa(aa).items()})

        epfo_path = msme_dir / "epfo.csv"
        if epfo_path.exists():
            epfo = pd.read_csv(epfo_path)
            row.update({f"epfo_{k}": v for k, v in summarize_epfo(epfo).items()})

        rows.append(row)

    return rows


def persona_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean key metrics grouped by persona_type."""
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("gst_", "upi_", "aa_", "epfo_")) or c == "source_count"
    ]
    summary = df.groupby("persona_type")[metric_cols].mean().round(3)
    return summary


def print_distinguishability_report(summary: pd.DataFrame) -> None:
    """Print side-by-side checks that personas separate on key metrics."""
    checks = [
        ("gst_avg_filing_delay_days", "healthy_established", "risky_declining", "lt"),
        ("gst_avg_monthly_turnover", "healthy_established", "risky_declining", "gt"),
        ("gst_turnover_trend", "healthy_established", "risky_declining", "gt"),
        ("upi_avg_daily_inflow", "healthy_established", "risky_declining", "gt"),
        ("upi_inflow_cv", "risky_volatile", "healthy_established", "gt"),
        ("aa_total_bounces", "risky_declining", "healthy_established", "gt"),
        ("aa_overdraft_day_rate", "risky_declining", "healthy_established", "gt"),
        ("epfo_avg_churn_rate", "risky_declining", "healthy_established", "gt"),
        ("gst_turnover_cv", "seasonal_business", "healthy_established", "gt"),
        ("source_count", "healthy_established", "healthy_ntc", "gt"),
    ]

    print("\n=== Persona Distinguishability Checks ===")
    passed = 0
    for metric, a, b, direction in checks:
        if metric not in summary.columns:
            continue
        va, vb = summary.loc[a, metric], summary.loc[b, metric]
        ok = va < vb if direction == "lt" else va > vb
        status = "PASS" if ok else "FAIL"
        passed += int(ok)
        print(f"  [{status}] {metric}: {a}={va:.3f} vs {b}={vb:.3f} (expect {a} {'<' if direction == 'lt' else '>'} {b})")
    print(f"\n{passed}/{len(checks)} checks passed.\n")


def plot_persona_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Synthetic MSME Persona Separation (cohort means)", fontsize=13)

    plots = [
        ("gst_avg_filing_delay_days", "GST Avg Filing Delay (days)", axes[0, 0]),
        ("gst_avg_monthly_turnover", "GST Avg Monthly Turnover (INR)", axes[0, 1]),
        ("upi_avg_daily_inflow", "UPI Avg Daily Inflow (INR)", axes[0, 2]),
        ("aa_total_bounces", "AA Total Bounces (12 mo)", axes[1, 0]),
        ("upi_inflow_cv", "UPI Inflow CV", axes[1, 1]),
        ("epfo_avg_churn_rate", "EPFO Avg Churn Rate", axes[1, 2]),
    ]

    persona_order = [
        "healthy_established",
        "healthy_ntc",
        "seasonal_business",
        "risky_declining",
        "risky_volatile",
    ]
    colors = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#c0392b"]

    for col, title, ax in plots:
        if col not in df.columns:
            ax.set_visible(False)
            continue
        means = df.groupby("persona_type")[col].mean().reindex(persona_order)
        ax.bar(range(len(means)), means.values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([p.replace("_", "\n") for p in persona_order], fontsize=7, rotation=0)

    plt.tight_layout()
    out_path = output_dir / "persona_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def main() -> None:
    rows = _load_or_generate(RAW_DIR)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} MSMEs across personas:")
    print(df["persona_type"].value_counts().sort_index().to_string())

    summary = persona_summary_table(df)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Per-Persona Summary Statistics (cohort means) ===")
    print(summary.to_string())

    print_distinguishability_report(summary)
    plot_persona_comparison(df, OUTPUT_DIR)

    summary.to_csv(OUTPUT_DIR / "persona_summary.csv")
    print(f"Saved summary CSV -> {OUTPUT_DIR / 'persona_summary.csv'}")


if __name__ == "__main__":
    main()
