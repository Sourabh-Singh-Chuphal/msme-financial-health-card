"""Compliance Discipline features from GST filings and EPFO records.

Compliance answers: "Does this business honour its legal obligations?"
Banks treat GST compliance as a proxy for management quality — a business
that consistently files late, mismatches GSTR-1 vs GSTR-3B, and fails to
deposit PF on time is raising behavioural red flags that underwriters notice
even before looking at financials.

Data sources consumed:
    Primary:   GST  → filing_delay, late_filing_rate, GSTR mismatch rate
    Secondary: EPFO → PF contribution regularity, employee churn rate

Missing-source behaviour:
    Both absent  → score=None
    Only GST     → score from GST metrics only (confidence reduced)
    Only EPFO    → score from EPFO metrics only (confidence reduced)
    Both present → full score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import DimensionFeatures, _clamp
from src.ingestion.schemas import IngestionResult

DIMENSION = "compliance"
_MAX_MONTHS = 12


def compute(result: IngestionResult) -> DimensionFeatures:
    """Compute compliance features for one MSME."""
    gst = result.validated_data.get("gst")
    epfo = result.validated_data.get("epfo")

    has_gst = gst is not None and not gst.empty
    has_epfo = epfo is not None and not epfo.empty

    if not has_gst and not has_epfo:
        return DimensionFeatures(
            dimension=DIMENSION,
            score=None,
            sub_features={},
            missing_reason="Neither GST nor EPFO data available for compliance assessment.",
            confidence=0.0,
        )

    sub: dict[str, float | None] = {}
    gst_component: float | None = None
    epfo_component: float | None = None

    if has_gst:
        gst_component, sub = _gst_compliance(gst, sub)

    if has_epfo:
        epfo_component, sub = _epfo_compliance(epfo, sub)

    score = _blend(gst_component, epfo_component, has_gst, has_epfo)
    confidence = _confidence(gst, epfo, has_gst, has_epfo)

    return DimensionFeatures(
        dimension=DIMENSION,
        score=round(score, 2),
        sub_features=sub,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# GST sub-computation
# ---------------------------------------------------------------------------

def _gst_compliance(gst: pd.DataFrame, sub: dict) -> tuple[float, dict]:
    # --- late filing rate
    filing_delay = pd.to_numeric(gst["filing_delay_days"], errors="coerce")
    late_filing_rate = float((filing_delay > 0).mean())
    sub["late_filing_rate"] = round(late_filing_rate, 4)
    late_score = _clamp((1.0 - late_filing_rate) * 100.0)

    # --- average filing delay (only for late filings)
    avg_delay = float(filing_delay.clip(lower=0).mean())
    sub["avg_filing_delay_days"] = round(avg_delay, 2)
    # 0 days → 100, 20 days → 0
    delay_score = _clamp(100.0 - avg_delay * 5.0)

    # --- GSTR-1 / GSTR-3B mismatch frequency
    if "mismatch_flag" in gst.columns:
        mismatch_rate = float(gst["mismatch_flag"].astype(bool).mean())
    else:
        mismatch_rate = 0.0
    sub["mismatch_rate"] = round(mismatch_rate, 4)
    # Any mismatch is bad; 20% mismatch rate → score 0
    mismatch_score = _clamp((1.0 - mismatch_rate) * 100.0 - mismatch_rate * 100.0)

    # Weighted GST compliance score
    component = 0.45 * late_score + 0.30 * delay_score + 0.25 * mismatch_score
    return _clamp(component), sub


# ---------------------------------------------------------------------------
# EPFO sub-computation
# ---------------------------------------------------------------------------

def _epfo_compliance(epfo: pd.DataFrame, sub: dict) -> tuple[float, dict]:
    # --- PF contribution regularity (on-time rate)
    on_time = epfo["pf_contribution_on_time"].astype(bool)
    pf_regularity = float(on_time.mean())
    sub["pf_regularity_rate"] = round(pf_regularity, 4)
    pf_score = pf_regularity * 100.0

    # --- employee churn rate (high churn signals instability / labour dispute)
    churn = pd.to_numeric(epfo["employee_churn_rate"], errors="coerce").dropna()
    avg_churn = float(churn.mean()) if len(churn) > 0 else 0.0
    sub["avg_churn_rate"] = round(avg_churn, 4)
    # 5% churn = industry normal → small penalty; 20% → heavy penalty
    churn_score = _clamp(100.0 - avg_churn * 300.0)

    # --- headcount stability: declining headcount is a compliance red flag
    if "employee_count" in epfo.columns and len(epfo) >= 2:
        headcounts = pd.to_numeric(
            epfo["employee_count"], errors="coerce"
        ).dropna().values
        x = np.arange(len(headcounts), dtype=float)
        slope = float(np.polyfit(x, headcounts, 1)[0])
        mean_hc = float(np.mean(headcounts))
        hc_trend = slope / mean_hc if mean_hc > 0 else 0.0
        sub["headcount_stability_trend"] = round(hc_trend, 4)
        # Growing or flat → 50-100; declining → 0-50
        hc_score = _clamp(50.0 + hc_trend * 2000.0)
    else:
        sub["headcount_stability_trend"] = None
        hc_score = 50.0

    component = 0.50 * pf_score + 0.30 * churn_score + 0.20 * hc_score
    return _clamp(component), sub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend(
    gst_score: float | None,
    epfo_score: float | None,
    has_gst: bool,
    has_epfo: bool,
) -> float:
    if has_gst and has_epfo:
        return 0.60 * gst_score + 0.40 * epfo_score
    if has_gst:
        return gst_score
    return epfo_score


def _confidence(
    gst: pd.DataFrame | None,
    epfo: pd.DataFrame | None,
    has_gst: bool,
    has_epfo: bool,
) -> float:
    n_gst = len(gst) if gst is not None else 0
    n_epfo = len(epfo) if epfo is not None else 0
    source_frac = (int(has_gst) + int(has_epfo)) / 2
    history = max(n_gst, n_epfo) / _MAX_MONTHS
    return round(source_frac * min(history, 1.0), 4)
