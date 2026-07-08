"""Unit tests for the MSME credit scoring engine.

Verifies:
1. Weight redistribution: when dimensions are missing, active weights sum to 1.0.
2. Blend ratio shift: composite scorer weights rule-based more heavily for low completeness,
   and ML more heavily for high completeness.
3. Monotonicity: degrading a dimension score or an ingestion-level parameter (e.g., bounce rate)
   results in a lower or equal final score.
"""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from src.features.base import FeatureVector, DimensionFeatures
from src.ingestion.schemas import IngestionResult
from src.features.pipeline import compute as feature_compute
from src.scoring.rule_based_scorer import score_features as rule_score
from src.scoring.ml_scorer import score_features as ml_score
from src.scoring.composite_scorer import compute_score as composite_score


# Helper to build a healthy ingestion result
def _make_healthy_ingestion() -> IngestionResult:
    # GST (12 months)
    gst_df = pd.DataFrame([{
        "tax_period": f"2025-{m:02d}-01",
        "return_type": "GSTR-1/3B",
        "due_date": f"2025-{m+1:02d}-20" if m < 12 else "2026-01-20",
        "filing_date": f"2025-{m+1:02d}-15" if m < 12 else "2026-01-15",
        "filing_delay_days": -5,
        "turnover_reported_gstr1": 1000000.0,
        "turnover_reported_gstr3b": 1000000.0,
        "itc_claimed": 120000.0,
        "mismatch_flag": False
    } for m in range(1, 13)])

    # UPI (12 months daily equivalent)
    upi_df = pd.DataFrame([{
        "date": f"2025-07-{d:02d}",
        "inflow_amount": 50000.0,
        "outflow_amount": 35000.0,
        "transaction_count": 10,
        "unique_counterparty_count": 5,
        "refund_rate": 0.0
    } for d in range(1, 31)])

    # AA Daily (12 months daily equivalent)
    aa_daily_df = pd.DataFrame([{
        "date": f"2025-07-{d:02d}",
        "closing_balance": 300000.0,
        "bounce_count": 0,
        "emi_debit": 0.0,
        "overdraft_used": False
    } for d in range(1, 31)])

    # AA Monthly (12 months)
    aa_monthly_df = pd.DataFrame([{
        "month": f"2025-{m:02d}",
        "average_monthly_balance": 300000.0,
        "bounce_return_count": 0,
        "emi_debits_total": 0.0,
        "overdraft_usage_days": 0
    } for m in range(1, 13)])

    # EPFO (12 months)
    epfo_df = pd.DataFrame([{
        "month": f"2025-{m:02d}",
        "employee_count": 10,
        "wage_bill": 150000.0,
        "pf_contribution_amount": 18000.0,
        "pf_contribution_on_time": True,
        "employee_churn_rate": 0.0
    } for m in range(1, 13)])

    return IngestionResult(
        msme_id="TEST_MSME_HEALTHY",
        sources_present=["gst", "upi", "aa", "epfo"],
        sources_absent=[],
        completeness_score=1.0,
        validated_data={
            "gst": gst_df,
            "upi": upi_df,
            "aa_daily": aa_daily_df,
            "aa_monthly": aa_monthly_df,
            "epfo": epfo_df
        },
        validation_warnings=[],
        can_score=True
    )


# ---------------------------------------------------------------------------
# Weight Redistribution Tests
# ---------------------------------------------------------------------------

def test_weight_redistribution_sums_to_one() -> None:
    # All 5 dimensions present
    fv_all = FeatureVector(
        msme_id="FV_ALL",
        stability=DimensionFeatures("stability", 80.0, {}, confidence=1.0),
        cashflow=DimensionFeatures("cashflow", 85.0, {}, confidence=1.0),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=1.0),
        growth=DimensionFeatures("growth", 75.0, {}, confidence=1.0),
        repayment=DimensionFeatures("repayment", 80.0, {}, confidence=1.0),
        sources_present=["gst", "upi", "aa", "epfo"],
        completeness_score=1.0
    )
    score_all = rule_score(fv_all)
    assert sum(score_all.weights_used.values()) == pytest.approx(1.0)
    assert score_all.weights_used["repayment"] > 0.0

    # Repayment and Growth missing (repayment.score = None, growth.score = None)
    fv_partial = FeatureVector(
        msme_id="FV_PARTIAL",
        stability=DimensionFeatures("stability", 80.0, {}, confidence=1.0),
        cashflow=DimensionFeatures("cashflow", 85.0, {}, confidence=1.0),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=1.0),
        growth=DimensionFeatures("growth", None, {}, confidence=0.0),
        repayment=DimensionFeatures("repayment", None, {}, confidence=0.0),
        sources_present=["gst", "upi"],
        completeness_score=0.5
    )
    score_partial = rule_score(fv_partial)
    assert sum(score_partial.weights_used.values()) == pytest.approx(1.0)
    assert score_partial.weights_used["repayment"] == 0.0
    assert score_partial.weights_used["growth"] == 0.0
    assert score_partial.weights_used["stability"] > 0.0
    assert score_partial.weights_used["cashflow"] > 0.0
    assert score_partial.weights_used["compliance"] > 0.0


# ---------------------------------------------------------------------------
# Composite Blend Ratio Shift Tests
# ---------------------------------------------------------------------------

def test_composite_blend_ratio_shift() -> None:
    # 1. High completeness (>= 0.75) -> 30% rule-based / 70% ML
    fv_high = FeatureVector(
        msme_id="FV_HIGH",
        stability=DimensionFeatures("stability", 80.0, {}, confidence=1.0),
        cashflow=DimensionFeatures("cashflow", 85.0, {}, confidence=1.0),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=1.0),
        growth=DimensionFeatures("growth", 75.0, {}, confidence=1.0),
        repayment=DimensionFeatures("repayment", 80.0, {}, confidence=1.0),
        sources_present=["gst", "upi", "aa", "epfo"],
        completeness_score=1.0
    )
    res_high = composite_score(fv_high)
    assert res_high.blend_ratio_used["rule_based"] == 0.30
    assert res_high.blend_ratio_used["ml"] == 0.70

    # 2. Medium completeness (0.5 to 0.75) -> 50% rule-based / 50% ML
    fv_med = FeatureVector(
        msme_id="FV_MED",
        stability=DimensionFeatures("stability", 80.0, {}, confidence=0.8),
        cashflow=DimensionFeatures("cashflow", 85.0, {}, confidence=0.8),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=0.8),
        growth=DimensionFeatures("growth", None, {}, confidence=0.0),
        repayment=DimensionFeatures("repayment", 80.0, {}, confidence=0.8),
        sources_present=["gst", "upi", "aa"],
        completeness_score=0.6
    )
    res_med = composite_score(fv_med)
    assert res_med.blend_ratio_used["rule_based"] == 0.50
    assert res_med.blend_ratio_used["ml"] == 0.50

    # 3. Low completeness (< 0.5) -> 80% rule-based / 20% ML
    fv_low = FeatureVector(
        msme_id="FV_LOW",
        stability=DimensionFeatures("stability", 80.0, {}, confidence=0.4),
        cashflow=DimensionFeatures("cashflow", None, {}, confidence=0.0),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=0.4),
        growth=DimensionFeatures("growth", None, {}, confidence=0.0),
        repayment=DimensionFeatures("repayment", None, {}, confidence=0.0),
        sources_present=["gst"],
        completeness_score=0.25
    )
    res_low = composite_score(fv_low)
    assert res_low.blend_ratio_used["rule_based"] == 0.80
    assert res_low.blend_ratio_used["ml"] == 0.20


# ---------------------------------------------------------------------------
# Monotonicity Tests
# ---------------------------------------------------------------------------

def test_score_monotonicity_direct() -> None:
    """Direct monotonicity: degrading a dimension score degrades the overall score."""
    fv_healthy = FeatureVector(
        msme_id="FV_HEALTHY",
        stability=DimensionFeatures("stability", 85.0, {}, confidence=1.0),
        cashflow=DimensionFeatures("cashflow", 85.0, {}, confidence=1.0),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=1.0),
        growth=DimensionFeatures("growth", 75.0, {}, confidence=1.0),
        repayment=DimensionFeatures("repayment", 80.0, {}, confidence=1.0),
        sources_present=["gst", "upi", "aa", "epfo"],
        completeness_score=1.0
    )

    fv_degraded = FeatureVector(
        msme_id="FV_DEGRADED",
        stability=DimensionFeatures("stability", 85.0, {}, confidence=1.0),
        # Degrade cashflow from 85.0 to 45.0
        cashflow=DimensionFeatures("cashflow", 45.0, {}, confidence=1.0),
        compliance=DimensionFeatures("compliance", 90.0, {}, confidence=1.0),
        growth=DimensionFeatures("growth", 75.0, {}, confidence=1.0),
        # Degrade repayment from 80.0 to 30.0
        repayment=DimensionFeatures("repayment", 30.0, {}, confidence=1.0),
        sources_present=["gst", "upi", "aa", "epfo"],
        completeness_score=1.0
    )

    score_healthy = rule_score(fv_healthy).overall_score
    score_degraded = rule_score(fv_degraded).overall_score

    assert score_degraded < score_healthy, (
        f"Degraded score ({score_degraded}) should be lower than healthy score ({score_healthy})"
    )


def test_score_monotonicity_ingestion_level() -> None:
    """Ingestion-level monotonicity: worsening the bounce rate degrades the overall score."""
    # 1. Compute features and score for healthy MSME
    ingestion_healthy = _make_healthy_ingestion()
    fv_healthy = feature_compute(ingestion_healthy)
    final_healthy = composite_score(fv_healthy).overall_score

    # 2. Create degraded MSME (introduce bank statement bounces)
    ingestion_degraded = _make_healthy_ingestion()
    
    # Degrade monthly bank statement: add 5 bounced transactions per month
    aa_monthly_bad = ingestion_degraded.validated_data["aa_monthly"].copy()
    aa_monthly_bad["bounce_return_count"] = 5
    ingestion_degraded.validated_data["aa_monthly"] = aa_monthly_bad

    # Degrade daily bank statement: add 1 bounce every 6 days
    aa_daily_bad = ingestion_degraded.validated_data["aa_daily"].copy()
    for d in range(5, 30, 6):
        aa_daily_bad.loc[d, "bounce_count"] = 1
    ingestion_degraded.validated_data["aa_daily"] = aa_daily_bad

    fv_degraded = feature_compute(ingestion_degraded)
    final_degraded = composite_score(fv_degraded).overall_score

    # Assert final score is lower
    assert final_degraded < final_healthy, (
        f"MSME with transaction bounces should score lower. "
        f"Healthy: {final_healthy}, Degraded (Bounces): {final_degraded}"
    )

    # 3. Assert repayment capacity dimension specifically is degraded
    rep_healthy = fv_healthy.repayment.score
    rep_degraded = fv_degraded.repayment.score
    assert rep_degraded < rep_healthy, (
        f"Repayment score must drop with bounces. "
        f"Healthy: {rep_healthy}, Degraded: {rep_degraded}"
    )
