"""Financial Stability features from GST filings and AA bank statements.

Stability answers: "Is this business consistent?"
A business that files GST on time every month, has stable (seasonality-adjusted)
turnover, and maintains a steady bank balance is stable.  High variance in any
of these is a warning signal regardless of the absolute level.

Data sources consumed:
    Primary:   GST  → filing_regularity, deseasonalized_turnover_cv
    Secondary: AA   → balance_cv, min_balance_health, avg_balance_adequacy

Seasonality handling:
    We deseasonalize GST turnover before computing its coefficient of variation.
    This ensures a predictable seasonal business (e.g. a wedding caterer) is not
    penalised for expected Diwali/monsoon swings — their underlying consistency
    is the signal, not raw monthly amplitude.

Missing-source behaviour:
    Both absent → score=None
    Only GST    → score from GST metrics only (confidence reduced)
    Only AA     → score from AA metrics only (confidence reduced)
    Both present → full score (confidence = 1.0 × history_factor)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._seasonality import deseasonalize
from src.features.base import DimensionFeatures, _clamp
from src.ingestion.schemas import IngestionResult

DIMENSION = "stability"
_MAX_MONTHS = 12


def compute(result: IngestionResult) -> DimensionFeatures:
    """Compute stability features for one MSME."""
    gst = result.validated_data.get("gst")
    aa_daily = result.validated_data.get("aa_daily")
    aa_monthly = result.validated_data.get("aa_monthly")

    has_gst = gst is not None and not gst.empty
    has_aa = (aa_daily is not None and not aa_daily.empty) or (
        aa_monthly is not None and not aa_monthly.empty
    )

    if not has_gst and not has_aa:
        return DimensionFeatures(
            dimension=DIMENSION,
            score=None,
            sub_features={},
            missing_reason="Neither GST nor AA data available for stability assessment.",
            confidence=0.0,
        )

    sub: dict[str, float | None] = {}
    gst_component_score: float | None = None
    aa_component_score: float | None = None

    # ------------------------------------------------------------------ GST
    if has_gst:
        gst_component_score, sub = _gst_stability(gst, sub)

    # ------------------------------------------------------------------- AA
    if has_aa:
        aa_component_score, sub = _aa_stability(aa_daily, aa_monthly, sub)

    # ------------------- blend components based on availability
    score = _blend(gst_component_score, aa_component_score, has_gst, has_aa)
    confidence = _confidence(gst, aa_monthly, aa_daily, has_gst, has_aa)

    return DimensionFeatures(
        dimension=DIMENSION,
        score=round(score, 2),
        sub_features=sub,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# GST sub-computation
# ---------------------------------------------------------------------------

def _gst_stability(
    gst: pd.DataFrame, sub: dict
) -> tuple[float, dict]:
    """Return (component_score 0-100, updated sub_features)."""

    # --- filing regularity
    filing_delay = pd.to_numeric(gst["filing_delay_days"], errors="coerce")
    late_filing_rate = float((filing_delay > 0).mean())
    avg_delay = float(filing_delay.clip(lower=0).mean())
    filing_regularity = 1.0 - late_filing_rate

    sub["filing_regularity"] = round(filing_regularity, 4)
    sub["avg_filing_delay_days"] = round(avg_delay, 2)

    filing_score = filing_regularity * 100.0
    delay_score = _clamp(100.0 - avg_delay * 4.0)  # 25-day avg delay → 0

    # --- turnover consistency (deseasonalized)
    turnover_vals = pd.to_numeric(
        gst["turnover_reported_gstr1"], errors="coerce"
    ).dropna().values

    if len(turnover_vals) >= 2:
        try:
            months = pd.to_datetime(gst["tax_period"]).dt.month.values[
                : len(turnover_vals)
            ]
            adj = deseasonalize(turnover_vals, months)
            mean_adj = float(np.mean(adj))
            cv = float(np.std(adj) / mean_adj) if mean_adj > 0 else None
        except Exception:
            cv = None
        sub["turnover_cv"] = round(cv, 4) if cv is not None else None
        turnover_stability_score = (
            _clamp(100.0 - cv * 150.0) if cv is not None else 50.0
        )
    else:
        sub["turnover_cv"] = None
        turnover_stability_score = 50.0  # neutral — not enough data

    # --- mismatch (a GST-specific compliance/stability signal)
    if "mismatch_flag" in gst.columns:
        mismatch_rate = float(gst["mismatch_flag"].astype(bool).mean())
    else:
        mismatch_rate = 0.0
    sub["gst_mismatch_rate"] = round(mismatch_rate, 4)
    mismatch_score = _clamp(100.0 - mismatch_rate * 200.0)

    # GST component = weighted sum of filing, delay, turnover_stability, mismatch
    component = (
        0.35 * filing_score
        + 0.20 * delay_score
        + 0.30 * turnover_stability_score
        + 0.15 * mismatch_score
    )
    return _clamp(component), sub


# ---------------------------------------------------------------------------
# AA sub-computation
# ---------------------------------------------------------------------------

def _aa_stability(
    aa_daily: pd.DataFrame | None,
    aa_monthly: pd.DataFrame | None,
    sub: dict,
) -> tuple[float, dict]:
    """Return (component_score 0-100, updated sub_features)."""

    balance_cv_score = 50.0
    avg_balance_score = 50.0
    overdraft_score = 100.0

    # --- balance volatility from daily data
    if aa_daily is not None and not aa_daily.empty:
        balances = pd.to_numeric(
            aa_daily["closing_balance"], errors="coerce"
        ).dropna().values

        if len(balances) >= 2:
            mean_bal = float(np.mean(balances))
            std_bal = float(np.std(balances))
            # Use absolute mean (balance can be negative in overdraft)
            cv = std_bal / abs(mean_bal) if abs(mean_bal) > 0 else None
            sub["balance_cv"] = round(cv, 4) if cv is not None else None
            balance_cv_score = (
                _clamp(100.0 - cv * 100.0) if cv is not None else 50.0
            )
        else:
            sub["balance_cv"] = None

        # --- overdraft episodes
        overdraft_used = aa_daily.get("overdraft_used")
        if overdraft_used is not None:
            od_rate = float(overdraft_used.astype(bool).mean())
            sub["overdraft_day_rate_stability"] = round(od_rate, 4)
            overdraft_score = _clamp(100.0 - od_rate * 250.0)
        else:
            sub["overdraft_day_rate_stability"] = None

        sub["min_balance"] = (
            float(np.min(balances)) if len(balances) > 0 else None
        )
    else:
        sub["balance_cv"] = None
        sub["overdraft_day_rate_stability"] = None
        sub["min_balance"] = None

    # --- average balance from monthly summary
    if aa_monthly is not None and not aa_monthly.empty:
        avg_balances = pd.to_numeric(
            aa_monthly["average_monthly_balance"], errors="coerce"
        ).dropna().values
        sub["avg_monthly_balance"] = (
            round(float(np.mean(avg_balances)), 2) if len(avg_balances) > 0 else None
        )
        # Score: positive average balance is better; normalise against 500k baseline
        avg_b = float(np.mean(avg_balances)) if len(avg_balances) > 0 else 0.0
        avg_balance_score = _clamp(50.0 + avg_b / 10_000.0)
    else:
        sub["avg_monthly_balance"] = None
        avg_balance_score = 50.0

    # AA component = balance_cv (quality), overdraft (health), avg_balance (level)
    component = (
        0.40 * balance_cv_score
        + 0.35 * overdraft_score
        + 0.25 * avg_balance_score
    )
    return _clamp(component), sub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend(
    gst_score: float | None,
    aa_score: float | None,
    has_gst: bool,
    has_aa: bool,
) -> float:
    if has_gst and has_aa:
        return 0.55 * gst_score + 0.45 * aa_score
    if has_gst:
        return gst_score
    return aa_score


def _confidence(
    gst: pd.DataFrame | None,
    aa_monthly: pd.DataFrame | None,
    aa_daily: pd.DataFrame | None,
    has_gst: bool,
    has_aa: bool,
) -> float:
    n_gst = len(gst) if gst is not None else 0
    n_aa = (
        len(aa_monthly)
        if aa_monthly is not None and not aa_monthly.empty
        else (
            int(
                pd.to_datetime(aa_daily["date"]).dt.to_period("M").nunique()
            )
            if aa_daily is not None and not aa_daily.empty
            else 0
        )
    )
    source_frac = (int(has_gst) + int(has_aa)) / 2
    history = max(n_gst, n_aa) / _MAX_MONTHS
    return round(source_frac * min(history, 1.0), 4)
