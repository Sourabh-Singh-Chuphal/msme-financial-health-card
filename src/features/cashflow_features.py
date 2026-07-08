"""Cash Flow Health features from UPI transactions and AA bank statements.

Cash flow answers: "Can this business actually pay its bills?"
Even a high-revenue MSME can have a cashflow problem if outflows eat all inflows
or if balance volatility leaves them exposed in lean months.

Data sources consumed:
    Primary:   UPI → net_cashflow_ratio, inflow_cv, avg_daily_inflow
    Secondary: AA  → avg_monthly_balance, bounce_rate, days_of_cash_buffer

Seasonality handling:
    We measure the TREND in recent UPI inflows using deseasonalized values so
    a merchant whose October inflow is 1.35× their April inflow doesn't appear
    to be in decline when they compare July to November — the seasonal dip is
    expected, not a cashflow signal.  The ratio-based metrics (net_cashflow_ratio,
    inflow_cv) are computed on seasonality-adjusted inflows.

Missing-source behaviour:
    Both absent → score=None
    Only UPI    → score from UPI metrics only (confidence reduced)
    Only AA     → score from AA metrics only (confidence reduced)
    Both present → full score (confidence = 1.0 × history_factor)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._seasonality import deseasonalize
from src.features.base import DimensionFeatures, _clamp
from src.ingestion.schemas import IngestionResult

DIMENSION = "cashflow"
_MAX_MONTHS = 12


def compute(result: IngestionResult) -> DimensionFeatures:
    """Compute cashflow features for one MSME."""
    upi = result.validated_data.get("upi")
    aa_daily = result.validated_data.get("aa_daily")
    aa_monthly = result.validated_data.get("aa_monthly")

    has_upi = upi is not None and not upi.empty
    has_aa = (aa_daily is not None and not aa_daily.empty) or (
        aa_monthly is not None and not aa_monthly.empty
    )

    if not has_upi and not has_aa:
        return DimensionFeatures(
            dimension=DIMENSION,
            score=None,
            sub_features={},
            missing_reason="Neither UPI nor AA data available for cashflow assessment.",
            confidence=0.0,
        )

    sub: dict[str, float | None] = {}
    upi_component: float | None = None
    aa_component: float | None = None

    if has_upi:
        upi_component, sub = _upi_cashflow(upi, sub)

    if has_aa:
        aa_component, sub = _aa_cashflow(aa_daily, aa_monthly, sub, upi)

    score = _blend(upi_component, aa_component, has_upi, has_aa)
    confidence = _confidence(upi, aa_monthly, aa_daily, has_upi, has_aa)

    return DimensionFeatures(
        dimension=DIMENSION,
        score=round(score, 2),
        sub_features=sub,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# UPI sub-computation
# ---------------------------------------------------------------------------

def _upi_cashflow(upi: pd.DataFrame, sub: dict) -> tuple[float, dict]:
    inflow = pd.to_numeric(upi["inflow_amount"], errors="coerce").fillna(0.0)
    outflow = pd.to_numeric(upi["outflow_amount"], errors="coerce").fillna(0.0)

    total_inflow = float(inflow.sum())
    total_outflow = float(outflow.sum())

    # Net cashflow ratio: (inflow - outflow) / inflow
    net_ratio = (
        (total_inflow - total_outflow) / total_inflow
        if total_inflow > 0
        else 0.0
    )
    sub["net_cashflow_ratio"] = round(net_ratio, 4)
    # ratio=0.20 → 90, ratio=0 → 50, ratio=-0.20 → 10
    net_ratio_score = _clamp(50.0 + net_ratio * 200.0)

    avg_daily_inflow = float(inflow.mean())
    sub["avg_daily_inflow"] = round(avg_daily_inflow, 2)

    # Inflow volatility — deseasonalize first so Diwali spikes don't inflate CV
    if "date" in upi.columns and len(inflow) >= 2:
        try:
            months = pd.to_datetime(upi["date"]).dt.month.values
            adj_inflow = deseasonalize(inflow.values, months)
            mean_adj = float(np.mean(adj_inflow))
            inflow_cv = (
                float(np.std(adj_inflow) / mean_adj) if mean_adj > 0 else None
            )
        except Exception:
            inflow_cv = None
    else:
        inflow_cv = None
    sub["inflow_cv"] = round(inflow_cv, 4) if inflow_cv is not None else None
    inflow_stability_score = (
        _clamp(100.0 - inflow_cv * 100.0) if inflow_cv is not None else 50.0
    )

    # Refund rate — high refunds signal customer dissatisfaction or fraud
    if "refund_rate" in upi.columns:
        active = upi[upi["inflow_amount"] > 0]
        avg_refund = (
            float(
                pd.to_numeric(active["refund_rate"], errors="coerce").mean()
            )
            if len(active) > 0
            else 0.0
        )
        sub["avg_refund_rate"] = round(avg_refund, 4)
        refund_score = _clamp(100.0 - avg_refund * 400.0)
    else:
        sub["avg_refund_rate"] = None
        refund_score = 75.0

    component = (
        0.40 * net_ratio_score
        + 0.30 * inflow_stability_score
        + 0.30 * refund_score
    )
    return _clamp(component), sub


# ---------------------------------------------------------------------------
# AA sub-computation
# ---------------------------------------------------------------------------

def _aa_cashflow(
    aa_daily: pd.DataFrame | None,
    aa_monthly: pd.DataFrame | None,
    sub: dict,
    upi: pd.DataFrame | None,
) -> tuple[float, dict]:
    bounce_score = 80.0
    buffer_score = 50.0
    balance_trend_score = 50.0

    # --- bounce rate from monthly
    if aa_monthly is not None and not aa_monthly.empty:
        avg_bal = pd.to_numeric(
            aa_monthly["average_monthly_balance"], errors="coerce"
        ).mean()
        bounces = pd.to_numeric(
            aa_monthly["bounce_return_count"], errors="coerce"
        )
        avg_bounces = float(bounces.mean())
        sub["avg_monthly_bounces"] = round(avg_bounces, 2)
        bounce_score = _clamp(100.0 - avg_bounces * 15.0)

        sub["avg_monthly_balance_cashflow"] = (
            round(float(avg_bal), 2) if not np.isnan(avg_bal) else None
        )

        # Balance trend: is the average balance improving?
        bal_series = pd.to_numeric(
            aa_monthly["average_monthly_balance"], errors="coerce"
        ).dropna().values
        if len(bal_series) >= 3:
            x = np.arange(len(bal_series), dtype=float)
            slope = float(np.polyfit(x, bal_series, 1)[0])
            mean_bal = float(np.mean(bal_series))
            monthly_growth = slope / abs(mean_bal) if abs(mean_bal) > 0 else 0.0
            sub["balance_trend_monthly"] = round(monthly_growth, 4)
            balance_trend_score = _clamp(50.0 + monthly_growth * 2000.0)
        else:
            sub["balance_trend_monthly"] = None
    else:
        sub["avg_monthly_bounces"] = None
        sub["avg_monthly_balance_cashflow"] = None
        sub["balance_trend_monthly"] = None

    # --- days of cash buffer: avg_balance / avg_daily_outflow
    if aa_daily is not None and not aa_daily.empty:
        daily_balances = pd.to_numeric(
            aa_daily["closing_balance"], errors="coerce"
        ).dropna()
        avg_balance_daily = float(daily_balances.mean())

        # Estimate daily outflow from UPI if available, else EMI + implied
        if upi is not None and not upi.empty:
            avg_daily_outflow = float(
                pd.to_numeric(upi["outflow_amount"], errors="coerce").mean()
            )
        elif "emi_debit" in aa_daily.columns:
            avg_daily_outflow = float(
                pd.to_numeric(aa_daily["emi_debit"], errors="coerce").fillna(0).sum()
                / max(len(aa_daily), 1)
            )
        else:
            avg_daily_outflow = max(avg_balance_daily / 30, 1.0)

        if avg_daily_outflow > 0:
            cash_buffer_days = max(0.0, avg_balance_daily / avg_daily_outflow)
            sub["days_of_cash_buffer"] = round(cash_buffer_days, 1)
            # 90 days buffer → 100, 0 days → 0
            buffer_score = _clamp(cash_buffer_days / 90.0 * 100.0)
        else:
            sub["days_of_cash_buffer"] = None
    else:
        sub["days_of_cash_buffer"] = None

    component = (
        0.35 * bounce_score
        + 0.35 * buffer_score
        + 0.30 * balance_trend_score
    )
    return _clamp(component), sub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend(
    upi_score: float | None,
    aa_score: float | None,
    has_upi: bool,
    has_aa: bool,
) -> float:
    if has_upi and has_aa:
        return 0.55 * upi_score + 0.45 * aa_score
    if has_upi:
        return upi_score
    return aa_score


def _confidence(
    upi: pd.DataFrame | None,
    aa_monthly: pd.DataFrame | None,
    aa_daily: pd.DataFrame | None,
    has_upi: bool,
    has_aa: bool,
) -> float:
    n_upi = (
        int(pd.to_datetime(upi["date"]).dt.to_period("M").nunique())
        if upi is not None and not upi.empty
        else 0
    )
    n_aa = (
        len(aa_monthly)
        if aa_monthly is not None and not aa_monthly.empty
        else (
            int(pd.to_datetime(aa_daily["date"]).dt.to_period("M").nunique())
            if aa_daily is not None and not aa_daily.empty
            else 0
        )
    )
    source_frac = (int(has_upi) + int(has_aa)) / 2
    history = max(n_upi, n_aa) / _MAX_MONTHS
    return round(source_frac * min(history, 1.0), 4)
