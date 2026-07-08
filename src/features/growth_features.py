"""Growth Trajectory features from GST, UPI, and EPFO data.

Growth answers: "Is this business expanding or contracting?"
We measure at three time horizons (3 / 6 / 12 months) to distinguish
short-term acceleration from sustained long-run trajectory.  Recent trends
receive somewhat higher weight for credit decisions.

Data sources consumed:
    GST  → turnover trend at 3m / 6m / 12m (deseasonalized)
    UPI  → transaction inflow trend at 3m / 12m (deseasonalized)
    EPFO → employee count trend (signals real-economy growth vs revenue alone)

Seasonality handling — the most important feature of this module:
    Raw month-over-month revenue comparisons produce misleading signals for
    seasonal businesses.  We apply ratio-to-baseline deseasonalization using
    the Indian retail seasonal calendar (Diwali, monsoon, wedding season).

    Example: a sweet shop's November revenue is ~1.55× their July revenue.
    Without adjustment, Nov→Dec looks like a 25% decline.  After dividing
    by the seasonal factor, the underlying trend is correctly identified.

    is_deseasonalized=True is recorded in sub_features so downstream
    explainability code can surface "growth score calculated on
    seasonality-adjusted data" to credit officers.

Missing-source behaviour:
    All three absent → score=None
    Any one present  → partial score; confidence reflects coverage.
    Score=None returned only if zero sources contribute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._seasonality import (
    compute_deseasonalized_trend,
    trend_to_score,
)
from src.features.base import DimensionFeatures, _clamp
from src.ingestion.schemas import IngestionResult

DIMENSION = "growth"
_MAX_MONTHS = 12


def compute(result: IngestionResult) -> DimensionFeatures:
    """Compute growth trajectory features for one MSME."""
    gst = result.validated_data.get("gst")
    upi = result.validated_data.get("upi")
    epfo = result.validated_data.get("epfo")

    has_gst = gst is not None and not gst.empty
    has_upi = upi is not None and not upi.empty
    has_epfo = epfo is not None and not epfo.empty

    if not has_gst and not has_upi and not has_epfo:
        return DimensionFeatures(
            dimension=DIMENSION,
            score=None,
            sub_features={},
            missing_reason=(
                "No GST, UPI, or EPFO data available for growth trajectory assessment."
            ),
            confidence=0.0,
        )

    sub: dict[str, float | None] = {"is_deseasonalized": True}
    component_scores: list[float] = []
    component_weights: list[float] = []

    # ------------------------------------------------------------------ GST
    if has_gst:
        gst_vals, gst_months = _extract_gst_series(gst)
        if gst_vals is not None:
            _add_gst_trends(
                gst_vals, gst_months, sub, component_scores, component_weights
            )

    # ------------------------------------------------------------------ UPI
    if has_upi:
        upi_vals, upi_months = _extract_upi_monthly(upi)
        if upi_vals is not None:
            _add_upi_trends(
                upi_vals, upi_months, sub, component_scores, component_weights
            )

    # ----------------------------------------------------------------- EPFO
    if has_epfo:
        _add_epfo_trend(epfo, sub, component_scores, component_weights)

    if not component_scores:
        return DimensionFeatures(
            dimension=DIMENSION,
            score=None,
            sub_features=sub,
            missing_reason="Insufficient data points to compute any trend.",
            confidence=0.0,
        )

    total_w = sum(component_weights)
    raw_score = sum(s * w for s, w in zip(component_scores, component_weights)) / total_w
    score = round(_clamp(raw_score), 2)
    confidence = _confidence(gst, upi, epfo, has_gst, has_upi, has_epfo)

    return DimensionFeatures(
        dimension=DIMENSION,
        score=score,
        sub_features=sub,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# GST trend helpers
# ---------------------------------------------------------------------------

def _extract_gst_series(
    gst: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract sorted (turnover_values, month_indices) from GST DataFrame."""
    try:
        df = gst.copy()
        df["_dt"] = pd.to_datetime(df["tax_period"], errors="coerce")
        df = df.dropna(subset=["_dt"]).sort_values("_dt")
        vals = pd.to_numeric(
            df["turnover_reported_gstr1"], errors="coerce"
        ).dropna().values
        months = df["_dt"].dt.month.values[: len(vals)]
        return (vals, months) if len(vals) >= 2 else (None, None)
    except Exception:
        return None, None


def _add_gst_trends(
    vals: np.ndarray,
    months: np.ndarray,
    sub: dict,
    scores: list,
    weights: list,
) -> None:
    n = len(vals)

    # 12-month trend (all data)
    t12 = compute_deseasonalized_trend(vals, months)
    sub["gst_trend_12m"] = round(t12, 6) if t12 is not None else None
    s12 = trend_to_score(t12)
    if s12 is not None:
        scores.append(s12)
        weights.append(0.14)  # lower weight — slow-moving signal

    # 6-month trend (last 6 months)
    if n >= 6:
        t6 = compute_deseasonalized_trend(vals, months, last_n=6)
        sub["gst_trend_6m"] = round(t6, 6) if t6 is not None else None
        s6 = trend_to_score(t6)
        if s6 is not None:
            scores.append(s6)
            weights.append(0.16)
    else:
        sub["gst_trend_6m"] = None

    # 3-month trend (last 3 months) — most recent signal, highest weight
    if n >= 3:
        t3 = compute_deseasonalized_trend(vals, months, last_n=3)
        sub["gst_trend_3m"] = round(t3, 6) if t3 is not None else None
        s3 = trend_to_score(t3)
        if s3 is not None:
            scores.append(s3)
            weights.append(0.20)
    else:
        sub["gst_trend_3m"] = None


# ---------------------------------------------------------------------------
# UPI monthly aggregation & trend
# ---------------------------------------------------------------------------

def _extract_upi_monthly(
    upi: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Aggregate daily UPI inflows to monthly, return (values, month_indices)."""
    try:
        df = upi.copy()
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["_dt"])
        df["inflow_amount"] = pd.to_numeric(df["inflow_amount"], errors="coerce")
        monthly = (
            df.groupby(df["_dt"].dt.to_period("M"))["inflow_amount"]
            .sum()
            .sort_index()
        )
        if len(monthly) < 2:
            return None, None
        vals = monthly.values.astype(float)
        # Month number from period index
        months = np.array([p.month for p in monthly.index])
        return vals, months
    except Exception:
        return None, None


def _add_upi_trends(
    vals: np.ndarray,
    months: np.ndarray,
    sub: dict,
    scores: list,
    weights: list,
) -> None:
    n = len(vals)

    # 12-month UPI trend
    t12 = compute_deseasonalized_trend(vals, months)
    sub["upi_trend_12m"] = round(t12, 6) if t12 is not None else None
    s12 = trend_to_score(t12)
    if s12 is not None:
        scores.append(s12)
        weights.append(0.16)

    # 3-month UPI trend — high-frequency recent signal
    if n >= 3:
        t3 = compute_deseasonalized_trend(vals, months, last_n=3)
        sub["upi_trend_3m"] = round(t3, 6) if t3 is not None else None
        s3 = trend_to_score(t3)
        if s3 is not None:
            scores.append(s3)
            weights.append(0.20)
    else:
        sub["upi_trend_3m"] = None


# ---------------------------------------------------------------------------
# EPFO headcount trend
# ---------------------------------------------------------------------------

def _add_epfo_trend(
    epfo: pd.DataFrame,
    sub: dict,
    scores: list,
    weights: list,
) -> None:
    try:
        hc = pd.to_numeric(epfo["employee_count"], errors="coerce").dropna().values
        if len(hc) < 2:
            sub["headcount_trend"] = None
            return
        x = np.arange(len(hc), dtype=float)
        slope = float(np.polyfit(x, hc, 1)[0])
        mean_hc = float(np.mean(hc))
        monthly_rate = slope / mean_hc if mean_hc > 0 else 0.0
        sub["headcount_trend"] = round(monthly_rate, 6)
        s = trend_to_score(monthly_rate)
        if s is not None:
            scores.append(s)
            weights.append(0.14)
    except Exception:
        sub["headcount_trend"] = None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _confidence(
    gst: pd.DataFrame | None,
    upi: pd.DataFrame | None,
    epfo: pd.DataFrame | None,
    has_gst: bool,
    has_upi: bool,
    has_epfo: bool,
) -> float:
    n_sources = int(has_gst) + int(has_upi) + int(has_epfo)
    source_frac = n_sources / 3

    n_gst = len(gst) if gst is not None else 0
    n_epfo = len(epfo) if epfo is not None else 0
    n_upi = (
        int(pd.to_datetime(upi["date"]).dt.to_period("M").nunique())
        if upi is not None and not upi.empty
        else 0
    )
    history = max(n_gst, n_epfo, n_upi) / _MAX_MONTHS
    return round(source_frac * min(history, 1.0), 4)
