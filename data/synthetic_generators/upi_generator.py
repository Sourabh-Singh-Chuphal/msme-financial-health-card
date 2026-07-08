"""Synthetic UPI transaction logs for MSME cashflow assessment.

UPI is often the only digital payment trail for New-to-Credit micro-retailers.
This generator produces daily inflow/outflow patterns whose volume, regularity,
and counterparty diversity differ by business type — retail vs services vs
manufacturing — mirroring how analysts infer business health from payment rails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.synthetic_generators.utils import (
    BusinessType,
    GenerationWindow,
    apply_trend,
    clamp,
    iter_days,
    seasonality_multiplier,
)


@dataclass(frozen=True)
class UPIRhythm:
    """Business-type baseline for daily UPI activity."""

    base_daily_inflow: float
    base_daily_outflow_ratio: float
    base_txn_count: int
    base_unique_counterparties: int
    base_refund_rate: float
    weekend_factor: float
    inflow_volatility: float


BUSINESS_RHYTHMS: dict[BusinessType, UPIRhythm] = {
    BusinessType.RETAIL: UPIRhythm(
        base_daily_inflow=45_000,
        base_daily_outflow_ratio=0.72,
        base_txn_count=85,
        base_unique_counterparties=40,
        base_refund_rate=0.025,
        weekend_factor=1.25,
        inflow_volatility=0.15,
    ),
    BusinessType.SERVICES: UPIRhythm(
        base_daily_inflow=120_000,
        base_daily_outflow_ratio=0.55,
        base_txn_count=12,
        base_unique_counterparties=8,
        base_refund_rate=0.008,
        weekend_factor=0.65,
        inflow_volatility=0.10,
    ),
    BusinessType.MANUFACTURING: UPIRhythm(
        base_daily_inflow=280_000,
        base_daily_outflow_ratio=0.80,
        base_txn_count=6,
        base_unique_counterparties=5,
        base_refund_rate=0.004,
        weekend_factor=0.45,
        inflow_volatility=0.20,
    ),
}


@dataclass(frozen=True)
class UPIProfile:
    """Persona-level modifiers applied on top of business-type rhythm."""

    inflow_scale: float = 1.0
    monthly_inflow_growth: float = 0.003
    inflow_noise_std: float = 0.08
    inflow_volatility: float = 0.0
    refund_rate_delta: float = 0.0
    txn_count_scale: float = 1.0
    counterparty_scale: float = 1.0
    seasonality_strength: float = 0.35
    missing_day_probability: float = 0.02  # days with zero UPI (cash-only)


def _weekend_multiplier(dow: int, rhythm: UPIRhythm) -> float:
    """Saturday/Sunday uplift or drop depending on business type."""
    if dow >= 5:
        return rhythm.weekend_factor
    return 1.0


def generate_upi_logs(
    window: GenerationWindow,
    business_type: BusinessType,
    profile: UPIProfile,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate daily UPI transaction logs for the generation window."""
    rhythm = BUSINESS_RHYTHMS[business_type]
    rows: list[dict] = []
    month_index = 0
    prev_month = None

    for day in iter_days(window):
        if prev_month is None or day.month != prev_month:
            if prev_month is not None:
                month_index += 1
            prev_month = day.month

        if rng.random() < profile.missing_day_probability:
            rows.append(
                {
                    "date": day.isoformat(),
                    "inflow_amount": 0.0,
                    "outflow_amount": 0.0,
                    "transaction_count": 0,
                    "unique_counterparty_count": 0,
                    "refund_rate": 0.0,
                }
            )
            continue

        seasonal = seasonality_multiplier(day.month, profile.seasonality_strength)
        weekend = _weekend_multiplier(day.weekday(), rhythm)
        base_inflow = apply_trend(
            rhythm.base_daily_inflow * profile.inflow_scale,
            month_index,
            profile.monthly_inflow_growth,
            profile.inflow_noise_std,
            rng,
        )
        vol = rhythm.inflow_volatility + profile.inflow_volatility
        if vol > 0:
            base_inflow *= float(rng.lognormal(0.0, vol))

        inflow = round(max(base_inflow * seasonal * weekend, 0.0), 2)
        outflow_ratio = clamp(
            rhythm.base_daily_outflow_ratio + float(rng.normal(0, 0.04)),
            0.35,
            0.95,
        )
        outflow = round(inflow * outflow_ratio, 2)

        txn_count = max(
            0,
            int(
                rng.poisson(
                    rhythm.base_txn_count
                    * profile.txn_count_scale
                    * (0.6 + 0.4 * (inflow / max(rhythm.base_daily_inflow, 1)))
                )
            ),
        )
        counterparties = max(
            0,
            int(
                rng.poisson(
                    rhythm.base_unique_counterparties
                    * profile.counterparty_scale
                    * clamp(inflow / max(rhythm.base_daily_inflow, 1), 0.3, 2.5)
                )
            ),
        )
        refund_rate = clamp(
            rhythm.base_refund_rate + profile.refund_rate_delta + float(rng.normal(0, 0.003)),
            0.0,
            0.15,
        )

        rows.append(
            {
                "date": day.isoformat(),
                "inflow_amount": inflow,
                "outflow_amount": outflow,
                "transaction_count": txn_count,
                "unique_counterparty_count": counterparties,
                "refund_rate": round(refund_rate, 4),
            }
        )

    return pd.DataFrame(rows)


def summarize_upi(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate UPI metrics for persona verification."""
    active = df[df["inflow_amount"] > 0]
    return {
        "avg_daily_inflow": float(df["inflow_amount"].mean()),
        "avg_daily_outflow": float(df["outflow_amount"].mean()),
        "avg_transaction_count": float(df["transaction_count"].mean()),
        "avg_unique_counterparties": float(df["unique_counterparty_count"].mean()),
        "avg_refund_rate": float(active["refund_rate"].mean()) if len(active) else 0.0,
        "inflow_cv": float(df["inflow_amount"].std() / max(df["inflow_amount"].mean(), 1)),
        "net_cashflow_ratio": float(
            (df["inflow_amount"].sum() - df["outflow_amount"].sum())
            / max(df["inflow_amount"].sum(), 1)
        ),
    }
