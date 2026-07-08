"""Synthetic EPFO (employee provident fund) records for MSME workforce signals.

EPFO data proxies business scale, wage-paying capacity, and employment stability.
Regular PF contributions and low churn indicate a viable enterprise; shrinking
headcount or irregular deposits often precede credit stress for labour-intensive MSMEs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.synthetic_generators.utils import (
    GenerationWindow,
    apply_trend,
    clamp,
    month_starts,
    seasonality_multiplier,
)


@dataclass(frozen=True)
class EPFOProfile:
    """Persona-level knobs for monthly EPFO filings."""

    base_employee_count: int = 18
    monthly_headcount_growth: float = 0.002
    base_avg_wage: float = 16_000.0
    wage_inflation_monthly: float = 0.004
    pf_regularity_rate: float = 0.98
    base_churn_rate: float = 0.03
    churn_trend: float = 0.0
    headcount_noise_std: float = 0.03
    seasonality_strength: float = 0.15


def _pf_contributed(wage_bill: float, on_time: bool) -> float:
    """Employer + employee PF is roughly 24% of wages when compliant."""
    if not on_time:
        return round(wage_bill * 0.24 * 0.5, 2)
    return round(wage_bill * 0.24, 2)


def generate_epfo_records(
    window: GenerationWindow,
    profile: EPFOProfile,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly EPFO contribution and headcount records."""
    rows: list[dict] = []

    for idx, month in enumerate(month_starts(window)):
        seasonal = seasonality_multiplier(month.month, profile.seasonality_strength)
        headcount_float = apply_trend(
            float(profile.base_employee_count),
            idx,
            profile.monthly_headcount_growth,
            profile.headcount_noise_std,
            rng,
        )
        headcount = max(int(round(headcount_float * seasonal)), 1)

        avg_wage = apply_trend(
            profile.base_avg_wage,
            idx,
            profile.wage_inflation_monthly,
            0.02,
            rng,
        )
        wage_bill = round(headcount * avg_wage, 2)

        churn = clamp(
            profile.base_churn_rate + profile.churn_trend * idx + float(rng.normal(0, 0.008)),
            0.0,
            0.45,
        )
        on_time = rng.random() < profile.pf_regularity_rate
        pf_amount = _pf_contributed(wage_bill, on_time)

        rows.append(
            {
                "month": month.strftime("%Y-%m"),
                "employee_count": headcount,
                "wage_bill": wage_bill,
                "pf_contribution_amount": pf_amount,
                "pf_contribution_on_time": on_time,
                "employee_churn_rate": round(churn, 4),
            }
        )

    return pd.DataFrame(rows)


def summarize_epfo(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate EPFO metrics for persona verification."""
    return {
        "avg_employee_count": float(df["employee_count"].mean()),
        "avg_wage_bill": float(df["wage_bill"].mean()),
        "pf_regularity_rate": float(df["pf_contribution_on_time"].mean()),
        "avg_churn_rate": float(df["employee_churn_rate"].mean()),
        "headcount_trend": float(
            np.polyfit(range(len(df)), df["employee_count"].values, 1)[0]
        )
        if len(df) > 1
        else 0.0,
    }
