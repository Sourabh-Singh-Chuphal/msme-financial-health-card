"""Synthetic GST (GSTR-1 / GSTR-3B) filing records for MSME credit assessment.

GST filings are the primary compliance and turnover signal for Indian MSMEs.
This generator produces monthly filing timelines with realistic seasonality
(festival-driven retail spikes), late-filing behaviour, and GSTR-1 vs GSTR-3B
mismatch flags that credit analysts watch for fraud or cash-flow stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.synthetic_generators.utils import (
    GenerationWindow,
    apply_trend,
    clamp,
    gst_due_date,
    month_starts,
    seasonality_multiplier,
)


@dataclass(frozen=True)
class GSTProfile:
    """Persona-level knobs that shape GST filing behaviour."""

    base_monthly_turnover: float
    monthly_turnover_growth: float = 0.005
    turnover_noise_std: float = 0.05
    itc_ratio_mean: float = 0.12
    itc_ratio_std: float = 0.02
    filing_delay_mean_days: float = 0.0
    filing_delay_std_days: float = 1.0
    late_filing_probability: float = 0.05
    mismatch_probability: float = 0.02
    seasonality_strength: float = 0.3
    turnover_volatility: float = 0.0  # extra month-to-month swing for volatile personas


def _sample_filing_date(
    due: date,
    rng: np.random.Generator,
    profile: GSTProfile,
) -> date:
    """Sample filing date with persona-specific delay distribution."""
    if rng.random() < profile.late_filing_probability:
        delay = int(
            max(
                1,
                rng.normal(
                    loc=max(profile.filing_delay_mean_days, 3.0),
                    scale=max(profile.filing_delay_std_days, 2.0),
                ),
            )
        )
    else:
        delay = int(
            max(
                0,
                rng.normal(loc=profile.filing_delay_mean_days, scale=profile.filing_delay_std_days),
            )
        )
    return due + timedelta(days=delay)


def _monthly_turnover(
    month_index: int,
    tax_period: date,
    profile: GSTProfile,
    rng: np.random.Generator,
) -> float:
    seasonal = seasonality_multiplier(tax_period.month, profile.seasonality_strength)
    base = apply_trend(
        profile.base_monthly_turnover,
        month_index,
        profile.monthly_turnover_growth,
        profile.turnover_noise_std,
        rng,
    )
    turnover = base * seasonal
    if profile.turnover_volatility > 0:
        turnover *= float(rng.lognormal(0.0, profile.turnover_volatility))
    return round(max(turnover, 10_000.0), 2)


def generate_gst_records(
    window: GenerationWindow,
    profile: GSTProfile,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly GSTR-1/3B filing records for a 12-month window."""
    rows: list[dict] = []
    months = month_starts(window)

    for idx, tax_period in enumerate(months):
        due = gst_due_date(tax_period)
        filing = _sample_filing_date(due, rng, profile)
        turnover_gstr1 = _monthly_turnover(idx, tax_period, profile, rng)

        itc_ratio = clamp(
            float(rng.normal(profile.itc_ratio_mean, profile.itc_ratio_std)),
            0.02,
            0.35,
        )
        itc_gstr3b = round(turnover_gstr1 * itc_ratio, 2)

        mismatch = False
        turnover_gstr3b = turnover_gstr1
        if rng.random() < profile.mismatch_probability:
            mismatch = True
            direction = rng.choice([-1, 1])
            delta_pct = float(rng.uniform(0.03, 0.12))
            turnover_gstr3b = round(turnover_gstr1 * (1 + direction * delta_pct), 2)
            itc_gstr3b = round(turnover_gstr3b * itc_ratio * float(rng.uniform(0.85, 1.15)), 2)

        rows.append(
            {
                "tax_period": tax_period.isoformat(),
                "return_type": "GSTR-1/3B",
                "due_date": due.isoformat(),
                "filing_date": filing.isoformat(),
                "filing_delay_days": (filing - due).days,
                "turnover_reported_gstr1": turnover_gstr1,
                "turnover_reported_gstr3b": turnover_gstr3b,
                "itc_claimed": itc_gstr3b,
                "mismatch_flag": mismatch,
            }
        )

    return pd.DataFrame(rows)


def summarize_gst(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate GST metrics for persona verification."""
    return {
        "avg_filing_delay_days": float(df["filing_delay_days"].mean()),
        "late_filing_rate": float((df["filing_delay_days"] > 0).mean()),
        "mismatch_rate": float(df["mismatch_flag"].mean()),
        "avg_monthly_turnover": float(df["turnover_reported_gstr1"].mean()),
        "turnover_trend": float(
            np.polyfit(range(len(df)), df["turnover_reported_gstr1"].values, 1)[0]
        )
        if len(df) > 1
        else 0.0,
        "turnover_cv": float(df["turnover_reported_gstr1"].std() / max(df["turnover_reported_gstr1"].mean(), 1)),
    }
