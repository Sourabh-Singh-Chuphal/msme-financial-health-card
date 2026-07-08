"""Synthetic Account Aggregator bank-statement data for MSME credit assessment.

AA-consented bank data reveals balance stability, cheque/EMI bounce history, and
overdraft reliance — signals banks use when formal financials are absent. This
generator produces daily closing balances and monthly aggregates that correlate
with persona cashflow health without requiring live AA infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from data.synthetic_generators.utils import (
    GenerationWindow,
    clamp,
    iter_days,
    month_starts,
)


@dataclass(frozen=True)
class AAProfile:
    """Persona-level knobs for bank-statement style AA data."""

    base_opening_balance: float = 250_000.0
    monthly_balance_growth: float = 0.004
    daily_net_inflow_mean: float = 8_000.0
    daily_net_inflow_std: float = 15_000.0
    emi_monthly_amount: float = 35_000.0
    emi_day_of_month: int = 5
    bounce_base_rate: float = 0.01
    bounce_trend: float = 0.0
    overdraft_limit: float = 100_000.0
    overdraft_usage_probability: float = 0.05
    balance_volatility: float = 0.08


def _monthly_emi_debit(day: date, profile: AAProfile) -> float:
    if day.day == profile.emi_day_of_month:
        return profile.emi_monthly_amount
    return 0.0


def generate_aa_statements(
    window: GenerationWindow,
    profile: AAProfile,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate daily AA bank-statement records with monthly summary fields."""
    daily_rows: list[dict] = []
    balance = profile.base_opening_balance
    month_index = 0
    prev_month: int | None = None

    monthly_balances: dict[str, list[float]] = {}
    monthly_bounces: dict[str, int] = {}
    monthly_overdraft_days: dict[str, int] = {}

    for day in iter_days(window):
        if prev_month is None or day.month != prev_month:
            if prev_month is not None:
                month_index += 1
            prev_month = day.month

        month_key = day.strftime("%Y-%m")
        monthly_balances.setdefault(month_key, [])
        monthly_bounces.setdefault(month_key, 0)
        monthly_overdraft_days.setdefault(month_key, 0)

        net = float(rng.normal(profile.daily_net_inflow_mean, profile.daily_net_inflow_std))
        if profile.balance_volatility > 0:
            net *= float(rng.lognormal(0.0, profile.balance_volatility))

        balance = max(balance + net - _monthly_emi_debit(day, profile), -profile.overdraft_limit)
        if day.day == 1 and month_index > 0:
            balance *= 1.0 + profile.monthly_balance_growth

        bounce_rate = clamp(profile.bounce_base_rate + profile.bounce_trend * month_index, 0.0, 0.35)
        bounce_count = 1 if rng.random() < bounce_rate else 0
        monthly_bounces[month_key] += bounce_count

        overdraft_used = balance < 0
        if overdraft_used:
            monthly_overdraft_days[month_key] += 1

        daily_rows.append(
            {
                "date": day.isoformat(),
                "closing_balance": round(balance, 2),
                "bounce_count": bounce_count,
                "emi_debit": _monthly_emi_debit(day, profile),
                "overdraft_used": overdraft_used,
            }
        )
        monthly_balances[month_key].append(balance)

    daily_df = pd.DataFrame(daily_rows)

    monthly_summary_rows: list[dict] = []
    for month_key in month_starts(window):
        key = month_key.strftime("%Y-%m")
        balances = monthly_balances.get(key, [])
        monthly_summary_rows.append(
            {
                "month": key,
                "average_monthly_balance": round(float(np.mean(balances)), 2) if balances else 0.0,
                "bounce_return_count": monthly_bounces.get(key, 0),
                "emi_debits_total": profile.emi_monthly_amount,
                "overdraft_usage_days": monthly_overdraft_days.get(key, 0),
            }
        )

    daily_df.attrs["monthly_summary"] = pd.DataFrame(monthly_summary_rows)
    return daily_df


def summarize_aa(daily_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate AA metrics for persona verification."""
    monthly = daily_df.attrs.get("monthly_summary")
    if monthly is None or monthly.empty:
        monthly = pd.DataFrame()

    return {
        "avg_closing_balance": float(daily_df["closing_balance"].mean()),
        "avg_monthly_balance": float(monthly["average_monthly_balance"].mean()) if len(monthly) else 0.0,
        "total_bounces": float(daily_df["bounce_count"].sum()),
        "avg_monthly_bounces": float(monthly["bounce_return_count"].mean()) if len(monthly) else 0.0,
        "overdraft_day_rate": float(daily_df["overdraft_used"].mean()),
        "balance_cv": float(daily_df["closing_balance"].std() / max(abs(daily_df["closing_balance"].mean()), 1)),
        "min_balance": float(daily_df["closing_balance"].min()),
    }
