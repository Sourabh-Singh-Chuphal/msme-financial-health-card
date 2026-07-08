"""Repayment Capacity features from Account Aggregator bank statements.

Repayment answers: "How likely is this MSME to service a new loan?"
We look at existing financial obligations and how well they are managed:
  - EMI burden: what fraction of average cash inflow already goes to loan repayments?
  - Overdraft frequency: is the business routinely in overdraft? (short-term liquidity stress)
  - Bounce rate: how often are debits returned (cheques / NACH mandates bounced)?

AA is the ONLY source for these signals.  Unlike stability/cashflow which can
partially score from a single source, repayment cannot be meaningfully assessed
without bank statement data — fabricating a 50 here would be dishonest and
misleading to underwriters.  If AA is absent, score=None.

Data source consumed:
    AA daily  → overdraft usage, bounce count
    AA monthly → EMI debits, bounce return count, average balance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import DimensionFeatures, _clamp
from src.ingestion.schemas import IngestionResult

DIMENSION = "repayment"
_MAX_MONTHS = 12


def compute(result: IngestionResult) -> DimensionFeatures:
    """Compute repayment capacity features for one MSME."""
    aa_daily = result.validated_data.get("aa_daily")
    aa_monthly = result.validated_data.get("aa_monthly")

    has_daily = aa_daily is not None and not aa_daily.empty
    has_monthly = aa_monthly is not None and not aa_monthly.empty

    if not has_daily and not has_monthly:
        return DimensionFeatures(
            dimension=DIMENSION,
            score=None,
            sub_features={},
            missing_reason=(
                "AA bank statement data is required for repayment capacity "
                "assessment and is not available for this MSME."
            ),
            confidence=0.0,
        )

    sub: dict[str, float | None] = {}

    # ----------------------------------------------------------------- overdraft
    od_score, sub = _overdraft_score(aa_daily, sub)

    # ------------------------------------------------------------------- bounce
    bounce_score, sub = _bounce_score(aa_daily, aa_monthly, sub)

    # ---------------------------------------------------------------- EMI burden
    emi_score, sub = _emi_score(aa_monthly, sub)

    # Composite: bounce is the strongest underwriter signal (cheque returns
    # trigger bank internal alerts); overdraft frequency follows; EMI last.
    raw = 0.40 * bounce_score + 0.35 * od_score + 0.25 * emi_score
    score = round(_clamp(raw), 2)

    confidence = _confidence(aa_daily, aa_monthly, has_daily, has_monthly)

    return DimensionFeatures(
        dimension=DIMENSION,
        score=score,
        sub_features=sub,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Sub-score computations
# ---------------------------------------------------------------------------

def _overdraft_score(
    aa_daily: pd.DataFrame | None, sub: dict
) -> tuple[float, dict]:
    if aa_daily is None or aa_daily.empty:
        sub["overdraft_day_rate"] = None
        sub["consecutive_overdraft_max"] = None
        return 75.0, sub  # partial penalty — no data but not zero

    od_col = aa_daily.get("overdraft_used")
    if od_col is None:
        sub["overdraft_day_rate"] = None
        sub["consecutive_overdraft_max"] = None
        return 75.0, sub

    od_flags = od_col.astype(bool).values
    od_rate = float(od_flags.mean())
    sub["overdraft_day_rate"] = round(od_rate, 4)

    # Longest consecutive overdraft run (signals chronic liquidity stress)
    max_run = _max_consecutive(od_flags)
    sub["consecutive_overdraft_max"] = int(max_run)

    # 0% overdraft → 100; 33% → 0; penalise long runs extra
    base = _clamp(100.0 - od_rate * 300.0)
    run_penalty = _clamp(max_run / 30.0 * 20.0)  # up to 20 pts for 30-day run
    score = _clamp(base - run_penalty)
    return score, sub


def _bounce_score(
    aa_daily: pd.DataFrame | None,
    aa_monthly: pd.DataFrame | None,
    sub: dict,
) -> tuple[float, dict]:
    # Prefer monthly summary (more reliable count)
    if aa_monthly is not None and not aa_monthly.empty and "bounce_return_count" in aa_monthly.columns:
        bounces = pd.to_numeric(
            aa_monthly["bounce_return_count"], errors="coerce"
        ).fillna(0)
        avg_monthly_bounces = float(bounces.mean())
        total_months = len(aa_monthly)
        sub["avg_monthly_bounces"] = round(avg_monthly_bounces, 2)
        sub["total_bounce_months"] = int(total_months)
        # 0 bounces → 100; 6 bounces/month → 10; 7+ → near 0
        score = _clamp(100.0 - avg_monthly_bounces * 15.0)
        return score, sub

    # Fallback to daily bounce counts
    if aa_daily is not None and not aa_daily.empty and "bounce_count" in aa_daily.columns:
        bounces = pd.to_numeric(
            aa_daily["bounce_count"], errors="coerce"
        ).fillna(0)
        avg_daily_bounces = float(bounces.mean())
        # Convert to monthly equivalent (multiply by ~30)
        avg_monthly_equiv = avg_daily_bounces * 30
        sub["avg_monthly_bounces"] = round(avg_monthly_equiv, 2)
        sub["total_bounce_months"] = None
        score = _clamp(100.0 - avg_monthly_equiv * 15.0)
        return score, sub

    sub["avg_monthly_bounces"] = None
    sub["total_bounce_months"] = None
    return 70.0, sub  # neutral-ish — no data available


def _emi_score(
    aa_monthly: pd.DataFrame | None, sub: dict
) -> tuple[float, dict]:
    if aa_monthly is None or aa_monthly.empty:
        sub["emi_burden_ratio"] = None
        sub["avg_monthly_emi"] = None
        return 70.0, sub  # no EMI data — assume moderate

    avg_balance = pd.to_numeric(
        aa_monthly["average_monthly_balance"], errors="coerce"
    ).mean()
    emi_total = pd.to_numeric(
        aa_monthly["emi_debits_total"], errors="coerce"
    ).mean()

    avg_emi = float(emi_total) if not np.isnan(emi_total) else 0.0
    avg_bal = float(avg_balance) if not np.isnan(avg_balance) else 1.0

    sub["avg_monthly_emi"] = round(avg_emi, 2)

    # EMI burden: EMI as fraction of average monthly balance (proxy for income)
    # A business with avg balance 500k and EMI 50k has 10% burden
    if abs(avg_bal) > 0:
        emi_ratio = avg_emi / abs(avg_bal)
    else:
        emi_ratio = 0.0
    emi_ratio = max(0.0, emi_ratio)
    sub["emi_burden_ratio"] = round(emi_ratio, 4)

    # 0% burden → 100; 50% → 0; >50% → 0
    score = _clamp(100.0 - emi_ratio * 200.0)
    return score, sub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_consecutive(flags: np.ndarray) -> int:
    """Count longest consecutive True run in a boolean array."""
    max_run = current = 0
    for f in flags:
        if f:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _confidence(
    aa_daily: pd.DataFrame | None,
    aa_monthly: pd.DataFrame | None,
    has_daily: bool,
    has_monthly: bool,
) -> float:
    n_monthly = len(aa_monthly) if aa_monthly is not None else 0
    n_daily_months = (
        int(pd.to_datetime(aa_daily["date"]).dt.to_period("M").nunique())
        if aa_daily is not None and not aa_daily.empty
        else 0
    )
    history = max(n_monthly, n_daily_months) / _MAX_MONTHS
    source_frac = (int(has_daily) + int(has_monthly)) / 2
    return round(source_frac * min(history, 1.0), 4)
