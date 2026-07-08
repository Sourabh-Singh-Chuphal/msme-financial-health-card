"""Shared utilities for synthetic MSME data generation.

Centralises RNG seeding, Indian retail seasonality, and date helpers so every
generator produces coherent 12-month timelines that align across data sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Iterator

import numpy as np


class BusinessType(str, Enum):
    """MSME sector — drives distinct UPI and cashflow rhythms."""

    RETAIL = "retail"
    SERVICES = "services"
    MANUFACTURING = "manufacturing"


# Month-index seasonality multipliers (Jan=0 … Dec=11) tuned for Indian MSMEs.
# Diwali (Oct–Nov), wedding season (Nov–Feb), Holi (Mar), monsoon dip (Jul–Aug).
DEFAULT_SEASONALITY: tuple[float, ...] = (
    1.05,  # Jan — wedding season tail
    1.08,  # Feb
    1.12,  # Mar — Holi
    1.00,  # Apr
    0.95,  # May
    0.90,  # Jun
    0.85,  # Jul — monsoon
    0.88,  # Aug
    0.95,  # Sep — pre-festive build-up
    1.35,  # Oct — Diwali build-up
    1.55,  # Nov — Diwali peak
    1.15,  # Dec — holiday sales
)


@dataclass(frozen=True)
class GenerationWindow:
    """Inclusive 12-month window ending on the last day of the prior calendar month."""

    start: date
    end: date

    @classmethod
    def last_n_months(cls, n: int = 12, anchor: date | None = None) -> GenerationWindow:
        anchor = anchor or date.today().replace(day=1)
        end = anchor - timedelta(days=1)
        start = (anchor.replace(day=1) - timedelta(days=1)).replace(day=1)
        for _ in range(n - 1):
            start = (start.replace(day=1) - timedelta(days=1)).replace(day=1)
        return cls(start=start, end=end)


def make_rng(seed: int, salt: int = 0) -> np.random.Generator:
    """Deterministic RNG; salt lets each MSME diverge while cohort stays reproducible."""
    return np.random.default_rng(seed + salt)


def month_starts(window: GenerationWindow) -> list[date]:
    """First day of each month in the generation window."""
    months: list[date] = []
    cursor = window.start.replace(day=1)
    while cursor <= window.end:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def iter_days(window: GenerationWindow) -> Iterator[date]:
    """Yield every calendar day in the window."""
    cursor = window.start
    while cursor <= window.end:
        yield cursor
        cursor += timedelta(days=1)


def gst_due_date(tax_period: date) -> date:
    """GSTR-3B due date: 20th of the month following the tax period."""
    if tax_period.month == 12:
        return date(tax_period.year + 1, 1, 20)
    return date(tax_period.year, tax_period.month + 1, 20)


def seasonality_multiplier(month: int, strength: float, profile: tuple[float, ...] | None = None) -> float:
    """Blend flat (1.0) with seasonal profile; strength=0 → no seasonality."""
    base = profile or DEFAULT_SEASONALITY
    seasonal = base[month - 1]
    return 1.0 + strength * (seasonal - 1.0)


def apply_trend(base: float, month_index: int, monthly_growth: float, noise_std: float, rng: np.random.Generator) -> float:
    """Compound monthly growth with optional Gaussian noise."""
    trend = base * ((1 + monthly_growth) ** month_index)
    if noise_std > 0:
        trend *= float(rng.lognormal(0.0, noise_std))
    return max(trend, 0.0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
