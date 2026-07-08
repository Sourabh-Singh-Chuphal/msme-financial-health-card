"""Unit tests for AA synthetic bank-statement generation."""

from __future__ import annotations

import numpy as np

from data.synthetic_generators.aa_generator import AAProfile, generate_aa_statements, summarize_aa


def test_generate_aa_daily_and_monthly(window, rng) -> None:
    profile = AAProfile()
    daily = generate_aa_statements(window, profile, rng)
    monthly = daily.attrs["monthly_summary"]
    assert len(daily) == (window.end - window.start).days + 1
    assert len(monthly) == 12
    assert "average_monthly_balance" in monthly.columns


def test_rising_bounce_profile(window) -> None:
    healthy = AAProfile(bounce_base_rate=0.005, bounce_trend=0.0)
    risky = AAProfile(bounce_base_rate=0.08, bounce_trend=0.02)
    rng = np.random.default_rng(3)
    df_ok = generate_aa_statements(window, healthy, rng)
    df_bad = generate_aa_statements(window, risky, rng)
    assert summarize_aa(df_bad)["total_bounces"] > summarize_aa(df_ok)["total_bounces"]
