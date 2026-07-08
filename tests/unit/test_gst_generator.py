"""Unit tests for GST synthetic record generation."""

from __future__ import annotations

import numpy as np

from data.synthetic_generators.gst_generator import GSTProfile, generate_gst_records, summarize_gst


def test_generate_gst_records_has_twelve_months(window, rng) -> None:
    profile = GSTProfile(base_monthly_turnover=500_000)
    df = generate_gst_records(window, profile, rng)
    assert len(df) == 12
    assert set(df.columns) >= {
        "filing_date",
        "due_date",
        "turnover_reported_gstr1",
        "itc_claimed",
        "mismatch_flag",
    }


def test_late_filing_persona_has_higher_delay(window) -> None:
    on_time = GSTProfile(
        base_monthly_turnover=800_000,
        late_filing_probability=0.0,
        filing_delay_mean_days=0,
    )
    late = GSTProfile(
        base_monthly_turnover=800_000,
        late_filing_probability=0.95,
        filing_delay_mean_days=10,
    )
    df_on = generate_gst_records(window, on_time, np.random.default_rng(1))
    df_late = generate_gst_records(window, late, np.random.default_rng(1))
    assert summarize_gst(df_late)["avg_filing_delay_days"] > summarize_gst(df_on)["avg_filing_delay_days"]


def test_seasonal_profile_has_higher_turnover_cv(window) -> None:
    flat = GSTProfile(base_monthly_turnover=600_000, seasonality_strength=0.0)
    seasonal = GSTProfile(base_monthly_turnover=600_000, seasonality_strength=0.85)
    df_flat = generate_gst_records(window, flat, np.random.default_rng(7))
    df_season = generate_gst_records(window, seasonal, np.random.default_rng(7))
    assert summarize_gst(df_season)["turnover_cv"] > summarize_gst(df_flat)["turnover_cv"]
