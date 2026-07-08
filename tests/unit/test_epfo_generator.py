"""Unit tests for EPFO synthetic record generation."""

from __future__ import annotations

import numpy as np

from data.synthetic_generators.epfo_generator import EPFOProfile, generate_epfo_records, summarize_epfo


def test_generate_epfo_twelve_months(window, rng) -> None:
    profile = EPFOProfile()
    df = generate_epfo_records(window, profile, rng)
    assert len(df) == 12
    assert df["employee_count"].min() >= 1


def test_declining_headcount_trend(window) -> None:
    growing = EPFOProfile(base_employee_count=20, monthly_headcount_growth=0.01)
    shrinking = EPFOProfile(base_employee_count=20, monthly_headcount_growth=-0.012)
    rng = np.random.default_rng(11)
    df_up = generate_epfo_records(window, growing, rng)
    df_down = generate_epfo_records(window, shrinking, rng)
    assert summarize_epfo(df_down)["headcount_trend"] < summarize_epfo(df_up)["headcount_trend"]
