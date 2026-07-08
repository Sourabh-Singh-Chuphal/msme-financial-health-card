"""Unit tests for UPI synthetic log generation."""

from __future__ import annotations

import numpy as np

from data.synthetic_generators.upi_generator import UPIProfile, generate_upi_logs, summarize_upi
from data.synthetic_generators.utils import BusinessType


def test_generate_upi_covers_full_window(window, rng) -> None:
    profile = UPIProfile()
    df = generate_upi_logs(window, BusinessType.RETAIL, profile, rng)
    assert len(df) == (window.end - window.start).days + 1
    assert df["inflow_amount"].ge(0).all()


def test_retail_has_more_transactions_than_manufacturing(window) -> None:
    profile = UPIProfile(inflow_volatility=0.0, missing_day_probability=0.0)
    rng = np.random.default_rng(99)
    retail = generate_upi_logs(window, BusinessType.RETAIL, profile, rng)
    mfg = generate_upi_logs(window, BusinessType.MANUFACTURING, profile, rng)
    assert summarize_upi(retail)["avg_transaction_count"] > summarize_upi(mfg)["avg_transaction_count"]


def test_volatile_profile_has_higher_inflow_cv(window) -> None:
    stable = UPIProfile(inflow_volatility=0.0, missing_day_probability=0.0)
    volatile = UPIProfile(inflow_volatility=0.40, missing_day_probability=0.0)
    rng = np.random.default_rng(5)
    df_stable = generate_upi_logs(window, BusinessType.SERVICES, stable, rng)
    df_vol = generate_upi_logs(window, BusinessType.SERVICES, volatile, rng)
    assert summarize_upi(df_vol)["inflow_cv"] > summarize_upi(df_stable)["inflow_cv"]
