"""Canonical Pydantic schemas for all 4 MSME alternate data sources.

Everything defined here is the internal contract: ingestors write to it,
feature engineering reads from it. Pydantic v2 enforces correctness at the
boundary so downstream layers never need to re-validate.

Design principles:
- Row-level models (GSTRecord, UPIRecord, …) validate individual records.
  They raise ValidationError on bad data so the validator layer can drop
  and warn without crashing the pipeline.
- Result models (ValidationWarning, IngestionResult) carry the validated
  DataFrames plus metadata (completeness, warnings) downstream.
- IngestionError signals a file-level problem (unparseable file), distinct
  from a row-level ValidationError.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Raised when a data source file exists but cannot be parsed at all.

    Distinct from per-row ValidationError: this means the file itself is
    broken (wrong columns, binary garbage, etc.).  The validator layer
    catches this and records it as a severity='error' warning — it never
    propagates out of the validate() call.
    """

    def __init__(self, source: str, path: Path, reason: str) -> None:
        self.source = source
        self.path = path
        self.reason = reason
        super().__init__(f"[{source}] Cannot parse {path.name}: {reason}")


# ---------------------------------------------------------------------------
# Row-level record schemas
# ---------------------------------------------------------------------------

class GSTRecord(BaseModel):
    """One month of GSTR-1/3B filing data.

    Rejects negative turnover and ITC — these are genuinely malformed values
    (a business cannot report negative revenue in a GST filing).
    filing_delay_days may be negative (filed before due date) — that is valid.
    """

    tax_period: date
    return_type: str
    due_date: date
    filing_date: date
    filing_delay_days: int
    turnover_reported_gstr1: float
    turnover_reported_gstr3b: float
    itc_claimed: float
    mismatch_flag: bool

    @field_validator(
        "turnover_reported_gstr1",
        "turnover_reported_gstr3b",
        "itc_claimed",
    )
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError(f"NaN is not a valid financial amount")
        if v < 0:
            raise ValueError(f"Financial amount must be ≥ 0, got {v:.2f}")
        return v

    @field_validator("filing_delay_days")
    @classmethod
    def must_not_be_nan_delay(cls, v: int) -> int:
        # Pandas Int64 NA becomes pd.NA; guard against it
        if v is None:
            raise ValueError("filing_delay_days must not be null")
        return v


class UPIRecord(BaseModel):
    """One day of UPI transaction activity."""

    date: date
    inflow_amount: float
    outflow_amount: float
    transaction_count: int
    unique_counterparty_count: int
    refund_rate: float

    @field_validator("inflow_amount", "outflow_amount")
    @classmethod
    def must_be_non_negative_amount(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("NaN is not a valid UPI amount")
        if v < 0:
            raise ValueError(f"UPI amount must be ≥ 0, got {v:.2f}")
        return v

    @field_validator("transaction_count", "unique_counterparty_count")
    @classmethod
    def must_be_non_negative_count(cls, v: int) -> int:
        if v is None:
            raise ValueError("Count must not be null")
        if v < 0:
            raise ValueError(f"Count must be ≥ 0, got {v}")
        return v

    @field_validator("refund_rate")
    @classmethod
    def must_be_valid_rate(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("NaN is not a valid refund rate")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"refund_rate must be in [0, 1], got {v:.4f}")
        return v


class AADailyRecord(BaseModel):
    """One day of Account Aggregator bank statement data.

    closing_balance CAN be negative (overdraft usage) — that is a valid
    financial state, not a malformed record.
    """

    date: date
    closing_balance: float
    bounce_count: int
    emi_debit: float
    overdraft_used: bool

    @field_validator("closing_balance")
    @classmethod
    def balance_must_not_be_nan(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("closing_balance must not be NaN")
        return v

    @field_validator("bounce_count")
    @classmethod
    def bounce_must_be_non_negative(cls, v: int) -> int:
        if v is None:
            raise ValueError("bounce_count must not be null")
        if v < 0:
            raise ValueError(f"bounce_count must be ≥ 0, got {v}")
        return v

    @field_validator("emi_debit")
    @classmethod
    def emi_must_be_non_negative(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("emi_debit must not be NaN")
        if v < 0:
            raise ValueError(f"emi_debit must be ≥ 0, got {v:.2f}")
        return v


class AAMonthlyRecord(BaseModel):
    """Monthly summary from Account Aggregator statements."""

    month: str  # "YYYY-MM"
    average_monthly_balance: float
    bounce_return_count: int
    emi_debits_total: float
    overdraft_usage_days: int

    @field_validator("month")
    @classmethod
    def must_be_valid_month_string(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError(f"month must be 'YYYY-MM', got '{v}'")
        return v

    @field_validator("average_monthly_balance")
    @classmethod
    def balance_must_not_be_nan_monthly(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("average_monthly_balance must not be NaN")
        return v

    @field_validator("bounce_return_count", "overdraft_usage_days")
    @classmethod
    def counts_must_be_non_negative(cls, v: int) -> int:
        if v is None:
            raise ValueError("Count must not be null")
        if v < 0:
            raise ValueError(f"Count must be ≥ 0, got {v}")
        return v

    @field_validator("emi_debits_total")
    @classmethod
    def emi_total_must_be_non_negative(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("emi_debits_total must not be NaN")
        if v < 0:
            raise ValueError(f"emi_debits_total must be ≥ 0, got {v:.2f}")
        return v


class EPFORecord(BaseModel):
    """One month of EPFO (Provident Fund) contribution data."""

    month: str  # "YYYY-MM"
    employee_count: int
    wage_bill: float
    pf_contribution_amount: float
    pf_contribution_on_time: bool
    employee_churn_rate: float

    @field_validator("month")
    @classmethod
    def must_be_valid_month_string_epfo(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError(f"month must be 'YYYY-MM', got '{v}'")
        return v

    @field_validator("employee_count")
    @classmethod
    def employee_count_must_be_positive(cls, v: int) -> int:
        if v is None:
            raise ValueError("employee_count must not be null")
        if v < 1:
            raise ValueError(f"employee_count must be ≥ 1, got {v}")
        return v

    @field_validator("wage_bill", "pf_contribution_amount")
    @classmethod
    def financial_must_be_non_negative(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("Financial amount must not be NaN")
        if v < 0:
            raise ValueError(f"Financial amount must be ≥ 0, got {v:.2f}")
        return v

    @field_validator("employee_churn_rate")
    @classmethod
    def churn_must_be_valid_rate(cls, v: float) -> float:
        if math.isnan(v):
            raise ValueError("employee_churn_rate must not be NaN")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"employee_churn_rate must be in [0, 1], got {v:.4f}")
        return v


# ---------------------------------------------------------------------------
# Result schemas (carry validated data downstream)
# ---------------------------------------------------------------------------

class ValidationWarning(BaseModel):
    """A non-fatal data quality issue found during ingestion."""

    source: str
    message: str
    severity: Literal["info", "warning", "error"]
    rows_affected: int = 0


class IngestionResult(BaseModel):
    """Complete result of validating all available data for one MSME.

    This is the contract object passed to the feature engineering layer.
    The layer must use completeness_score and sources_present to decide
    which features it can compute and how to set confidence flags.

    validated_data keys:
        "gst"        → pd.DataFrame (monthly GST records) or None
        "upi"        → pd.DataFrame (daily UPI records) or None
        "aa_daily"   → pd.DataFrame (daily AA statements) or None
        "aa_monthly" → pd.DataFrame (monthly AA summary) or None
        "epfo"       → pd.DataFrame (monthly EPFO records) or None
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    msme_id: str
    sources_present: list[str]   # logical names: subset of ["gst","upi","aa","epfo"]
    sources_absent: list[str]    # complement of sources_present
    completeness_score: float    # 0.0–1.0; see validator._compute_completeness()
    validated_data: dict[str, Any]  # str → pd.DataFrame | None
    validation_warnings: list[ValidationWarning]
    can_score: bool              # False only when sources_present is empty
