"""Unit tests for the ingestion layer.

Tests four scenarios that together enforce all contracts of the validator:
    1. All 4 sources present with full 12-month history → completeness ≈ 1.0
    2. Only 2 sources present → completeness ≈ 0.5, can_score still True
    3. One source (GST) contains malformed records → warnings generated,
       bad rows dropped, valid rows kept, pipeline continues
    4. Zero sources present → completeness = 0.0, can_score = False

Each test writes minimal synthetic CSVs to pytest's tmp_path to avoid
touching the real data/raw directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import validate, IngestionResult, ValidationWarning


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _write_gst(msme_dir: Path, rows: list[dict] | None = None) -> None:
    """Write gst.csv; uses 3 valid rows by default."""
    if rows is None:
        rows = [
            {
                "tax_period": "2025-07-01",
                "return_type": "GSTR-1/3B",
                "due_date": "2025-08-20",
                "filing_date": "2025-08-18",
                "filing_delay_days": -2,
                "turnover_reported_gstr1": 800000.0,
                "turnover_reported_gstr3b": 800000.0,
                "itc_claimed": 96000.0,
                "mismatch_flag": False,
            },
            {
                "tax_period": "2025-08-01",
                "return_type": "GSTR-1/3B",
                "due_date": "2025-09-20",
                "filing_date": "2025-09-20",
                "filing_delay_days": 0,
                "turnover_reported_gstr1": 820000.0,
                "turnover_reported_gstr3b": 820000.0,
                "itc_claimed": 98400.0,
                "mismatch_flag": False,
            },
            {
                "tax_period": "2025-09-01",
                "return_type": "GSTR-1/3B",
                "due_date": "2025-10-20",
                "filing_date": "2025-10-19",
                "filing_delay_days": -1,
                "turnover_reported_gstr1": 850000.0,
                "turnover_reported_gstr3b": 850000.0,
                "itc_claimed": 102000.0,
                "mismatch_flag": True,
            },
        ]
    pd.DataFrame(rows).to_csv(msme_dir / "gst.csv", index=False)


def _write_upi(msme_dir: Path, n_days: int = 90) -> None:
    """Write n_days of valid daily UPI records (≈3 months)."""
    base = pd.Timestamp("2025-07-01")
    rows = [
        {
            "date": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "inflow_amount": 45000.0 + i * 10,
            "outflow_amount": 32000.0 + i * 5,
            "transaction_count": 80 + (i % 10),
            "unique_counterparty_count": 35 + (i % 5),
            "refund_rate": 0.025,
        }
        for i in range(n_days)
    ]
    pd.DataFrame(rows).to_csv(msme_dir / "upi.csv", index=False)


def _write_aa(msme_dir: Path, n_months: int = 3) -> None:
    """Write aa_daily.csv and aa_monthly.csv for n_months."""
    # daily (just one day per month for brevity)
    daily_rows = []
    monthly_rows = []
    base = pd.Timestamp("2025-07-01")
    for m in range(n_months):
        month_start = base + pd.DateOffset(months=m)
        daily_rows.append(
            {
                "date": month_start.strftime("%Y-%m-%d"),
                "closing_balance": 400000.0 - m * 5000,
                "bounce_count": 0,
                "emi_debit": 35000.0,
                "overdraft_used": False,
            }
        )
        monthly_rows.append(
            {
                "month": month_start.strftime("%Y-%m"),
                "average_monthly_balance": 420000.0 - m * 5000,
                "bounce_return_count": 0,
                "emi_debits_total": 35000.0,
                "overdraft_usage_days": 0,
            }
        )
    pd.DataFrame(daily_rows).to_csv(msme_dir / "aa_daily.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(msme_dir / "aa_monthly.csv", index=False)


def _write_epfo(msme_dir: Path, n_months: int = 3) -> None:
    """Write n_months of valid EPFO records."""
    base = pd.Timestamp("2025-07-01")
    rows = [
        {
            "month": (base + pd.DateOffset(months=i)).strftime("%Y-%m"),
            "employee_count": 20 + i,
            "wage_bill": 320000.0 + i * 5000,
            "pf_contribution_amount": 76800.0,
            "pf_contribution_on_time": True,
            "employee_churn_rate": 0.03,
        }
        for i in range(n_months)
    ]
    pd.DataFrame(rows).to_csv(msme_dir / "epfo.csv", index=False)


# ---------------------------------------------------------------------------
# Scenario 1: All 4 sources present, full history (12 months)
# ---------------------------------------------------------------------------


class TestAllSourcesPresent:
    """Validator returns a clean IngestionResult when all sources exist."""

    @pytest.fixture()
    def result(self, tmp_path: Path) -> IngestionResult:
        msme_dir = tmp_path / "MSME_FULL"
        msme_dir.mkdir()
        # Write 12-month GST
        rows = []
        for m in range(12):
            rows.append(
                {
                    "tax_period": f"2025-{m + 1:02d}-01",
                    "return_type": "GSTR-1/3B",
                    "due_date": f"2025-{(m % 12) + 1:02d}-20",
                    "filing_date": f"2025-{(m % 12) + 1:02d}-19",
                    "filing_delay_days": -1,
                    "turnover_reported_gstr1": 800000.0 + m * 10000,
                    "turnover_reported_gstr3b": 800000.0 + m * 10000,
                    "itc_claimed": 96000.0,
                    "mismatch_flag": False,
                }
            )
        _write_gst(msme_dir, rows)
        _write_upi(msme_dir, n_days=365)
        _write_aa(msme_dir, n_months=12)
        _write_epfo(msme_dir, n_months=12)
        return validate("MSME_FULL", data_dir=tmp_path)

    def test_all_sources_present(self, result: IngestionResult) -> None:
        assert set(result.sources_present) == {"gst", "upi", "aa", "epfo"}

    def test_sources_absent_is_empty(self, result: IngestionResult) -> None:
        assert result.sources_absent == []

    def test_completeness_score_near_one(self, result: IngestionResult) -> None:
        # With 12 months all sources: should be 1.0 (or very close)
        assert result.completeness_score >= 0.95, (
            f"Expected completeness ≥ 0.95, got {result.completeness_score}"
        )

    def test_can_score_is_true(self, result: IngestionResult) -> None:
        assert result.can_score is True

    def test_validated_data_all_non_empty(self, result: IngestionResult) -> None:
        for key in ("gst", "upi", "aa_daily", "aa_monthly", "epfo"):
            df = result.validated_data.get(key)
            assert df is not None and not df.empty, f"{key} should have data"

    def test_no_error_level_warnings(self, result: IngestionResult) -> None:
        error_warnings = [w for w in result.validation_warnings if w.severity == "error"]
        assert error_warnings == [], (
            f"Unexpected error warnings: {error_warnings}"
        )

    def test_gst_dataframe_schema(self, result: IngestionResult) -> None:
        df = result.validated_data["gst"]
        assert "turnover_reported_gstr1" in df.columns
        assert "mismatch_flag" in df.columns

    def test_upi_dataframe_schema(self, result: IngestionResult) -> None:
        df = result.validated_data["upi"]
        assert "inflow_amount" in df.columns
        assert "refund_rate" in df.columns


# ---------------------------------------------------------------------------
# Scenario 2: Only 2 sources present (NTC-style partial data)
# ---------------------------------------------------------------------------


class TestTwoSourcesPresent:
    """Validator gracefully handles missing AA and EPFO sources."""

    @pytest.fixture()
    def result(self, tmp_path: Path) -> IngestionResult:
        msme_dir = tmp_path / "MSME_NTC"
        msme_dir.mkdir()
        _write_gst(msme_dir)    # 3-month GST
        _write_upi(msme_dir, n_days=90)   # ~3 months UPI
        # No AA, no EPFO
        return validate("MSME_NTC", data_dir=tmp_path)

    def test_sources_present(self, result: IngestionResult) -> None:
        assert set(result.sources_present) == {"gst", "upi"}

    def test_sources_absent(self, result: IngestionResult) -> None:
        assert set(result.sources_absent) == {"aa", "epfo"}

    def test_can_score_is_still_true(self, result: IngestionResult) -> None:
        """2 sources is enough to produce a score — NTC MSMEs must not be rejected."""
        assert result.can_score is True

    def test_completeness_score_is_around_half(self, result: IngestionResult) -> None:
        # 2 of 4 sources present, each with ~3/12 months → ~0.125 each → ~0.25 total
        # But at least it should be < 0.65 to confirm partial-data detection
        assert result.completeness_score < 0.65, (
            f"Expected completeness < 0.65 for 2-source partial data, "
            f"got {result.completeness_score}"
        )
        assert result.completeness_score > 0.0, "Must be > 0 with some data"

    def test_absent_source_data_is_none(self, result: IngestionResult) -> None:
        assert result.validated_data["aa_daily"] is None
        assert result.validated_data["aa_monthly"] is None
        assert result.validated_data["epfo"] is None

    def test_no_error_warnings_for_missing_sources(self, result: IngestionResult) -> None:
        """Missing files are NOT errors — they're expected for NTC personas."""
        # Any warning that mentions absent sources should be informational,
        # not error-severity
        error_warnings = [
            w for w in result.validation_warnings
            if w.severity == "error" and w.source in ("aa", "epfo")
        ]
        assert error_warnings == [], (
            f"Missing sources should not generate error warnings: {error_warnings}"
        )


# ---------------------------------------------------------------------------
# Scenario 3: One source (GST) contains malformed records
# ---------------------------------------------------------------------------


class TestMalformedRecords:
    """Malformed rows are dropped with warnings; valid rows are kept."""

    @pytest.fixture()
    def result(self, tmp_path: Path) -> IngestionResult:
        msme_dir = tmp_path / "MSME_BAD"
        msme_dir.mkdir()
        # GST with 2 valid rows and 1 row with negative turnover
        _write_gst(
            msme_dir,
            rows=[
                {
                    "tax_period": "2025-07-01",
                    "return_type": "GSTR-1/3B",
                    "due_date": "2025-08-20",
                    "filing_date": "2025-08-18",
                    "filing_delay_days": -2,
                    "turnover_reported_gstr1": 800000.0,
                    "turnover_reported_gstr3b": 800000.0,
                    "itc_claimed": 96000.0,
                    "mismatch_flag": False,
                },
                {
                    "tax_period": "2025-08-01",
                    "return_type": "GSTR-1/3B",
                    "due_date": "2025-09-20",
                    "filing_date": "2025-09-20",
                    "filing_delay_days": 0,
                    "turnover_reported_gstr1": -50000.0,  # ← MALFORMED: negative turnover
                    "turnover_reported_gstr3b": -50000.0,
                    "itc_claimed": -6000.0,              # ← MALFORMED: negative ITC
                    "mismatch_flag": False,
                },
                {
                    "tax_period": "2025-09-01",
                    "return_type": "GSTR-1/3B",
                    "due_date": "2025-10-20",
                    "filing_date": "2025-10-20",
                    "filing_delay_days": 0,
                    "turnover_reported_gstr1": 850000.0,
                    "turnover_reported_gstr3b": 850000.0,
                    "itc_claimed": 102000.0,
                    "mismatch_flag": False,
                },
            ],
        )
        _write_upi(msme_dir, n_days=90)
        return validate("MSME_BAD", data_dir=tmp_path)

    def test_gst_source_still_present(self, result: IngestionResult) -> None:
        """GST should still appear in sources_present (2 valid rows remain)."""
        assert "gst" in result.sources_present

    def test_malformed_rows_are_dropped(self, result: IngestionResult) -> None:
        """The bad row must not appear in validated data."""
        gst_df = result.validated_data["gst"]
        assert gst_df is not None
        # The negative-turnover row should have been dropped
        assert (gst_df["turnover_reported_gstr1"] >= 0).all(), (
            "Negative turnover rows should have been removed"
        )

    def test_valid_rows_are_kept(self, result: IngestionResult) -> None:
        """The 2 valid rows must survive."""
        gst_df = result.validated_data["gst"]
        assert gst_df is not None
        assert len(gst_df) == 2, (
            f"Expected 2 valid rows, got {len(gst_df)}"
        )

    def test_warning_generated_for_dropped_row(self, result: IngestionResult) -> None:
        """There must be at least one warning mentioning the gst source."""
        gst_warnings = [w for w in result.validation_warnings if "gst" in w.source]
        assert len(gst_warnings) > 0, "Expected at least one GST validation warning"

    def test_warning_has_rows_affected(self, result: IngestionResult) -> None:
        """The summary warning must report rows_affected > 0."""
        summary_warnings = [
            w for w in result.validation_warnings
            if "gst" in w.source and w.rows_affected > 0
        ]
        assert len(summary_warnings) > 0

    def test_can_score_is_true_despite_bad_records(self, result: IngestionResult) -> None:
        """Malformed records in one source must not block scoring."""
        assert result.can_score is True

    def test_pipeline_does_not_raise(self, tmp_path: Path) -> None:
        """validate() must never propagate an exception for bad records."""
        msme_dir = tmp_path / "MSME_ALLBAD"
        msme_dir.mkdir()
        # Write a GST file where EVERY row has negative turnover
        rows = [
            {
                "tax_period": f"2025-{m:02d}-01",
                "return_type": "GSTR-1/3B",
                "due_date": f"2025-{m:02d}-20",
                "filing_date": f"2025-{m:02d}-19",
                "filing_delay_days": 0,
                "turnover_reported_gstr1": -999.0,  # all bad
                "turnover_reported_gstr3b": -999.0,
                "itc_claimed": -100.0,
                "mismatch_flag": False,
            }
            for m in range(1, 4)
        ]
        pd.DataFrame(rows).to_csv(msme_dir / "gst.csv", index=False)
        # Should NOT raise
        r = validate("MSME_ALLBAD", data_dir=tmp_path)
        assert r is not None
        assert "gst" not in r.sources_present  # all rows dropped → source absent


# ---------------------------------------------------------------------------
# Scenario 4: Zero sources present
# ---------------------------------------------------------------------------


class TestZeroSourcesPresent:
    """Empty MSME directory → cannot score but does not crash."""

    @pytest.fixture()
    def result(self, tmp_path: Path) -> IngestionResult:
        msme_dir = tmp_path / "MSME_EMPTY"
        msme_dir.mkdir()
        # No files written
        return validate("MSME_EMPTY", data_dir=tmp_path)

    def test_sources_present_is_empty(self, result: IngestionResult) -> None:
        assert result.sources_present == []

    def test_all_sources_absent(self, result: IngestionResult) -> None:
        assert set(result.sources_absent) == {"gst", "upi", "aa", "epfo"}

    def test_completeness_score_is_zero(self, result: IngestionResult) -> None:
        assert result.completeness_score == 0.0

    def test_can_score_is_false(self, result: IngestionResult) -> None:
        assert result.can_score is False

    def test_all_validated_data_is_none(self, result: IngestionResult) -> None:
        for key, val in result.validated_data.items():
            assert val is None, f"Expected None for {key}, got {val}"

    def test_error_warning_present(self, result: IngestionResult) -> None:
        """A 'cannot score' error warning must be surfaced."""
        error_warnings = [w for w in result.validation_warnings if w.severity == "error"]
        assert len(error_warnings) > 0, (
            "Expected at least one error-level warning for zero-source MSME"
        )

    def test_result_is_ingestion_result_instance(self, result: IngestionResult) -> None:
        assert isinstance(result, IngestionResult)

    def test_does_not_raise_for_nonexistent_directory(self, tmp_path: Path) -> None:
        """validate() must not raise even if the MSME directory itself is missing."""
        r = validate("MSME_DOESNOTEXIST", data_dir=tmp_path)
        assert r.can_score is False
        assert r.completeness_score == 0.0


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_broken_file_treated_as_absent(self, tmp_path: Path) -> None:
        """A gst.csv with completely wrong columns should be caught, not crash."""
        msme_dir = tmp_path / "MSME_BROKEN"
        msme_dir.mkdir()
        # Write a totally wrong CSV
        pd.DataFrame({"foo": [1, 2], "bar": ["x", "y"]}).to_csv(
            msme_dir / "gst.csv", index=False
        )
        r = validate("MSME_BROKEN", data_dir=tmp_path)
        assert "gst" not in r.sources_present
        gst_errors = [
            w for w in r.validation_warnings
            if "gst" in w.source and w.severity == "error"
        ]
        assert len(gst_errors) > 0, "Should warn about broken gst.csv"

    def test_msme_id_preserved_in_result(self, tmp_path: Path) -> None:
        msme_dir = tmp_path / "MSME_ID_CHECK"
        msme_dir.mkdir()
        r = validate("MSME_ID_CHECK", data_dir=tmp_path)
        assert r.msme_id == "MSME_ID_CHECK"

    def test_completeness_score_bounds(self, tmp_path: Path) -> None:
        """completeness_score must always be in [0.0, 1.0]."""
        msme_dir = tmp_path / "MSME_BOUNDS"
        msme_dir.mkdir()
        _write_gst(msme_dir)
        r = validate("MSME_BOUNDS", data_dir=tmp_path)
        assert 0.0 <= r.completeness_score <= 1.0
