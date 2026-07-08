"""MSME data validator — orchestrates all ingestors into one IngestionResult.

This module is the ingestion boundary: everything that enters the feature
engineering layer must pass through validate().  The validator NEVER raises
an exception to its caller — all errors are captured in validation_warnings
and the pipeline continues on whatever data is available.

Completeness scoring formula:
    Each of the 4 logical sources contributes up to 0.25.
    Within each source: score = 0.25 × min(months_of_history / 12, 1.0)
    Total completeness_score ∈ [0.0, 1.0].

    "Months of history" is measured as:
        GST / EPFO   → number of valid rows (each row = 1 month)
        UPI          → number of distinct calendar months in the date column
        AA           → number of rows in aa_monthly; fallback to aa_daily months
    This penalises partial-history sources (3 months of UPI = 0.0625 vs 12 months = 0.25).

can_score:
    True if at least one source is present with at least one valid row.
    False only when zero sources are available — caller should surface this
    as "Insufficient data to score" to the end user.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.ingestion.aa_ingestor import AARawData, ingest_aa
from src.ingestion.epfo_ingestor import ingest_epfo
from src.ingestion.gst_ingestor import ingest_gst
from src.ingestion.schemas import (
    AADailyRecord,
    AAMonthlyRecord,
    EPFORecord,
    GSTRecord,
    IngestionError,
    IngestionResult,
    UPIRecord,
    ValidationWarning,
)
from src.ingestion.upi_ingestor import ingest_upi

_ALL_SOURCES = ("gst", "upi", "aa", "epfo")
_MAX_MONTHS = 12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(msme_id: str, data_dir: Path | str = Path("data/raw")) -> IngestionResult:
    """Validate all available data sources for a single MSME.

    Args:
        msme_id: The MSME identifier (e.g. "MSME_000001").
        data_dir: Root directory that contains per-MSME subdirectories.

    Returns:
        IngestionResult with validated DataFrames, completeness_score, and
        any warnings produced during validation.  NEVER raises.
    """
    data_dir = Path(data_dir)
    msme_dir = data_dir / msme_id
    warnings: list[ValidationWarning] = []
    validated: dict[str, Any] = {
        "gst": None,
        "upi": None,
        "aa_daily": None,
        "aa_monthly": None,
        "epfo": None,
    }

    # ------------------------------------------------------------------ GST
    gst_raw = _safe_ingest("gst", lambda: ingest_gst(msme_dir), warnings)
    if gst_raw is not None:
        validated["gst"], w = _validate_rows(
            gst_raw, GSTRecord, source="gst", id_col="tax_period"
        )
        warnings.extend(w)

    # ------------------------------------------------------------------ UPI
    upi_raw = _safe_ingest("upi", lambda: ingest_upi(msme_dir), warnings)
    if upi_raw is not None:
        validated["upi"], w = _validate_rows(
            upi_raw, UPIRecord, source="upi", id_col="date"
        )
        warnings.extend(w)

    # ------------------------------------------------------------------- AA
    aa_raw: AARawData | None = _safe_ingest(
        "aa", lambda: ingest_aa(msme_dir), warnings
    )
    if aa_raw is not None:
        if aa_raw.daily is not None:
            validated["aa_daily"], w = _validate_rows(
                aa_raw.daily, AADailyRecord, source="aa_daily", id_col="date"
            )
            warnings.extend(w)
        if aa_raw.monthly is not None:
            validated["aa_monthly"], w = _validate_rows(
                aa_raw.monthly,
                AAMonthlyRecord,
                source="aa_monthly",
                id_col="month",
            )
            warnings.extend(w)

    # ----------------------------------------------------------------- EPFO
    epfo_raw = _safe_ingest("epfo", lambda: ingest_epfo(msme_dir), warnings)
    if epfo_raw is not None:
        validated["epfo"], w = _validate_rows(
            epfo_raw, EPFORecord, source="epfo", id_col="month"
        )
        warnings.extend(w)

    # ----------------------------- Determine which sources are usable
    sources_present = _determine_sources_present(validated)
    sources_absent = [s for s in _ALL_SOURCES if s not in sources_present]

    completeness = _compute_completeness(validated, sources_present)
    can_score = len(sources_present) > 0

    if not can_score:
        warnings.append(
            ValidationWarning(
                source="all",
                message=(
                    f"No valid data found for {msme_id}. "
                    "Cannot produce a financial health score."
                ),
                severity="error",
                rows_affected=0,
            )
        )

    return IngestionResult(
        msme_id=msme_id,
        sources_present=sources_present,
        sources_absent=sources_absent,
        completeness_score=completeness,
        validated_data=validated,
        validation_warnings=warnings,
        can_score=can_score,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_ingest(
    source: str,
    ingest_fn: Any,
    warnings: list[ValidationWarning],
) -> Any:
    """Call an ingestor, catch IngestionError and unexpected exceptions.

    Returns the ingestor result on success, None on any failure.  Failures
    are recorded as warnings so the pipeline continues.
    """
    try:
        return ingest_fn()
    except IngestionError as exc:
        warnings.append(
            ValidationWarning(
                source=source,
                message=f"File-level parse failure — treating source as absent. "
                        f"Reason: {exc.reason}",
                severity="error",
                rows_affected=0,
            )
        )
        return None
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            ValidationWarning(
                source=source,
                message=f"Unexpected ingestion error: {exc!r}",
                severity="error",
                rows_affected=0,
            )
        )
        return None


def _validate_rows(
    df: pd.DataFrame,
    model: type,
    source: str,
    id_col: str,
) -> tuple[pd.DataFrame, list[ValidationWarning]]:
    """Validate each row against a Pydantic model.

    Valid rows are kept; invalid rows are dropped.
    Returns (cleaned_df, list_of_warnings).
    """
    warnings: list[ValidationWarning] = []
    valid_indices: list[int] = []
    dropped = 0

    for idx, row in df.iterrows():
        row_dict = _row_to_dict(row)
        try:
            model(**row_dict)
            valid_indices.append(idx)
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            dropped += 1
            identifier = row_dict.get(id_col, f"row {idx}")
            msg = _extract_error_message(exc)
            warnings.append(
                ValidationWarning(
                    source=source,
                    message=f"Dropped record [{identifier}]: {msg}",
                    severity="warning",
                    rows_affected=1,
                )
            )

    clean_df = df.loc[valid_indices].reset_index(drop=True) if valid_indices else pd.DataFrame(columns=df.columns)

    if dropped > 0:
        warnings.insert(
            0,
            ValidationWarning(
                source=source,
                message=f"{dropped} of {len(df)} records failed validation and were dropped.",
                severity="warning" if dropped < len(df) else "error",
                rows_affected=dropped,
            ),
        )

    return clean_df, warnings


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a pandas row to a plain dict, handling pandas NA types."""
    result: dict[str, Any] = {}
    for key, val in row.items():
        # pandas NA / NaT → None so Pydantic sees a missing required field
        if pd.isna(val) if not isinstance(val, (bool, str)) else False:
            result[key] = None
        else:
            result[key] = val
    return result


def _extract_error_message(exc: Exception) -> str:
    """Pull the first meaningful error message from a Pydantic or generic exception."""
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            loc = " → ".join(str(l) for l in errors[0].get("loc", []))
            msg = errors[0].get("msg", str(exc))
            return f"{loc}: {msg}" if loc else msg
    return str(exc)


def _determine_sources_present(validated: dict[str, Any]) -> list[str]:
    """Return list of logical source names that have at least one valid row."""
    present: list[str] = []

    def _has_rows(key: str) -> bool:
        df = validated.get(key)
        return df is not None and not df.empty

    if _has_rows("gst"):
        present.append("gst")
    if _has_rows("upi"):
        present.append("upi")
    # AA is a single logical source backed by two files
    if _has_rows("aa_daily") or _has_rows("aa_monthly"):
        present.append("aa")
    if _has_rows("epfo"):
        present.append("epfo")

    return present


def _compute_completeness(
    validated: dict[str, Any], sources_present: list[str]
) -> float:
    """Compute data_completeness_score ∈ [0.0, 1.0].

    Each of the 4 sources contributes up to 0.25.  Within each source the
    contribution scales with months of history: fewer months = less confidence.
    """
    total = 0.0

    if "gst" in sources_present:
        df = validated.get("gst")
        months = len(df) if df is not None else 0
        total += 0.25 * min(months / _MAX_MONTHS, 1.0)

    if "upi" in sources_present:
        df = validated.get("upi")
        if df is not None and not df.empty and "date" in df.columns:
            try:
                n_months = pd.to_datetime(df["date"]).dt.to_period("M").nunique()
            except Exception:
                n_months = 0
        else:
            n_months = 0
        total += 0.25 * min(n_months / _MAX_MONTHS, 1.0)

    if "aa" in sources_present:
        monthly_df = validated.get("aa_monthly")
        daily_df = validated.get("aa_daily")
        if monthly_df is not None and not monthly_df.empty:
            n_months = len(monthly_df)
        elif daily_df is not None and not daily_df.empty and "date" in daily_df.columns:
            try:
                n_months = pd.to_datetime(daily_df["date"]).dt.to_period("M").nunique()
            except Exception:
                n_months = 0
        else:
            n_months = 0
        total += 0.25 * min(n_months / _MAX_MONTHS, 1.0)

    if "epfo" in sources_present:
        df = validated.get("epfo")
        months = len(df) if df is not None else 0
        total += 0.25 * min(months / _MAX_MONTHS, 1.0)

    return round(total, 4)
