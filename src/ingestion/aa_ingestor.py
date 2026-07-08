"""Account Aggregator (AA) bank statement ingestor.

Reads BOTH aa_daily.csv and aa_monthly.csv for one MSME.  Either file can
be absent independently.  The logical source 'aa' is present if at least
one of the two files exists and is non-empty after validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.ingestion.schemas import IngestionError

SOURCE_NAME = "aa"

REQUIRED_DAILY_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "closing_balance",
        "bounce_count",
        "emi_debit",
        "overdraft_used",
    }
)

REQUIRED_MONTHLY_COLUMNS: frozenset[str] = frozenset(
    {
        "month",
        "average_monthly_balance",
        "bounce_return_count",
        "emi_debits_total",
        "overdraft_usage_days",
    }
)


class AARawData(NamedTuple):
    """Container for both AA file reads before validation."""

    daily: pd.DataFrame | None
    monthly: pd.DataFrame | None


def ingest_aa(msme_dir: Path) -> AARawData:
    """Read and type-coerce AA daily and monthly files for one MSME.

    Returns:
        AARawData namedtuple with .daily and .monthly fields.
        Each may be None if the corresponding file is absent.

    Raises:
        IngestionError: if a file exists but cannot be parsed / has wrong schema.
    """
    return AARawData(
        daily=_read_daily(msme_dir),
        monthly=_read_monthly(msme_dir),
    )


def _read_daily(msme_dir: Path) -> pd.DataFrame | None:
    csv_path = msme_dir / "aa_daily.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise IngestionError(
            source=f"{SOURCE_NAME}:daily", path=csv_path, reason=str(exc)
        ) from exc

    missing = REQUIRED_DAILY_COLUMNS - set(df.columns)
    if missing:
        raise IngestionError(
            source=f"{SOURCE_NAME}:daily",
            path=csv_path,
            reason=f"Missing required columns: {sorted(missing)}",
        )

    df = df[list(REQUIRED_DAILY_COLUMNS)].copy()

    # date: keep as ISO string
    df["closing_balance"] = pd.to_numeric(df["closing_balance"], errors="coerce")
    df["bounce_count"] = pd.to_numeric(df["bounce_count"], errors="coerce").astype(
        "Int64"
    )
    df["emi_debit"] = pd.to_numeric(df["emi_debit"], errors="coerce")
    df["overdraft_used"] = df["overdraft_used"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )

    return df


def _read_monthly(msme_dir: Path) -> pd.DataFrame | None:
    csv_path = msme_dir / "aa_monthly.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise IngestionError(
            source=f"{SOURCE_NAME}:monthly", path=csv_path, reason=str(exc)
        ) from exc

    missing = REQUIRED_MONTHLY_COLUMNS - set(df.columns)
    if missing:
        raise IngestionError(
            source=f"{SOURCE_NAME}:monthly",
            path=csv_path,
            reason=f"Missing required columns: {sorted(missing)}",
        )

    df = df[list(REQUIRED_MONTHLY_COLUMNS)].copy()

    # month: keep as "YYYY-MM" string; Pydantic validates the format
    df["average_monthly_balance"] = pd.to_numeric(
        df["average_monthly_balance"], errors="coerce"
    )
    df["bounce_return_count"] = pd.to_numeric(
        df["bounce_return_count"], errors="coerce"
    ).astype("Int64")
    df["emi_debits_total"] = pd.to_numeric(df["emi_debits_total"], errors="coerce")
    df["overdraft_usage_days"] = pd.to_numeric(
        df["overdraft_usage_days"], errors="coerce"
    ).astype("Int64")

    return df
