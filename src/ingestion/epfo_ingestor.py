"""EPFO contribution records ingestor.

Reads epfo.csv (monthly rows) from an MSME directory.
Missing file → None.  Broken file → IngestionError.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.schemas import IngestionError

SOURCE_NAME = "epfo"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "month",
        "employee_count",
        "wage_bill",
        "pf_contribution_amount",
        "pf_contribution_on_time",
        "employee_churn_rate",
    }
)


def ingest_epfo(msme_dir: Path) -> pd.DataFrame | None:
    """Read and type-coerce monthly EPFO records for one MSME.

    Args:
        msme_dir: Path to the MSME's data folder.

    Returns:
        DataFrame with columns matching EPFORecord, or None if epfo.csv absent.

    Raises:
        IngestionError: if epfo.csv exists but cannot be parsed.
    """
    csv_path = msme_dir / "epfo.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise IngestionError(
            source=SOURCE_NAME, path=csv_path, reason=str(exc)
        ) from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise IngestionError(
            source=SOURCE_NAME,
            path=csv_path,
            reason=f"Missing required columns: {sorted(missing)}",
        )

    df = df[list(REQUIRED_COLUMNS)].copy()

    # month: keep as "YYYY-MM" string
    df["employee_count"] = pd.to_numeric(
        df["employee_count"], errors="coerce"
    ).astype("Int64")
    df["wage_bill"] = pd.to_numeric(df["wage_bill"], errors="coerce")
    df["pf_contribution_amount"] = pd.to_numeric(
        df["pf_contribution_amount"], errors="coerce"
    )
    df["employee_churn_rate"] = pd.to_numeric(
        df["employee_churn_rate"], errors="coerce"
    )
    df["pf_contribution_on_time"] = df["pf_contribution_on_time"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )

    return df
