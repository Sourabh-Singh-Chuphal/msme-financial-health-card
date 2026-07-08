"""UPI transaction log ingestor.

Reads upi.csv (daily rows) from an MSME directory.
Missing file → None.  Structurally broken file → IngestionError.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.schemas import IngestionError

SOURCE_NAME = "upi"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "inflow_amount",
        "outflow_amount",
        "transaction_count",
        "unique_counterparty_count",
        "refund_rate",
    }
)


def ingest_upi(msme_dir: Path) -> pd.DataFrame | None:
    """Read and type-coerce daily UPI transaction logs for one MSME.

    Args:
        msme_dir: Path to the MSME's data folder.

    Returns:
        DataFrame with columns matching UPIRecord, or None if upi.csv absent.

    Raises:
        IngestionError: if upi.csv exists but cannot be parsed.
    """
    csv_path = msme_dir / "upi.csv"
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

    # date kept as ISO string; Pydantic parses "YYYY-MM-DD"
    df["inflow_amount"] = pd.to_numeric(df["inflow_amount"], errors="coerce")
    df["outflow_amount"] = pd.to_numeric(df["outflow_amount"], errors="coerce")
    df["transaction_count"] = pd.to_numeric(
        df["transaction_count"], errors="coerce"
    ).astype("Int64")
    df["unique_counterparty_count"] = pd.to_numeric(
        df["unique_counterparty_count"], errors="coerce"
    ).astype("Int64")
    df["refund_rate"] = pd.to_numeric(df["refund_rate"], errors="coerce")

    return df
