"""GST filing data ingestor.

Reads gst.csv from an MSME directory and returns a typed DataFrame ready
for row-level Pydantic validation.  Returns None if the file is absent
(source simply wasn't available for this MSME — normal for NTC personas).
Raises IngestionError only when the file exists but is structurally broken.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.schemas import IngestionError

SOURCE_NAME = "gst"

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "tax_period",
        "return_type",
        "due_date",
        "filing_date",
        "filing_delay_days",
        "turnover_reported_gstr1",
        "turnover_reported_gstr3b",
        "itc_claimed",
        "mismatch_flag",
    }
)


def ingest_gst(msme_dir: Path) -> pd.DataFrame | None:
    """Read and type-coerce GST data for one MSME.

    Args:
        msme_dir: Path to the MSME's data folder (e.g. data/raw/MSME_000001).

    Returns:
        DataFrame with columns matching GSTRecord, or None if gst.csv absent.

    Raises:
        IngestionError: if gst.csv exists but cannot be parsed or has wrong schema.
    """
    csv_path = msme_dir / "gst.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, dtype=str)  # read everything as str first
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

    # Keep date columns as ISO strings — Pydantic parses "YYYY-MM-DD" natively.
    # Convert numeric columns with errors="coerce" so bad values become NaN
    # (caught by field validators rather than crashing here).
    df["filing_delay_days"] = pd.to_numeric(df["filing_delay_days"], errors="coerce")
    df["turnover_reported_gstr1"] = pd.to_numeric(
        df["turnover_reported_gstr1"], errors="coerce"
    )
    df["turnover_reported_gstr3b"] = pd.to_numeric(
        df["turnover_reported_gstr3b"], errors="coerce"
    )
    df["itc_claimed"] = pd.to_numeric(df["itc_claimed"], errors="coerce")

    # Normalize bool: "True"/"False"/"1"/"0" → Python bool
    df["mismatch_flag"] = df["mismatch_flag"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )

    return df
