"""Public API for the ingestion layer.

Import from here in all downstream modules — never import individual
ingestor files directly.  This keeps the internal implementation flexible.
"""

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
from src.ingestion.validator import validate

__all__ = [
    # Entry point
    "validate",
    # Result types
    "IngestionResult",
    "ValidationWarning",
    "IngestionError",
    # Record schemas (useful for feature engineering layer)
    "GSTRecord",
    "UPIRecord",
    "AADailyRecord",
    "AAMonthlyRecord",
    "EPFORecord",
]
