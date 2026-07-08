"""Feature pipeline — orchestrates all 5 dimension modules into a FeatureVector.

Usage:
    from src.features.pipeline import compute
    from src.ingestion.validator import validate

    ingestion = validate("MSME_000001", data_dir=Path("data/raw"))
    feature_vector = compute(ingestion)

The pipeline is fault-tolerant: if any individual dimension's compute()
raises an unexpected exception it is caught and converted to a DimensionFeatures
with score=None and an error message — the other 4 dimensions still run.
This ensures one buggy sub-feature never silences the entire assessment.
"""

from __future__ import annotations

import traceback

from src.features import (
    cashflow_features,
    compliance_features,
    growth_features,
    repayment_features,
    stability_features,
)
from src.features.base import DimensionFeatures, FeatureVector
from src.ingestion.schemas import IngestionResult


def compute(result: IngestionResult) -> FeatureVector:
    """Run all 5 feature dimensions for one MSME and return a FeatureVector.

    Args:
        result: Validated ingestion result from src.ingestion.validator.validate().

    Returns:
        FeatureVector with all 5 DimensionFeatures objects populated.
        Never raises — all errors are contained in the returned object.
    """
    stability = _safe_compute("stability", stability_features.compute, result)
    cashflow = _safe_compute("cashflow", cashflow_features.compute, result)
    compliance = _safe_compute("compliance", compliance_features.compute, result)
    growth = _safe_compute("growth", growth_features.compute, result)
    repayment = _safe_compute("repayment", repayment_features.compute, result)

    return FeatureVector(
        msme_id=result.msme_id,
        stability=stability,
        cashflow=cashflow,
        compliance=compliance,
        growth=growth,
        repayment=repayment,
        sources_present=result.sources_present,
        completeness_score=result.completeness_score,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _safe_compute(
    dimension: str,
    fn: object,
    result: IngestionResult,
) -> DimensionFeatures:
    """Call a dimension's compute() function; return a None-score DimensionFeatures on error."""
    try:
        return fn(result)
    except Exception as exc:  # noqa: BLE001
        return DimensionFeatures(
            dimension=dimension,
            score=None,
            sub_features={},
            missing_reason=(
                f"Unexpected error during {dimension} feature computation: "
                f"{type(exc).__name__}: {exc}"
            ),
            confidence=0.0,
        )
