"""Shared data types for the feature engineering layer.

DimensionFeatures and FeatureVector are the internal contracts between
feature engineering and the scoring engine.  All 5 dimension modules
return DimensionFeatures; the pipeline assembles them into a FeatureVector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class DimensionFeatures:
    """Output of one feature dimension's compute() call.

    score:
        0-100 composite for this dimension.  None when required source(s)
        are absent — the composite scorer must handle None, never substitute 0.
    sub_features:
        All intermediate values that fed the score.  Keys are stable strings
        used as ML feature names and SHAP explanation targets.
    missing_reason:
        Human-readable explanation when score is None.
    confidence:
        0-1; reflects source coverage AND history depth.
        A 3-month AA source gives lower confidence than a 12-month one.
    dimension:
        Name string for logging / serialisation.
    """

    dimension: str
    score: float | None
    sub_features: dict[str, float | None]
    missing_reason: str | None = None
    confidence: float = 1.0

    def is_available(self) -> bool:
        """True when a score could be computed."""
        return self.score is not None


@dataclass
class FeatureVector:
    """Complete feature representation for one MSME.

    Assembles the 5 dimension results from the pipeline.  Provides
    to_flat_dict() so the ML scorer can consume one row per MSME.
    """

    msme_id: str
    stability: DimensionFeatures
    cashflow: DimensionFeatures
    compliance: DimensionFeatures
    growth: DimensionFeatures
    repayment: DimensionFeatures
    sources_present: list[str]
    completeness_score: float

    def dimensions(self) -> dict[str, DimensionFeatures]:
        return {
            "stability": self.stability,
            "cashflow": self.cashflow,
            "compliance": self.compliance,
            "growth": self.growth,
            "repayment": self.repayment,
        }

    def available_scores(self) -> dict[str, float]:
        """Return only dimensions that produced a score."""
        return {
            name: dim.score
            for name, dim in self.dimensions().items()
            if dim.score is not None
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten all dimension scores + sub-features into one dict for ML.

        Key convention:  <dimension>_<sub_feature_name>
        Score keys:      <dimension>_score
        Confidence keys: <dimension>_confidence
        """
        out: dict[str, Any] = {
            "msme_id": self.msme_id,
            "completeness_score": self.completeness_score,
            "n_sources": len(self.sources_present),
        }
        for dim_name, dim in self.dimensions().items():
            out[f"{dim_name}_score"] = dim.score
            out[f"{dim_name}_confidence"] = dim.confidence
            out[f"{dim_name}_available"] = dim.is_available()
            for feat_name, feat_val in dim.sub_features.items():
                out[f"{dim_name}_{feat_name}"] = feat_val
        return out
