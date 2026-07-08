"""Rule-based scoring engine for MSMEs.

Computes a weighted average of the 5 dimension scores (stability, cashflow,
compliance, growth, repayment) using weights loaded from config/scoring_weights.yaml.
Handles missing dimensions by dynamically redistributing their weights
proportionally among the available dimensions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import yaml

from src.features.base import FeatureVector

# Fallback default weights if config file is not readable
DEFAULT_WEIGHTS = {
    "stability": 0.15,
    "cashflow": 0.25,
    "compliance": 0.15,
    "growth": 0.15,
    "repayment": 0.30,
}

@dataclass
class RuleBasedScore:
    """Result of rule-based credit scoring."""
    overall_score: float
    dimension_scores: dict[str, float | None]
    weights_used: dict[str, float]
    confidence: float


def load_weights(config_path: Path | str = "config/scoring_weights.yaml") -> dict[str, float]:
    """Load scoring weights from a YAML config file.

    Falls back to DEFAULT_WEIGHTS on any error.
    """
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_WEIGHTS.copy()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            weights = yaml.safe_load(f)
        if not isinstance(weights, dict):
            return DEFAULT_WEIGHTS.copy()
        # Verify keys and numeric values
        validated_weights = {}
        for dim in DEFAULT_WEIGHTS:
            val = weights.get(dim)
            if val is not None and isinstance(val, (int, float)) and val >= 0:
                validated_weights[dim] = float(val)
            else:
                validated_weights[dim] = DEFAULT_WEIGHTS[dim]
        # Normalize weights so they sum to 1.0
        total = sum(validated_weights.values())
        if total > 0:
            for k in validated_weights:
                validated_weights[k] /= total
        else:
            return DEFAULT_WEIGHTS.copy()
        return validated_weights
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def score_features(fv: FeatureVector, config_path: Path | str = "config/scoring_weights.yaml") -> RuleBasedScore:
    """Compute the rule-based credit score for a FeatureVector.

    If any dimension score is None, its weight is distributed proportionally
    to the active dimensions. If all dimensions are None, overall_score is 0.0.
    """
    weights = load_weights(config_path)
    dims = fv.dimensions()  # returns dict[str, DimensionFeatures]

    active_dims = {}
    active_weights_raw = {}
    active_confidences = {}

    for name, dim in dims.items():
        if dim.score is not None:
            active_dims[name] = dim.score
            active_weights_raw[name] = weights.get(name, DEFAULT_WEIGHTS.get(name, 0.2))
            active_confidences[name] = dim.confidence

    if not active_dims:
        # No data at all to score
        return RuleBasedScore(
            overall_score=0.0,
            dimension_scores={name: dim.score for name, dim in dims.items()},
            weights_used={name: 0.0 for name in dims},
            confidence=0.0
        )

    # Redistribute weights proportionally
    total_raw_weight = sum(active_weights_raw.values())
    weights_used = {}
    
    for name in dims:
        if name in active_dims:
            weights_used[name] = active_weights_raw[name] / total_raw_weight if total_raw_weight > 0 else 1.0 / len(active_dims)
        else:
            weights_used[name] = 0.0

    # Compute overall score as weighted average
    overall_score = sum(active_dims[name] * weights_used[name] for name in active_dims)

    # Aggregate confidence as the weighted average of active dimension confidences
    overall_confidence = sum(active_confidences[name] * weights_used[name] for name in active_dims)

    return RuleBasedScore(
        overall_score=round(overall_score, 2),
        dimension_scores={name: dim.score for name, dim in dims.items()},
        weights_used=weights_used,
        confidence=round(overall_confidence, 4)
    )
