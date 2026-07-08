"""Composite scorer that blends rule-based and ML credit scores.

Applies a dynamic weighting strategy based on the data completeness_score:
- High completeness (>= 0.75) -> 70% ML / 30% Rule-Based.
- Medium completeness (0.5 to 0.75) -> 50% ML / 50% Rule-Based.
- Low completeness (< 0.5) -> 20% ML / 80% Rule-Based (protects against ML instability on sparse data).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.features.base import FeatureVector
from src.scoring.rule_based_scorer import score_features as rule_score_features, RuleBasedScore
from src.scoring.ml_scorer import score_features as ml_score_features, MLScore


@dataclass
class FinalScore:
    """Consolidated financial health assessment."""
    overall_score: float
    dimension_scores: dict[str, float | None]
    confidence_band: Literal["High", "Medium", "Low"]
    confidence_score: float
    rule_score: RuleBasedScore
    ml_score: MLScore
    blend_ratio_used: dict[str, float]


def get_confidence_band(confidence: float) -> Literal["High", "Medium", "Low"]:
    """Categorize confidence score into High, Medium, or Low band."""
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.50:
        return "Medium"
    else:
        return "Low"


def compute_score(
    fv: FeatureVector,
    weights_path: Path | str = "config/scoring_weights.yaml",
    model_path: Path | str = "config/model.json"
) -> FinalScore:
    """Compute the final credit health score by blending rule-based and ML scoring.

    Dynamically adjusts the blend ratio based on data completeness to leverage
    ML patterns when data is rich, while falling back to explainable rules
    when data is thin.
    """
    # 1. Compute individual scores
    rule_score = rule_score_features(fv, weights_path)
    ml_score = ml_score_features(fv, model_path)

    # 2. Determine blend weights based on completeness_score
    # Let w_rule be the weight of the rule-based score
    c = fv.completeness_score
    
    if ml_score.overall_score is None:
        # Fallback entirely to rule-based if ML score is unavailable
        w_rule = 1.0
    elif c >= 0.75:
        w_rule = 0.30
    elif c >= 0.50:
        w_rule = 0.50
    else:
        # Thin data -> trust rule-based more for explainability
        w_rule = 0.80

    w_ml = 1.0 - w_rule

    # 3. Blend the scores
    r_val = rule_score.overall_score
    m_val = ml_score.overall_score if ml_score.overall_score is not None else r_val

    blended_score = w_rule * r_val + w_ml * m_val

    # 4. Blend confidence
    # Rule confidence represents present dimension depth. ML confidence is data completeness.
    blended_conf = w_rule * rule_score.confidence + w_ml * ml_score.confidence

    return FinalScore(
        overall_score=round(blended_score, 2),
        dimension_scores=rule_score.dimension_scores,
        confidence_band=get_confidence_band(blended_conf),
        confidence_score=round(blended_conf, 4),
        rule_score=rule_score,
        ml_score=ml_score,
        blend_ratio_used={
            "rule_based": round(w_rule, 2),
            "ml": round(w_ml, 2)
        }
    )
