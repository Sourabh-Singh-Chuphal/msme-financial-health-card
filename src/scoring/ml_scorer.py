"""ML-based credit scorer.

Loads the trained XGBoost model from config/model.json, prepares the input
FeatureVector into a flat pandas DataFrame aligned with the model's training
features, and predicts the Probability of Default (PD). The final ML score
is calculated as (1.0 - PD) * 100.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.base import FeatureVector

# Global cache for the loaded model
_MODEL_CACHE: XGBClassifier | None = None


@dataclass
class MLScore:
    """Result of ML-based credit scoring."""
    overall_score: float | None
    confidence: float
    probability_of_default: float | None
    missing_reason: str | None = None


def load_model(model_path: Path | str = "config/model.json") -> XGBClassifier:
    """Load and cache the trained XGBoost model.

    Raises FileNotFoundError if model has not been trained yet.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please run train_model.py first."
        )

    model = XGBClassifier()
    model.load_model(str(path))
    _MODEL_CACHE = model
    return model


def score_features(fv: FeatureVector, model_path: Path | str = "config/model.json") -> MLScore:
    """Score a FeatureVector using the trained XGBoost model.

    Returns the score scaled 0-100 (high = low risk).
    Handles missing values natively via XGBoost's NaN support.
    """
    try:
        model = load_model(model_path)
    except FileNotFoundError as exc:
        return MLScore(
            overall_score=None,
            confidence=0.0,
            probability_of_default=None,
            missing_reason=f"Model not loaded: {exc}"
        )

    # 1. Flatten the feature vector
    flat_feats = fv.to_flat_dict()
    # Drop non-feature key
    if "msme_id" in flat_feats:
        flat_feats.pop("msme_id")

    # 2. Convert to DataFrame and align with training features
    df = pd.DataFrame([flat_feats])
    
    # Retrieve feature names from XGBoost booster
    try:
        feature_names = model.get_booster().feature_names
    except Exception:
        feature_names = None

    if feature_names:
        # Reorder and align columns. Missing columns become NaN, which XGBoost handles natively
        aligned_df = pd.DataFrame(index=[0])
        for col in feature_names:
            if col in df.columns:
                val = df.at[0, col]
                # Map None/NaN to np.nan
                aligned_df.at[0, col] = np.nan if pd.isna(val) else val
            else:
                aligned_df.at[0, col] = np.nan
        X = aligned_df
    else:
        # Fallback if feature names are not available
        X = df.copy()

    # Convert object columns to float/bool/numeric where possible
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col], errors="raise")
            except Exception:
                pass

    # 3. Predict probability of default (PD)
    try:
        pd_prob = float(model.predict_proba(X)[0, 1])
        # Score is high when probability of default is low
        overall_score = (1.0 - pd_prob) * 100.0
        
        # ML confidence is based on the completeness score of the input data
        confidence = fv.completeness_score

        return MLScore(
            overall_score=round(overall_score, 2),
            confidence=round(confidence, 4),
            probability_of_default=round(pd_prob, 4)
        )
    except Exception as exc:
        return MLScore(
            overall_score=None,
            confidence=0.0,
            probability_of_default=None,
            missing_reason=f"Prediction error: {exc}"
        )
