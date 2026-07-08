"""SHAP-based explanation wrapper for the MSME credit classifier.

Loads the trained XGBoost model and calculates SHAP values (feature contributions)
for any given MSME's feature vector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.ingestion.validator import validate
from src.features.pipeline import compute as feature_compute
from src.scoring.ml_scorer import load_model

# Global cache for the SHAP explainer
_EXPLAINER_CACHE: shap.TreeExplainer | None = None


@dataclass
class ExplanationResult:
    """SHAP explanation details for a single MSME prediction."""
    msme_id: str
    shap_values: dict[str, float]
    base_value: float
    prediction_margin: float
    predicted_probability: float
    feature_values: dict[str, float | bool | None]


def get_explainer(model_path: Path | str = "config/model.json") -> shap.TreeExplainer:
    """Load the model and return/cache the SHAP TreeExplainer."""
    global _EXPLAINER_CACHE
    if _EXPLAINER_CACHE is not None:
        return _EXPLAINER_CACHE

    model = load_model(model_path)
    # Wrap model with shap TreeExplainer
    explainer = shap.TreeExplainer(model)
    _EXPLAINER_CACHE = explainer
    return explainer


def explain_score(
    msme_id: str,
    data_dir: Path | str = "data/raw",
    model_path: Path | str = "config/model.json"
) -> ExplanationResult:
    """Compute SHAP explanation values for a specific MSME's prediction.

    Loads the MSME's raw files, processes them through the feature pipeline,
    and runs the SHAP explainer to calculate exact log-odds contributions
    for every feature.
    """
    data_dir = Path(data_dir)
    model_path = Path(model_path)

    # 1. Ingest and extract features
    ingestion = validate(msme_id, data_dir=data_dir)
    fv = feature_compute(ingestion)
    flat_feats = fv.to_flat_dict()
    
    # msme_id is not a model feature
    flat_feats.pop("msme_id", None)

    # 2. Re-create the DataFrame exactly as it was aligned during scoring/training
    model = load_model(model_path)
    try:
        feature_names = model.get_booster().feature_names
    except Exception:
        feature_names = None

    if feature_names:
        aligned_dict = {}
        for col in feature_names:
            if col in flat_feats:
                val = flat_feats[col]
                aligned_dict[col] = np.nan if pd.isna(val) else val
            else:
                aligned_dict[col] = np.nan
        df = pd.DataFrame([aligned_dict])
    else:
        df = pd.DataFrame([flat_feats])

    # Convert object columns to float/bool/numeric where possible
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                pass

    # 3. Compute predictions and SHAP values
    explainer = get_explainer(model_path)
    
    # Predict margin and probability
    booster = model.get_booster()
    dmat = xgb.DMatrix(df)
    margin = float(booster.predict(dmat, output_margin=True)[0])
    prob = float(model.predict_proba(df)[0, 1])

    # Get SHAP values for the first row
    shap_vals = explainer.shap_values(df)[0]
    base_val = float(explainer.expected_value)

    # Reconstruct dictionary of shap values per feature
    features_list = df.columns.tolist()
    shap_dict = {feat: float(shap_vals[i]) for i, feat in enumerate(features_list)}

    # Reconstruct dictionary of actual feature values
    feat_val_dict = {}
    for col in df.columns:
        val = df.at[0, col]
        feat_val_dict[col] = None if pd.isna(val) else val

    return ExplanationResult(
        msme_id=msme_id,
        shap_values=shap_dict,
        base_value=base_val,
        prediction_margin=margin,
        predicted_probability=prob,
        feature_values=feat_val_dict
    )
