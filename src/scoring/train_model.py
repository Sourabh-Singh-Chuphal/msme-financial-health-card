"""Model training script for MSME credit scoring.

Orchestrates features generation across all 200 synthetic MSMEs, simulates a
realistic and non-leaky default label for each based on underlying features
plus noise, trains an XGBoost classifier, and saves it to config/model.json.
Prints AUC-ROC, precision, recall, and confusion matrix.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.validator import validate
from src.features.pipeline import compute as feature_compute


def generate_labels_and_features(
    manifest_path: Path, data_dir: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate FeatureVectors and simulate default labels for all MSMEs.

    To avoid trivial leakage, default outcomes are simulated probabilistically
    based on a combination of dimension scores and persona-based baseline risk,
    with added logistic noise.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    msme_ids = manifest["msme_ids"]

    feature_dicts = []
    labels = []

    # Seed for reproducibility
    rng = np.random.default_rng(101)

    print(f"Generating features and labels for {len(msme_ids)} MSMEs...")

    for msme_id in msme_ids:
        # Load metadata to get persona
        meta_path = data_dir / msme_id / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        persona = meta.get("persona_type", "unknown")

        # Ingest and extract features
        ingestion = validate(msme_id, data_dir=data_dir)
        fv = feature_compute(ingestion)
        flat_feats = fv.to_flat_dict()

        # Compute probability of default (PD) using a combination of features and persona
        # 1. Base log-odds (logit) of default based on persona
        persona_logit = {
            "healthy_established": -3.5,  # ~3% default rate
            "healthy_ntc":         -2.8,  # ~6% default rate
            "seasonal_business":   -2.2,  # ~10% default rate
            "risky_volatile":      -0.6,  # ~35% default rate
            "risky_declining":      0.2,  # ~55% default rate
        }.get(persona, -2.0)

        # 2. Adjust log-odds based on computed feature scores (if available)
        # Low scores increase risk, high scores decrease risk
        score_adjust = 0.0
        dims = ["stability", "cashflow", "compliance", "growth", "repayment"]
        valid_dims_count = 0
        for dim in dims:
            score = flat_feats.get(f"{dim}_score")
            if score is not None:
                # Center around 65 (medium health)
                # Lower score -> positive risk adjustment, higher score -> negative risk adjustment
                score_adjust += 0.04 * (65.0 - score)
                valid_dims_count += 1
        
        # Add random logistic noise to make it probabilistic
        noise = rng.logistic(0, 0.8)
        
        total_logit = persona_logit + score_adjust + noise
        pd_prob = 1.0 / (1.0 + np.exp(-total_logit))

        # Determine default outcome
        default_label = 1 if rng.uniform(0, 1) < pd_prob else 0

        # Store results
        feature_dicts.append(flat_feats)
        labels.append(default_label)

    features_df = pd.DataFrame(feature_dicts)
    labels_series = pd.Series(labels, name="default")

    return features_df, labels_series


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    manifest_path = data_dir / "manifest.json"
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    model_path = config_dir / "model.json"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}. Please run synthetic cohort generation first.")
        sys.exit(1)

    # 1. Generate data
    df_raw, y = generate_labels_and_features(manifest_path, data_dir)
    
    # 2. Save labels to metadata files for downstream reference and transparency
    # (Optional but useful for consistent scoring / verification later)
    # Let's align features
    df = df_raw.copy()
    
    # Columns to drop (non-feature columns)
    drop_cols = ["msme_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert object columns to float/bool/numeric where possible (XGBoost handles nan)
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col], errors="raise")
            except Exception:
                pass

    # 3. Train/test split (80/20)
    # Use stratify on y to keep default rates balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nData split summary:")
    print(f"  Training set size: {X_train.shape[0]} (defaults: {y_train.sum()})")
    print(f"  Testing set size : {X_test.shape[0]} (defaults: {y_test.sum()})")

    # 4. Train XGBoost classifier
    # We use hyperparameters suited for a small dataset with missing values
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    [[TN: {cm[0,0]:2d}, FP: {cm[0,1]:2d}]")
    print(f"     [FN: {cm[1,0]:2d}, VP/TP: {cm[1,1]:2d}]]")

    # Leakage check: AUC close to 1.0 is suspicious
    if auc > 0.98:
        print("\n[WARNING] Suspiciously high AUC (> 0.98). Check for leakage of persona variables or deterministic labels.")
    elif auc >= 0.75:
        print("\n[PASS] Model performance is in a realistic, non-leaky range (0.75 - 0.98).")
    else:
        print("\n[NOTE] Model AUC-ROC is lower than 0.75. This indicates a weak classifier or very high noise.")

    # Save model
    model.save_model(str(model_path))
    print(f"\nModel successfully trained and saved -> {model_path}")

    # Also save the generated labels for the 200 MSMEs to a mapping file in data/raw for verification
    label_map = {msme_id: int(label) for msme_id, label in zip(df["msme_id"], y)}
    label_file = data_dir / "simulated_labels.json"
    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"Simulated default labels saved -> {label_file}")


if __name__ == "__main__":
    main()
