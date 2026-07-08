"""Unit tests for the explainability layer.

Enforces the core mathematical property of SHAP: the sum of the feature
contributions must equal the predicted margin minus the base value.
Also verifies that the reason codes are ranked, filtered, and returned
correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.explainability.shap_explainer import explain_score
from src.explainability.reason_codes import get_reason_codes


def test_shap_mathematical_sum_property() -> None:
    """Mathematical verification: sum of SHAP values must equal prediction_margin - base_value."""
    manifest_path = Path("data/raw/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Synthetic data cohort not generated. Skipping integration test.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    msme_ids = manifest["msme_ids"]
    assert len(msme_ids) > 0, "Manifest contains no MSME IDs"

    # Use the first MSME in the cohort
    sample_id = msme_ids[0]
    
    explanation = explain_score(sample_id, data_dir="data/raw", model_path="config/model.json")

    # Math assertion: sum(SHAP) = prediction - base_value
    shap_sum = sum(explanation.shap_values.values())
    expected_difference = explanation.prediction_margin - explanation.base_value

    # Check with 1e-4 tolerance due to float representation in TreeExplainer / DMatrix
    assert shap_sum == pytest.approx(expected_difference, abs=1e-4), (
        f"SHAP sum {shap_sum:.6f} does not equal margin - base_value ({expected_difference:.6f})"
    )


def test_reason_codes_ranking_and_bounds() -> None:
    """Verify that reason codes are successfully mapped, formatted, and limited to top 3."""
    manifest_path = Path("data/raw/manifest.json")
    if not manifest_path.exists():
        pytest.skip("Synthetic data cohort not generated. Skipping integration test.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sample_id = manifest["msme_ids"][0]

    explanation = explain_score(sample_id, data_dir="data/raw", model_path="config/model.json")
    strengths, risks = get_reason_codes(explanation)

    # Assert limits (at most 3 codes returned)
    assert len(strengths) <= 3
    assert len(risks) <= 3

    # Assert correct sign categorization
    for rc in strengths:
        # Strength = reduces default risk -> negative SHAP
        assert rc.shap_value < 0, f"Strength feature {rc.feature_name} has positive SHAP"
        assert len(rc.description) > 0

    for rc in risks:
        # Risk = increases default risk -> positive SHAP
        assert rc.shap_value > 0, f"Risk feature {rc.feature_name} has negative SHAP"
        assert len(rc.description) > 0

    # Assert sorted by descending absolute SHAP value
    for i in range(len(strengths) - 1):
        assert abs(strengths[i].shap_value) >= abs(strengths[i+1].shap_value)
    for i in range(len(risks) - 1):
        assert abs(risks[i].shap_value) >= abs(risks[i+1].shap_value)
