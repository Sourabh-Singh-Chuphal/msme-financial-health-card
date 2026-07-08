"""End-to-End integration test for MSME credit decisioning flow.

Tests the full lifecycle for all 5 credit personas:
1. Retrieve a representative ID for each persona type.
2. Complete the Account Aggregator (AA) consent handshake via POST /consent.
3. Compute the composite credit score via GET /score/{msme_id}.
4. Extract explainability reason codes via GET /explain/{msme_id}.

Asserts risk band classifications conform to the core design specifications
(e.g., healthy_ntc is not penalized into High Risk due to a thin data profile).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)
MANIFEST_PATH = Path("data/raw/manifest.json")
DATA_DIR = Path("data/raw")


def get_persona_msme_id(target_persona: str) -> str:
    """Finds the first MSME ID matching the target persona in the synthetic cohort."""
    if not MANIFEST_PATH.exists():
        pytest.skip("Synthetic manifest not found. Skipping E2E test.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    msme_ids = manifest["msme_ids"]

    for msme_id in msme_ids:
        meta_path = DATA_DIR / msme_id / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("persona_type") == target_persona:
                return msme_id

    pytest.skip(f"No MSME found for persona {target_persona}")


@pytest.mark.parametrize("persona,expected_risk_bands,score_threshold_assertion", [
    (
        "healthy_established",
        ["LOW RISK", "MEDIUM RISK"],
        lambda score: score >= 75.0
    ),
    (
        "healthy_ntc",
        ["LOW RISK", "MEDIUM RISK"],  # Core thesis: New-to-credit must NOT be High Risk
        lambda score: score >= 60.0    # Should be at least Medium Risk or Low Risk
    ),
    (
        "risky_declining",
        ["MEDIUM RISK", "HIGH RISK"],
        lambda score: score < 85.0     # Must show visible credit degradation
    ),
    (
        "risky_volatile",
        ["LOW RISK", "MEDIUM RISK", "HIGH RISK"],
        lambda score: score <= 100.0   # General bound validation
    ),
    (
        "seasonal_business",
        ["LOW RISK", "MEDIUM RISK"],
        lambda score: score >= 75.0    # Must not be penalized for seasonality
    )
])
def test_full_underwriting_flow_by_persona(
    persona: str,
    expected_risk_bands: list[str],
    score_threshold_assertion: object
) -> None:
    """Runs the full E2E credit scoring and explainability loop for the target persona."""
    # 1. Fetch persona MSME reference ID
    msme_id = get_persona_msme_id(persona)

    # 2. Account Aggregator Consent Handshake (POST /consent)
    consent_resp = client.post(
        "/consent",
        json={
            "msme_id": msme_id,
            "data_sources": ["gst", "upi", "aa", "epfo"],
            "expiry_minutes": 15
        }
    )
    assert consent_resp.status_code == 200
    token = consent_resp.json()["consent_token"]
    assert token.startswith("CONSENT_TKN_")

    # 3. Compute Composite Credit Score (GET /score/{msme_id})
    score_resp = client.get(f"/score/{msme_id}?consent_token={token}")
    assert score_resp.status_code == 200
    score_data = score_resp.json()

    assert score_data["msme_id"] == msme_id
    score = score_data["overall_score"]
    assert 0.0 <= score <= 100.0
    
    # Assert score meets mathematical threshold constraints
    assert score_threshold_assertion(score), f"Score {score:.2f} failed threshold checks for {persona}"

    # Determine risk band based on score threshold
    if score >= 80.0:
        actual_risk_band = "LOW RISK"
    elif score >= 60.0:
        actual_risk_band = "MEDIUM RISK"
    else:
        actual_risk_band = "HIGH RISK"

    assert actual_risk_band in expected_risk_bands, (
        f"Persona '{persona}' classified into '{actual_risk_band}', expected one of {expected_risk_bands}"
    )

    # 4. Extract Credit Explanations (GET /explain/{msme_id})
    explain_resp = client.get(f"/explain/{msme_id}?consent_token={token}")
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()

    assert explain_data["msme_id"] == msme_id
    assert "top_strengths" in explain_data
    assert "top_risks" in explain_data

    # Both strengths and risks should be list format
    assert isinstance(explain_data["top_strengths"], list)
    assert isinstance(explain_data["top_risks"], list)
