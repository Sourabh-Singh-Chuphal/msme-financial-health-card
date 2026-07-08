"""Integration tests for the MSME credit API.

Verifies:
1. Happy path onboarding, consent token handshake, scoring, and explainability.
2. Authorization checks: rejection of calls with missing or invalid tokens.
3. Partial data processing: scoring for a partial-data MSME (NTC) succeeds with 200,
   but outputs a lower confidence and None for missing dimensions.
4. Nonexistent targets: clean HTTP 404 responses for invalid IDs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_api_happy_path_flow() -> None:
    # 1. Onboard a new MSME
    onboard_resp = client.post(
        "/onboard",
        json={
            "name": "Acme Traders",
            "sector": "Retail",
            "registration_number": "GSTIN99887766XX",
            "annual_turnover_est": 5000000.0
        }
    )
    assert onboard_resp.status_code == 201
    onboard_data = onboard_resp.json()
    msme_id = onboard_data["msme_id"]
    assert msme_id.startswith("MSME_REG_")

    # 2. Request Account Aggregator consent
    consent_resp = client.post(
        "/consent",
        json={
            "msme_id": msme_id,
            "data_sources": ["gst", "upi"],
            "expiry_minutes": 30
        }
    )
    assert consent_resp.status_code == 200
    consent_data = consent_resp.json()
    token = consent_data["consent_token"]
    assert token.startswith("CONSENT_TKN_")

    # 3. Retrieve credit score using the consent token (Query Param)
    score_resp = client.get(f"/score/{msme_id}?consent_token={token}")
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    assert score_data["msme_id"] == msme_id
    assert 0 <= score_data["overall_score"] <= 100
    assert "confidence_band" in score_data

    # 4. Retrieve credit score using Bearer Authorization Header
    score_hdr_resp = client.get(
        f"/score/{msme_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert score_hdr_resp.status_code == 200
    score_hdr_data = score_hdr_resp.json()
    assert score_hdr_data["overall_score"] == score_data["overall_score"]

    # 5. Retrieve credit explanation reasons
    explain_resp = client.get(f"/explain/{msme_id}?consent_token={token}")
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()
    assert "top_strengths" in explain_data
    assert "top_risks" in explain_data


def test_api_missing_or_invalid_consent() -> None:
    # Use one of the pre-populated synthetic MSMEs
    msme_id = "MSME_000001"

    # 1. Missing Token -> 401 Unauthorized
    score_missing_resp = client.get(f"/score/{msme_id}")
    assert score_missing_resp.status_code == 401
    assert "Missing Account Aggregator consent token" in score_missing_resp.json()["detail"]

    # 2. Invalid Token -> 403 Forbidden
    score_invalid_resp = client.get(f"/score/{msme_id}?consent_token=CONSENT_TKN_INVALID")
    assert score_invalid_resp.status_code == 403
    assert "Invalid or unrecognized consent token" in score_invalid_resp.json()["detail"]


def test_api_partial_data_msme() -> None:
    """NTC MSME has partial data: scoring must succeed with 200, but output lower confidence."""
    # MSME_000041 is healthy_ntc (only GST and UPI sources present)
    msme_id = "MSME_000041"

    # 1. Create consent token
    consent_resp = client.post(
        "/consent",
        json={"msme_id": msme_id, "data_sources": ["gst", "upi"]}
    )
    assert consent_resp.status_code == 200
    token = consent_resp.json()["consent_token"]

    # 2. Score
    score_resp = client.get(f"/score/{msme_id}?consent_token={token}")
    assert score_resp.status_code == 200
    score_data = score_resp.json()

    # Verify scores are generated and repayment capacity (AA dependent) is None
    assert score_data["overall_score"] > 70.0
    assert score_data["dimension_scores"]["repayment"] is None
    # At least one dimension score must be computed for a valid NTC score
    active_scores = [val for val in score_data["dimension_scores"].values() if val is not None]
    assert len(active_scores) > 0

    # Verify confidence band is Medium/Low
    assert score_data["confidence_band"] in ("Medium", "Low")
    assert score_data["completeness_score"] < 1.0


def test_api_nonexistent_ids() -> None:
    non_id = "MSME_999999"

    # 1. Requesting consent for non-existent MSME -> 404 Not Found
    consent_resp = client.post(
        "/consent",
        json={"msme_id": non_id}
    )
    assert consent_resp.status_code == 404

    # 2. Scoring a non-existent MSME (with valid token for a different ID)
    # Onboard one valid MSME to get a valid token
    onboard_resp = client.post(
        "/onboard",
        json={"name": "Temp", "sector": "Retail", "registration_number": "REG"}
    )
    valid_id = onboard_resp.json()["msme_id"]
    consent_valid = client.post("/consent", json={"msme_id": valid_id})
    token = consent_valid.json()["consent_token"]

    # Requesting score for non-existent ID with a token for a different ID -> 403 Forbidden
    score_mismatch_resp = client.get(f"/score/{non_id}?consent_token={token}")
    assert score_mismatch_resp.status_code == 403
    assert "token does not match" in score_mismatch_resp.json()["detail"]
