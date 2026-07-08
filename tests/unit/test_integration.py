"""Unit tests for the ecosystem integration adapters (AA, OCEN, ULI).

Verifies that:
1. Consent artifacts follow the ReBIT model and include required fields.
2. OCEN payloads are generated with reasonable terms matching risk bands.
3. ULI profiles aggregate identity and alternate data indexes correctly.
"""

from __future__ import annotations

import json
from src.integration.aa_consent_flow import generate_mock_consent_artifact
from src.integration.ocen_adapter import generate_ocen_payload, push_to_lsp
from src.integration.uli_adapter import map_to_uli_profile


def test_aa_consent_artifact_generation() -> None:
    msme_id = "MSME_000001"
    customer_handle = "merchant@anumati"
    sources = ["gst", "upi", "aa"]

    artifact = generate_mock_consent_artifact(msme_id, customer_handle, sources)

    assert artifact.consent_id is not None
    assert artifact.customer.id == customer_handle
    assert "GST" in artifact.fi_types
    assert "UPI" in artifact.fi_types
    assert "DEPOSIT" in artifact.fi_types
    assert "EPFO" not in artifact.fi_types
    assert artifact.purpose.code == "102"  # Credit Assessment
    assert artifact.signature == "MOCK_DIGITAL_SIGNATURE_OF_ACCOUNT_AGGREGATOR"


def test_ocen_payload_and_lsp_push() -> None:
    msme_id = "MSME_000001"
    score_res = {
        "overall_score": 85.5,
        "confidence_band": "High",
        "completeness_score": 1.0,
        "sources_present": ["gst", "upi", "aa", "epfo"],
        "dimension_scores": {"stability": 80.0, "cashflow": 90.0}
    }
    token = "CONSENT_TKN_MOCK"

    payload = generate_ocen_payload(msme_id, score_res, token)

    assert payload.borrower_id == msme_id
    assert payload.consent_ref == token
    assert payload.health_score_summary.overall_score == 85.5
    assert payload.recommended_terms.max_suggested_credit_limit == 500000.0
    assert payload.recommended_terms.pricing_tier == "Prime"

    # Test transmission log mock
    receipt = push_to_lsp(payload)
    assert receipt["status"] == "QUEUED_FOR_OFFERS"
    assert "ocenTxId" in receipt


def test_uli_merchant_profile_mapping() -> None:
    msme_id = "MSME_000001"
    pan = "ABCDE1234F"
    score_res = {
        "overall_score": 92.0,
        "confidence_band": "High",
        "completeness_score": 1.0,
        "sources_present": ["gst", "upi", "aa", "epfo"]
    }
    features = {
        "filing_regularity": 0.95,
        "gst_mismatch_rate": 0.0,
        "avg_daily_inflow": 15000.0,
        "net_cashflow_ratio": 0.20,
        "employee_count": 12,
        "overdraft_day_rate": 0.0,
        "avg_monthly_bounces": 0.0
    }

    profile = map_to_uli_profile(msme_id, pan, score_res, features)

    assert profile.uli_transaction_id.startswith("ULI-TXN-")
    assert profile.merchant_pan == "XXXXXX234F"
    assert profile.gstin_verified is True
    assert profile.composite_credit_health_score == 92.0
    assert profile.alternate_data_summary.gst_filing_punctuality == 0.95
    assert profile.alternate_data_summary.epfo_active_employees == 12
    assert profile.alternate_data_summary.upi_net_inflow_surplus_inr == 90000.0  # 15000 * 30 * 0.20
