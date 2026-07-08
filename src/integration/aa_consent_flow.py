"""Account Aggregator (AA) Consent Flow Module.

Models the official ReBIT (Reserve Bank Information Technology) / Sahamati
consent artifact schema used for alternate data pulling in the Indian fintech ecosystem.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal
from pydantic import BaseModel, Field


class PurposeModel(BaseModel):
    code: str = Field(
        "102",
        description="ReBIT purpose code. Code '102' indicates 'Credit Assessment' (mandatory for loan underwriting)."
    )
    refUri: str = Field(
        "https://api.sahamati.org.in/purposes/102.json",
        description="Reference URI to the standard purpose schema."
    )
    text: str = Field(
        "MSME Credit Risk Assessment and Underwriting Analysis",
        description="Underwriter-facing text describing the consent usage."
    )


class DataRangeModel(BaseModel):
    from_date: datetime = Field(..., alias="from", description="Start of historical data pull range.")
    to_date: datetime = Field(..., alias="to", description="End of historical data pull range.")


class IdentifierModel(BaseModel):
    type: Literal["PAN", "GSTIN", "MOBILE", "VPA"] = Field(..., description="ID type.")
    value: str = Field(..., description="Actual identifier value.")


class CustomerModel(BaseModel):
    id: str = Field(..., description="Customer AA handle (e.g. mobile@sahamati or merchant@nodal_aa).")


class DataConsumerModel(BaseModel):
    id: str = Field("FIU-LENDER-999", description="Financial Information User (FIU) identifier.")


class ConsentArtifact(BaseModel):
    """The standard ReBIT-compliant Consent Artifact schema.

    This represents the signed XML/JSON object returned by the Account Aggregator
    once the MSME merchant approves the consent request via their AA app.
    """
    consent_id: str = Field(..., alias="consentId", description="Unique global consent transaction ID.")
    consent_start: datetime = Field(default_factory=datetime.utcnow, alias="consentStart")
    consent_expiry: datetime = Field(..., alias="consentExpiry", description="Expiry time of the consent artifact itself.")
    consent_mode: Literal["STORE", "VIEW"] = Field("STORE", alias="consentMode")
    consent_types: list[Literal["PROFILE", "TRANSACTIONS", "SUMMARY"]] = Field(
        default=["PROFILE", "TRANSACTIONS"],
        alias="consentTypes"
    )
    fi_types: list[Literal["DEPOSIT", "TERM_DEPOSIT", "GST", "EPFO", "UPI"]] = Field(
        default=["DEPOSIT", "GST", "UPI"],
        alias="fiTypes",
        description="Financial Information types requested. Maps our GST/UPI/AA/EPFO alternate data sources."
    )
    purpose: PurposeModel = Field(default_factory=PurposeModel)
    customer: CustomerModel
    data_consumer: DataConsumerModel = Field(default_factory=DataConsumerModel, alias="dataConsumer")
    data_range: DataRangeModel = Field(..., alias="dataRange")
    signature: str = Field(
        "MOCK_DIGITAL_SIGNATURE_OF_ACCOUNT_AGGREGATOR",
        description="Cryptographic signature of the AA verifying authenticity and non-repudiation."
    )


def generate_mock_consent_artifact(
    msme_id: str,
    customer_handle: str,
    sources: list[str],
    expiry_minutes: int = 60
) -> ConsentArtifact:
    """Generates a ReBIT-compliant consent artifact for demo purposes.

    Simulates the actual data response from an AA provider (e.g. Anumati, OneMoney, CAMS).
    """
    now = datetime.utcnow()
    
    # Map sources string to FI Types
    fi_map = {
        "gst": "GST",
        "upi": "UPI",
        "aa": "DEPOSIT",
        "epfo": "EPFO"
    }
    fi_types = [fi_map[s] for s in sources if s in fi_map]
    if not fi_types:
        fi_types = ["DEPOSIT", "GST", "UPI"]

    # Calculate 12 months historical range (standard credit review window)
    data_from = now - timedelta(days=365)

    import uuid
    consent_id = str(uuid.uuid4())

    return ConsentArtifact(
        consentId=consent_id,
        consentStart=now,
        consentExpiry=now + timedelta(minutes=expiry_minutes),
        consentMode="STORE",
        consentTypes=["PROFILE", "TRANSACTIONS"],
        fiTypes=fi_types,
        customer=CustomerModel(id=customer_handle),
        dataRange=DataRangeModel(**{"from": data_from, "to": now})
    )
