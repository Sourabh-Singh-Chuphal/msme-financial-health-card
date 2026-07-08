"""FastAPI application for the MSME Financial Health Card.

Simulates ULI/OCEN/AA readiness. Exposes onboarding, Account Aggregator (AA)
consent flow, composite credit scoring, and SHAP explanations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Depends, HTTPException, Query, Header, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingestion.validator import validate
from src.features.pipeline import compute as feature_compute
from src.scoring.composite_scorer import compute_score
from src.explainability.shap_explainer import explain_score
from src.explainability.reason_codes import get_reason_codes

app = FastAPI(
    title="MSME Financial Health Card API",
    description="ULI/OCEN/AA-ready real-time credit scoring API for New-to-Credit MSMEs.",
    version="1.0.0"
)

# Allow all origins so the Vercel frontend can reach the Render-hosted API
# (Tighten to your Vercel domain after first deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Database (Pre-populate with the 200 synthetic MSMEs)
# ---------------------------------------------------------------------------
ONBOARDED_MSMES: dict[str, dict[str, Any]] = {}
ACTIVE_CONSENTS: dict[str, dict[str, Any]] = {}

# Pre-populate 200 synthetic MSME IDs so they can be scored immediately
for i in range(1, 201):
    msme_id = f"MSME_{i:06d}"
    ONBOARDED_MSMES[msme_id] = {
        "name": f"Synthetic MSME {i:06d}",
        "sector": "Retail" if i % 2 == 0 else "Manufacturing",
        "registration_number": f"GSTIN{i:06d}XXXXXXXX",
        "synthetic_id": msme_id
    }


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class OnboardRequest(BaseModel):
    name: str = Field(..., description="Legal business name of the MSME.")
    sector: str = Field(..., description="Industry sector (e.g. Retail, Manufacturing, Services).")
    registration_number: str = Field(..., description="GSTIN, Udyam, or PAN registration number.")
    annual_turnover_est: float | None = Field(None, description="Estimated annual turnover in INR.")


class OnboardResponse(BaseModel):
    msme_id: str = Field(..., description="Unique generated MSME Identifier for subsequent API calls.")
    status: str = Field("onboarded", description="Onboarding status.")


class ConsentRequest(BaseModel):
    msme_id: str = Field(..., description="Target MSME ID requesting consent.")
    data_sources: list[str] = Field(
        default=["gst", "upi", "aa", "epfo"],
        description="List of alternate data channels to request access to."
    )
    expiry_minutes: int = Field(60, description="Consent validity duration in minutes.")


class ConsentResponse(BaseModel):
    consent_token: str = Field(..., description="Secret token representing the active data sharing consent.")
    expiry: datetime = Field(..., description="ISO timestamp when consent expires.")
    msme_id: str = Field(..., description="Target MSME ID.")
    status: str = Field("active", description="Consent state.")


class DimensionScoreModel(BaseModel):
    stability: float | None = Field(..., description="Operations and tax filing stability score (0-100).")
    cashflow: float | None = Field(..., description="Cash surplus and buffer score (0-100).")
    compliance: float | None = Field(..., description="Statutory compliance discipline score (0-100).")
    growth: float | None = Field(..., description="Turnover and headcount growth trend score (0-100).")
    repayment: float | None = Field(..., description="Debt burden and return track record score (0-100).")


class ScoreResponse(BaseModel):
    msme_id: str = Field(..., description="MSME Identifier.")
    overall_score: float = Field(..., description="Blended final credit score (0-100).")
    dimension_scores: DimensionScoreModel = Field(..., description="Breakdown of individual dimension scores.")
    confidence_band: Literal["High", "Medium", "Low"] = Field(..., description="Data depth confidence category.")
    confidence_score: float = Field(..., description="Raw confidence score (0-1).")
    completeness_score: float = Field(..., description="Fraction of available data sources (0-1).")
    blend_ratio_used: dict[str, float] = Field(..., description="Weights used between Rule-Based and ML models.")
    sources_present: list[str] = Field(..., description="List of validated data sources utilized.")


class ReasonCodeModel(BaseModel):
    feature_name: str = Field(..., description="Internal feature key.")
    shap_value: float = Field(..., description="SHAP value in log-odds (negative means strength, positive means risk).")
    actual_value: Any = Field(..., description="Actual feature value extracted from the MSME profile.")
    description: str = Field(..., description="Human-readable explanation of the adjustment.")


class ExplainResponse(BaseModel):
    msme_id: str = Field(..., description="MSME Identifier.")
    predicted_probability_of_default: float = Field(..., description="XGBoost predicted probability of default.")
    base_value: float = Field(..., description="Expected base log-odds of default.")
    prediction_margin: float = Field(..., description="Raw log-odds output margin.")
    top_strengths: list[ReasonCodeModel] = Field(..., description="Top 3 positive credit influences.")
    top_risks: list[ReasonCodeModel] = Field(..., description="Top 3 risk factors.")


class HealthResponse(BaseModel):
    status: Literal["healthy"] = Field("healthy")
    api_version: str = Field("1.0.0")
    model_loaded: bool = Field(..., description="Boolean indicating if the XGBoost classifier is loaded.")


# ---------------------------------------------------------------------------
# Consent Dependency
# ---------------------------------------------------------------------------

def verify_consent(
    msme_id: str,
    consent_token: str | None = Query(None, description="Active consent token."),
    authorization: str | None = Header(None, description="Bearer token header.")
) -> str:
    """Verifies that there is a valid, active Account Aggregator consent token for the target MSME.

    Simulates ULI/OCEN compliance where data access requires explicit, time-bounded user consent.
    """
    token = None
    if consent_token:
        token = consent_token
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Account Aggregator consent token. Authorize via consent_token query param or Bearer header."
        )

    consent = ACTIVE_CONSENTS.get(token)
    if not consent:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unrecognized consent token."
        )

    if consent["msme_id"] != msme_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent token does not match the requested MSME ID."
        )

    if datetime.utcnow() > consent["expiry"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consent token has expired."
        )

    return token


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects the root URL to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.post(
    "/onboard",
    response_model=OnboardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Onboard MSME",
    tags=["Onboarding"]
)
def onboard_msme(request: OnboardRequest) -> OnboardResponse:
    """Onboards a new MSME applicant into the credit system.

    Simulates the initial lender portal or ULI interface onboarding flow.
    For demonstration purposes, newly onboarded MSMEs are mapped to a healthy
    synthetic profile (`MSME_000001`) so they can immediately be scored and explained.
    """
    new_id = f"MSME_REG_{uuid.uuid4().hex[:6].upper()}"
    ONBOARDED_MSMES[new_id] = {
        "name": request.name,
        "sector": request.sector,
        "registration_number": request.registration_number,
        # Map to a synthetic ID to allow scoring/explaining during demo
        "synthetic_id": "MSME_000001"
    }
    return OnboardResponse(msme_id=new_id)


@app.post(
    "/consent",
    response_model=ConsentResponse,
    summary="Create Data Consent Token",
    tags=["Consent Framework"]
)
def create_consent(request: ConsentRequest) -> ConsentResponse:
    """Simulates the Account Aggregator (AA) consent handshake.

    Performs the consent handshake per the Sahamati Account Aggregator architecture.
    Accepts the list of alternate data channels and returns a time-bound consent_token.
    This token is required to read scores or explanations for the MSME.
    """
    if request.msme_id not in ONBOARDED_MSMES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MSME ID {request.msme_id} is not onboarded."
        )

    token = f"CONSENT_TKN_{uuid.uuid4().hex[:12].upper()}"
    expiry = datetime.utcnow() + timedelta(minutes=request.expiry_minutes)

    ACTIVE_CONSENTS[token] = {
        "msme_id": request.msme_id,
        "sources": request.data_sources,
        "expiry": expiry
    }

    return ConsentResponse(
        consent_token=token,
        expiry=expiry,
        msme_id=request.msme_id
    )


@app.get(
    "/score/{msme_id}",
    response_model=ScoreResponse,
    summary="Compute Composite Financial Health Score",
    tags=["Scoring Engine"]
)
def get_score(
    msme_id: str,
    _token: str = Depends(verify_consent)
) -> ScoreResponse:
    """Retrieves the unified composite credit score for a specific MSME.

    Simulates the final ULI credit decisioning point.
    Runs the full pipeline:
    1. Alternate data validation (`src/ingestion`)
    2. Seasonality-adjusted dimension feature engineering (`src/features`)
    3. Composite scoring (`src/scoring`)

    Handles missing sources gracefully by using the rule-based weight redistribution
    and XGBoost's native missing value support. Returns HTTP 200 with reduced confidence.
    """
    msme_info = ONBOARDED_MSMES.get(msme_id)
    if not msme_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MSME ID {msme_id} not found."
        )

    synthetic_id = msme_info["synthetic_id"]
    data_dir = Path("data/raw")

    try:
        # Run pipeline
        ingestion = validate(synthetic_id, data_dir=data_dir)
        if not ingestion.can_score:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="MSME has no available alternate data sources. Cannot compute score."
            )

        fv = feature_compute(ingestion)
        final_score = compute_score(fv)

        return ScoreResponse(
            msme_id=msme_id,
            overall_score=final_score.overall_score,
            dimension_scores=DimensionScoreModel(
                stability=final_score.dimension_scores.get("stability"),
                cashflow=final_score.dimension_scores.get("cashflow"),
                compliance=final_score.dimension_scores.get("compliance"),
                growth=final_score.dimension_scores.get("growth"),
                repayment=final_score.dimension_scores.get("repayment")
            ),
            confidence_band=final_score.confidence_band,
            confidence_score=final_score.confidence_score,
            completeness_score=fv.completeness_score,
            blend_ratio_used=final_score.blend_ratio_used,
            sources_present=fv.sources_present
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score MSME: {str(exc)}"
        )


@app.get(
    "/explain/{msme_id}",
    response_model=ExplainResponse,
    summary="Compute Credit Explanation Details",
    tags=["Explainable AI"]
)
def get_explain(
    msme_id: str,
    _token: str = Depends(verify_consent)
) -> ExplainResponse:
    """Retrieves credit strengths and risk factors for a specific MSME.

    Simulates the loan officer dashboard or ULI transparency interface.
    Extracts raw Tree SHAP values from the trained XGBoost model and maps them to
    specific, human-readable reason codes detailing operational strengths and risks.
    """
    msme_info = ONBOARDED_MSMES.get(msme_id)
    if not msme_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MSME ID {msme_id} not found."
        )

    synthetic_id = msme_info["synthetic_id"]

    try:
        explanation = explain_score(synthetic_id, data_dir="data/raw")
        strengths, risks = get_reason_codes(explanation)

        return ExplainResponse(
            msme_id=msme_id,
            predicted_probability_of_default=explanation.predicted_probability,
            base_value=explanation.base_value,
            prediction_margin=explanation.prediction_margin,
            top_strengths=[
                ReasonCodeModel(
                    feature_name=rc.feature_name,
                    shap_value=rc.shap_value,
                    actual_value=rc.actual_value,
                    description=rc.description
                ) for rc in strengths
            ],
            top_risks=[
                ReasonCodeModel(
                    feature_name=rc.feature_name,
                    shap_value=rc.shap_value,
                    actual_value=rc.actual_value,
                    description=rc.description
                ) for rc in risks
            ]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain MSME credit: {str(exc)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Check",
    tags=["Infrastructure"]
)
def health_check() -> HealthResponse:
    """Basic infrastructure liveness route. Verifies that the classifier can be loaded."""
    model_loaded = False
    try:
        # Try loading explainer cache or checking model file
        model_path = Path("config/model.json")
        model_loaded = model_path.exists()
    except Exception:
        pass

    return HealthResponse(model_loaded=model_loaded)
