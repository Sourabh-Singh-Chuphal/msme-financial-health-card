"""Print sample reason codes for 5 MSMEs representing each persona.

Examines one MSME per cohort persona type, outputs their final blended score,
their probability of default, and prints the top 3 strengths and top 3 risk
factors based on SHAP values.

Run from project root:
    python notebooks/print_sample_reasons.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.shap_explainer import explain_score
from src.explainability.reason_codes import get_reason_codes
from src.scoring.composite_scorer import compute_score
from src.features.pipeline import compute as feature_compute
from src.ingestion.validator import validate

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def find_one_of_each_persona() -> dict[str, str]:
    """Find one MSME ID for each of the 5 persona types."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    msme_ids = manifest["msme_ids"]

    selected = {}
    personas_needed = {
        "healthy_established",
        "healthy_ntc",
        "risky_declining",
        "risky_volatile",
        "seasonal_business",
    }

    for msme_id in msme_ids:
        meta_path = DATA_DIR / msme_id / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        persona = meta.get("persona_type")
        if persona in personas_needed and persona not in selected:
            selected[persona] = msme_id
            if len(selected) == len(personas_needed):
                break

    return selected


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}. Run synthetic cohort generators first.")
        sys.exit(1)

    selected_msmes = find_one_of_each_persona()

    print("=" * 80)
    print("CREDIT EXPLAINER SANITY CHECK: SAMPLE REASON CODES BY PERSONA")
    print("=" * 80)

    for persona, msme_id in selected_msmes.items():
        # Score the MSME
        ingestion = validate(msme_id, data_dir=DATA_DIR)
        fv = feature_compute(ingestion)
        final_res = compute_score(fv)

        # Compute SHAP explanation
        explanation = explain_score(msme_id, data_dir=DATA_DIR)
        strengths, risks = get_reason_codes(explanation)

        print(f"\nPersona Type : {persona.upper()}")
        print(f"MSME ID      : {msme_id}")
        print(f"Final Score  : {final_res.overall_score:.2f} (Confidence Band: {final_res.confidence_band})")
        print(f"ML Risk (PD) : {final_res.ml_score.probability_of_default * 100:.1f}%" if final_res.ml_score.overall_score is not None else "ML Risk (PD) : N/A")
        print("-" * 80)
        
        print("Top Credit Strengths (Positive Influences):")
        if strengths:
            for idx, rc in enumerate(strengths, 1):
                print(f"  {idx}. [SHAP: {rc.shap_value:.4f}] {rc.description} (Value: {rc.actual_value})")
        else:
            print("  None identified.")

        print("\nTop Risk Factors (Negative Influences):")
        if risks:
            for idx, rc in enumerate(risks, 1):
                print(f"  {idx}. [SHAP:  {rc.shap_value:.4f}] {rc.description} (Value: {rc.actual_value})")
        else:
            print("  None identified.")
            
        print("=" * 80)


if __name__ == "__main__":
    main()
