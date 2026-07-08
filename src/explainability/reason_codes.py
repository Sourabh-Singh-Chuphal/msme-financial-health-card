"""Reason codes generator for credit health card explainability.

Translates raw SHAP log-odds contribution values into structured,
human-readable credit strengths (positive contributions) and risk factors
(negative contributions).
"""

from __future__ import annotations

from dataclasses import dataclass
from src.explainability.shap_explainer import ExplanationResult


@dataclass
class ReasonCode:
    """A single human-readable reason for a score adjustment."""
    feature_name: str
    shap_value: float
    actual_value: float | bool | None
    description: str


# Mapping of features to templates.
# Format: { feature_name: (strength_template, risk_template) }
# If a template contains {val}, it will be formatted with the formatted value.
TEMPLATES = {
    # Data Completeness
    "completeness_score": (
        "Strong data completeness provides high confidence in financial profile.",
        "Limited historical records or missing data sources reduce overall underwriting confidence."
    ),
    "n_sources": (
        "Multiple alternate data sources available to build a robust profile.",
        "Thin data profile with very few active alternate data channels."
    ),

    # Stability Features
    "stability_score": (
        "High overall stability score indicates consistent business operations.",
        "Lower stability score indicates volatility in operations or cash balances."
    ),
    "stability_filing_regularity": (
        "Consistent and punctual GST return filing history.",
        "Irregular or delayed GST filing behavior in recent periods."
    ),
    "stability_avg_filing_delay_days": (
        "GST returns are filed well before the due date on average.",
        "Substantial average filing delays ({val:.1f} days) in recent GST returns."
    ),
    "stability_turnover_cv": (
        "Highly stable and consistent monthly GST turnover (low volatility).",
        "Significant month-over-month volatility in GST turnover."
    ),
    "stability_gst_mismatch_rate": (
        "Minimal mismatch discrepancies between GSTR-1 and GSTR-3B filings.",
        "Frequent mismatch discrepancies ({val:.1%}) between GSTR-1 and GSTR-3B filings."
    ),
    "stability_balance_cv": (
        "Maintains stable daily closing balances in bank statement logs.",
        "High daily closing balance volatility indicates unstable cash management."
    ),
    "stability_overdraft_day_rate_stability": (
        "Zero or minimal overdraft episodes, showing comfortable liquidity.",
        "Frequent overdraft usage ({val:.1%}) indicates short-term liquidity pressure."
    ),

    # Cash Flow Features
    "cashflow_score": (
        "Excellent cash flow surplus and buffer capacity.",
        "Cash flow constraints identified due to high outflows or thin buffers."
    ),
    "cashflow_net_cashflow_ratio": (
        "Strong net cash inflow surplus ({val:.1%}) relative to total inflows.",
        "Negative or tight net cash flow ratio ({val:.1%}), showing high relative outflows."
    ),
    "cashflow_days_of_cash_buffer": (
        "Comfortable cash buffer of {val:.1f} days to cover standard operational expenses.",
        "Thin operational cash buffer of only {val:.1f} days, leaving business vulnerable to shocks."
    ),
    "cashflow_inflow_cv": (
        "Steady and predictable daily UPI inflows.",
        "High volatility in daily UPI inflow volumes."
    ),
    "cashflow_avg_refund_rate": (
        "Very low customer refund rate, indicating transaction reliability.",
        "Elevated customer refund rate ({val:.1%}), indicating operational or delivery issues."
    ),
    "cashflow_avg_monthly_bounces": (
        "Clean bank statement history with zero bounced transactions.",
        "Frequent bounced debit transactions ({val:.1f} per month) indicate payment stress."
    ),
    "cashflow_balance_trend_monthly": (
        "Upward trend in average monthly bank balances.",
        "Downward trend in average monthly bank balances."
    ),

    # Compliance Features
    "compliance_score": (
        "Outstanding compliance discipline across GST and employee benefits.",
        "Compliance gaps identified in timely filing or statutory deposits."
    ),
    "compliance_pf_regularity_rate": (
        "Employee EPFO provident fund contributions are consistently deposited on time.",
        "Delayed EPFO contribution deposits in recent employee wage cycles."
    ),
    "compliance_avg_churn_rate": (
        "Low employee turnover rate indicates organizational and team stability.",
        "Elevated employee churn rate ({val:.1%}) indicates potential labor or operational instability."
    ),
    "compliance_headcount_stability_trend": (
        "Stable or growing workforce headcount over the past year.",
        "Workforce size has steadily declined, indicating potential business contraction."
    ),

    # Growth Features
    "growth_score": (
        "Business shows strong positive growth trajectory.",
        "Business is experiencing negative growth or contraction."
    ),
    "growth_gst_trend_12m": (
        "Healthy long-term upward trend in deseasonalized GST turnover.",
        "Long-term decline in deseasonalized monthly GST turnover."
    ),
    "growth_gst_trend_3m": (
        "Accelerating short-term growth in recent GST turnover.",
        "Short-term contraction in recent GST turnover."
    ),
    "growth_upi_trend_12m": (
        "Strong long-term upward trend in monthly UPI collections.",
        "Long-term decline in monthly UPI transaction volume."
    ),
    "growth_upi_trend_3m": (
        "Strong recent acceleration in UPI transaction volume.",
        "Recent decline in monthly UPI transaction volumes."
    ),
    "growth_headcount_trend": (
        "Expanding employee workforce indicates company scaling.",
        "Shrinking workforce indicates business contraction."
    ),

    # Repayment Features
    "repayment_score": (
        "Excellent debt servicing capacity with minimal risk indicators.",
        "Constrained repayment capacity due to existing burden or bank returns."
    ),
    "repayment_emi_burden_ratio": (
        "Very low existing EMI burden relative to monthly bank balances.",
        "High existing EMI burden (EMIs consume {val:.1%} of monthly balances)."
    ),
    "repayment_consecutive_overdraft_max": (
        "No chronic or consecutive overdraft stress periods.",
        "Chronic liquidity stress with consecutive overdraft usage for {val} days."
    ),
    "repayment_avg_monthly_bounces": (
        "Consistent payment track record with zero bounced debits.",
        "Frequent bounced debits ({val:.1f} per month) indicate severe repayment stress."
    )
}


def get_reason_codes(explanation: ExplanationResult) -> tuple[list[ReasonCode], list[ReasonCode]]:
    """Convert raw SHAP values into ranked strengths and risks.

    In credit scoring:
    - Negative SHAP reduces default probability -> Credit Strength (Positive Reason)
    - Positive SHAP increases default probability -> Risk Factor (Negative Reason)

    Returns:
        (top_3_strengths, top_3_risks) sorted by absolute influence (magnitude of SHAP).
    """
    strengths = []
    risks = []

    for feat, shap_val in explanation.shap_values.items():
        if abs(shap_val) < 1e-4:
            continue

        actual_val = explanation.feature_values.get(feat)
        
        # Determine templates
        templates = TEMPLATES.get(feat)
        if templates:
            strength_tpl, risk_tpl = templates
        else:
            # Fallback dynamic template
            strength_tpl = f"Positive contribution from {feat} (value: {{val}})"
            risk_tpl = f"Risk factor from {feat} (value: {{val}})"

        # Format description
        is_strength = shap_val < 0
        tpl = strength_tpl if is_strength else risk_tpl
        
        try:
            if actual_val is None:
                description = tpl.split("(")[0].strip()  # Strip out formatted parts if None
            elif isinstance(actual_val, bool):
                description = tpl
            else:
                description = tpl.format(val=actual_val)
        except Exception:
            description = tpl.replace("{val}", str(actual_val)) if actual_val is not None else tpl

        rc = ReasonCode(
            feature_name=feat,
            shap_value=shap_val,
            actual_value=actual_val,
            description=description
        )

        if is_strength:
            strengths.append(rc)
        else:
            risks.append(rc)

    # Sort both lists by absolute SHAP value (descending) to find the most influential reasons
    strengths_sorted = sorted(strengths, key=lambda x: abs(x.shap_value), reverse=True)
    risks_sorted = sorted(risks, key=lambda x: abs(x.shap_value), reverse=True)

    return strengths_sorted[:3], risks_sorted[:3]
