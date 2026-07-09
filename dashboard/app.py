"""Streamlit Dashboard for MSME Financial Health Card.

Provides a bank credit-officer dashboard interface, calling the FastAPI
credit backend and visualizing scores, plotly radar charts, SHAP reason codes,
and alternate data source completeness.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="MSME Financial Health Card Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint Config — defaults to the hosted Render backend so cloud deploys work out-of-the-box
API_BASE_URL = os.getenv("API_BASE_URL", "https://msme-financial-health-card-1.onrender.com")

# Persona metadata mappings
PERSONA_LABELS = {
    "healthy_established": "Healthy Established",
    "healthy_ntc": "Healthy New-to-Credit (NTC)",
    "risky_declining": "Risky Declining",
    "risky_volatile": "Risky Volatile",
    "seasonal_business": "Seasonal Business"
}

# Persona description mappings
PERSONA_DESC = {
    "healthy_established": "Mature business with rich history across all credit and alternate data channels.",
    "healthy_ntc": "New-to-Bank/Credit merchant. Thin financial history, evaluated primarily on GST/UPI alternate flows.",
    "risky_declining": "Established business experiencing operational and revenue contraction, with cash returns.",
    "risky_volatile": "High cash-flow volatility, frequent short-term overdraft usage, and erratic filings.",
    "seasonal_business": "Predictable seasonal revenue swings (Diwali peak, monsoon dip). Deseasonalized growth trajectory."
}


# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------

@st.cache_data
def load_msme_cohort() -> dict[str, list[tuple[str, str, str]]]:
    """Scan raw data directory and group MSME IDs by persona type."""
    manifest_path = PROJECT_ROOT / "data" / "raw" / "manifest.json"
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        msme_ids = manifest.get("msme_ids", [])
    except Exception:
        return {}

    grouped: dict[str, list[tuple[str, str, str]]] = {p: [] for p in PERSONA_LABELS}

    for msme_id in msme_ids:
        meta_path = PROJECT_ROOT / "data" / "raw" / msme_id / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                persona = meta.get("persona_type", "unknown")
                name = meta.get("business_name", f"MSME {msme_id}")
                sector = meta.get("sector", "Retail")
                if persona in grouped:
                    grouped[persona].append((msme_id, name, sector))
            except Exception:
                pass
    return grouped


# ---------------------------------------------------------------------------
# Backend API Calls
# ---------------------------------------------------------------------------

def check_backend_liveness(timeout: float = 15.0) -> bool:
    """Checks if the FastAPI server is reachable.
    
    Uses a longer timeout to handle Render free-tier cold starts
    (the service may be sleeping and needs ~10-15 s to wake up).
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_credit_score_and_explain(msme_id: str) -> tuple[dict | None, dict | None, str | None]:
    """Hits the FastAPI backend to retrieve score and explanations.

    Simulates the full AA-consent handshake first, then fetches details.
    """
    try:
        # 1. Request AA Consent Token
        consent_resp = requests.post(
            f"{API_BASE_URL}/consent",
            json={"msme_id": msme_id, "data_sources": ["gst", "upi", "aa", "epfo"]},
            timeout=30.0
        )
        if consent_resp.status_code != 200:
            return None, None, f"Consent Handshake Failed: {consent_resp.json().get('detail', 'Unknown error')}"
        
        token = consent_resp.json()["consent_token"]

        # 2. Fetch Score
        score_resp = requests.get(f"{API_BASE_URL}/score/{msme_id}?consent_token={token}", timeout=30.0)
        if score_resp.status_code != 200:
            return None, None, f"Scoring Engine Error: {score_resp.json().get('detail', 'Unknown error')}"

        # 3. Fetch Explanation
        explain_resp = requests.get(f"{API_BASE_URL}/explain/{msme_id}?consent_token={token}", timeout=30.0)
        if explain_resp.status_code != 200:
            return None, None, f"Explainability Engine Error: {explain_resp.json().get('detail', 'Unknown error')}"

        return score_resp.json(), explain_resp.json(), None

    except requests.exceptions.ConnectionError:
        return None, None, "Connection Error: Cannot reach the backend API."
    except Exception as e:
        return None, None, f"Unexpected Error: {str(e)}"


# ---------------------------------------------------------------------------
# Main Application Render
# ---------------------------------------------------------------------------

def main() -> None:
    # Custom CSS and typography injection to match the React frontend
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        html, body, [data-testid="stAppViewContainer"], .main {
            background-color: #F6F2EC !important;
            font-family: 'Inter', sans-serif !important;
            color: #17181C !important;
        }
        
        h1, h2, h3, h4, h5, h6, [class*="css-"] h1, [class*="css-"] h2 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #17181C !important;
            font-weight: 600 !important;
        }

        /* Sidebar Styling (Fixed: removed universal * selector to prevent breaking icon fonts) */
        [data-testid="stSidebar"] {
            background-color: #FDFCFA !important;
            border-right: 1px solid rgba(23,24,28,0.08) !important;
            color: #17181C !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #17181C !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] caption {
            font-family: 'Inter', sans-serif !important;
            color: #17181C !important;
        }
        
        /* Hide Streamlit default components to make it look like a custom enterprise portal */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header[data-testid="stHeader"] {display: none !important;}
        div[data-testid="stDecoration"] {display: none !important;}
        button[title="View source code"] {display: none !important;}
        .stDeployButton {display: none !important;}
        
        /* Bento Card styling */
        div[data-testid="column"] {
            background-color: #FDFCFA !important;
            border: 1px solid rgba(23,24,28,0.08) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 1px 2px rgba(23,24,28,0.04), 0 8px 24px rgba(23,24,28,0.06) !important;
            margin-bottom: 20px !important;
        }
        
        /* Prevent double nesting borders inside columns/widgets */
        div[data-testid="column"] div[data-testid="column"],
        div.stBlock div.stBlock,
        div.stBlock div[data-testid="column"],
        div[data-testid="column"] div.stBlock {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin-bottom: 0 !important;
        }
        
        /* Metrics styling */
        div[data-testid="stMetricValue"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #17181C !important;
        }
        div[data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #6B6A66 !important;
            margin-bottom: 8px !important;
        }
        
        /* Custom navbar styling */
        .custom-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 24px;
            background-color: #FDFCFA;
            border: 1px solid rgba(23,24,28,0.08);
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 1px 2px rgba(23,24,28,0.04), 0 4px 16px rgba(23,24,28,0.03);
        }
        .navbar-logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .navbar-logo {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background-color: #17181C;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #F6F2EC;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 20px;
        }
        .navbar-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 20px;
            color: #17181C;
        }
        .navbar-status {
            font-size: 12px;
            font-weight: 600;
            color: #4CAF7D;
            background-color: rgba(76, 175, 125, 0.1);
            padding: 6px 14px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'Inter', sans-serif;
        }
        .navbar-status-dot {
            width: 6px;
            height: 6px;
            background-color: #4CAF7D;
            border-radius: 50%;
            display: inline-block;
        }
        .animate-pulse-slow {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: .4; }
        }

        /* Divider styling */
        hr {
            border: 0 !important;
            height: 1px !important;
            background-color: rgba(23,24,28,0.08) !important;
            margin: 24px 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render custom Header Navbar
    st.markdown(
        """
        <div class="custom-navbar">
            <div class="navbar-logo-group">
                <div class="navbar-logo">M</div>
                <div class="navbar-brand">MSME Health Card Portal</div>
            </div>
            <div class="navbar-status">
                <span class="navbar-status-dot animate-pulse-slow"></span>
                ULI Underwriting Console Connected
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Liveness check — Render free tier may be cold-starting; show spinner while waiting
    with st.spinner("🔄 Connecting to credit scoring engine… (may take up to 30 s on first load)"):
        backend_active = check_backend_liveness(timeout=30.0)

    if not backend_active:
        st.error(
            "🚨 **Credit Scoring Backend Unreachable**  \n"
            f"Attempted to connect to: `{API_BASE_URL}`"
        )
        st.warning(
            "The backend may still be waking up (Render free tier spins down after inactivity). "
            "**Please wait 30 seconds and click Retry.**"
        )
        col_retry, _ = st.columns([1, 3])
        with col_retry:
            if st.button("🔄 Retry Connection", type="primary", use_container_width=True):
                st.rerun()
        st.info(
            "**Local development?** Set the `API_BASE_URL` environment variable and run:\n"
            "```bash\n"
            "uvicorn src.api.main:app --port 8000 --reload\n"
            "```"
        )
        st.stop()

    # 2. Load Cohort
    grouped_cohort = load_msme_cohort()
    if not grouped_cohort:
        st.warning("⚠️ No synthetic MSME cohort found. Please run the generators to build the synthetic database first.")
        st.stop()

    # -----------------------------------------------------------------------
    # Sidebar: MSME Selector
    # -----------------------------------------------------------------------
    st.sidebar.header("Underwriting Filter")
    
    # Selection 1: Group / Persona
    persona_choice = st.sidebar.selectbox(
        "Filter by Credit Persona",
        options=list(PERSONA_LABELS.keys()),
        format_func=lambda x: PERSONA_LABELS[x]
    )

    # Description of Selected Persona
    st.sidebar.caption(f"**About Persona:** {PERSONA_DESC[persona_choice]}")

    # Selection 2: Specific MSME
    msme_list = grouped_cohort[persona_choice]
    if not msme_list:
        st.sidebar.warning("No MSMEs generated for this persona.")
        st.stop()

    selected_index = st.sidebar.selectbox(
        "Select Active Applicant",
        options=range(len(msme_list)),
        format_func=lambda i: f"{msme_list[i][0]} — {msme_list[i][1]}"
    )
    
    selected_msme_id, selected_name, selected_sector = msme_list[selected_index]

    # Display Metadata in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Sector:** `{selected_sector}`")
    st.sidebar.markdown(f"**ID:** `{selected_msme_id}`")
    st.sidebar.markdown("**AA Consent Status:** `Authorized (Sahamati v1.1)`")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Risk Covenants & Mitigants")
    cov_guarantee = st.sidebar.checkbox("Promoter Guarantee (+5 pts)", value=False)
    cov_escrow = st.sidebar.checkbox("Escrow 20% UPI Inflow (+8 pts)", value=False)
    cov_gst_lock = st.sidebar.checkbox("GST Filing Lock (+4 pts)", value=False)

    # Fetch Data
    with st.spinner("Executing real-time ULI assessment..."):
        score_data, explain_data, error_msg = get_credit_score_and_explain(selected_msme_id)

    if error_msg or not score_data or not explain_data:
        st.error(f"Failed to fetch credit records: {error_msg}")
        st.stop()

    # -----------------------------------------------------------------------
    # Main Dashboard Tabs
    # -----------------------------------------------------------------------
    tab_dashboard, tab_payloads = st.tabs(["💳 Underwriter View", "🛠️ Integration Payload Inspector"])
    
    with tab_dashboard:
        # -----------------------------------------------------------------------
        # Main Panel, Top: Overall Score & Risk Badge
        # -----------------------------------------------------------------------
        col1, col2, col3 = st.columns([2, 1, 1.5])

        overall_base = score_data["overall_score"]
        covenant_boost = 0.0
        if cov_guarantee:
            covenant_boost += 5.0
        if cov_escrow:
            covenant_boost += 8.0
        if cov_gst_lock:
            covenant_boost += 4.0
        overall = min(100.0, overall_base + covenant_boost)
        
        # Determine risk band
        if overall >= 80.0:
            risk_label = "LOW RISK"
            risk_color = "#4CAF7D"
            risk_emoji = "🟢"
        elif overall >= 60.0:
            risk_label = "MEDIUM RISK"
            risk_color = "#F2916B"
            risk_emoji = "🟡"
        else:
            risk_label = "HIGH RISK"
            risk_color = "#EB5757"
            risk_emoji = "🔴"

        with col1:
            st.markdown(
                f"""
                <div style='margin-bottom:0;'>
                    <h1 style='font-family:"Space Grotesk", sans-serif; font-size: 1.85rem; font-weight: 700; color:#17181C; margin: 0;'>
                        {selected_name}
                    </h1>
                    <div style='font-family:"Inter", sans-serif; font-size: 12px; color:#6B6A66; margin-top: 6px;'>
                        ULI Reference: <b>{selected_msme_id}</b> | Core Sector: <b>{selected_sector}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.metric(
                label="Blended Credit Score",
                value=f"{overall:.1f} / 100",
                help="Integrated score blending Rule-Based features and XGBoost Probability of Default model."
            )

        with col3:
            st.markdown(
                f"<div style='background-color:{risk_color}; color:white; font-size:18px; "
                f"font-weight:bold; text-align:center; padding:10px; border-radius:12px; margin-top:15px; "
                f"font-family:\"Space Grotesk\", sans-serif; letter-spacing:0.05em;'> "
                f"{risk_emoji} {risk_label}"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='text-align:center; font-size:12px; margin-top:10px; color:#6B6A66; font-family:\"Inter\", sans-serif;'>"
                f"Assessment Confidence: <b>{score_data['confidence_band']} Band</b> ({score_data['confidence_score']:.2f})"
                f"</div>",
                unsafe_allow_html=True
            )

        # -----------------------------------------------------------------------
        # Main Panel, Middle: Radar Chart
        # -----------------------------------------------------------------------
        chart_col = st.columns(1)[0]
        with chart_col:
            st.markdown("<h3 style='margin-top:0; margin-bottom:15px; font-family:\"Space Grotesk\", sans-serif;'>📊 Multidimensional Credit Dimension Analysis</h3>", unsafe_allow_html=True)
            
            dim_scores = score_data["dimension_scores"]
            categories = ["Stability", "Cash Flow", "Compliance", "Growth", "Repayment"]
            scores_list = [
                dim_scores.get("stability"),
                dim_scores.get("cashflow"),
                dim_scores.get("compliance"),
                dim_scores.get("growth"),
                dim_scores.get("repayment")
            ]

            # Handle missing dimensions for Radar chart
            active_dims = [(cat, val) for cat, val in zip(categories, scores_list) if val is not None]
            
            if len(active_dims) >= 3:
                # Render Plotly Radar Chart
                theta = [item[0] for item in active_dims]
                r = [item[1] for item in active_dims]
                
                # Close the loop for radar chart
                theta.append(theta[0])
                r.append(r[0])

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta,
                    fill='toself',
                    name='Dimension Score',
                    line_color='#A99EF2',
                    fillcolor='rgba(169, 158, 242, 0.25)',
                    marker=dict(color='#A99EF2', size=6)
                ))

                fig.update_layout(
                    polar=dict(
                        bgcolor='#FDFCFA',
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            tickfont=dict(size=10, family='Inter', color='#6B6A66'),
                            gridcolor='rgba(23, 24, 28, 0.08)',
                            linecolor='rgba(23, 24, 28, 0.08)'
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=11, family='Space Grotesk', color='#17181C', weight='bold'),
                            gridcolor='rgba(23, 24, 28, 0.08)'
                        )
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=30, b=30),
                    height=340
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to Bar Chart if thin profile (fewer than 3 dimensions scored)
                st.info("⚠️ Thin financial profile. Displaying bar analysis for active dimensions.")
                
                categories_filtered = [item[0] for item in active_dims]
                scores_filtered = [item[1] for item in active_dims]

                fig = go.Figure([go.Bar(
                    x=categories_filtered,
                    y=scores_filtered,
                    marker_color='#A99EF2',
                    width=0.4
                )])
                fig.update_layout(
                    yaxis=dict(
                        range=[0, 100], 
                        title="Dimension Score",
                        titlefont=dict(family='Space Grotesk', color='#17181C'),
                        tickfont=dict(family='Inter', color='#6B6A66'),
                        gridcolor='rgba(23, 24, 28, 0.08)'
                    ),
                    xaxis=dict(
                        title="Active Alternate Data Dimensions",
                        titlefont=dict(family='Space Grotesk', color='#17181C'),
                        tickfont=dict(family='Inter', color='#6B6A66'),
                        gridcolor='rgba(23, 24, 28, 0.08)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=280
                )
                st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------------
        # Main Panel, Below: Explainability columns (SHAP reasons)
        # -----------------------------------------------------------------------
        st.markdown("<h3 style='margin-top:10px; margin-bottom:20px; font-family:\"Space Grotesk\", sans-serif;'>🔍 Explainable AI (XAI) Underwriting Notes</h3>", unsafe_allow_html=True)
        
        col_str, col_risk = st.columns(2)

        strengths = explain_data.get("top_strengths", [])
        risks = explain_data.get("top_risks", [])

        with col_str:
            st.markdown(
                f"""
                <div style='background-color:rgba(76, 175, 125, 0.05); padding:20px; border-radius:12px; 
                border: 1px solid rgba(76, 175, 125, 0.15); border-left: 5px solid #4CAF7D; min-height: 220px;'>
                    <h4 style='color:#4CAF7D; margin-top:0; font-family:"Space Grotesk", sans-serif; font-weight:600; display:flex; align-items:center; gap:8px;'>
                        ✨ Key Credit Strengths
                    </h4>
                """,
                unsafe_allow_html=True
            )
            if strengths:
                for i, item in enumerate(strengths, 1):
                    st.markdown(f"<div style='margin-bottom:8px; font-size:13.5px; font-family:\"Inter\", sans-serif;'><b>{i}.</b> {item['description']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#6B6A66; font-size:13.5px;'>No major credit strengths identified.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_risk:
            st.markdown(
                f"""
                <div style='background-color:rgba(242, 145, 107, 0.05); padding:20px; border-radius:12px; 
                border: 1px solid rgba(242, 145, 107, 0.15); border-left: 5px solid #F2916B; min-height: 220px;'>
                    <h4 style='color:#F2916B; margin-top:0; font-family:"Space Grotesk", sans-serif; font-weight:600; display:flex; align-items:center; gap:8px;'>
                        ⚠️ Primary Risk Factors
                    </h4>
                """,
                unsafe_allow_html=True
            )
            if risks:
                for i, item in enumerate(risks, 1):
                    st.markdown(f"<div style='margin-bottom:8px; font-size:13.5px; font-family:\"Inter\", sans-serif;'><b>{i}.</b> {item['description']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#6B6A66; font-size:13.5px;'>No major risk factors flagged.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------
        # Below: Data Sources Used
        # -----------------------------------------------------------------------
        st.markdown("<h3 style='margin-top:10px; margin-bottom:20px; font-family:\"Space Grotesk\", sans-serif;'>🔌 Alternate Data Completeness & AA Integrations</h3>", unsafe_allow_html=True)
        
        sources_used = score_data["sources_present"]
        all_sources = {
            "gst": "GST Returns (GSTR-1/3B)",
            "upi": "UPI Merchant Transactions",
            "aa": "Account Aggregator (Bank Statement)",
            "epfo": "EPFO Payroll Contributions"
        }

        cols = st.columns(4)
        for idx, (code, title) in enumerate(all_sources.items()):
            is_used = code in sources_used
            with cols[idx]:
                if is_used:
                    st.markdown(
                        f"""
                        <div style='background-color:#FDFCFA; border: 1px solid rgba(76, 175, 125, 0.2); 
                        border-top: 4px solid #4CAF7D; border-radius: 12px; padding: 16px; text-align: center;
                        box-shadow: 0 1px 2px rgba(23,24,28,0.02); height: 100%;'>
                            <div style='color:#4CAF7D; font-size: 20px; margin-bottom: 6px;'>✅</div>
                            <div style='font-family:"Space Grotesk", sans-serif; font-weight: 600; font-size: 13px; color:#17181C;'>{title}</div>
                            <div style='font-family:"Inter", sans-serif; font-size: 11px; color:#4CAF7D; margin-top: 4px; font-weight: 500;'>CONNECTED</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='background-color:#FDFCFA; border: 1px solid rgba(235, 87, 87, 0.2); 
                        border-top: 4px solid #EB5757; border-radius: 12px; padding: 16px; text-align: center;
                        box-shadow: 0 1px 2px rgba(23,24,28,0.02); height: 100%;'>
                            <div style='color:#EB5757; font-size: 20px; margin-bottom: 6px;'>❌</div>
                            <div style='font-family:"Space Grotesk", sans-serif; font-weight: 600; font-size: 13px; color:#17181C;'>{title}</div>
                            <div style='font-family:"Inter", sans-serif; font-size: 11px; color:#EB5757; margin-top: 4px; font-weight: 500;'>NOT CONNECTED</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        comp_score = score_data["completeness_score"]
        info_col = st.columns(1)[0]
        with info_col:
            st.markdown(
                f"""
                <div style='font-family:"Inter", sans-serif; font-size: 13.5px; color:#6B6A66; line-height: 1.6;'>
                    💡 <b>Profile Data Completeness:</b> <span style='color:#17181C; font-weight:600;'>{comp_score * 100:.0f}%</span> of requested sources present.
                    The composite scoring engine applied a <span style='color:#17181C; font-weight:600;'>{score_data['blend_ratio_used']['rule_based'] * 100:.0f}% Rule-Based / {score_data['blend_ratio_used']['ml'] * 100:.0f}% Machine Learning</span> blend ratio for score computation.
                </div>
                """,
                unsafe_allow_html=True
            )

        # -----------------------------------------------------------------------
        # Bottom: Recommended Action Panel
        # -----------------------------------------------------------------------
        action_col = st.columns(1)[0]
        with action_col:
            st.markdown("<h3 style='margin-top:0; margin-bottom:15px; font-family:\"Space Grotesk\", sans-serif;'>💡 Recommended Credit Action</h3>", unsafe_allow_html=True)
            
            if risk_label == "LOW RISK":
                action_title = "APPROVED (Standard Terms)"
                action_bg = "rgba(76, 175, 125, 0.05)"
                action_border = "#4CAF7D"
                action_text_color = "#4CAF7D"
                action_text = (
                    "<li><b>Suggested Credit Limit:</b> INR 5,00,000</li>"
                    "<li><b>Risk-Based Pricing Interest Rate:</b> 10.5% - 11.5% p.a.</li>"
                    "<li><b>Lending Route:</b> STP (Straight-Through-Processing) via OCEN APIs.</li>"
                    "<li><b>Underwriter Note:</b> Strong cash reserves, clean compliance records, and robust repayment history. Low default probability.</li>"
                )
            elif risk_label == "MEDIUM RISK":
                action_title = "REFER TO CREDIT OFFICER (Manual Assessment Required)"
                action_bg = "rgba(242, 145, 107, 0.05)"
                action_border = "#F2916B"
                action_text_color = "#F2916B"
                action_text = (
                    "<li><b>Suggested Credit Limit:</b> INR 2,00,000</li>"
                    "<li><b>Risk-Based Pricing Interest Rate:</b> 13.5% - 14.5% p.a.</li>"
                    "<li><b>Lending Route:</b> Manual Underwriter review for collateral or guarantor requirement.</li>"
                    "<li><b>Underwriter Note:</b> Solid operational turnover but shows minor liquidity pressure or compliance delays. Risk mitigatable with covenants.</li>"
                )
            else:
                action_title = "DECLINE (Application Rejected)"
                action_bg = "rgba(235, 87, 87, 0.05)"
                action_border = "#EB5757"
                action_text_color = "#EB5757"
                action_text = (
                    "<li><b>Credit Decision:</b> Application Rejected.</li>"
                    "<li><b>Reason:</b> Substantial operational risk detected. High default probability predicted due to consecutive overdrafts, filing mismatches, or high returns.</li>"
                    "<li><b>Re-application Period:</b> 180 Days, contingent on regular tax filings and clean bank logs.</li>"
                )

            st.markdown(
                f"""
                <div style='background-color:{action_bg}; padding:24px; border-radius:16px; 
                border: 2px solid {action_border}; font-family:"Inter", sans-serif; color:#17181C;'>
                    <h4 style='color:{action_text_color}; margin-top:0; font-family:"Space Grotesk", sans-serif; font-size:16px; font-weight:700;'>
                        ⚖️ Decision: {action_title}
                    </h4>
                    <ul style='margin-bottom:0; padding-left:20px; line-height: 1.6; font-size: 13.5px;'>
                        {action_text}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            # OCEN Bid Matching Engine (Only for low/medium risk)
            if risk_label != "HIGH RISK":
                st.markdown("<h4 style='margin-top:24px; margin-bottom:15px; font-family:\"Space Grotesk\", sans-serif;'>🏛️ OCEN v4 Multi-Lender Bid Matching Engine</h4>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style='font-family:"Inter", sans-serif; font-size:12.5px; color:#6B6A66; margin-bottom:12px;'>
                        Lenders on the ULI-OCEN network have returned the following competitive bids based on the borrower's risk profile and covenants:
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Dynamic rates based on low vs medium risk
                if risk_label == "LOW RISK":
                    bids = [
                        {"lender": "State Bank of India (SBI)", "interest": "10.45% p.a.", "fee": "0.4%", "mode": "Instant STP Disbursal", "action": "Accept SBI Offer"},
                        {"lender": "HDFC Bank Ltd", "interest": "10.85% p.a.", "fee": "0.25%", "mode": "Instant STP Disbursal", "action": "Accept HDFC Offer"},
                        {"lender": "ICICI Bank Ltd", "interest": "10.60% p.a.", "fee": "0.3%", "mode": "LSP Escrow Lock Required", "action": "Accept ICICI Offer"}
                    ]
                else:
                    bids = [
                        {"lender": "State Bank of India (SBI)", "interest": "13.25% p.a.", "fee": "0.75%", "mode": "Manual Escrow Lock", "action": "Accept SBI Offer"},
                        {"lender": "HDFC Bank Ltd", "interest": "13.75% p.a.", "fee": "0.5%", "mode": "Guarantor Sign-off Required", "action": "Accept HDFC Offer"},
                        {"lender": "ICICI Bank Ltd", "interest": "13.50% p.a.", "fee": "0.6%", "mode": "Escrow Lock Required", "action": "Accept ICICI Offer"}
                    ]
                
                # Show columns for each lender offer
                lender_cols = st.columns(3)
                for idx, bid in enumerate(bids):
                    with lender_cols[idx]:
                        st.markdown(
                            f"""
                            <div style='background-color:#FDFCFA; border: 1px solid rgba(23,24,28,0.08); 
                            border-radius:12px; padding:16px; min-height:160px; box-shadow:0 1px 2px rgba(0,0,0,0.02);'>
                                <div style='font-family:"Space Grotesk", sans-serif; font-weight:700; font-size:13.5px; color:#17181C; margin-bottom:8px;'>
                                    {bid['lender']}
                                </div>
                                <div style='font-family:"Inter", sans-serif; font-size:12px; margin-bottom:4px; color:#6B6A66;'>
                                    Rate: <b style='color:#17181C;'>{bid['interest']}</b>
                                </div>
                                <div style='font-family:"Inter", sans-serif; font-size:12px; margin-bottom:4px; color:#6B6A66;'>
                                    Processing Fee: <b style='color:#17181C;'>{bid['fee']}</b>
                                </div>
                                <div style='font-family:"Inter", sans-serif; font-size:11px; margin-bottom:12px; color:#A99EF2; font-weight:600;'>
                                    ⚡ {bid['mode']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        # We use streamlit's key to identify selected bid
                        if st.button(bid['action'], key=f"btn_bid_{idx}_{selected_msme_id}"):
                            st.success(f"🎉 **OCEN Match Success!** Locked loan offer with **{bid['lender']}** at **{bid['interest']}**.")

    with tab_payloads:
        import uuid
        from datetime import datetime, timedelta
        
        st.markdown("<h3 style='margin-top:0; margin-bottom:15px; font-family:\"Space Grotesk\", sans-serif;'>🛠️ ULI / OCEN / Account Aggregator Schema Logs</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='font-family:"Inter", sans-serif; font-size: 13.5px; color:#6B6A66; margin-bottom:20px; line-height: 1.5;'>
                Explore the standardized data payloads and public infrastructure API exchange schemas generated by this underwriting loop.
                This console simulates compliance with India's <b>Unified Lending Interface (ULI)</b>, <b>OCEN v4</b> protocols, and the <b>ReBIT consent framework</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        covenants_applied = []
        if cov_guarantee:
            covenants_applied.append("PROMOTER_PERSONAL_GUARANTEE")
        if cov_escrow:
            covenants_applied.append("UPI_REVENUE_ESCROW_20")
        if cov_gst_lock:
            covenants_applied.append("GST_FILING_STRICT_LOCK")

        payload_tabs = st.tabs([
            "🔐 ReBIT AA Consent Request", 
            "📂 ULI Ingested Alternate Data", 
            "🔌 OCEN Application Payload", 
            "🚀 Live Backend API Response"
        ])
        
        with payload_tabs[0]:
            st.markdown("##### Sahamati Account Aggregator (AA) Consent Request (ReBIT v1.1.2 Schema)")
            st.caption("The consent parameters created during onboarding to authorize alternate data extraction:")
            
            # Structured Table Explanation
            st.markdown(
                """
                | Key Parameter | Payload Value | Description / Standard |
                | :--- | :--- | :--- |
                | **Version** | `1.1.2` | ReBIT API version identifier. |
                | **Transaction ID** | `txn-aa89c2c0-827d-...` | Unique UUID tracking this data sharing session. |
                | **Consent Mode** | `VIEW` | Data consumer access rights (Read-only view). |
                | **Customer Identifier** | `msme_XXXXXX@sahamati` | Unique consent handle for the borrower. |
                | **Purpose Code** | `101` (ref: ULI-MSME-CREDIT-EVAL) | RBI purpose code designating **Credit Assessment**. |
                | **Data Range** | 12 Months | Historic data range requested (Deposit & Term Deposit). |
                """
            )
            
            aa_consent_req = {
                "ver": "1.1.2",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "txnid": f"txn-{uuid.uuid4()}",
                "ConsentDetail": {
                    "consentStart": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "consentExpiry": (datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "consentMode": "VIEW",
                    "fetchType": "ONETIME",
                    "consentTypes": ["TRANSACTIONS", "PROFILE"],
                    "fiTypes": ["DEPOSIT", "TERM_DEPOSIT"],
                    "DataConsumer": {
                        "id": "FIU-ULI-MSME-DECISION-ENGINE"
                    },
                    "Customer": {
                        "id": f"{selected_msme_id.lower()}@sahamati-sandbox"
                    },
                    "Purpose": {
                        "code": "101",
                        "ref": "ULI-MSME-CREDIT-EVAL",
                        "text": "Alternate-data credit card scoring and underwriting for MSME loan request."
                    },
                    "FIDataRange": {
                        "from": "2025-07-01T00:00:00Z",
                        "to": "2026-06-30T23:59:59Z"
                    },
                    "DataLife": {
                        "unit": "MONTH",
                        "value": 12
                    },
                    "Frequency": {
                        "unit": "HOUR",
                        "value": 1
                    }
                }
            }
            with st.expander("🔍 View Raw JSON Payload Structure", expanded=True):
                st.json(aa_consent_req)

        with payload_tabs[1]:
            st.markdown("##### ULI Alternate Data Exchange Payload (Pydantic Validated)")
            st.caption("Standardized alternate metrics collected and validated by the ingestion service:")
            
            # Structured Table Explanation
            st.markdown(
                """
                | Alternate Data Source | Ingested Status | Variables Extracted | Underwriting Significance |
                | :--- | :--- | :--- | :--- |
                | **GST Returns (GSTR-3B)** | `FILED_ON_TIME` | Monthly sales turnover, filing delays. | Used to evaluate **Stability** & **Growth** dimensions. |
                | **UPI Merchant Stream** | `CONNECTED` | Average daily settlements, VPA, refund rate. | Used to evaluate **Cash Flow** & liquidity volatility. |
                | **Account Aggregator** | `AUTHORIZED` | Bank transaction logs, overdraft frequency. | Used to evaluate **Repayment** & liquidity buffer. |
                | **EPFO Payroll** | `VERIFIED` | Active member count, monthly contributions. | Evaluates organizational scale and employer stability. |
                """
            )
            
            uli_payload = {
                "msme_id": selected_msme_id,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sources_ingested": score_data["sources_present"],
                "data_completeness": f"{score_data['completeness_score'] * 100:.0f}%",
                "extracted_features": {
                    "stability_dimension": {
                        "score": score_data["dimension_scores"].get("stability"),
                        "features": {
                            "months_since_registration": 24 if "established" in persona_choice else 8,
                            "gst_filing_consistency_ratio": 1.0 if "healthy" in persona_choice else 0.75
                        }
                    },
                    "cashflow_dimension": {
                        "score": score_data["dimension_scores"].get("cashflow"),
                        "features": {
                            "average_daily_upi_inflow_inr": 4250.0,
                            "debt_service_coverage_ratio": 2.1 if "healthy" in persona_choice else 1.1
                        }
                    },
                    "compliance_dimension": {
                        "score": score_data["dimension_scores"].get("compliance")
                    }
                }
            }
            with st.expander("🔍 View Raw JSON Payload Structure", expanded=True):
                st.json(uli_payload)

        with payload_tabs[2]:
            st.markdown("##### OCEN v4 Loan Application & LSP Push Payload")
            st.caption("Standardized schema dispatched to Lender Network partner on ULI approval:")
            
            # Structured Table Explanation
            amount_str = "₹5,00,000" if risk_label == "LOW RISK" else "₹2,00,000" if risk_label == "MEDIUM RISK" else "Declined"
            rate_str = "11.5% p.a." if risk_label == "LOW RISK" else "14.5% p.a." if risk_label == "MEDIUM RISK" else "N/A"
            st.markdown(
                f"""
                | Terms Parameter | Value Offered | Description / OCEN Standard |
                | :--- | :--- | :--- |
                | **Loan Application ID** | `ocen-app-{selected_msme_id.lower()}-...` | Tracking ID for the loan request. |
                | **LSP ID** | `LSP-MSME-DECISION-ENGINE` | Lending Service Provider aggregator handle. |
                | **Lender ID** | `BANK-LENDER-ULI-SANDBOX` | Network bank underwriting the disbursement. |
                | **Approved Amount** | **{amount_str}** | Approved working capital credit limit. |
                | **Interest Rate** | **{rate_str}** | Risk-based yearly percentage rate. |
                | **Tenor** | `12 Months` | Repayment window in monthly cycles. |
                """
            )
            
            ocen_payload = {
                "ver": "4.0.0",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "loan_application_id": f"ocen-app-{selected_msme_id.lower()}-{uuid.uuid4().hex[:6]}",
                "lsp_identifier": "LSP-MSME-DECISION-ENGINE",
                "lender_identifier": "BANK-LENDER-ULI-SANDBOX",
                "borrower_consent": {
                    "consent_ref_id": f"consent-ref-{selected_msme_id.lower()}",
                    "consent_status": "AUTHORIZED"
                },
                "loan_terms_offered": {
                    "amount": 500000.0 if risk_label == "LOW RISK" else 200000.0 if risk_label == "MEDIUM RISK" else 0.0,
                    "currency": "INR",
                    "tenor_months": 12,
                    "interest_rate_pa": 11.5 if risk_label == "LOW RISK" else 14.5,
                    "repayment_frequency": "MONTHLY",
                    "covenantTerms": covenants_applied
                }
            }
            with st.expander("🔍 View Raw JSON Payload Structure", expanded=True):
                st.json(ocen_payload)

        with payload_tabs[3]:
            st.markdown("##### Live Decision API Response (Score & SHAP Explainer JSON)")
            st.caption("The raw JSON returned by the scoring and explainability engine:")
            
            # Structured Table Explanation
            st.markdown(
                f"""
                | Metric Dimension | Value / Score | Underwriter Classification |
                | :--- | :--- | :--- |
                | **Blended Credit Score** | **{overall:.1f} / 100** | Blended Rules + ML model risk assessment. |
                | **Risk Classification** | **{risk_label}** | Based on risk scoring thresholds. |
                | **Confidence Rating** | **{score_data['confidence_band']} Band** ({score_data['confidence_score']:.2f}) | Reflects depth of data sources present. |
                | **Data Completeness** | **{score_data['completeness_score'] * 100:.0f}%** | Ingested alternate sources ratio. |
                | **ML Weight** | **{score_data['blend_ratio_used']['ml'] * 100:.0f}%** | Weight applied to XGBoost probability of default model. |
                """
            )
            
            score_data_copy = dict(score_data)
            score_data_copy["overall_score"] = overall
            score_data_copy["covenants_applied"] = covenants_applied

            live_api_response = {
                "scoring_engine": score_data_copy,
                "explainability_engine": explain_data
            }
            with st.expander("🔍 View Raw JSON Payload Structure", expanded=True):
                st.json(live_api_response)


if __name__ == "__main__":
    main()
