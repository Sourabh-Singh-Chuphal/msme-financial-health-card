# Known Limitations

This document lists the simulated vs. production-ready areas of the MSME Financial Health Card repository. Being upfront about these boundaries demonstrates engineering maturity and helps you navigate queries from judges.

---

## 1. Data Layer

### Simulated
* **Synthetic Cohort**: All 200 MSME profiles are synthetic datasets generated locally. While they accurately model specific credit personas (seasonal, declining, volatile, healthy), they are not sourced from real-world MSME businesses.
* **Absence of Bureau Records**: The current ingestion pipeline only uses alternate data (GST, UPI, EPFO, bank statement). In a real bank credit environment, these alternate data scores would be combined with a traditional Bureau Score (e.g., CIBIL or CMR) when available.

### Production-Ready
* **Pydantic Validation Schemas**: Raw source validators strictly enforce schemas and drop malformed rows, generating clean warning counts.
* **Dimension Feature Engine**: Seasonality adjustments (ratio-to-baseline deseasonalization) are fully implemented and tested.

---

## 2. Machine Learning Model

### Simulated
* **Default Label Simulation**: Since we do not have historical loan default history for our synthetic merchants, default labels (`y=0` or `y=1`) are simulated probabilistically using a feature-based log-odds model with added noise to avoid leakage.
* **Small Sample Size**: The model is trained on a 160-sample split of our synthetic cohort. A production credit risk model would require training on tens of thousands of real historical loan records.

### Production-Ready
* **Tree-SHAP Explainability**: The SHAP explainer extracts true mathematical log-odds contributions from the XGBoost booster, enforcing the additive sum constraint ($\sum s_i = \text{margin} - \text{base\_value}$).

---

## 3. Ecosystem Integrations (AA / OCEN / ULI)

### Simulated
* **Sahamati AA Gateway**: We mock the AA gateway consent authorization. Live gateways require a registered Financial Information User (FIU) license, certificate exchanges, and digital signature validation.
* **ULI / OCEN APIs**: The adapters construct illustrative payloads modeled on OCEN v4 and PTPL (Public Tech Platform for Frictionless Lending) structures. We do not make external HTTP network transfers to live public sandboxes.

### Production-Ready
* **Consent-Bound Endpoints**: Routes require token validation and reject requests with expired tokens or mismatched client IDs.
* **Pydantic Adapter Schemas**: Mapped payloads conform to standard schema models, ready for serialization and transport.
