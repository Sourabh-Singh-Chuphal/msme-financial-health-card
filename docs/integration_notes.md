# Ecosystem Integration Notes (AA / OCEN / ULI)

This document provides a plain-language summary of how the MSME Financial Health Card integrates with standard Indian digital public infrastructure (DPI) lending networks. It clarifies what is fully functional in this implementation versus what is simulated/illustrative to help you answer questions confidently during judge evaluations.

---

## 1. Account Aggregator (AA) Consent Flow

### Real Capabilities
* **ReBIT Standard Schema**: The [`ConsentArtifact`](file:///c:/Users/soura/Projects/msme-financial-health-card/src/integration/aa_consent_flow.py) is implemented using Pydantic models mapping directly to ReBIT (Reserve Bank Information Technology) / Sahamati standard JSON schemas.
* **Purpose Code Mapping**: Consent requests utilize ReBIT standard code `102` which is the official designated code for **"Credit Assessment"** (preventing data access for non-credit activities).
* **Token Validation**: The API scoring routes strictly enforce consent token verification: validating matching MSME IDs, expiration periods, and required data types.

### Simulated / Mocked Areas
* **Cryptographic Signatures**: In production, the Account Aggregator digitally signs the consent artifact using public/private key pairs. We mock this signature as a static string `MOCK_DIGITAL_SIGNATURE_OF_ACCOUNT_AGGREGATOR`.
* **Sahamati Gateway Handshake**: Instead of communicating with an active AA Gateway (which requires a registered Financial Information User [FIU] license and public certificate exchanges), we mock the consent handshake in memory by registering a `consent_token` mapped to the MSME ID.

---

## 2. Open Credit Enablement Network (OCEN) Adapter

### Real Capabilities
* **LSP Loan Referral Payload**: The [`OCENLoanReferralPayload`](file:///c:/Users/soura/Projects/msme-financial-health-card/src/integration/ocen_adapter.py) replicates the structure of a Loan Service Provider (LSP) handoff object transmitting computed credit metrics and suggested loan terms to a lender.
* **Spec Alignment**: Modeled on the OCEN v4 schema guidelines, transferring data like `borrowerId`, `healthScoreSummary`, and `recommendedTerms`.
* **Dynamic Terms Inference**: Recommended terms (credit limits, interest rate caps, and tenor periods) are dynamically computed from our composite scorer's risk bands (Prime vs. Standard vs. Decline).

### Simulated / Mocked Areas
* **Gateway Transmission**: The `push_to_lsp()` function logs the serialized JSON payload and returns a simulated transaction receipt ID (`TXN-OCEN-XXX`). It does not make external HTTP calls to an actual OCEN gateway, as these require closed-network banking credentials.

---

## 3. Unified Lending Interface (ULI) Adapter

### Real Capabilities
* **Data Aggregation**: The [`ULIMerchantHealthProfile`](file:///c:/Users/soura/Projects/msme-financial-health-card/src/integration/uli_adapter.py) aggregates data points across ULI's standard alternate data sources (GST, UPI, EPFO, and bank statements) alongside the composite health score.
* **DPI Philosophy**: Aligns with RBI's Public Tech Platform for Frictionless Lending (PTPFL) goal: allowing lenders to pull cross-domain data via a single transaction endpoint.

### Simulated / Mocked Areas
* **JSON Structure Schema**: Because ULI specification documentation is primarily focused on high-level connection architectures (governed by the RBI Innovation Hub) rather than a fixed JSON data schema, our exchange structure is **illustrative**. It represents a standard design pattern for ULI integration, but the exact keys are not bound to a rigid official spec.

---

## 4. Integration Verification Summary

If a judge asks: **"Is this system actually integrated with AA, OCEN, and ULI?"**
> **"Our data schemas and consent handshakes are fully modeled on the ReBIT/Sahamati and OCEN specification architectures. We have implemented standard Pydantic models for the ReBIT consent artifact and the OCEN v4 loan referral payload. Because live integrations require formal regulatory licenses (FIU/FIP registrations) and certificate exchanges, the network gateways themselves are simulated in our integration layer. The code is structurally ready to plug into live gateways once API keys and certificates are provided."**
