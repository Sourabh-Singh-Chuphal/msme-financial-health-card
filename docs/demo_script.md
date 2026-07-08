# 3-Minute Live Demo Walkthrough Script

This script guides you step-by-step through a 3-minute hackathon presentation of the **MSME Financial Health Card**.

---

## Part 1: The Problem (0 - 30 seconds)
**[Action: Show Title Slide or Landing Page]**

> *"Traditional credit scoring shuts out viable MSME borrowers because they lack formal credit logs and tax documents, causing high rejection rates for New-to-Bank and New-to-Credit enterprises. We solve this by aggregating rich alternate digital data—GST filings, UPI merchant streams, EPFO payrolls, and bank logs—into a unified, explainable credit assessment system ready for India's public digital platforms."*

---

## Part 2: The Money Moment - Healthy NTC (30 seconds - 1.5 minutes)
**[Action: Go to Sidebar, filter by 'Healthy New-to-Credit (NTC)' and select 'MSME_000041' (or any other NTC MSME)]**

> *"Let's look at a New-to-Credit merchant. Traditionally, this business is a hard 'no-go' for credit officers due to the absence of credit bureau scores or formal balance sheets. In our system, the Account Aggregator consent token is verified, and we run the alternate data pipeline.*
>
> *Under traditional rules, this merchant is invisible. But our system evaluates their alternate data and awards a blended credit score of **89.1 / 100 — placing them safely in the LOW to MEDIUM RISK band**. Notice that the score confidence is flagged as **Medium** because we are operating on partial data (only GST and UPI are present, with no bank records). However, instead of crashing or declining them, our composite scorer dynamically redistributes weights, relying heavily on the GST filing and UPI transaction flows. This merchant gets approved for a suggested **INR 2,00,000 credit limit at a standard interest rate cap of 13.5% p.a.**"*

---

## Part 3: Explainability Panel (1.5 minutes - 2.0 minutes)
**[Action: Scroll down to the Explainable AI (XAI) Underwriting Notes]**

> *"To make this assessment acceptable to credit officers, we need transparency. Our explainability panel is driven by a tree-based SHAP (SHapley Additive exPlanations) engine. We convert raw math log-odds into human-readable credit strengths and risk factors.*
>
> *For this NTC merchant, the top strength is their **outstanding compliance discipline in filing their GST returns**. The top risk factor is not a generic code, but a specific warning: an **elevated customer refund rate of 2.6%**, which indicates operational or delivery friction. The credit officer immediately understands why the score is 89.1 and what risks to monitor."*

---

## Part 4: Ecosystem Integration Points (2.0 minutes - 2.5 minutes)
**[Action: Scroll to 'Alternate Data Completeness & AA Integrations' showing the checklist]**

> *"This entire architecture is built for ecosystem readiness. We model the official ReBIT standard JSON schema for the **Account Aggregator consent artifact**, using standard Purpose Code `102` for Credit Assessment. We also map our credit outputs onto the **Open Credit Enablement Network (OCEN) v4 loan referral payload** to enable lender-agnostic credit offers, and format alternate data vectors into **Unified Lending Interface (ULI)** profiles for frictionless public tech data pulling."*

---

## Part 5: Impact Framing (2.5 minutes - 3.0 minutes)
**[Action: Show the Recommended Credit Action box at the bottom]**

> *"By blending robust rule-based logic with XGBoost risk classification, we achieve two critical bank metrics: first, we expand the onboarding of credit-invisible, viable NTC MSMEs; second, we preserve and protect portfolio quality through strict risk boundaries and explainable AI-backed risk controls. Our entire 106-test suite passes with 83% coverage. Thank you!"*
