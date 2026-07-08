# MSME Financial Health Card

An AI/ML-driven underwriting engine and credit scoring platform that aggregates alternate data sources (GST, UPI, EPFO, Account Aggregator bank statement logs) to compute a multidimensional credit card score for New-to-Credit (NTC) and New-to-Bank (NTB) MSMEs.

Designed for seamless integration with India's **Unified Lending Interface (ULI)**, **OCEN**, and the **Account Aggregator (AA) consent framework**.

---

## 🚀 Quick Start Setup (5 Commands Total)

Get the API, dashboard, and machine learning models up and running on your system with these simple commands:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the 200-MSME Synthetic Cohort
Builds the synthetic borrower database representing different risk personas (established, NTC, seasonal, volatile, declining):
```bash
python data/synthetic_generators/generate_cohort.py
```

### 3. Train the XGBoost Credit Classifier
Simulates risk outcomes using feature adjustments plus random noise (anti-leakage design), trains the model, and prints performance stats:
```bash
python src/scoring/train_model.py
```

### 4. Start the FastAPI API Service (Ecosystem Simulator)
Starts the backend scoring and explanation server:
```bash
uvicorn src.api.main:app --port 8000
```

### 5. Launch the Streamlit Credit Dashboard
Launches the bank credit-officer dashboard console in your browser:
```bash
streamlit run dashboard/app.py
```

---

## 🧪 Running the Test Suite

Execute the full suite of **111 unit, integration, and end-to-end tests** (including monotonicity and SHAP sum checks) with coverage metrics:
```bash
pytest
```

---

## 📁 Repository Map

* **`src/`**: Credit decisioning core application
  * **`ingestion/`**: Alternate data validators & schemas (Pydantic)
  * **`features/`**: Deseasonalized multi-dimensional credit metrics (Stability, Cash Flow, Compliance, Growth, Repayment)
  * **`scoring/`**: Dynamic scorer blending Rule-Based logic and XGBoost ML
  * **`explainability/`**: SHAP explainer transforming model outputs to credit reason codes
  * **`api/`**: FastAPI ULI/OCEN/AA endpoint routers
  * **`integration/`**: ReBIT consent structures, ULI data exchanges, and OCEN adapters
* **`dashboard/`**: Underwriter visual console (`app.py` in Streamlit)
* **`docs/`**: Demo scripts, known limitations, and adapter integration notes
* **`tests/`**: Unit, integration, and E2E test scripts
