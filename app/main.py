"""
Financial Fraud Detection Web App
BA870-AC820 | Spring 2026

Run with:  streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.data_loader import get_dataset, get_data_source
from utils.models import train_model, predict_batch

st.set_page_config(
    page_title="Financial Fraud Risk Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data & model once (cached) ─────────────────────────────────────────
@st.cache_data
def load_data():
    return get_dataset()

@st.cache_resource
def load_model(data_hash):
    df = get_dataset()
    model, scaler, cv_scores = train_model(df)
    return model, scaler, cv_scores

df = load_data()
model, scaler, cv_scores = load_model(len(df))
df = predict_batch(model, scaler, df)

# Store in session state so pages can access them
st.session_state["df"] = df
st.session_state["model"] = model
st.session_state["scaler"] = scaler

# ── Home page ────────────────────────────────────────────────────────────────
st.title("Financial Fraud Risk Analysis")

st.markdown(
    "This tool applies the Beneish M-Score and logistic regression to real company filings "
    "sourced from the SEC EDGAR XBRL API to identify potential earnings manipulation. "
    "It also includes an assessment module that benchmarks human judgment against the model."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Companies in Dataset", len(df))
with col2:
    st.metric("Confirmed Fraud Cases", int(df["is_fraud"].sum()))
with col3:
    st.metric("Model CV Accuracy", f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

st.markdown("---")

st.markdown("""
### Pages

| Page | Description |
|------|-------------|
| **Fraud Risk Dashboard** | M-score distribution, per-company ratio breakdown, model comparison |
| **Fraud Assessment** | Classify anonymized firms and compare your judgment to the model |
| **Model Performance** | Accuracy results, confusion matrix, and decision pattern analysis |

---

### Beneish M-Score

An accounting-based model developed by Beneish (1999) that uses 8 financial ratios
derived from annual filings to detect earnings manipulation.

| Ratio | Description |
|-------|-------------|
| DSRI | Days Sales in Receivables Index |
| GMI | Gross Margin Index |
| AQI | Asset Quality Index |
| SGI | Sales Growth Index |
| DEPI | Depreciation Index |
| SGAI | SG&A Expense Index |
| TATA | Total Accruals to Total Assets |
| LVGI | Leverage Index |

Threshold: M > −2.22 indicates likely manipulation.
""")
