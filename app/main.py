"""
Financial Fraud Detection Web App

Run with:  streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.data_loader import get_dataset, get_data_source
from utils.models import train_model, predict_batch

st.set_page_config(
    page_title="Fraud Detector",
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
st.title("Financial Fraud Detection Analysis App")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Companies in Dataset", len(df))
with col2:
    st.metric("Known Fraud Cases", int(df["is_fraud"].sum()))
with col3:
    st.metric("Model CV Accuracy", f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
with col4:
    st.metric("Data Source", get_data_source())

st.markdown("---")

st.markdown("""
### What this app does

This app uses **financial statement data** and the **Beneish M-Score** model to flag
potential earnings manipulation. It also lets you test your own judgment against
the machine learning model.

| Page | What you can do |
|------|----------------|
| **Dashboard** | Explore fraud risk scores and key financial ratios for each firm |
| **Game** | Try to identify fraudulent firms yourself — then compare with the model |
| **Insights** | See how human intuition stacks up against data-driven predictions |

---

### About the Beneish M-Score

The M-Score is an accounting-based model developed by Messod Beneish (1999).
It uses 8 financial ratios to detect earnings manipulation:

| Ratio | What it measures |
|-------|-----------------|
| DSRI | Days Sales in Receivables Index |
| GMI | Gross Margin Index |
| AQI | Asset Quality Index |
| SGI | Sales Growth Index |
| DEPI | Depreciation Index |
| SGAI | SG&A Expense Index |
| TATA | Total Accruals to Total Assets |
| LVGI | Leverage Index |

**Score > −2.22 → likely manipulator.**

---

### Navigation

Use the **sidebar** to navigate between pages.
""")

st.info("Select a page from the sidebar to get started.")
