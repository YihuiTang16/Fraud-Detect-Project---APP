"""
Page 1 — Analytics Dashboard
Shows fraud risk scores, Beneish ratio breakdown, and model comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import get_dataset
from utils.models import train_model, predict_batch, get_feature_importances
from utils.features import FEATURE_COLS, FEATURE_DESCRIPTIONS, mscore_label, mscore_color, MSCORE_THRESHOLD

st.set_page_config(page_title="Dashboard", layout="wide")

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return get_dataset()

@st.cache_resource
def load_model(n):
    df = get_dataset()
    return train_model(df)

df = load_data()
model, scaler, cv_scores = load_model(len(df))
df = predict_batch(model, scaler, df)

# ── Page header ──────────────────────────────────────────────────────────────
st.title("Analytics Dashboard")
st.caption("Explore fraud risk scores and financial ratios for each company in the dataset.")
st.markdown("---")

# ── Summary metrics ──────────────────────────────────────────────────────────
total = len(df)
n_fraud = int(df["is_fraud"].sum())
n_flagged_mscore = int((df["mscore"] > MSCORE_THRESHOLD).sum())
n_flagged_lr = int((df["lr_prob"] >= 0.5).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Companies", total)
c2.metric("Known Fraud Cases", n_fraud)
c3.metric("M-Score Flags (> −2.22)", n_flagged_mscore)
c4.metric("LR Model Flags (prob ≥ 50%)", n_flagged_lr)

st.markdown("---")

# ── Overview scatter plot ────────────────────────────────────────────────────
st.subheader("M-Score Distribution")

fig_scatter = px.scatter(
    df,
    x="company",
    y="mscore",
    color=df["is_fraud"].map({1: "Fraud", 0: "Clean"}),
    color_discrete_map={"Fraud": "#e74c3c", "Clean": "#2ecc71"},
    hover_data={"mscore": ":.3f", "lr_prob": ":.2%", "year": True, "sector": True},
    labels={"company": "Company", "mscore": "M-Score", "color": "Label"},
    title="Beneish M-Score by Company  (dashed line = −2.22 threshold)",
)
fig_scatter.add_hline(
    y=MSCORE_THRESHOLD,
    line_dash="dash",
    line_color="gray",
    annotation_text="Threshold (−2.22)",
    annotation_position="top right",
)
fig_scatter.update_layout(xaxis_tickangle=-45, height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ── Company deep-dive ────────────────────────────────────────────────────────
st.subheader("Company Deep Dive")

company_list = df["company"].tolist()
selected = st.selectbox("Select a company", company_list)
row = df[df["company"] == selected].iloc[0]

col_left, col_right = st.columns([1, 2])

with col_left:
    label = "FRAUD" if row["is_fraud"] == 1 else "CLEAN"
    st.markdown(f"### {selected} ({int(row['year'])})")
    st.markdown(f"**Actual label:** {label}")
    st.markdown(f"**Sector:** {row['sector']}")

    m = row["mscore"]
    color = mscore_color(m)
    risk = mscore_label(m)
    st.markdown(f"**M-Score:** `{m:.3f}` — :{color}[{risk}]")
    st.markdown(f"**LR Fraud Probability:** `{row['lr_prob']:.1%}`")
    st.markdown(f"**LR Prediction:** {'Fraud' if row['lr_pred'] == 1 else 'Clean'}")

    if row["scandal"]:
        st.info(f"📋 **Scandal:** {row['scandal']}")

with col_right:
    # Radar / bar chart of the 8 features
    feature_vals = [row[f] for f in FEATURE_COLS]
    fig_bar = go.Figure(go.Bar(
        x=FEATURE_COLS,
        y=feature_vals,
        marker_color=["#e74c3c" if v > 1.1 else "#f39c12" if v > 1.0 else "#2ecc71" for v in feature_vals],
        text=[f"{v:.3f}" for v in feature_vals],
        textposition="outside",
    ))
    fig_bar.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Index = 1.0")
    fig_bar.update_layout(
        title=f"Beneish Components — {selected}",
        yaxis_title="Index Value",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Feature descriptions
with st.expander("What do these ratios mean?"):
    for feat, desc in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"- **{feat}**: {desc}")

st.markdown("---")

# ── Model comparison table ────────────────────────────────────────────────────
st.subheader("Full Dataset — Model Comparison")

display_df = df[["company", "year", "sector", "is_fraud", "mscore", "mscore_flag", "lr_prob", "lr_pred"]].copy()
display_df.columns = ["Company", "Year", "Sector", "Actual Fraud", "M-Score", "M-Score Flag", "LR Prob", "LR Pred"]
display_df["M-Score"] = display_df["M-Score"].map("{:.3f}".format)
display_df["LR Prob"] = display_df["LR Prob"].map("{:.1%}".format)

def highlight_fraud(row):
    if row["is_fraud"] == 1:
        return ["background-color: #f8d7da; color: #000000"] * len(row)
    else:
        return [""] * len(row)

st.dataframe(
    display_df.style.apply(highlight_fraud, axis=1),
    use_container_width=True,
    height=420,
)

st.markdown("---")

# ── Feature importance ───────────────────────────────────────────────────────
st.subheader("Logistic Regression — Feature Importance")
st.caption("Coefficient magnitude indicates how strongly each ratio influences the model's fraud prediction.")

fi = get_feature_importances(model, scaler)
fig_fi = px.bar(
    fi,
    x="abs_importance",
    y="feature",
    orientation="h",
    color="coefficient",
    color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
    labels={"abs_importance": "|Coefficient|", "feature": "Feature", "coefficient": "Coef"},
    title="Feature Importance (|LR Coefficient|)",
)
fig_fi.update_layout(height=380)
st.plotly_chart(fig_fi, use_container_width=True)
