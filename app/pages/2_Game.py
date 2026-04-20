"""
Page 2 — Human vs. Model Game
Users classify anonymized firms as fraud / not fraud, then see model results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import get_dataset, get_game_sample
from utils.models import train_model, predict_batch
from utils.features import FEATURE_COLS, FEATURE_DESCRIPTIONS, mscore_label, mscore_color

st.set_page_config(page_title="Game", layout="wide")

# ── Load data & model ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return get_dataset()

@st.cache_resource
def load_model(n):
    df = get_dataset()
    return train_model(df)

df_full = load_data()
model, scaler, cv_scores = load_model(len(df_full))
df_full = predict_batch(model, scaler, df_full)

# ── Session state initialization ─────────────────────────────────────────────
if "game_df" not in st.session_state:
    st.session_state.game_df = get_game_sample(df_full, n=10)
if "game_idx" not in st.session_state:
    st.session_state.game_idx = 0
if "game_results" not in st.session_state:
    st.session_state.game_results = []
if "revealed" not in st.session_state:
    st.session_state.revealed = False

game_df = st.session_state.game_df
total_rounds = len(game_df)
idx = st.session_state.game_idx

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Fraud Assessment Game")
st.caption("Can you spot financial fraud? Review anonymized company financials and classify each firm as fraud or clean. Results are compared against the model after each submission.")

# ── Progress bar ──────────────────────────────────────────────────────────────
completed = len(st.session_state.game_results)
progress = completed / total_rounds
st.progress(progress, text=f"Round {min(idx + 1, total_rounds)} of {total_rounds}")
st.markdown("---")

# ── Game over screen ──────────────────────────────────────────────────────────
if idx >= total_rounds:
    st.success("You've completed all rounds! Head to the **Insights** page to see how you did.")
    results_df = pd.DataFrame(st.session_state.game_results)

    human_acc = (results_df["human_correct"]).mean()
    model_acc = (results_df["model_correct"]).mean()

    c1, c2 = st.columns(2)
    c1.metric("Your Accuracy", f"{human_acc:.0%}")
    c2.metric("Model Accuracy", f"{model_acc:.0%}")

    if st.button("Play Again"):
        st.session_state.game_df = get_game_sample(df_full, n=10, seed=completed + 1)
        st.session_state.game_idx = 0
        st.session_state.game_results = []
        st.session_state.revealed = False
        st.rerun()
    st.stop()

# ── Current firm ──────────────────────────────────────────────────────────────
firm = game_df.iloc[idx]

st.subheader(f"Company: **{firm['anon_id']}**  |  Sector: {firm['sector']}  |  Year: {int(firm['year'])}")
st.markdown("*Company name is hidden. Use the financial ratios below to make your judgment.*")

# ── Ratio display ─────────────────────────────────────────────────────────────
col_chart, col_table = st.columns([2, 1])

with col_chart:
    vals = [firm[f] for f in FEATURE_COLS]
    fig = go.Figure(go.Bar(
        x=FEATURE_COLS,
        y=vals,
        marker_color=["#e74c3c" if v > 1.15 else "#f39c12" if v > 1.0 else "#2ecc71" for v in vals],
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Financial Ratios (index > 1.0 = elevated)",
        yaxis_title="Index Value",
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.markdown("**Ratio Summary**")
    ratio_df = pd.DataFrame({
        "Ratio": FEATURE_COLS,
        "Value": [f"{firm[f]:.3f}" for f in FEATURE_COLS],
        "Signal": ["High" if firm[f] > 1.15 else "↑ Elevated" if firm[f] > 1.0 else "Normal"
                   for f in FEATURE_COLS],
    })
    st.dataframe(ratio_df, use_container_width=True, hide_index=True)

with st.expander("What do these ratios mean?"):
    for feat, desc in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"- **{feat}**: {desc}")

st.markdown("---")

# ── User input ────────────────────────────────────────────────────────────────
if not st.session_state.revealed:
    st.subheader("Your Prediction")

    col_guess, col_conf = st.columns(2)
    with col_guess:
        user_choice = st.radio(
            "Is this company committing fraud?",
            options=["Fraud", "Not Fraud"],
            horizontal=True,
        )
    with col_conf:
        confidence = st.slider("How confident are you? (%)", 50, 100, 70, step=5)

    if st.button("Submit & Reveal", type="primary", use_container_width=True):
        st.session_state.user_choice = user_choice
        st.session_state.user_conf = confidence
        st.session_state.revealed = True
        st.rerun()

# ── Reveal ────────────────────────────────────────────────────────────────────
if st.session_state.revealed:
    user_choice = st.session_state.get("user_choice", "Not Fraud")
    confidence = st.session_state.get("user_conf", 70)
    user_pred = 1 if "Fraud" in user_choice and "Not" not in user_choice else 0

    actual = int(firm["is_fraud"])
    model_pred = int(firm["lr_pred"])
    model_prob = float(firm["lr_prob"])
    mscore = float(firm["mscore"])

    human_correct = int(user_pred == actual)
    model_correct = int(model_pred == actual)

    # Record result
    if len(st.session_state.game_results) == idx:
        st.session_state.game_results.append({
            "firm": firm["anon_id"],
            "actual": actual,
            "user_pred": user_pred,
            "user_confidence": confidence,
            "model_pred": model_pred,
            "model_prob": model_prob,
            "human_correct": human_correct,
            "model_correct": model_correct,
        })

    st.markdown("---")
    st.subheader("Results Revealed")

    r1, r2, r3 = st.columns(3)
    actual_str = "FRAUD" if actual == 1 else "CLEAN"
    r1.metric("Actual Label", actual_str)
    r2.metric("Your Answer", "FRAUD" if user_pred == 1 else "NOT FRAUD",
              delta="✓ Correct" if human_correct else "✗ Wrong",
              delta_color="normal" if human_correct else "inverse")
    r3.metric("Model Answer", f"{'FRAUD' if model_pred == 1 else 'NOT FRAUD'} ({model_prob:.0%})",
              delta="✓ Correct" if model_correct else "✗ Wrong",
              delta_color="normal" if model_correct else "inverse")

    st.markdown(f"**Beneish M-Score:** `{mscore:.3f}` — {mscore_label(mscore)}")
    if firm["scandal"]:
        st.warning(f"**Known scandal:** {firm['scandal']}")

    if st.button("Next Company →", type="primary", use_container_width=True):
        st.session_state.game_idx += 1
        st.session_state.revealed = False
        st.rerun()
