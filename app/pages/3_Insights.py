"""
Page 3 — Insights
Human vs. model accuracy comparison, leaderboard, and common mistakes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import get_dataset
from utils.models import train_model, predict_batch
from utils.features import MSCORE_THRESHOLD

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Model Performance")
st.caption("Accuracy summary and decision pattern analysis for the assessment session.")
st.markdown("---")

# ── Check if game has been played ────────────────────────────────────────────
results = st.session_state.get("game_results", [])

if not results:
    st.info("Complete the Fraud Assessment first. This page will display results once an assessment session has been run.")

    # Show model-level stats regardless
    st.markdown("---")
    st.subheader("Model Performance Overview")

    @st.cache_data
    def load_full():
        df = get_dataset()
        model, scaler, cv_scores = train_model(df)
        df = predict_batch(model, scaler, df)
        return df, cv_scores

    df, cv_scores = load_full()

    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-Val Accuracy", f"{cv_scores.mean():.1%}")
    c2.metric("CV Std Dev", f"± {cv_scores.std():.1%}")
    c3.metric("Fraud Base Rate", f"{df['is_fraud'].mean():.1%}")

    # Confusion matrix on full dataset
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df["is_fraud"], df["lr_pred"])
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Clean", "Fraud"],
        y=["Clean", "Fraud"],
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix — Full Dataset (in-sample)",
    )
    fig_cm.update_layout(height=380)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.stop()

# ── Results exist — full insights ────────────────────────────────────────────
results_df = pd.DataFrame(results)

n_played = len(results_df)
human_acc = results_df["human_correct"].mean()
model_acc = results_df["model_correct"].mean()
avg_conf = results_df["user_confidence"].mean()

# ── Top summary ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rounds Played", n_played)
c2.metric("Your Accuracy", f"{human_acc:.0%}")
c3.metric("Model Accuracy", f"{model_acc:.0%}")
c4.metric("Your Avg Confidence", f"{avg_conf:.0f}%")

if human_acc > model_acc:
    st.success(f"Your accuracy ({human_acc:.0%}) exceeded the model ({model_acc:.0%}) on this session.")
elif human_acc == model_acc:
    st.info(f"Your accuracy matched the model ({human_acc:.0%}) on this session.")
else:
    diff = model_acc - human_acc
    st.warning(f"The model outperformed by {diff:.0%} ({model_acc:.0%} vs. {human_acc:.0%}).")

st.markdown("---")

# ── Human vs model bar chart ─────────────────────────────────────────────────
st.subheader("Accuracy Comparison")

fig_acc = go.Figure(go.Bar(
    x=["Analyst", "ML Model"],
    y=[human_acc, model_acc],
    marker_color=["#3498db", "#e74c3c"],
    text=[f"{human_acc:.0%}", f"{model_acc:.0%}"],
    textposition="outside",
    width=0.4,
))
fig_acc.update_layout(
    yaxis=dict(range=[0, 1.15], tickformat=".0%"),
    title="Analyst vs. Model Accuracy",
    height=380,
    showlegend=False,
)
fig_acc.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Baseline (50%)")
st.plotly_chart(fig_acc, use_container_width=True)

st.markdown("---")

# ── Case-by-case results ──────────────────────────────────────────────────────
st.subheader("Case-by-Case Results")

display = results_df[["firm", "actual", "user_pred", "user_confidence", "model_pred", "model_prob",
                        "human_correct", "model_correct"]].copy()
display.columns = ["Firm", "Actual", "Analyst", "Confidence", "Model", "Model Prob",
                   "Analyst Correct", "Model Correct"]
display["Actual"] = display["Actual"].map({1: "Fraud", 0: "Clean"})
display["Analyst"] = display["Analyst"].map({1: "Fraud", 0: "Clean"})
display["Model"] = display["Model"].map({1: "Fraud", 0: "Clean"})
display["Model Prob"] = display["Model Prob"].map("{:.0%}".format)
display["Confidence"] = display["Confidence"].map("{}%".format)
display["Analyst Correct"] = display["Analyst Correct"].map({1: "Yes", 0: "No"})
display["Model Correct"] = display["Model Correct"].map({1: "Yes", 0: "No"})

st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Decision pattern analysis ─────────────────────────────────────────────────
st.subheader("Decision Pattern Analysis")

mistakes = results_df[results_df["human_correct"] == 0]
if len(mistakes) == 0:
    st.success("No misclassifications — all cases correctly identified.")
else:
    col_a, col_b = st.columns(2)

    with col_a:
        false_neg = mistakes[mistakes["actual"] == 1]  # missed fraud
        false_pos = mistakes[mistakes["actual"] == 0]  # flagged clean as fraud
        st.markdown(f"""
**Misclassification breakdown:**
- False negatives (fraud classified as clean): **{len(false_neg)}** case(s)
- False positives (clean classified as fraud): **{len(false_pos)}** case(s)

False negatives carry higher practical cost in fraud detection contexts.
        """)

    with col_b:
        # Confidence when wrong vs right
        right_conf = results_df[results_df["human_correct"] == 1]["user_confidence"].mean()
        wrong_conf = results_df[results_df["human_correct"] == 0]["user_confidence"].mean()

        fig_conf = go.Figure(go.Bar(
            x=["Correct", "Incorrect"],
            y=[right_conf, wrong_conf],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{right_conf:.0f}%", f"{wrong_conf:.0f}%"],
            textposition="outside",
        ))
        fig_conf.update_layout(
            title="Average Confidence — Correct vs. Incorrect",
            yaxis=dict(range=[0, 110]),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ── Session log ───────────────────────────────────────────────────────────────
st.subheader("Session Log")

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

with st.form("add_score"):
    player_name = st.text_input("Name")
    submitted = st.form_submit_button("Save Result")
    if submitted and player_name.strip():
        st.session_state.leaderboard.append({
            "Name": player_name.strip(),
            "Accuracy": f"{human_acc:.0%}",
            "Cases": n_played,
            "Avg Confidence": f"{avg_conf:.0f}%",
        })
        st.success(f"Result saved for {player_name}.")

if st.session_state.leaderboard:
    lb_df = pd.DataFrame(st.session_state.leaderboard)
    st.dataframe(lb_df, use_container_width=True, hide_index=True)
else:
    st.caption("No results saved for this session.")
