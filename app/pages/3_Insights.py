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

st.set_page_config(page_title="Insights", page_icon="💡", layout="wide")

st.title("💡 Insights")
st.caption("How does human intuition compare to the machine learning model?")
st.markdown("---")

# ── Check if game has been played ────────────────────────────────────────────
results = st.session_state.get("game_results", [])

if not results:
    st.info("👈 Play the **Game** first to generate data. Come back here afterwards to see how you did!")

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
        title="Model Confusion Matrix (full dataset)",
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
    st.success(f"🏆 You beat the model! Your accuracy ({human_acc:.0%}) > Model ({model_acc:.0%})")
elif human_acc == model_acc:
    st.info(f"🤝 It's a tie! Both you and the model scored {human_acc:.0%}")
else:
    diff = model_acc - human_acc
    st.warning(f"🤖 The model won this round by {diff:.0%}. Better luck next time!")

st.markdown("---")

# ── Human vs model bar chart ─────────────────────────────────────────────────
st.subheader("Accuracy Comparison")

fig_acc = go.Figure(go.Bar(
    x=["You", "ML Model"],
    y=[human_acc, model_acc],
    marker_color=["#3498db", "#e74c3c"],
    text=[f"{human_acc:.0%}", f"{model_acc:.0%}"],
    textposition="outside",
    width=0.4,
))
fig_acc.update_layout(
    yaxis=dict(range=[0, 1.15], tickformat=".0%"),
    title="Human vs. Model Accuracy",
    height=380,
    showlegend=False,
)
fig_acc.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random guess (50%)")
st.plotly_chart(fig_acc, use_container_width=True)

st.markdown("---")

# ── Round-by-round breakdown ──────────────────────────────────────────────────
st.subheader("Round-by-Round Breakdown")

display = results_df[["firm", "actual", "user_pred", "user_confidence", "model_pred", "model_prob",
                        "human_correct", "model_correct"]].copy()
display.columns = ["Firm", "Actual", "Your Guess", "Confidence", "Model Guess", "Model Prob",
                   "You Correct", "Model Correct"]
display["Actual"] = display["Actual"].map({1: "🚨 Fraud", 0: "✅ Clean"})
display["Your Guess"] = display["Your Guess"].map({1: "🚨 Fraud", 0: "✅ Clean"})
display["Model Guess"] = display["Model Guess"].map({1: "🚨 Fraud", 0: "✅ Clean"})
display["Model Prob"] = display["Model Prob"].map("{:.0%}".format)
display["Confidence"] = display["Confidence"].map("{}%".format)
display["You Correct"] = display["You Correct"].map({1: "✓", 0: "✗"})
display["Model Correct"] = display["Model Correct"].map({1: "✓", 0: "✗"})

st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Common mistakes analysis ──────────────────────────────────────────────────
st.subheader("Decision Pattern Analysis")

mistakes = results_df[results_df["human_correct"] == 0]
if len(mistakes) == 0:
    st.success("You made no mistakes — perfect score!")
else:
    col_a, col_b = st.columns(2)

    with col_a:
        false_neg = mistakes[mistakes["actual"] == 1]  # missed fraud
        false_pos = mistakes[mistakes["actual"] == 0]  # flagged clean as fraud
        st.markdown(f"""
**Your mistake breakdown:**
- Missed fraud (called it clean): **{len(false_neg)}** case(s)
- False alarm (called clean company fraud): **{len(false_pos)}** case(s)

*Missing real fraud is typically the more costly error in practice.*
        """)

    with col_b:
        # Confidence when wrong vs right
        right_conf = results_df[results_df["human_correct"] == 1]["user_confidence"].mean()
        wrong_conf = results_df[results_df["human_correct"] == 0]["user_confidence"].mean()

        fig_conf = go.Figure(go.Bar(
            x=["Correct answers", "Wrong answers"],
            y=[right_conf, wrong_conf],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{right_conf:.0f}%", f"{wrong_conf:.0f}%"],
            textposition="outside",
        ))
        fig_conf.update_layout(
            title="Avg Confidence — Correct vs Wrong",
            yaxis=dict(range=[0, 110]),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ── Leaderboard (session-based) ───────────────────────────────────────────────
st.subheader("Leaderboard")

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

with st.form("add_score"):
    player_name = st.text_input("Enter your name to save your score")
    submitted = st.form_submit_button("Save Score")
    if submitted and player_name.strip():
        st.session_state.leaderboard.append({
            "Name": player_name.strip(),
            "Accuracy": f"{human_acc:.0%}",
            "Rounds": n_played,
            "Avg Confidence": f"{avg_conf:.0f}%",
        })
        st.success(f"Score saved for {player_name}!")

if st.session_state.leaderboard:
    lb_df = pd.DataFrame(st.session_state.leaderboard)
    st.dataframe(lb_df, use_container_width=True, hide_index=True)
else:
    st.caption("No scores saved yet — submit your name above.")
