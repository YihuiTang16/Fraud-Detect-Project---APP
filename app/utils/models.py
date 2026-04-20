"""
Model training and prediction utilities.

MVP model: Logistic Regression trained on Beneish M-score features.
Uses StandardScaler for normalization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from .features import FEATURE_COLS


def train_model(df: pd.DataFrame):
    """
    Train a Logistic Regression model on the dataset.
    Returns (model, scaler, cv_scores).
    """
    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    model.fit(X_scaled, y)

    # Cross-validation (leave-one-out for small dataset)
    cv = StratifiedKFold(n_splits=min(5, y.sum()), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    return model, scaler, cv_scores


def predict(model, scaler, features: dict) -> tuple:
    """
    Predict fraud probability for a single firm.
    features: dict with keys matching FEATURE_COLS.
    Returns (prediction: int, fraud_probability: float).
    """
    X = pd.DataFrame([features])[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(model.predict(X_scaled)[0])
    return pred, round(prob, 4)


def predict_batch(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """Add lr_pred and lr_prob columns to df."""
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    df = df.copy()
    df["lr_prob"] = model.predict_proba(X_scaled)[:, 1].round(4)
    df["lr_pred"] = model.predict(X_scaled)
    return df


def get_feature_importances(model, scaler) -> pd.DataFrame:
    """Return a DataFrame of feature coefficients (as importance proxy)."""
    coefs = model.coef_[0]
    return pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": coefs,
        "abs_importance": np.abs(coefs),
    }).sort_values("abs_importance", ascending=False).reset_index(drop=True)
