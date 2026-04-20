"""
Dataset loader for the Fraud Detection app.

Primary source: data/processed/fraud_dataset_real.csv
  30 real companies (8 confirmed fraud, 22 clean) with Beneish M-score components
  computed from SEC EDGAR XBRL filings (no login required).

Fallback: hardcoded seed data (29 firms, 12 historical fraud cases).
  Used only if the real CSV is missing (e.g., first run before the pipeline runs).
  To regenerate the real dataset: python scripts/build_real_dataset.py
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from .features import compute_mscore, FEATURE_COLS

# ── Path resolution ──────────────────────────────────────────────────────────
# Works regardless of where `streamlit run` is invoked from.
_HERE = os.path.dirname(__file__)                              # app/utils/
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REAL_CSV = os.path.join(_PROJECT_ROOT, "data", "processed", "fraud_dataset_real.csv")


def _load_real() -> pd.DataFrame | None:
    """Load the real EDGAR dataset if it exists."""
    if not os.path.exists(_REAL_CSV):
        return None
    df = pd.read_csv(_REAL_CSV)
    # Ensure anon_id is present
    if "anon_id" not in df.columns:
        df["anon_id"] = ["Firm-" + chr(65 + i % 26) + str(i + 1) for i in range(len(df))]
    return df.reset_index(drop=True)


# ── Seed fallback (used only when real CSV is absent) ────────────────────────
_SEED_RECORDS = [
    # ── Known fraud cases (historical) ────────────────────────────────────────
    {
        "company": "Enron", "year": 2000, "is_fraud": 1,
        "DSRI": 1.465, "GMI": 1.382, "AQI": 1.254, "SGI": 1.341,
        "DEPI": 1.083, "SGAI": 0.923, "TATA": 0.074, "LVGI": 1.179,
        "sector": "Energy", "scandal": "Off-balance-sheet SPEs, mark-to-market abuse",
    },
    {
        "company": "WorldCom", "year": 2001, "is_fraud": 1,
        "DSRI": 1.012, "GMI": 1.452, "AQI": 1.874, "SGI": 1.028,
        "DEPI": 1.531, "SGAI": 0.945, "TATA": 0.089, "LVGI": 1.342,
        "sector": "Telecom", "scandal": "$11B in operating costs capitalized as assets",
    },
    {
        "company": "Tyco International", "year": 2001, "is_fraud": 1,
        "DSRI": 1.283, "GMI": 1.195, "AQI": 1.321, "SGI": 1.412,
        "DEPI": 1.045, "SGAI": 1.134, "TATA": 0.062, "LVGI": 1.243,
        "sector": "Conglomerate", "scandal": "Unauthorized loans and false accounting",
    },
    {
        "company": "HealthSouth", "year": 2002, "is_fraud": 1,
        "DSRI": 1.521, "GMI": 1.334, "AQI": 1.412, "SGI": 1.187,
        "DEPI": 1.123, "SGAI": 0.987, "TATA": 0.083, "LVGI": 1.356,
        "sector": "Healthcare", "scandal": "$2.7B earnings overstatement",
    },
    {
        "company": "Waste Management", "year": 1997, "is_fraud": 1,
        "DSRI": 1.123, "GMI": 1.245, "AQI": 1.543, "SGI": 1.034,
        "DEPI": 1.456, "SGAI": 0.876, "TATA": 0.056, "LVGI": 1.187,
        "sector": "Waste Services", "scandal": "Inflated earnings via improper depreciation",
    },
    {
        "company": "Sunbeam", "year": 1997, "is_fraud": 1,
        "DSRI": 1.867, "GMI": 1.124, "AQI": 1.234, "SGI": 1.256,
        "DEPI": 0.987, "SGAI": 0.934, "TATA": 0.078, "LVGI": 1.012,
        "sector": "Consumer Goods", "scandal": "Channel stuffing and bill-and-hold fraud",
    },
    {
        "company": "Rite Aid", "year": 1999, "is_fraud": 1,
        "DSRI": 1.345, "GMI": 1.312, "AQI": 1.123, "SGI": 1.198,
        "DEPI": 1.067, "SGAI": 1.089, "TATA": 0.045, "LVGI": 1.423,
        "sector": "Retail Pharmacy", "scandal": "Overstated income by $1.6B",
    },
    {
        "company": "Qwest Communications", "year": 2001, "is_fraud": 1,
        "DSRI": 1.234, "GMI": 1.423, "AQI": 1.345, "SGI": 1.567,
        "DEPI": 1.234, "SGAI": 0.912, "TATA": 0.067, "LVGI": 1.234,
        "sector": "Telecom", "scandal": "Swapped capacity assets to inflate revenue",
    },
    {
        "company": "Symbol Technologies", "year": 2002, "is_fraud": 1,
        "DSRI": 1.678, "GMI": 1.234, "AQI": 1.156, "SGI": 1.089,
        "DEPI": 1.023, "SGAI": 1.045, "TATA": 0.091, "LVGI": 1.067,
        "sector": "Technology", "scandal": "Revenue recognition manipulation",
    },
    {
        "company": "Bristol-Myers Squibb", "year": 2002, "is_fraud": 1,
        "DSRI": 1.432, "GMI": 1.087, "AQI": 1.287, "SGI": 1.234,
        "DEPI": 1.034, "SGAI": 1.123, "TATA": 0.053, "LVGI": 1.145,
        "sector": "Pharma", "scandal": "Channel stuffing inflated revenue by $2.5B",
    },
    {
        "company": "Lucent Technologies", "year": 2000, "is_fraud": 1,
        "DSRI": 1.523, "GMI": 1.345, "AQI": 1.432, "SGI": 1.123,
        "DEPI": 1.087, "SGAI": 0.987, "TATA": 0.069, "LVGI": 1.234,
        "sector": "Telecom", "scandal": "Improper revenue recognition on vendor financing",
    },
    {
        "company": "Xerox", "year": 2000, "is_fraud": 1,
        "DSRI": 1.345, "GMI": 1.267, "AQI": 1.234, "SGI": 1.145,
        "DEPI": 1.056, "SGAI": 1.067, "TATA": 0.058, "LVGI": 1.178,
        "sector": "Technology", "scandal": "Accelerated equipment lease revenue by $6.4B",
    },
    # ── Clean firms ────────────────────────────────────────────────────────────
    {
        "company": "Apple", "year": 2022, "is_fraud": 0,
        "DSRI": 0.934, "GMI": 0.987, "AQI": 0.956, "SGI": 1.078,
        "DEPI": 1.023, "SGAI": 0.978, "TATA": -0.034, "LVGI": 0.987,
        "sector": "Technology", "scandal": "",
    },
    {
        "company": "Microsoft", "year": 2022, "is_fraud": 0,
        "DSRI": 0.967, "GMI": 1.012, "AQI": 0.978, "SGI": 1.123,
        "DEPI": 1.045, "SGAI": 0.989, "TATA": -0.028, "LVGI": 0.945,
        "sector": "Technology", "scandal": "",
    },
    {
        "company": "Johnson & Johnson", "year": 2022, "is_fraud": 0,
        "DSRI": 0.945, "GMI": 0.976, "AQI": 1.012, "SGI": 1.034,
        "DEPI": 0.987, "SGAI": 1.023, "TATA": -0.019, "LVGI": 1.023,
        "sector": "Healthcare", "scandal": "",
    },
    {
        "company": "Procter & Gamble", "year": 2022, "is_fraud": 0,
        "DSRI": 0.923, "GMI": 0.989, "AQI": 0.967, "SGI": 1.056,
        "DEPI": 1.012, "SGAI": 0.967, "TATA": -0.023, "LVGI": 1.034,
        "sector": "Consumer Goods", "scandal": "",
    },
    {
        "company": "Coca-Cola", "year": 2022, "is_fraud": 0,
        "DSRI": 0.978, "GMI": 1.023, "AQI": 0.989, "SGI": 1.089,
        "DEPI": 1.034, "SGAI": 0.978, "TATA": -0.031, "LVGI": 1.012,
        "sector": "Beverages", "scandal": "",
    },
    {
        "company": "Walmart", "year": 2022, "is_fraud": 0,
        "DSRI": 0.956, "GMI": 0.978, "AQI": 1.034, "SGI": 1.045,
        "DEPI": 0.978, "SGAI": 1.012, "TATA": -0.027, "LVGI": 0.989,
        "sector": "Retail", "scandal": "",
    },
    {
        "company": "3M", "year": 2022, "is_fraud": 0,
        "DSRI": 0.987, "GMI": 1.012, "AQI": 0.978, "SGI": 0.989,
        "DEPI": 1.023, "SGAI": 1.034, "TATA": -0.022, "LVGI": 1.023,
        "sector": "Industrials", "scandal": "",
    },
    {
        "company": "Intel", "year": 2022, "is_fraud": 0,
        "DSRI": 0.945, "GMI": 0.989, "AQI": 0.967, "SGI": 1.012,
        "DEPI": 1.012, "SGAI": 0.956, "TATA": -0.025, "LVGI": 0.978,
        "sector": "Technology", "scandal": "",
    },
    {
        "company": "IBM", "year": 2022, "is_fraud": 0,
        "DSRI": 0.978, "GMI": 1.023, "AQI": 1.012, "SGI": 0.978,
        "DEPI": 1.034, "SGAI": 1.023, "TATA": -0.018, "LVGI": 1.045,
        "sector": "Technology", "scandal": "",
    },
    {
        "company": "Pfizer", "year": 2022, "is_fraud": 0,
        "DSRI": 0.912, "GMI": 0.967, "AQI": 0.956, "SGI": 1.134,
        "DEPI": 0.989, "SGAI": 0.978, "TATA": -0.029, "LVGI": 1.012,
        "sector": "Pharma", "scandal": "",
    },
    {
        "company": "Abbott Laboratories", "year": 2022, "is_fraud": 0,
        "DSRI": 0.967, "GMI": 1.034, "AQI": 0.989, "SGI": 1.056,
        "DEPI": 1.023, "SGAI": 0.989, "TATA": -0.021, "LVGI": 0.967,
        "sector": "Healthcare", "scandal": "",
    },
    {
        "company": "Caterpillar", "year": 2022, "is_fraud": 0,
        "DSRI": 0.934, "GMI": 0.978, "AQI": 0.967, "SGI": 1.089,
        "DEPI": 1.012, "SGAI": 0.956, "TATA": -0.026, "LVGI": 1.023,
        "sector": "Industrials", "scandal": "",
    },
    {
        "company": "Colgate-Palmolive", "year": 2022, "is_fraud": 0,
        "DSRI": 0.956, "GMI": 1.012, "AQI": 0.978, "SGI": 1.034,
        "DEPI": 0.978, "SGAI": 1.012, "TATA": -0.032, "LVGI": 1.034,
        "sector": "Consumer Goods", "scandal": "",
    },
    {
        "company": "ExxonMobil", "year": 2022, "is_fraud": 0,
        "DSRI": 0.923, "GMI": 0.989, "AQI": 0.945, "SGI": 1.145,
        "DEPI": 1.023, "SGAI": 0.934, "TATA": -0.028, "LVGI": 0.978,
        "sector": "Energy", "scandal": "",
    },
    {
        "company": "Nike", "year": 2022, "is_fraud": 0,
        "DSRI": 0.945, "GMI": 0.978, "AQI": 0.989, "SGI": 1.067,
        "DEPI": 0.978, "SGAI": 1.023, "TATA": -0.019, "LVGI": 0.989,
        "sector": "Consumer Goods", "scandal": "",
    },
    {
        "company": "Merck", "year": 2022, "is_fraud": 0,
        "DSRI": 0.934, "GMI": 0.967, "AQI": 0.978, "SGI": 1.089,
        "DEPI": 1.023, "SGAI": 0.967, "TATA": -0.025, "LVGI": 0.978,
        "sector": "Pharma", "scandal": "",
    },
]


def _load_seed() -> pd.DataFrame:
    df = pd.DataFrame(_SEED_RECORDS)
    df["mscore"] = compute_mscore(df).round(4)
    df["mscore_flag"] = df["mscore"].apply(lambda m: 1 if m > -2.22 else 0)
    df["anon_id"] = ["Firm-" + chr(65 + i % 26) + str(i + 1) for i in range(len(df))]
    return df.reset_index(drop=True)


# ── Public API ───────────────────────────────────────────────────────────────

def get_dataset() -> pd.DataFrame:
    """
    Return the labeled dataset with M-scores.

    Loads from data/processed/fraud_dataset_real.csv (SEC EDGAR pipeline)
    when available, otherwise falls back to the hardcoded seed dataset.
    """
    df = _load_real()
    if df is not None:
        return df
    return _load_seed()


def get_data_source() -> str:
    """Return a human-readable string describing which data source is active."""
    if os.path.exists(_REAL_CSV):
        return "SEC EDGAR XBRL (real filings)"
    return "Seed data (fallback)"


def get_game_sample(df: pd.DataFrame, n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Return a balanced sample for the game (equal fraud / non-fraud)."""
    fraud = df[df["is_fraud"] == 1].sample(
        n=min(n // 2, int((df["is_fraud"] == 1).sum())), random_state=seed
    )
    clean = df[df["is_fraud"] == 0].sample(
        n=min(n // 2, int((df["is_fraud"] == 0).sum())), random_state=seed
    )
    return pd.concat([fraud, clean]).sample(frac=1, random_state=seed).reset_index(drop=True)
