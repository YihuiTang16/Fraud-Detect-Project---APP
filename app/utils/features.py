"""
Beneish M-Score feature computation.

The Beneish M-Score is an accounting-based model that uses 8 financial ratios
to flag potential earnings manipulation.

Formula:
  M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
          + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

Threshold: M > -2.22  →  likely manipulator
"""

import pandas as pd
import numpy as np

FEATURE_COLS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]

MSCORE_WEIGHTS = {
    "intercept": -4.84,
    "DSRI":  0.920,
    "GMI":   0.528,
    "AQI":   0.404,
    "SGI":   0.892,
    "DEPI":  0.115,
    "SGAI": -0.172,
    "TATA":  4.679,
    "LVGI": -0.327,
}

MSCORE_THRESHOLD = -2.22

FEATURE_DESCRIPTIONS = {
    "DSRI":  "Days Sales in Receivables Index — receivables growing faster than sales signals channel stuffing",
    "GMI":   "Gross Margin Index — deteriorating margins create pressure to manipulate",
    "AQI":   "Asset Quality Index — rising intangibles / deferred costs relative to total assets",
    "SGI":   "Sales Growth Index — high-growth firms face greater incentive to manipulate",
    "DEPI":  "Depreciation Index — falling depreciation rate may indicate expense capitalization",
    "SGAI":  "SG&A Index — rising selling & admin expenses relative to sales",
    "TATA":  "Total Accruals to Total Assets — high accruals suggest low earnings quality",
    "LVGI":  "Leverage Index — rising debt increases default risk and manipulation incentive",
}


def compute_mscore(df: pd.DataFrame) -> pd.Series:
    """Compute the Beneish M-score for each row in df."""
    m = MSCORE_WEIGHTS["intercept"]
    for col, w in MSCORE_WEIGHTS.items():
        if col == "intercept":
            continue
        m = m + w * df[col]
    return m


def mscore_label(mscore: float) -> str:
    """Return a human-readable risk label for a given M-score."""
    if mscore > -1.78:
        return "High Risk"
    elif mscore > -2.22:
        return "Elevated Risk"
    else:
        return "Low Risk"


def mscore_color(mscore: float) -> str:
    """Return a color string for display."""
    if mscore > -1.78:
        return "red"
    elif mscore > -2.22:
        return "orange"
    else:
        return "green"
