# Financial Fraud Detection Web App
**BA870-AC820 | Spring 2026**

A multi-page Streamlit app that detects potential financial statement manipulation
using the Beneish M-Score and Logistic Regression, with a gamified interface that
benchmarks human intuition against the model.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (from this folder)
streamlit run app/main.py
```

The app opens at `http://localhost:8501`. Use the sidebar to navigate between pages.

---

## What This App Does

This app applies the **Beneish M-Score** — an accounting-based model from the
finance literature — to real company filings to flag potential earnings manipulation.
It also includes a game that lets you test your own fraud-detection judgment against
the model.

| Page | Description |
|------|-------------|
| **Home** | Dataset overview and model accuracy |
| **Dashboard** | M-score scatter plot, per-company deep dive, feature importance |
| **Game** | 10-round interactive game: guess fraud vs. clean, then compare with the model |
| **Insights** | Your accuracy vs. model accuracy, mistake analysis, confidence calibration |

---

## Data Source

**Source:** [SEC EDGAR XBRL API](https://www.sec.gov/developer) — free, no login required.

Financial statement data (10-K annual filings) was pulled from
`data.sec.gov/api/xbrl/companyfacts/` for 30 companies. Fraud labels are based
on confirmed SEC enforcement actions (Accounting and Auditing Enforcement Releases
and litigation releases).

| Category | Count |
|----------|-------|
| Confirmed fraud cases | 8 |
| Clean benchmark firms (S&P 500) | 22 |
| **Total** | **30** |

**Fraud cases:** KHC (Kraft Heinz), GRPN (Groupon), UAA (Under Armour), GE,
MDXG (MiMedx), BHC (Bausch Health), LL (Lumber Liquidators), LCI (Lannett).

The dataset is pre-built and included at `data/fraud_dataset_real.csv`.

---

## Model Summary

### Beneish M-Score (rule-based baseline)

Formula from Beneish (1999):

```
M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
         + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI
```

| Feature | Description |
|---------|-------------|
| DSRI | Days Sales in Receivables Index |
| GMI | Gross Margin Index |
| AQI | Asset Quality Index |
| SGI | Sales Growth Index |
| DEPI | Depreciation Index |
| SGAI | SG&A Expense Index |
| TATA | Total Accruals to Total Assets |
| LVGI | Leverage Index |

**Threshold:** M > −2.22 → likely manipulator

### Logistic Regression (ML model)

Trained on the 8 Beneish features with StandardScaler normalization.
5-fold stratified cross-validation. CV accuracy: ~73% on real data.

### Key Finding

Most confirmed fraud companies in this dataset score *below* the −2.22 M-Score
threshold because their fraud types (non-GAAP metric manipulation, reserve
understatement, pricing disclosure) differ from the receivables/revenue manipulation
the M-Score was calibrated to detect. This limitation is documented and analyzed
in the modeling notebook.

---

## Project Structure

```
submission/
├── app/
│   ├── main.py                  # Home page + model loader
│   ├── pages/
│   │   ├── 1_Dashboard.py       # Analytics dashboard
│   │   ├── 2_Game.py            # Human vs. model game
│   │   └── 3_Insights.py        # Results and insights
│   └── utils/
│       ├── features.py          # Beneish M-score formula
│       ├── data_loader.py       # Dataset loader (real CSV → seed fallback)
│       └── models.py            # Logistic Regression training + prediction
├── data/
│   └── fraud_dataset_real.csv   # Pre-built dataset from SEC EDGAR
├── notebooks/
│   └── fraud_detection_modeling.ipynb   # Full ML pipeline (Colab-friendly)
├── requirements.txt
└── README.md
```

---

## Modeling Notebook

`notebooks/fraud_detection_modeling.ipynb` is fully executable on Google Colab
(no local files required — real data is embedded inline as a fallback).

Covers: EDA → M-Score computation → Logistic Regression → Evaluation →
Feature Importance → Key Findings.

---

## Tech Stack

- Python 3.10+
- Streamlit, pandas, scikit-learn, plotly, numpy

## Reference

Beneish, M. D. (1999). *The Detection of Earnings Manipulation.*
Financial Analysts Journal, 55(5), 24–36.
