"""
Real-data pipeline: SEC EDGAR XBRL → Beneish M-Score features → labeled dataset.

Data source:   SEC EDGAR company facts API (free, no login required)
               https://data.sec.gov/api/xbrl/companyfacts/CIK{...}.json

Fraud labels:  Confirmed SEC enforcement actions (AAER / litigation releases)
               See inline scandal notes for source references.

Output:        data/processed/fraud_dataset_real.csv

Usage:
    python scripts/build_real_dataset.py

Requires: requests, pandas  (no yfinance, no login)
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── EDGAR API settings ────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "FraudDetectResearch BA870-AC820 research-noreply@bu.edu",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
REQUEST_DELAY = 0.15  # seconds between requests (stay under 10/s SEC limit)

# ── Company list ──────────────────────────────────────────────────────────────
# fraud_year_range: list of fiscal years to try (most preferred first).
# We compute Beneish ratios for year t vs year t-1. The first year in the range
# that yields at least 6 of 8 complete Beneish components is used.
#
# CIK numbers verified against https://www.sec.gov/files/company_tickers.json
# Fraud labels sourced from SEC AAER / Litigation Releases (public record).

COMPANIES = [
    # ── Confirmed fraud (SEC enforcement actions) ──────────────────────────────
    {
        "ticker": "KHC", "cik": "0001637459", "is_fraud": 1,
        "fraud_year_range": [2017, 2018, 2016], "sector": "Consumer Goods",
        "scandal": "SEC settlement 2019 ($62M): procurement accounting fraud, inflated cost savings",
    },
    {
        "ticker": "GRPN", "cik": "0001490281", "is_fraud": 1,
        "fraud_year_range": [2020, 2019, 2021], "sector": "Technology",
        "scandal": "SEC settlement 2023: misclassified expenses to inflate non-GAAP adjusted EBITDA",
    },
    {
        # CIK 0001336917 verified; fraud_year 2019 is within SEC settlement period (2015-2019)
        "ticker": "UAA", "cik": "0001336917", "is_fraud": 1,
        "fraud_year_range": [2019, 2020, 2018], "sector": "Consumer Goods",
        "scandal": "SEC settlement 2021 ($9M): revenue pull-forward across quarters to meet targets",
    },
    {
        # CIK 0000040545 = General Electric Company (not GE Capital 0000040554)
        # fraud_year 2016 is within the SEC settlement period (2015-2016)
        "ticker": "GE", "cik": "0000040545", "is_fraud": 1,
        "fraud_year_range": [2016, 2017], "sector": "Industrials",
        "scandal": "SEC settlement 2023 ($200M): insurance reserve understatement, power segment failures",
    },
    {
        # MDXG: SEC enforcement 2019 for fraud 2012-2018; first post-restatement filing is 2019
        "ticker": "MDXG", "cik": "0001376339", "is_fraud": 1,
        "fraud_year_range": [2021, 2020, 2022], "sector": "Healthcare",
        "scandal": "SEC enforcement 2019: improper revenue recognition and channel stuffing 2012-2018",
    },
    {
        # Bausch Health (formerly Valeant Pharma): channel stuffing through Philidor
        "ticker": "BHC", "cik": "0000885590", "is_fraud": 1,
        "fraud_year_range": [2022, 2021, 2023], "sector": "Pharma",
        "scandal": "SEC settlement 2020: channel stuffing at Philidor; related-party concealment",
    },
    {
        # Lumber Liquidators / LL Flooring: SEC settlement 2019 for securities fraud
        "ticker": "LL", "cik": "0001396033", "is_fraud": 1,
        "fraud_year_range": [2019, 2018, 2020], "sector": "Retail",
        "scandal": "SEC settlement 2019 ($33M): misleading statements on product safety compliance",
    },
    {
        # Lannett Company: SEC settlement 2020 for price-fixing disclosures
        "ticker": "LCI", "cik": "0000057725", "is_fraud": 1,
        "fraud_year_range": [2019, 2018, 2020], "sector": "Pharma",
        "scandal": "SEC settlement 2020: misleading disclosures about generic drug pricing scheme",
    },
    # ── Clean firms — S&P 500, no known SEC enforcement actions ───────────────
    {"ticker": "AAPL", "cik": "0000320193", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Technology", "scandal": ""},
    {"ticker": "MSFT", "cik": "0000789019", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Technology", "scandal": ""},
    {"ticker": "JNJ", "cik": "0000200406", "is_fraud": 0,
     "fraud_year_range": [2022, 2023], "sector": "Healthcare", "scandal": ""},
    {"ticker": "PG", "cik": "0000080424", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Consumer Goods", "scandal": ""},
    {"ticker": "KO", "cik": "0000021344", "is_fraud": 0,
     "fraud_year_range": [2022, 2023], "sector": "Beverages", "scandal": ""},
    {"ticker": "WMT", "cik": "0000104169", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Retail", "scandal": ""},
    {"ticker": "MMM", "cik": "0000066740", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Industrials", "scandal": ""},
    {"ticker": "INTC", "cik": "0000050863", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Technology", "scandal": ""},
    {"ticker": "IBM", "cik": "0000051143", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Technology", "scandal": ""},
    {"ticker": "PFE", "cik": "0000078003", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Pharma", "scandal": ""},
    {"ticker": "ABT", "cik": "0000001800", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Healthcare", "scandal": ""},
    {"ticker": "CAT", "cik": "0000018230", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Industrials", "scandal": ""},
    {"ticker": "CL", "cik": "0000021665", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Consumer Goods", "scandal": ""},
    {"ticker": "XOM", "cik": "0000034088", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Energy", "scandal": ""},
    {"ticker": "NKE", "cik": "0000320187", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Consumer Goods", "scandal": ""},
    {"ticker": "MRK", "cik": "0000310158", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Pharma", "scandal": ""},
    {"ticker": "HD", "cik": "0000354950", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Retail", "scandal": ""},
    {"ticker": "CVX", "cik": "0000093410", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Energy", "scandal": ""},
    {"ticker": "LLY", "cik": "0000059478", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Pharma", "scandal": ""},
    {"ticker": "AMZN", "cik": "0001018724", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Technology", "scandal": ""},
    {"ticker": "V", "cik": "0001403161", "is_fraud": 0,
     "fraud_year_range": [2023, 2022], "sector": "Financials", "scandal": ""},
    {"ticker": "UNH", "cik": "0000072333", "is_fraud": 0,
     "fraud_year_range": [2022, 2021], "sector": "Healthcare", "scandal": ""},
]

# ── XBRL concept aliases ──────────────────────────────────────────────────────
# For each field, try aliases in order. Use the first one that has annual data.
CONCEPT_ALIASES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenuesNetOfInterestExpense",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",      # modern standard (post-2018)
        "CostOfGoodsSold",                  # older standard
        "CostOfRevenue",
        "CostOfSales",
        "CostOfGoodsAndServices",
        "CostOfServicesSold",
    ],
    "gross_profit": [                       # used to derive COGS if direct not available
        "GrossProfit",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableMember",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
    ],
    "total_assets": [
        "Assets",
    ],
    "depreciation": [
        "Depreciation",
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "lt_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
        "SeniorLongTermNotes",
    ],
    "current_liab": [
        "LiabilitiesCurrent",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByOperatingActivities",
    ],
}


# ── EDGAR fetch helpers ───────────────────────────────────────────────────────

def fetch_company_facts(cik: str) -> dict | None:
    """Download company facts JSON from EDGAR. Uses file cache."""
    cik_padded = cik.lstrip("0").zfill(10)
    cache_path = RAW_DIR / f"CIK{cik_padded}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            return None
        data = resp.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        time.sleep(REQUEST_DELAY)
        return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def get_annual_series(facts: dict, field: str) -> dict[int, float]:
    """
    Return {fiscal_year: value} merged across ALL aliases for the field.
    For each fiscal year, we keep the value from the MOST RECENTLY FILED record
    across all aliases. This handles companies that changed XBRL tags over time
    (e.g. old SalesRevenueNet → new RevenueFromContractWithCustomer).
    """
    aliases = CONCEPT_ALIASES.get(field, [field])
    gaap = facts.get("facts", {}).get("us-gaap", {})

    # merged[fy] = (filed_date_str, value)
    merged: dict[int, tuple] = {}

    for concept in aliases:
        if concept not in gaap:
            continue
        usd_vals = gaap[concept].get("units", {}).get("USD", [])
        for rec in usd_vals:
            if rec.get("form") not in ("10-K", "10-K/A", "20-F", "20-F/A"):
                continue
            if rec.get("fp") != "FY":
                continue
            fy = rec.get("fy")
            if fy is None:
                continue
            filed = rec.get("filed", "")
            if fy not in merged or filed > merged[fy][0]:
                merged[fy] = (filed, rec.get("val"))

    return {fy: val for fy, (_, val) in merged.items()}


# ── Beneish feature computation ───────────────────────────────────────────────

def compute_beneish_row(series: dict, year_t: int) -> dict | None:
    """
    Compute all 8 Beneish components for year_t vs year_t-1.
    Returns None if fewer than 6 components can be computed.
    """
    year_tm1 = year_t - 1

    def v(field, yr):
        val = series.get(field, {}).get(yr)
        return float(val) if val is not None else None

    def safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    rev_t  = v("revenue", year_t)
    rev_m  = v("revenue", year_tm1)
    ta_t   = v("total_assets", year_t)
    ta_m   = v("total_assets", year_tm1)

    # Revenue and total assets are required for denominator safety
    if any(x is None or x == 0 for x in [rev_t, rev_m, ta_t, ta_m]):
        return None

    # COGS — fallback: Revenue - GrossProfit
    cogs_t = v("cogs", year_t) or (rev_t - v("gross_profit", year_t) if v("gross_profit", year_t) else None)
    cogs_m = v("cogs", year_tm1) or (rev_m - v("gross_profit", year_tm1) if v("gross_profit", year_tm1) else None)

    rec_t  = v("receivables",    year_t)
    rec_m  = v("receivables",    year_tm1)
    ca_t   = v("current_assets", year_t)
    ca_m   = v("current_assets", year_tm1)
    ppe_t  = v("ppe_net",        year_t)
    ppe_m  = v("ppe_net",        year_tm1)
    dep_t  = v("depreciation",   year_t)
    dep_m  = v("depreciation",   year_tm1)
    sga_t  = v("sga",            year_t)
    sga_m  = v("sga",            year_tm1)
    ltd_t  = v("lt_debt",        year_t) or 0.0
    ltd_m  = v("lt_debt",        year_tm1) or 0.0
    cl_t   = v("current_liab",   year_t) or 0.0
    cl_m   = v("current_liab",   year_tm1) or 0.0
    ni_t   = v("net_income",     year_t)
    cfo_t  = v("cfo",            year_t)

    # ── Compute each component ────────────────────────────────────────────────
    components: dict[str, float | None] = {}

    # DSRI
    dsr_t = safe_div(rec_t, rev_t)
    dsr_m = safe_div(rec_m, rev_m)
    components["DSRI"] = safe_div(dsr_t, dsr_m)

    # GMI  (prior gross margin / current gross margin)
    gm_t = safe_div(rev_t - (cogs_t or 0), rev_t) if cogs_t is not None else None
    gm_m = safe_div(rev_m - (cogs_m or 0), rev_m) if cogs_m is not None else None
    components["GMI"] = safe_div(gm_m, gm_t)

    # AQI
    def aq(ca, ppe, ta):
        if ca is None or ppe is None or ta is None or ta == 0:
            return None
        return 1.0 - (ca + ppe) / ta
    components["AQI"] = safe_div(aq(ca_t, ppe_t, ta_t), aq(ca_m, ppe_m, ta_m))

    # SGI
    components["SGI"] = safe_div(rev_t, rev_m)

    # DEPI
    def dep_rate(dep, ppe):
        if dep is None or ppe is None or (dep + ppe) == 0:
            return None
        return dep / (dep + ppe)
    components["DEPI"] = safe_div(dep_rate(dep_m, ppe_m), dep_rate(dep_t, ppe_t))

    # SGAI
    sga_ratio_t = safe_div(sga_t, rev_t)
    sga_ratio_m = safe_div(sga_m, rev_m)
    components["SGAI"] = safe_div(sga_ratio_t, sga_ratio_m)

    # LVGI
    lev_t = safe_div(ltd_t + cl_t, ta_t) if (ltd_t + cl_t) > 0 else None
    lev_m = safe_div(ltd_m + cl_m, ta_m) if (ltd_m + cl_m) > 0 else None
    components["LVGI"] = safe_div(lev_t, lev_m)

    # TATA
    components["TATA"] = safe_div((ni_t or 0) - (cfo_t or 0), ta_t) if (ni_t is not None or cfo_t is not None) else None

    # ── Accept row if >= 6 of 8 components present ────────────────────────────
    n_computed = sum(1 for v in components.values() if v is not None)
    if n_computed < 6:
        return None

    # Fill any missing with neutral values (1.0 for indices, 0.0 for TATA)
    result = {}
    for k, val in components.items():
        if val is not None:
            result[k] = round(float(val), 6)
        else:
            result[k] = 0.0 if k == "TATA" else 1.0

    return result


def compute_mscore(row: dict) -> float:
    return (-4.84
            + 0.920 * row["DSRI"]
            + 0.528 * row["GMI"]
            + 0.404 * row["AQI"]
            + 0.892 * row["SGI"]
            + 0.115 * row["DEPI"]
            - 0.172 * row["SGAI"]
            + 4.679 * row["TATA"]
            - 0.327 * row["LVGI"])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    records = []
    total = len(COMPANIES)

    for i, co in enumerate(COMPANIES):
        ticker = co["ticker"]
        cik    = co["cik"]
        label  = "FRAUD" if co["is_fraud"] else "clean"
        print(f"[{i+1:2d}/{total}] {ticker:6s} ({label}) ...", end=" ", flush=True)

        facts = fetch_company_facts(cik)
        if facts is None:
            print("SKIP (fetch failed)")
            continue

        # Pull all field series
        series = {field: get_annual_series(facts, field) for field in CONCEPT_ALIASES}

        # Try each year in the range until a row with ≥ 6 components is found
        row_features = None
        used_year = None
        for yr in co["fraud_year_range"]:
            row_features = compute_beneish_row(series, yr)
            if row_features:
                used_year = yr
                break

        if row_features is None:
            print(f"SKIP (insufficient data for years {co['fraud_year_range']})")
            continue

        row = {
            "company": ticker,
            "year": used_year,
            "is_fraud": co["is_fraud"],
            "sector": co["sector"],
            "scandal": co["scandal"],
            **row_features,
        }
        row["mscore"] = round(compute_mscore(row), 4)
        row["mscore_flag"] = 1 if row["mscore"] > -2.22 else 0
        records.append(row)
        print(f"OK  year={used_year}  M={row['mscore']:+.3f}")

    df = pd.DataFrame(records)
    df["anon_id"] = ["Firm-" + chr(65 + i % 26) + str(i + 1) for i in range(len(df))]

    out_path = PROCESSED_DIR / "fraud_dataset_real.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")
    print(f"  Fraud: {int(df['is_fraud'].sum())}  |  Clean: {int((df['is_fraud']==0).sum())}")
    print(f"  M-score range: {df['mscore'].min():.3f} to {df['mscore'].max():.3f}")
    return df


if __name__ == "__main__":
    # Clear stale cache so re-runs pick up corrected CIKs
    stale = ["CIK0000040554.json"]   # old wrong GE CIK
    for fname in stale:
        p = RAW_DIR / fname
        if p.exists():
            p.unlink()
            print(f"Removed stale cache: {fname}")

    print("Building real dataset from SEC EDGAR XBRL API...\n")
    df = build_dataset()

    print("\nFull results:")
    print(df[["company", "year", "is_fraud", "mscore", "mscore_flag"]].to_string(index=False))
