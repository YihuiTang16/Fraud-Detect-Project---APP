# Financial Fraud Detection Web App

This project is a simple Streamlit app that tries to detect potential financial fraud using accounting ratios (Beneish M-score) and a basic machine learning model.

I also added a small interactive game where users can guess whether a company is fraudulent or not, and then compare their answers with the model.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## What I Built

The project has three main parts:

* **Dashboard**
  Shows fraud scores, key financial ratios, and lets you explore each company

* **Game page**
  You guess whether a company is fraudulent (10 rounds), then see how the model did

* **Insights page**
  Compares human vs. model accuracy and highlights common mistakes

---

## Data

The data comes from the SEC EDGAR XBRL API (official company filings).

I manually constructed a small dataset (~30 companies):

* 8 confirmed fraud cases (based on SEC enforcement actions)
* 22 non-fraud companies (mostly large public firms)

The dataset is saved in:

```
data/fraud_dataset_real.csv
```

---

## Model

### 1. Beneish M-Score

I used the standard Beneish M-score formula from finance literature.
The idea is: if the score is higher than −2.22, the company might be manipulating earnings.

### 2. Logistic Regression

I also trained a simple logistic regression model using the same features.
Used standard scaling + cross-validation.

Result:

* Accuracy is around **70%+**
* Not perfect (which is expected for real-world data)

---

## What I Found (Important)

One interesting thing is that many fraud companies in this dataset are **not flagged by M-score**.

This is probably because:

* M-score mainly detects revenue/receivables manipulation
* But modern fraud cases are more complex (non-GAAP metrics, reserves, disclosures, etc.)

So the model works, but has clear limitations.

---

## Project Structure

```
app/            → Streamlit app
data/           → dataset
notebooks/      → modeling notebook
requirements.txt
README.md
```

---

## Notebook

The notebook (`fraud_detection_modeling.ipynb`) shows the full process:

* data preparation
* feature calculation
* model training
* evaluation

It can run on Colab.

---

## Tech Used

* Python
* pandas
* scikit-learn
* streamlit
* plotly

---

## Reference

Beneish (1999), *The Detection of Earnings Manipulation*
