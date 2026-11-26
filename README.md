# FictiPay Customer Churn Prediction

**bKash presents NSUCEC Datathon (Round 1)** — Predicting 30-day customer inactivity on a mobile wallet platform from 200M+ transactions, 360M+ daily balances, and KYC profiles, paired with Optuna-tuned LightGBM and SHAP explainability.

![Competition](https://img.shields.io/badge/bKash%20%C3%97%20NSUCEC-Datathon%20Round%201-E2136E)
![Task](https://img.shields.io/badge/Task-30--Day%20Churn%20Prediction-brightgreen)
![Metric](https://img.shields.io/badge/Metric-AUC--ROC-blue)
![Stack](https://img.shields.io/badge/Stack-LightGBM%20%7C%20Optuna%20%7C%20SHAP%20%7C%20Pandas-orange)

---

## Overview

FictiPay is a digital wallet platform serving millions of retail, merchant, and biller accounts. In Round 1, the goal is to predict which retail customers will become **inactive over the next 30 days** (`2024-04-01 → 2024-04-30`), using features engineered exclusively from the 3-month observation window (`2024-01-01 → 2024-03-31`).

### Problem & Data Scale
- **Churn Definition:** `CHURN = 1` if a customer makes **zero transactions** in April 2024 (`2024-04-01` to `2024-04-30`); `CHURN = 0` otherwise.
- **Dataset Size (~1.8 GB compressed Parquet):**
  - `transactions/trx_2024-01..03.parquet`: **~200M rows** (`TrxID`, `TRX_DATETIME`, `SRC_ACCOUNT`, `DST_ACCOUNT`, `TRX_TYPE`, `TRX_AMT`)
  - `dayend_balance/balance_2024-01..03.parquet`: **~360M rows** (`ACCOUNT_ID`, `DATE`, `AVAILABLE_BALANCE`)
  - `kyc.parquet`: **~2M rows** (`ACCOUNT_ID`, `ACCOUNT_TYPE`, `ACCOUNT_OPEN_DATE`, `GENDER`, `REGION`)
  - `train_labels.csv`: **595,000 customer accounts** (`75,435` churned — **12.7%**, `519,565` retained — **87.3%**)
  - `test.csv`: **255,000 customer accounts**
- **Evaluation Metric:** **AUC-ROC** on `CHURN_PROB ∈ [0, 1]`.
- **Full Dataset Specification:** See [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md).

---

## Solution Architecture

```text
Raw Parquet Tables (~200M txns, ~360M balances, ~2M KYC)
        │
        ▼
[1] src/pipeline.py
    • Month-by-month customer-filtered Parquet loading & aggregation
    • 60+ engineered features across KYC, Transaction, Trend & Balance domains
    • Explicit handling of zero-activity accounts (recency = 91d, zero fills)
    • Output: artifacts/features_all.parquet
        │
        ▼
[2] src/train.py
    • Class-imbalance weighting (`scale_pos_weight ≈ 6.89`)
    • Optuna TPE tuning (30 trials × 5-Fold Stratified CV, maximizing AUC-ROC)
    • 5-Fold LightGBM ensemble with Out-Of-Fold (OOF) evaluation
      (AUC-ROC, Average Precision, Precision@10%, Recall@10%)
    • SHAP TreeExplainer global bar & beeswarm plots + gain/split importances
    • Outputs: artifacts/predictions.csv, artifacts/metrics_summary.csv, results/explainability/*
```

---

## Feature Engineering Breakdown

[`src/pipeline.py`](src/pipeline.py) builds a unified customer-level feature matrix from three raw data sources:

| Domain | Key Engineered Features |
|:---|:---|
| **KYC & Tenure** | `account_age_days`, `open_year`, `open_month`, `is_female`, encoded `region_enc` |
| **Transaction Volume & Diversity** | Total count, sum/mean/std/median/min/max amounts, `active_days_total`, `unique_dst_total`, `trx_freq_per_day`, `amt_per_active_day` |
| **Transaction Types & Flow** | Per-type counts, amounts, and share ratios for `P2P`, `MerchantPay`, `BillPay`, `CashIn`, `CashOut`, plus normalized `cashin_vs_cashout` net flow |
| **Recency & Sliding Windows** | `recency_days`, `first_seen_days`, and rolling **7-day** & **30-day** counts, means, and sums |
| **Monthly Trends & Momentum** | Jan/Feb/Mar counts & amounts, MoM deltas (`mom_count_jan_feb`, `mom_count_feb_mar`), `mom_trend_acceleration`, `trx_count_q1_slope`, `active_month_count`, per-month activity flags |
| **Day-End Balance Dynamics** | Global mean/std/min/max/median, `bal_cv`, `zero_bal_days` & `zero_bal_ratio`, Jan/Feb/Mar balance means & stds, `bal_trend_slope`, `bal_mom_feb_mar`, 7d/30d recent balance means, and `bal_recent_vs_total` |

---

## Modeling & Explainability

[`src/train.py`](src/train.py) implements an end-to-end training, evaluation, and interpretability pipeline:

1. **Imbalance-Aware LightGBM:** Computes `scale_pos_weight = (1 - p) / p` (~6.9×) to counter the 12.7% churn rate.
2. **Hyperparameter Optimization:** 30-trial Optuna TPE search evaluated via **5-Fold Stratified CV** over `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, and `min_split_gain`.
3. **Diagnostic & Business Metrics:** Reports OOF **AUC-ROC**, **Average Precision**, **Precision@10%**, and **Recall@10%** (targeting the top decile of high-risk customers for retention campaigns).
4. **SHAP Explainability Suite:** Automatically generates and saves under `results/explainability/`:
   - `shap_summary_bar.png` & `shap_beeswarm.png` (Top 25 drivers on 5,000-customer sample)
   - `feature_importance.png` & `feature_importance.csv` (gain & split importances)
   - `tuning_progress.png` (Optuna trial trajectory) & `oof_roc_curve.png` (OOF probability distribution + ROC curve)

---

## Repository Structure

```text
.
├── README.md                    # Project showcase & documentation
├── .gitignore                   # Excludes raw data (data/), parquet artifacts, submissions & caches
├── requirements.txt             # Python dependencies
├── docs/
│   └── DATASET_SPEC.md          # Official competition dataset & problem specification
└── src/
    ├── pipeline.py              # End-to-end feature engineering pipeline
    └── train.py                 # Optuna-tuned 5-Fold LightGBM, SHAP explainability & submission
```

> **Note on Dataset Files:** The raw FictiPay Parquet dataset (~1.8 GB) and generated artifacts (`artifacts/`) are excluded from Git via `.gitignore` to stay well within GitHub's 100 MB file limit. Place the competition's `public/` directory at `data/public/` to reproduce locally.

---

## Reproducing & Usage

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place Competition Data
Place the competition's `public/` folder inside `data/`:
```text
data/public/
├── kyc.parquet
├── train_labels.csv
├── test.csv
├── sample_submission.csv
├── dayend_balance/
│   ├── balance_2024-01.parquet
│   ├── balance_2024-02.parquet
│   └── balance_2024-03.parquet
└── transactions/
    ├── trx_2024-01.parquet
    ├── trx_2024-02.parquet
    └── trx_2024-03.parquet
```

### 3. Run the End-to-End Pipeline
Run both stages from the repository root:
```bash
# Stage 1: Build customer feature matrix -> artifacts/features_all.parquet
python3 src/pipeline.py

# Stage 2: Optuna tuning + 5-Fold LightGBM + SHAP plots -> artifacts/predictions.csv
python3 src/train.py
```

---

*Built for **bKash presents NSUCEC Datathon (Round 1)** — FictiPay Customer Churn Prediction.*
