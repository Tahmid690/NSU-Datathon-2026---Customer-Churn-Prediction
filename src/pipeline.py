"""
FictiPay Churn Prediction Pipeline
===================================
Observation window: 2024-01-01 → 2024-03-31
Prediction window:  2024-04-01 → 2024-04-30
Target: AUC-ROC >= 0.986
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data/public")
OUT_DIR  = Path("./artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBS_END   = pd.Timestamp("2024-03-31")
OBS_START = pd.Timestamp("2024-01-01")

# ──────────────────────────────────────────────
# 1. LOAD LABELS & TEST IDs
# ──────────────────────────────────────────────
print("=== Loading labels & test IDs ===")
train_labels = pd.read_csv(DATA_DIR / "train_labels.csv")
test_ids     = pd.read_csv(DATA_DIR / "test.csv")

all_ids = pd.concat([
    train_labels[["ACCOUNT_ID"]],
    test_ids[["ACCOUNT_ID"]]
], ignore_index=True)

customer_set = set(all_ids["ACCOUNT_ID"])
print(f"  Train: {len(train_labels):,}  |  Test: {len(test_ids):,}  |  Total customers: {len(customer_set):,}")


# ──────────────────────────────────────────────
# 2. KYC FEATURES
# ──────────────────────────────────────────────
print("\n=== Processing KYC features ===")
kyc = pd.read_parquet(DATA_DIR / "kyc.parquet")
kyc = kyc[kyc["ACCOUNT_ID"].isin(customer_set)].copy()

kyc["ACCOUNT_OPEN_DATE"] = pd.to_datetime(kyc["ACCOUNT_OPEN_DATE"])
kyc["account_age_days"]  = (OBS_END - kyc["ACCOUNT_OPEN_DATE"]).dt.days
kyc["is_female"]         = (kyc["GENDER"] == "Female").astype(np.int8)

# Encode region and gender
region_map = {r: i for i, r in enumerate(kyc["REGION"].unique())}
kyc["region_enc"] = kyc["REGION"].map(region_map).astype(np.int16)

# Account open year/month
kyc["open_year"]  = kyc["ACCOUNT_OPEN_DATE"].dt.year.astype(np.int16)
kyc["open_month"] = kyc["ACCOUNT_OPEN_DATE"].dt.month.astype(np.int8)

kyc_feat = kyc[["ACCOUNT_ID", "account_age_days", "is_female",
                 "region_enc", "open_year", "open_month"]].copy()

print(f"  KYC features shape: {kyc_feat.shape}")
del kyc; gc.collect()


# ──────────────────────────────────────────────
# 3. TRANSACTION FEATURES
# ──────────────────────────────────────────────
print("\n=== Processing Transaction features ===")

TRX_MONTHS = ["2024-01", "2024-02", "2024-03"]
TRX_TYPES  = ["P2P", "MerchantPay", "BillPay", "CashIn", "CashOut"]

# Window boundaries (days before OBS_END)
W7  = OBS_END - pd.Timedelta(days=7)
W30 = OBS_END - pd.Timedelta(days=30)
W90 = OBS_END - pd.Timedelta(days=90)  # full period

# Collect per-month aggregations then merge
monthly_records = {m: [] for m in TRX_MONTHS}

agg_parts = []

for month_str in TRX_MONTHS:
    fpath = DATA_DIR / "transactions" / f"trx_{month_str}.parquet"
    print(f"  Loading {fpath.name} ...", end=" ", flush=True)
    trx = pd.read_parquet(fpath, columns=["TrxID", "TRX_DATETIME", "SRC_ACCOUNT", "TRX_TYPE", "TRX_AMT"])
    print(f"{len(trx):,} rows", flush=True)

    trx["TRX_DATETIME"] = pd.to_datetime(trx["TRX_DATETIME"])

    # Filter only customer source accounts
    trx = trx[trx["SRC_ACCOUNT"].isin(customer_set)].copy()
    trx.rename(columns={"SRC_ACCOUNT": "ACCOUNT_ID"}, inplace=True)

    # Month label for monthly trend
    trx["month"] = trx["TRX_DATETIME"].dt.month  # 1, 2, 3

    # Per-type flag columns
    for t in TRX_TYPES:
        trx[f"is_{t}"] = (trx["TRX_TYPE"] == t).astype(np.int8)

    agg_parts.append(trx)
    del trx; gc.collect()

print("  Concatenating all transactions ...", flush=True)
all_trx = pd.concat(agg_parts, ignore_index=True)
del agg_parts; gc.collect()

print(f"  Total customer transactions: {len(all_trx):,}")

# ── Global aggregates ──────────────────────────────────────────
print("  Computing global aggregates ...", flush=True)
grp = all_trx.groupby("ACCOUNT_ID")

trx_global = grp["TRX_AMT"].agg(
    trx_count_total="count",
    trx_amt_total="sum",
    trx_amt_mean="mean",
    trx_amt_std="std",
    trx_amt_median="median",
    trx_amt_max="max",
    trx_amt_min="min",
).reset_index()

# Active days
active_days = (all_trx.groupby("ACCOUNT_ID")["TRX_DATETIME"]
               .apply(lambda x: x.dt.date.nunique())
               .reset_index(name="active_days_total"))
trx_global = trx_global.merge(active_days, on="ACCOUNT_ID", how="left")

# Unique destinations
dst_all = pd.read_parquet(
    DATA_DIR / "transactions" / "trx_2024-01.parquet",
    columns=["SRC_ACCOUNT", "DST_ACCOUNT"]
)
for m in ["2024-02", "2024-03"]:
    dst_all = pd.concat([
        dst_all,
        pd.read_parquet(DATA_DIR / "transactions" / f"trx_{m}.parquet",
                        columns=["SRC_ACCOUNT", "DST_ACCOUNT"])
    ], ignore_index=True)

dst_all = dst_all[dst_all["SRC_ACCOUNT"].isin(customer_set)]
unique_dst = (dst_all.groupby("SRC_ACCOUNT")["DST_ACCOUNT"]
              .nunique()
              .reset_index(name="unique_dst_total"))
unique_dst.rename(columns={"SRC_ACCOUNT": "ACCOUNT_ID"}, inplace=True)
trx_global = trx_global.merge(unique_dst, on="ACCOUNT_ID", how="left")
del dst_all, unique_dst; gc.collect()

# ── Per-type counts ────────────────────────────────────────────
print("  Computing per-type counts ...", flush=True)
for t in TRX_TYPES:
    cnt = (all_trx[all_trx["TRX_TYPE"] == t]
           .groupby("ACCOUNT_ID")["TRX_AMT"]
           .agg(**{f"trx_{t}_count": "count", f"trx_{t}_amt": "sum"})
           .reset_index())
    trx_global = trx_global.merge(cnt, on="ACCOUNT_ID", how="left")
trx_global.fillna(0, inplace=True)

# Ratio features
total = trx_global["trx_count_total"].clip(lower=1)
for t in TRX_TYPES:
    trx_global[f"ratio_{t}"] = trx_global[f"trx_{t}_count"] / total

# CashIn/CashOut net ratio
trx_global["cashin_vs_cashout"] = (
    (trx_global["trx_CashIn_amt"] - trx_global["trx_CashOut_amt"]) /
    (trx_global["trx_CashIn_amt"] + trx_global["trx_CashOut_amt"] + 1)
)

# ── Recency features ───────────────────────────────────────────
print("  Computing recency ...", flush=True)
last_trx = (all_trx.groupby("ACCOUNT_ID")["TRX_DATETIME"]
            .max().reset_index(name="last_trx_date"))
first_trx = (all_trx.groupby("ACCOUNT_ID")["TRX_DATETIME"]
             .min().reset_index(name="first_trx_date"))

last_trx["recency_days"] = (OBS_END - last_trx["last_trx_date"]).dt.days
first_trx["first_seen_days"] = (OBS_END - first_trx["first_trx_date"]).dt.days

recency = last_trx[["ACCOUNT_ID","recency_days"]].merge(
    first_trx[["ACCOUNT_ID","first_seen_days"]], on="ACCOUNT_ID"
)
trx_global = trx_global.merge(recency, on="ACCOUNT_ID", how="left")
del last_trx, first_trx, recency; gc.collect()

# ── Sliding window: last 7 / 30 days ──────────────────────────
print("  Computing sliding window features ...", flush=True)
for window_days, label in [(7, "7d"), (30, "30d")]:
    cutoff = OBS_END - pd.Timedelta(days=window_days)
    w_trx  = all_trx[all_trx["TRX_DATETIME"] >= cutoff]
    w_agg  = w_trx.groupby("ACCOUNT_ID")["TRX_AMT"].agg(
        **{f"trx_count_{label}": "count",
           f"trx_amt_mean_{label}": "mean",
           f"trx_amt_sum_{label}": "sum"}
    ).reset_index()
    trx_global = trx_global.merge(w_agg, on="ACCOUNT_ID", how="left")

trx_global.fillna(0, inplace=True)

# ── Monthly trend features ─────────────────────────────────────
print("  Computing monthly trends ...", flush=True)
monthly_cnt = (all_trx.groupby(["ACCOUNT_ID", "month"])["TRX_AMT"]
               .agg(count="count", amt_sum="sum")
               .reset_index())

for mo, mo_label in [(1, "jan"), (2, "feb"), (3, "mar")]:
    mo_df = monthly_cnt[monthly_cnt["month"] == mo][["ACCOUNT_ID", "count", "amt_sum"]].copy()
    mo_df.rename(columns={"count": f"trx_count_{mo_label}",
                           "amt_sum": f"trx_amt_{mo_label}"}, inplace=True)
    trx_global = trx_global.merge(mo_df, on="ACCOUNT_ID", how="left")

trx_global.fillna(0, inplace=True)

# Month-over-month trends
trx_global["mom_count_jan_feb"] = trx_global["trx_count_feb"] - trx_global["trx_count_jan"]
trx_global["mom_count_feb_mar"] = trx_global["trx_count_mar"] - trx_global["trx_count_feb"]
trx_global["mom_trend_acceleration"] = trx_global["mom_count_feb_mar"] - trx_global["mom_count_jan_feb"]
trx_global["trx_count_q1_slope"] = (trx_global["trx_count_mar"] - trx_global["trx_count_jan"]) / 2.0

# Consecutive active months
def count_active_months(row):
    months_active = sum([row["trx_count_jan"] > 0,
                         row["trx_count_feb"] > 0,
                         row["trx_count_mar"] > 0])
    return months_active

trx_global["active_month_count"] = trx_global.apply(count_active_months, axis=1).astype(np.int8)

# Was active in March (most recent month)
trx_global["active_in_mar"] = (trx_global["trx_count_mar"] > 0).astype(np.int8)
trx_global["active_in_feb"] = (trx_global["trx_count_feb"] > 0).astype(np.int8)
trx_global["active_in_jan"] = (trx_global["trx_count_jan"] > 0).astype(np.int8)

# Transaction frequency (avg per active day)
trx_global["trx_freq_per_day"] = (
    trx_global["trx_count_total"] / (trx_global["active_days_total"].clip(lower=1))
)

# Amount per active day
trx_global["amt_per_active_day"] = (
    trx_global["trx_amt_total"] / (trx_global["active_days_total"].clip(lower=1))
)

del all_trx, monthly_cnt; gc.collect()
print(f"  Transaction features shape: {trx_global.shape}")


# ──────────────────────────────────────────────
# 4. BALANCE FEATURES
# ──────────────────────────────────────────────
print("\n=== Processing Balance features ===")

BAL_MONTHS = ["2024-01", "2024-02", "2024-03"]
bal_parts  = []

for month_str in BAL_MONTHS:
    fpath = DATA_DIR / "dayend_balance" / f"balance_{month_str}.parquet"
    print(f"  Loading {fpath.name} ...", end=" ", flush=True)
    bal = pd.read_parquet(fpath, columns=["ACCOUNT_ID", "DATE", "AVAILABLE_BALANCE"])
    bal = bal[bal["ACCOUNT_ID"].isin(customer_set)].copy()
    bal["DATE"] = pd.to_datetime(bal["DATE"])
    bal["month"] = bal["DATE"].dt.month
    print(f"{len(bal):,} rows", flush=True)
    bal_parts.append(bal)
    del bal; gc.collect()

print("  Concatenating balance data ...", flush=True)
all_bal = pd.concat(bal_parts, ignore_index=True)
del bal_parts; gc.collect()
print(f"  Total balance rows: {len(all_bal):,}")

# ── Global balance stats ───────────────────────────────────────
print("  Computing balance aggregates ...", flush=True)
grp_bal = all_bal.groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"]

bal_global = grp_bal.agg(
    bal_mean="mean",
    bal_std="std",
    bal_min="min",
    bal_max="max",
    bal_median="median",
).reset_index()

# Zero balance days
zero_bal = (all_bal.groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"]
            .apply(lambda x: (x == 0).sum())
            .reset_index(name="zero_bal_days"))
total_days = all_bal.groupby("ACCOUNT_ID")["DATE"].count().reset_index(name="total_bal_days")

bal_global = bal_global.merge(zero_bal, on="ACCOUNT_ID", how="left")
bal_global = bal_global.merge(total_days, on="ACCOUNT_ID", how="left")
bal_global["zero_bal_ratio"] = bal_global["zero_bal_days"] / bal_global["total_bal_days"].clip(lower=1)
bal_global["bal_cv"] = bal_global["bal_std"] / (bal_global["bal_mean"].abs() + 1)  # coefficient of variation

del zero_bal, total_days; gc.collect()

# ── Monthly balance trend ──────────────────────────────────────
print("  Computing monthly balance trends ...", flush=True)
monthly_bal = all_bal.groupby(["ACCOUNT_ID", "month"])["AVAILABLE_BALANCE"].agg(
    mean="mean", std="std"
).reset_index()

for mo, mo_label in [(1, "jan"), (2, "feb"), (3, "mar")]:
    mo_df = monthly_bal[monthly_bal["month"] == mo][["ACCOUNT_ID", "mean", "std"]].copy()
    mo_df.rename(columns={"mean": f"bal_mean_{mo_label}", "std": f"bal_std_{mo_label}"}, inplace=True)
    bal_global = bal_global.merge(mo_df, on="ACCOUNT_ID", how="left")

bal_global.fillna(0, inplace=True)

# Balance trend slope (Mar - Jan)
bal_global["bal_trend_slope"] = bal_global["bal_mean_mar"] - bal_global["bal_mean_jan"]
bal_global["bal_mom_feb_mar"]  = bal_global["bal_mean_mar"] - bal_global["bal_mean_feb"]

# Last 7 days balance
last7_bal  = all_bal[all_bal["DATE"] >= W7]
last7_mean = (last7_bal.groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"]
              .mean().reset_index(name="bal_mean_7d"))
bal_global = bal_global.merge(last7_mean, on="ACCOUNT_ID", how="left")

# Last 30 days balance
last30_bal  = all_bal[all_bal["DATE"] >= W30]
last30_mean = (last30_bal.groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"]
               .mean().reset_index(name="bal_mean_30d"))
bal_global = bal_global.merge(last30_mean, on="ACCOUNT_ID", how="left")
bal_global.fillna(0, inplace=True)

# Balance difference: recent vs earlier
bal_global["bal_recent_vs_total"] = bal_global["bal_mean_7d"] - bal_global["bal_mean"]

del all_bal, last7_bal, last30_bal, monthly_bal; gc.collect()
print(f"  Balance features shape: {bal_global.shape}")


# ──────────────────────────────────────────────
# 5. MERGE ALL FEATURES
# ──────────────────────────────────────────────
print("\n=== Merging feature tables ===")
feat = all_ids.copy()
feat = feat.merge(kyc_feat,   on="ACCOUNT_ID", how="left")
feat = feat.merge(trx_global, on="ACCOUNT_ID", how="left")
feat = feat.merge(bal_global, on="ACCOUNT_ID", how="left")

# Fill accounts with NO transactions at all (truly inactive)
trx_cols = [c for c in feat.columns if c.startswith("trx_") or c.startswith("ratio_") or
            c.startswith("mom_") or c in ["active_days_total","recency_days","first_seen_days",
                                           "unique_dst_total","cashin_vs_cashout","trx_freq_per_day",
                                           "amt_per_active_day","active_month_count",
                                           "active_in_mar","active_in_feb","active_in_jan",
                                           "trx_count_q1_slope","mom_trend_acceleration"]]

# For accounts with no transactions: recency = 91 (max), rest = 0
feat["recency_days"] = feat["recency_days"].fillna(91)
feat[trx_cols] = feat[trx_cols].fillna(0)

bal_cols = [c for c in feat.columns if c.startswith("bal_") or c == "zero_bal_days"
            or c == "total_bal_days" or c == "zero_bal_ratio"]
feat[bal_cols] = feat[bal_cols].fillna(0)

feat["account_age_days"] = feat["account_age_days"].fillna(feat["account_age_days"].median())

print(f"  Final feature matrix shape: {feat.shape}")
print(f"  Feature columns: {feat.shape[1] - 1}")

feat.to_parquet(OUT_DIR / "features_all.parquet", index=False)
print("  Saved features_all.parquet")
