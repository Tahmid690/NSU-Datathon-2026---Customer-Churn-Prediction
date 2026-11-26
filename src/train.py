"""
Model Training, Tuning, SHAP, and Submission
"""
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("./data/public")
OUT_DIR  = Path("./artifacts")
EXPL_DIR = Path("./results/explainability")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPL_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 1. LOAD FEATURES
# ──────────────────────────────────────────────
print("=== Loading feature matrix ===")
feat        = pd.read_parquet(OUT_DIR / "features_all.parquet")
train_lbl   = pd.read_csv(DATA_DIR / "train_labels.csv")
test_ids    = pd.read_csv(DATA_DIR / "test.csv")

train_df = feat.merge(train_lbl, on="ACCOUNT_ID", how="inner")
test_df  = feat.merge(test_ids,  on="ACCOUNT_ID", how="inner")

print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
print(f"  Churn rate: {train_df['CHURN'].mean()*100:.2f}%")

# Feature columns (exclude ID and label)
FEAT_COLS = [c for c in feat.columns if c != "ACCOUNT_ID"]
print(f"  Feature count: {len(FEAT_COLS)}")

X_train = train_df[FEAT_COLS].astype(np.float32)
y_train = train_df["CHURN"].values
X_test  = test_df[FEAT_COLS].astype(np.float32)

del feat; gc.collect()


# ──────────────────────────────────────────────
# 2. CLASS IMBALANCE — quantify
# ──────────────────────────────────────────────
churn_rate = y_train.mean()
scale_pos  = (1 - churn_rate) / churn_rate  # ~7x weight for churn
print(f"\n=== Class Imbalance ===")
print(f"  Churn=1: {y_train.sum():,}  ({churn_rate*100:.2f}%)")
print(f"  Churn=0: {(1-y_train).sum():,}  ({(1-churn_rate)*100:.2f}%)")
print(f"  scale_pos_weight (class weight): {scale_pos:.2f}")


# ──────────────────────────────────────────────
# 3. OPTUNA HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
print("\n=== Optuna Hyperparameter Tuning (30 trials) ===")

CV_FOLDS = 5
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

trial_scores = []

def lgb_cv_score(params, X, y):
    scores = []
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        cb_params = dict(params)
        n_rounds = cb_params.pop("n_estimators", 1000)

        model = lgb.train(
            cb_params,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        pred = model.predict(X_val)
        scores.append(roc_auc_score(y_val, pred))

    return np.mean(scores)


def objective(trial):
    params = {
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "n_jobs":            -1,
        "seed":              42,
        "verbosity":         -1,
        "scale_pos_weight":  scale_pos,

        "n_estimators":      trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 63, 511),
        "max_depth":         trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
    }
    score = lgb_cv_score(params, X_train, y_train)
    trial_scores.append(score)
    return score


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "n_jobs":           -1,
    "seed":             42,
    "verbosity":        -1,
    "scale_pos_weight": scale_pos,
})

print(f"\n  Best CV AUC: {study.best_value:.6f}")
print(f"  Best params: {study.best_params}")

# ── Tuning progress plot ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(trial_scores)+1), trial_scores, "o-", color="#4A90D9", alpha=0.7)
ax.axhline(max(trial_scores), color="red", linestyle="--", label=f"Best: {max(trial_scores):.5f}")
ax.set_xlabel("Trial")
ax.set_ylabel("CV AUC-ROC")
ax.set_title("Optuna Tuning Progress")
ax.legend()
plt.tight_layout()
plt.savefig(EXPL_DIR / "tuning_progress.png", dpi=150)
plt.close()
print("  Saved tuning_progress.png")


# ──────────────────────────────────────────────
# 4. FINAL MODEL TRAINING (full CV + OOF preds)
# ──────────────────────────────────────────────
print("\n=== Final Model Training (5-fold CV) ===")

n_rounds = best_params.pop("n_estimators", 1000)

oof_preds  = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_models = []
fold_aucs   = []

for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds        += model.predict(X_test) / CV_FOLDS
    fold_models.append(model)

    auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_aucs.append(auc)
    print(f"  Fold {fold_i+1}/{CV_FOLDS}  AUC={auc:.6f}  n_trees={model.num_trees()}")

overall_auc = roc_auc_score(y_train, oof_preds)
overall_ap  = average_precision_score(y_train, oof_preds)

# Precision/Recall@10%
threshold_10pct = np.percentile(oof_preds, 90)
y_top10  = (oof_preds >= threshold_10pct).astype(int)
prec_10  = precision_score(y_train, y_top10, zero_division=0)
rec_10   = recall_score(y_train, y_top10, zero_division=0)

print(f"\n  === OOF Metrics ===")
print(f"  AUC-ROC:         {overall_auc:.6f}")
print(f"  Avg Precision:   {overall_ap:.6f}")
print(f"  Precision@10%:   {prec_10:.6f}")
print(f"  Recall@10%:      {rec_10:.6f}")
print(f"  Fold AUCs: {[f'{a:.5f}' for a in fold_aucs]}")


# ──────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ──────────────────────────────────────────────
print("\n=== SHAP Explainability ===")
explainer   = shap.TreeExplainer(fold_models[0])
shap_sample = X_train.sample(min(5000, len(X_train)), random_state=42)
shap_values = explainer.shap_values(shap_sample)

# Handle list output (binary classification)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# SHAP summary bar plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_vals, shap_sample, plot_type="bar",
                  max_display=25, show=False)
plt.title("SHAP Feature Importance (Top 25)")
plt.tight_layout()
plt.savefig(EXPL_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# SHAP beeswarm plot
plt.figure(figsize=(12, 12))
shap.summary_plot(shap_vals, shap_sample, max_display=25, show=False)
plt.title("SHAP Beeswarm Plot (Top 25)")
plt.tight_layout()
plt.savefig(EXPL_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Saved shap_summary_bar.png and shap_beeswarm.png")

# Feature importance from best model
fi = pd.DataFrame({
    "feature": FEAT_COLS,
    "importance_gain": fold_models[0].feature_importance(importance_type="gain"),
    "importance_split": fold_models[0].feature_importance(importance_type="split"),
})
fi = fi.sort_values("importance_gain", ascending=False).reset_index(drop=True)
fi.to_csv(EXPL_DIR / "feature_importance.csv", index=False)
print("  Saved feature_importance.csv")

# Feature importance plot
fig, ax = plt.subplots(figsize=(12, 10))
top_fi = fi.head(30)
ax.barh(top_fi["feature"][::-1], top_fi["importance_gain"][::-1], color="#4A90D9")
ax.set_xlabel("Gain")
ax.set_title("Top 30 Features by Gain")
plt.tight_layout()
plt.savefig(EXPL_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved feature_importance.png")


# ──────────────────────────────────────────────
# 6. SUBMISSION FILE
# ──────────────────────────────────────────────
print("\n=== Generating Submission ===")
submission = pd.DataFrame({
    "ACCOUNT_ID": test_df["ACCOUNT_ID"].values,
    "CHURN_PROB": test_preds,
})
submission.to_csv(OUT_DIR / "predictions.csv", index=False)
print(f"  Saved predictions.csv ({len(submission):,} rows)")
print(f"  Pred stats: min={test_preds.min():.4f}  max={test_preds.max():.4f}  mean={test_preds.mean():.4f}")


# ──────────────────────────────────────────────
# 7. PERFORMANCE REPORT
# ──────────────────────────────────────────────
print("\n=== Performance Report ===")

# OOF distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(oof_preds[y_train==0], bins=50, alpha=0.7, label="Non-Churn", color="#4A90D9")
axes[0].hist(oof_preds[y_train==1], bins=50, alpha=0.7, label="Churn",     color="#E8543A")
axes[0].set_xlabel("Predicted Churn Probability")
axes[0].set_ylabel("Count")
axes[0].set_title("OOF Prediction Distribution")
axes[0].legend()

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_train, oof_preds)
axes[1].plot(fpr, tpr, color="#4A90D9", label=f"AUC={overall_auc:.5f}")
axes[1].plot([0,1],[0,1], "k--", alpha=0.3)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("OOF ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig(EXPL_DIR / "oof_roc_curve.png", dpi=150)
plt.close()

# Save metrics summary
metrics_df = pd.DataFrame({
    "metric": ["AUC-ROC", "Avg Precision", "Precision@10%", "Recall@10%",
                "Fold1 AUC", "Fold2 AUC", "Fold3 AUC", "Fold4 AUC", "Fold5 AUC"],
    "value":  [overall_auc, overall_ap, prec_10, rec_10] + fold_aucs
})
metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
print(metrics_df.to_string(index=False))

print("\n✅ Pipeline complete!")
print(f"   AUC-ROC:       {overall_auc:.6f}")
print(f"   Avg Precision: {overall_ap:.6f}")
print(f"   Precision@10%: {prec_10:.6f}")
print(f"   Recall@10%:    {rec_10:.6f}")
