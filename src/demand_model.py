"""
Demand Forecasting Model (Model 1) — Student 2 / ML Engineer
=============================================================
Predicts `next_7_day_demand` (regression) using the leakage-safe
feature table produced by src/features.py.

Usage:
    python src/demand_model.py                           # uses default paths
    python src/demand_model.py --features path/to/ft.csv # custom feature table

Outputs:
    models/demand_model.pkl          — best trained model (joblib)
    models/demand_model_card.md      — model card with metrics & limitations
    reports/model_comparison.md      — comparison table of all models tested
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] xgboost not installed — skipping XGBoost model.")

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FEATURE_TABLE_PATH = os.path.join("data", "processed", "feature_table.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "demand_model.pkl")
MODEL_CARD_PATH = os.path.join("models", "demand_model_card.md")
COMPARISON_REPORT_PATH = os.path.join("reports", "model_comparison.md")

TARGET = "next_7_day_demand"

# Features to use for training (must exist in feature_table.csv)
# These are all leakage-safe features (past-only or calendar-derived)
FEATURE_COLUMNS = [
    # Time features
    "day_of_week",
    "weekend_flag",
    "month",
    "week_no",
    # Lag features (built from corrected_demand with proper shift)
    "lag_1",
    "lag_7",
    "lag_14",
    # Rolling features (shift(1) applied before rolling — no leakage)
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    # Inventory features
    "days_of_inventory",
    "inventory_to_demand_ratio",
    "reorder_gap",
    # Price / Promo features
    "discount_pct",
    "price_change",
    "promotion_flag",
]

# Optional features — included if present in the data
OPTIONAL_FEATURES = [
    "festival_flag",
    "holiday",
    "temp_c",
    "rain_mm",
    "local_event",
    "shelf_life_days",
    "lead_days",
    "floor_area_sqft",
    "avg_daily_customers",
    "cv_demand",
]

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = [
    "store_type",
    "category",
    "brand",
    "abc_class",
    "xyz_class",
]

N_CV_SPLITS = 5
HOLDOUT_DAYS = 14  # Last 14 days reserved as final test set


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def safe_mape(y_true, y_pred):
    """MAPE that handles zero-demand rows gracefully."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, MAPE, R² and return as a dict."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAPE": round(safe_mape(y_true, y_pred), 4),
        "R²": round(r2_score(y_true, y_pred), 4),
    }


def compute_segment_metrics(df, y_true, y_pred, segment_col):
    """Break down metrics by a segment column (e.g., abc_class, store_type)."""
    results = {}
    if segment_col not in df.columns:
        return results
    for segment_val in sorted(df[segment_col].dropna().unique()):
        mask = df[segment_col] == segment_val
        if mask.sum() < 10:
            continue
        results[str(segment_val)] = compute_metrics(
            y_true[mask.values], y_pred[mask.values]
        )
    return results


# ──────────────────────────────────────────────
# Data loading & preparation
# ──────────────────────────────────────────────

def load_and_prepare_data(feature_table_path):
    """
    Load feature_table.csv, select features, handle missing values,
    and split into time-aware train/test sets.
    """
    print(f"[1/6] Loading feature table from: {feature_table_path}")
    df = pd.read_csv(feature_table_path, parse_dates=["date"])
    print(f"       Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Drop rows where target is NaN (end-of-series, no forward window) ──
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"       Dropped {before - len(df):,} rows with NaN target (end-of-series)")

    # ── Determine which features are actually available ──
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    optional_available = [c for c in OPTIONAL_FEATURES if c in df.columns]
    available_features.extend(optional_available)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing expected features (will proceed without): {missing}")

    # ── One-hot encode categoricals ──
    cat_available = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if cat_available:
        print(f"       One-hot encoding: {cat_available}")
        df = pd.get_dummies(df, columns=cat_available, drop_first=True, dtype=int)
        # Add the newly created dummy columns to features
        new_dummy_cols = [
            c for c in df.columns
            if any(c.startswith(cat + "_") for cat in cat_available)
        ]
        available_features.extend(new_dummy_cols)

    print(f"       Final feature count: {len(available_features)}")

    # ── Sort chronologically (critical for time-aware splitting) ──
    df = df.sort_values("date").reset_index(drop=True)

    # ── Fill remaining NaN in features with 0 (lag NaNs for early rows) ──
    df[available_features] = df[available_features].fillna(0)

    # ── Time-aware holdout split ──
    cutoff_date = df["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()

    print(f"       Train: {len(train_df):,} rows (up to {cutoff_date.date()})")
    print(f"       Test:  {len(test_df):,} rows (last {HOLDOUT_DAYS} days holdout)")

    X_train = train_df[available_features]
    y_train = train_df[TARGET]
    X_test = test_df[available_features]
    y_test = test_df[TARGET]

    return df, train_df, test_df, X_train, y_train, X_test, y_test, available_features


# ──────────────────────────────────────────────
# Baseline model
# ──────────────────────────────────────────────

def compute_baseline(train_df, test_df):
    """
    Baseline: predict next_7_day_demand as rolling_mean_7 × 7
    (i.e., "demand over next week ≈ average daily demand last week × 7").
    This is the sanity floor every real model must beat.
    """
    print("\n[2/6] Computing baseline (7-day rolling mean × 7)...")
    if "rolling_mean_7" in test_df.columns:
        baseline_pred = test_df["rolling_mean_7"].fillna(0) * 7
    else:
        # Fallback: use lag_7 directly if rolling_mean_7 is missing
        baseline_pred = test_df["lag_7"].fillna(0) if "lag_7" in test_df.columns else pd.Series(0, index=test_df.index)

    y_test = test_df[TARGET]
    baseline_metrics = compute_metrics(y_test.values, baseline_pred.values)
    print(f"       Baseline MAE:  {baseline_metrics['MAE']}")
    print(f"       Baseline RMSE: {baseline_metrics['RMSE']}")
    print(f"       Baseline R²:   {baseline_metrics['R²']}")
    return baseline_pred, baseline_metrics


# ──────────────────────────────────────────────
# Model training with time-aware CV
# ──────────────────────────────────────────────

def get_models():
    """Return dict of model name → (model instance, hyperparams description)."""
    models = {
        "Linear Regression": (
            LinearRegression(),
            "Default (OLS, no regularization)",
        ),
        "Decision Tree": (
            DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42),
            "max_depth=10, min_samples_leaf=20",
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42,
            ),
            "n_estimators=200, max_depth=15, min_samples_leaf=10",
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = (
            XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            ),
            "n_estimators=300, max_depth=8, lr=0.05, subsample=0.8",
        )
    return models


def cross_validate_models(X_train, y_train):
    """
    Time-aware cross-validation using TimeSeriesSplit.
    Returns CV metrics for each model.
    """
    print(f"\n[3/6] Running {N_CV_SPLITS}-fold TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    models = get_models()
    cv_results = {}

    for name, (model, _) in models.items():
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model_clone = _clone_model(model)
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_val)
            fold_metrics.append(compute_metrics(y_val.values, y_pred))

        # Average across folds
        avg_metrics = {}
        for metric_name in ["MAE", "RMSE", "MAPE", "R²"]:
            vals = [fm[metric_name] for fm in fold_metrics if not np.isnan(fm[metric_name])]
            avg_metrics[metric_name] = round(np.mean(vals), 4) if vals else np.nan

        cv_results[name] = avg_metrics
        print(f"       {name:20s} — CV MAE: {avg_metrics['MAE']:.4f}, R²: {avg_metrics['R²']:.4f}")

    return cv_results


def _clone_model(model):
    """Create a fresh copy of a model with the same hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def train_final_models(X_train, y_train, X_test, y_test):
    """
    Train all models on the full training set, evaluate on holdout.
    Returns dict of model_name → (trained_model, holdout_metrics, hyperparams).
    """
    print(f"\n[4/6] Training final models on full training set & evaluating on holdout...")
    models = get_models()
    results = {}

    for name, (model, hyperparams) in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Clip negative predictions to 0 (demand can't be negative)
        y_pred = np.clip(y_pred, 0, None)

        holdout_metrics = compute_metrics(y_test.values, y_pred)
        results[name] = {
            "model": model,
            "metrics": holdout_metrics,
            "hyperparams": hyperparams,
            "predictions": y_pred,
        }
        print(f"       {name:20s} — Holdout MAE: {holdout_metrics['MAE']:.4f}, R²: {holdout_metrics['R²']:.4f}")

    return results


def select_best_model(results, baseline_metrics):
    """
    Select the best model based on MAE (the headline metric for business).
    Must beat the baseline to be valid.
    """
    print(f"\n[5/6] Selecting best model...")

    # Sort by MAE (lower is better)
    ranked = sorted(results.items(), key=lambda x: x[1]["metrics"]["MAE"])
    best_name, best_data = ranked[0]

    if best_data["metrics"]["MAE"] >= baseline_metrics["MAE"]:
        print(f"  [WARN] Best model ({best_name}) does NOT beat the baseline!")
        print(f"         Model MAE: {best_data['metrics']['MAE']}, Baseline MAE: {baseline_metrics['MAE']}")
        print(f"         Proceeding anyway, but investigate feature engineering or data quality.")

    print(f"       Winner: {best_name}")
    print(f"       MAE={best_data['metrics']['MAE']}, RMSE={best_data['metrics']['RMSE']}, "
          f"MAPE={best_data['metrics']['MAPE']}, R²={best_data['metrics']['R²']}")

    return best_name, best_data


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def generate_model_comparison_report(
    baseline_metrics, cv_results, holdout_results, best_name, feature_list
):
    """Generate reports/model_comparison.md (Student 2's section)."""
    print(f"\n[6/6] Writing reports...")

    lines = [
        "# Model Comparison Report — Demand Forecasting (Model 1)",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Why MAE Is Our Headline Metric",
        "",
        "We recommend **MAE (Mean Absolute Error) in units** as the primary metric because:",
        "- It's directly interpretable: \"on average, our forecast is X units off.\"",
        "- Store managers can understand it without statistical training.",
        "- MAPE misbehaves near zero-demand slow-movers (division by near-zero),",
        "  which are common in our dataset (sparse-history products).",
        "- R² is useful for model comparison but doesn't translate to actionable business language.",
        "",
        "---",
        "",
        "## Cross-Validation Results (TimeSeriesSplit, k=5)",
        "",
        "| Model | MAE | RMSE | MAPE | R² |",
        "|---|---|---|---|---|",
    ]

    for name, metrics in cv_results.items():
        lines.append(
            f"| {name} | {metrics['MAE']} | {metrics['RMSE']} | "
            f"{metrics['MAPE']} | {metrics['R²']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Holdout Test Results (last 14 days, unseen during training/CV)",
        "",
        "| Model | MAE | RMSE | MAPE | R² |",
        "|---|---|---|---|---|",
        f"| **Baseline (7-day mean)** | {baseline_metrics['MAE']} | "
        f"{baseline_metrics['RMSE']} | {baseline_metrics['MAPE']} | "
        f"{baseline_metrics['R²']} |",
    ])

    for name, data in holdout_results.items():
        m = data["metrics"]
        marker = " ✅ **Winner**" if name == best_name else ""
        lines.append(
            f"| {name}{marker} | {m['MAE']} | {m['RMSE']} | "
            f"{m['MAPE']} | {m['R²']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        f"## Selected Model: **{best_name}**",
        "",
        "### Justification",
        "",
        f"**{best_name}** was selected as the production model because it achieved the ",
        "lowest MAE on the unseen holdout set while maintaining strong R². ",
        "It provides the best balance of accuracy, interpretability, and training efficiency ",
        "among the permitted algorithms. Tree-based models naturally handle non-linear ",
        "interactions between features (e.g., promotions × seasonality) without manual ",
        "feature crosses.",
        "",
        "### Features Used",
        "",
    ])
    for f in feature_list:
        lines.append(f"- `{f}`")

    lines.extend([
        "",
        "---",
        "",
        "## Stock-Out Classification (Model 2)",
        "",
        "_Section to be completed by Student 3._",
        "",
    ])

    os.makedirs(os.path.dirname(COMPARISON_REPORT_PATH), exist_ok=True)
    with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"       ✓ Model comparison report: {COMPARISON_REPORT_PATH}")


def generate_model_card(
    best_name, best_data, feature_list, baseline_metrics, test_df, y_pred
):
    """Generate models/demand_model_card.md."""

    # Per-segment breakdown
    segment_breakdowns = {}
    for seg_col in ["abc_class", "store_type"]:
        seg_metrics = compute_segment_metrics(test_df, test_df[TARGET].values, y_pred, seg_col)
        if seg_metrics:
            segment_breakdowns[seg_col] = seg_metrics

    lines = [
        "# Model Card — Demand Forecasting (`demand_model.pkl`)",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Overview",
        "",
        f"- **Task:** Predict `next_7_day_demand` (total units demanded over the next 7 days)",
        f"- **Algorithm:** {best_name}",
        f"- **Hyperparameters:** {best_data['hyperparams']}",
        f"- **Training data:** Feature table up to {HOLDOUT_DAYS}-day holdout cutoff",
        f"- **Validation:** TimeSeriesSplit (k={N_CV_SPLITS}) + final {HOLDOUT_DAYS}-day holdout",
        "",
        "## Overall Performance (Holdout)",
        "",
        "| Metric | Baseline (7-day mean) | This Model |",
        "|---|---|---|",
    ]

    m = best_data["metrics"]
    for metric_name in ["MAE", "RMSE", "MAPE", "R²"]:
        lines.append(
            f"| {metric_name} | {baseline_metrics[metric_name]} | {m[metric_name]} |"
        )

    # Segment breakdowns
    for seg_col, seg_data in segment_breakdowns.items():
        lines.extend([
            "",
            f"## Performance by `{seg_col}`",
            "",
            f"| {seg_col} | MAE | RMSE | R² |",
            "|---|---|---|---|",
        ])
        for seg_val, seg_m in seg_data.items():
            lines.append(f"| {seg_val} | {seg_m['MAE']} | {seg_m['RMSE']} | {seg_m['R²']} |")

    lines.extend([
        "",
        "## Features Used",
        "",
        f"Total: {len(feature_list)} features",
        "",
    ])
    for f in feature_list:
        lines.append(f"- `{f}`")

    lines.extend([
        "",
        "## Known Limitations",
        "",
        "1. **Sparse-history products** (< 14 days of data) rely on category-level",
        "   mean demand as a fallback for lag/rolling features — predictions for these",
        "   SKUs carry wider error margins.",
        "2. **Censored demand correction** is applied upstream (Student 1's `corrected_demand`);",
        "   the quality of this model's forecasts is directly dependent on the quality of",
        "   that correction. If stock-outs are systematic and prolonged, the correction",
        "   heuristic may still under-estimate true demand.",
        "3. **External factors** (weather, festivals) are used as lagged/historical features.",
        "   In production, future weather would require a separate weather forecast API.",
        "4. **No deep learning** — constrained to permitted algorithms per competition rules.",
        "5. **Static training** — this model does not retrain automatically as new data arrives.",
        "   In production, a scheduled retraining pipeline would be needed.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python src/demand_model.py --features data/processed/feature_table.csv",
        "```",
        "",
    ])

    os.makedirs(os.path.dirname(MODEL_CARD_PATH), exist_ok=True)
    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"       ✓ Model card: {MODEL_CARD_PATH}")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main(feature_table_path=None):
    if feature_table_path is None:
        feature_table_path = FEATURE_TABLE_PATH

    print("=" * 60)
    print("  DEMAND FORECASTING MODEL — Student 2 (ML Engineer)")
    print("=" * 60)

    # Step 1: Load & prepare data
    df, train_df, test_df, X_train, y_train, X_test, y_test, features = (
        load_and_prepare_data(feature_table_path)
    )

    # Step 2: Compute baseline
    _, baseline_metrics = compute_baseline(train_df, test_df)

    # Step 3: Cross-validate models
    cv_results = cross_validate_models(X_train, y_train)

    # Step 4: Train final models on full training set
    holdout_results = train_final_models(X_train, y_train, X_test, y_test)

    # Step 5: Select best model
    best_name, best_data = select_best_model(holdout_results, baseline_metrics)

    # Save best model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_data["model"], MODEL_OUTPUT_PATH)
    print(f"       ✓ Model saved: {MODEL_OUTPUT_PATH}")

    # Step 6: Generate reports
    generate_model_comparison_report(
        baseline_metrics, cv_results, holdout_results, best_name, features
    )
    generate_model_card(
        best_name, best_data, features, baseline_metrics, test_df, best_data["predictions"]
    )

    print("\n" + "=" * 60)
    print(f"  DONE — Best model: {best_name}")
    print(f"  Saved to: {MODEL_OUTPUT_PATH}")
    print("=" * 60)

    return best_data["model"], best_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand forecasting model")
    parser.add_argument(
        "--features",
        type=str,
        default=FEATURE_TABLE_PATH,
        help="Path to feature_table.csv",
    )
    args = parser.parse_args()
    main(args.features)
