"""
Quick Model Tester — Check demand predictions for any Store × Product
=====================================================================
Usage:
    python src/test_model.py                        # interactive mode
    python src/test_model.py --store S01 --product P101  # specific query
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Paths
MODEL_PATH = os.path.join("models", "demand_model.pkl")
FEATURE_TABLE_PATH = os.path.join("data", "processed", "feature_table.csv")


def load_model_and_data():
    print("Loading model and feature table...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURE_TABLE_PATH, parse_dates=["date"])
    print(f"  Model: {type(model).__name__}")
    print(f"  Data:  {len(df):,} rows")
    return model, df


def get_feature_columns(df):
    """Get the same feature columns used during training."""
    FEATURE_COLS = [
        "day_of_week", "weekend_flag", "month", "week_no",
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "days_of_inventory", "inventory_to_demand_ratio", "reorder_gap",
        "discount_pct", "price_change", "promotion_flag",
    ]
    optional = [
        "festival_flag", "holiday", "temp_c", "rain_mm", "local_event",
        "shelf_life_days", "lead_days", "floor_area_sqft", "avg_daily_customers",
        "cv_demand",
    ]
    features = [c for c in FEATURE_COLS + optional if c in df.columns]

    # One-hot encode categoricals (must match training)
    cat_cols = [c for c in ["store_type", "category", "brand", "abc_class", "xyz_class"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        dummy_cols = [c for c in df.columns if any(c.startswith(cat + "_") for cat in cat_cols)]
        features.extend(dummy_cols)

    return df, features


def predict_for_store_product(model, df, features, store_id, product_id):
    """Show predictions for a specific store × product."""
    mask = (df["store_id"] == store_id) & (df["product_id"] == product_id)
    subset = df[mask].copy()

    if len(subset) == 0:
        print(f"\n  No data found for Store={store_id}, Product={product_id}")
        print(f"  Available stores:  {sorted(df['store_id'].unique())}")
        print(f"  Available products (sample): {sorted(df['product_id'].unique())[:10]}")
        return

    subset = subset.sort_values("date")
    subset[features] = subset[features].fillna(0)

    # Predict
    subset["predicted_demand"] = np.clip(model.predict(subset[features]), 0, None)

    # Show last 10 days
    display_cols = ["date", "store_id", "product_id"]
    if "corrected_demand" in subset.columns:
        display_cols.append("corrected_demand")
    if "next_7_day_demand" in subset.columns:
        display_cols.append("next_7_day_demand")
    display_cols.append("predicted_demand")
    if "rolling_mean_7" in subset.columns:
        display_cols.append("rolling_mean_7")
    if "closing_stock" in subset.columns:
        display_cols.append("closing_stock")

    last_10 = subset[display_cols].tail(10)

    print(f"\n{'='*70}")
    print(f"  Store: {store_id} | Product: {product_id}")
    print(f"  Total records: {len(subset)} | Showing last 10 days")
    print(f"{'='*70}")
    print(last_10.to_string(index=False))

    # Summary
    if "next_7_day_demand" in subset.columns:
        valid = subset.dropna(subset=["next_7_day_demand"])
        if len(valid) > 0:
            mae = np.mean(np.abs(valid["next_7_day_demand"] - valid["predicted_demand"]))
            print(f"\n  MAE for this product: {mae:.2f} units")

    # Latest prediction
    latest = subset.iloc[-1]
    print(f"\n  Latest prediction ({latest['date'].date()}):")
    print(f"    Predicted next 7-day demand: {latest['predicted_demand']:.1f} units")
    if "closing_stock" in latest:
        stock = latest["closing_stock"]
        pred = latest["predicted_demand"]
        gap = stock - pred
        if gap < 0:
            print(f"    Current stock: {stock:.0f} | SHORTFALL of {abs(gap):.0f} units!")
        else:
            print(f"    Current stock: {stock:.0f} | Surplus of {gap:.0f} units")


def interactive_mode(model, df, features):
    """Let user pick store and product interactively."""
    stores = sorted(df["store_id"].unique())
    products = sorted(df["product_id"].unique())

    print(f"\nAvailable Stores:  {stores}")
    print(f"Available Products ({len(products)} total): {products[:10]}... \n")

    while True:
        store = input("Enter Store ID (e.g., S01) or 'quit': ").strip().upper()
        if store.lower() == "quit":
            break

        product = input("Enter Product ID (e.g., P101) or 'quit': ").strip().upper()
        if product.lower() == "quit":
            break

        predict_for_store_product(model, df.copy(), features, store, product)
        print()


def main():
    parser = argparse.ArgumentParser(description="Test demand model predictions")
    parser.add_argument("--store", type=str, help="Store ID (e.g., S01)")
    parser.add_argument("--product", type=str, help="Product ID (e.g., P101)")
    args = parser.parse_args()

    model, df = load_model_and_data()
    df, features = get_feature_columns(df)

    if args.store and args.product:
        predict_for_store_product(model, df, features, args.store, args.product)
    else:
        interactive_mode(model, df, features)


if __name__ == "__main__":
    main()
