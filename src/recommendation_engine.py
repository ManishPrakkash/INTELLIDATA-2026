"""
Recommendation Engine (Student 3 / Decision Intelligence).

Turns Model 1's demand forecast (Student 2's models/demand_model.pkl) and Model 2's
stock-out predictions (models/stockout_model.pkl) into the exact manager-ready output
table specified by the brief:

    Store | Product | Current Stock | 7-Day Forecast | Stock-out Prob. | Risk | Recommended Order

Business logic (docs/RESEARCH.md §1, §4, §5):
    Safety Stock       = Z(service_level) * rolling_std_7 * sqrt(lead_days)
    Recommended Stock  = 7-day forecast + Safety Stock
    Reorder Quantity   = max(0, Recommended Stock - Current Stock - Incoming Stock)
    Risk Tier          = High >=0.70 / Medium 0.40-0.70 / Low <0.40 (fixed thresholds)

Known limitation (disclosed, not hidden): master_table.csv/feature_table.csv have no
purchase-order / "stock already in transit" table, so `incoming_stock` is 0 for every
row. Real event data may include this -- if so, read it in and replace the placeholder.
"""
import os

import joblib
import numpy as np
import pandas as pd

DATA_DIR = "data/processed"
MODELS_DIR = "models"

# ABC-XYZ segment -> target service level -> Z-score (RESEARCH.md §4)
SEGMENT_Z = {
    "AX": 2.33, "AY": 2.17, "AZ": 2.05,
    "BX": 1.75, "BY": 1.65, "BZ": 1.48,
    "CX": 1.28, "CY": 1.15, "CZ": 1.04,
}
DEFAULT_Z = 1.65

RISK_HIGH, RISK_MEDIUM = 0.70, 0.40


def risk_tier(prob: float) -> str:
    if prob >= RISK_HIGH:
        return "HIGH"
    if prob >= RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def explain_row(row: pd.Series, population: pd.DataFrame) -> str:
    """Rule-based, per-row directional-delta explanation matching the brief's exact
    example style ("Promotion active +31%", etc.)."""
    reasons = []

    if row.get("promotion_flag", 0) == 1:
        promo_mean = population.loc[population["promotion_flag"] == 1, "corrected_demand"].mean()
        base_mean = population.loc[population["promotion_flag"] == 0, "corrected_demand"].mean()
        if base_mean and base_mean > 0:
            lift = (promo_mean - base_mean) / base_mean
            reasons.append(f"Promotion active (+{lift:.0%} typical demand lift)")

    if row.get("weekend_flag", row.get("weekend", 0)) == 1:
        reasons.append("Weekend approaching")

    lag_1, lag_7 = row.get("lag_1"), row.get("lag_7")
    if pd.notna(lag_1) and pd.notna(lag_7) and lag_7:
        growth = (lag_1 - lag_7) / lag_7
        if growth > 0.10:
            reasons.append(f"Recent sales growth (+{growth:.0%} vs last week)")

    if row.get("festival_flag", row.get("festival", 0)) == 1:
        reasons.append("Local festival")
    if row.get("holiday", 0) == 1:
        reasons.append("Public holiday")

    temp = row.get("temp_c")
    if pd.notna(temp):
        cat_mean_temp = population.loc[population["category"] == row.get("category"), "temp_c"].mean()
        if pd.notna(cat_mean_temp) and temp - cat_mean_temp > 2:
            reasons.append("Temperature above seasonal average")

    if row.get("days_of_inventory", np.inf) < 3:
        reasons.append("Less than 3 days of inventory remaining")

    if not reasons:
        reasons.append("Demand pattern in line with recent history")

    return ", ".join(reasons[:4])


def main():
    feat = pd.read_csv(os.path.join(DATA_DIR, "feature_table.csv"))
    abc_xyz = pd.read_csv(os.path.join(DATA_DIR, "abc_xyz_analysis.csv"))

    stockout_bundle = joblib.load(os.path.join(MODELS_DIR, "stockout_model.pkl"))
    so_model, so_encoders, so_feature_cols = (
        stockout_bundle["model"], stockout_bundle["encoders"], stockout_bundle["feature_cols"]
    )

    demand_model_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    demand_model = joblib.load(demand_model_path) if os.path.exists(demand_model_path) else None

    df = feat.merge(abc_xyz[["product_id", "abc_class", "xyz_class", "cv_demand"]],
                     on="product_id", how="left")
    df["abc_class"] = df["abc_class"].fillna("C")
    df["xyz_class"] = df["xyz_class"].fillna("Z")
    df["cv_demand"] = df["cv_demand"].fillna(df["cv_demand"].median())

    cat_fallback = df.groupby("category")["corrected_demand"].transform("mean")
    for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"]:
        df[col] = df[col].fillna(cat_fallback)
    df["rolling_std_7"] = df["rolling_std_7"].fillna(df["rolling_std_7"].median())
    df["days_of_inventory"] = df["days_of_inventory"].replace([np.inf, -np.inf], np.nan)
    df["days_of_inventory"] = df["days_of_inventory"].fillna(df["days_of_inventory"].median())
    df["price_change"] = df["price_change"].fillna(0)

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].dropna(subset=["lag_1", "rolling_mean_7"]).copy()

    for col in ["store_type", "category", "brand", "abc_class", "xyz_class"]:
        enc = so_encoders[col]
        latest[col + "_enc"] = latest[col].astype(str).map(
            {cls: i for i, cls in enumerate(enc.classes_)}
        ).fillna(-1).astype(int)

    X_latest_so = latest[so_feature_cols]
    latest["stockout_probability"] = so_model.predict_proba(X_latest_so)[:, 1]
    latest["risk_tier"] = latest["stockout_probability"].apply(risk_tier)

    if demand_model is not None:
        dm_model = demand_model.get("model", demand_model) if isinstance(demand_model, dict) else demand_model
        dm_feature_cols = demand_model.get("feature_cols") if isinstance(demand_model, dict) else None
        try:
            if dm_feature_cols:
                latest["forecast_7day"] = dm_model.predict(latest[dm_feature_cols]).round().clip(min=0)
            else:
                latest["forecast_7day"] = (latest["rolling_mean_7"] * 7).round().clip(lower=0)
        except Exception as e:
            print(f"Warning: demand_model.pkl prediction failed ({e}); falling back to rolling-mean baseline.")
            latest["forecast_7day"] = (latest["rolling_mean_7"] * 7).round().clip(lower=0)
    else:
        latest["forecast_7day"] = (latest["rolling_mean_7"] * 7).round().clip(lower=0)

    latest["z_score"] = latest["abc_class"].astype(str).str.cat(latest["xyz_class"].astype(str)).map(SEGMENT_Z).fillna(DEFAULT_Z)
    latest["safety_stock"] = (latest["z_score"] * latest["rolling_std_7"].fillna(0) * np.sqrt(latest["lead_days"])).round()
    latest["recommended_stock"] = latest["forecast_7day"] + latest["safety_stock"]
    latest["current_stock"] = latest["closing_stock"]
    latest["incoming_stock"] = 0  # documented limitation -- no purchase-order table in this dataset
    latest["reorder_qty"] = (latest["recommended_stock"] - latest["current_stock"] - latest["incoming_stock"]).clip(lower=0).round()

    population = feat[pd.to_datetime(feat["date"]) >= (latest_date - pd.Timedelta(days=60))]
    latest["reason"] = latest.apply(lambda r: explain_row(r, population), axis=1)

    def manager_action(risk):
        return {
            "HIGH": "Raise replenishment order today.",
            "MEDIUM": "Monitor closely; place order within 2-3 days if trend continues.",
            "LOW": "No action needed.",
        }[risk]

    latest["manager_action"] = latest["risk_tier"].apply(manager_action)
    latest["abc_xyz"] = latest["abc_class"].astype(str) + latest["xyz_class"].astype(str)

    out = latest[[
        "store_id", "product_id", "category", "brand", "city", "current_stock", "forecast_7day",
        "stockout_probability", "risk_tier", "reorder_qty", "reason", "manager_action", "abc_xyz",
    ]].rename(columns={
        "store_id": "Store", "product_id": "Product", "city": "City", "current_stock": "Current Stock",
        "forecast_7day": "7-Day Forecast", "stockout_probability": "Stock-out Prob.",
        "risk_tier": "Risk", "reorder_qty": "Recommended Order",
    })

    out = out.sort_values("Stock-out Prob.", ascending=False)
    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(os.path.join(DATA_DIR, "recommendations.csv"), index=False)

    print(f"Wrote {len(out)} rows to data/processed/recommendations.csv (as of {latest_date.date()})")
    print(f"Used {'real Model 1 forecast' if demand_model is not None else 'rolling-mean baseline'} for 7-Day Forecast.")
    print(out["Risk"].value_counts())
    print("\nTop 5 highest-risk rows:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
