# Model Card — Demand Forecasting (`demand_model.pkl`)

_Generated: 2026-09-29 12:15_

## Overview

- **Task:** Predict `next_7_day_demand` (total units demanded over the next 7 days)
- **Algorithm:** Random Forest
- **Hyperparameters:** n_estimators=200, max_depth=15, min_samples_leaf=10
- **Training data:** Feature table up to 14-day holdout cutoff
- **Validation:** TimeSeriesSplit (k=5) + final 14-day holdout

## Overall Performance (Holdout)

| Metric | Baseline (7-day mean) | This Model |
|---|---|---|
| MAE | 12.9638 | 11.5688 |
| RMSE | 17.519 | 15.7721 |
| MAPE | 1.3069 | 1.2105 |
| R² | 0.0253 | 0.21 |

## Features Used

Total: 38 features

- `day_of_week`
- `weekend_flag`
- `month`
- `week_no`
- `lag_1`
- `lag_7`
- `lag_14`
- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_std_7`
- `days_of_inventory`
- `inventory_to_demand_ratio`
- `reorder_gap`
- `discount_pct`
- `price_change`
- `promotion_flag`
- `festival_flag`
- `holiday`
- `temp_c`
- `rain_mm`
- `local_event`
- `shelf_life_days`
- `lead_days`
- `floor_area_sqft`
- `avg_daily_customers`
- `store_type_Hypermarket`
- `store_type_Supermarket`
- `category_Dairy`
- `category_Frozen`
- `category_Household`
- `category_Personal Care`
- `category_Snacks`
- `brand_Crispo`
- `brand_FizzUp`
- `brand_GlowCare`
- `brand_HomeShine`
- `brand_PureLeaf`
- `brand_SnapFresh`

## Known Limitations

1. **Sparse-history products** (< 14 days of data) rely on category-level
   mean demand as a fallback for lag/rolling features — predictions for these
   SKUs carry wider error margins.
2. **Censored demand correction** is applied upstream (Student 1's `corrected_demand`);
   the quality of this model's forecasts is directly dependent on the quality of
   that correction. If stock-outs are systematic and prolonged, the correction
   heuristic may still under-estimate true demand.
3. **External factors** (weather, festivals) are used as lagged/historical features.
   In production, future weather would require a separate weather forecast API.
4. **No deep learning** — constrained to permitted algorithms per competition rules.
5. **Static training** — this model does not retrain automatically as new data arrives.
   In production, a scheduled retraining pipeline would be needed.

## Reproducibility

```bash
python src/demand_model.py --features data/processed/feature_table.csv
```
