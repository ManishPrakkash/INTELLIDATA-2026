# Model Comparison Report — Demand Forecasting (Model 1)

_Generated: 2026-09-29 12:15_

## Why MAE Is Our Headline Metric

We recommend **MAE (Mean Absolute Error) in units** as the primary metric because:
- It's directly interpretable: "on average, our forecast is X units off."
- Store managers can understand it without statistical training.
- MAPE misbehaves near zero-demand slow-movers (division by near-zero),
  which are common in our dataset (sparse-history products).
- R² is useful for model comparison but doesn't translate to actionable business language.

---

## Cross-Validation Results (TimeSeriesSplit, k=5)

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | 9.3805 | 11.9534 | 0.4993 | -0.0219 |
| Decision Tree | 8.6881 | 11.3167 | 0.4268 | 0.0729 |
| Random Forest | 8.1518 | 10.6288 | 0.414 | 0.1916 |
| XGBoost | 8.2785 | 10.7562 | 0.4254 | 0.1668 |

---

## Holdout Test Results (last 14 days, unseen during training/CV)

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Baseline (7-day mean)** | 12.9638 | 17.519 | 1.3069 | 0.0253 |
| Linear Regression | 12.6029 | 15.683 | 1.4197 | 0.2189 |
| Decision Tree | 12.3023 | 16.6302 | 1.2829 | 0.1217 |
| Random Forest ✅ **Winner** | 11.5688 | 15.7721 | 1.2105 | 0.21 |
| XGBoost | 11.5828 | 15.7381 | 1.2008 | 0.2134 |

---

## Selected Model: **Random Forest**

### Justification

**Random Forest** was selected as the production model because it achieved the 
lowest MAE on the unseen holdout set while maintaining strong R². 
It provides the best balance of accuracy, interpretability, and training efficiency 
among the permitted algorithms. Tree-based models naturally handle non-linear 
interactions between features (e.g., promotions × seasonality) without manual 
feature crosses.

### Features Used

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

---

## Stock-Out Classification (Model 2)

_Section to be completed by Student 3._

## Model 2 -- Stock-out Risk Classification (Student 3)

        model  cv_f1_mean  test_accuracy  test_precision  test_recall  test_f1  test_roc_auc
Random Forest       1.000          1.000           1.000        1.000    1.000         1.000
Decision Tree       0.990          0.999           1.000        0.997    0.999         0.999
      XGBoost       0.984          0.997           0.989        1.000    0.994         1.000

**Selected model:** Random Forest. Justification: highest F1 on the untouched final holdout
(last 14 days), balancing precision (don't cry wolf on every SKU) against recall (don't
miss real stock-outs) -- both a missed stock-out and a false alarm carry a real business
cost, so F1 is fairer to optimize than Accuracy or Recall alone. Class imbalance (~16%
positive) handled via class_weight='balanced'
rather than SMOTE, per docs/RESEARCH.md §6.
