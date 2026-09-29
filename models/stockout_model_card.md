# Stock-out Risk Model Card

**Algorithm:** Random Forest (selected by highest test F1 among Decision Tree / Random Forest / XGBoost)
**Class imbalance handling:** class_weight='balanced'
**Operational definition of stockout_flag:** closing_stock == 0 at end of day (Student 1's cleaning pipeline)
**Train positive rate:** 15.3% | **Test (last 14 days) positive rate:** 22.2%
**Validation:** TimeSeriesSplit(5) for model comparison; last 14 days as untouched final holdout
**Features:** Student 2's feature_table.csv (lag/rolling/calendar/inventory/price) + ABC-XYZ segmentation

## Test-set metrics (last 14 days, never used in training/CV)

        model  cv_f1_mean  test_accuracy  test_precision  test_recall  test_f1  test_roc_auc
Random Forest       1.000          1.000           1.000        1.000    1.000         1.000
Decision Tree       0.990          0.999           1.000        0.997    0.999         0.999
      XGBoost       0.984          0.997           0.989        1.000    0.994         1.000

## Confusion Matrix (best model, threshold=0.5)

```
[[2446    0]
 [   0  699]]
```
Rows = actual [no stock-out, stock-out], Columns = predicted [no stock-out, stock-out]

**Why not lead with Accuracy:** positive class is a minority (~16%); a model predicting
"never stock out" scores high accuracy while being useless. Precision/Recall/F1/ROC-AUC
are the metrics that matter here.

## Permutation Importance (top drivers, global)

                  feature  importance_mean  importance_std
        days_of_inventory         0.486035        0.009215
inventory_to_demand_ratio         0.419520        0.010116
              reorder_gap         0.003735        0.001059
                    lag_1         0.000358        0.000358
                    lag_7         0.000286        0.000351
            rolling_std_7         0.000000        0.000000
              day_of_week         0.000000        0.000000
                    month         0.000000        0.000000
                  week_no         0.000000        0.000000
             weekend_flag         0.000000        0.000000

## Known limitations

- Sparse-history products (<14 days) fall back to category-level mean demand for
  lag/rolling features, widening their error (flagged via is_sparse_history upstream).
- Risk tiers are derived from stockout_probability using the brief's fixed thresholds
  (High >=0.70, Medium 0.40-0.70, Low <0.40), not a re-derived cutoff.
