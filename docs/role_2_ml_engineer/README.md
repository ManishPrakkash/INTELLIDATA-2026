# Student 2 — ML Engineer
## Feature Engineering, Forecasting, Validation

You turn Student 1's clean master table into a leakage-safe feature set and a trustworthy 7-day demand forecaster. Model 1 (`next_7_day_demand`, regression) is entirely yours; you also hand Student 3 a stable, well-documented feature table so their stock-out classifier isn't built on shifting sand.

Read first: [root plan](../../README.md) §2–3, and [`docs/RESEARCH.md`](../RESEARCH.md) §1 (safety stock — you don't compute it, but you must understand it), §2 (time-aware CV, leakage), §3 (censored demand — you consume Student 1's `is_censored`/`corrected_demand` columns).

Your outputs, exactly:
- `src/features.py` (reusable, unit-testable feature functions)
- `data/processed/feature_table.csv`
- `notebooks/02_feature_engineering.ipynb`, `notebooks/03_demand_forecasting.ipynb`
- `models/demand_model.pkl` + `models/demand_model_card.md`
- `reports/model_comparison.md` (shared file — Student 3 appends their classification section)

---

## Hour-by-Hour Plan

### Hour 0–1 — Setup & Pre-Build
Same environment setup as everyone. `data/raw/` already has a full practice dataset in place (root [`README.md`](../../README.md) §8) — while Student 1 cleans it into `master_table.csv`, **build and unit-test your feature functions against that data right away**; you don't need to wait or generate anything yourself. `src/features.py` should be largely done and merely need re-pointing at the final `master_table.csv` at hour 5. This is the single highest-leverage use of your first hour. (If the official dataset replaces the practice one mid-event, your feature functions are schema-driven and shouldn't need changes — just a re-run.)

### Hour 1–5 — Build Feature Functions (against synthetic data)

Implement each group as a small, named function in `src/features.py` operating on a DataFrame sorted by `(store_id, product_id, date)`:

| Group | Features | Implementation note |
|---|---|---|
| Time | `day_of_week, weekend_flag, month, week_no, festival_flag` | derived directly from `date` and the joined `external_factors` columns — trivially safe, no leakage risk |
| Lag | `lag_1, lag_7, lag_14` | `groupby(['store_id','product_id'])['corrected_demand'].shift(k)` — **use `corrected_demand`, not raw `quantity_sold`**, per RESEARCH.md §3 |
| Rolling | `rolling_mean_7, rolling_mean_14, rolling_std_7` | `groupby(...).shift(1).rolling(w).mean()/.std()` — **the `.shift(1)` before `.rolling()` is the single most common leakage bug**: without it, today's own value leaks into today's rolling average |
| Inventory | `days_of_inventory, inventory_to_demand_ratio, reorder_gap` | `days_of_inventory = closing_stock / rolling_mean_7` (guard divide-by-zero → cap or NaN→impute); `reorder_gap = reorder_lvl - closing_stock` |
| Price/Promo | `discount_pct, price_change, promotion_flag` | `price_change = avg_selling_price - avg_selling_price.shift(1)` |
| Store/Product | `store_type, category, brand, shelf_life_days, lead_time(=lead_days)` | static joins, one-hot or target/frequency-encode as needed for the chosen algorithm |
| Segmentation | `abc_class, xyz_class, cv_demand` | compute once globally per product (ABC on total revenue, XYZ on demand CV) — reuse Student 1's EDA numbers directly, don't recompute from scratch |

**Target:** `next_7_day_demand = ` forward sum of `corrected_demand` over the next 7 days, computed with `shift(-7).rolling(7, min_periods=7).sum()`-style logic **applied only for building the training target**, never included as a feature.

Handle the **sparse-history trap** explicitly: for rows where a product has <14 days of history, lag/rolling features will be NaN — fall back to the category-level mean demand (documented, not silently zero-filled) per Student 1's `is_sparse_history` flag.

### Hour 5 — CHECKPOINT 1
Pull the real `data/processed/master_table.csv`, re-run `src/features.py` against it, fix any schema mismatches with Student 1 immediately, push `data/processed/feature_table.csv`.

### Hour 6–13 — Model 1: Demand Forecasting (Round 2 core)

1. **Baseline first, always** — the brief requires it, and it's your sanity floor: predict `next_7_day_demand` as the previous 7-day mean (`rolling_mean_7 × 7`). Compute its MAE/RMSE/MAPE/R² — every real model must beat this or something is wrong.
2. **Time-aware split.** Sort by `date`. Use `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` for model comparison, then a final untouched holdout of the **last 7–14 days** for the number you report as "true" performance. Never shuffle before splitting.
3. **Train and compare, from the permitted list only:** Linear Regression (interpretable baseline-plus), Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor. (SVM/KNN/Naive Bayes are classification-oriented for this brief's purposes — skip for regression unless you want an extra comparison row; not required.)
4. **Metrics:** MAE, RMSE, MAPE, R² for every model, in `reports/model_comparison.md`. Explicitly write **which metric matters most to the business and why** — recommend **MAE in units** as the headline (it's directly interpretable as "average units off," which store managers understand; MAPE misbehaves near zero-demand slow-movers, which are common here).
5. **Pick a winner**, justify it in 2–3 sentences (accuracy vs. interpretability vs. training time trade-off), save it: `joblib.dump(model, 'models/demand_model.pkl')`.
6. Write `models/demand_model_card.md`: algorithm, hyperparameters, feature list, metrics (overall + broken down by `abc_class`/`store_type` — an honest per-segment breakdown is a differentiator per RESEARCH.md §9), and known limitations (e.g., "sparse-history products rely on category fallback and carry wider error").

### Hour 13 — CHECKPOINT 2
Freeze `feature_table.csv` and `demand_model.pkl`. Hand both to Student 3 with a 5-minute walkthrough of column meanings.

### Hour 13–20 — Support Role (Round 3)

- Build `src/run_pipeline.py`: one script that runs cleaning → features → both models → recommendation engine end-to-end from raw CSVs, so the whole repo is reproducible with one command (this is an explicit mandatory deliverable: "reproducible pipeline").
- Help Student 3 wire the trained demand model into the Streamlit dashboard and the What-If simulator (re-predicting on a modified feature row when a slider changes — no retraining needed, just re-run `.predict()` on an edited copy of the feature row).
- Support residual/error analysis for the "Model Performance" dashboard tab: plot actual vs. predicted, and a residual histogram.

### Hour 20–24 — Integration, Pitch, Polish
Join the bug bash. In the pitch, you own the "what we can predict + model performance + why it can be trusted" section — lead with the baseline-vs-model comparison and the per-segment error breakdown as evidence of honesty, not just accuracy.

---

## Deliverables Checklist

- [ ] `src/features.py` — all feature groups implemented, leakage-checked
- [ ] `data/processed/feature_table.csv`
- [ ] Baseline (7-day mean) metrics reported alongside every trained model
- [ ] Time-aware CV (`TimeSeriesSplit`) — no random shuffling anywhere near dates
- [ ] `reports/model_comparison.md` — MAE/RMSE/MAPE/R² per algorithm + business-metric justification
- [ ] `models/demand_model.pkl` + `models/demand_model_card.md`
- [ ] `src/run_pipeline.py` — one-command reproducibility

## Common Pitfalls to Avoid

- Forgetting `.shift(1)` before `.rolling()` — this single bug silently leaks the target into the training features and will produce suspiciously great validation scores that collapse on the real holdout.
- Using `train_test_split(..., shuffle=True)` anywhere near time-ordered data.
- Training on raw `quantity_sold` instead of `corrected_demand` — you'll systematically under-forecast exactly the SKUs the business cares most about (the frequent stock-outs).
- Reporting only one global metric — segment-level breakdown is what makes the model trustworthy to judges and to a store manager.
- Using Deep Learning (LSTM/Prophet-with-neural-components/etc.) — **disqualifying** per the brief. Stick to the seven permitted algorithms.
