# Data Dictionary — Shared Contract Between All Three Roles

This is the schema every role reads from and writes to. If you need a column that isn't here, **add it to this file in the same commit** that adds it to the data, so the other two teammates always know what exists without having to ask.

Grain of every table below (unless stated otherwise): **one row = one `date` × one `store_id` × one `product_id`.**

## Raw Inputs (from `data/raw/`, untouched)

| File | Grain | Key columns |
|---|---|---|
| `transactions.csv` | one row per sale | `transaction_id, date, store_id, product_id, quantity, selling_price, discount_pct, promotion_flag, customer_id, payment_mode, hour` |
| `products.csv` | one row per product | `product_id, category, sub_category, brand, mrp, cost_price, shelf_life_days, supplier_id` |
| `stores.csv` | one row per store | `store_id, city, store_type, floor_area_sqft, avg_daily_customers, region` |
| `inventory.csv` | one row per date×store×product | `date, store, product, opening, received, sold, closing, reorder_lvl, lead_days` |
| `external_factors.csv` | one row per date×city | `date, city, temp_c, rain_mm, holiday, festival, weekend, local_event` |

## `data/processed/master_table.csv` — owned by Student 1

Produced by aggregating `transactions.csv` to daily store×product level, then left-joining `products`, `stores`, `inventory`, and `external_factors` (via each store's `city`).

| Column | Type | Notes |
|---|---|---|
| `date` | date | ISO `YYYY-MM-DD` |
| `store_id` | string | |
| `product_id` | string | |
| `city`, `store_type`, `region`, `floor_area_sqft`, `avg_daily_customers` | — | from `stores.csv` |
| `category`, `sub_category`, `brand`, `mrp`, `cost_price`, `shelf_life_days`, `supplier_id` | — | from `products.csv`, **category standardized** (case/spacing fixed) |
| `quantity_sold` | float | aggregated daily units, from `transactions.csv` (or `inventory.sold` reconciled with it — document any mismatch in the DQ report) |
| `revenue` | float | `quantity_sold × selling_price` (avg daily selling price if multiple transactions) |
| `avg_selling_price`, `avg_discount_pct`, `promotion_flag` | — | daily aggregates from `transactions.csv` |
| `opening_stock`, `received_stock`, `closing_stock`, `reorder_lvl`, `lead_days` | — | from `inventory.csv` |
| `is_censored` | 0/1 | **new** — 1 if this date's sales were constrained by a stock-out (see RESEARCH.md §3) |
| `corrected_demand` | float | **new** — censored-demand-corrected daily demand signal (raw `quantity_sold` when `is_censored==0`) |
| `temp_c`, `rain_mm`, `holiday`, `festival`, `weekend`, `local_event` | — | from `external_factors.csv`, joined on `date` + `city` |
| `stockout_flag` | 0/1 | **operational definition — Student 3 owns the exact rule, documented in their README** |

Data quality flags to retain as columns (don't silently drop rows — flag, then decide): `is_duplicate_removed`, `is_qty_invalid`, `is_inventory_mismatch`, `is_imputed_temp` (or similar) as needed — full accounting goes in `reports/data_quality_report.md`.

## `data/processed/feature_table.csv` — owned by Student 2

Everything in `master_table.csv` **plus**:

| Feature group | Columns |
|---|---|
| Time | `day_of_week, weekend_flag, month, week_no, festival_flag` |
| Lag (on `corrected_demand`) | `lag_1, lag_7, lag_14` |
| Rolling (on `corrected_demand`) | `rolling_mean_7, rolling_mean_14, rolling_std_7` |
| Inventory | `days_of_inventory, inventory_to_demand_ratio, reorder_gap` |
| Price / Promo | `discount_pct, price_change, promotion_flag` |
| Store / Product | `store_type, category, brand, shelf_life_days, lead_time` |
| Segmentation (Student 3 also reads these) | `abc_class` (A/B/C), `xyz_class` (X/Y/Z), `cv_demand` |
| Target | `next_7_day_demand` (sum of `corrected_demand` over the next 7 days — **future window, training-only column, never used as a predictor**) |

**Leakage rule (non-negotiable):** any column used as an `X` feature must be computable using only information available strictly before the 7-day window it predicts. See RESEARCH.md §2.3 for the exact checklist.

## `data/processed/stockout_labels.csv` — owned by Student 3

| Column | Type | Notes |
|---|---|---|
| `date, store_id, product_id` | — | join key back to `feature_table.csv` |
| `stockout_flag` | 0/1 | operational definition, e.g. `closing_stock == 0` OR `corrected_demand > (opening_stock + received_stock)` within the next 7 days — pick one, document it clearly, it must match what `master_table.stockout_flag` says |
| `stockout_probability` | float [0,1] | model output |
| `risk_tier` | High/Medium/Low | per the brief's fixed thresholds (0.70 / 0.40) |

## Model Artifacts

| File | Owner | Contract |
|---|---|---|
| `models/demand_model.pkl` | Student 2 | `.predict(X: feature_table columns)` → `next_7_day_demand` (float, units) |
| `models/stockout_model.pkl` | Student 3 | `.predict_proba(X: feature_table columns)` → `stockout_probability` |
| `models/demand_model_card.md`, `models/stockout_model_card.md` | respective owner | algorithm chosen, hyperparameters, metrics, known limitations |

## Final Output Table (what the dashboard renders — matches the brief's example exactly)

| Store | Product | Current Stock | 7-Day Forecast | Stock-out Prob. | Risk | Recommended Order |
|---|---|---|---|---|---|---|

Produced by `src/recommendation_engine.py` (Student 3), consuming both models + `master_table.csv` (for current/incoming stock) + `products.csv` (for cost/margin, used by the newsvendor nudge).
