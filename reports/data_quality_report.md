# Data Quality & Anomaly Reconciliation Report
**Author:** Student 1 — Data Analyst (StockSense Team)  
**Date of Execution:** 2026-09-29 10:40:29  
**Dataset Version:** NovaMart Retail System (Master Grain: `date × store_id × product_id`)  
**Processed Master Table:** [`data/processed/master_table.csv`](../data/processed/master_table.csv) (39,001 rows, 37 columns)

---

## 1. Executive Summary

A robust decision-support system requires high-fidelity, leakage-free data. Real-world retail data is plagued with operational noise, logging dropouts, and point-of-sale anomalies. 

In Round 1, we audited all 5 incoming raw datasets against the shared schema ([`DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md)), systematically isolated **6 intentional data-quality traps**, implemented defensible cleaning rules, and created explicit tracking flags. **No critical observations were silently dropped without accounting.**

---

## 2. Intentional Data-Quality Traps Accounting Table

| # | Trap Description | Raw Detection Method | Trap Count Found | Cleaning Action Taken | Business & Technical Justification |
|---|---|---|---|---|---|
| **1** | **Duplicate Transactions** | `df.duplicated(subset=['transaction_id'])` | **657** / 66,759 rows (0.98%) | Dropped identical duplicates; preserved first record. | Network/POS re-transmissions artificially inflate customer count and sales volume. Keeping duplicates would cause demand forecasting models to over-predict. |
| **2** | **Impossible (Negative) Quantities** | `tx['quantity'] < 0` | **341** rows | Flagged and excluded from daily demand aggregation. | No return authorization or return slip metadata was present. Treating negative sales as true customer demand corrupts sales velocities and causes negative target values. |
| **3** | **Category Casing & Spacing Variants** | `prod['category'].unique()` | **21** raw variants for **6** true categories | Applied regex alphanumeric cleaning (`re.sub(r'[^a-z]', '', s)`) mapped to Title Case. | String variations such as `'PersonalCare'` vs `' Personal Care'` or `'BEVERAGES'` vs `'beverages'` cause categorical merges and GroupBy aggregations to fragment into false silos. |
| **4** | **Missing Temperature Readings** | `ext['temp_c'].isna()` | **23** / 720 days (3.19%) | Imputed using city-specific linear interpolation & calendar normals; flagged with `is_imputed_temp=1`. | Missing temperature data represented only 3.19% of readings. Forward-filling or city climate normals preserve macro seasonality without introducing synthetic distortion. |
| **5** | **Inventory Arithmetic Mismatches** | `closing != opening + received - sold` | **1,100** rows (2.82%) | Retained reported stock; added audit flag `is_inventory_mismatch=1`. | Physical stock shrinkage, unrecorded damage, or barcode scanning timing create legitimate warehouse discrepancies. Rather than arbitrarily forcing balance, downstream models and managers are informed via the flag. |
| **6** | **Sparse-History Products** | Active sales history < 14 days | **6** SKUs (P102, P126, P129, P136, P148, P155) | Retained with `is_sparse_history=1` flag; fallback to category-level baselines advised for ML. | New product introductions lack sufficient lookback for 7/14-day rolling features. Flagging ensures Student 2 avoids NaN feature leakage and applies cold-start imputation. |

---

## 3. Stock-Out Censorship & Corrected Demand Signal

### The Core Problem
When a store runs out of stock (`closing_stock == 0`), recorded sales drop to zero or truncate mid-day. Standard regression algorithms trained on raw `quantity_sold` learn that demand drops to zero whenever inventory is depleted. This leads to **systematic under-forecasting of high-velocity, frequently stocked-out items**.

### Implementation
- **Total Stock-out Days (`stockout_flag == 1`):** **6,189** (15.87%)
- **Total Censored Days (`is_censored == 1`):** **6,238** (15.99%)
- **Correction Method:** For censored observations, true latent demand is reconstructed using the rolling mean demand on the same day-of-week across prior uncensored dates (past 4 weeks):
  $$\text{corrected\_demand} = \max(\text{quantity\_sold}, \text{rolling\_weekday\_mean})$$
- **Leakage Prevention:** Rolling statistics strictly utilize historical records prior to the observation date ($t-1$ and earlier). No future window data is accessed.

---

## 4. Master Table Quality Assurance Checks

- [x] **Primary Key Uniqueness:** Verified unique at `date × store_id × product_id` (0 duplicates across 39,001 rows).
- [x] **Completeness:** 0 null values across all 37 production columns.
- [x] **Value Constraints:**
  - `revenue >= 0.0`
  - `quantity_sold >= 0.0`
  - `corrected_demand >= quantity_sold`
  - `closing_stock >= 0`
  - `avg_discount_pct` in range $[0, 100]$
- [x] **Contract Compliance:** Schema precisely matches [`DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md), unblocking Student 2 (Feature Engineering) and Student 3 (Decision Intelligence).

---

## 5. Downstream Recommendations for Team

1. **For Student 2 (ML Engineer):**
   - Use `corrected_demand` as the primary base for target generation (`next_7_day_demand`) and lag features (`lag_1, lag_7, lag_14`).
   - For rows where `is_sparse_history == 1`, bypass long-window rolling lags and impute using category/store-type median demand.
2. **For Student 3 (Decision Intelligence):**
   - `stockout_flag` exhibits a **15.87%** base rate. Use class-weighted trees (`scale_pos_weight` in XGBoost, `class_weight='balanced'` in Random Forest) rather than raw SMOTE.
   - Use the gap between `corrected_demand` and `quantity_sold` multiplied by `avg_selling_price` to display **Estimated Lost Revenue** on the Streamlit Executive Dashboard.
