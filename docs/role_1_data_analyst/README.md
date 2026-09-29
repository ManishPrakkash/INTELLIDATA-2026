# Student 1 — Data Analyst
## Cleaning, Integration, EDA, Statistics, Data Quality

You own the foundation everyone else builds on. If `master_table.csv` is late, wrong, or ambiguous, both Model 1 (Student 2) and Model 2 + dashboard (Student 3) stall. Speed and clear documentation matter as much as correctness — **hand off early, hand off often.**

Read first: [root plan](../../README.md) §2 (architecture), §6 (timeline), §7 (data contract) and [`docs/RESEARCH.md`](../RESEARCH.md) §2.3 (leakage), §3 (censored demand — you compute the flag, Student 2 uses it).

Your outputs, exactly:
- `data/processed/master_table.csv` — the single source of truth (schema in [`DATA_DICTIONARY.md`](../DATA_DICTIONARY.md))
- `reports/data_quality_report.md`
- `notebooks/01_eda.ipynb`
- `reports/statistical_tests.md`

---

## Hour-by-Hour Plan

### Hour 0–1 — Setup
1. `git pull`, create your venv, `pip install -r requirements.txt`.
2. `data/raw/` already contains a full 180-day, 4-store, 60-product practice dataset (see root [`README.md`](../../README.md) §8 for exact row counts and injected trap counts) — no need to generate anything to get started. If the official NovaMart CSVs are handed out at the event, drop them into `data/raw/` (same filenames) and re-run everything from here; nothing downstream is hardcoded to the practice values.
3. Open every raw CSV and just look: `df.info()`, `df.describe()`, `df.head(20)`, `df['category'].unique()`. Do not assume the schema — verify column names/types match [`DATA_DICTIONARY.md`](../DATA_DICTIONARY.md); update that file if reality differs. Note: the practice data's `category` column has 20 distinct raw-string variants across only 6 true categories — a good first thing to standardize.

### Hour 1–5 — Cleaning, Integration, Aggregation (Round 1 core)

Work through the brief's **Intentional Data Quality Traps** table systematically. For every issue you find: **count it, decide an action, write one sentence of justification** — that sentence goes straight into `data_quality_report.md`. Never silently drop rows without logging the count and reason.

| Trap | How to detect | Action to take |
|---|---|---|
| Missing values (`temp_c` blank) | `df.isna().sum()` | Impute with same-city rolling mean, or forward-fill; **do not** impute more than ~10% of a column without flagging it as a limitation |
| Duplicate rows (same `transaction_id` repeated) | `df.duplicated(subset='transaction_id')` | Drop true duplicates (identical rows); keep the first occurrence; log count |
| Category inconsistency (`Beverages`/`beverage`/`BEVERAGES`) | `df['category'].str.lower().str.strip().unique()` | Standardize to a canonical Title Case set; build a mapping dict, don't hand-fix row by row |
| Impossible quantity (`quantity = -4`) | `df[df['quantity'] < 0]` | These are not "returns" unless a `return_flag` exists — treat as data-entry errors: flag and exclude from demand aggregation, but **count** them (mention as a limitation, don't just vanish them) |
| Inventory mismatch (`closing != opening + received - sold`) | compute `expected_closing` and diff | Where diff is small (±rounding) reconcile silently; where large, flag `is_inventory_mismatch=1` and keep both values — let downstream decide, don't guess |
| Sparse history (new product, <7 days of data) | `groupby(product_id).size()` | Keep the rows, but flag `is_sparse_history=1`; note in the DQ report that Student 2 should use a category-level fallback (e.g., category median demand) for these, not a lag-based feature |

**Then, in order:**
1. Standardize `date` to a proper datetime, `store_id`/`product_id` to consistent string dtype (watch for whitespace).
2. Aggregate `transactions.csv` to **daily store × product**: `quantity_sold = sum(quantity)`, `revenue = sum(quantity*selling_price)`, `avg_selling_price = mean(selling_price)`, `avg_discount_pct = mean(discount_pct)`, `promotion_flag = max(promotion_flag)` (promo active if any transaction that day was promotional).
3. Merge in `products.csv` (on `product_id`), `stores.csv` (on `store_id`), `inventory.csv` (on `date`+`store`+`product`), `external_factors.csv` (on `date`+`city`, where `city` comes from the store join — do this join *after* the store merge).
4. Compute `is_censored` and a first-pass `corrected_demand` per [`RESEARCH.md`](../RESEARCH.md) §3 — even a simple version now unblocks Student 2 completely; refine later if time allows.
5. Write `data/processed/master_table.csv`. **Push it as soon as it exists**, even before EDA is finished — Student 2 needs it to start feature engineering hours earlier than if you wait for "perfect."

### Hour 5 — CHECKPOINT 1
Announce the master table is live. Walk Student 2 and Student 3 through any column you're unsure about. Fix schema issues together immediately rather than each guessing independently.

### Hour 5–6 — EDA (Round 1)

Every chart must answer a stated business question — don't make decorative charts. Produce all six, saved as figures + one-paragraph interpretation each in the notebook:

| Business Question | Chart |
|---|---|
| Which categories generate most revenue? | Category revenue share (bar) + Pareto (cumulative % line) — **this doubles as your ABC input for Student 3** |
| Which stores are growing or declining? | Daily/weekly revenue trend, one line per store |
| Do promotions increase units sold? | Boxplot or bar: mean `quantity_sold` promo vs non-promo |
| How does weekend behaviour differ? | Weekday vs weekend mean demand, by category |
| Which products are volatile? | Coefficient of variation (`std/mean`) of daily demand per product, ranked — **this doubles as your XYZ input for Student 3** |
| Which stores repeatedly stock out? | Heatmap: store × category, color = stock-out rate |

### Hour 5–6 (parallel with above) — Statistical Reasoning

Perform **at least 3** tests. For each, write in `reports/statistical_tests.md`: business question, H0, H1, test used, statistic, p-value, and a one-sentence business interpretation (not just "p<0.05, reject H0" — say what that *means* for a store manager).

1. **Do promotions significantly increase sales?** Two-sample t-test (or Mann-Whitney U if demand isn't normal — check with a quick histogram/Shapiro test first) comparing `quantity_sold` on promo=1 vs promo=0 days.
2. **Does mean demand differ across store types?** One-way ANOVA (or Kruskal-Wallis) across `store_type` groups.
3. **Is stock-out frequency associated with promotion status?** Chi-square test of independence on a 2×2 contingency table (`promotion_flag` × `stockout_flag`).

Use `scipy.stats` (`ttest_ind`, `mannwhitneyu`, `f_oneway`, `kruskal`, `chi2_contingency`). Always check the assumption (normality/independence) before picking the parametric vs non-parametric version — stating that you checked is itself a scoring point under "Statistical Reasoning."

### Hour 6–13 — Support Role (Round 2, while Students 2 & 3 build models)

You are not idle — you are the fact-checker:
- Validate that Student 2's engineered features have sane distributions (no infinities from divide-by-zero in ratios like `inventory_to_demand_ratio`).
- Sanity-check that lag/rolling features don't leak (spot-check a few rows manually).
- Start drafting the **Executive Summary narrative** for the dashboard — plain-English bullets from your EDA (e.g., "Beverages drive 34% of revenue but have the highest demand volatility").
- Finalize `reports/data_quality_report.md` fully (issue / count / action / justification table, one row per trap).

### Hour 13–20 — Support Role (Round 3)

- Validate the KPI formulas on the dashboard (Revenue, Units Sold, Stock-out Rate, Inventory Turnover, Days of Inventory, Promotion Lift, Estimated Lost Sales) against your own EDA numbers — you are the person who will notice if a KPI tile is silently wrong.
- Help write the plain-English "Why?" reason strings for the recommendation cards, since you understand the business patterns best.

### Hour 20–24 — Integration, Pitch, Polish
Join the full-team bug bash, co-write the pitch (you present the "what we discovered" and "why the result can be trusted" data-quality narrative — this is your strongest section), help proofread the README links.

---

## Deliverables Checklist

- [ ] `data/processed/master_table.csv` (correct grain: date × store × product, pushed by hour 5)
- [ ] `reports/data_quality_report.md` (every trap: issue, count, action, justification)
- [ ] `notebooks/01_eda.ipynb` (6 business-question charts, each with a written interpretation)
- [ ] `reports/statistical_tests.md` (≥3 tests: H0, H1, test, p-value, business interpretation)
- [ ] ABC revenue ranking and demand-CV ranking available for Student 3 (can live in the EDA notebook, just make sure the numbers are exported to a CSV or clearly copy-pasteable)

## Common Pitfalls to Avoid

- Dropping rows with missing/invalid data **without counting them first** — always log before you filter.
- Fixing category casing "by eye" row by row instead of a documented mapping — not reproducible, looks sloppy to judges.
- Aggregating transactions to daily level using `mean()` instead of `sum()` for quantity — a very common, very wrong mistake.
- Joining `external_factors.csv` on `store_id` instead of `city` — the brief's schema joins external factors by city, not store.
- Waiting until EDA and stats are "perfect" before sharing `master_table.csv` — push early, iterate later.
