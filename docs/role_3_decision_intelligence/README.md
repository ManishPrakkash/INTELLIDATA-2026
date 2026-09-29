# Student 3 — Decision Intelligence
## Classification, Explainability, Visualization, Recommendation

You turn two models into a decision a store manager will actually act on, and you build the artifact the judges will click through and remember. This role carries most of the "out of the box" differentiation — own it with that in mind.

Read first: [root plan](../../README.md) §3 (differentiation strategy — this is largely your job to implement), and [`docs/RESEARCH.md`](../RESEARCH.md) §4 (ABC-XYZ), §5 (newsvendor), §6 (imbalance), §7 (explainability), §8 (dashboard patterns).

Your outputs, exactly:
- `data/processed/stockout_labels.csv`
- `notebooks/04_stockout_classification.ipynb`
- `models/stockout_model.pkl` + `models/stockout_model_card.md`
- `src/recommendation_engine.py`
- `reports/explainability.md`
- `dashboard/app.py` (Streamlit)
- co-owner of `reports/final_pitch.pdf`

---

## Hour-by-Hour Plan

### Hour 0–1 — Setup & Pre-Build
Standard environment setup. `data/raw/` already has a full practice dataset in place (root [`README.md`](../../README.md) §8), so you don't need to wait for Students 1 & 2 to finish anything before starting: **build the entire Streamlit shell right away**, page layout, the six mandatory tabs, dummy tables/charts with placeholder numbers pulled from the practice data. At hour 13 you'll swap dummy data for real predictions — the layout work shouldn't wait on anyone. If the official dataset replaces the practice one mid-event, only the numbers change, not your layout code.

### Hour 1–6 — Design the Recommendation Card & Operational Definitions

Before any modelling, nail down two definitions in writing (put them at the top of `stockout_model_card.md` — judges will ask "what exactly counts as a stock-out?"):

1. **Operational definition of `stockout_flag`.** Recommended: a date×store×product row is a stock-out event if `closing_stock == 0` at end of day, OR the corrected demand estimate exceeds available stock (`opening_stock + received_stock`) at any point in the next 7 days. Pick one, state it precisely, keep it consistent with Student 1's `master_table.stockout_flag` column.
2. **The recommendation card layout**, matching the brief's example exactly:
   ```
   STORE {store} - {product}
   Predicted 7-day demand: {forecast} units
   Current stock: {current} units | Incoming stock: {incoming} units
   Stock-out probability: {prob}% | Risk: {tier}
   Recommended additional replenishment: {reorder_qty} units
   Why? {top 3-4 reason phrases, e.g. "Weekend approaching, promotion running, 16% recent sales growth, local festival."}
   MANAGER ACTION: {one-line imperative}
   ```

### Hour 6–13 — Model 2: Stock-out Risk Classification (Round 2, parallel with Student 2)

You can start on Student 2's **in-progress** feature branch as soon as `feature_table.csv` has a first version (don't wait for their final polish) — this is what makes true parallelism work.

1. **Handle class imbalance correctly** (RESEARCH.md §6): check the positive-class rate first (`stockout_flag.mean()`). In the current practice dataset it's ~16% overall (but ranges 0–100% per SKU — some products are chronically at risk, others never stock out, which is realistic and gives your model real signal to find). Still imbalanced enough to matter: use `class_weight='balanced'` (Random Forest) or `scale_pos_weight = n_neg/n_pos` (XGBoost) **before** reaching for SMOTE. Document that you tried weighting first — this ordering itself is a scoring point.
2. **Train and compare, from the permitted list:** Logistic-style baseline is not on the list, so start with **Decision Tree**, then **Random Forest**, **XGBoost**, and optionally **Naive Bayes** or **KNN** as a contrast (they'll likely underperform on this kind of tabular/imbalanced problem — showing that comparison and explaining *why* tree ensembles win is itself good analysis).
3. **Evaluate:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix — for imbalanced problems, **lead with Recall/F1/ROC-AUC in your narrative**, not Accuracy (a model predicting "never stock out" can hit 90%+ accuracy on a rare event and still be useless — say this explicitly in your report, it shows judges you understand the trap).
4. Append your section to `reports/model_comparison.md` (shared file with Student 2 — keep formatting consistent).
5. Convert `stockout_probability` → `risk_tier` using the brief's fixed thresholds (High ≥0.70, Medium 0.40–0.70, Low <0.40) — **do not** re-derive your own thresholds, the brief fixes these.
6. Save `models/stockout_model.pkl`, write `models/stockout_model_card.md`.

### Hour 6–13 (interleaved) — Explainability

1. Compute **permutation importance** (`sklearn.inspection.permutation_importance`) on the validation set for the stock-out model — more reliable than raw Gini/gain importance (RESEARCH.md §7).
2. Build the **per-prediction reason generator**: for a given high-risk row, compute simple directional deltas by comparing the row's feature values against the population/category average for features known to matter (promotion active, weekend approaching, recent sales growth = compare `lag_1` vs `lag_7`, festival flag, temperature/rain anomaly) and phrase them exactly in the brief's style: `"Promotion active +31%"`. This does not require SHAP — a documented rule-based delta against the permutation-importance-ranked features is sufficient and fully within the sklearn-only constraint.
3. Write `reports/explainability.md`: global importance ranking + 3–4 worked examples of the per-prediction explanation, in prose a store manager (not a data scientist) would understand.

### Hour 8–13 (as soon as EDA revenue/CV numbers exist) — ABC-XYZ Segmentation & Recommendation Engine

This is your headline differentiator (RESEARCH.md §4–§5). Build `src/recommendation_engine.py`:

1. **ABC**: rank products by total revenue (from Student 1's Pareto chart data), cumulative-% cutoffs → `abc_class`.
2. **XYZ**: rank products by demand coefficient of variation (from Student 1's volatility chart) → `xyz_class`.
3. **Segment-specific service level** → `Z` score per (ABC,XYZ) combination, e.g. AX/AY → 97–99% (Z≈2.05–2.33), CZ → 85–90% (Z≈1.04–1.28). Put this mapping in a small, readable config dict — judges love seeing an explicit, tunable policy table, not a hardcoded magic number.
4. **Safety Stock** = `Z × σ_demand(rolling_std_7 or product-level std) × sqrt(lead_days)`.
5. **Recommended Stock** = `demand_model.predict(next_7_day_demand) + Safety Stock`.
6. **Reorder Quantity** = `max(0, Recommended Stock - current_stock - incoming_stock)`.
7. **Optional depth layer — newsvendor nudge**: compute `critical_ratio = (mrp-cost_price) / ((mrp-cost_price) + cost_price*holding_rate)` per product (pick a reasonable `holding_rate`, e.g. 1.5–2%/week, state your assumption) and use it to nudge the Z-score up/down slightly within the segment's band. Implement this only after steps 1–6 work end-to-end — it's a bonus, not a blocker.
8. Output the **final manager-ready table** exactly matching the brief's example columns: `Store, Product, Current Stock, 7-Day Forecast, Stock-out Prob., Risk, Recommended Order` — write to `data/processed/recommendations.csv` for the dashboard to read.

### Hour 13 — CHECKPOINT 2
Receive frozen `feature_table.csv` + `demand_model.pkl` from Student 2. Wire real predictions into your recommendation engine and dashboard, replacing placeholder data.

### Hour 13–20 — Dashboard Build (Round 3 core)

Build `dashboard/app.py` in Streamlit with exactly these tabs/sections (mandatory minimum from the brief):

| Section | Must show |
|---|---|
| Executive Summary | Revenue, growth, stock-out rate, inventory value, products at risk, **Estimated Lost Sales in ₹** (censored-demand gap × price — your strongest number) |
| Demand Intelligence | Actual vs. forecast chart, category trend, store trend, forecast error over time |
| Inventory Risk | High/Medium/Low risk table + heatmap (store × category) |
| Manager Action Centre | Filterable recommendation cards / table (the brief's exact card format), sortable by risk |
| Model Performance | MAE/RMSE/MAPE/R² and Accuracy/Precision/Recall/F1/ROC-AUC/confusion matrix, plus the per-segment breakdown from Student 2 |
| Explainability | Global permutation importance chart + per-prediction reason display when a row is selected |

**Bonus — What-If simulator:** sidebar sliders/toggles for discount %, promotion on/off, supplier delay (+N lead days), festival flag. On change, copy the selected row's feature vector, edit the relevant columns, re-run `.predict()`/`.predict_proba()` on both saved models (no retraining), and re-render the recommendation card live. This is explicitly called out as optional in the brief and is a strong differentiator for a 24-hour build.

### Hour 20–24 — Integration, Pitch, Polish
Run the full `streamlit run dashboard/app.py` against real, final model outputs during the team bug bash. Co-write `reports/final_pitch.pdf` — you lead the "which products/stores need action now" and "how the recommendation reduces business loss" sections, since the dashboard and recommendation engine are yours. Rehearse a **2-minute demo path**: Executive Summary → one High-risk recommendation card with its "Why?" → What-If slider live → close on the ₹ business-impact number.

---

## Deliverables Checklist

- [ ] `data/processed/stockout_labels.csv` with a clearly documented operational definition
- [ ] Class imbalance handled (weighting tried before/instead of blind SMOTE), documented
- [ ] `reports/model_comparison.md` classification section: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix — with an explicit note on why Accuracy alone is misleading here
- [ ] `models/stockout_model.pkl` + `models/stockout_model_card.md`
- [ ] `reports/explainability.md` — permutation importance + worked per-prediction examples in plain English
- [ ] `src/recommendation_engine.py` — ABC-XYZ segmentation, segment-specific safety stock, reorder quantity, risk tier
- [ ] `data/processed/recommendations.csv` matching the brief's exact output table columns
- [ ] `dashboard/app.py` — all six mandatory sections + (bonus) What-If simulator
- [ ] Recommendation card matches the brief's exact wording/format

## Common Pitfalls to Avoid

- Reporting only Accuracy on an imbalanced stock-out target — a trivial "always predict no stock-out" model can look great on Accuracy alone; lead with Recall/F1/ROC-AUC instead.
- Inventing your own risk-tier thresholds instead of using the brief's fixed 0.70/0.40 cutoffs.
- Building the dashboard against hardcoded numbers and forgetting to wire in the real models before the final run — start the wiring the moment `feature_table.csv`/`demand_model.pkl` land at hour 13, don't leave it to hour 20.
- A "reason" explanation that's just the raw permutation-importance ranking with no per-row context — the brief's example is specific to *that* store-product on *that* day; genericity reads as a black box, which the brief explicitly says is unacceptable.
- Skipping the ABC-XYZ segmentation because "the mandatory formula alone is enough" — the mandatory formula is table stakes; the segmentation is what makes this submission distinguishable from every other team's.
