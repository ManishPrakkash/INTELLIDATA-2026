# StockSense — Research Dossier (Planning Phase)

This document backs every non-obvious technical decision in the [master plan](../README.md) with a formula, a rationale, and a source. Read this once as a team before Round 1 starts — it is what turns "we built a model" into "we built a defensible business system."

---

## 1. Inventory Formulas We Will Actually Use

### 1.1 Safety Stock (statistical method — preferred over the crude min/max method)

The brief's simple version: `Recommended Stock = Forecast Demand + Safety Stock`. Two ways to compute safety stock exist:

- **Crude range method:** `(max daily sales × max lead time) − (avg daily sales × avg lead time)`. Simple but ignores how demand is actually distributed — punished by judges as "rule-based, not data-driven."
- **Statistical (service-level) method — what we use:**

  ```
  Safety Stock = Z(service_level) × σ_demand × √(lead_time_days)
  ```

  where `σ_demand` is the store-product's rolling standard deviation of daily demand and `Z` is the standard-normal inverse CDF for the target service level (e.g., Z=1.65 for 95%, Z=1.28 for 90%, Z=2.33 for 99%). This is the industry-standard formula and lets us set **different service levels per ABC-XYZ segment** (§3 below) instead of one flat number for every SKU.

  Source: [Fishbowl — 6 Safety Stock Formula Variations](https://www.fishbowlinventory.com/blog/calculating-the-safety-stock-formula-6-variations-key-use-cases), [Lokad — Safety Stock with Sales Forecasting](https://www.lokad.com/calculate-safety-stocks-with-sales-forecasting/), [ABC Supply Chain — Safety Stock Formula & Calculation](https://abcsupplychain.com/safety-stock-formula-calculation/)

### 1.2 Reorder Point & Reorder Quantity (mandatory formula from the brief, kept exactly)

```
Reorder Quantity = max(0, Recommended Stock − Current Stock − Incoming Stock)
Recommended Stock = Forecast(next_7_day_demand) + Safety Stock
```

Standard reorder-point theory confirms this shape: `Reorder Point = (avg daily sales × lead time) + safety stock` — our formula is the multi-day, forecast-driven generalization of the same idea.
Source: [Netstock — Reorder Point Formula](https://www.netstock.com/blog/reorder-point-formula/), [GAINS — Reorder Point vs Safety Stock](https://gainsystems.com/blog/reorder-point-vs-safety-stock-balancing-inventory-in-retail/)

### 1.3 Risk Tiers (mandatory, kept exactly as specified)

```
High Risk:    stockout_probability >= 0.70
Medium Risk:  0.40 <= stockout_probability < 0.70
Low Risk:     stockout_probability < 0.40
```

---

## 2. Time-Aware Modelling Practice

### 2.1 Why random K-fold CV is disqualifying here

The brief explicitly states: *"Use time-aware train/validation/test splitting. Do not randomly mix future dates into training."* Random K-fold on time series leaks future information into training folds and inflates validation scores — a classic and heavily-penalized mistake.

### 2.2 What we use instead: expanding-window walk-forward validation

- Use `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` on the chronologically sorted master table, per store-product or globally with a `date` sort key.
- Each fold trains on all data *before* a cutoff and validates on the immediately following block — this "simulates real-world forecasting" because the model only ever sees the past.
- Final holdout: reserve the **last 7–14 days** of the whole dataset as a true test set, touched only once, after model selection is locked.

Sources: [Medium — Respect the Order: CV in Time Series](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1), [MachineLearningMastery — Backtest ML Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/), [TowardsDataScience — 4 Things to Do When Applying CV with Time Series](https://towardsdatascience.com/4-things-to-do-when-applying-cross-validation-with-time-series-c6a5674ebf3a/)

### 2.3 Leakage checklist (apply to every engineered feature)

A feature is safe only if it could have been *computed and known* at the moment we make the prediction (before the 7 days it forecasts). Concretely:
- `lag_1/7/14`, `rolling_mean/std_7/14` — safe (past only).
- Anything computed using `sold`, `closing_stock`, or `stockout_flag` **from the target week itself** — leakage, forbidden.
- Calendar features (`day_of_week`, `festival_flag`, `weekend_flag`) for the *target* week — safe, because calendars are known in advance.
- Weather (`temp_c`, `rain_mm`) for the target week — **not safe** in real life (weather forecasts are uncertain) unless explicitly modeled as a forecast-with-uncertainty; for the hackathon we use only *lagged/historical* weather as a feature and note this limitation explicitly in the model card (judges reward teams who flag their own leakage risk rather than silently ignore it).

---

## 3. Censored-Demand Correction — The Single Biggest Differentiator

**The problem:** `transactions.csv` and `inventory.csv` only record what was *sold*, not what was *wanted*. On a day a product stocks out mid-day, observed sales are truncated — the true demand was higher. If we train `next_7_day_demand` directly on raw `sold`, the model **systematically learns to under-forecast exactly the SKUs that stock out most often** — the vicious cycle: under-forecast → under-order → stock out again → training data still censored. This is a known, published problem in retail forecasting (e.g., the 2025 "FreshRetailNet-50K" stockout-annotated censored demand dataset).

**What we do about it (practical, hackathon-scoped version):**
1. Flag censored observations: a date×store×product row is censored when `closing_stock == 0` (or `sold` implies stock hit zero before end of day, detectable via `opening + received - sold <= 0`).
2. For censored rows, replace raw `sold` with a **corrected demand estimate** before computing lags/rolling stats and the `next_7_day_demand` target:
   - Simple, defensible heuristic (what we implement): `corrected_demand = max(sold, rolling_mean_demand_same_weekday_last_4_weeks)` — i.e., never let a stock-out day drag the demand signal *down* below what a normal day of that type would have shown.
   - Optional stretch (if time permits): fit a per-category Tobit-style adjustment (censored regression) instead of the heuristic — mention this as a "future work" line in the pitch even if not implemented; naming it correctly signals depth of understanding.
3. Report both numbers on the dashboard: **raw historical sales** vs. **corrected demand estimate**, with the gap between them multiplied by `selling_price` reported as **Estimated Lost Sales** (already a mandatory KPI in the brief) — this is the single most quotable number in the pitch.

Sources: [arXiv 2505.16319 — FreshRetailNet-50K: Stockout-Annotated Censored Demand Dataset](https://arxiv.org/abs/2505.16319), [MDPI/Inventions — A Reproducible, Leakage-Free Pipeline for Censored Demand Forecasting](https://doi.org/10.3390/inventions11050089)

---

## 4. ABC-XYZ Segmentation (differentiated inventory policy, not one-size-fits-all)

- **ABC** — rank products by revenue contribution (`quantity × selling_price`), cumulative %, Pareto-style: A = top ~70–80% of revenue, B = next ~15%, C = remainder. (The brief already asks for a Pareto/category-revenue chart in EDA — ABC reuses that exact analysis at product grain.)
- **XYZ** — rank products by **coefficient of variation** of daily demand (`std/mean`): X = CV < 0.5 (stable), Y = 0.5 ≤ CV < 1.0 (variable), Z = CV ≥ 1.0 (erratic). The brief already asks for a "which products are volatile — CV / rolling std" chart in EDA — XYZ reuses that exact analysis too, so this is near-zero extra work for a large payoff.
- **Business use:** apply a *higher* service level / Z-score (e.g., 97–99%) to AX/AY items (high value, must not stock out) and a *lower*, cheaper service level (85–90%) to CZ items (low value, erratic, not worth over-investing in). This single idea converts two mandatory EDA charts into a genuinely smarter, segment-aware recommendation engine.

Sources: [Medium — ABC-XYZ Inventory Classification with Python](https://medium.com/@ulas_yilmaz/abc-xyz-inventory-classification-with-python-50ebee552fe4), [SCMDojo — ABC XYZ Analysis Complete Guide](https://www.scmdojo.com/blog/abc-xyz-analysis-complete-guide), [ABC Supply Chain — ABC XYZ Analysis Guide](https://abcsupplychain.com/abc-xyz-analysis/)

## 5. Newsvendor Critical Ratio (money-aware reorder quantity)

Classic single-period inventory theory: the cost-optimal order quantity satisfies

```
Critical Ratio (CR) = Cu / (Cu + Co)
Cu (underage cost, i.e., cost of a stock-out) ≈ (selling_price − cost_price)   [lost margin]
Co (overage cost, i.e., cost of overstock)   ≈ (cost_price × holding_rate)     [capital lock-up / spoilage risk]
```

We use `CR` to nudge the Z-score/service-level used in the safety-stock formula (§1.1, §4) per product: high-margin, low-holding-cost items (`CR` close to 1) justify a higher service level than low-margin or high-spoilage items (short `shelf_life_days`, e.g., Dairy) where overstocking is riskier than a stock-out. This is an optional depth layer — implement if time allows after the mandatory formulas are working; mention explicitly in the pitch either way, since naming the correct economic framework is itself a credibility signal.

Source: [arXiv 2007.03870 — A New Generalized Newsvendor Model with Random Demand](https://arxiv.org/pdf/2007.03870), [arXiv 2106.15865 — Non-Parametric Generalised Newsvendor Model](https://arxiv.org/pdf/2106.15865)

---

## 6. Class Imbalance for Stock-out Classification

Stock-outs are rare events — the classifier's positive class will be a minority. Research consensus for tree-based models (which is all we're allowed — no deep learning):

- **Prefer class weighting over synthetic oversampling for tree ensembles.** SMOTE was designed for algorithms that benefit from denser neighborhoods (KNN, SVM); tree models like Random Forest/XGBoost already split on the raw distribution well, and SMOTE risks creating redundant/noisy synthetic points and overfitting.
- **XGBoost:** set `scale_pos_weight ≈ n_negative / n_positive`.
- **Random Forest / sklearn linear models:** set `class_weight='balanced'`.
- **Threshold tuning:** since the brief defines risk tiers directly from `stockout_probability` (0.70/0.40 cut points), do **not** blindly use the default 0.5 classification threshold for the binary `stockout_flag` metric reporting — report Precision/Recall/F1 at a threshold chosen by inspecting the Precision-Recall curve, and separately report the tiered probabilities for the business layer.
- If class weighting alone under-performs, SMOTE (or SMOTE-ENN) is an acceptable second attempt — but document that plain class-weighting was tried first, which is the methodologically correct order and a good thing to show in the model comparison table.

Sources: [Medium — XGBoost and Imbalanced Datasets: Strategies](https://medium.com/@dicee/xgboost-and-imbalanced-datasets-strategies-for-handling-class-imbalance-cdd810b3905c), [Medium — Handling Imbalanced Data with XGBoost: class_weight](https://medium.com/@thedatabeast/handling-imbalanced-data-with-xgboost-class-weight-parameter-c67b7257515b), [MachineLearningMastery — Algorithm Showdown on Imbalanced Data](https://machinelearningmastery.com/algorithm-showdown-logistic-regression-vs-random-forest-vs-xgboost-on-imbalanced-data/)

---

## 7. Explainability Without Deep Learning

The brief bans deep learning but mandates explainability "understandable to a store manager." Two model-native/model-agnostic options, both compatible with the permitted algorithm list:

- **Tree-based (Gini/gain) importance** — built into Random Forest/XGBoost, fast, but biased toward high-cardinality features and can be misleading (measures training-time impurity reduction, not real-world predictive contribution).
- **Permutation importance (preferred, what we use)** — model-agnostic: shuffle one feature's values in the validation set, measure how much the chosen metric degrades. This is what `sklearn.inspection.permutation_importance` computes, and it is considered more reliable than raw Gini importance because it measures actual predictive contribution on held-out data, not training-time splits.
- Present importance **per-prediction** (not just globally) for the mandatory "why is this product at risk" explanation card — approximate this without SHAP (not required, sklearn-only is fine) by reporting each risky feature's directional contribution using simple rule-based deltas (e.g., "promotion active" → show the average probability lift for promo=1 vs promo=0 rows in similar products), exactly matching the style of the example table in the brief (`Promotion active +31%`, etc.).

Sources: [MachineLearningMastery — Feature Importance and Selection with XGBoost](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/), [Medium — Permutation Feature Importance Explained](https://medium.com/data-science-explained/permutation-feature-importance-explained-76d1398b1c6f), [TowardsDataScience — A Guide to 21 Feature Importance Methods](https://towardsdatascience.com/a-guide-to-21-feature-importance-methods-and-packages-in-machine-learning-with-code-85a841f8b319/)

---

## 8. Dashboard / Prototype Patterns

Streamlit is the fastest path to a working, judge-clickable prototype within 24 hours (explicitly suggested by the brief as an optional tool). Patterns worth reusing, observed across public reference implementations of retail demand-forecasting dashboards:
- A form/sidebar to pick Store × Product, with a resulting **recommendation card** (matches the brief's exact example card format).
- A **What-If panel**: sliders/toggles for discount %, promotion on/off, supplier delay (+N days), festival flag — recomputes forecast → stock-out probability → reorder quantity live by re-calling the already-trained models with modified feature rows (no retraining needed live).
- Tabs matching the brief's mandatory dashboard sections exactly: Executive Summary, Demand Intelligence, Inventory Risk, Manager Action Centre, Model Performance, Explainability.

Sources: [GitHub — Retail Demand Forecasting & Inventory Optimization (Streamlit + SHAP)](https://github.com/TheManifester/AI-Demand-Forecasting-Inventory-Optimization-), [GitHub — Demand Forecasting System (Streamlit + XGBoost, what-if form)](https://github.com/harshantla-cloud/DEMAND-FORECASTING-SYSTEM), [GitHub — Demand Forecasting and Inventory Optimization (scenario + batch dashboard)](https://github.com/Jai-kp/Demand-Forecasting-and-Inventory-Optimization)

---

## 9. How Judges Actually Score Hackathons (so we spend the last 2 hours correctly)

Published guidance from hackathon-judging retrospectives converges on a few consistent points directly relevant to our time budget:
- Teams that **stop coding ~2 hours before the deadline and rehearse the pitch** consistently beat teams with marginally better code but a weaker demo — judges spend only minutes per team, so the story must land fast.
- **Business impact must be tangible and quantified**, not asserted — this is exactly why we compute Estimated Lost Sales in ₹ (§3) rather than only reporting MAE/RMSE.
- Judges explicitly reward when "the data science component was novel / an out-of-the-box solution" — hence §3–§5 above are not optional nice-to-haves, they are the trophy-differentiating layer.
- Skip line-by-line code walkthroughs in the demo; show the manager-facing recommendation card and dashboard first, model internals second only if asked.

Sources: [RBC Borealis — Why Hackathons Are Innovation Playgrounds](https://rbcborealis.com/news/why-hackathons-are-innovation-playgrounds/), [Eventflare — Crafting Effective Hackathon Judging Criteria](https://eventflare.io/journal/crafting-effective-hackathon-judging-criteria-a-step-by-step-guide), [TAIKAI — Hackathon Judging: 6 Criteria to Pick Winning Projects](https://taikai.network/en/blog/hackathon-judging)

---

## 10. Summary — What This Changes vs. a "By-the-Book" Submission

| Mandatory brief item | Baseline approach | Our upgrade |
|---|---|---|
| `next_7_day_demand` target | Raw historical `sold` | Censored-demand corrected `sold` (§3) |
| Safety stock | One global buffer | ABC-XYZ-segmented service levels (§4) |
| Reorder quantity | Forecast + flat safety stock | + Newsvendor critical-ratio nudge using margin & shelf life (§5) |
| Class imbalance | Default classifier | `scale_pos_weight`/`class_weight` first, threshold-tuned, SMOTE only as documented fallback (§6) |
| Explainability | Global feature importance chart | Permutation importance + per-prediction directional deltas in the exact card style the brief shows (§7) |
| Dashboard | Static charts | Six mandatory sections + live What-If simulator (§8) |
| Pitch | Model metrics recap | Quantified ₹ business impact + honest model-trust panel (§9) |
