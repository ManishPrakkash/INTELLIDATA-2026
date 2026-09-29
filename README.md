# STOCKSENSE — IntelliData 2026 Data Science Hackathon

**"Predict Demand. Prevent Stock-outs. Power Better Decisions."**
Team submission for NovaMart Retail Pvt. Ltd. — a working decision-support system, not just a model.

> Team size: 3 | Duration: 24 hours | Rounds: Data & EDA → Feature Eng & ML → Intelligence Layer & Prototype
> Full official brief: [`IntelliData 2026 Challenge.pdf`](./IntelliData%202026%20Challenge.pdf)

This document is the master plan. It is the single source of truth for architecture, timeline, role contracts, and the differentiation strategy the team is executing to win. Each teammate also has a dedicated, hour-by-hour guide:

| Role | Owner | Guide |
|---|---|---|
| Student 1 — Data Analyst | Cleaning, integration, EDA, statistics, data quality | [`docs/role_1_data_analyst/README.md`](docs/role_1_data_analyst/README.md) |
| Student 2 — ML Engineer | Feature engineering, forecasting, validation | [`docs/role_2_ml_engineer/README.md`](docs/role_2_ml_engineer/README.md) |
| Student 3 — Decision Intelligence | Classification, explainability, visualization, recommendation | [`docs/role_3_decision_intelligence/README.md`](docs/role_3_decision_intelligence/README.md) |

Deep research backing every technical decision below (formulas, papers, benchmark practices, sources): [`docs/RESEARCH.md`](docs/RESEARCH.md)
Shared data contract every role must respect: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)
Where every real-world number in the practice dataset came from, with direct links — **cite this in the pitch**: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)

---

## 1. The Business Problem (restated in one paragraph)

NovaMart runs multi-city supermarkets/hypermarkets/express stores. Replenishment today is manual rule-of-thumb, causing **stock-outs** (lost revenue) and **overstock** (locked capital, expiry risk). We must, for every Store × Product: forecast next-7-day demand, estimate stock-out probability and risk tier (High/Medium/Low), and turn that into a reorder quantity + plain-English reason a store manager can act on today — plus a dashboard management can trust.

Grain of the master analytics table: **one row = one date × one store × one product.**

## 2. Solution Architecture

```
transactions.csv ─┐
products.csv      │
stores.csv        ├─► CLEAN & VALIDATE ─► AGGREGATE (daily, store×product) ─► MERGE ─► MASTER TABLE
inventory.csv      │        (Student 1)                                                     │
external_factors.csv┘                                                                        │
                                                                                              ▼
                                                                              EDA + STATISTICAL TESTS (Student 1)
                                                                                              │
                                                                                              ▼
                                                                         FEATURE ENGINEERING (Student 2)
                                                                     time / lag / rolling / inventory / price-promo
                                                                                              │
                                                    ┌─────────────────────────────────────────┴─────────────────────────────┐
                                                    ▼                                                                       ▼
                                    MODEL 1 — DEMAND FORECAST (regression)                          MODEL 2 — STOCK-OUT RISK (classification)
                                    baseline → Linear/Tree/RF/XGBoost                                Logistic/RF/XGBoost + imbalance handling
                                    (Student 2)                                                       (Student 3, built on Student 2's features)
                                                    └─────────────────────────────────────────┬─────────────────────────────┘
                                                                                              ▼
                                                                      EXPLAINABILITY (permutation importance)
                                                                      BUSINESS ACTION LAYER (safety stock, reorder qty,
                                                                      risk tier, censored-demand correction, newsvendor logic)
                                                                                              │
                                                                                              ▼
                                                                    STREAMLIT DASHBOARD + WHAT-IF SIMULATOR (Student 3)
                                                                    Executive Summary | Demand Intelligence | Inventory Risk |
                                                                    Manager Action Centre | Model Performance | Explainability
```

## 3. What Will Make This a Winning Entry (Differentiation Strategy)

Every hackathon team in the room will build "a forecast + a classifier + a bar chart." The evaluation rubric explicitly rewards **business understanding, explainability, and a real recommendation** — not model accuracy alone (ML Modelling is only 20% of the score; Business + Explainability + Viz + Pitch together are ~45%). We differentiate on substance the judges can verify in 5 minutes, not slideware:

1. **Censored-demand correction (biggest technical differentiator).** On days a product stocked out, `quantity sold` under-states *true* demand — the model trains on a lie. We detect stock-out days (`closing_stock == 0` or `sold < opening+received` mismatch) and reconstruct a corrected demand signal (time-weighted / rolling-average recovery, documented in RESEARCH.md) before computing `next_7_day_demand`. This is a subtlety almost no undergraduate team will catch, and it directly answers the judge question "why can this be trusted?" See [`docs/RESEARCH.md#censored-demand`](docs/RESEARCH.md#3-censored-demand-correction-the-single-biggest-differentiator).
2. **ABC-XYZ segmentation driving a *differentiated* safety-stock policy** instead of one global buffer. Class A/X (high value, stable) gets a tight service level; Class C/Z (low value, erratic) gets a cheap, generous buffer. This is textbook supply-chain practice most student teams skip entirely.
3. **Newsvendor critical-ratio reorder logic** layered on top of the mandatory `Forecast + Safety Stock` formula, using `cost_price` and `mrp` from `products.csv` to trade off holding cost vs. stock-out cost per product — ties the recommendation directly to money, not just units.
4. **Estimated Lost Sales in ₹, computed end-to-end** (censored-demand gap × selling price), reported on the Executive Summary tile so the pitch can say *"our system would have recovered ₹X in the last 30 days"* — a concrete, defensible number.
5. **Model-trust panel**: backtested WAPE/MAE by store-type and by ABC segment (not just one global number), so the dashboard can honestly say where the model is strong vs. weak — judges reward candour over false confidence.
6. **What-If simulator** (festival toggle / discount slider / supplier delay days) recomputing forecast → risk → reorder live — turns a static report into an interactive decision tool, satisfying the optional bonus explicitly listed in the brief.
7. **Everything is reproducible from raw CSV → dashboard with one command** (`src/run_pipeline.py`), and the repo includes a synthetic-data generator so the *entire* pipeline is dry-run and demo-able even before/independent of the official dataset drop.
8. **Our practice/development data is grounded in real, cited public data**, not arbitrary randomness: the holiday/festival calendar and city climate normals used while building and testing (before the official data lands) come from the actual 2026 Tamil Nadu holiday list and published climate records for these exact four cities — every figure has a direct source link in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md). This is a small thing to mention in the pitch, but it signals rigor: we didn't just simulate demand, we validated our assumptions against reality before the real data ever arrived.

## 4. Mandatory Deliverables Checklist (from the official brief)

- [ ] Cleaned master dataset (CSV) — `data/processed/master_table.csv`
- [ ] Data Quality Report — `reports/data_quality_report.md`
- [ ] EDA + modelling notebook(s) — `notebooks/`
- [ ] ≥3 statistical tests with H0/H1/test/p-value/interpretation — `reports/statistical_tests.md`
- [ ] Model comparison table + justification — `reports/model_comparison.md`
- [ ] Saved model / reproducible pipeline — `models/`, `src/run_pipeline.py`
- [ ] Visualization / dashboard / prototype — `dashboard/app.py`
- [ ] README with setup & usage (this file + role guides)
- [ ] Final presentation / pitch — `reports/final_pitch.pdf`
- [ ] GitHub repository link submitted as instructed

## 5. Repository Structure

```
STOCKSENSE_TEAM_NAME/
  data/
    raw/            # official CSVs land here untouched (or synthetic dry-run data)
    processed/      # master_table.csv, cleaned intermediates
  notebooks/         # 01_eda, 02_features, 03_demand_model, 04_stockout_model, 05_dashboard_dev
  src/               # cleaning.py, features.py, demand_model.py, stockout_model.py,
                      # recommendation_engine.py, generate_synthetic_data.py, run_pipeline.py
  models/            # serialized .pkl models + model_card.md per model
  dashboard/         # app.py (Streamlit)
  reports/           # data_quality_report.md, statistical_tests.md, model_comparison.md,
                      # explainability.md, final_pitch.pdf
  docs/              # RESEARCH.md, DATA_DICTIONARY.md, per-role READMEs (this planning layer)
  README.md
  requirements.txt
```

## 6. The 24-Hour Timeline

Rounds map to the brief's mandatory rounds; roles work **in parallel from hour 0** using the synthetic dataset so nobody is idle waiting on a "phase" to finish. Two hard sync checkpoints keep the three streams compatible.

| Hours | Phase | Student 1 (Data Analyst) | Student 2 (ML Engineer) | Student 3 (Decision Intelligence) |
|---|---|---|---|---|
| 0–1 | Kickoff | Clone repo (practice dataset already in `data/raw/`, §8); if official data is released, replace those files; confirm data dictionary | Same setup; review feature-engineering plan | Same setup; scaffold Streamlit skeleton against practice master table |
| 1–5 | **Round 1** — Data & EDA | Lead: cleaning, dedup, dtype/category fixes, inventory reconciliation, aggregation, merge → `master_table.csv`; data quality report | Study lag/rolling feature spec against synthetic data; pre-build feature functions (`src/features.py`) unit-tested on synthetic table | Build dashboard shell with dummy data (KPI tiles, table layout); draft recommendation card design |
| 5 | **CHECKPOINT 1** | Freeze & hand off `master_table.csv` + data dictionary confirmed | | |
| 5–6 | EDA + Stats | Finish 6 business-question charts + 3 statistical tests (`reports/statistical_tests.md`) | Re-run feature functions on the *real* master table; fix schema drift | Review EDA insights to decide which "reasons" go into explainability copy |
| 6–13 | **Round 2** — Feature Eng & ML | Support: validate feature distributions, sanity-check leakage per business question | Lead: full feature set, censored-demand correction, baseline (7-day mean) → Linear/Tree/RF/XGBoost regression, time-aware CV, metric comparison → `models/demand_model.pkl` | Build stock-out label logic + start classification model in parallel on Student 2's in-progress features branch |
| 13 | **CHECKPOINT 2** | Demand model frozen; feature table frozen; stock-out labels frozen | | |
| 13–20 | **Round 3** — Intelligence Layer | Support: write plain-English narrative for Exec Summary; validate KPI formulas | Support: expose model + feature pipeline via `src/run_pipeline.py`; help wire dashboard to real predictions | Lead: finalize classification model, permutation importance, ABC-XYZ segmentation, safety-stock/newsvendor recommendation engine, full Streamlit dashboard incl. What-If simulator |
| 20–22 | Integration | End-to-end run of `run_pipeline.py` on real data; bug bash across all three modules together | | |
| 22–23 | Pitch prep | All three co-write `final_pitch.pdf`; rehearse the 2-minute story (discovery → prediction → trust → action → business impact); cite [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) if asked where the practice data's calendar/climate assumptions came from | | |
| 23–24 | Final polish | Clean notebooks, finalize README links, push, verify GitHub renders correctly, submit | | |

**Rule:** nobody works in a silo past hour 5 without the other two knowing the current state of `data/processed/master_table.csv` — it is the shared contract. Post in your team chat every time you push a new version of it.

## 7. Cross-Role Data Contract (so parallel work doesn't collide)

| Producer | File | Consumer | Must contain |
|---|---|---|---|
| Student 1 | `data/processed/master_table.csv` | Students 2 & 3 | one row per date×store×product; demand, price, promo, stock, store/product attrs, external factors — see [`DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) |
| Student 1 | `reports/data_quality_report.md` | Whole team + judges | issue, count, action taken, justification |
| Student 2 | `data/processed/feature_table.csv` | Student 3 | all engineered features, leakage-safe, keyed by date/store/product |
| Student 2 | `models/demand_model.pkl` + `reports/model_comparison.md` | Student 3 | `.predict()`-ready regressor + metrics table |
| Student 3 | `data/processed/stockout_labels.csv` | Student 2 (for reference) | `stockout_flag` definition, used back-check against demand model residuals |
| Student 3 | `dashboard/app.py` | Whole team | reads only from `models/` and `data/processed/`, never recomputes cleaning logic |

## 8. Dataset Status — Read This First

**The official IntelliData 2026 brief does not include a downloadable dataset — only a schema and a handful of sample rows.** NovaMart's real CSVs are released to teams only at the event itself. There is nothing to "download" from the organizers before that.

To make sure nobody is idle waiting for it, `data/raw/` is **already populated** with a full-scale, schema-correct **practice dataset**, generated locally by `src/generate_synthetic_data.py` and committed to this repo so every teammate can start Round 1 immediately:

| File | Rows | Notes |
|---|---|---|
| `transactions.csv` | 66,759 | 2026-08-01 → 2027-01-27 (180 days), 4 stores × 60 products |
| `inventory.csv` | 39,001 | daily store×product stock ledger, with **lead-time-delayed replenishment** (orders arrive after each product's real `lead_days`, not instantly) so stock-outs are a genuine, learnable event rather than a rounding-error rarity |
| `products.csv` | 60 | 6 categories, casing intentionally inconsistent (see below) |
| `stores.csv` | 4 | S01 Coimbatore, S02 Chennai, S03 Madurai, S04 Salem — matches the brief exactly |
| `external_factors.csv` | 720 | daily × city weather/holiday/festival/event flags — **grounded in the real 2026–27 Tamil Nadu holiday calendar and real climate normals for these four cities**, not random noise; full citations in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) |

Injected data-quality traps present right now (Student 1 will find and handle these in Round 1 — counts below are what the *current* seed produced, re-running the generator with a different `--seed` will shuffle them):

| Trap | Count found |
|---|---|
| Duplicate `transaction_id` rows | 657 |
| Impossible (negative) `quantity` rows | 341 |
| Inventory arithmetic mismatches (`closing ≠ opening+received-sold`) | 1,100 |
| Missing `temp_c` values | 23 / 720 (3.2%) |
| Category casing/spacing variants (e.g. `Beverages` / `BEVERAGES` / ` Dairy` / `PersonalCare`) | 21 distinct raw strings across 6 true categories — note `PersonalCare` (no space) survives a naive `.str.lower().str.strip()` clean and still won't merge with `Personal Care`; Student 1 needs a stronger normalization (strip non-alphanumerics too) |
| Zero-closing-stock (stock-out) rows | 6,189 (**15.87%** overall; per-SKU rate ranges from 0% to 100%, median 6.4% — realistic heterogeneity, not a rare-event problem) |
| Sparse-history products (<14 days of transaction data) | 6 / 60 |

**Why the stock-out rate matters:** an earlier version of this generator replenished stock instantly whenever it ran low, which made real stock-outs occur in well under 0.1% of rows — statistically useless for training/evaluating Model 2 (a test fold could easily contain zero positive examples). Replenishment now has a real lead-time delay per product, so a demand spike while an order is in transit can genuinely cause a stock-out — the ~16% rate this produces is in line with typical retail stock-out benchmarks and gives Model 2 something real to learn.

**Known, inherent limitation (not fixable, matches the real brief):** the brief's own sample `stores.csv` lists exactly these four stores — S02 is the only Hypermarket and S03 the only Express store. Any "does demand differ by store type" statistical test is therefore partially confounded with the individual store's own effect, not store type in the abstract, for two of the three types. Disclose this explicitly in `reports/statistical_tests.md` rather than treating the ANOVA result as a clean answer — the real event data will very likely have the same limitation.

**When the real NovaMart CSVs are handed out at the event:** drop them into `data/raw/`, overwriting the practice files (same filenames, same schema per [`DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)). Every notebook and script downstream is schema-driven, not hardcoded to these specific values, so the switch should require zero code changes — only a re-run.

## 9. Setup & Usage

```bash
git clone <your-fork-url> STOCKSENSE_TEAM_NAME
cd STOCKSENSE_TEAM_NAME
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# data/raw/ already contains a full practice dataset (see §8). To regenerate it
# (different scale/seed, or after the real data replaces it and you want a fresh
# practice set again for testing):
python src/generate_synthetic_data.py --days 180 --products 60 --seed 42 --out data/raw/

# Full pipeline (cleaning -> features -> both models -> recommendations):
python src/run_pipeline.py

# Dashboard:
streamlit run dashboard/app.py
```

## 10. Evaluation Rubric Mapping (know where every hour of effort scores)

| Area | Weight | Where we deliver it |
|---|---|---|
| Business Understanding | 10% | This README §1, Exec Summary tile, final pitch |
| Data Cleaning & Quality | 15% | Student 1 — `reports/data_quality_report.md` |
| EDA & Statistical Reasoning | 15% | Student 1 — notebooks + `reports/statistical_tests.md` |
| Feature Engineering | 15% | Student 2 — `src/features.py`, leakage checklist |
| ML Modelling & Validation | 20% | Student 2 & 3 — `reports/model_comparison.md`, time-aware CV |
| Explainability & Recommendation Logic | 10% | Student 3 — permutation importance + recommendation engine |
| Visualization / Prototype | 10% | Student 3 — Streamlit dashboard |
| Final Pitch & Team Understanding | 5% | All — every member must be able to answer questions on *any* module |

## 11. Permitted ML Algorithms (hard constraint)

Linear Regression, Decision Trees, SVM, XGBoost, KNN, Naive Bayes, Random Forest. **No deep learning.** Everything in this plan is designed within that constraint.

## 12. Communication Rhythm

- Stand-up at hour 0, hour 5, hour 13, hour 20 (5 minutes each): what's done, what's blocking, what's next.
- Shared `data/processed/master_table.csv` is versioned in git — commit often with clear messages so teammates can `git pull` and keep moving.
- The practice dataset in `data/raw/` (§8) means hour 0–5 is never idle waiting on the organizers — swap in the real CSVs the moment they're released and re-run.
