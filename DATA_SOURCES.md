# Data Sources — Reference for Pitch & Audit Trail

The official NovaMart CSVs (see root [`README.md`](../README.md) §8) are not published anywhere online — they are handed out at the event. Everything in `data/raw/` right now is a **practice dataset** we generate ourselves with [`src/generate_synthetic_data.py`](../src/generate_synthetic_data.py). To make that practice data more than random noise, two of its dimensions — the **holiday/festival calendar** and the **city climate normals** — are grounded in real, cited, publicly available data for the exact four cities and time window (Aug 2026 – Jan 2027) used in the brief.

**Use this file to answer, live in the pitch, the question "where did your data come from?"** — every number below has a direct link. Nothing here is presented as the official NovaMart dataset; it is disclosed as our own research-grounded practice/development data, built ahead of the official data drop.

---

## 1. Tamil Nadu / India Public Holiday & Festival Calendar (2026–27)

Used to set realistic `holiday` / `festival` flags in `external_factors.csv` instead of randomly scattered ones — `TN_CALENDAR_2026_27` in [`src/generate_synthetic_data.py`](../src/generate_synthetic_data.py).

| Date | Event | Flag(s) set | Status | Source |
|---|---|---|---|---|
| 2026-08-15 | Independence Day | holiday | Official (national, fixed date every year) | [Public Holidays in India 2026 – timeanddate.com](https://www.timeanddate.com/holidays/india/2026) |
| 2026-08-26 | Eid-e-Milad / Onam window | holiday + festival | Official (Tamil Nadu 2026 gazetted list) | [Tamil Nadu govt issues 2026 public holiday list — The News Minute](https://www.thenewsminute.com/tamil-nadu/tamil-nadu-govt-issues-2026-public-holiday-list-with-24-declared-holidays) |
| 2026-09-04 | Janmashtami | festival | Official (2026 restricted/gazetted holiday list) | [India Public Holidays 2026 & Festivals — Zimyo](https://www.zimyo.com/insights/public-holidays-in-india/) |
| 2026-09-14 | Ganesh Chaturthi | festival | Official (2026 restricted holiday list) | [Zimyo — List of National & Public Holidays in India 2026](https://www.zimyo.com/insights/public-holidays-in-india/) |
| 2026-10-02 | Gandhi Jayanti | holiday | Official (national, fixed date every year) | [timeanddate.com — Holidays and Observances in India 2026](https://www.timeanddate.com/holidays/india/2026) |
| 2026-10-20 | Dussehra | holiday + festival | Official (2026 gazetted list) | [timeanddate.com — Holidays and Observances in India 2026](https://www.timeanddate.com/holidays/india/2026) |
| 2026-11-08 | Deepavali (Diwali) | holiday + festival | Official (2026 gazetted list; falls on a Sunday) | [Tamil Nadu govt issues 2026 public holiday list — The News Minute](https://www.thenewsminute.com/tamil-nadu/tamil-nadu-govt-issues-2026-public-holiday-list-with-24-declared-holidays) |
| 2026-12-25 | Christmas | holiday + festival | Official (national, fixed date every year) | [timeanddate.com — Holidays and Observances in India 2026](https://www.timeanddate.com/holidays/india/2026) |
| 2027-01-01 | New Year's Day | holiday | Official (fixed date every year) | [BankBazaar — Holidays 2026](https://www.bankbazaar.com/indian-holiday-calendar.html) |
| 2027-01-14 | Pongal | holiday + festival | **Projected** — TN's solar-calendar-fixed harvest festival; 2027's official state gazette wasn't published at generation time, so the date is projected from the confirmed 2026 date (Jan 15, 2026) and the festival's fixed solar-calendar pattern (Jan 14–15 every year) | [Tamil Nadu Holidays 2026 — BankBazaar](https://www.bankbazaar.com/indian-holiday/tamil-nadu-holidays.html) |
| 2027-01-15 | Thiruvalluvar Day | holiday + festival | Projected, same basis as Pongal above | [Tamil Nadu Public Holidays List 2026 — greytHR](https://www.greythr.com/wiki/acts/st-tamil-nadu/holidays/2026/) |
| 2027-01-16 | Uzhavar Thirunal | holiday | Projected, same basis as Pongal above | [Tamil Nadu Public Holidays List 2026 — greytHR](https://www.greythr.com/wiki/acts/st-tamil-nadu/holidays/2026/) |
| 2027-01-26 | Republic Day | holiday | Official (national, fixed date every year) | [timeanddate.com — Holidays and Observances in India 2026](https://www.timeanddate.com/holidays/india/2026) |

Additional general references consulted for cross-checking the 2026 list: [factoHR — Holiday List Tamil Nadu](https://factohr.com/holiday-list-in-india/tamil-nadu/), [greytHR — Tamil Nadu Public Holidays 2026](https://www.greythr.com/wiki/acts/st-tamil-nadu/holidays/2026/), [Sakshi Post — TN Government Holiday List 2026](https://www.sakshipost.com/news/national/tamil-nadu-government-holiday-list-2026-pongal-deepavali-christmas-more-476361).

`local_event` flags remain randomly generated (no public calendar exists for store-level local events — the brief treats these as store/city-specific, not a published dataset).

---

## 2. City Climate Normals (Temperature & Rainfall)

Used to draw realistic `temp_c` / `rain_mm` values in `external_factors.csv` around each city's real monthly average, instead of an arbitrary random walk — `CLIMATE_NORMALS` in [`src/generate_synthetic_data.py`](../src/generate_synthetic_data.py). Values are monthly averages; daily figures in the generated data are these averages plus Gaussian noise.

### Coimbatore

| Month | Avg Temp (°C) | Avg Rain (mm) | Status | Source |
|---|---|---|---|---|
| Aug | 24.5 | 104 | Directly published | [climate-data.org — Coimbatore](https://en.climate-data.org/asia/india/tamil-nadu/coimbatore-2788/) |
| Sep | 25.0 | 87 | Directly published | [climate-data.org — Coimbatore](https://en.climate-data.org/asia/india/tamil-nadu/coimbatore-2788/) |
| Oct | 24.8 | 181 | Directly published | [climate-data.org — Coimbatore](https://en.climate-data.org/asia/india/tamil-nadu/coimbatore-2788/) |
| Nov | 23.8 | 140 | Interpolated (Oct→Dec) | derived from [climate-data.org — Coimbatore](https://en.climate-data.org/asia/india/tamil-nadu/coimbatore-2788/) |
| Dec | 23.2 | 60 | Temp published; rainfall interpolated | [WeatherSpark — Coimbatore](https://weatherspark.com/y/108544/Average-Weather-in-Coimbatore-Tamil-Nadu-India-Year-Round) |
| Jan | 23.9 | 13 | Directly published | [climate-data.org — Coimbatore](https://en.climate-data.org/asia/india/tamil-nadu/coimbatore-2788/) |

### Chennai

| Month | Avg Temp (°C) | Avg Rain (mm) | Status | Source |
|---|---|---|---|---|
| Aug | 29.5 | 120 | Interpolated (Jul→Sep) | derived from [climate-data.org — Chennai](https://en.climate-data.org/asia/india/tamil-nadu/chennai-1003222/) |
| Sep | 28.7 | 110 | Directly published | [climate-data.org — Chennai](https://en.climate-data.org/asia/india/tamil-nadu/chennai-1003222/) |
| Oct | 27.3 | 223 | Directly published | [climate-data.org — Chennai](https://en.climate-data.org/asia/india/tamil-nadu/chennai-1003222/) |
| Nov | 25.7 | 228 | Directly published | [climate-data.org — Chennai](https://en.climate-data.org/asia/india/tamil-nadu/chennai-1003222/) |
| Dec | 24.6 | 113 | Directly published | [climate-data.org — Chennai](https://en.climate-data.org/asia/india/tamil-nadu/chennai-1003222/) |
| Jan | 24.0 | 25 | Temp published (timeanddate); rainfall interpolated | [timeanddate.com — Chennai Climate](https://www.timeanddate.com/weather/india/chennai/climate) |

### Madurai

| Month | Avg Temp (°C) | Avg Rain (mm) | Status | Source |
|---|---|---|---|---|
| Aug | 31.4 | 95 | Directly published | [climatestotravel.com — Madurai](https://www.climatestotravel.com/climate/india/madurai) |
| Sep | 29.3 | 84 | Directly published | [climate-data.org — Madurai](https://en.climate-data.org/asia/india/tamil-nadu/madurai-5892/) |
| Oct | 27.3 | 180 | Directly published | [climate-data.org — Madurai](https://en.climate-data.org/asia/india/tamil-nadu/madurai-5892/) |
| Nov | 25.4 | 168 | Directly published | [climate-data.org — Madurai](https://en.climate-data.org/asia/india/tamil-nadu/madurai-5892/) |
| Dec | 24.6 | 68 | Directly published | [climate-data.org — Madurai](https://en.climate-data.org/asia/india/tamil-nadu/madurai-5892/) |
| Jan | 26.7 | 20 | Temp published; rainfall interpolated | [climatestotravel.com — Madurai](https://www.climatestotravel.com/climate/india/madurai) |

### Salem

| Month | Avg Temp (°C) | Avg Rain (mm) | Status | Source |
|---|---|---|---|---|
| Aug | 27.5 | 140 | Derived from day/night averages + rain-day count | [WeatherSpark — Salem](https://weatherspark.com/y/109371/Average-Weather-in-Salem-Tamil-Nadu-India-Year-Round) |
| Sep | 26.8 | 120 | Derived from day/night averages + rain-day count | [WeatherSpark — Salem](https://weatherspark.com/y/109371/Average-Weather-in-Salem-Tamil-Nadu-India-Year-Round) |
| Oct | 25.8 | 124 | Rainfall directly published (wettest month, 4.9in); temp derived | [WeatherSpark — Salem](https://weatherspark.com/y/109371/Average-Weather-in-Salem-Tamil-Nadu-India-Year-Round) |
| Nov | 24.5 | 95 | Derived from day/night averages + rain-day count | [WeatherSpark — Salem](https://weatherspark.com/y/109371/Average-Weather-in-Salem-Tamil-Nadu-India-Year-Round) |
| Dec | 23.5 | 25 | Derived (dry season ends ~Dec 4, per source) | [WeatherSpark — Salem](https://weatherspark.com/y/109371/Average-Weather-in-Salem-Tamil-Nadu-India-Year-Round) |
| Jan | 24.0 | 5 | Derived from day/night averages, dry season | [WeatherSpark — Salem](https://weatherspark.com/y/109371/Average-Weather-in-Salem-Tamil-Nadu-India-Year-Round) |

**Note on Salem:** the primary public source (WeatherSpark) reports day/night extremes and rainy-day counts rather than a single mean temperature and mm-rainfall table the way climate-data.org does for the other three cities. We derived a mean temperature as the midpoint of day/night, and approximated monthly rainfall from the reported rainy-day counts and Salem's published annual total (1206mm/111 rainy days). This is disclosed here precisely so it can be defended if a judge asks — it is the one city where our figures are a derived approximation rather than a directly published monthly mean.

**Feb–Jul (outside the brief's Aug–Jan window):** the generator falls back to the nearest sourced month for these (see `_climate_for()` in the script) and these values are explicitly *not* independently verified — call this out if the team ever generates data outside the default window.

---

## 3. How This Should Be Used in the Pitch

Say this, not more: *"Our development/practice dataset — built before the official data was available — grounds its external factors in the real 2026 Tamil Nadu holiday calendar and published climate normals for these exact four cities, rather than random noise, so our feature engineering and model behavior were validated against realistic seasonal and calendar effects ahead of time."*

Do **not** say "we used real NovaMart data" or "we sourced the official dataset" — that is false and easily challenged. This is our own preparatory research, clearly disclosed, applied only to the practice dataset that gets replaced by the official one at the event (root [`README.md`](../README.md) §8).

## 4. Background Research Consulted (not directly encoded into data, informs methodology)

These didn't feed numbers into the generator, but shaped modelling and business-logic decisions in [`RESEARCH.md`](RESEARCH.md) — listed here for a single audit trail of "how we know what we know":

- [Fishbowl — Safety Stock Formula: 6 Variations](https://www.fishbowlinventory.com/blog/calculating-the-safety-stock-formula-6-variations-key-use-cases)
- [Lokad — Calculate Safety Stocks with Sales Forecasting](https://www.lokad.com/calculate-safety-stocks-with-sales-forecasting/)
- [Netstock — Reorder Point Formula](https://www.netstock.com/blog/reorder-point-formula/)
- [MachineLearningMastery — Backtest ML Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)
- [Medium — XGBoost and Imbalanced Datasets: Strategies](https://medium.com/@dicee/xgboost-and-imbalanced-datasets-strategies-for-handling-class-imbalance-cdd810b3905c)
- [MachineLearningMastery — Feature Importance and Selection with XGBoost](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)
- [arXiv 2505.16319 — FreshRetailNet-50K: Stockout-Annotated Censored Demand Dataset](https://arxiv.org/abs/2505.16319)
- [Medium — ABC-XYZ Inventory Classification with Python](https://medium.com/@ulas_yilmaz/abc-xyz-inventory-classification-with-python-50ebee552fe4)
- [arXiv 2007.03870 — A New Generalized Newsvendor Model with Random Demand](https://arxiv.org/pdf/2007.03870)
- [TAIKAI — Hackathon Judging: 6 Criteria to Pick Winning Projects](https://taikai.network/en/blog/hackathon-judging)

Full context for each is in [`RESEARCH.md`](RESEARCH.md).
