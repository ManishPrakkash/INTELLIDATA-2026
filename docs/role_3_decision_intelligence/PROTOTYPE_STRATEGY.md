# Prototype Differentiation Strategy — "NovaMart Ops Copilot"

**Problem this solves:** a six-tab Streamlit dashboard with bar charts and a DataFrame table is what ~90% of teams will submit. The brief asks for a "management-ready visualization/prototype" — that's a UX bar, not a chart-count bar. This document is the concrete, buildable plan for standing out on that specific axis, sized to fit inside your existing [hour 13–20 dashboard-build window](README.md#hour-1320--dashboard-build-round-3-core).

**The reframe:** stop building "a dashboard people explore." Build **an assistant that tells the manager what to do the moment they open it** — the six mandatory sections still exist (nothing here removes a rubric requirement), but they're delivered through a different interaction model.

---

## Research backing this (see full citations below)

- Modern inventory-ops tools are moving from "monitor a dashboard" to **exception-based, threshold-triggered alerts** — managers act on what's flagged, not what they went looking for. This directly matches the brief's own framing: *"tell store managers what action to take before the problem occurs."*
- A **single composite health score (0–100)** is a validated executive-communication pattern — it's what lets a judge understand your entire system's state in one glance, before they've read a single chart.
- A **rule-based (keyword/intent-matched) chatbot requires no ML or deep learning** — pure conditional logic over your existing `recommendations.csv`. This is fully compliant with the brief's "ML models only, from the permitted list" restriction, because it isn't one of the two mandatory modelling problems — it's a UI convenience layer sitting on top of outputs you already computed.
- Plotly Dash is more customizable/impressive than Streamlit for this kind of interactive, card-based layout, but costs meaningfully more build time. Given you're covering classification + explainability + recommendation + dashboard solo, **the recommendation below is: stay on Streamlit, but stop using its defaults** (custom CSS-styled HTML cards instead of `st.dataframe`/`st.table`, custom color system, custom layout) — this gets ~80% of the visual differentiation of switching frameworks for ~20% of the risk.

---

## The Plan — 4 Layers, Ordered by Effort/Impact

### Layer 1 (must-build, ~3–4 hrs): Exception-First Home Screen

Replace "Executive Summary tab with 5 KPI tiles" as the *landing view* with:

1. **NovaMart Health Score** — one number, 0–100, top-center, large font, color-coded (red <60, amber 60–80, green >80). Transparent formula (must be explainable — judges will ask):
   ```
   Health Score = 100 × [ 0.4×(1 − stockout_rate) + 0.35×(1 − estimated_lost_sales/revenue) + 0.25×inventory_turnover_normalized ]
   ```
   Show the three sub-components as small labeled bars under the score, not hidden — this *is* your Executive Summary requirement, just led by one number instead of a wall of tiles.
2. **"Today's Priorities"** — instead of a `st.dataframe` of every store×product row, render the top 5–10 **High-risk** rows as styled HTML/CSS cards (message-style: colored left border, product name bold, one-line "why," a clear reorder-quantity call-to-action), literally styled to resemble a phone notification/WhatsApp message. This is the single highest-leverage visual change you can make — a card that *looks like* something a manager would actually get paged with reads as "real product," a `st.dataframe` reads as "student project."
3. The remaining KPI tiles (Revenue, Units Sold, Inventory Value, Stock-out Rate) still appear, just **below** the health score and priority cards, not above them — exception-first ordering.

### Layer 2 (should-build, ~2 hrs): Rule-Based "Ask the Copilot" Box

A text input, e.g. `"Show me high-risk products in Chennai"` or `"What should I reorder this week for Store S01?"`. Implementation (no ML/DL, ~150 lines of plain Python):

1. Lowercase + tokenize the input.
2. Keyword-match against known entities already in your data: city names, store IDs, category names, risk tiers (`high`/`medium`/`low`), and a small set of intents (`show`/`list`/`reorder`/`why`/`compare`).
3. Build a pandas filter from whatever matched (`recommendations_df.query(...)` style), fall back to "I didn't catch that — try mentioning a store, city, category, or risk level" if nothing matched (a graceful, honest failure mode beats a fake-confident wrong answer).
4. Render the filtered result using the same card component as Layer 1.
5. **Bonus explainability moment:** show a small caption under the answer — `Matched: risk_tier=High, city=Chennai` — this doubles as a transparent, inspectable "here's exactly why you got these results" note, which is free credibility for the Explainability rubric line.

This is the single most "wait, that's not just a dashboard" moment in a 3-minute judge demo, and it costs about as much time as one well-built dashboard tab.

### Layer 3 (should-build, ~1 hr): "Present Mode"

A toggle/button that auto-sequences through: Health Score → biggest single risk (with its full recommendation card and "why") → live What-If slider move that changes the recommendation in front of the judges → the ₹ Estimated Lost Sales figure as the closing number. Judges spend minutes per team — a guided 90-second path through your strongest evidence beats hoping they click the right tabs themselves. Implement as literally just a `st.session_state` step counter with "Next" advancing between pre-selected views — this is cheap.

### Layer 4 (nice-to-have, ~30 min): Phone-Frame Preview Toggle

A "📱 View as store manager" toggle that renders the Priority Cards panel inside a narrow (375px), phone-shaped CSS container. Trivial to build (a `max-width` div + rounded corners + a fake status-bar strip), but it signals you understood *who actually uses this* — a store manager on the floor, not an analyst at a desk — which is exactly the kind of user-empathy detail the research says separates a "good project" from a "winning" one.

---

## Proof This Combination Is Buildable in a Hackathon Timeframe

Worth knowing before you commit time to this: a comparable hackathon submission — **Cognizant Technoverse 2026, "Demand Forecasting & Inventory Optimization," Team 38 (GITAM University)** — combined demand forecasting, stock-out-risk analysis, replenishment recommendations, a What-If simulator, **and a rule-based "AI Agent"** in a single Streamlit dashboard. That's effectively Layers 1–3 of this plan, built and submitted successfully at hackathon scope. One difference worth noting for time-budgeting: that team had **four people**, with one person dedicated solely to dashboard/UX. You're doing this solo alongside the classification model and explainability work — which is exactly why the Layer ordering below (must/should/nice-to-have) matters: build Layer 1 fully before touching Layer 2, and don't start Layer 2 unless Layer 1 is done with time to spare.

Also validated independently: **SayanGhorai/smart-retail-supply-chain** on GitHub implements ABC-XYZ segmentation feeding forecast-driven inventory decisions — the same core idea as [`RESEARCH.md`](../RESEARCH.md) §4, confirming it's a recognized real pattern, not an invented one.

## Implementation Reference (concrete, copy-adaptable patterns)

### Layer 1 — HTML/CSS alert cards in Streamlit

Streamlit's own components (`st.dataframe`, `st.metric`) can't produce a notification-style card — you need `st.markdown(..., unsafe_allow_html=True)` with your own CSS. Minimal pattern:

```python
def render_alert_card(store, product, risk_tier, reorder_qty, reason, forecast, stockout_prob):
    color = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}[risk_tier]
    st.markdown(f"""
    <div style="border-left: 6px solid {color}; background: #1e1e1e; border-radius: 8px;
                padding: 14px 18px; margin-bottom: 10px;">
      <div style="font-weight: 700; font-size: 1.05em;">{store} — {product}</div>
      <div style="color: {color}; font-weight: 600;">{risk_tier} RISK · {stockout_prob:.0%} stock-out probability</div>
      <div style="margin-top: 6px;">Forecast: <b>{forecast} units</b> · Reorder: <b>{reorder_qty} units</b></div>
      <div style="margin-top: 6px; opacity: 0.85;">Why? {reason}</div>
    </div>
    """, unsafe_allow_html=True)
```
Call this once per high-risk row instead of `st.dataframe(recommendations_df)`. Source pattern: [Streamlit docs discussion — unsafe_allow_html usage](https://discuss.streamlit.io/t/are-you-using-html-in-markdown-tell-us-why/96), [DEV Community — Styling Streamlit Metrics in Custom CSS](https://dev.to/barrisam/how-to-style-streamlit-metrics-in-custom-css-4h14).

### Layer 2 — Rule-based intent parser ("Ask the Copilot")

This is exactly the deterministic-parser pattern used by Snips NLU (an open-source rule-based NLU engine that guarantees correct matches on known patterns, unlike a statistical model): regex/keyword rules map straight to a known set of intents and entities, no ML involved.

```python
import re

CITIES = ["chennai", "coimbatore", "madurai", "salem"]
TIERS = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}

def parse_query(text: str) -> dict:
    text = text.lower()
    filters = {}
    for city in CITIES:
        if city in text:
            filters["city"] = city.title()
    for tier_word, tier_val in TIERS.items():
        if tier_word in text:
            filters["risk_tier"] = tier_val
    store_match = re.search(r"\bs0[1-4]\b", text)
    if store_match:
        filters["store_id"] = store_match.group(0).upper()
    intent = "reorder" if "reorder" in text else ("why" if "why" in text else "show")
    return {"intent": intent, "filters": filters}
```
Apply `filters` as a pandas `.query()`/boolean-mask against `recommendations.csv`, render results with the Layer 1 card function, and show the matched `filters` dict as the transparency caption. Source: [Snips NLU Tutorial — deterministic/regex parser](https://snips-nlu.readthedocs.io/en/latest/tutorial.html), [DataCamp — Intent classification with regex](https://campus.datacamp.com/courses/building-chatbots-in-python/understanding-natural-language?ex=2).

### Layer 3 — Present Mode via `st.session_state`

```python
if "present_step" not in st.session_state:
    st.session_state.present_step = 0
steps = ["health_score", "top_risk", "whatif_live", "impact_number"]

col1, col2 = st.columns(2)
if col1.button("◀ Back") and st.session_state.present_step > 0:
    st.session_state.present_step -= 1
if col2.button("Next ▶") and st.session_state.present_step < len(steps) - 1:
    st.session_state.present_step += 1

render_step(steps[st.session_state.present_step])   # dispatch to the matching view function
```
Source pattern: [Streamlit Docs — Add statefulness to apps (session_state)](https://docs.streamlit.io/develop/concepts/architecture/session-state), [archydeberker/streamlit-wizard on GitHub](https://github.com/archydeberker/streamlit-wizard).

### Layer 4 — Phone-frame CSS container

No JS/dependency needed — a border + border-radius + fixed width does it:

```python
st.markdown("""
<style>
.phone-frame { max-width: 380px; margin: 0 auto; border: 10px solid #111;
               border-radius: 36px; padding: 10px; background: #000; }
.phone-frame .screen { background: #fff; border-radius: 24px; padding: 12px;
                        max-height: 700px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)
# then wrap the priority-cards render inside: st.markdown('<div class="phone-frame"><div class="screen">', unsafe_allow_html=True) ... st.markdown('</div></div>', unsafe_allow_html=True)
```
Source: [CodePen — Simple device mockups in CSS (trevoreyre)](https://codepen.io/trevoreyre/pen/dvNwqG), [freefrontend — 20+ Pure CSS iPhone Examples](https://freefrontend.com/iphones-in-css/).

### On the Health Score formula — an honest caveat

There is no single standardized formula for a composite retail health score — even the Balanced Scorecard literature confirms weighting is a design choice assembled from business priorities, not a calculated constant. **State your weights as a stated assumption in the dashboard itself** (a small "ℹ️ Formula" expander showing the exact weighted formula) rather than presenting the 0–100 number as if it were an industry-standard metric — this is more defensible under judge questioning than pretending it's not a design choice. Source: [BSC Designer — Supply Chain Strategy Scorecard with KPIs](https://bscdesigner.com/supply-chain.htm), [Profit.co — Balanced Scorecard in Retail](https://www.profit.co/blog/strategy/balanced-scorecard-for-the-retail-industry/).

## What NOT to Do (given your remaining time budget)

- **Don't rewrite in Plotly Dash or React.** More customizable, yes — but you're solo on classification + explainability + recommendation engine + this dashboard, and a framework switch this late risks not finishing. Get the differentiation from Layers 1–3 in Streamlit with custom CSS, not from a framework swap.
- **Don't build a real LLM/chatbot.** Layer 2 must stay rule-based both because it's fast and because it keeps you unambiguously inside the brief's "ML only, no deep learning" constraint — a real NLU model would also invite exactly the "is this in scope" question you don't want from a judge.
- **Don't drop any of the six mandatory dashboard sections** (Executive Summary, Demand Intelligence, Inventory Risk, Manager Action Centre, Model Performance, Explainability) — they still all need to exist; this plan changes *how they're presented and ordered*, not whether they're present. Keep them as secondary tabs/scroll sections beneath the exception-first home view.

## Where This Plugs Into Your Existing Plan

No change to your hours 0–13 (dashboard shell pre-build, classification model, explainability). This replaces the generic "six mandatory sections" build described in your [main README's Hour 13–20 block](README.md#hour-1320--dashboard-build-round-3-core) with the four layers above — same six sections, same six hours, different execution.

---

## Sources

**Design philosophy:**
- [dev.to — Designing a Zero-Friction AI Dashboard (exception-based alerting over constant monitoring)](https://dev.to/harshibhupathirajucreator/designing-a-zero-friction-ai-dashboard-for-3-am-pagerduty-alerts-5daa)
- [aglowiditsolutions.com — Mobile Apps for Retail Management (exception-based alerts, single-handed mobile UX for store staff)](https://aglowiditsolutions.com/blog/mobile-apps-for-retail-management/)
- [SimpleKPI — Customizing KPI Dashboards: Storytelling & the single 0–100 health score pattern](https://www.simplekpi.com/Blog/the-art-of-customizing-kpi-dashboards)
- [FanRuan — Executive Summary Dashboard: 9 Steps to Design KPIs](https://www.fanruan.com/en/blog/executive-summary-dashboard)
- [BSC Designer — Supply Chain Strategy Scorecard with KPIs](https://bscdesigner.com/supply-chain.htm)
- [Profit.co — How the Balanced Scorecard Works in Retail](https://www.profit.co/blog/strategy/balanced-scorecard-for-the-retail-industry/)

**Rule-based "Ask the Copilot" (Layer 2):**
- [Elshad Karimov — Build a Simple Rule-Based Chatbot in Python (no AI libraries)](https://elshad-karimov.medium.com/build-a-simple-rule-based-chatbot-in-python-without-ai-libraries-3e6848cc6017)
- [HeroThemes — Rule-Based Chatbots: A Beginner's Guide](https://herothemes.com/blog/rule-based-chatbots/)
- [Snips NLU Tutorial — deterministic/regex intent parser](https://snips-nlu.readthedocs.io/en/latest/tutorial.html)
- [DataCamp — Intent classification with regex](https://campus.datacamp.com/courses/building-chatbots-in-python/understanding-natural-language?ex=2)

**Streamlit implementation patterns:**
- [Streamlit Discuss — Are you using HTML in Markdown? (unsafe_allow_html patterns)](https://discuss.streamlit.io/t/are-you-using-html-in-markdown-tell-us-why/96)
- [DEV Community — How to Style Streamlit Metrics in Custom CSS](https://dev.to/barrisam/how-to-style-streamlit-metrics-in-custom-css-4h14)
- [Streamlit Docs — Add statefulness to apps (session_state)](https://docs.streamlit.io/develop/concepts/architecture/session-state)
- [archydeberker/streamlit-wizard — GitHub](https://github.com/archydeberker/streamlit-wizard)
- [Analytics India Magazine — Streamlit vs Plotly Dash Comparison](https://analyticsindiamag.com/ai-trends/streamlit-vs-plotlydash-comparison-with-python-examples)
- [Plotly — Best Streamlit Alternatives for Production-Grade Data Apps](https://plotly.com/blog/best-streamlit-alternatives-production-data-apps/)
- [Medium — 10 Best Streamlit Layout & Design Tips](https://medium.com/data-science-collective/wait-this-was-built-in-streamlit-10-best-streamlit-design-tips-for-dashboards-2b0f50067622)

**Phone-frame mockup (Layer 4):**
- [CodePen — Simple device mockups in CSS (trevoreyre)](https://codepen.io/trevoreyre/pen/dvNwqG)
- [freefrontend — 20+ Pure CSS iPhone Examples](https://freefrontend.com/iphones-in-css/)

**Proof of feasibility — comparable hackathon submissions:**
- Cognizant Technoverse 2026, "Demand Forecasting & Inventory Optimization" — Team 38, GITAM University (demand forecast + stock-out risk + replenishment + What-If + rule-based AI Agent + Streamlit, built by a 4-person team)
- [SayanGhorai/smart-retail-supply-chain — GitHub (independent ABC-XYZ segmentation + forecast-driven inventory decisions)](https://github.com/SayanGhorai/smart-retail-supply-chain)
