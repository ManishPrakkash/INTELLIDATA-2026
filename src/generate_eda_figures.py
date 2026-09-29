"""
src/generate_eda_figures.py - Generates the 6 mandatory EDA figures & ABC-XYZ segmentation
Owned by Student 1 (Data Analyst) — StockSense / IntelliData 2026
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/processed/master_table.csv")
df["date"] = pd.to_datetime(df["date"])

print("Generating 6 Business Question Visualizations...")

# -------------------------------------------------------------
# Chart 1: Category Revenue Share + Pareto (Cumulative % Line)
# -------------------------------------------------------------
print("[1/6] Chart 1: Category Revenue & Pareto Analysis...")
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
cat_rev["revenue_lakhs"] = cat_rev["revenue"] / 100000
cat_rev["cum_pct"] = (cat_rev["revenue"].cumsum() / cat_rev["revenue"].sum()) * 100

fig, ax1 = plt.subplots(figsize=(10, 5.5))
colors = sns.color_palette("mako", len(cat_rev))
bars = ax1.bar(cat_rev["category"], cat_rev["revenue_lakhs"], color=colors, edgecolor="black", alpha=0.85, width=0.55)
ax1.set_ylabel("Total Revenue (₹ Lakhs)", color="#1b4965")
ax1.tick_params(axis="y", labelcolor="#1b4965")
ax1.set_xlabel("Merchandise Category")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Value labels on bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"₹{yval:.1f}L", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Secondary axis for Pareto line
ax2 = ax1.twinx()
ax2.plot(cat_rev["category"], cat_rev["cum_pct"], color="#d62828", marker="o", linewidth=2.5, markersize=8)
ax2.set_ylabel("Cumulative Revenue Contribution (%)", color="#d62828")
ax2.tick_params(axis="y", labelcolor="#d62828")
ax2.set_ylim(0, 110)
ax2.axhline(80, color="#d62828", linestyle=":", alpha=0.8, label="80% Pareto Cutoff")

# Annotation for 80% cutoff
for idx, row in cat_rev.iterrows():
    ax2.annotate(f"{row['cum_pct']:.1f}%", (row["category"], row["cum_pct"] + 3), ha="center", fontsize=9, fontweight="bold", color="#d62828")

plt.title("NovaMart Category Revenue Share & Pareto Analysis (ABC Driver)", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_category_pareto.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# Chart 2: Store Growth / Revenue Trend Over Time
# -------------------------------------------------------------
print("[2/6] Chart 2: Store Revenue Trends (Weekly Aggregation)...")
df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
store_trends = df.groupby(["week", "store_id", "store_type"])["revenue"].sum().reset_index()

fig, ax = plt.subplots(figsize=(11, 5.5))
palette = {"S01": "#2a9d8f", "S02": "#e76f51", "S03": "#f4a261", "S04": "#264653"}
labels = {
    "S01": "S01 (Coimbatore - Supermarket)",
    "S02": "S02 (Chennai - Hypermarket)",
    "S03": "S03 (Madurai - Express)",
    "S04": "S04 (Salem - Supermarket)"
}

for sid, grp in store_trends.groupby("store_id"):
    ax.plot(grp["week"], grp["revenue"] / 1000, label=labels.get(sid, sid), color=palette.get(sid), linewidth=2.2, marker="s", markersize=4)

ax.set_title("Weekly Store Revenue Trajectory (Aug 2026 – Jan 2027)", pad=15)
ax.set_ylabel("Weekly Net Revenue (₹ Thousands)")
ax.set_xlabel("Calendar Week")
ax.legend(frameon=True, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_store_trends.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# Chart 3: Promotion Lift on Demand
# -------------------------------------------------------------
print("[3/6] Chart 3: Promotion Lift on Units Sold by Category...")
promo_cat = df.groupby(["category", "promotion_flag"])["quantity_sold"].mean().unstack().reset_index()
promo_cat.columns = ["category", "No_Promo", "Promo_Active"]
promo_cat["Lift_Pct"] = ((promo_cat["Promo_Active"] - promo_cat["No_Promo"]) / promo_cat["No_Promo"]) * 100

x = np.arange(len(promo_cat))
width = 0.35

fig, ax = plt.subplots(figsize=(10.5, 5.5))
rects1 = ax.bar(x - width/2, promo_cat["No_Promo"], width, label="Standard Days (Promo=0)", color="#6c757d", alpha=0.85)
rects2 = ax.bar(x + width/2, promo_cat["Promo_Active"], width, label="Promotion Days (Promo=1)", color="#0077b6", alpha=0.9)

ax.set_ylabel("Average Daily Units Sold / SKU")
ax.set_title("Impact of Active Promotional Campaigns on Demand Volume by Category", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(promo_cat["category"])
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

# Add Lift % annotations
for i, row in promo_cat.iterrows():
    top = max(row["No_Promo"], row["Promo_Active"])
    ax.annotate(f"+{row['Lift_Pct']:.1f}%", xy=(x[i] + width/2, row["Promo_Active"] + 0.15),
                ha="center", fontsize=9.5, fontweight="bold", color="#0077b6")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_promotion_lift.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# Chart 4: Weekday vs. Weekend Demand Patterns
# -------------------------------------------------------------
print("[4/6] Chart 4: Weekend vs Weekday Demand Patterns...")
weekend_cat = df.groupby(["category", "weekend"])["quantity_sold"].mean().unstack().reset_index()
weekend_cat.columns = ["category", "Weekday", "Weekend"]
weekend_cat["Weekend_Lift"] = ((weekend_cat["Weekend"] - weekend_cat["Weekday"]) / weekend_cat["Weekday"]) * 100

fig, ax = plt.subplots(figsize=(10.5, 5.5))
rects1 = ax.bar(x - width/2, weekend_cat["Weekday"], width, label="Monday – Friday", color="#4a5568", alpha=0.85)
rects2 = ax.bar(x + width/2, weekend_cat["Weekend"], width, label="Saturday & Sunday", color="#38b000", alpha=0.9)

ax.set_ylabel("Average Daily Units Sold / SKU")
ax.set_title("Customer Shopping Volume: Weekday vs Weekend Spikes Across Categories", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(weekend_cat["category"])
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

for i, row in weekend_cat.iterrows():
    ax.annotate(f"+{row['Weekend_Lift']:.1f}%", xy=(x[i] + width/2, row["Weekend"] + 0.15),
                ha="center", fontsize=9.5, fontweight="bold", color="#2d6a4f")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_weekend_effect.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# Chart 5: Product Demand Volatility (CV Ranking / XYZ Input)
# -------------------------------------------------------------
print("[5/6] Chart 5: Product Volatility & XYZ Segmentation...")
prod_metrics = df.groupby(["product_id", "category"]).agg(
    total_rev=("revenue", "sum"),
    mean_demand=("corrected_demand", "mean"),
    std_demand=("corrected_demand", "std")
).reset_index()

prod_metrics["cv_demand"] = (prod_metrics["std_demand"] / prod_metrics["mean_demand"]).round(3)
prod_metrics = prod_metrics.sort_values("cv_demand", ascending=False).reset_index(drop=True)

# Select top 15 most volatile and bottom 10 most stable
top_volatile = prod_metrics.head(15).copy()
least_volatile = prod_metrics.tail(10).copy()
sample_chart = pd.concat([top_volatile, least_volatile]).sort_values("cv_demand", ascending=True)

fig, ax = plt.subplots(figsize=(11, 7.5))
colors = ["#2b9348" if cv < 0.50 else "#e9c46a" if cv < 1.00 else "#d90429" for cv in sample_chart["cv_demand"]]
bars = ax.barh(sample_chart["product_id"] + " (" + sample_chart["category"] + ")", sample_chart["cv_demand"], color=colors, alpha=0.85)

ax.axvline(0.50, color="#2b9348", linestyle="--", linewidth=1.5, label="X Class Threshold (CV < 0.50)")
ax.axvline(1.00, color="#d90429", linestyle="--", linewidth=1.5, label="Z Class Threshold (CV >= 1.00)")

ax.set_xlabel("Coefficient of Variation (CV = Std Dev / Mean Demand)")
ax.set_title("Product Demand Volatility Spectrum (XYZ Classification Driver)", pad=15)
ax.legend(loc="lower right", frameon=True)
ax.grid(axis="x", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_product_volatility.png"), dpi=300)
plt.close()


# -------------------------------------------------------------
# Chart 6: Store × Category Stock-Out Frequency Heatmap
# -------------------------------------------------------------
print("[6/6] Chart 6: Store x Category Stock-Out Rate Heatmap...")
heatmap_data = df.groupby(["store_id", "category"])["stockout_flag"].mean().unstack() * 100
heatmap_data.index = ["S01 (Coimbatore)", "S02 (Chennai)", "S03 (Madurai)", "S04 (Salem)"]

fig, ax = plt.subplots(figsize=(9.5, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Stock-Out Occurrence Rate (%)'},
            linewidths=1, linecolor="white", annot_kws={"size": 11, "weight": "bold"}, ax=ax)
ax.set_title("NovaMart Network Vulnerability: Stock-Out Rate by Store × Category (%)", pad=15)
ax.set_ylabel("Retail Store Location")
ax.set_xlabel("Merchandise Category")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_stockout_heatmap.png"), dpi=300)
plt.close()

# -------------------------------------------------------------
# Export Full ABC-XYZ Analysis Table for Student 2 & Student 3
# -------------------------------------------------------------
print("Exporting ABC-XYZ Analysis Table to data/processed/abc_xyz_analysis.csv...")
prod_metrics["rev_share"] = (prod_metrics["total_rev"] / prod_metrics["total_rev"].sum()).round(4)
prod_metrics = prod_metrics.sort_values("total_rev", ascending=False).reset_index(drop=True)
prod_metrics["cum_rev_share"] = prod_metrics["rev_share"].cumsum().round(4)

# ABC: A <= 70%, B <= 90%, C > 90%
prod_metrics["abc_class"] = np.where(prod_metrics["cum_rev_share"] <= 0.70, "A",
                            np.where(prod_metrics["cum_rev_share"] <= 0.90, "B", "C"))

# XYZ: X < 0.50, Y < 1.00, Z >= 1.00
prod_metrics["xyz_class"] = np.where(prod_metrics["cv_demand"] < 0.50, "X",
                            np.where(prod_metrics["cv_demand"] < 1.00, "Y", "Z"))
prod_metrics["abc_xyz"] = prod_metrics["abc_class"] + prod_metrics["xyz_class"]

out_abc_xyz = "data/processed/abc_xyz_analysis.csv"
prod_metrics.to_csv(out_abc_xyz, index=False)
print(f"Export complete: {out_abc_xyz} ({len(prod_metrics)} products classified)")
print("All 6 figures generated successfully!")
