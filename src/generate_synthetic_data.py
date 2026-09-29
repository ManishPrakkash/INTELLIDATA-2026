"""
Synthetic data generator for StockSense (IntelliData 2026).

Produces transactions.csv, products.csv, stores.csv, inventory.csv and
external_factors.csv that match the official schema exactly, with the same
family of intentional data-quality traps described in the brief (missing
values, duplicate rows, category-casing inconsistency, impossible
quantities, inventory-arithmetic mismatches, sparse-history new products).

Purpose: let all three teammates build and test the full pipeline
(cleaning -> EDA -> features -> models -> dashboard) before the official
dataset is released, so no hour of the 24 is spent idle.

Usage:
    python src/generate_synthetic_data.py --days 120 --stores 4 --products 40 --out data/raw/
"""
import argparse
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

CATEGORIES = {
    "Dairy": (["Milk", "Curd", "Cheese", "Butter"], 3, 15),
    "Beverages": (["Soft Drink", "Juice", "Water"], 30, 240),
    "Snacks": (["Biscuits", "Chips", "Namkeen"], 60, 270),
    "Personal Care": (["Shampoo", "Soap", "Toothpaste"], 365, 900),
    "Household": (["Detergent", "Cleaner"], 365, 900),
    "Frozen": (["Ice Cream", "Frozen Veg"], 60, 180),
}
BRANDS = ["Aavin", "GlowCare", "FizzUp", "Crispo", "HomeShine", "PureLeaf", "SnapFresh"]
CITIES = {
    "S01": ("Coimbatore", "Supermarket", "West"),
    "S02": ("Chennai", "Hypermarket", "North"),
    "S03": ("Madurai", "Express", "South"),
    "S04": ("Salem", "Supermarket", "Central"),
}
PAYMENT_MODES = ["UPI", "Card", "Cash"]


def make_category_variant(cat: str) -> str:
    """Randomly mangle category casing/spacing to simulate the inconsistency trap."""
    variants = [cat, cat.upper(), cat.lower(), f" {cat}", cat.replace(" ", "")]
    return random.choice(variants)


def gen_products(n_products: int, rng: random.Random) -> pd.DataFrame:
    rows = []
    cat_names = list(CATEGORIES.keys())
    for i in range(1, n_products + 1):
        pid = f"P{100 + i}"
        cat = rng.choice(cat_names)
        subs, life_min, life_max = CATEGORIES[cat]
        sub = rng.choice(subs)
        cost = round(rng.uniform(15, 90), 2)
        mrp = round(cost * rng.uniform(1.15, 1.9), 2)
        rows.append(
            {
                "product_id": pid,
                "category": make_category_variant(cat),
                "sub_category": sub,
                "brand": rng.choice(BRANDS),
                "mrp": mrp,
                "cost_price": cost,
                "shelf_life_days": rng.randint(life_min, life_max),
                "supplier_id": f"SUP{rng.randint(1, 10):02d}",
            }
        )
    return pd.DataFrame(rows)


def gen_stores() -> pd.DataFrame:
    rows = []
    sizes = {"Supermarket": (7000, 8600, 1000, 1300), "Hypermarket": (16000, 19000, 2500, 3000),
             "Express": (4000, 4500, 650, 800)}
    for sid, (city, stype, region) in CITIES.items():
        floor_lo, floor_hi, cust_lo, cust_hi = sizes[stype]
        rows.append(
            {
                "store_id": sid,
                "city": city,
                "store_type": stype,
                "floor_area_sqft": random.randint(floor_lo, floor_hi),
                "avg_daily_customers": random.randint(cust_lo, cust_hi),
                "region": region,
            }
        )
    return pd.DataFrame(rows)


def gen_external_factors(dates, cities, rng: random.Random) -> pd.DataFrame:
    rows = []
    festival_days = set(rng.sample(range(len(dates)), max(1, len(dates) // 30)))
    local_event_days = set(rng.sample(range(len(dates)), max(1, len(dates) // 20)))
    for city in cities:
        base_temp = rng.uniform(26, 34)
        for i, d in enumerate(dates):
            weekend = 1 if d.weekday() >= 5 else 0
            temp = round(base_temp + rng.uniform(-3, 3) + 2 * np.sin(i / 15), 1)
            rain = round(max(0, rng.gauss(1.5, 3)), 1) if rng.random() < 0.3 else 0.0
            holiday = 1 if (i in festival_days and rng.random() < 0.5) else 0
            festival = 1 if i in festival_days else 0
            local_event = 1 if i in local_event_days else 0
            # missing-value trap: ~4% of temperature readings blank
            temp_val = np.nan if rng.random() < 0.04 else temp
            rows.append(
                {
                    "date": d.isoformat(),
                    "city": city,
                    "temp_c": temp_val,
                    "rain_mm": rain,
                    "holiday": holiday,
                    "festival": festival,
                    "weekend": weekend,
                    "local_event": local_event,
                }
            )
    return pd.DataFrame(rows)


def gen_transactions_and_inventory(products: pd.DataFrame, dates, rng: random.Random,
                                    new_product_cutoff_days: int):
    """Simulate daily demand per store-product with trend/seasonality/promo/festival effects,
    derive transactions, then derive inventory with occasional stock-outs and injected
    arithmetic mismatches / duplicate transaction ids / impossible quantities."""
    tx_rows = []
    inv_rows = []
    tx_counter = 1000
    n_days = len(dates)

    # a handful of products are "new" (sparse history trap): only appear in the last few days
    sparse_products = set(rng.sample(list(products["product_id"]), max(1, len(products) // 10)))

    for store_id, (city, store_type, region) in CITIES.items():
        for _, prod in products.iterrows():
            pid = prod["product_id"]
            base_demand = rng.uniform(5, 40) * (1.6 if store_type == "Hypermarket" else 1.0)
            trend = rng.uniform(-0.01, 0.02)
            opening = rng.randint(80, 400)
            reorder_lvl = int(base_demand * rng.uniform(4, 8))
            lead_days = rng.choice([1, 1, 2, 3, 4])

            start_idx = 0
            if pid in sparse_products:
                start_idx = n_days - rng.randint(3, 7)  # only last few days have data

            for i in range(start_idx, n_days):
                d = dates[i]
                weekend = d.weekday() >= 5
                promo = 1 if rng.random() < 0.12 else 0
                discount = round(rng.uniform(5, 25), 1) if promo else 0.0
                seasonal = 1 + 0.25 * np.sin(i / 7) + trend * i
                weekend_mult = 1.3 if weekend else 1.0
                promo_mult = 1.35 if promo else 1.0
                demand = max(0, rng.gauss(base_demand * seasonal * weekend_mult * promo_mult, base_demand * 0.25))
                demand = int(round(demand))

                sellable = min(demand, opening)
                stockout = sellable < demand and opening <= 0
                mrp = prod["mrp"]
                price = round(mrp * (1 - discount / 100), 2)

                # generate 0-3 transactions covering the day's sellable units
                units_left = sellable
                n_tx = rng.randint(1, 3) if units_left > 0 else 0
                for _ in range(n_tx):
                    if units_left <= 0:
                        break
                    qty = rng.randint(1, max(1, min(5, units_left)))
                    units_left -= qty
                    tx_counter += 1
                    row = {
                        "transaction_id": f"T{tx_counter}",
                        "date": d.isoformat(),
                        "store_id": store_id,
                        "product_id": pid,
                        "quantity": qty,
                        "selling_price": price,
                        "discount_pct": discount,
                        "promotion_flag": promo,
                        "customer_id": f"C{rng.randint(1, 5000):05d}",
                        "payment_mode": rng.choice(PAYMENT_MODES),
                        "hour": rng.randint(8, 21),
                    }
                    tx_rows.append(row)
                    # duplicate-row trap: ~1% chance the same transaction is double-logged
                    if rng.random() < 0.01:
                        tx_rows.append(dict(row))
                    # impossible-quantity trap: ~0.5% chance of a negative quantity row
                    if rng.random() < 0.005:
                        bad = dict(row)
                        tx_counter += 1
                        bad["transaction_id"] = f"T{tx_counter}"
                        bad["quantity"] = -abs(qty)
                        tx_rows.append(bad)

                received = 0
                if opening <= reorder_lvl or rng.random() < 0.1:
                    received = int(base_demand * rng.uniform(3, 6))
                closing = max(0, opening + received - sellable)

                # inventory-mismatch trap: ~3% of rows get a corrupted closing value
                closing_reported = closing
                if rng.random() < 0.03:
                    closing_reported = closing + rng.choice([-5, -3, 3, 5, 10])

                inv_rows.append(
                    {
                        "date": d.isoformat(),
                        "store": store_id,
                        "product": pid,
                        "opening": opening,
                        "received": received,
                        "sold": sellable,
                        "closing": max(0, closing_reported),
                        "reorder_lvl": reorder_lvl,
                        "lead_days": lead_days,
                    }
                )
                opening = closing

    return pd.DataFrame(tx_rows), pd.DataFrame(inv_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=120, help="number of days of history to simulate")
    ap.add_argument("--stores", type=int, default=4, help="ignored; stores fixed to match brief sample (S01-S04)")
    ap.add_argument("--products", type=int, default=40, help="number of synthetic products")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/raw/", help="output directory")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    start = date(2026, 8, 1)
    dates = [start + timedelta(days=i) for i in range(args.days)]

    products = gen_products(args.products, rng)
    stores = gen_stores()
    cities = sorted({v[0] for v in CITIES.values()})
    external = gen_external_factors(dates, cities, rng)
    transactions, inventory = gen_transactions_and_inventory(products, dates, rng, new_product_cutoff_days=7)

    products.to_csv(os.path.join(args.out, "products.csv"), index=False)
    stores.to_csv(os.path.join(args.out, "stores.csv"), index=False)
    external.to_csv(os.path.join(args.out, "external_factors.csv"), index=False)
    transactions.to_csv(os.path.join(args.out, "transactions.csv"), index=False)
    inventory.to_csv(os.path.join(args.out, "inventory.csv"), index=False)

    print(f"Synthetic data written to {args.out}")
    print(f"  products.csv          : {len(products)} rows")
    print(f"  stores.csv             : {len(stores)} rows")
    print(f"  external_factors.csv   : {len(external)} rows")
    print(f"  transactions.csv       : {len(transactions)} rows")
    print(f"  inventory.csv          : {len(inventory)} rows")
    print("Note: this data intentionally contains the same quality traps as the official "
          "dataset (missing temps, duplicate transaction rows, negative quantities, "
          "inventory-arithmetic mismatches, sparse new-product history). Do not expect it "
          "to be clean -- that is the point.")


if __name__ == "__main__":
    main()
