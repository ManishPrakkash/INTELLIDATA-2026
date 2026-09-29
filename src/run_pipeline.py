"""
StockSense — End-to-End Reproducible Pipeline
===============================================
One command to go from raw CSVs → clean master table → features → 
trained models → recommendation outputs.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --skip-cleaning   # if master_table.csv already exists
    python src/run_pipeline.py --skip-features    # if feature_table.csv already exists

This is a mandatory deliverable: the entire repo must be reproducible
with a single command.
"""

import os
import sys
import time
import argparse
import pandas as pd

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

MASTER_TABLE_PATH = os.path.join(DATA_PROCESSED, "master_table.csv")
FEATURE_TABLE_PATH = os.path.join(DATA_PROCESSED, "feature_table.csv")
DEMAND_MODEL_PATH = os.path.join(MODELS_DIR, "demand_model.pkl")

# Add src/ to path so we can import our modules
sys.path.insert(0, SRC_DIR)


def print_banner(step_num, total, title):
    """Print a clear step banner for pipeline progress."""
    print()
    print("=" * 60)
    print(f"  STEP {step_num}/{total}: {title}")
    print("=" * 60)


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# Step 1: Data Cleaning → master_table.csv
# ──────────────────────────────────────────────
def step_cleaning(skip=False):
    """
    Run Student 1's cleaning pipeline to produce master_table.csv.
    If cleaning.py doesn't exist yet, check if master_table.csv is already there.
    """
    print_banner(1, 4, "DATA CLEANING → master_table.csv")

    if skip:
        print("  [SKIP] --skip-cleaning flag set")
        if os.path.exists(MASTER_TABLE_PATH):
            df = pd.read_csv(MASTER_TABLE_PATH)
            print(f"  Using existing master_table.csv ({len(df):,} rows)")
            return True
        else:
            print("  [ERROR] master_table.csv not found and cleaning was skipped!")
            return False

    cleaning_script = os.path.join(SRC_DIR, "cleaning.py")
    if os.path.exists(cleaning_script):
        print("  Running src/cleaning.py...")
        try:
            import cleaning
            if hasattr(cleaning, "main"):
                cleaning.main()
            print("  ✓ Cleaning complete")
            return True
        except Exception as e:
            print(f"  [ERROR] Cleaning failed: {e}")
            return False
    else:
        # Student 1 hasn't created cleaning.py yet — check if CSV exists
        if os.path.exists(MASTER_TABLE_PATH):
            df = pd.read_csv(MASTER_TABLE_PATH)
            print(f"  [INFO] cleaning.py not found, but master_table.csv exists ({len(df):,} rows)")
            print(f"  Proceeding with existing master_table.csv")
            return True
        else:
            print("  [ERROR] Neither cleaning.py nor master_table.csv found!")
            print("  Student 1 must provide data/processed/master_table.csv first.")
            return False


# ──────────────────────────────────────────────
# Step 2: Feature Engineering → feature_table.csv
# ──────────────────────────────────────────────
def step_features(skip=False):
    """Run Student 2's feature engineering pipeline."""
    print_banner(2, 4, "FEATURE ENGINEERING → feature_table.csv")

    if skip:
        print("  [SKIP] --skip-features flag set")
        if os.path.exists(FEATURE_TABLE_PATH):
            df = pd.read_csv(FEATURE_TABLE_PATH)
            print(f"  Using existing feature_table.csv ({len(df):,} rows)")
            return True
        else:
            print("  [ERROR] feature_table.csv not found and features was skipped!")
            return False

    print("  Loading master_table.csv...")
    df = pd.read_csv(MASTER_TABLE_PATH, parse_dates=["date"])
    print(f"  Loaded {len(df):,} rows")

    print("  Running feature engineering (src/features.py)...")
    try:
        from features import build_features
        df_features = build_features(df)
        df_features.to_csv(FEATURE_TABLE_PATH, index=False)
        print(f"  ✓ Feature table saved: {FEATURE_TABLE_PATH} ({len(df_features):,} rows, {len(df_features.columns)} cols)")
        return True
    except Exception as e:
        print(f"  [ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────
# Step 3: Demand Model Training → demand_model.pkl
# ──────────────────────────────────────────────
def step_demand_model():
    """Run Student 2's demand forecasting model training."""
    print_banner(3, 4, "DEMAND FORECASTING MODEL → demand_model.pkl")

    try:
        from demand_model import main as train_demand_model
        model, model_name = train_demand_model(FEATURE_TABLE_PATH)
        print(f"  ✓ Demand model trained: {model_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Demand model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────
# Step 4: Stock-out Classification (Student 3)
# ──────────────────────────────────────────────
def step_stockout_model():
    """Run Student 3's stock-out classification model if available."""
    print_banner(4, 4, "STOCK-OUT CLASSIFICATION MODEL (Student 3)")

    stockout_script = os.path.join(SRC_DIR, "stockout_model.py")
    if os.path.exists(stockout_script):
        try:
            import stockout_model
            if hasattr(stockout_model, "main"):
                stockout_model.main()
            print("  ✓ Stock-out model trained")
            return True
        except Exception as e:
            print(f"  [ERROR] Stock-out model failed: {e}")
            return False
    else:
        print("  [INFO] stockout_model.py not found — Student 3 has not created it yet.")
        print("  Skipping. The demand model pipeline is complete on its own.")
        return True


# ──────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="StockSense — End-to-end reproducible pipeline"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip cleaning step (use existing master_table.csv)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering (use existing feature_table.csv)",
    )
    args = parser.parse_args()

    print()
    print("+" + "=" * 58 + "+")
    print("|    STOCKSENSE — End-to-End Reproducible Pipeline       |")
    print("+" + "=" * 58 + "+")

    start_time = time.time()
    ensure_dirs()

    steps = [
        ("Data Cleaning", lambda: step_cleaning(skip=args.skip_cleaning)),
        ("Feature Engineering", lambda: step_features(skip=args.skip_features)),
        ("Demand Forecasting", step_demand_model),
        ("Stock-out Classification", step_stockout_model),
    ]

    results = {}
    for step_name, step_fn in steps:
        success = step_fn()
        results[step_name] = "✅ Pass" if success else "❌ Fail"
        if not success and step_name in ["Data Cleaning", "Feature Engineering", "Demand Forecasting"]:
            print(f"\n  [ABORT] Critical step '{step_name}' failed. Stopping pipeline.")
            break

    elapsed = time.time() - start_time

    print()
    print("+" + "=" * 58 + "+")
    print("|    PIPELINE SUMMARY                                    |")
    print("+" + "-" * 58 + "+")
    for step_name, status in results.items():
        print(f"|  {status} {step_name:<50s}  |")
    print("+" + "-" * 58 + "+")
    print(f"|  Total time: {elapsed:.1f}s{' ' * (44 - len(f'{elapsed:.1f}'))}|")
    print("+" + "=" * 58 + "+")

    # Check final outputs
    print("\nOutput files:")
    for path, label in [
        (MASTER_TABLE_PATH, "Master Table"),
        (FEATURE_TABLE_PATH, "Feature Table"),
        (DEMAND_MODEL_PATH, "Demand Model"),
        (os.path.join(MODELS_DIR, "demand_model_card.md"), "Model Card"),
        (os.path.join(REPORTS_DIR, "model_comparison.md"), "Model Comparison"),
    ]:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {label}: {path}")


if __name__ == "__main__":
    main()
