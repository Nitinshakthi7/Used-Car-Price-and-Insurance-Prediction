import os
import sys

# ── Ensure imports work when running from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess
from regression import run_regression_models

# ──────────────────────────────────────────────
# Paths  (update 'cars.csv' to your actual filename if different)
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'pre-owned cars.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')


def main():
    print("=" * 55)
    print("   SUPERVISED LEARNING PROJECT — PRE-OWNED CARS")
    print("=" * 55)

    # ── STAGE 1: Preprocessing ──────────────────────────────
    print("\n>>> STAGE 1: DATA PREPROCESSING")
    X_train, X_test, y_train, y_test = preprocess(DATA_PATH)

    # ── STAGE 2: Regression ─────────────────────────────────
    print("\n>>> STAGE 2: MODEL TRAINING (LINEAR & KNN)")
    trained_models = run_regression_models(
        X_train, X_test, y_train, y_test,
        outputs_dir=OUTPUTS_DIR
    )

    # ──────────────────────────────────────────────────────
    # STAGE 3: Classification  (classification.py — future)
    # ──────────────────────────────────────────────────────
    print("\n>>> STAGE 3: CLASSIFICATION MODEL")
    print("    [Future] classification.py will predict 'has_insurance'.")
    print("    Planned : Logistic Regression / Decision Tree / KNN")

    print("\n" + "=" * 55)
    print("   Run complete.")
    print(f"   Outputs saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
