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
    # STAGE 3: Classification  (Predicting 'has_insurance')
    # ──────────────────────────────────────────────────────
    print("\n>>> STAGE 3: CLASSIFICATION MODEL (has_insurance)")
    print("[SYSTEM] Re-running preprocessing to isolate 'has_insurance' target...")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = preprocess(DATA_PATH, target='has_insurance')
    
    from classification import run_classification_models
    clf_model = run_classification_models(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf,
        outputs_dir=OUTPUTS_DIR
    )

    print("\n" + "=" * 55)
    print("   Run complete. All Stages Executed Successfully.")
    print(f"   Outputs saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
