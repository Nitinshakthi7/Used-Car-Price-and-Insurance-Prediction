import os
import sys

# ── Ensure src/ is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess
from decision_tree import run_decision_tree_model, run_decision_tree_classifier

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'pre-owned cars.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')


def main():
    print("=" * 55)
    print("   DECISION TREE MODEL EXECUTION")
    print("=" * 55)

    # ════════════════════════════════════════════════
    #  PART A — REGRESSION: Predict Car Price
    # ════════════════════════════════════════════════
    print("\n>>> PART A: REGRESSION — Predict Car Price")
    print(">>> STAGE 1: Data Preprocessing (target = price)")
    X_train, X_test, y_train, y_test = preprocess(DATA_PATH, target='price')

    print("\n>>> STAGE 2: Decision Tree Regression (with GridSearchCV)")
    run_decision_tree_model(
        X_train, X_test, y_train, y_test,
        outputs_dir=OUTPUTS_DIR
    )

    # ════════════════════════════════════════════════
    #  PART B — CLASSIFICATION: Predict Has Insurance
    # ════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print(">>> PART B: CLASSIFICATION — Predict Insurance Status")
    print(">>> STAGE 1: Data Preprocessing (target = has_insurance)")
    X_train_c, X_test_c, y_train_c, y_test_c = preprocess(DATA_PATH, target='has_insurance')

    print("\n>>> STAGE 2: Decision Tree Classification (with GridSearchCV)")
    run_decision_tree_classifier(
        X_train_c, X_test_c, y_train_c, y_test_c,
        outputs_dir=OUTPUTS_DIR
    )

    print("\n" + "=" * 55)
    print("   Execution complete.")
    print(f"   Outputs saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
