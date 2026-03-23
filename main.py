import os
import sys

# ── Ensure imports work when running from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess
from regression import run_regression_models
from classification import run_classification_models
from decision_tree import run_decision_tree_model, run_decision_tree_classifier
from knn_classifier import run_knn_classifier

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'pre-owned cars.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')


def main():
    print("=" * 55)
    print("   SUPERVISED LEARNING PROJECT — PRE-OWNED CARS")
    print("=" * 55)

    # ════════════════════════════════════════════════════
    # REGRESSION — Predict Car Price
    # ════════════════════════════════════════════════════
    print("\n>>> STAGE 1: DATA PREPROCESSING (target = price)")
    X_train, X_test, y_train, y_test = preprocess(DATA_PATH, target='price')

    print("\n>>> STAGE 2: REGRESSION MODELS (Linear & KNN)")
    run_regression_models(
        X_train, X_test, y_train, y_test,
        outputs_dir=OUTPUTS_DIR
    )

    print("\n>>> STAGE 3: DECISION TREE REGRESSION")
    run_decision_tree_model(
        X_train, X_test, y_train, y_test,
        outputs_dir=OUTPUTS_DIR
    )

    # ════════════════════════════════════════════════════
    # CLASSIFICATION — Predict Has Insurance
    # ════════════════════════════════════════════════════
    print("\n>>> STAGE 4: DATA PREPROCESSING (target = has_insurance)")
    X_train_c, X_test_c, y_train_c, y_test_c = preprocess(DATA_PATH, target='has_insurance')

    print("\n>>> STAGE 5: LOGISTIC REGRESSION (Classification)")
    run_classification_models(
        X_train_c, X_test_c, y_train_c, y_test_c,
        outputs_dir=OUTPUTS_DIR
    )

    print("\n>>> STAGE 6: DECISION TREE CLASSIFICATION")
    run_decision_tree_classifier(
        X_train_c, X_test_c, y_train_c, y_test_c,
        outputs_dir=OUTPUTS_DIR
    )

    # ── KNN Classification ──
    print("\n>>> STAGE 7: KNN CLASSIFICATION")
    run_knn_classifier(
        X_train_c, X_test_c, y_train_c, y_test_c,
        outputs_dir=OUTPUTS_DIR
    )

    print("\n" + "=" * 55)
    print("   Run complete. All Stages Executed Successfully.")
    print(f"   Outputs saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
