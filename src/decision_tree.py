import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV


# ══════════════════════════════════════════════════════════════════════
#  PART A — REGRESSION  (Predict Car Price)
# ══════════════════════════════════════════════════════════════════════

def run_decision_tree_model(X_train, X_test, y_train, y_test, outputs_dir='../outputs'):
    """
    Train a Decision Tree Regressor with GridSearchCV tuning to predict car prices.
    Evaluates both baseline and tuned models, saves the best model, and plots results.

    Returns
    -------
    best_model : trained & tuned DecisionTreeRegressor
    """

    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "=" * 55)
    print("  DECISION TREE — REGRESSION (Car Price)")
    print("=" * 55)

    # ── STEP 1: Scale features ────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── STEP 2: Baseline (untuned) ─────────────────────────────────
    print("\n--- [Baseline] Untuned Decision Tree ---")
    baseline = DecisionTreeRegressor(random_state=42)
    baseline.fit(X_train_sc, y_train)
    y_pred_b = baseline.predict(X_test_sc)

    mae_b  = mean_absolute_error(y_test, y_pred_b)
    rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_b))
    r2_b   = r2_score(y_test, y_pred_b)
    print(f"  MAE  : ₹{mae_b:>12,.2f}")
    print(f"  RMSE : ₹{rmse_b:>12,.2f}")
    print(f"  R²   : {r2_b:>12.4f}")

    # ── STEP 3: GridSearchCV ───────────────────────────────────────
    print("\n--- [GridSearchCV] Hyperparameter Tuning ---")
    param_grid = {
        'max_depth'        : [5, 8, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf' : [1, 2, 4, 8],
        'max_features'     : ['sqrt', 'log2', None],
    }
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
    )
    grid.fit(X_train_sc, y_train)

    print("\n  Best Parameters:")
    for k, v in grid.best_params_.items():
        print(f"    {k:<22}: {v}")
    print(f"  Best CV R²   : {grid.best_score_:.4f}")

    # ── STEP 4: Evaluate tuned model ──────────────────────────────
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_sc)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n--- [Tuned] Test Set Performance ---")
    print(f"  MAE  : ₹{mae:>12,.2f}")
    print(f"  RMSE : ₹{rmse:>12,.2f}")
    print(f"  R²   : {r2:>12.4f}")

    print("\n--- Improvement (Baseline → Tuned) ---")
    print(f"  R²  : {r2_b:.4f} → {r2:.4f}  (Δ {r2 - r2_b:+.4f})")
    print(f"  MAE : ₹{mae_b:,.0f} → ₹{mae:,.0f}  (Δ ₹{mae - mae_b:+,.0f})")

    # Save model
    model_path = os.path.join(outputs_dir, 'decision_tree_regressor.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n[INFO] Tuned regressor saved → {model_path}")

    # ── STEP 5: Plot ───────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=y_test, y=y_pred, alpha=0.6,
        color="#1ABC9C", edgecolor="w", linewidth=0.5, s=50,
        label="Predicted vs Actual (Tuned)"
    )
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             color='#E74C3C', linestyle='--', linewidth=2,
             label='Perfect Prediction Line')

    plt.title(
        f'Decision Tree (Tuned): Actual vs. Predicted Car Prices\nR² = {r2:.4f}',
        fontsize=15, fontweight='bold', pad=15
    )
    plt.xlabel('Actual Price (₹)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
    plt.ticklabel_format(style='plain', axis='both')
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(outputs_dir, 'decision_tree_regression_predicted.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Regression plot saved → {plot_path}")
    print("\n[INFO] Regression Decision Tree complete.")

    return best_model


# ══════════════════════════════════════════════════════════════════════
#  PART B — CLASSIFICATION  (Predict Insurance: Yes / No)
# ══════════════════════════════════════════════════════════════════════

def run_decision_tree_classifier(X_train, X_test, y_train, y_test, outputs_dir='../outputs'):
    """
    Train a Decision Tree Classifier with GridSearchCV tuning to predict
    whether a car has insurance (has_insurance: 0 = No, 1 = Yes).

    Evaluates both baseline and tuned models, saves the best model,
    and generates a confusion matrix plot.

    Returns
    -------
    best_model : trained & tuned DecisionTreeClassifier
    """

    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "=" * 55)
    print("  DECISION TREE — CLASSIFICATION (Has Insurance?)")
    print("=" * 55)

    # ── STEP 1: Scale features ────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── STEP 2: Baseline (untuned) ─────────────────────────────────
    print("\n--- [Baseline] Untuned Decision Tree Classifier ---")
    baseline = DecisionTreeClassifier(random_state=42)
    baseline.fit(X_train_sc, y_train)
    y_pred_b = baseline.predict(X_test_sc)

    acc_b = accuracy_score(y_test, y_pred_b)
    print(f"  Accuracy : {acc_b:.4f}  ({acc_b*100:.2f}%)")
    print("\n  Classification Report (Baseline):")
    print(classification_report(y_test, y_pred_b,
                                target_names=["No Insurance", "Has Insurance"]))

    # ── STEP 3: GridSearchCV ───────────────────────────────────────
    print("\n--- [GridSearchCV] Hyperparameter Tuning ---")
    param_grid = {
        'max_depth'        : [3, 5, 8, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf' : [1, 2, 4, 8],
        'max_features'     : ['sqrt', 'log2', None],
        'criterion'        : ['gini', 'entropy'],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_train_sc, y_train)

    print("\n  Best Parameters:")
    for k, v in grid.best_params_.items():
        print(f"    {k:<22}: {v}")
    print(f"  Best CV Accuracy : {grid.best_score_:.4f}  ({grid.best_score_*100:.2f}%)")

    # ── STEP 4: Evaluate tuned model ──────────────────────────────
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)

    print("\n--- [Tuned] Test Set Performance ---")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("\n  Classification Report (Tuned):")
    print(classification_report(y_test, y_pred,
                                target_names=["No Insurance", "Has Insurance"]))

    print("\n--- Improvement (Baseline → Tuned) ---")
    print(f"  Accuracy : {acc_b*100:.2f}% → {acc*100:.2f}%  (Δ {(acc - acc_b)*100:+.2f}%)")

    # Save model
    model_path = os.path.join(outputs_dir, 'decision_tree_classifier.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n[INFO] Tuned classifier saved → {model_path}")

    # ── STEP 5: Confusion Matrix Plot ─────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Insurance", "Has Insurance"]
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format='d'
    )
    ax.set_title(
        f'Decision Tree (Tuned): Insurance Prediction\nAccuracy = {acc*100:.2f}%',
        fontsize=14, fontweight='bold', pad=15
    )
    plt.tight_layout()

    cm_path = os.path.join(outputs_dir, 'decision_tree_classifier_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved → {cm_path}")

    # ── STEP 6: Class Distribution Bar Plot ───────────────────────
    labels  = ["No Insurance", "Has Insurance"]
    actuals = [int((y_test == 0).sum()), int((y_test == 1).sum())]
    preds   = [int((y_pred == 0).sum()), int((y_pred == 1).sum())]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, actuals, width, label='Actual',    color='#3498DB', alpha=0.85)
    bars2 = ax.bar(x + width/2, preds,   width, label='Predicted', color='#1ABC9C', alpha=0.85)

    ax.set_xlabel('Insurance Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count',            fontsize=12, fontweight='bold')
    ax.set_title('Decision Tree: Actual vs. Predicted Insurance Distribution',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.bar_label(bars1, padding=3, fontsize=10)
    ax.bar_label(bars2, padding=3, fontsize=10)
    sns.set_theme(style="whitegrid")
    plt.tight_layout()

    dist_path = os.path.join(outputs_dir, 'decision_tree_classifier_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[INFO] Distribution plot saved → {dist_path}")

    print("\n[INFO] Classification Decision Tree complete.")
    return best_model
