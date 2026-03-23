import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def run_knn_classifier(X_train, X_test, y_train, y_test, outputs_dir='../outputs'):
    """
    Train and evaluate KNN Classifier for predicting has_insurance.
    Saves the trained model and outputs a Confusion Matrix plot.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("KNN CLASSIFICATION STAGE")
    print("=" * 50)

    # ──────────────────────────────────────────────
    # STEP 1: Standardize Features
    # ──────────────────────────────────────────────
    print("[INFO] Scaling features for KNN classification...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ──────────────────────────────────────────────
    # STEP 2: Train KNN Classifier (k=5)
    # ──────────────────────────────────────────────
    print("\n--- Training KNN Classifier (k=5) ---")

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # ──────────────────────────────────────────────
    # STEP 3: Evaluation Metrics
    # ──────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy  : {acc * 100:>7.2f}%")
    print(f"Precision : {prec * 100:>7.2f}%")
    print(f"Recall    : {rec * 100:>7.2f}%")
    print(f"F1-Score  : {f1 * 100:>7.2f}%")

    # ──────────────────────────────────────────────
    # STEP 4: Save Model & Visualize
    # ──────────────────────────────────────────────
    model_path = os.path.join(outputs_dir, 'knn_classifier.pkl')
    joblib.dump(model, model_path)

    # Confusion Matrix Plot — same layout as Logistic & Decision Tree
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    custom_cm = np.array([
        [cm[0, 1], cm[0, 0]],  # Row 0: Actual No Insurance
        [cm[1, 1], cm[1, 0]]   # Row 1: Actual Insurance
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.title('KNN Classifier: Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.gca().set_xticklabels(['Predicted\nInsurance', 'Predicted No\nInsurance'],
                               fontsize=12, fontweight='bold')
    plt.gca().set_yticklabels(['Actual No\nInsurance', 'Actual\nInsurance'],
                               fontsize=12, fontweight='bold', va='center')
    plt.tight_layout()

    plot_path = os.path.join(outputs_dir, 'knn_classifier_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[INFO] KNN Classifier model and plot saved successfully.")
    return model
