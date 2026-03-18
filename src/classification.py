import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_classification_models(X_train, X_test, y_train, y_test, outputs_dir='../outputs'):
    """
    Train and evaluate Logistic Regression for Classification (predicting has_insurance).
    Saves the trained model and outputs a Confusion Matrix plot.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("CLASSIFICATION MODELING STAGE")
    print("=" * 50)

    # ──────────────────────────────────────────────
    # STEP 1: Standardize Features
    # ──────────────────────────────────────────────
    print("[INFO] Scaling features for classification...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler specifically for the classification module
    scaler_path = os.path.join(outputs_dir, 'scaler_clf.pkl')
    joblib.dump(scaler, scaler_path)

    # ──────────────────────────────────────────────
    # STEP 2: Train Logistic Regression
    # ──────────────────────────────────────────────
    print("\n--- Training Logistic Regression ---")
    
    # We use a high max_iter to ensure convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # ──────────────────────────────────────────────
    # STEP 3: Evaluation Metrics
    # ──────────────────────────────────────────────
    # Using weighted average since classes might be imbalanced (True vs False)
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
    model_path = os.path.join(outputs_dir, 'logistic_model.pkl')
    joblib.dump(model, model_path)
    
    # Confusion Matrix Plot
    # Ensure labels are [False, True] explicitly
    cm = confusion_matrix(y_test, y_pred, labels=[False, True])
    
    # The standard sklearn cm outputs Cols: [Predicted False, Predicted True]
    # The user's image requests Cols: [Predicted Insurance, Predicted No Insurance]
    # We must swap the columns to match their diagram.
    custom_cm = np.array([
        [cm[0, 1], cm[0, 0]],  # Row 0: Actual No Insurance
        [cm[1, 1], cm[1, 0]]   # Row 1: Actual Insurance
    ])
    
    plt.figure(figsize=(7, 6))
    
    # Styling the heatmap
    sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.title('Logistic Regression: Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    
    # Apply the exact labels from the user's provided diagram over the ticks
    plt.gca().set_xticklabels(['Predicted\nInsurance', 'Predicted No\nInsurance'], fontsize=12, fontweight='bold')
    plt.gca().set_yticklabels(['Actual No\nInsurance', 'Actual\nInsurance'], fontsize=12, fontweight='bold', va='center')
    
    plt.tight_layout()
    
    plot_path = os.path.join(outputs_dir, 'logistic_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[INFO] Logistic Regression model and plot saved successfully.")
    return model
