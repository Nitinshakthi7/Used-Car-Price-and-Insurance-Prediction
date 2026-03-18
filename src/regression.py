import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_regression_models(X_train, X_test, y_train, y_test, outputs_dir='../outputs'):
    """
    Train both Linear Regression (Baseline) and KNN Regression (Improvement).
    Evaluates both models, saves them as .pkl files, and generates plots.

    Returns
    -------
    models : dict of trained models
    """

    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("REGRESSION MODELING STAGE")
    print("=" * 50)

    # ──────────────────────────────────────────────
    # STEP 1: Standardize Features
    # ──────────────────────────────────────────────
    # Both Linear Regression and KNN require feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    scaler_path = os.path.join(outputs_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Features standardized. Scaler saved → {scaler_path}")

    models_to_run = {
        "Linear Regression": LinearRegression(),
        "KNN Regression (k=5)": KNeighborsRegressor(n_neighbors=5)
    }

    trained_models = {}
    
    # ──────────────────────────────────────────────
    # STEP 2: Train & Evaluate Both Models
    # ──────────────────────────────────────────────
    for name, model in models_to_run.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        
        print(f"MAE  : ₹{mae:>12,.2f}")
        print(f"RMSE : ₹{rmse:>12,.2f}")
        print(f"R²   : {r2:>12.4f}")

        # Save model
        safe_name = name.split()[0].lower() + "_model.pkl" # 'linear_model.pkl' or 'knn_model.pkl'
        model_path = os.path.join(outputs_dir, safe_name)
        joblib.dump(model, model_path)
        trained_models[name] = model

        # ──────────────────────────────────────────────
        # STEP 3: Generate Plot for each model
        # ──────────────────────────────────────────────
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 8))

        color = "#2E86C1" if "Linear" in name else "#9B59B6" # Blue for Linear, Purple for KNN
        
        sns.scatterplot(
            x=y_test, y=y_pred, alpha=0.6,
            color=color, edgecolor="w", linewidth=0.5, s=50,
            label="Predicted vs Actual"
        )

        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot(
            [min_val, max_val], [min_val, max_val],
            color='#E74C3C', linestyle='--', linewidth=2,
            label='Perfect Prediction Line'
        )

        plt.title(f'{name}: Actual vs. Predicted Car Prices', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Actual Price (₹)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Price (₹)', fontsize=12, fontweight='bold')
        plt.ticklabel_format(style='plain', axis='both')
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()

        plot_path = os.path.join(outputs_dir, f'{name.split()[0].lower()}_actual_vs_predicted.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print("\n[INFO] All models trained, evaluated, and plots saved.")
    return trained_models
