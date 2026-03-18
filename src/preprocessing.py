import pandas as pd
import numpy as np
import joblib 
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
def load_data(filepath):
    """Load the CSV dataset from the given filepath."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {filepath}")
    return df


# ──────────────────────────────────────────────
# 2. Explore Data
# ──────────────────────────────────────────────
def explore_data(df):
    """Print basic exploration info about the dataset."""
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    print(f"Shape          : {df.shape}")
    print(f"\nColumn Names   :\n{df.columns.tolist()}")


# ──────────────────────────────────────────────
# 3. Handle Missing Values
# ──────────────────────────────────────────────
def handle_missing_values(df, target):
    """
    - Drop rows where the target variable is missing.
    - Fill numeric columns with median.
    - Fill categorical columns with mode.
    """
    print(f"\n[STEP] Handling missing values (Target: {target})...")

    # Drop rows with missing target
    if target in df.columns:
        df = df.dropna(subset=[target])

    # Numeric columns → fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns → fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ──────────────────────────────────────────────
# 4. Drop Irrelevant Columns
# ──────────────────────────────────────────────
def drop_irrelevant_columns(df):
    """
    Drop columns not useful for prediction:
    - reg_year   : registration date (not meaningful as-is)
    - reg_number : vehicle ID / plate number
    - title      : full car title string (redundant)
    - overall_cost: cost metric (risk of data leakage with price)
    """
    print("\n[STEP] Dropping irrelevant columns...")

    cols_to_drop = ['reg_year', 'reg_number', 'title', 'overall_cost']
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing, errors='ignore')

    return df


# ──────────────────────────────────────────────
# 5. Encode Categorical Variables
# ──────────────────────────────────────────────
def encode_categorical(df, outputs_dir='../outputs'):
    """
    Apply Label Encoding to all remaining object-type columns.
    Saves the encoders if needed, or just encodes.
    """
    print("\n[STEP] Encoding categorical variables...")

    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [ENCODED] {col}")

    return df


# ──────────────────────────────────────────────
# 6. Full Preprocessing Pipeline
# ──────────────────────────────────────────────
def preprocess(filepath, target='price', test_size=0.2, random_state=42):
    """
    End-to-end preprocessing pipeline.
    """
    df = load_data(filepath)
    explore_data(df)
    df = handle_missing_values(df, target)
    df = drop_irrelevant_columns(df)
    df = encode_categorical(df)

    # ── Separate features and target ──
    drop_cols = [target]
    
    # If we are doing Regression (`price`), drop the future Classification target
    if target == 'price' and 'has_insurance' in df.columns:
        drop_cols.append('has_insurance')
    
    # If we are doing Classification (`has_insurance`), we KEEP `price` as a powerful feature!

    X = df.drop(columns=drop_cols)
    y = df[target]

    print(f"\n[INFO] Features used     : len({len(X.columns)})")
    print(f"[INFO] Target variable   : {target}")
    
    # ── Train-Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=(y if target=='has_insurance' else None)
    )

    print(f"[INFO] Training samples : {X_train.shape[0]}")
    print(f"[INFO] Testing  samples : {X_test.shape[0]}")
    print("\n[INFO] Preprocessing complete.")

    return X_train, X_test, y_train, y_test
