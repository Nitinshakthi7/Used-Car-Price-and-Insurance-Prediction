import pandas as pd
import numpy as np
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
    print(f"\nFirst 5 Rows   :\n{df.head()}")
    print(f"\nData Types     :\n{df.dtypes}")
    print(f"\nMissing Values :\n{df.isnull().sum()}")


# ──────────────────────────────────────────────
# 3. Handle Missing Values
# ──────────────────────────────────────────────
def handle_missing_values(df):
    """
    - Drop rows where target (price) is missing.
    - Fill numeric columns with median.
    - Fill categorical columns with mode.
    """
    print("\n[STEP] Handling missing values...")

    # Drop rows with missing target
    df = df.dropna(subset=['price'])

    # Numeric columns → fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns → fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[INFO] Missing values after handling:\n{df.isnull().sum()}")
    return df


# ──────────────────────────────────────────────
# 4. Drop Irrelevant Columns
# ──────────────────────────────────────────────
def drop_irrelevant_columns(df):
    """
    Drop columns not useful for price prediction:
    - reg_year   : registration date (not meaningful as-is)
    - reg_number : vehicle ID / plate number
    - title      : full car title string (redundant)
    - overall_cost: cost metric (risk of data leakage with price)
    """
    print("\n[STEP] Dropping irrelevant columns...")

    cols_to_drop = ['reg_year', 'reg_number', 'title', 'overall_cost']
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing, errors='ignore')

    print(f"[INFO] Dropped columns: {existing if existing else 'None'}")
    return df


# ──────────────────────────────────────────────
# 5. Encode Categorical Variables
# ──────────────────────────────────────────────
def encode_categorical(df):
    """
    Apply Label Encoding to all remaining object-type columns.
    This handles: brand, model, fuel_type, transmission,
                  ownership (e.g. '1st owner'), spare_key (Yes/No),
                  has_insurance (TRUE/FALSE) if still present.
    """
    print("\n[STEP] Encoding categorical variables...")

    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  [ENCODED] {col}")

    print("[INFO] Categorical encoding complete.")
    return df


# ──────────────────────────────────────────────
# 6. Full Preprocessing Pipeline
# ──────────────────────────────────────────────
def preprocess(filepath, target='price', test_size=0.2, random_state=42):
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    filepath     : str   – path to the CSV dataset
    target       : str   – column name of the regression target
    test_size    : float – proportion of data held out for testing
    random_state : int   – seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_data(filepath)
    explore_data(df)
    df = handle_missing_values(df)
    df = drop_irrelevant_columns(df)
    df = encode_categorical(df)

    # ── Separate features and target ──
    # Drop 'has_insurance' — that is the classification target (future stage)
    drop_cols = [target]
    if 'has_insurance' in df.columns:
        drop_cols.append('has_insurance')

    X = df.drop(columns=drop_cols)
    y = df[target]

    print(f"\n[INFO] Features used     : {X.columns.tolist()}")
    print(f"[INFO] Target variable   : {target}")
    print(f"[INFO] Feature matrix    : {X.shape}")

    # ── Train-Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n[INFO] Training samples : {X_train.shape[0]}")
    print(f"[INFO] Testing  samples : {X_test.shape[0]}")
    print("\n[INFO] Preprocessing complete. Ready for model training.")

    return X_train, X_test, y_train, y_test
