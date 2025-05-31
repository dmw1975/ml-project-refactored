"""Debug stratification differences."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add paths
project_root = Path(__file__).parent.absolute()
new_path = Path(__file__).parent / "esg_ml_clean"
sys.path.append(str(project_root))
sys.path.insert(0, str(new_path))

# Import from old pipeline
from data import load_features_data, get_base_and_yeo_features, load_scores_data

# Import new pipeline
from src.data.loader import DataLoader

print("=== Debugging Stratification Differences ===\n")

# OLD PIPELINE
print("1. Old Pipeline Split:")
feature_df = load_features_data()
base_features, _, _, _ = get_base_and_yeo_features(feature_df)
scores = load_scores_data()

# Align data
common_index = base_features.index.intersection(scores.index)
X = base_features.loc[common_index]
y = scores.loc[common_index]

# Simple split (no stratification)
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   - Train indices (first 10): {X_train_old.index[:10].tolist()}")
print(f"   - Test indices (first 10): {X_test_old.index[:10].tolist()}")

# NEW PIPELINE
print("\n2. New Pipeline Split:")
config = {'data_dir': 'data'}
loader = DataLoader(config)

# Default split (with stratification?)
X_train_new, X_test_new, y_train_new, y_test_new = loader.get_train_test_split(
    dataset_type='base',
    test_size=0.2,
    random_state=42,
    stratify_by='sector'  # This is the default
)

print(f"   - Train indices (first 10): {X_train_new.index[:10].tolist()}")
print(f"   - Test indices (first 10): {X_test_new.index[:10].tolist()}")

# Check for sector columns
sector_cols = [col for col in X.columns if col.startswith('gics_sector_')]
print(f"\n3. Sector columns found: {len(sector_cols)}")
if sector_cols:
    print(f"   - Example sectors: {sector_cols[:3]}")

# Compare indices
print(f"\n4. Index comparison:")
print(f"   - Train indices match: {all(X_train_old.index == X_train_new.index)}")
print(f"   - Test indices match: {all(X_test_old.index == X_test_new.index)}")

# Try new pipeline without stratification
print("\n5. New Pipeline Split WITHOUT stratification:")
X_train_no_strat, X_test_no_strat, y_train_no_strat, y_test_no_strat = loader.get_train_test_split(
    dataset_type='base',
    test_size=0.2,
    random_state=42,
    stratify_by=None  # No stratification
)

print(f"   - Train indices match old: {all(X_train_old.index == X_train_no_strat.index)}")
print(f"   - Test indices match old: {all(X_test_old.index == X_test_no_strat.index)}")

# Check actual values
print(f"\n6. Value comparison (no stratification):")
print(f"   - X_train values match: {X_train_old.equals(X_train_no_strat)}")
print(f"   - X_test values match: {X_test_old.equals(X_test_no_strat)}")
print(f"   - y_train values match: {y_train_old.equals(y_train_no_strat)}")
print(f"   - y_test values match: {y_test_old.equals(y_test_no_strat)}")

print("\n=== Analysis Complete ===")