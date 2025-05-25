#!/usr/bin/env python3
"""Debug script to trace where tree models lose samples."""

import pandas as pd
import numpy as np
from data_tree_models import get_tree_model_datasets, perform_stratified_split_for_tree_models
from config import settings

print("=== Debugging Tree Model Data Size ===")

# Step 1: Load raw data
print("\n1. Loading raw tree model datasets...")
datasets, y = get_tree_model_datasets()

print(f"\nDatasets returned:")
for name, df in datasets.items():
    print(f"  {name}: {df.shape}")

print(f"\nTarget (y) shape: {y.shape}")

# Step 2: Check a specific dataset through the full pipeline
print("\n2. Following 'Base' dataset through pipeline...")
X = datasets['Base']
print(f"Base dataset shape before split: {X.shape}")

# Step 3: Perform the split
X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
    X, y, test_size=0.2, random_state=42
)

print(f"\nAfter stratified split:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Total samples: {len(X_train) + len(X_test)}")

# Step 4: Check if there's any filtering in the stratified split
print("\n3. Checking for any NaN values that might cause filtering...")
print(f"  NaN in X: {X.isna().sum().sum()}")
print(f"  NaN in y: {y.isna().sum()}")

# Step 5: Check unique sectors for stratification
if 'gics_sector' in X.columns:
    print(f"\n4. Unique sectors in dataset: {X['gics_sector'].nunique()}")
    print(f"  Sector value counts:")
    print(X['gics_sector'].value_counts())
    
    # Check if any sectors have very few samples
    sector_counts = X['gics_sector'].value_counts()
    small_sectors = sector_counts[sector_counts < 2]
    if len(small_sectors) > 0:
        print(f"\n  WARNING: Found sectors with < 2 samples (cannot stratify):")
        print(small_sectors)