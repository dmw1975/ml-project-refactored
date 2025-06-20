#!/usr/bin/env python3
"""Trace where NaN values come from in tree models data."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))

from src.data.data_categorical import load_tree_models_data

print("=" * 80)
print("TRACING NaN VALUES IN TREE MODELS DATA")
print("=" * 80)

# Load tree models data
tree_features, tree_target = load_tree_models_data()
print(f"\nTree models data loaded: {tree_features.shape}")
print(f"Total NaN values: {tree_features.isnull().sum().sum()}")

# Find columns with NaN
nan_cols = tree_features.columns[tree_features.isnull().any()].tolist()
print(f"\nColumns with NaN values ({len(nan_cols)}):")
for col in nan_cols:
    nan_count = tree_features[col].isnull().sum()
    print(f"  {col}: {nan_count} NaN values ({nan_count/len(tree_features)*100:.1f}%)")

# Find rows with NaN
rows_with_nan = tree_features.isnull().any(axis=1)
print(f"\nRows with at least one NaN: {rows_with_nan.sum()} out of {len(tree_features)} ({rows_with_nan.sum()/len(tree_features)*100:.1f}%)")

# Check which issuers have NaN values
if rows_with_nan.any():
    issuers_with_nan = tree_features.index[rows_with_nan].tolist()
    print(f"\nFirst 10 issuers with NaN values:")
    for issuer in issuers_with_nan[:10]:
        print(f"  {issuer}")

# Let's check the processed tree models dataset directly
print("\n" + "-" * 40)
print("CHECKING PROCESSED TREE MODELS DATASET FILE:")
tree_df = pd.read_csv('data/processed/tree_models_dataset.csv')
print(f"Shape: {tree_df.shape}")
print(f"Total NaN: {tree_df.isnull().sum().sum()}")

if tree_df.isnull().any().any():
    nan_cols_file = tree_df.columns[tree_df.isnull().any()].tolist()
    print(f"\nColumns with NaN in file:")
    for col in nan_cols_file:
        print(f"  {col}: {tree_df[col].isnull().sum()} NaN")

# Compare with raw tree models data
print("\n" + "-" * 40)
print("CHECKING RAW TREE MODELS DATA:")
raw_tree = pd.read_csv('data/raw/combined_df_for_tree_models.csv')
print(f"Shape: {raw_tree.shape}")
print(f"Total NaN: {raw_tree.isnull().sum().sum()}")

# So the issue must be in the categorical data processing
# Let's check each categorical column
print("\n" + "-" * 40)
print("CHECKING CATEGORICAL COLUMNS IN PROCESSED DATA:")
categorical_cols = ['gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile', 
                   'cntry_of_risk', 'top_1_shareholder_location',
                   'top_2_shareholder_location', 'top_3_shareholder_location']

for col in categorical_cols:
    if col in tree_features.columns:
        unique_vals = tree_features[col].value_counts(dropna=False)
        if pd.isna(unique_vals.index).any():
            print(f"\n{col} has NaN values:")
            print(f"  NaN count: {tree_features[col].isnull().sum()}")
            print(f"  Unique values (including NaN): {len(unique_vals)}")

print("\n" + "=" * 80)