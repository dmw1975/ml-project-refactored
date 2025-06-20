#!/usr/bin/env python3
"""Trace the exact 45-row difference between linear and tree models."""

import pandas as pd
import numpy as np

print("=" * 80)
print("TRACING THE 45-ROW DIFFERENCE")
print("=" * 80)

# Load processed datasets
tree_df = pd.read_csv('data/processed/tree_models_dataset.csv')
linear_df = pd.read_csv('data/processed/linear_models_dataset.csv')

print(f"\nProcessed data shapes:")
print(f"Tree models: {tree_df.shape}")
print(f"Linear models: {linear_df.shape}")

# Check NaN in each
print(f"\nNaN counts:")
print(f"Tree models: {tree_df.isnull().sum().sum()} total NaN values")
print(f"Linear models: {linear_df.isnull().sum().sum()} total NaN values")

# Rows with NaN
tree_nan_rows = tree_df.isnull().any(axis=1).sum()
linear_nan_rows = linear_df.isnull().any(axis=1).sum()
print(f"\nRows with NaN:")
print(f"Tree models: {tree_nan_rows} rows with NaN")
print(f"Linear models: {linear_nan_rows} rows with NaN")

# Simulate what happens when we drop NaN
tree_no_nan = tree_df.dropna()
linear_no_nan = linear_df.dropna()

print(f"\nAfter dropping NaN:")
print(f"Tree models: {tree_no_nan.shape[0]} rows (dropped {tree_df.shape[0] - tree_no_nan.shape[0]})")
print(f"Linear models: {linear_no_nan.shape[0]} rows (dropped {linear_df.shape[0] - linear_no_nan.shape[0]})")

# Now let's trace what actually happens in the models
print("\n" + "-" * 40)
print("SIMULATING ACTUAL MODEL DATA LOADING:")

# Load scores
scores_df = pd.read_csv('data/raw/score.csv')
scores_df = scores_df.set_index('issuer_name')
scores = scores_df['esg_score']

# Tree models approach (from data_categorical.py)
tree_indexed = tree_df.set_index('issuer_name') if 'issuer_name' in tree_df.columns else tree_df
common_tree = tree_indexed.index.intersection(scores.index)
print(f"\nTree models after index alignment: {len(common_tree)} rows")

# Linear models approach
linear_indexed = linear_df.set_index('issuer_name') if 'issuer_name' in linear_df.columns else linear_df
common_linear = linear_indexed.index.intersection(scores.index)
print(f"Linear models after index alignment: {len(common_linear)} rows")

# Now simulate the NaN dropping that happens in tree models
tree_features = tree_indexed.loc[common_tree]
tree_target = scores.loc[common_tree]

# Check for NaN
tree_nan_mask = tree_features.isnull().any(axis=1) | tree_target.isnull()
tree_clean_count = (~tree_nan_mask).sum()
print(f"\nTree models after dropping NaN: {tree_clean_count} rows")
print(f"Difference from linear models: {len(common_linear) - tree_clean_count} rows")

# Let's find which specific issuers are dropped
if tree_nan_mask.any():
    dropped_issuers = tree_features.index[tree_nan_mask]
    print(f"\nIssuers dropped from tree models due to NaN ({len(dropped_issuers)}):")
    for i, issuer in enumerate(dropped_issuers[:10]):
        print(f"  {i+1}. {issuer}")
    if len(dropped_issuers) > 10:
        print(f"  ... and {len(dropped_issuers) - 10} more")

# Check if these issuers exist in linear models
print(f"\nChecking if dropped issuers exist in linear models:")
dropped_in_linear = [issuer for issuer in dropped_issuers if issuer in linear_indexed.index]
print(f"Found {len(dropped_in_linear)} out of {len(dropped_issuers)} dropped issuers in linear models")

# Final verification
print("\n" + "-" * 40)
print("FINAL SUMMARY:")
print(f"Linear models: {len(common_linear)} rows (no NaN dropping)")
print(f"Tree models: {tree_clean_count} rows (after NaN dropping)")
print(f"Difference: {len(common_linear) - tree_clean_count} rows")

# Double check by counting unique categorical values
print("\n" + "-" * 40)
print("CATEGORICAL VALUE DISTRIBUTION:")
cat_cols = ['gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile', 'cntry_of_risk',
            'top_1_shareholder_location', 'top_2_shareholder_location', 'top_3_shareholder_location']

for col in cat_cols:
    if col in tree_features.columns:
        nan_count = tree_features[col].isnull().sum()
        if nan_count > 0:
            print(f"\n{col}: {nan_count} NaN values")

print("\n" + "=" * 80)