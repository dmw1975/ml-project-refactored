#!/usr/bin/env python3
"""Trace data loading to find where 45 rows are being dropped."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))

from src.data.data import load_features_data, load_scores_data, get_base_and_yeo_features
from src.data.data_categorical import load_tree_models_data, load_linear_models_data

# Import load_tree_models_from_csv after fixing the import
import sys
sys.path.append('src/data')
from data_tree_models import load_tree_models_from_csv

print("=" * 80)
print("TRACING DATA LOADING TO FIND WHERE 45 ROWS ARE DROPPED")
print("=" * 80)

# Check raw data
print("\n1. RAW DATA FILES:")
print("-" * 40)
for file in ['combined_df_for_tree_models.csv', 'combined_df_for_linear_models.csv', 'score.csv']:
    df = pd.read_csv(f'data/raw/{file}')
    print(f"{file}: {df.shape[0]} rows, {df.shape[1]} columns")

# Load features and scores separately
print("\n2. LOADING FEATURES AND SCORES (src.data.data):")
print("-" * 40)
features = load_features_data()
print(f"Features loaded: {features.shape[0]} rows, {features.shape[1]} columns")
print(f"Features NaN count: {features.isnull().sum().sum()}")

scores = load_scores_data()
print(f"Scores loaded: {len(scores)} rows")
print(f"Scores NaN count: {scores.isnull().sum()}")

# Check if indices match
print(f"\nFeatures index type: {type(features.index)}, first 5: {features.index[:5].tolist()}")
print(f"Scores index type: {type(scores.index)}, first 5: {scores.index[:5].tolist()}")

# Get base and yeo features
print("\n3. GET BASE AND YEO FEATURES:")
print("-" * 40)
try:
    LR_Base, LR_Yeo, base_cols, yeo_cols = get_base_and_yeo_features(features)
    print(f"LR_Base shape: {LR_Base.shape}")
    print(f"LR_Yeo shape: {LR_Yeo.shape}")
    print(f"Base NaN count: {LR_Base.isnull().sum().sum()}")
    print(f"Yeo NaN count: {LR_Yeo.isnull().sum().sum()}")
except Exception as e:
    print(f"Error: {e}")

# Load tree models data via data_tree_models
print("\n4. LOAD TREE MODELS DATA (src.data.data_tree_models):")
print("-" * 40)
try:
    X, y, cat_features = load_tree_models_from_csv()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X NaN count: {X.isnull().sum().sum()}")
    print(f"y NaN count: {y.isnull().sum()}")
    
    # Check indices
    print(f"\nX index name: {X.index.name}, unique count: {X.index.nunique()}")
    print(f"y index name: {y.index.name}, unique count: {y.index.nunique()}")
    
    # Check for duplicate indices
    if X.index.duplicated().any():
        print(f"WARNING: X has {X.index.duplicated().sum()} duplicate indices!")
    if y.index.duplicated().any():
        print(f"WARNING: y has {y.index.duplicated().sum()} duplicate indices!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Load via data_categorical
print("\n5. LOAD VIA DATA_CATEGORICAL:")
print("-" * 40)
try:
    # Tree models
    tree_features, tree_target = load_tree_models_data()
    print(f"Tree models - Features: {tree_features.shape}, Target: {len(tree_target)}")
    print(f"Tree features NaN: {tree_features.isnull().sum().sum()}")
    print(f"Tree target NaN: {tree_target.isnull().sum()}")
    
    # Linear models
    linear_features, linear_target = load_linear_models_data()
    print(f"Linear models - Features: {linear_features.shape}, Target: {len(linear_target)}")
    print(f"Linear features NaN: {linear_features.isnull().sum().sum()}")
    print(f"Linear target NaN: {linear_target.isnull().sum()}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Check alignment issues
print("\n6. CHECKING ALIGNMENT ISSUES:")
print("-" * 40)

# Load raw tree models data
tree_raw = pd.read_csv('data/raw/combined_df_for_tree_models.csv')
scores_raw = pd.read_csv('data/raw/score.csv')

print(f"Tree raw issuer_name unique: {tree_raw['issuer_name'].nunique()}")
print(f"Scores raw issuer_name unique: {scores_raw['issuer_name'].nunique()}")

# Check for mismatches
tree_issuers = set(tree_raw['issuer_name'])
score_issuers = set(scores_raw['issuer_name'])

print(f"\nIssuers in tree but not in scores: {len(tree_issuers - score_issuers)}")
print(f"Issuers in scores but not in tree: {len(score_issuers - tree_issuers)}")

# Check the actual intersection after index alignment
tree_raw_indexed = tree_raw.set_index('issuer_name')
scores_raw_indexed = scores_raw.set_index('issuer_name')
common_issuers = tree_raw_indexed.index.intersection(scores_raw_indexed.index)
print(f"Common issuers after intersection: {len(common_issuers)}")

print("\n7. SIMULATING TREE MODEL NaN DROPPING:")
print("-" * 40)
# Simulate what happens in enhanced_xgboost_categorical.py
X_sim = tree_raw.set_index('issuer_name')
y_sim = scores_raw.set_index('issuer_name')['esg_score']

# Align
common_idx = X_sim.index.intersection(y_sim.index)
X_aligned = X_sim.loc[common_idx]
y_aligned = y_sim.loc[common_idx]

print(f"After alignment: X={X_aligned.shape}, y={len(y_aligned)}")

# Check for NaN
if X_aligned.isnull().any().any() or y_aligned.isnull().any():
    print("Found NaN values!")
    mask = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
    X_clean = X_aligned[mask]
    y_clean = y_aligned[mask]
    print(f"After dropping NaN: X={X_clean.shape}, y={len(y_clean)}")
    print(f"Rows dropped: {len(X_aligned) - len(X_clean)}")
else:
    print("No NaN values found in aligned data!")

print("\n" + "=" * 80)