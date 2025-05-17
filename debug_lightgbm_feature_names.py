#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to investigate the LightGBM feature name mapping issue.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import shap
import warnings

from config import settings
from utils import io

# Configure warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def debug_lightgbm_feature_names():
    """Debug LightGBM feature name mapping."""
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print("Loaded LightGBM model data")
    except Exception as e:
        print(f"Error loading LightGBM models: {e}")
        return
    
    # Get a representative model
    model_name = next(iter(lightgbm_models.keys()))
    model_info = lightgbm_models[model_name]
    print(f"Using model: {model_name}")
    
    # Check if feature_name_mapping exists
    if 'feature_name_mapping' not in model_info:
        print("Error: No feature_name_mapping found in model_info")
        return
    
    # Get feature name mapping
    feature_name_mapping = model_info['feature_name_mapping']
    print(f"Found feature_name_mapping with {len(feature_name_mapping)} entries")
    
    # Print first 10 entries to see the structure
    print("\nFirst 10 entries in feature_name_mapping:")
    for i, (k, v) in enumerate(list(feature_name_mapping.items())[:10]):
        print(f"  {k} -> {v}")
    
    # Check if cleaned_feature_names exists and matches keys in mapping
    if 'cleaned_feature_names' not in model_info:
        print("Error: No cleaned_feature_names found in model_info")
        return
    
    cleaned_feature_names = model_info['cleaned_feature_names']
    print(f"Found cleaned_feature_names with {len(cleaned_feature_names)} entries")
    
    # Print first 10 entries
    print("\nFirst 10 entries in cleaned_feature_names:")
    for i, name in enumerate(cleaned_feature_names[:10]):
        print(f"  {i}: {name}")
        if name in feature_name_mapping:
            print(f"     Maps to: {feature_name_mapping[name]}")
        else:
            print(f"     No mapping found for this name")
    
    # Check if X_test_clean exists and columns match cleaned_feature_names
    if 'X_test_clean' not in model_info:
        print("Error: No X_test_clean found in model_info")
        return
    
    X_test_clean = model_info['X_test_clean']
    print(f"Found X_test_clean with {X_test_clean.shape[1]} columns")
    
    # Print first 10 columns
    print("\nFirst 10 columns in X_test_clean:")
    for i, col in enumerate(X_test_clean.columns[:10]):
        print(f"  {i}: {col}")
        if col in feature_name_mapping:
            print(f"     Maps to: {feature_name_mapping[col]}")
        else:
            print(f"     No mapping found for this column")
    
    # Test creating inverse mapping
    inverse_mapping = {v: k for k, v in feature_name_mapping.items()}
    print(f"\nCreated inverse_mapping with {len(inverse_mapping)} entries")
    
    # Test mapping a few columns
    print("\nTesting column mapping for a few columns:")
    for i, col in enumerate(X_test_clean.columns[:5]):
        orig_name = inverse_mapping.get(col, col)
        print(f"  {col} -> {orig_name}")
    
    # Create DataFrame with original feature names
    X_mapped = X_test_clean.copy()
    
    # Check if every column can be mapped
    unmapped_cols = []
    for col in X_mapped.columns:
        if col not in feature_name_mapping:
            unmapped_cols.append(col)
    
    print(f"\nFound {len(unmapped_cols)} columns that could not be mapped")
    if unmapped_cols:
        print(f"First 5 unmapped columns: {unmapped_cols[:5]}")
    
    # Try correcting the inverse mapping
    print("\nLet's try creating an index-based mapping instead")
    # Create mapping from index to original name
    index_to_original = {}
    for i, cleaned_name in enumerate(cleaned_feature_names):
        mapped_name = feature_name_mapping.get(cleaned_name)
        if mapped_name:
            index_to_original[i] = mapped_name
        else:
            index_to_original[i] = cleaned_name  # Keep as is if not in mapping
    
    print(f"Created index_to_original mapping with {len(index_to_original)} entries")
    
    # Print first 10 mappings
    print("\nFirst 10 entries in index_to_original:")
    for i in range(10):
        if i in index_to_original:
            print(f"  {i}: feature_{i} -> {index_to_original[i]}")
        else:
            print(f"  {i}: No mapping found")

if __name__ == "__main__":
    debug_lightgbm_feature_names()