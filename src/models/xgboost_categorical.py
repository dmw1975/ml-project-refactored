#!/usr/bin/env python3
"""
Redirect to enhanced XGBoost categorical implementation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.archive.enhanced_xgboost_categorical import train_enhanced_xgboost_categorical
from src.data.data_categorical import load_tree_models_data, get_categorical_features
from src.data.data import get_base_and_yeo_features, add_random_feature
import pandas as pd
import numpy as np
import pickle
from src.config import settings

def train_xgboost_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """
    Call the enhanced implementation and return results.
    The enhanced version returns a dictionary with both basic and optuna models.
    """
    return train_enhanced_xgboost_categorical(X, y, dataset_name, categorical_columns, test_size, random_state)

def get_base_and_yeo_features_categorical():
    """Get base and Yeo-Johnson transformed features for categorical models."""
    features, _ = load_tree_models_data()
    
    # Get quantitative columns (exclude categorical)
    categorical_cols = get_categorical_features()
    quantitative_cols = [col for col in features.columns if col not in categorical_cols]
    
    # Split into base and yeo columns
    base_cols = [col for col in quantitative_cols if not col.startswith('yeo_joh_')]
    yeo_cols = [col for col in quantitative_cols if col.startswith('yeo_joh_')]
    
    # Create base features (quantitative + categorical)
    base_features = features[base_cols + categorical_cols].copy()
    
    # Create yeo features (yeo quantitative + categorical)
    yeo_features = features[yeo_cols + categorical_cols].copy()
    
    return base_features, yeo_features

def add_random_feature_categorical(features):
    """Add random feature to categorical dataset."""
    features_copy = features.copy()
    np.random.seed(42)
    features_copy['random_feature'] = np.random.randn(len(features_copy))
    return features_copy

def train_xgboost_categorical_models(datasets=['all']):
    """
    Train XGBoost models with categorical features on specified datasets.
    This function provides compatibility with the existing pipeline.
    """
    print("🌳 Training Enhanced XGBoost models with native categorical features...")
    
    # Load categorical data
    features, target = load_tree_models_data()
    categorical_columns = get_categorical_features()
    
    # Get feature sets
    base_features_df, yeo_features_df = get_base_and_yeo_features_categorical()
    
    all_results = {}
    
    # Define dataset configurations
    dataset_configs = {
        'Base': (base_features_df, 'Base features'),
        'Yeo': (yeo_features_df, 'Yeo-Johnson transformed features'),
    }
    
    # Add random feature variants
    if 'all' in datasets or any('Random' in d for d in datasets):
        for config_name, (feature_subset, description) in list(dataset_configs.items()):
            random_data = add_random_feature_categorical(features)
            dataset_configs[f'{config_name}_Random'] = (
                random_data[feature_subset.columns.tolist() + ['random_feature']], 
                f'{description} + random feature'
            )
    
    # Filter datasets based on request
    if 'all' not in datasets:
        dataset_configs = {k: v for k, v in dataset_configs.items() if k in datasets}
    
    # Train models
    for dataset_name, (X, description) in dataset_configs.items():
        print(f"\n{'='*60}")
        print(f"Training on {dataset_name}: {description}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        try:
            # Call enhanced implementation which returns both basic and optuna models
            results = train_enhanced_xgboost_categorical(
                X, target, dataset_name, categorical_columns
            )
            
            # Add all results from this dataset to the main results dict
            for model_key, model_data in results.items():
                all_results[model_key] = model_data
                
        except Exception as e:
            print(f"Error training XGBoost on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    if all_results:
        combined_path = settings.MODEL_DIR / 'xgboost_models.pkl'
        with open(combined_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n✅ All XGBoost models saved to: {combined_path}")
    
    print(f"\n✅ XGBoost categorical training completed! {len(all_results)} models trained.")
    return all_results
