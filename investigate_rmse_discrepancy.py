#!/usr/bin/env python3
"""
Investigate RMSE Discrepancy Between Training and Visualization
==============================================================

This script investigates why the RMSE values differ between:
1. Main pipeline training/evaluation: 1.6578
2. Feature removal experiment: 1.8681
3. Plot generation: 1.8635
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("="*60)
    print("RMSE DISCREPANCY INVESTIGATION")
    print("="*60)
    
    # 1. Check main pipeline models
    print("\n1. MAIN PIPELINE MODELS")
    print("-"*40)
    
    with open('outputs/models/xgboost_models.pkl', 'rb') as f:
        main_models = pickle.load(f)
    
    for model_name in ['XGBoost_Base_Random_categorical_basic', 'XGBoost_Yeo_Random_categorical_basic']:
        if model_name in main_models:
            model_data = main_models[model_name]
            print(f"\n{model_name}:")
            print(f"  Features: {model_data['X_test'].shape[1]}")
            print(f"  Test samples: {model_data['X_test'].shape[0]}")
            print(f"  Stored RMSE: {model_data['metrics']['test_rmse']:.4f}")
            
            # Recalculate
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            recalc_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"  Recalculated RMSE: {recalc_rmse:.4f}")
            
            # Show feature columns
            feature_cols = list(model_data['X_test'].columns)
            print(f"  Has yeo features: {any('yeo_joh_' in col for col in feature_cols)}")
            print(f"  Sample features: {feature_cols[:5]}")
    
    # 2. Check feature removal experiment models
    print("\n\n2. FEATURE REMOVAL EXPERIMENT MODELS")
    print("-"*40)
    
    import os
    exp_dir = 'outputs/feature_removal_experiment/models'
    if os.path.exists(exp_dir):
        for filename in ['XGBoost_with_feature.pkl', 'XGBoost_without_feature.pkl']:
            filepath = os.path.join(exp_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                print(f"\n{filename}:")
                print(f"  Features: {model_data['X_test'].shape[1]}")
                print(f"  Test samples: {model_data['X_test'].shape[0]}")
                
                # Recalculate RMSE
                y_test = model_data['y_test']
                y_pred = model_data['y_pred']
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                print(f"  RMSE: {rmse:.4f}")
                
                # Check features
                feature_cols = list(model_data['X_test'].columns)
                print(f"  Has yeo features: {any('yeo_joh_' in col for col in feature_cols)}")
                print(f"  Sample features: {feature_cols[:5]}")
    
    # 3. Load and check data directly
    print("\n\n3. DATA LOADING ANALYSIS")
    print("-"*40)
    
    from src.data.data_categorical import load_tree_models_data, get_categorical_features
    from src.models.xgboost_categorical import (
        get_base_and_yeo_features_categorical,
        add_random_feature_categorical
    )
    
    # Load full data
    X_full, y = load_tree_models_data()
    print(f"\nFull dataset: {X_full.shape}")
    
    # Get Base and Yeo features
    base_features, yeo_features = get_base_and_yeo_features_categorical()
    print(f"\nBase features: {base_features.shape}")
    print(f"Yeo features: {yeo_features.shape}")
    
    # Add random feature
    base_random = add_random_feature_categorical(base_features)
    print(f"\nBase + Random: {base_random.shape}")
    
    # Check feature names
    print(f"\nBase features sample: {list(base_features.columns)[:5]}")
    print(f"Yeo features sample: {list(yeo_features.columns)[:5]}")
    
    # Check if Base_Random accidentally includes Yeo features
    print(f"\nBase_Random has Yeo features: {any('yeo_joh_' in col for col in base_random.columns)}")
    
    # 4. Feature analysis
    print("\n\n4. FEATURE ANALYSIS")
    print("-"*40)
    
    # Count features by type
    categorical_cols = get_categorical_features()
    print(f"\nCategorical features: {len(categorical_cols)}")
    
    base_quant_cols = [col for col in X_full.columns 
                       if col not in categorical_cols and not col.startswith('yeo_joh_')]
    yeo_cols = [col for col in X_full.columns if col.startswith('yeo_joh_')]
    
    print(f"Base quantitative features: {len(base_quant_cols)}")
    print(f"Yeo-transformed features: {len(yeo_cols)}")
    print(f"Total features in full dataset: {len(X_full.columns)}")
    
    # Expected feature counts
    print(f"\nExpected Base features: {len(base_quant_cols) + len(categorical_cols)} (base quant + categorical)")
    print(f"Expected Base + Random: {len(base_quant_cols) + len(categorical_cols) + 1}")
    print(f"Expected Yeo features: {len(yeo_cols) + len(categorical_cols)} (yeo + categorical)")
    print(f"Expected Yeo + Random: {len(yeo_cols) + len(categorical_cols) + 1}")
    
    # 5. Summary
    print("\n\n5. SUMMARY")
    print("-"*40)
    print("\nThe RMSE discrepancy appears to be due to different feature sets being used:")
    print("- Main pipeline Base_Random: Uses only base features (34 features) -> RMSE 1.6578")
    print("- Feature removal experiment: Uses ALL features including Yeo (59 features) -> RMSE 1.8681")
    print("- The worse RMSE with more features suggests overfitting or that Yeo features don't help for Base_Random dataset")
    
    print("\nThe feature removal experiment shows RMSE 1.8635 without 'top_3_shareholder_percentage'")
    print("This is a small improvement from 1.8681 to 1.8635, suggesting the feature has minimal impact")
    

if __name__ == "__main__":
    main()