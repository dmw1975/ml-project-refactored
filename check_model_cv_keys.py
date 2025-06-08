#!/usr/bin/env python3
"""Check what CV keys are in each model type."""

import pickle
from pathlib import Path

model_dir = Path("outputs/models")

# Check each model file
for model_file in ["xgboost_models.pkl", "lightgbm_models.pkl", "catboost_models.pkl", "elasticnet_models.pkl", "linear_regression_models.pkl"]:
    filepath = model_dir / model_file
    if filepath.exists():
        print(f"\n=== {model_file} ===")
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        
        # Check first model in file
        if models:
            first_model_name = list(models.keys())[0]
            first_model = models[first_model_name]
            print(f"First model: {first_model_name}")
            print(f"Keys: {list(first_model.keys())}")
            
            # Check for CV-related keys
            cv_keys = [k for k in first_model.keys() if 'cv' in k.lower() or 'cross' in k.lower()]
            print(f"CV-related keys: {cv_keys}")
            
            # Check for cv_scores specifically
            if 'cv_scores' in first_model:
                print(f"  cv_scores shape: {first_model['cv_scores'].shape if hasattr(first_model['cv_scores'], 'shape') else len(first_model['cv_scores'])}")
            if 'cv_mse' in first_model:
                print(f"  cv_mse: {first_model['cv_mse']}")
            if 'cv_mse_std' in first_model:
                print(f"  cv_mse_std: {first_model['cv_mse_std']}")