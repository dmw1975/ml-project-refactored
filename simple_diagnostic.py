#!/usr/bin/env python3
"""Simple diagnostic to understand model exclusion."""

import pickle
from pathlib import Path

# Check each model's metrics
for model_file in ["lightgbm_models.pkl", "catboost_models.pkl"]:
    print(f"\n=== {model_file} ===")
    filepath = Path("outputs/models") / model_file
    
    with open(filepath, 'rb') as f:
        models = pickle.load(f)
    
    for model_name, model_data in models.items():
        print(f"\n{model_name}:")
        
        # Check for metrics
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            print(f"  Has 'metrics' dict: {list(metrics.keys())}")
        else:
            print(f"  No 'metrics' dict")
        
        # Check for individual metric keys
        for key in ['RMSE', 'MAE', 'R2', 'MSE']:
            if key in model_data:
                print(f"  Has '{key}': {model_data[key]}")
        
        # Check for CV scores
        cv_keys = [k for k in model_data.keys() if 'cv' in k.lower()]
        if cv_keys:
            print(f"  CV-related keys: {cv_keys}")