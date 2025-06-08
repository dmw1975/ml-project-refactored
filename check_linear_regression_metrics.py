#!/usr/bin/env python3
"""Check if Linear Regression models have metrics."""

import pickle
from pathlib import Path

# Load Linear Regression models
lr_file = Path("outputs/models/linear_regression_models.pkl")
if lr_file.exists():
    with open(lr_file, 'rb') as f:
        lr_models = pickle.load(f)
    
    print("Linear Regression Models Analysis:")
    print("=" * 60)
    
    for model_name, model_data in lr_models.items():
        print(f"\nModel: {model_name}")
        if isinstance(model_data, dict):
            print(f"  Keys: {list(model_data.keys())}")
            
            # Check for metrics
            if 'metrics' in model_data:
                print(f"  ✓ HAS METRICS: {model_data['metrics']}")
            else:
                print("  ✗ NO 'metrics' key")
                
            # Check for other metric-related keys
            metric_keys = [k for k in model_data.keys() if 'metric' in k.lower() or 'score' in k.lower() or 'rmse' in k.lower() or 'mae' in k.lower()]
            if metric_keys:
                print(f"  Other metric-related keys: {metric_keys}")
                for key in metric_keys:
                    print(f"    {key}: {model_data[key]}")
        else:
            print(f"  Not a dictionary, type: {type(model_data)}")
else:
    print("Linear Regression models file not found!")