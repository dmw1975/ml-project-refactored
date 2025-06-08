#!/usr/bin/env python3
"""Check model keys."""

import pickle
from pathlib import Path

# Check first model in each file
for model_file in ["lightgbm_models.pkl", "catboost_models.pkl"]:
    filepath = Path("outputs/models") / model_file
    
    with open(filepath, 'rb') as f:
        models = pickle.load(f)
    
    # Get first model
    first_name = list(models.keys())[0]
    first_model = models[first_name]
    
    print(f"\n{model_file}: {first_name}")
    print(f"Has 'model_name': {'model_name' in first_model}")
    print(f"Has 'model': {'model' in first_model}")
    if 'model' in first_model:
        print(f"Model class: {first_model['model'].__class__.__name__}")