#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple test to check CatBoost and XGBoost models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pickle

def check_models():
    """Check available models."""
    models_dir = Path("outputs/models")
    
    print("Checking for CatBoost and XGBoost models...")
    print("="*60)
    
    # List all model files
    model_files = list(models_dir.glob("*.pkl"))
    
    catboost_models = []
    xgboost_models = []
    
    for model_file in model_files:
        if 'catboost' in model_file.name.lower():
            catboost_models.append(model_file)
        elif 'xgboost' in model_file.name.lower():
            xgboost_models.append(model_file)
    
    print(f"\nFound {len(catboost_models)} CatBoost models:")
    for model in catboost_models:
        print(f"  - {model.name}")
        # Try to load and check the model
        try:
            with open(model, 'rb') as f:
                data = pickle.load(f)
            print(f"    ✓ Loaded successfully")
            print(f"    - Keys: {list(data.keys())[:5]}...")
            print(f"    - Has model: {'model' in data}")
            print(f"    - Has X_test: {'X_test' in data}")
            print(f"    - Model type: {type(data.get('model'))}")
        except Exception as e:
            print(f"    ❌ Error loading: {e}")
    
    print(f"\nFound {len(xgboost_models)} XGBoost models:")
    for model in xgboost_models:
        print(f"  - {model.name}")
        # Try to load and check the model
        try:
            with open(model, 'rb') as f:
                data = pickle.load(f)
            print(f"    ✓ Loaded successfully")
            print(f"    - Keys: {list(data.keys())[:5]}...")
            print(f"    - Has model: {'model' in data}")
            print(f"    - Has X_test: {'X_test' in data}")
            print(f"    - Model type: {type(data.get('model'))}")
        except Exception as e:
            print(f"    ❌ Error loading: {e}")

if __name__ == "__main__":
    check_models()