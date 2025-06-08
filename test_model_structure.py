#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test to check the structure of CatBoost and XGBoost models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pickle

def check_model_structure():
    """Check model structure."""
    models_dir = Path("outputs/models")
    
    # Load CatBoost models
    catboost_file = models_dir / "catboost_models.pkl"
    if catboost_file.exists():
        print("CatBoost Models Structure:")
        print("="*60)
        with open(catboost_file, 'rb') as f:
            catboost_models = pickle.load(f)
        
        # Check one model in detail
        for model_name in list(catboost_models.keys())[:1]:  # Just check first model
            print(f"\nModel: {model_name}")
            model_data = catboost_models[model_name]
            print(f"Type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"Keys: {list(model_data.keys())}")
                print(f"Has 'model': {'model' in model_data}")
                print(f"Has 'X_test': {'X_test' in model_data}")
                print(f"Has 'y_test': {'y_test' in model_data}")
                print(f"Has 'model_name': {'model_name' in model_data}")
                
                if 'model' in model_data:
                    print(f"Model type: {type(model_data['model'])}")
                if 'X_test' in model_data:
                    print(f"X_test type: {type(model_data['X_test'])}")
                    if hasattr(model_data['X_test'], 'shape'):
                        print(f"X_test shape: {model_data['X_test'].shape}")
    
    # Load XGBoost models
    xgboost_file = models_dir / "xgboost_models.pkl"
    if xgboost_file.exists():
        print("\n\nXGBoost Models Structure:")
        print("="*60)
        with open(xgboost_file, 'rb') as f:
            xgboost_models = pickle.load(f)
        
        # Check one model in detail
        for model_name in list(xgboost_models.keys())[:1]:  # Just check first model
            print(f"\nModel: {model_name}")
            model_data = xgboost_models[model_name]
            print(f"Type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"Keys: {list(model_data.keys())}")
                print(f"Has 'model': {'model' in model_data}")
                print(f"Has 'X_test': {'X_test' in model_data}")
                print(f"Has 'y_test': {'y_test' in model_data}")
                print(f"Has 'model_name': {'model_name' in model_data}")
                
                if 'model' in model_data:
                    print(f"Model type: {type(model_data['model'])}")
                if 'X_test' in model_data:
                    print(f"X_test type: {type(model_data['X_test'])}")
                    if hasattr(model_data['X_test'], 'shape'):
                        print(f"X_test shape: {model_data['X_test'].shape}")
    
    # Compare with LightGBM structure
    lightgbm_file = models_dir / "lightgbm_models.pkl"
    if lightgbm_file.exists():
        print("\n\nLightGBM Models Structure (for comparison):")
        print("="*60)
        with open(lightgbm_file, 'rb') as f:
            lightgbm_models = pickle.load(f)
        
        # Check one model in detail
        for model_name in list(lightgbm_models.keys())[:1]:  # Just check first model
            print(f"\nModel: {model_name}")
            model_data = lightgbm_models[model_name]
            print(f"Type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"Keys: {list(model_data.keys())}")
                print(f"Has 'model': {'model' in model_data}")
                print(f"Has 'X_test': {'X_test' in model_data}")
                print(f"Has 'y_test': {'y_test' in model_data}")
                print(f"Has 'model_name': {'model_name' in model_data}")

if __name__ == "__main__":
    check_model_structure()