#!/usr/bin/env python3
"""Verify that LightGBM and CatBoost data exists and is accessible."""

import sys
from pathlib import Path
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings

def verify_model_data():
    """Verify all model data exists and document locations."""
    
    print("=" * 80)
    print("MODEL DATA AVAILABILITY VERIFICATION")
    print("=" * 80)
    
    # Check model pickle files
    models_dir = settings.MODEL_DIR
    model_files = {
        'Linear Regression': 'linear_regression_models.pkl',
        'ElasticNet': 'elasticnet_models.pkl',
        'XGBoost': 'xgboost_models.pkl',
        'LightGBM': 'lightgbm_models.pkl',
        'CatBoost': 'catboost_models.pkl'
    }
    
    all_models_data = {}
    
    print("\n1. MODEL PICKLE FILES:")
    print("-" * 40)
    for model_type, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
                all_models_data[model_type] = models
                print(f"✓ {model_type}: {filepath}")
                print(f"  Contains {len(models)} models:")
                for name in models.keys():
                    print(f"    - {name}")
        else:
            print(f"✗ {model_type}: NOT FOUND at {filepath}")
    
    # Check baseline comparison data
    print("\n2. BASELINE COMPARISON DATA:")
    print("-" * 40)
    baselines_dir = settings.DATA_DIR / "baselines"
    if baselines_dir.exists():
        for file in baselines_dir.glob("*.pkl"):
            print(f"  ✓ {file.name}")
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    print(f"    Contains models: {list(data.keys())}")
    
    # Check metrics CSV files
    print("\n3. METRICS CSV FILES:")
    print("-" * 40)
    metrics_dir = settings.METRICS_DIR
    if metrics_dir.exists():
        for file in metrics_dir.glob("*.csv"):
            print(f"  ✓ {file.name}")
    
    # Check dataset comparison results
    print("\n4. DATASET COMPARISON RESULTS:")
    print("-" * 40)
    
    # Look for any stored comparison results
    for potential_dir in [settings.DATA_DIR, settings.METRICS_DIR, settings.OUTPUT_DIR]:
        for file in potential_dir.rglob("*comparison*.pkl"):
            print(f"  ✓ {file}")
    
    # Analyze data format consistency
    print("\n5. DATA FORMAT ANALYSIS:")
    print("-" * 40)
    
    # Compare data structures
    for model_type, models in all_models_data.items():
        print(f"\n{model_type}:")
        if models:
            first_model_name = list(models.keys())[0]
            first_model_data = models[first_model_name]
            if isinstance(first_model_data, dict):
                print(f"  Data type: dict")
                print(f"  Keys: {list(first_model_data.keys())[:10]}...")  # First 10 keys
                
                # Check for common required fields
                required_fields = ['model', 'y_test', 'y_pred']
                missing_fields = [f for f in required_fields if f not in first_model_data]
                if missing_fields:
                    print(f"  ⚠ Missing fields: {missing_fields}")
                else:
                    print(f"  ✓ All required fields present")
            else:
                print(f"  Data type: {type(first_model_data)}")
    
    return all_models_data

if __name__ == "__main__":
    verify_model_data()