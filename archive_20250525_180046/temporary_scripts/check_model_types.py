#!/usr/bin/env python3
"""Check which models are missing model_type field."""

import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import settings


def check_all_models():
    """Check all models for missing model_type."""
    model_dir = settings.MODEL_DIR
    
    files = [
        'linear_regression_models.pkl',
        'elasticnet_models.pkl',
        'xgboost_models.pkl',
        'lightgbm_models.pkl',
        'catboost_models.pkl'
    ]
    
    for filename in files:
        filepath = model_dir / filename
        if filepath.exists():
            print(f"\n{filename}:")
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
            
            for name, model_data in models.items():
                if isinstance(model_data, dict):
                    model_type = model_data.get('model_type', 'MISSING')
                    model_name = model_data.get('model_name', 'MISSING')
                    has_model = 'model' in model_data
                    has_y_pred = 'y_pred' in model_data or 'y_test_pred' in model_data
                    
                    if model_type == 'MISSING' or not has_model or not has_y_pred:
                        print(f"  {name}:")
                        print(f"    model_type: {model_type}")
                        print(f"    model_name: {model_name}")
                        print(f"    has model: {has_model}")
                        print(f"    has predictions: {has_y_pred}")
                        print(f"    keys: {list(model_data.keys())[:10]}")
                else:
                    print(f"  {name}: Not a dict! Type: {type(model_data)}")


if __name__ == "__main__":
    check_all_models()