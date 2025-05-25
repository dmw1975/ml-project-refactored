#!/usr/bin/env python3
"""Fix model_type field in consolidated model files."""

import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import settings


def fix_model_types():
    """Add model_type field to all models that are missing it."""
    model_dir = settings.MODEL_DIR
    
    # Fix XGBoost models
    xgb_file = model_dir / 'xgboost_models.pkl'
    if xgb_file.exists():
        print("Fixing XGBoost models...")
        with open(xgb_file, 'rb') as f:
            models = pickle.load(f)
        
        updated = 0
        for name, model_data in models.items():
            if isinstance(model_data, dict):
                if 'model_type' not in model_data or not model_data['model_type']:
                    model_data['model_type'] = 'xgboost'
                    updated += 1
                # Also ensure model_name is set
                if 'model_name' not in model_data:
                    model_data['model_name'] = name
        
        with open(xgb_file, 'wb') as f:
            pickle.dump(models, f)
        print(f"✓ Updated {updated} XGBoost models")
    
    # Fix LightGBM models
    lgb_file = model_dir / 'lightgbm_models.pkl'
    if lgb_file.exists():
        print("Fixing LightGBM models...")
        with open(lgb_file, 'rb') as f:
            models = pickle.load(f)
        
        updated = 0
        for name, model_data in models.items():
            if isinstance(model_data, dict):
                if 'model_type' not in model_data or not model_data['model_type']:
                    model_data['model_type'] = 'lightgbm'
                    updated += 1
                # Also ensure model_name is set
                if 'model_name' not in model_data:
                    model_data['model_name'] = name
        
        with open(lgb_file, 'wb') as f:
            pickle.dump(models, f)
        print(f"✓ Updated {updated} LightGBM models")
    
    # Fix CatBoost models
    cb_file = model_dir / 'catboost_models.pkl'
    if cb_file.exists():
        print("Fixing CatBoost models...")
        with open(cb_file, 'rb') as f:
            models = pickle.load(f)
        
        updated = 0
        for name, model_data in models.items():
            if isinstance(model_data, dict):
                if 'model_type' not in model_data or not model_data['model_type']:
                    model_data['model_type'] = 'catboost'
                    updated += 1
                # Also ensure model_name is set
                if 'model_name' not in model_data:
                    model_data['model_name'] = name
                # Ensure required fields for CatBoost adapter
                if 'y_pred' not in model_data and 'y_test_pred' in model_data:
                    model_data['y_pred'] = model_data['y_test_pred']
        
        with open(cb_file, 'wb') as f:
            pickle.dump(models, f)
        print(f"✓ Updated {updated} CatBoost models")
    
    print("\n✅ Model type fixes completed!")


if __name__ == "__main__":
    fix_model_types()