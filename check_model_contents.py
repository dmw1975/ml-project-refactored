#!/usr/bin/env python3
"""Check contents of CatBoost and LightGBM model files."""

import pickle
import os
from pathlib import Path
import json

def check_model_file(filepath):
    """Check contents of a model pickle file."""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print('='*60)
    
    if not os.path.exists(filepath):
        print(f"ERROR: File does not exist!")
        return
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type of loaded data: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys in dictionary: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n--- Model: {key} ---")
                
                if isinstance(value, dict):
                    print(f"  Model dict keys: {list(value.keys())}")
                    
                    # Check for model object
                    if 'model' in value:
                        model = value['model']
                        print(f"  Model type: {type(model)}")
                        print(f"  Model class: {model.__class__.__name__ if hasattr(model, '__class__') else 'N/A'}")
                    
                    # Check for test data
                    if 'test_data' in value:
                        test_data = value['test_data']
                        print(f"  Test data present: Yes")
                        if isinstance(test_data, dict):
                            print(f"  Test data keys: {list(test_data.keys())}")
                            if 'X' in test_data:
                                print(f"  X_test shape: {test_data['X'].shape if hasattr(test_data['X'], 'shape') else 'N/A'}")
                            if 'y' in test_data:
                                print(f"  y_test shape: {test_data['y'].shape if hasattr(test_data['y'], 'shape') else 'N/A'}")
                    else:
                        print(f"  Test data present: No")
                    
                    # Check for metrics
                    if 'metrics' in value:
                        metrics = value['metrics']
                        print(f"  Metrics present: Yes")
                        if isinstance(metrics, dict):
                            print(f"  Metric keys: {list(metrics.keys())}")
                            for metric_key, metric_value in metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    print(f"    {metric_key}: {metric_value:.4f}")
                    else:
                        print(f"  Metrics present: No")
                    
                    # Check for CV scores
                    if 'cv_scores' in value:
                        cv_scores = value['cv_scores']
                        print(f"  CV scores present: Yes")
                        if hasattr(cv_scores, '__len__'):
                            print(f"  Number of folds: {len(cv_scores)}")
                            print(f"  CV scores type: {type(cv_scores)}")
                    else:
                        print(f"  CV scores present: No")
                    
                    # Check for CV models
                    if 'cv_models' in value:
                        cv_models = value['cv_models']
                        print(f"  CV models present: Yes")
                        if hasattr(cv_models, '__len__'):
                            print(f"  Number of CV models: {len(cv_models)}")
                    else:
                        print(f"  CV models present: No")
                        
                else:
                    print(f"  Value type: {type(value)}")
                    
    except Exception as e:
        print(f"ERROR loading file: {e}")
        import traceback
        traceback.print_exc()

# Check CatBoost and LightGBM model files
model_dir = Path("/mnt/d/ml_project_refactored/outputs/models")

print("Checking CatBoost and LightGBM model files...")

check_model_file(model_dir / "catboost_models.pkl")
check_model_file(model_dir / "lightgbm_models.pkl")

# Also check XGBoost for comparison
print("\n\nFor comparison, checking XGBoost (which works):")
check_model_file(model_dir / "xgboost_models.pkl")