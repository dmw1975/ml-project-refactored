#!/usr/bin/env python3
"""
Debug XGBoost feature importance issue.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_model
from src.config import settings


def main():
    """Debug XGBoost feature importance."""
    print("Loading XGBoost models to debug feature importance...")
    
    # Load XGBoost models
    xgb_models = load_model("xgboost_models.pkl", settings.MODEL_DIR)
    
    if not xgb_models:
        print("No XGBoost models found!")
        return
    
    # Check first model
    first_model_name = list(xgb_models.keys())[0]
    first_model_data = xgb_models[first_model_name]
    
    print(f"\nChecking model: {first_model_name}")
    print(f"Model data keys: {list(first_model_data.keys())}")
    
    # Check all model names
    print("\nAll XGBoost model names:")
    for name in xgb_models.keys():
        print(f"  - {name}")
    
    # Check feature importance
    if 'feature_importance' in first_model_data:
        fi = first_model_data['feature_importance']
        print(f"\nFeature importance type: {type(fi)}")
        
        if isinstance(fi, pd.DataFrame):
            print(f"DataFrame shape: {fi.shape}")
            print(f"DataFrame columns: {list(fi.columns)}")
            print(f"First 5 rows:")
            print(fi.head())
        else:
            print(f"Feature importance is not a DataFrame: {fi}")
    else:
        print("No 'feature_importance' key found")
    
    # Check model object
    model = first_model_data.get('model')
    if model:
        print(f"\nModel type: {type(model)}")
        if hasattr(model, 'feature_importances_'):
            print(f"Model has feature_importances_: shape={model.feature_importances_.shape}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Model has feature_names_in_: {len(model.feature_names_in_)} features")
            print(f"First 5 feature names: {list(model.feature_names_in_[:5])}")
    
    # Check feature names
    if 'feature_names' in first_model_data:
        fn = first_model_data['feature_names']
        print(f"\nFeature names type: {type(fn)}")
        print(f"Number of features: {len(fn) if hasattr(fn, '__len__') else 'N/A'}")
        if isinstance(fn, list) and len(fn) > 0:
            print(f"First 5 feature names: {fn[:5]}")


if __name__ == "__main__":
    main()