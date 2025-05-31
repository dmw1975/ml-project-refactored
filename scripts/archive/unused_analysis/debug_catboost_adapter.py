#!/usr/bin/env python3
"""Debug CatBoost adapter feature importance issue."""

import os
import pickle
from pathlib import Path

# Import the CatBoost adapter
from visualization_new.adapters.catboost_adapter import CatBoostAdapter

def debug_catboost_models():
    """Debug CatBoost models to understand feature importance issue."""
    models_dir = Path("outputs/models")
    
    # Check for catboost_models.pkl
    catboost_file = models_dir / "catboost_models.pkl"
    
    if not catboost_file.exists():
        print(f"CatBoost models file not found: {catboost_file}")
        return
    
    print(f"Loading CatBoost models from: {catboost_file}")
    
    try:
        with open(catboost_file, 'rb') as f:
            all_models = pickle.load(f)
        
        print(f"Found {len(all_models)} CatBoost models")
        
        for model_name, model_data in list(all_models.items())[:2]:  # Check first 2 models
            print(f"\n{'='*60}")
            print(f"Debugging model: {model_name}")
            print('='*60)
            
            try:
                # Create adapter
                adapter = CatBoostAdapter(model_data)
                
                # Check what's in model_data
                print("\nModel data keys:")
                for key in sorted(model_data.keys()):
                    if key not in ['X_train', 'X_test', 'y_train', 'y_test', 'y_pred', 'model']:
                        print(f"  {key}: {type(model_data[key])}")
                
                # Check if feature_importance exists
                if 'feature_importance' in model_data:
                    fi = model_data['feature_importance']
                    print(f"\nFeature importance found:")
                    print(f"  Type: {type(fi)}")
                    if hasattr(fi, 'shape'):
                        print(f"  Shape: {fi.shape}")
                    if hasattr(fi, 'columns'):
                        print(f"  Columns: {list(fi.columns)}")
                        if isinstance(fi, pd.DataFrame):
                            print(f"  First few rows:")
                            print(fi.head())
                
                # Try to get feature importance
                print("\nCalling get_feature_importance()...")
                try:
                    fi_df = adapter.get_feature_importance()
                    print(f"Success! Shape: {fi_df.shape}")
                    print(f"Columns: {list(fi_df.columns)}")
                    print(f"First few rows:")
                    print(fi_df.head())
                except Exception as e:
                    print(f"Error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"Error processing model: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error loading models file: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import pandas as pd
    debug_catboost_models()