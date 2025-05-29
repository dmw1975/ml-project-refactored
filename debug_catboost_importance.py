#!/usr/bin/env python3
"""Debug CatBoost feature importance structure."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils import io
from config import settings

def main():
    """Check CatBoost feature importance structure."""
    # Load CatBoost models
    models = io.load_model('catboost_models.pkl', settings.MODEL_DIR)
    
    print("Checking CatBoost feature importance structure...\n")
    
    for model_name, model_data in models.items():
        print(f"{model_name}:")
        
        if 'feature_importance' in model_data:
            fi = model_data['feature_importance']
            print(f"  Type: {type(fi)}")
            
            if hasattr(fi, 'shape'):
                print(f"  Shape: {fi.shape}")
            
            if hasattr(fi, 'columns'):
                print(f"  Columns: {list(fi.columns)}")
            
            if hasattr(fi, 'head'):
                print("  First few rows:")
                print(fi.head())
        else:
            print("  No feature_importance field")
        
        print()

if __name__ == "__main__":
    main()