#!/usr/bin/env python3
"""Check the structure of loaded models."""

import pickle
from pathlib import Path

def check_structure():
    """Check structure of model files."""
    models_dir = Path("outputs/models")
    
    # Check linear regression models
    lr_file = models_dir / "linear_regression_models.pkl"
    if lr_file.exists():
        with open(lr_file, 'rb') as f:
            lr_models = pickle.load(f)
        
        print("Linear Regression Models Structure:")
        print(f"Type: {type(lr_models)}")
        if isinstance(lr_models, dict):
            print(f"Keys: {list(lr_models.keys())}")
            # Check structure of first model
            first_key = list(lr_models.keys())[0]
            first_model = lr_models[first_key]
            print(f"\nFirst model ({first_key}) keys:")
            for k in sorted(first_model.keys()):
                print(f"  - {k}: {type(first_model[k])}")

if __name__ == "__main__":
    check_structure()