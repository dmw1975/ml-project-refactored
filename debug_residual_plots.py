#!/usr/bin/env python3
"""Debug why XGBoost and CatBoost residual plots are missing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization_new.plots.residuals import plot_all_residuals
from visualization_new.utils.io import load_all_models

def main():
    """Debug residual plot generation."""
    print("Loading all models...")
    models = load_all_models()
    
    print(f"\nTotal models loaded: {len(models)}")
    for model_name in sorted(models.keys()):
        print(f"  - {model_name}")
    
    print("\nGenerating residual plots...")
    figures = plot_all_residuals(models)
    
    print(f"\nTotal residual plots generated: {len(figures)}")
    
    # Check which models had errors
    print("\nChecking model data structure...")
    for model_name, model_data in models.items():
        if 'xgboost' in model_name.lower() or 'catboost' in model_name.lower():
            print(f"\n{model_name}:")
            print(f"  Keys: {list(model_data.keys())}")
            print(f"  Has predictions: {'predictions' in model_data}")
            print(f"  Has y_test: {'y_test' in model_data}")
            if 'predictions' in model_data:
                print(f"  Predictions shape: {model_data['predictions'].shape if hasattr(model_data['predictions'], 'shape') else len(model_data['predictions'])}")
            if 'y_test' in model_data:
                print(f"  y_test shape: {model_data['y_test'].shape if hasattr(model_data['y_test'], 'shape') else len(model_data['y_test'])}")

if __name__ == "__main__":
    main()