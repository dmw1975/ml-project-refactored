#!/usr/bin/env python3
"""Test to find the exact error in CatBoost residual plots."""

import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization_new.plots.residuals import plot_residuals
from visualization_new.utils.io import load_all_models

def main():
    """Test residual plot generation for CatBoost."""
    print("Loading all models...")
    models = load_all_models()
    
    # Test only CatBoost models
    catboost_models = {k: v for k, v in models.items() if 'catboost' in k.lower()}
    
    print(f"\nTesting {len(catboost_models)} CatBoost models:")
    
    for model_name, model_data in catboost_models.items():
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Disable saving to focus on the error
            config = {
                'save': False,
                'show': False,
                'format_for_export': False
            }
            
            fig = plot_residuals(model_data, config)
            print(f"✓ Successfully created residual plot for {model_name}")
            
        except Exception as e:
            print(f"✗ Error creating residual plot for {model_name}:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            
            # Additional debugging
            print("\nModel data keys:", list(model_data.keys()))
            if 'y_test' in model_data:
                print(f"y_test type: {type(model_data['y_test'])}")
                if hasattr(model_data['y_test'], 'shape'):
                    print(f"y_test shape: {model_data['y_test'].shape}")
            if 'y_pred' in model_data:
                print(f"y_pred type: {type(model_data['y_pred'])}")
                if hasattr(model_data['y_pred'], 'shape'):
                    print(f"y_pred shape: {model_data['y_pred'].shape}")

if __name__ == "__main__":
    main()