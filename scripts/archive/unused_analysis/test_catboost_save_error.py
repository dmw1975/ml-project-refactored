#!/usr/bin/env python3
"""Test to find the exact error when saving CatBoost residual plots."""

import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization_new.plots.residuals import plot_residuals
from visualization_new.utils.io import load_all_models

def main():
    """Test residual plot generation and saving for CatBoost."""
    print("Loading all models...")
    models = load_all_models()
    
    # Test only CatBoost models
    catboost_models = {k: v for k, v in models.items() if 'catboost' in k.lower()}
    
    print(f"\nTesting saving for {len(catboost_models)} CatBoost models:")
    
    for model_name, model_data in catboost_models.items():
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Enable saving and format_for_export to test the full pipeline
            config = {
                'save': True,
                'show': False,
                'format_for_export': True,
                'output_dir': Path('./test_outputs/residuals')
            }
            
            fig = plot_residuals(model_data, config)
            print(f"✓ Successfully created and saved residual plot for {model_name}")
            
        except Exception as e:
            print(f"✗ Error creating/saving residual plot for {model_name}:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    main()