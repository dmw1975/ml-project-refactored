#!/usr/bin/env python3
"""Test residual plot generation with fixed adapters."""

from pathlib import Path
from visualization_new.utils.io import load_all_models
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.plots.residuals import plot_residuals

def test_residual_plots():
    """Test generating residual plots."""
    print("Loading models...")
    all_models = load_all_models()
    
    # Create output directory
    viz_dir = Path("outputs/visualizations/residuals")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Test one model from each type
    test_models = [
        'LR_Base',
        'ElasticNet_LR_Base_basic',
        'XGBoost_Base_categorical_basic',
        'LightGBM_Base_categorical_basic',
        'CatBoost_Base_categorical_basic'
    ]
    
    for model_name in test_models:
        print(f"\nTesting: {model_name}")
        
        if model_name in all_models:
            try:
                model_data = all_models[model_name]
                adapter = get_adapter_for_model(model_data)
                
                output_path = viz_dir / f"{model_name}_residuals_test.png"
                plot_residuals(adapter, str(output_path))
                print(f"✓ Success! Created: {output_path}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ Model not found")

if __name__ == "__main__":
    test_residual_plots()