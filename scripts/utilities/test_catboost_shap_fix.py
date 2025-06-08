#!/usr/bin/env python3
"""Test CatBoost SHAP fix on a single problematic model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings
from scripts.utilities.generate_shap_visualizations import (
    load_grouped_models, compute_shap_for_model, create_shap_plots
)
from src.visualization.core.style import setup_visualization_style


def test_catboost_shap():
    """Test SHAP generation for a problematic CatBoost model."""
    setup_visualization_style()
    
    # Load models
    all_models = load_grouped_models()
    
    # Test with a problematic model
    test_model = "CatBoost_Base_Random_categorical_basic"
    
    print(f"Testing SHAP generation for: {test_model}")
    print("=" * 60)
    
    if test_model not in all_models:
        print(f"Model {test_model} not found!")
        return False
    
    model_data = all_models[test_model]
    
    # Compute SHAP values
    print("\nComputing SHAP values...")
    shap_values, X_sample = compute_shap_for_model(test_model, model_data, max_samples=50)
    
    if shap_values is None:
        print("Failed to compute SHAP values!")
        return False
    
    print(f"✓ SHAP values computed successfully!")
    print(f"  Shape: {shap_values.shape}")
    
    # Generate plots
    print("\nGenerating SHAP plots...")
    success = create_shap_plots(test_model, model_data, shap_values, X_sample)
    
    if success:
        print(f"\n✓ Successfully generated SHAP plots for {test_model}")
        
        # Check created files
        shap_dir = settings.VISUALIZATION_DIR / 'shap' / test_model
        if shap_dir.exists():
            plots = list(shap_dir.glob("*.png"))
            print(f"\nCreated {len(plots)} plots:")
            for plot in plots:
                print(f"  - {plot.name}")
        
        return True
    else:
        print(f"\n✗ Failed to generate SHAP plots for {test_model}")
        return False


if __name__ == "__main__":
    success = test_catboost_shap()
    sys.exit(0 if success else 1)