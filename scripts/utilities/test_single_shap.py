#!/usr/bin/env python3
"""Test SHAP computation for a single model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from scripts.utilities.compute_and_visualize_shap import (
    compute_shap_values, generate_shap_visualizations, setup_visualization_style
)
from src.utils.io import load_model
from src.config import settings

# Create PATHS dictionary from settings
PATHS = {
    'outputs': {
        'models': str(settings.MODEL_DIR),
        'visualizations': str(settings.VISUALIZATION_DIR),
        'shap': str(settings.OUTPUT_DIR / 'shap')
    }
}

def test_single_model():
    """Test SHAP computation for CatBoost_Base_categorical_basic."""
    setup_visualization_style()
    
    model_name = "CatBoost_Base_categorical_basic"
    print(f"Testing SHAP for {model_name}")
    
    # Load model
    model_data = load_model(f"{model_name}.pkl", PATHS['outputs']['models'])
    if model_data is None:
        print("Failed to load model")
        return
    
    # Ensure model_name is set
    model_data['model_name'] = model_name
    
    # Compute SHAP values
    shap_values, X_sample = compute_shap_values(model_data, max_samples=30)
    
    if shap_values is None:
        print("Failed to compute SHAP values")
        return
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"X_sample shape: {X_sample.shape}")
    
    # Generate visualizations
    success = generate_shap_visualizations(model_name, model_data, shap_values, X_sample)
    
    if success:
        print("Successfully generated SHAP visualizations!")
    else:
        print("Failed to generate visualizations")

if __name__ == "__main__":
    test_single_model()