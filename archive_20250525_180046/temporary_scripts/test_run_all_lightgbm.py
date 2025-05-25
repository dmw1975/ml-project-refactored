"""Test script to run LightGBM training and visualization."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

if __name__ == "__main__":
    # First, ensure output directories exist
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)
    os.makedirs("outputs/feature_importance", exist_ok=True)
    
    # Train LightGBM models
    print("Training LightGBM models...")
    from models.lightgbm_model import train_lightgbm_models
    lightgbm_models = train_lightgbm_models(n_trials=10)  # Use fewer trials for faster execution
    
    # Generate feature importance for all LightGBM models
    print("\nGenerating feature importance for all LightGBM models...")
    from test_lightgbm_importance import test_lightgbm_importance
    importance_results = test_lightgbm_importance()
    
    # Generate visualizations
    print("\nGenerating LightGBM visualizations...")
    from visualization.lightgbm_plots import visualize_lightgbm_models
    visualize_lightgbm_models()
    
    # Generate model comparison
    print("\nComparing all models...")
    from test_model_comparison import compare_all_models
    compare_all_models()
    
    print("\nAll LightGBM tasks completed successfully!")