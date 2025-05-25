#!/usr/bin/env python
"""
Test script for feature importance visualization.

This script verifies that the top_20_features_avg_importance.png file
is correctly saved only in the main features directory and not in subdirectories.
"""

import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from visualization_new.utils.io import load_all_models
from visualization_new.viz_factory import create_feature_importance_comparison
from visualization_new.core.interfaces import VisualizationConfig
from config import settings

def test_feature_importance_comparison():
    """Test the feature importance comparison visualization."""
    print("Loading all models...")
    all_models = load_all_models()

    print(f"Found {len(all_models)} models")

    # Use only tree-based models to avoid the dimension error
    model_names = []
    tree_based_models = {}
    for name, model_data in all_models.items():
        # Only use XGBoost, LightGBM, CatBoost models
        if any(prefix in name.lower() for prefix in ["xgb", "lightgbm", "catboost"]):
            tree_based_models[name] = model_data
            model_names.append(name)
            print(f"Selected model: {name}")

    # Only use 5 models to keep visualization simple
    model_list = list(tree_based_models.values())[:5]

    # Define configurations for different output directories
    configs = [
        # Main features directory
        {
            "output_dir": settings.VISUALIZATION_DIR / "features",
            "dpi": 300,
            "format": "png",
            "save": True,
            "show": False,
            "top_n": 20
        },
        # Model-specific directories
        {
            "output_dir": settings.VISUALIZATION_DIR / "features" / "catboost",
            "dpi": 300,
            "format": "png",
            "save": True,
            "show": False,
            "top_n": 20
        },
        {
            "output_dir": settings.VISUALIZATION_DIR / "features" / "lightgbm",
            "dpi": 300,
            "format": "png",
            "save": True,
            "show": False,
            "top_n": 20
        },
        {
            "output_dir": settings.VISUALIZATION_DIR / "features" / "xgboost",
            "dpi": 300,
            "format": "png",
            "save": True,
            "show": False,
            "top_n": 20
        }
    ]

    # Run comparison for each config to test redirection logic
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: Output directory = {config['output_dir']}")
        try:
            create_feature_importance_comparison(model_list, config)
            print("Visualization created successfully")
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    # Check where the files were actually saved
    print("\nChecking for top_20_features_avg_importance.png files:")
    main_file = settings.VISUALIZATION_DIR / "features" / "top_20_features_avg_importance.png"
    if main_file.exists():
        print(f"✓ Main file exists: {main_file}")
    else:
        print(f"✗ Main file missing: {main_file}")

    # Check subdirectories for top_20_features_avg_importance.png
    for subdir in ["catboost", "lightgbm", "elasticnet", "xgboost"]:
        subdir_file = settings.VISUALIZATION_DIR / "features" / subdir / "top_20_features_avg_importance.png"
        if subdir_file.exists():
            print(f"✗ Duplicate file found in subdirectory: {subdir_file}")
        else:
            print(f"✓ No duplicate of avg_importance in subdirectory: {subdir}")

    # Check for top_features_heatmap.png files
    print("\nChecking for top_features_heatmap.png files:")
    main_heatmap = settings.VISUALIZATION_DIR / "features" / "top_features_heatmap.png"
    if main_heatmap.exists():
        print(f"✓ Main heatmap exists: {main_heatmap}")
    else:
        print(f"✗ Main heatmap missing: {main_heatmap}")

    # Check subdirectories for top_features_heatmap.png
    for subdir in ["catboost", "lightgbm", "elasticnet", "xgboost"]:
        subdir_heatmap = settings.VISUALIZATION_DIR / "features" / subdir / "top_features_heatmap.png"
        if subdir_heatmap.exists():
            print(f"✗ Duplicate heatmap found in subdirectory: {subdir_heatmap}")
        else:
            print(f"✓ No duplicate heatmap in subdirectory: {subdir}")

if __name__ == "__main__":
    test_feature_importance_comparison()