#!/usr/bin/env python3
"""Generate only the model comparison SHAP plot."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.plots.shap_plots import create_model_comparison_shap_plot
from scripts.utilities.generate_shap_visualizations import load_grouped_models
from src.visualization.core.style import setup_visualization_style

def main():
    """Generate model comparison SHAP plot."""
    print("Generating model comparison SHAP plot...")
    
    # Set plot style
    setup_visualization_style()
    
    # Load models
    all_models = load_grouped_models()
    
    # Filter for tree models
    tree_models = {
        name: data for name, data in all_models.items() 
        if any(model_type in name for model_type in ["CatBoost", "LightGBM", "XGBoost"])
    }
    
    print(f"Found {len(tree_models)} tree models")
    
    # Create output directory
    shap_dir = settings.VISUALIZATION_DIR / 'shap'
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plot
    try:
        comparison_path = create_model_comparison_shap_plot(tree_models, shap_dir)
        if comparison_path and comparison_path.exists():
            print(f"✓ Successfully created: {comparison_path}")
            return True
        else:
            print("✗ Failed to create comparison plot")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)