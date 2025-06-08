#!/usr/bin/env python3
"""
Fix XGBoost feature importance plot names.
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_model
from src.config import settings
from src.visualization.plots.features import plot_feature_importance
from src.visualization.core.interfaces import VisualizationConfig


def main():
    """Fix XGBoost plot names by regenerating with proper names."""
    print("Fixing XGBoost feature importance plot names...")
    
    # Load XGBoost models
    xgb_models = load_model("xgboost_models.pkl", settings.MODEL_DIR)
    
    if not xgb_models:
        print("No XGBoost models found!")
        return
    
    # Output directory
    output_dir = settings.VISUALIZATION_DIR / "features" / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old Unknown plot
    unknown_plot = output_dir / "Unknown_top_features.png"
    if unknown_plot.exists():
        print(f"Removing {unknown_plot}")
        unknown_plot.unlink()
    
    # Generate plots with proper names
    success_count = 0
    for model_name, model_data in xgb_models.items():
        try:
            # Add model_name to the data
            model_data['model_name'] = model_name
            
            # Create config
            config = VisualizationConfig(
                output_dir=output_dir,
                save=True,
                show=False,
                format='png',
                dpi=300,
                top_n=15,
                show_error=True,
                show_values=True,
                grid=True,
                figsize=(10, 8)
            )
            
            # Generate plot
            fig = plot_feature_importance(model_data, config)
            success_count += 1
            print(f"  ✓ Created plot for {model_name}")
            
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {e}")
    
    print(f"\nSuccessfully created {success_count}/{len(xgb_models)} plots")
    
    # List final plots
    print("\nFinal XGBoost plots:")
    for plot in sorted(output_dir.glob("*.png")):
        print(f"  - {plot.name}")


if __name__ == "__main__":
    main()