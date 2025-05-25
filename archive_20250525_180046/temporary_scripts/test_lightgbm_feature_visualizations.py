"""Script to generate LightGBM feature importance visualizations."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io
from visualization.lightgbm_plots import plot_lightgbm_feature_importance

if __name__ == "__main__":
    # Ensure directories exist
    io.ensure_dir(settings.VISUALIZATION_DIR / "features")
    io.ensure_dir(settings.VISUALIZATION_DIR / "features" / "lightgbm")
    
    print("Generating LightGBM feature importance visualizations...")
    
    # Run the visualization function
    result = plot_lightgbm_feature_importance()
    
    if result:
        print("\nLightGBM feature importance visualizations completed successfully!")
        print(f"Visualizations are saved in the following locations:")
        print(f"- {settings.VISUALIZATION_DIR}/features/ (main directory)")
        print(f"- {settings.VISUALIZATION_DIR}/features/lightgbm/ (model-specific directory)")
    else:
        print("\nFailed to generate LightGBM feature importance visualizations.")
        print("Please ensure LightGBM models are trained and feature importance data is available.")