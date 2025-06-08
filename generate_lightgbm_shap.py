#!/usr/bin/env python3
"""Generate SHAP visualizations for LightGBM models only."""

import sys
from pathlib import Path
import pickle
import numpy as np
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.plots.shap_plots import generate_shap_plots
from src.visualization.core.style import setup_visualization_style

warnings.filterwarnings('ignore')

def main():
    """Generate SHAP visualizations for LightGBM models."""
    print("=" * 80)
    print("GENERATING LIGHTGBM SHAP VISUALIZATIONS")
    print("=" * 80)
    
    setup_visualization_style()
    
    # Load LightGBM models
    model_file = settings.MODEL_DIR / "lightgbm_models.pkl"
    if not model_file.exists():
        print(f"LightGBM models not found: {model_file}")
        return
    
    with open(model_file, 'rb') as f:
        lightgbm_models = pickle.load(f)
    
    print(f"\nFound {len(lightgbm_models)} LightGBM models")
    
    # Process each model
    success_count = 0
    for model_name, model_data in lightgbm_models.items():
        print(f"\nProcessing {model_name}...")
        
        # Check if SHAP already exists
        shap_dir = settings.VISUALIZATION_DIR / "SHAP" / model_name
        if shap_dir.exists() and len(list(shap_dir.glob("*.png"))) > 0:
            print(f"  ✓ SHAP visualizations already exist, skipping")
            success_count += 1
            continue
        
        try:
            # Create SHAP directory
            shap_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate SHAP plots
            plots = generate_shap_plots(model_data, model_name)
            
            if plots:
                print(f"  ✓ Generated {len(plots)} SHAP plots")
                success_count += 1
            else:
                print(f"  ✗ No SHAP plots generated")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Successfully processed {success_count}/{len(lightgbm_models)} models")
    print("=" * 80)

if __name__ == "__main__":
    main()