#!/usr/bin/env python3
"""Fix residual plots for XGBoost and CatBoost models."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization_new.plots.residuals import plot_residuals
from visualization_new.utils.io import load_all_models
from visualization_new.core.registry import get_adapter_for_model
from config import settings

def main():
    """Generate missing residual plots."""
    print("Loading all models...")
    models = load_all_models()
    
    # Filter to only XGBoost and CatBoost models
    target_models = {k: v for k, v in models.items() 
                     if 'xgboost' in k.lower() or 'catboost' in k.lower()}
    
    print(f"\nGenerating residual plots for {len(target_models)} models...")
    
    output_dir = settings.VISUALIZATION_DIR / 'residuals'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for model_name, model_data in target_models.items():
        try:
            print(f"\nProcessing {model_name}...")
            
            # Wrap model data in adapter
            adapter = get_adapter_for_model(model_data)
            
            # Create plot
            fig = plot_residuals(adapter, config={'output_dir': output_dir})
            
            # Save plot
            filename = f"{model_name.lower().replace(' ', '_')}_residuals"
            fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            print(f"  Saved {filename}.png")
            
            # Close figure to avoid memory issues
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()