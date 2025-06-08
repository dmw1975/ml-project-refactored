#!/usr/bin/env python3
"""Fix CV distribution plots showing 'Unknown' model names."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.visualization.utils.io import load_all_models
from src.visualization.plots.cv_distributions import plot_cv_distributions
from src.config.settings import VISUALIZATION_DIR


def fix_model_names_in_data(models):
    """
    Ensure all models have proper model_name fields.
    
    Args:
        models: Dictionary of models
        
    Returns:
        Fixed models dictionary
    """
    fixed_models = {}
    
    for key, model_data in models.items():
        # The key itself contains the model name
        if isinstance(model_data, dict):
            # If model_name is missing or 'Unknown', use the key
            if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
                model_data['model_name'] = key
                print(f"Fixed model name for {key}: {model_data['model_name']}")
            else:
                print(f"Model {key} already has name: {model_data['model_name']}")
        
        fixed_models[key] = model_data
    
    return fixed_models


def main():
    """Regenerate CV distribution plots with proper model names."""
    print("Loading all models...")
    models = load_all_models()
    
    print(f"\nFound {len(models)} models total")
    
    # Fix model names
    print("\nFixing model names...")
    models = fix_model_names_in_data(models)
    
    # Filter models with CV data
    cv_models = []
    for key, model_data in models.items():
        if isinstance(model_data, dict):
            # Check for CV data
            if any(k in model_data for k in ['cv_scores', 'cv_fold_scores', 'cv_mean', 'cv_mse']):
                cv_models.append(model_data)
                print(f"  ✓ {key} has CV data")
            else:
                print(f"  ✗ {key} has no CV data")
    
    print(f"\nFound {len(cv_models)} models with CV data")
    
    if not cv_models:
        print("No models with CV data found!")
        return
    
    # Create CV distribution plots
    print("\nCreating CV distribution plots...")
    output_dir = VISUALIZATION_DIR / "performance" / "cv_distributions"
    
    cv_config = {
        'save': True,
        'output_dir': output_dir,
        'dpi': 300,
        'format': 'png'
    }
    
    try:
        figures = plot_cv_distributions(cv_models, cv_config)
        print(f"\nCreated {len(figures)} CV distribution plots:")
        for name, fig in figures.items():
            print(f"  - {name}")
            # Close figure to free memory
            if hasattr(fig, 'close'):
                fig.close()
    except Exception as e:
        print(f"Error creating CV distribution plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()