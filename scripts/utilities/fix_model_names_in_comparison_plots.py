#!/usr/bin/env python3
"""
Fix model names in comparison plots by ensuring all model data has proper model_name keys.
This addresses the issue where comparison plots show "unknown" for dataset names.
"""

import sys
from pathlib import Path
import pickle
import traceback

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.config import settings
from src.utils.io import load_model, save_model
from src.visualization.plots.optimization import plot_hyperparameter_comparison
from src.visualization.core.interfaces import VisualizationConfig


def fix_model_names_in_pickle(model_file):
    """
    Fix model names in a pickle file by ensuring each model has a model_name key.
    
    Args:
        model_file: Name of the pickle file (e.g., 'xgboost_models.pkl')
        
    Returns:
        bool: True if successful
    """
    try:
        # Load models
        models = load_model(model_file, settings.MODEL_DIR)
        if not models:
            print(f"No models found in {model_file}")
            return False
        
        # Fix model names
        fixed_count = 0
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                # Ensure model_name is set
                if 'model_name' not in model_data or not model_data['model_name']:
                    model_data['model_name'] = model_name
                    fixed_count += 1
                    print(f"  Fixed model_name for {model_name}")
                
                # Also ensure it matches the key
                elif model_data['model_name'] != model_name:
                    print(f"  Updated model_name from '{model_data['model_name']}' to '{model_name}'")
                    model_data['model_name'] = model_name
                    fixed_count += 1
        
        if fixed_count > 0:
            # Save the updated models
            save_model(models, model_file, settings.MODEL_DIR)
            print(f"✓ Fixed {fixed_count} model names in {model_file}")
        else:
            print(f"✓ All model names already correct in {model_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error fixing model names in {model_file}: {e}")
        traceback.print_exc()
        return False


def regenerate_comparison_plots(model_family, models):
    """
    Regenerate comparison plots for a model family.
    
    Args:
        model_family: 'xgboost', 'lightgbm', or 'catboost'
        models: Dictionary of models
    """
    print(f"\nRegenerating {model_family} comparison plots...")
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance" / model_family
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = VisualizationConfig(
        output_dir=output_dir,
        format="png",
        dpi=300,
        save=True,
        show=False
    )
    
    # Get models with best_params (Optuna models)
    optuna_models = []
    for model_name, model_data in models.items():
        if 'best_params' in model_data and model_data['best_params']:
            # Ensure model_name is set
            model_data['model_name'] = model_name
            optuna_models.append(model_data)
    
    if not optuna_models:
        print(f"  No Optuna models found for {model_family}")
        return
    
    # Generate comparison plots for common hyperparameters
    hyperparameters = {
        'xgboost': ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'colsample_bytree'],
        'lightgbm': ['learning_rate', 'num_leaves', 'n_estimators', 'feature_fraction', 'bagging_fraction'],
        'catboost': ['learning_rate', 'depth', 'iterations', 'l2_leaf_reg']
    }
    
    params_to_plot = hyperparameters.get(model_family, [])
    
    for param in params_to_plot:
        try:
            # Check if any model has this parameter
            has_param = any(param in model.get('best_params', {}) for model in optuna_models)
            
            if has_param:
                print(f"  Generating {param} comparison plot...")
                output_path = plot_hyperparameter_comparison(
                    optuna_models,
                    param,
                    config,
                    model_family
                )
                
                if output_path:
                    print(f"    ✓ Created {Path(output_path).name}")
                else:
                    print(f"    ✗ Failed to create {param} comparison")
            else:
                print(f"  Skipping {param} - not found in any model")
                
        except Exception as e:
            print(f"    ✗ Error creating {param} comparison: {e}")
            traceback.print_exc()


def main():
    """Main function to fix model names and regenerate comparison plots."""
    print("Fixing model names in comparison plots...")
    print("=" * 60)
    
    # Model files to fix
    model_files = {
        'xgboost': 'xgboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'catboost': 'catboost_models.pkl'
    }
    
    # Fix model names in each file
    print("\nStep 1: Fixing model names in pickle files...")
    for model_family, model_file in model_files.items():
        print(f"\nProcessing {model_family}...")
        if fix_model_names_in_pickle(model_file):
            # Load the fixed models
            models = load_model(model_file, settings.MODEL_DIR)
            if models:
                # Regenerate comparison plots
                print(f"\nStep 2: Regenerating comparison plots for {model_family}...")
                regenerate_comparison_plots(model_family, models)
    
    print("\n" + "=" * 60)
    print("✅ Done! Model names have been fixed and comparison plots regenerated.")
    print("\nCheck the following directories for updated plots:")
    for model_family in model_files.keys():
        plot_dir = settings.VISUALIZATION_DIR / "performance" / model_family
        if plot_dir.exists():
            comparison_plots = list(plot_dir.glob("*_best_*_comparison.png"))
            if comparison_plots:
                print(f"\n{model_family.upper()}:")
                for plot in comparison_plots:
                    print(f"  - {plot.name}")


if __name__ == "__main__":
    main()