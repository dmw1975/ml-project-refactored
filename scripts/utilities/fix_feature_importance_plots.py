#!/usr/bin/env python3
"""
Fix feature importance plot generation issues.
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_all_models
from src.visualization.plots.features import plot_feature_importance
from src.visualization.core.interfaces import VisualizationConfig
from src.config import settings


def clean_redundant_folders():
    """Remove redundant nested folders."""
    features_dir = settings.VISUALIZATION_DIR / "features"
    
    print("Cleaning redundant folders...")
    
    # Fix redundant folders
    redundant_paths = [
        features_dir / "elasticnet" / "elasticnet",
        features_dir / "lightgbm" / "lightgbm", 
        features_dir / "linear" / "linear"
    ]
    
    for redundant_path in redundant_paths:
        if redundant_path.exists():
            parent_dir = redundant_path.parent
            print(f"  Moving files from {redundant_path} to {parent_dir}")
            
            # Move all files up one level
            for file in redundant_path.glob("*.png"):
                target = parent_dir / file.name
                shutil.move(str(file), str(target))
                print(f"    Moved {file.name}")
            
            # Remove empty directory
            redundant_path.rmdir()
            print(f"    Removed empty directory: {redundant_path}")


def generate_missing_plots():
    """Generate missing feature importance plots for all models."""
    print("\nGenerating feature importance plots for all models...")
    
    # Load all models
    all_models = load_all_models()
    
    # Group models by type
    model_groups = {
        'catboost': {},
        'lightgbm': {},
        'xgboost': {},
        'elasticnet': {},
        'linear': {}
    }
    
    # Categorize models
    for name, data in all_models.items():
        name_lower = name.lower()
        if 'catboost' in name_lower:
            model_groups['catboost'][name] = data
        elif 'lightgbm' in name_lower:
            model_groups['lightgbm'][name] = data
        elif 'xgboost' in name_lower or 'xgb' in name_lower:
            model_groups['xgboost'][name] = data
        elif 'elasticnet' in name_lower:
            model_groups['elasticnet'][name] = data
        elif name_lower.startswith('lr_'):
            model_groups['linear'][name] = data
    
    # Summary
    print("\nModel summary:")
    for model_type, models in model_groups.items():
        print(f"  {model_type}: {len(models)} models")
    
    # Generate plots for each model type
    for model_type, models in model_groups.items():
        if not models:
            continue
            
        print(f"\nProcessing {model_type} models...")
        
        # Create output directory (only one level deep)
        output_dir = settings.VISUALIZATION_DIR / "features" / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots
        success_count = 0
        for model_name, model_data in models.items():
            try:
                # Debug info
                model_obj = model_data.get('model')
                has_feature_importance = 'feature_importance' in model_data
                has_feature_names = 'feature_names' in model_data
                
                print(f"  Processing {model_name}:")
                print(f"    - Has model object: {model_obj is not None}")
                print(f"    - Has feature_importance: {has_feature_importance}")
                print(f"    - Has feature_names: {has_feature_names}")
                
                if model_obj is not None:
                    if hasattr(model_obj, 'feature_importances_'):
                        print(f"    - Model has feature_importances_: True")
                    elif hasattr(model_obj, 'get_feature_importance'):
                        print(f"    - Model has get_feature_importance(): True")
                
                # Create config with proper output directory
                config = VisualizationConfig(
                    output_dir=output_dir,  # This is the only place we set output_dir
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
                print(f"    ✓ Created plot for {model_name}")
                
            except Exception as e:
                print(f"    ✗ Error with {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  Successfully created {success_count}/{len(models)} plots")


def main():
    """Main function."""
    # Clean up redundant folders
    clean_redundant_folders()
    
    # Generate all feature importance plots
    generate_missing_plots()
    
    # List final structure
    print("\nFinal directory structure:")
    features_dir = settings.VISUALIZATION_DIR / "features"
    for model_type_dir in sorted(features_dir.iterdir()):
        if model_type_dir.is_dir():
            plots = list(model_type_dir.glob("*.png"))
            print(f"  {model_type_dir.name}/: {len(plots)} plots")
            if len(plots) <= 3:
                for plot in sorted(plots):
                    print(f"    - {plot.name}")
            else:
                for plot in sorted(plots)[:3]:
                    print(f"    - {plot.name}")
                print(f"    ... and {len(plots) - 3} more")


if __name__ == "__main__":
    main()