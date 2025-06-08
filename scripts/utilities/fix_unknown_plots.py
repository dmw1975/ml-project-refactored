#!/usr/bin/env python3
"""
Fix Unknown residual and SHAP plots by regenerating them with proper model names.
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_all_models
from src.config import settings
from src.visualization.viz_factory import create_residual_plot
from src.visualization.plots.shap_plots import create_shap_summary_plot, create_shap_dependence_plots
from src.visualization.core.interfaces import VisualizationConfig


def fix_unknown_residual_plots():
    """Fix Unknown residual plots."""
    print("\nFixing Unknown residual plots...")
    
    # Check if Unknown directory exists
    unknown_dir = settings.VISUALIZATION_DIR / "residuals" / "Unknown"
    if unknown_dir.exists():
        print(f"Found Unknown residual directory: {unknown_dir}")
        
        # Remove Unknown directory
        shutil.rmtree(unknown_dir)
        print("  Removed Unknown directory")
    
    # Load all models
    all_models = load_all_models()
    
    # Find models that might have been labeled as Unknown
    models_to_fix = []
    for model_name, model_data in all_models.items():
        if 'xgboost' in model_name.lower() or 'catboost' in model_name.lower():
            # Check if model_name is properly set
            if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
                models_to_fix.append((model_name, model_data))
    
    if models_to_fix:
        print(f"\nRegenerating residual plots for {len(models_to_fix)} models...")
        for model_name, model_data in models_to_fix:
            try:
                # Ensure model_name is set
                model_data['model_name'] = model_name
                
                # Create config
                config = VisualizationConfig(
                    output_dir=settings.VISUALIZATION_DIR / "residuals",
                    save=True,
                    show=False,
                    format='png',
                    dpi=300
                )
                
                # Generate residual plot
                create_residual_plot(model_data, config)
                print(f"  ✓ Created residual plot for {model_name}")
                
            except Exception as e:
                print(f"  ✗ Error creating residual plot for {model_name}: {e}")


def fix_unknown_shap_plots():
    """Fix Unknown SHAP plots."""
    print("\n\nFixing Unknown SHAP plots...")
    
    # Check if Unknown directory exists
    unknown_dir = settings.VISUALIZATION_DIR / "shap" / "Unknown"
    if unknown_dir.exists():
        print(f"Found Unknown SHAP directory: {unknown_dir}")
        
        # Remove Unknown directory
        shutil.rmtree(unknown_dir)
        print("  Removed Unknown directory")
    
    # Load all models
    all_models = load_all_models()
    
    # Find models that might have been labeled as Unknown
    models_to_fix = []
    for model_name, model_data in all_models.items():
        if 'xgboost' in model_name.lower() or 'catboost' in model_name.lower():
            # Check if model_name is properly set
            if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
                models_to_fix.append((model_name, model_data))
    
    if models_to_fix:
        print(f"\nRegenerating SHAP plots for {len(models_to_fix)} models...")
        for model_name, model_data in models_to_fix:
            try:
                # Ensure model_name is set
                model_data['model_name'] = model_name
                
                # Determine model type for output directory
                if 'xgboost' in model_name.lower():
                    model_type = 'xgboost'
                elif 'catboost' in model_name.lower():
                    model_type = 'catboost'
                else:
                    model_type = 'unknown'
                
                output_dir = settings.VISUALIZATION_DIR / "shap" / model_type
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create config
                config = VisualizationConfig(
                    output_dir=output_dir,
                    save=True,
                    show=False,
                    format='png',
                    dpi=300
                )
                
                # Generate SHAP summary plot
                create_shap_summary_plot(model_data, config)
                print(f"  ✓ Created SHAP summary plot for {model_name}")
                
                # Generate SHAP dependence plots (top 5 features)
                create_shap_dependence_plots(model_data, config, top_n=5)
                print(f"  ✓ Created SHAP dependence plots for {model_name}")
                
            except Exception as e:
                print(f"  ✗ Error creating SHAP plots for {model_name}: {e}")


def check_final_structure():
    """Check final visualization structure."""
    print("\n\nChecking final visualization structure...")
    
    # Check residuals
    residuals_dir = settings.VISUALIZATION_DIR / "residuals"
    if residuals_dir.exists():
        subdirs = [d for d in residuals_dir.iterdir() if d.is_dir()]
        print(f"\nResidual plot directories ({len(subdirs)}):")
        for subdir in sorted(subdirs):
            plot_count = len(list(subdir.glob("*.png")))
            print(f"  - {subdir.name}: {plot_count} plots")
    
    # Check SHAP
    shap_dir = settings.VISUALIZATION_DIR / "shap"
    if shap_dir.exists():
        subdirs = [d for d in shap_dir.iterdir() if d.is_dir()]
        print(f"\nSHAP plot directories ({len(subdirs)}):")
        for subdir in sorted(subdirs):
            plot_count = len(list(subdir.glob("*.png")))
            print(f"  - {subdir.name}: {plot_count} plots")


def main():
    """Main function."""
    print("Fixing Unknown plots...")
    
    # Note: Model names should already be fixed from previous script
    # This script just removes Unknown directories since the models now have proper names
    
    # Fix residual plots
    fix_unknown_residual_plots()
    
    # Fix SHAP plots
    fix_unknown_shap_plots()
    
    # Check final structure
    check_final_structure()
    
    print("\nDone!")


if __name__ == "__main__":
    main()