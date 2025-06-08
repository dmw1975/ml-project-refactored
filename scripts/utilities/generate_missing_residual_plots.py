#!/usr/bin/env python3
"""
Generate missing residual plots for CatBoost and XGBoost models.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.io import load_all_models
from src.config import settings
from src.visualization.viz_factory import create_residual_plot
from src.visualization.core.interfaces import VisualizationConfig


def check_existing_residual_plots():
    """Check which residual plots already exist."""
    residuals_dir = settings.VISUALIZATION_DIR / "residuals"
    if not residuals_dir.exists():
        return []
    
    existing_plots = []
    for plot_file in residuals_dir.glob("*.png"):
        existing_plots.append(plot_file.stem.replace("_residuals", ""))
    
    return existing_plots


def generate_missing_residual_plots():
    """Generate residual plots for all models, especially CatBoost and XGBoost."""
    print("Checking for missing residual plots...")
    
    # Get existing plots
    existing_plots = check_existing_residual_plots()
    print(f"Found {len(existing_plots)} existing residual plots")
    
    # Load all models
    all_models = load_all_models()
    print(f"Loaded {len(all_models)} models total")
    
    # Create output directory if it doesn't exist
    output_dir = settings.VISUALIZATION_DIR / "residuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track what we generate
    generated_count = 0
    skipped_count = 0
    error_count = 0
    
    # Group models by type for summary
    model_types = {
        'catboost': [],
        'xgboost': [],
        'lightgbm': [],
        'elasticnet': [],
        'linear': []
    }
    
    # Process each model
    for model_name, model_data in all_models.items():
        # Ensure model_name is set in the data
        if 'model_name' not in model_data or model_data.get('model_name') == 'Unknown':
            model_data['model_name'] = model_name
        
        # Categorize model
        model_name_lower = model_name.lower()
        if 'catboost' in model_name_lower:
            model_types['catboost'].append(model_name)
        elif 'xgboost' in model_name_lower:
            model_types['xgboost'].append(model_name)
        elif 'lightgbm' in model_name_lower:
            model_types['lightgbm'].append(model_name)
        elif 'elasticnet' in model_name_lower:
            model_types['elasticnet'].append(model_name)
        else:
            model_types['linear'].append(model_name)
        
        # Check if plot already exists
        if model_name in existing_plots:
            print(f"  [SKIP] {model_name} - residual plot already exists")
            skipped_count += 1
            continue
        
        # Generate residual plot
        try:
            print(f"  [GENERATING] {model_name}...")
            
            # Create config for residual plot
            config = VisualizationConfig(
                output_dir=output_dir,
                save=True,
                show=False,
                format='png',
                dpi=300,
                figsize=(12, 10)
            )
            
            # Generate the plot
            fig = create_residual_plot(model_data, config)
            
            if fig is not None:
                # Close the figure to free memory
                plt.close(fig)
                generated_count += 1
                print(f"    ✓ Successfully generated residual plot")
            else:
                error_count += 1
                print(f"    ✗ Failed to generate residual plot")
                
        except Exception as e:
            error_count += 1
            print(f"    ✗ Error generating residual plot: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total models processed: {len(all_models)}")
    print(f"Plots generated: {generated_count}")
    print(f"Plots skipped (already exist): {skipped_count}")
    print(f"Errors: {error_count}")
    
    print("\nModels by type:")
    for model_type, models in model_types.items():
        print(f"  {model_type}: {len(models)} models")
        if model_type in ['catboost', 'xgboost'] and models:
            # Show which CatBoost/XGBoost models were processed
            for model in sorted(models):
                status = "✓" if model not in existing_plots else "already existed"
                print(f"    - {model} [{status}]")
    
    # Check final state
    print("\nChecking final residual plots...")
    final_plots = list((settings.VISUALIZATION_DIR / "residuals").glob("*.png"))
    print(f"Total residual plots now: {len(final_plots)}")
    
    # Remove any Unknown plots
    unknown_plot = output_dir / "Unknown_residuals.png"
    if unknown_plot.exists():
        print(f"\nRemoving Unknown residual plot...")
        unknown_plot.unlink()
        print("  ✓ Removed Unknown_residuals.png")


def verify_residual_plots():
    """Verify that all expected models have residual plots."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    residuals_dir = settings.VISUALIZATION_DIR / "residuals"
    all_models = load_all_models()
    
    # Check each model
    missing_models = []
    for model_name in all_models.keys():
        plot_path = residuals_dir / f"{model_name}_residuals.png"
        if not plot_path.exists():
            missing_models.append(model_name)
    
    if missing_models:
        print(f"WARNING: {len(missing_models)} models still missing residual plots:")
        for model in missing_models:
            print(f"  - {model}")
    else:
        print("✓ All models have residual plots!")
    
    # List all residual plots
    all_plots = sorted(residuals_dir.glob("*.png"))
    print(f"\nTotal residual plots: {len(all_plots)}")
    
    # Group by model type
    plot_counts = {
        'catboost': 0,
        'xgboost': 0,
        'lightgbm': 0,
        'elasticnet': 0,
        'linear': 0,
        'unknown': 0
    }
    
    for plot in all_plots:
        plot_name = plot.stem.lower()
        if 'catboost' in plot_name:
            plot_counts['catboost'] += 1
        elif 'xgboost' in plot_name:
            plot_counts['xgboost'] += 1
        elif 'lightgbm' in plot_name:
            plot_counts['lightgbm'] += 1
        elif 'elasticnet' in plot_name:
            plot_counts['elasticnet'] += 1
        elif plot_name.startswith('lr_'):
            plot_counts['linear'] += 1
        else:
            plot_counts['unknown'] += 1
    
    print("\nResidual plots by model type:")
    for model_type, count in plot_counts.items():
        if count > 0:
            print(f"  {model_type}: {count} plots")


def main():
    """Main function."""
    print("Generating missing residual plots for CatBoost and XGBoost models...")
    print("="*60)
    
    # Generate missing plots
    generate_missing_residual_plots()
    
    # Verify results
    verify_residual_plots()
    
    print("\nDone!")


if __name__ == "__main__":
    main()