#!/usr/bin/env python3
"""Generate SHAP plots for CatBoost models that are missing them."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings
from scripts.utilities.generate_shap_visualizations import (
    load_grouped_models, compute_shap_for_model, create_shap_plots
)
from src.visualization.core.style import setup_visualization_style


def generate_missing_catboost_shap():
    """Generate SHAP plots for CatBoost models with missing plots."""
    setup_visualization_style()
    
    # Load models
    all_models = load_grouped_models()
    
    # List of CatBoost models that need SHAP plots
    models_to_process = [
        "CatBoost_Base_Random_categorical_basic",
        "CatBoost_Base_Random_categorical_optuna",
        "CatBoost_Yeo_Random_categorical_basic",
        "CatBoost_Yeo_Random_categorical_optuna",
        "CatBoost_Yeo_categorical_basic",
        "CatBoost_Yeo_categorical_optuna"
    ]
    
    print(f"Generating SHAP plots for {len(models_to_process)} CatBoost models...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, model_name in enumerate(models_to_process, 1):
        print(f"\n[{i}/{len(models_to_process)}] Processing: {model_name}")
        print("-" * 40)
        
        if model_name not in all_models:
            print(f"  ✗ Model not found!")
            failed += 1
            continue
        
        model_data = all_models[model_name]
        
        # Check if plots already exist
        shap_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
        if shap_dir.exists():
            existing_plots = list(shap_dir.glob("*.png"))
            if len(existing_plots) > 0:
                print(f"  ℹ {len(existing_plots)} plots already exist, skipping...")
                successful += 1
                continue
        
        # Compute SHAP values
        print("  Computing SHAP values...")
        shap_values, X_sample = compute_shap_for_model(model_name, model_data, max_samples=50)
        
        if shap_values is None:
            print("  ✗ Failed to compute SHAP values!")
            failed += 1
            continue
        
        # Generate plots
        print("  Generating SHAP plots...")
        success = create_shap_plots(model_name, model_data, shap_values, X_sample)
        
        if success:
            successful += 1
            print(f"  ✓ Successfully generated SHAP plots")
        else:
            failed += 1
            print(f"  ✗ Failed to generate SHAP plots")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SHAP Generation Summary:")
    print(f"  Total models: {len(models_to_process)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")
    
    # Verify all CatBoost folders now have plots
    print("\nVerifying all CatBoost models have SHAP plots:")
    catboost_models = {name: data for name, data in all_models.items() if "CatBoost" in name}
    
    for model_name in sorted(catboost_models.keys()):
        shap_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
        if shap_dir.exists():
            plot_count = len(list(shap_dir.glob("*.png")))
            status = "✓" if plot_count > 0 else "✗"
            print(f"  {status} {model_name}: {plot_count} plots")
        else:
            print(f"  ✗ {model_name}: No directory")
    
    return successful == len(models_to_process)


if __name__ == "__main__":
    success = generate_missing_catboost_shap()
    sys.exit(0 if success else 1)