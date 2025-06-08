#!/usr/bin/env python3
"""Generate missing SHAP plots for CatBoost and LightGBM models."""

import os
import sys
import pickle
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.visualization.plots.shap_plots import plot_shap_summary, plot_shap_waterfall, plot_shap_dependence, plot_shap_categorical
from src.visualization.core.style import set_plot_style
from src.config.settings import PATHS


def get_model_list():
    """Get list of all CatBoost and LightGBM models."""
    models_dir = Path(PATHS['outputs']['models'])
    
    catboost_models = []
    lightgbm_models = []
    
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        if "CatBoost" in model_name:
            catboost_models.append(model_name)
        elif "LightGBM" in model_name:
            lightgbm_models.append(model_name)
    
    return catboost_models, lightgbm_models


def check_shap_values_exist(model_name):
    """Check if SHAP values exist for a model."""
    shap_file = Path(PATHS['outputs']['shap']) / f"{model_name}_shap_values.pkl"
    return shap_file.exists()


def load_shap_data(model_name):
    """Load SHAP values and related data."""
    shap_file = Path(PATHS['outputs']['shap']) / f"{model_name}_shap_values.pkl"
    
    if not shap_file.exists():
        return None, None
    
    with open(shap_file, 'rb') as f:
        shap_data = pickle.load(f)
    
    # Extract components
    shap_values = shap_data['shap_values']
    X_test = shap_data['X_test']
    
    return shap_values, X_test


def generate_shap_plots_for_model(model_name, shap_values, X_test, max_samples=100):
    """Generate all SHAP plots for a single model."""
    print(f"\n  Generating SHAP plots for {model_name}...")
    
    # Create output directory
    output_dir = Path(PATHS['outputs']['visualizations']) / 'shap' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit samples for performance
    if len(X_test) > max_samples:
        print(f"  Limiting to {max_samples} samples for visualization performance")
        sample_indices = list(range(max_samples))
        X_test_sample = X_test.iloc[sample_indices]
        shap_values_sample = shap_values[sample_indices]
    else:
        X_test_sample = X_test
        shap_values_sample = shap_values
    
    try:
        # 1. Summary plot
        print("    - Creating summary plot...")
        plot_shap_summary(
            shap_values_sample, 
            X_test_sample,
            save_path=output_dir / f"{model_name}_shap_summary.png"
        )
        
        # 2. Waterfall plot (for first instance)
        print("    - Creating waterfall plot...")
        plot_shap_waterfall(
            shap_values_sample[0],
            X_test_sample.iloc[0],
            feature_names=X_test_sample.columns.tolist(),
            save_path=output_dir / f"{model_name}_shap_waterfall.png"
        )
        
        # 3. Dependence plots for top features
        print("    - Creating dependence plots...")
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test_sample.columns,
            'importance': abs(shap_values_sample).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Plot top 5 continuous features
        continuous_features = []
        categorical_features = []
        
        for feature in feature_importance['feature']:
            if X_test_sample[feature].dtype in ['float64', 'int64'] and X_test_sample[feature].nunique() > 10:
                continuous_features.append(feature)
            elif X_test_sample[feature].nunique() <= 10:
                categorical_features.append(feature)
        
        # Plot continuous features
        for feature in continuous_features[:5]:
            plot_shap_dependence(
                feature,
                shap_values_sample,
                X_test_sample,
                save_path=output_dir / f"{model_name}_shap_dependence_{feature.replace(' ', '_')}.png"
            )
        
        # 4. Categorical plots
        print("    - Creating categorical plots...")
        for feature in categorical_features[:3]:
            plot_shap_categorical(
                feature,
                shap_values_sample,
                X_test_sample,
                save_path=output_dir / f"{model_name}_shap_categorical_{feature.replace(' ', '_')}.png"
            )
        
        print(f"    ✓ Successfully generated SHAP plots for {model_name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error generating plots for {model_name}: {str(e)}")
        return False


def main():
    """Main function to generate missing SHAP plots."""
    print("Generating missing SHAP plots for CatBoost and LightGBM models...")
    
    # Set plot style
    set_plot_style()
    
    # Get model lists
    catboost_models, lightgbm_models = get_model_list()
    
    print(f"\nFound {len(catboost_models)} CatBoost models")
    print(f"Found {len(lightgbm_models)} LightGBM models")
    
    all_models = catboost_models + lightgbm_models
    successful = 0
    failed = 0
    no_shap_values = 0
    
    # Process each model
    for i, model_name in enumerate(all_models, 1):
        print(f"\nProcessing model {i}/{len(all_models)}: {model_name}")
        
        # Check if SHAP values exist
        if not check_shap_values_exist(model_name):
            print(f"  ✗ No SHAP values found for {model_name}")
            no_shap_values += 1
            continue
        
        # Load SHAP data
        shap_values, X_test = load_shap_data(model_name)
        
        if shap_values is None:
            print(f"  ✗ Failed to load SHAP values for {model_name}")
            failed += 1
            continue
        
        # Generate plots
        if generate_shap_plots_for_model(model_name, shap_values, X_test):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SHAP Plot Generation Summary:")
    print(f"  Total models: {len(all_models)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  No SHAP values: {no_shap_values}")
    print(f"{'='*60}")
    
    # Check what was actually created
    print("\nChecking generated plots...")
    for model_type in ['CatBoost', 'LightGBM']:
        model_count = 0
        plot_count = 0
        for model_name in all_models:
            if model_type in model_name:
                model_count += 1
                model_dir = Path(PATHS['outputs']['visualizations']) / 'shap' / model_name
                if model_dir.exists():
                    plots = list(model_dir.glob("*.png"))
                    plot_count += len(plots)
        
        print(f"\n{model_type}: {model_count} models, {plot_count} plots generated")


if __name__ == "__main__":
    main()