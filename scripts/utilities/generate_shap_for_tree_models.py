#!/usr/bin/env python3
"""Generate SHAP visualizations for all tree-based models."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.plots.shap_plots import SHAPVisualizer
from src.visualization.core.style import setup_visualization_style


def load_grouped_models():
    """Load all tree-based models from grouped pickle files."""
    models_dir = settings.MODEL_DIR
    all_models = {}
    
    # Load CatBoost models
    catboost_file = models_dir / "catboost_models.pkl"
    if catboost_file.exists():
        with open(catboost_file, 'rb') as f:
            catboost_models = pickle.load(f)
            all_models.update(catboost_models)
            print(f"Loaded {len(catboost_models)} CatBoost models")
    
    # Load LightGBM models
    lightgbm_file = models_dir / "lightgbm_models.pkl"
    if lightgbm_file.exists():
        with open(lightgbm_file, 'rb') as f:
            lightgbm_models = pickle.load(f)
            all_models.update(lightgbm_models)
            print(f"Loaded {len(lightgbm_models)} LightGBM models")
    
    # Load XGBoost models
    xgboost_file = models_dir / "xgboost_models.pkl"
    if xgboost_file.exists():
        with open(xgboost_file, 'rb') as f:
            xgboost_models = pickle.load(f)
            all_models.update(xgboost_models)
            print(f"Loaded {len(xgboost_models)} XGBoost models")
    
    return all_models


def compute_shap_for_model(model_name, model_data, max_samples=50):
    """Compute SHAP values for a single model."""
    print(f"\n  Computing SHAP values for {model_name}...")
    
    model = model_data.get('model')
    X_test = model_data.get('X_test')
    
    if model is None or X_test is None:
        print(f"    ✗ Missing model or X_test data")
        return None, None
    
    # Limit samples for performance
    if len(X_test) > max_samples:
        sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
    else:
        X_sample = X_test
    
    try:
        # Create appropriate explainer based on model type
        if "CatBoost" in model_name:
            # For CatBoost, use the generic Explainer
            explainer = shap.Explainer(model, X_sample)
        elif "LightGBM" in model_name:
            # For LightGBM, use TreeExplainer
            explainer = shap.TreeExplainer(model)
        elif "XGBoost" in model_name:
            # For XGBoost, use TreeExplainer
            explainer = shap.TreeExplainer(model)
        else:
            print(f"    ✗ Unknown model type")
            return None, None
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle potential multiple outputs
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        print(f"    ✓ SHAP values computed: shape {shap_values.shape}")
        return shap_values, X_sample
        
    except Exception as e:
        print(f"    ✗ Error computing SHAP values: {str(e)}")
        return None, None


def create_shap_plots(model_name, model_data, shap_values, X_sample):
    """Create SHAP visualizations for a model."""
    print(f"  Generating SHAP plots...")
    
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model_name is in model_data
    model_data['model_name'] = model_name
    
    # Initialize visualizer
    visualizer = SHAPVisualizer(model_data)
    
    plots_created = 0
    
    try:
        # 1. Summary plot
        try:
            visualizer.create_shap_summary_plot(
                shap_values, 
                X_sample,
                output_dir / f"{model_name}_shap_summary.png"
            )
            plots_created += 1
            print(f"    ✓ Summary plot created")
        except Exception as e:
            print(f"    ✗ Summary plot failed: {e}")
        
        # 2. Waterfall plot (for first instance)
        try:
            visualizer.create_shap_waterfall_plot(
                shap_values,
                X_sample,
                instance_idx=0,
                output_path=output_dir / f"{model_name}_shap_waterfall.png"
            )
            plots_created += 1
            print(f"    ✓ Waterfall plot created")
        except Exception as e:
            print(f"    ✗ Waterfall plot failed: {e}")
        
        # 3. Dependence plots for top features
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Create dependence plots for top 3 features
        for i, (_, row) in enumerate(feature_importance.head(3).iterrows()):
            feature = row['feature']
            try:
                safe_feature_name = feature.replace(' ', '_').replace('/', '_').replace('\\', '_')
                visualizer.create_shap_dependence_plot(
                    shap_values,
                    X_sample,
                    feature,
                    output_dir / f"{model_name}_shap_dependence_{safe_feature_name}.png"
                )
                plots_created += 1
                print(f"    ✓ Dependence plot for '{feature}' created")
            except Exception as e:
                print(f"    ✗ Dependence plot for '{feature}' failed: {e}")
        
        # 4. Categorical plots
        categorical_features = []
        for col in X_sample.columns:
            if X_sample[col].dtype == 'object' or X_sample[col].nunique() <= 10:
                categorical_features.append(col)
        
        # Create plots for up to 2 categorical features
        for feature in categorical_features[:2]:
            try:
                safe_feature_name = feature.replace(' ', '_').replace('/', '_').replace('\\', '_')
                visualizer.create_categorical_shap_plot(
                    shap_values,
                    X_sample,
                    feature,
                    output_dir / f"{model_name}_shap_categorical_{safe_feature_name}.png"
                )
                plots_created += 1
                print(f"    ✓ Categorical plot for '{feature}' created")
            except Exception as e:
                print(f"    ✗ Categorical plot for '{feature}' failed: {e}")
        
        print(f"  Total plots created: {plots_created}")
        return plots_created > 0
        
    except Exception as e:
        print(f"  ✗ Error generating plots: {str(e)}")
        return False


def main():
    """Main function to generate SHAP visualizations."""
    print("Generating SHAP visualizations for tree-based models...")
    
    # Set plot style
    setup_visualization_style()
    
    # Load all models
    all_models = load_grouped_models()
    
    if not all_models:
        print("No models found!")
        return
    
    # Filter for tree-based models only
    tree_models = {
        name: data for name, data in all_models.items() 
        if any(model_type in name for model_type in ["CatBoost", "LightGBM", "XGBoost"])
    }
    
    print(f"\nFound {len(tree_models)} tree-based models to process")
    
    successful = 0
    failed = 0
    
    # Process each model
    for i, (model_name, model_data) in enumerate(tree_models.items(), 1):
        print(f"\n{'='*60}")
        print(f"Processing model {i}/{len(tree_models)}: {model_name}")
        print(f"{'='*60}")
        
        # Compute SHAP values
        shap_values, X_sample = compute_shap_for_model(model_name, model_data)
        
        if shap_values is None:
            failed += 1
            continue
        
        # Generate visualizations
        if create_shap_plots(model_name, model_data, shap_values, X_sample):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SHAP Visualization Summary:")
    print(f"  Total models: {len(tree_models)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")
    
    # Check results
    print("\nChecking generated visualizations...")
    shap_dir = settings.VISUALIZATION_DIR / 'shap'
    
    for model_type in ['CatBoost', 'LightGBM', 'XGBoost']:
        model_count = sum(1 for name in tree_models if model_type in name)
        plot_count = 0
        
        for name in tree_models:
            if model_type in name:
                model_dir = shap_dir / name
                if model_dir.exists():
                    plots = list(model_dir.glob("*.png"))
                    plot_count += len(plots)
        
        print(f"{model_type}: {model_count} models, {plot_count} plots")


if __name__ == "__main__":
    main()