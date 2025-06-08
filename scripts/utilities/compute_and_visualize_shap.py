#!/usr/bin/env python3
"""Compute SHAP values and generate visualizations for tree models."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings
from src.utils.io import load_model
from src.visualization.plots.shap_plots import SHAPVisualizer
from src.visualization.core.style import setup_visualization_style

# Create PATHS dictionary from settings
PATHS = {
    'outputs': {
        'models': str(settings.MODEL_DIR),
        'visualizations': str(settings.VISUALIZATION_DIR),
        'shap': str(settings.OUTPUT_DIR / 'shap')
    }
}


def get_tree_models():
    """Get list of all tree-based models."""
    models_dir = Path(PATHS['outputs']['models'])
    
    tree_models = []
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        if any(model_type in model_name for model_type in ["CatBoost", "LightGBM", "XGBoost"]):
            tree_models.append(model_name)
    
    return sorted(tree_models)


def compute_shap_values(model_data: Dict[str, Any], max_samples: int = 100) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for a model.
    
    Args:
        model_data: Model data dictionary
        max_samples: Maximum number of samples to use
        
    Returns:
        Tuple of (shap_values, X_sample)
    """
    model = model_data['model']
    X_test = model_data.get('X_test')
    
    if X_test is None:
        print("  No X_test data found")
        return None, None
    
    # Limit samples for performance
    if len(X_test) > max_samples:
        print(f"  Limiting to {max_samples} samples for SHAP computation")
        sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
    else:
        X_sample = X_test
    
    # Create explainer based on model type
    model_name = model_data.get('model_name', 'Unknown')
    
    try:
        if "CatBoost" in model_name:
            explainer = shap.Explainer(model, X_sample)
        elif "LightGBM" in model_name:
            explainer = shap.TreeExplainer(model)
        elif "XGBoost" in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            print(f"  Unknown model type for {model_name}")
            return None, None
        
        # Compute SHAP values
        print("  Computing SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Handle potential multiple outputs
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values, X_sample
        
    except Exception as e:
        print(f"  Error computing SHAP values: {str(e)}")
        return None, None


def generate_shap_visualizations(model_name: str, model_data: Dict[str, Any], 
                                shap_values: np.ndarray, X_sample: pd.DataFrame):
    """Generate SHAP visualizations for a model."""
    print(f"  Generating SHAP visualizations for {model_name}...")
    
    # Create output directory
    output_dir = Path(PATHS['outputs']['visualizations']) / 'shap' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = SHAPVisualizer(model_data)
    
    try:
        # 1. Summary plot
        print("    - Creating summary plot...")
        visualizer.create_shap_summary_plot(
            shap_values, 
            X_sample,
            output_dir / f"{model_name}_shap_summary.png"
        )
        
        # 2. Waterfall plot (for first instance)
        print("    - Creating waterfall plot...")
        visualizer.create_shap_waterfall_plot(
            shap_values,
            X_sample,
            instance_idx=0,
            output_path=output_dir / f"{model_name}_shap_waterfall.png"
        )
        
        # 3. Dependence plots for top features
        print("    - Creating dependence plots...")
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Plot top 5 features
        for i, feature in enumerate(feature_importance['feature'].head(5)):
            try:
                visualizer.create_shap_dependence_plot(
                    shap_values,
                    X_sample,
                    feature,
                    output_dir / f"{model_name}_shap_dependence_{feature.replace(' ', '_').replace('/', '_')}.png"
                )
            except Exception as e:
                print(f"      Warning: Could not create dependence plot for {feature}: {e}")
        
        # 4. Categorical plots for categorical features
        print("    - Creating categorical plots...")
        categorical_features = []
        for col in X_sample.columns:
            if X_sample[col].dtype == 'object' or X_sample[col].nunique() <= 10:
                categorical_features.append(col)
        
        for feature in categorical_features[:3]:
            try:
                visualizer.create_categorical_shap_plot(
                    shap_values,
                    X_sample,
                    feature,
                    output_dir / f"{model_name}_shap_categorical_{feature.replace(' ', '_').replace('/', '_')}.png"
                )
            except Exception as e:
                print(f"      Warning: Could not create categorical plot for {feature}: {e}")
        
        print(f"    ✓ Successfully generated SHAP visualizations for {model_name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error generating visualizations: {str(e)}")
        return False


def main():
    """Main function to compute SHAP values and generate visualizations."""
    print("Computing SHAP values and generating visualizations for tree models...")
    
    # Set plot style
    setup_visualization_style()
    
    # Get tree models
    tree_models = get_tree_models()
    print(f"\nFound {len(tree_models)} tree-based models")
    
    successful = 0
    failed = 0
    
    # Process each model
    for i, model_name in enumerate(tree_models, 1):
        print(f"\n{'='*60}")
        print(f"Processing model {i}/{len(tree_models)}: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model_path = Path(PATHS['outputs']['models']) / f"{model_name}.pkl"
        try:
            model_data = load_model(f"{model_name}.pkl", PATHS['outputs']['models'])
            
            # Ensure model_name is set
            model_data['model_name'] = model_name
            
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            failed += 1
            continue
        
        # Compute SHAP values
        shap_values, X_sample = compute_shap_values(model_data, max_samples=50)
        
        if shap_values is None:
            print(f"  ✗ Failed to compute SHAP values")
            failed += 1
            continue
        
        # Generate visualizations
        if generate_shap_visualizations(model_name, model_data, shap_values, X_sample):
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
    
    # Check what was created
    print("\nChecking generated plots...")
    shap_dir = Path(PATHS['outputs']['visualizations']) / 'shap'
    
    for model_type in ['CatBoost', 'LightGBM', 'XGBoost']:
        model_count = 0
        plot_count = 0
        
        for model_name in tree_models:
            if model_type in model_name:
                model_count += 1
                model_dir = shap_dir / model_name
                if model_dir.exists():
                    plots = list(model_dir.glob("*.png"))
                    plot_count += len(plots)
        
        print(f"\n{model_type}: {model_count} models, {plot_count} plots generated")


if __name__ == "__main__":
    main()