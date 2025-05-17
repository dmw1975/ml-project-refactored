#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified model interpretability visualization generator for tree-based models.

This script orchestrates the generation of SHAP plots for CatBoost and LightGBM models 
and specialized feature importance visualizations for XGBoost models (which have
compatibility issues with SHAP). This ensures comprehensive and consistent 
visualization of feature importance across all model types.

The script produces:
1. SHAP plots for CatBoost and LightGBM:
   - Summary plots showing feature impact distribution
   - Dependence plots for top features
   - Force plots explaining individual predictions
   - Model comparison plots
   
2. Feature importance visualizations for XGBoost:
   - Built-in feature importance plots
   - Permutation importance plots
   - Partial dependence plots for top features
"""

import subprocess
import os
from pathlib import Path
from config import settings

def ensure_output_directory():
    """Create shap directory under visualizations if it doesn't exist."""
    shap_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "shap"
    os.makedirs(shap_dir, exist_ok=True)
    return shap_dir

def main():
    """Main function to orchestrate the generation of model interpretability visualizations."""
    print("Generating model interpretability visualizations...")
    
    # Ensure output directory exists
    shap_dir = ensure_output_directory()
    
    # Run improved SHAP visualizations for CatBoost and LightGBM
    print("\n=== Generating SHAP visualizations for CatBoost and LightGBM ===")
    try:
        subprocess.run(["python", "improved_shap_visualizations.py"], check=True)
        print("Successfully generated SHAP visualizations for CatBoost and LightGBM")
    except subprocess.CalledProcessError as e:
        print(f"Error generating SHAP visualizations: {e}")
    
    # Run specialized XGBoost visualizations
    print("\n=== Generating specialized feature importance visualizations for XGBoost ===")
    try:
        subprocess.run(["python", "xgboost_shap_visualizations.py"], check=True)
        print("Successfully generated feature importance visualizations for XGBoost")
    except subprocess.CalledProcessError as e:
        print(f"Error generating XGBoost visualizations: {e}")
    
    # Run fixed LightGBM SHAP visualizations with proper feature names
    print("\n=== Generating fixed LightGBM SHAP visualizations with proper feature names ===")
    try:
        subprocess.run(["python", "fixed_lightgbm_shap_visualizations.py"], check=True)
        print("Successfully generated fixed LightGBM SHAP visualizations with proper feature names")
    except subprocess.CalledProcessError as e:
        print(f"Error generating fixed LightGBM visualizations: {e}")
    
    # Run fixed model comparison plot with proper feature names
    print("\n=== Generating fixed model comparison plot with proper feature names ===")
    try:
        subprocess.run(["python", "fixed_model_comparison.py"], check=True)
        print("Successfully generated fixed model comparison plot with proper feature names")
    except subprocess.CalledProcessError as e:
        print(f"Error generating fixed model comparison plot: {e}")
    
    # Generate consistent feature importance bar charts for all models
    print("\n=== Generating consistent feature importance bar charts for all models ===")
    try:
        subprocess.run(["python", "create_feature_importance_charts.py"], check=True)
        print("Successfully generated consistent feature importance bar charts for all models")
    except subprocess.CalledProcessError as e:
        print(f"Error generating feature importance bar charts: {e}")
    
    # Generate ElasticNet SHAP visualizations
    print("\n=== Generating ElasticNet SHAP visualizations ===")
    try:
        subprocess.run(["python", "elasticnet_shap_visualizations.py"], check=True)
        print("Successfully generated ElasticNet SHAP visualizations")
    except subprocess.CalledProcessError as e:
        print(f"Error generating ElasticNet SHAP visualizations: {e}")
    
    # Generate combined model comparison plot across all four model types
    print("\n=== Generating combined model comparison SHAP plot ===")
    try:
        subprocess.run(["python", "model_comparison_shap_plot.py"], check=True)
        print("Successfully generated combined model comparison SHAP plot")
    except subprocess.CalledProcessError as e:
        print(f"Error generating combined model comparison SHAP plot: {e}")
    
    print("\nModel interpretability visualization generation complete.")
    print(f"All visualizations saved to: {shap_dir}")
    
    # Create README file explaining the visualizations
    readme_path = shap_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("""# Model Interpretability Visualizations

This directory contains various visualizations for interpreting tree-based models:

## CatBoost and LightGBM SHAP Visualizations

### Summary Plots
Summary plots display the feature impact distribution across all samples, showing both magnitude and direction of impact. Features are ordered by their overall importance.

### Force Plots
Force plots explain individual predictions by showing how each feature pushes the prediction value away from the baseline.

### Model Comparison
The model comparison heatmap displays feature importance across different models, enabling direct comparison of feature significance.

## Fixed LightGBM SHAP Visualizations

These visualizations provide improved feature name mapping for LightGBM SHAP plots:

### Fixed Summary Plots
Similar to regular SHAP summary plots but with proper feature names instead of generic "feature_0", "feature_1", etc.

### Fixed Force Plots
Explain individual predictions with properly labeled features for clearer insights.

### Feature Importance Bar Charts
Bar charts showing the most important features with proper names and their importance values, created in a consistent format for all model types (LightGBM, CatBoost, and XGBoost) to enable direct visual comparison.

### Fixed Model Comparison
An improved model comparison heatmap that correctly displays actual feature names instead of generic ones (feature_0, feature_1, etc.) for easier interpretation and analysis across models.

### All-Model Comparison Plot
A comprehensive heatmap visualization showing normalized feature importance across all four model types (XGBoost, CatBoost, LightGBM, and ElasticNet). This plot enables direct comparison of feature significance between different modeling approaches and helps identify consistently important features.

## XGBoost Feature Importance Visualizations

Due to compatibility issues between XGBoost and SHAP, we provide alternative visualizations:

### Feature Importance Plots
These plots use XGBoost's built-in feature importance metrics, which indicate the relative importance of each feature.

### Permutation Importance
Permutation importance plots show how shuffling feature values affects model performance, providing a model-agnostic importance measure.

### Partial Dependence Plots
These plots show how predictions change as a function of feature values, revealing non-linear relationships and thresholds.

## ElasticNet SHAP Visualizations

Linear ElasticNet models can also be explained using SHAP values:

### Summary Plots
Summary plots show the distribution of feature impact for ElasticNet models, with features ordered by their overall importance.

### Force Plots
Force plots explain individual ElasticNet predictions by showing how each feature contributes to the final prediction.

### Feature Importance Bar Chart
A bar chart showing the most important features for ElasticNet models, based on SHAP values.
""")
    
    print(f"Created README file at {readme_path}")

if __name__ == "__main__":
    main()