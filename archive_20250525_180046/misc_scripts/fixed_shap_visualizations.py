#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed SHAP (SHapley Additive exPlanations) visualizations for tree-based models
with proper feature name mapping.

This script fixes the issue with generic feature names ("feature_0", "feature_1", etc.)
in LightGBM SHAP plots by correctly mapping them back to their original names
using the feature_name_mapping provided in the model data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import warnings

from config import settings
from utils import io

# Set up consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Configure warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def setup_output_directory():
    """Create shap directory under visualizations if it doesn't exist."""
    shap_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "shap"
    os.makedirs(shap_dir, exist_ok=True)
    return shap_dir

def fix_lightgbm_feature_names():
    """
    Fix LightGBM SHAP plots by creating versions with proper feature names.
    
    This function reads the LightGBM model data, extracts the feature name mapping,
    and creates new SHAP plots with proper feature names.
    """
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print("Loaded LightGBM model data")
    except Exception as e:
        print(f"Error loading LightGBM models: {e}")
        return
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Check if any model has the required data
    representative_model = None
    for model_name, model_info in lightgbm_models.items():
        if ('feature_name_mapping' in model_info and 
            'cleaned_feature_names' in model_info and 
            'X_test_clean' in model_info and 
            'model' in model_info):
            representative_model = model_name
            break
    
    if representative_model is None:
        print("No suitable LightGBM model found with feature name mapping.")
        return
    
    print(f"Using representative model: {representative_model}")
    model_info = lightgbm_models[representative_model]
    model = model_info['model']
    
    # Get feature name mapping (cleaned to original)
    feature_name_mapping = model_info['feature_name_mapping']
    cleaned_feature_names = model_info['cleaned_feature_names']
    X_test_clean = model_info['X_test_clean']
    
    print(f"Found {len(cleaned_feature_names)} cleaned feature names and mapping.")
    
    # Sample data for SHAP calculations
    sample_size = min(100, len(X_test_clean))
    X_sample = X_test_clean.sample(sample_size, random_state=42)
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create inverse mapping from cleaned to original names
    inverse_mapping = {v: k for k, v in feature_name_mapping.items()}
    
    # Create mapping from feature index to original name
    index_to_original = {}
    for i, cleaned_name in enumerate(cleaned_feature_names):
        if cleaned_name in inverse_mapping:
            index_to_original[i] = inverse_mapping[cleaned_name]
        else:
            index_to_original[i] = cleaned_name  # Keep as is if not in mapping
    
    # Create DataFrame with original names for plotting
    X_sample_mapped = X_sample.copy()
    X_sample_mapped.columns = [inverse_mapping.get(col, col) for col in X_sample.columns]
    
    # Create summary plot with original feature names
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=[inverse_mapping.get(col, col) for col in X_sample.columns],
        show=False,
        plot_size=(12, 8)
    )
    
    # Customize plot
    plt.title("LightGBM Feature Impact Distribution", fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plot_path = shap_dir / "lightgbm_shap_summary_fixed.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created fixed LightGBM SHAP summary plot at {plot_path}")
    
    # Calculate feature importance
    mean_abs_shap = np.abs(shap_values).mean(0)
    
    # Get top 5 feature indices
    top_indices = np.argsort(mean_abs_shap)[-5:]
    
    # Create dependence plots for top 5 features
    for index in reversed(top_indices):
        # Get original feature name if available
        cleaned_name = X_sample.columns[index]
        original_name = inverse_mapping.get(cleaned_name, cleaned_name)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            index, 
            shap_values, 
            X_sample,
            feature_names=[inverse_mapping.get(col, col) for col in X_sample.columns],
            show=False
        )
        
        # Customize plot
        plt.title(f"LightGBM SHAP Dependence: {original_name}", fontsize=14)
        
        # Save the plot
        plt.tight_layout()
        safe_feature_name = original_name.replace("/", "_").replace("\\", "_")
        plot_path = shap_dir / f"lightgbm_dependence_{safe_feature_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Created fixed LightGBM dependence plots for top features")

def main():
    """Main function to fix LightGBM SHAP visualizations."""
    print("Fixing LightGBM SHAP visualizations...")
    fix_lightgbm_feature_names()
    print("LightGBM SHAP visualization fixes complete.")

if __name__ == "__main__":
    main()