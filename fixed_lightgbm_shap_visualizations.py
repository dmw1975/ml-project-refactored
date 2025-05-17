#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed LightGBM SHAP visualizations with proper feature name mapping.

This script addresses the issue where LightGBM SHAP plots show generic feature names
(e.g., "feature_0", "feature_1") instead of the actual descriptive feature names.
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

def generate_fixed_lightgbm_shap_visualizations():
    """
    Generate SHAP visualizations for LightGBM models with proper feature names.
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
    
    # Find a model with feature name mapping and X_test_clean
    representative_model = None
    for model_name, model_info in lightgbm_models.items():
        if ('feature_name_mapping' in model_info and 
            'cleaned_feature_names' in model_info and 
            'X_test_clean' in model_info and 
            'model' in model_info):
            representative_model = model_name
            # Prefer optuna model if available
            if '_optuna' in model_name:
                break
    
    if representative_model is None:
        print("No suitable LightGBM model found with feature name mapping.")
        return
    
    print(f"Using model: {representative_model}")
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
    
    # Create a mapping dictionary from feature_0 to original name
    # The key insight: feature_0 -> market_cap_usd, feature_1 -> net_income_usd, etc.
    # This mapping is stored in feature_name_mapping, but we need to extract it properly
    feature_display_names = []
    for i, name in enumerate(X_sample.columns):
        # Get original name from the mapping
        if name in feature_name_mapping:
            feature_display_names.append(feature_name_mapping[name])
        else:
            feature_display_names.append(name)  # Keep original if not in mapping
    
    # Create summary plot with mapped feature names
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample,
        feature_names=feature_display_names,
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
    
    # Removed code for creating dependence plots to save space
    # Note: We still calculate top_indices for other potential uses
    top_n = 5
    top_indices = np.argsort(mean_abs_shap)[-top_n:]
    
    # Create force plots for a few examples
    sample_indices = list(range(min(5, len(X_sample))))
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(14, 3))
        
        # Create force plot with proper feature names
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[idx],
            features=X_sample.iloc[idx],
            feature_names=feature_display_names,
            matplotlib=True,
            show=False
        )
        
        # Customize plot
        plt.title(f"LightGBM SHAP Force Plot - Sample {i+1}", fontsize=14)
        
        # Save the plot
        plt.tight_layout()
        plot_path = shap_dir / f"lightgbm_force_plot_sample_{i+1}_fixed.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Created fixed LightGBM force plots")
    
    # Create a summary table of top 20 features by importance
    top_n_large = 20
    top_indices_large = np.argsort(mean_abs_shap)[-top_n_large:]
    
    summary_data = []
    for idx in reversed(top_indices_large):
        feature_name = feature_display_names[idx]
        importance = mean_abs_shap[idx]
        summary_data.append({
            'Feature': feature_name,
            'Importance': importance
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = shap_dir / "lightgbm_feature_importance_fixed.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved feature importance table to {csv_path}")
    
    # Bar chart creation removed to avoid creating lightgbm_feature_importance_bar_fixed.png
    
def main():
    """Main function to generate fixed LightGBM SHAP visualizations."""
    print("Generating fixed LightGBM SHAP visualizations...")
    generate_fixed_lightgbm_shap_visualizations()
    print("Fixed LightGBM SHAP visualization generation complete.")

if __name__ == "__main__":
    main()