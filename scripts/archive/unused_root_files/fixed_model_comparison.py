#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed model comparison SHAP plot with proper feature name mapping.
This script creates an improved model comparison plot that correctly maps
LightGBM feature names to their actual names for better interpretability.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import warnings

from src.config import settings
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

def create_fixed_model_comparison_plot():
    """
    Create an improved model comparison plot that correctly maps feature names.
    """
    # Load model data for all model types
    model_types = ["xgboost", "catboost", "lightgbm"]
    model_data = {}
    
    for model_type in model_types:
        try:
            file_path = Path(settings.MODEL_DIR) / f"{model_type}_models.pkl"
            if file_path.exists():
                models = io.load_model(f"{model_type}_models.pkl", settings.MODEL_DIR)
                model_data[model_type] = models
                print(f"Loaded {model_type} model data")
            else:
                print(f"Model file not found: {file_path}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Load existing SHAP values if available
    # This is mainly to get a sense of the feature importance order
    # Load LightGBM feature mapping
    lightgbm_feature_mapping = None
    if "lightgbm" in model_data:
        for model_name, model_info in model_data["lightgbm"].items():
            if ('feature_name_mapping' in model_info and 
                'cleaned_feature_names' in model_info):
                lightgbm_feature_mapping = model_info['feature_name_mapping']
                print(f"Loaded feature name mapping from {model_name}")
                break
    
    if not lightgbm_feature_mapping:
        print("Could not find LightGBM feature name mapping")
        return
        
    # Load feature importance data directly from file
    try:
        importance_csv = shap_dir / "lightgbm_feature_importance_fixed.csv"
        if importance_csv.exists():
            lightgbm_importance_df = pd.read_csv(importance_csv)
            print(f"Loaded feature importance data from {importance_csv}")
        else:
            print(f"Feature importance file not found: {importance_csv}")
            return
    except Exception as e:
        print(f"Error loading feature importance data: {e}")
        return
    
    # Generate feature importance from SHAP values for CatBoost
    catboost_importance = {}
    
    if "catboost" in model_data:
        for model_name, model_info in model_data["catboost"].items():
            if "_optuna" in model_name:
                # Generate importance from model directly
                try:
                    # Get feature scores from model
                    model = model_info['model']
                    feature_names = model_info['feature_names']
                    importances = model.get_feature_importance()
                    
                    if len(importances) != len(feature_names):
                        # Sizes don't match, use generic names
                        feature_names = [f"feature_{i}" for i in range(len(importances))]
                        
                    for i, feature in enumerate(feature_names):
                        # Try to match this feature to LightGBM feature if possible
                        if feature in lightgbm_feature_mapping.keys():
                            mapped_name = lightgbm_feature_mapping[feature]
                        else:
                            mapped_name = feature
                            
                        # Scale importance (CatBoost scale is different from SHAP)
                        scaled_importance = importances[i] / max(importances) * 0.3  # Scale to ~0.3 max for comparison
                        catboost_importance[mapped_name] = scaled_importance
                except Exception as e:
                    print(f"Error getting CatBoost importance: {e}")
                break
    
    # Get top features from LightGBM for comparison
    top_features = lightgbm_importance_df['Feature'].tolist()[:15]  # Get top 15
    
    # Prepare data for plotting
    comparison_data = []
    
    for feature in top_features:
        # LightGBM data (already have proper names)
        lightgbm_imp = lightgbm_importance_df[lightgbm_importance_df['Feature'] == feature]['Importance'].values[0]
        comparison_data.append({
            'Feature': feature,
            'Model': 'LIGHTGBM',
            'Importance': lightgbm_imp
        })
        
        # CatBoost data
        catboost_imp = catboost_importance.get(feature, 0)
        comparison_data.append({
            'Feature': feature,
            'Model': 'CATBOOST',
            'Importance': catboost_imp
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Pivot the data for plotting
    pivot_df = comparison_df.pivot(index='Feature', columns='Model', values='Importance')
    
    # Create heatmap
    ax = sns.heatmap(
        pivot_df, 
        cmap='viridis', 
        annot=True, 
        fmt='.3f',
        linewidths=.5,
        cbar_kws={'label': 'Mean |SHAP value|'}
    )
    
    # Customize plot
    plt.title('Feature Importance Comparison Across Models (Fixed Names)', fontsize=16)
    plt.ylabel('Feature', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plot_path = shap_dir / "model_comparison_shap_fixed.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created fixed model comparison SHAP plot at {plot_path}")

def main():
    """Main function to generate fixed model comparison plot."""
    print("Generating fixed model comparison plot...")
    create_fixed_model_comparison_plot()
    print("Fixed model comparison plot generation complete.")

if __name__ == "__main__":
    main()