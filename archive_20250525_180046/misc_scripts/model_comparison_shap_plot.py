#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model comparison SHAP plot generator.

This script creates a comparison visualization of SHAP feature importance
across all four model types: XGBoost, LightGBM, CatBoost, and ElasticNet.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

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

def load_model_data():
    """
    Load model data for all four model types.
    
    Returns:
        dict: Dictionary with model data by model type
    """
    model_data = {}
    
    # Load XGBoost models
    try:
        file_path = Path(settings.MODEL_DIR) / "xgboost_models.pkl"
        if file_path.exists():
            model_data['xgboost'] = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
            print(f"Loaded XGBoost model data")
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
    
    # Load CatBoost models
    try:
        file_path = Path(settings.MODEL_DIR) / "catboost_models.pkl"
        if file_path.exists():
            model_data['catboost'] = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
            print(f"Loaded CatBoost model data")
    except Exception as e:
        print(f"Error loading CatBoost models: {e}")
    
    # Load LightGBM models
    try:
        file_path = Path(settings.MODEL_DIR) / "lightgbm_models.pkl"
        if file_path.exists():
            model_data['lightgbm'] = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
            print(f"Loaded LightGBM model data")
    except Exception as e:
        print(f"Error loading LightGBM models: {e}")
    
    # Load ElasticNet models
    try:
        file_path = Path(settings.MODEL_DIR) / "elasticnet_models.pkl"
        if file_path.exists():
            model_data['elasticnet'] = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
            print(f"Loaded ElasticNet model data")
    except Exception as e:
        print(f"Error loading ElasticNet models: {e}")
    
    return model_data

def load_feature_importance_data(shap_dir):
    """
    Load feature importance data from CSV files.
    
    Args:
        shap_dir: Path to SHAP visualization directory
    
    Returns:
        dict: Dictionary with feature importance data by model type
    """
    feature_importance = {}
    
    # Load XGBoost feature importance
    try:
        xgb_path = shap_dir / "xgboost_feature_importance.csv"
        if xgb_path.exists():
            xgb_df = pd.read_csv(xgb_path)
            feature_importance['XGBoost'] = xgb_df
            print(f"Loaded XGBoost feature importance data")
    except Exception as e:
        print(f"Error loading XGBoost feature importance: {e}")
    
    # Load CatBoost feature importance
    try:
        catboost_path = shap_dir / "catboost_feature_importance.csv"
        if catboost_path.exists():
            catboost_df = pd.read_csv(catboost_path)
            feature_importance['CatBoost'] = catboost_df
            print(f"Loaded CatBoost feature importance data")
    except Exception as e:
        print(f"Error loading CatBoost feature importance: {e}")
    
    # Load LightGBM feature importance with proper feature names
    try:
        lgbm_path = shap_dir / "lightgbm_feature_importance_fixed.csv"
        if lgbm_path.exists():
            lgbm_df = pd.read_csv(lgbm_path)
            feature_importance['LightGBM'] = lgbm_df
            print(f"Loaded LightGBM feature importance data")
    except Exception as e:
        print(f"Error loading LightGBM feature importance: {e}")
    
    # Load ElasticNet feature importance
    try:
        elasticnet_path = shap_dir / "elasticnet_feature_importance.csv"
        if elasticnet_path.exists():
            elasticnet_df = pd.read_csv(elasticnet_path)
            feature_importance['ElasticNet'] = elasticnet_df
            print(f"Loaded ElasticNet feature importance data")
    except Exception as e:
        print(f"Error loading ElasticNet feature importance: {e}")
    
    return feature_importance

def create_comparison_dataframe(feature_importance_data, top_n=20):
    """
    Create a combined DataFrame for comparing feature importance across models.
    
    Args:
        feature_importance_data: Dictionary with feature importance data by model
        top_n: Number of top features to include per model
    
    Returns:
        DataFrame: Combined data for the comparison plot
    """
    if not feature_importance_data:
        print("No feature importance data available")
        return None
    
    # First, collect the top N features from each model
    top_features = set()
    for model_name, df in feature_importance_data.items():
        if df is not None and 'Feature' in df.columns and 'Importance' in df.columns:
            # Get top N features for this model
            top_model_features = df.sort_values('Importance', ascending=False).head(top_n)['Feature'].tolist()
            top_features.update(top_model_features)
    
    top_features = list(top_features)
    print(f"Collected {len(top_features)} unique top features across all models")
    
    # Create a DataFrame with importance values for each model
    comparison_data = pd.DataFrame({'Feature': top_features})
    comparison_data.set_index('Feature', inplace=True)
    
    # Add columns for each model's importance values
    for model_name, df in feature_importance_data.items():
        if df is not None and 'Feature' in df.columns and 'Importance' in df.columns:
            # Create a mapping of feature to importance
            feature_to_importance = dict(zip(df['Feature'], df['Importance']))
            
            # Add the column to our comparison DataFrame
            comparison_data[model_name] = comparison_data.index.map(
                lambda x: feature_to_importance.get(x, 0)
            )
    
    # Fill any missing values with 0
    comparison_data.fillna(0, inplace=True)
    
    # Normalize each model's values to [0, 1] for fair comparison
    for model_name in comparison_data.columns:
        max_val = comparison_data[model_name].max()
        if max_val > 0:  # Avoid division by zero
            comparison_data[model_name] = comparison_data[model_name] / max_val
    
    return comparison_data

def create_model_comparison_plot(comparison_data, shap_dir, top_n=30):
    """
    Create the model comparison heatmap plot.
    
    Args:
        comparison_data: DataFrame with normalized importance values
        shap_dir: Output directory for plots
        top_n: Maximum number of features to show in plot
    """
    if comparison_data is None or comparison_data.empty:
        print("No comparison data available for plotting")
        return
    
    # Ensure we don't try to plot more features than we have
    top_n = min(top_n, len(comparison_data))
    
    # Calculate the overall importance across all models
    comparison_data['Combined'] = comparison_data.sum(axis=1)
    
    # Sort by the combined importance and take top N features
    sorted_data = comparison_data.sort_values('Combined', ascending=False).head(top_n)
    
    # Drop the combined column now that we've used it for sorting
    sorted_data = sorted_data.drop('Combined', axis=1)
    
    # Create the plot
    plt.figure(figsize=(14, max(10, top_n * 0.3)))
    
    # Create a custom colormap that starts from white (for 0 values)
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Generate the heatmap
    ax = sns.heatmap(
        sorted_data,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Normalized Importance (SHAP value)"}
    )
    
    # Customize the plot
    plt.title("Feature Importance Comparison Across Models", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plot_path = shap_dir / "model_comparison_shap_all.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created model comparison plot at {plot_path}")

def main():
    """Main function to generate the model comparison SHAP plot."""
    print("Generating model comparison SHAP plot for all four model types...")
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Load feature importance data
    feature_importance_data = load_feature_importance_data(shap_dir)
    
    if not feature_importance_data:
        print("No feature importance data found. Run the individual SHAP visualization scripts first.")
        return
    
    # Create the comparison DataFrame
    comparison_data = create_comparison_dataframe(feature_importance_data)
    
    if comparison_data is not None:
        # Create the model comparison plot
        create_model_comparison_plot(comparison_data, shap_dir)
        print("Model comparison SHAP plot generation complete.")
    else:
        print("Could not create comparison data. Exiting.")
    
if __name__ == "__main__":
    main()