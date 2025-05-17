#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature importance bar chart generator for all tree-based models.

This script creates consistent feature importance bar charts for LightGBM, CatBoost, 
and XGBoost models using the same format and styling for better comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

def create_catboost_feature_importance_chart():
    """
    Create a feature importance bar chart for CatBoost model.
    """
    # Load CatBoost models
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
        print("Loaded CatBoost model data")
    except Exception as e:
        print(f"Error loading CatBoost models: {e}")
        return
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Choose representative model (prefer optuna)
    representative_model = None
    for model_name in catboost_models.keys():
        if '_optuna' in model_name:
            representative_model = model_name
            break
    
    # If no optuna model, use the first available
    if representative_model is None and catboost_models:
        representative_model = list(catboost_models.keys())[0]
    
    if representative_model is None:
        print("No CatBoost models found")
        return
    
    model_info = catboost_models[representative_model]
    model = model_info['model']
    
    try:
        # Get feature importance scores
        feature_importances = model.get_feature_importance()
        feature_names = model_info.get('feature_names', [])
        
        # If feature_names not available or length mismatch, create generic names
        if not feature_names or len(feature_names) != len(feature_importances):
            feature_names = [f"Feature_{i}" for i in range(len(feature_importances))]
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importances)],
            'Importance': feature_importances
        })
        
        # Sort by importance and take top 20
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        # Save to CSV
        csv_path = shap_dir / "catboost_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        print(f"Saved CatBoost feature importance table to {csv_path}")
        
        # Create the bar chart
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        
        # Add value annotations
        for i, importance in enumerate(importance_df.head(15)['Importance']):
            ax.text(importance + max(importance_df['Importance']) * 0.01, i, f"{importance:.4f}", va='center')
        
        plt.title("CatBoost Feature Importance", fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plot_path = shap_dir / "catboost_feature_importance_bar.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created CatBoost feature importance bar chart at {plot_path}")
    
    except Exception as e:
        print(f"Error creating CatBoost feature importance chart: {e}")

def create_xgboost_feature_importance_chart():
    """
    Create a feature importance bar chart for XGBoost model with proper feature names.
    """
    # Load XGBoost models
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
        print("Loaded XGBoost model data")
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
        return
    
    # Also load LightGBM models to get proper feature names mapping
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print("Loaded LightGBM model data for feature name mapping")
        
        # Get feature name mapping from LightGBM
        lightgbm_feature_mapping = None
        for model_name, model_info in lightgbm_models.items():
            if 'feature_name_mapping' in model_info and 'feature_names' in model_info:
                lightgbm_feature_mapping = model_info['feature_name_mapping']
                original_feature_names = model_info['feature_names']
                print(f"Found feature name mapping with {len(lightgbm_feature_mapping)} entries")
                break
    except Exception as e:
        print(f"Error loading LightGBM models for feature mapping: {e}")
        lightgbm_feature_mapping = None
        original_feature_names = None
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Choose representative model (prefer optuna)
    representative_model = None
    for model_name in xgboost_models.keys():
        if '_optuna' in model_name:
            representative_model = model_name
            break
    
    # If no optuna model, use the first available
    if representative_model is None and xgboost_models:
        representative_model = list(xgboost_models.keys())[0]
    
    if representative_model is None:
        print("No XGBoost models found")
        return
    
    model_info = xgboost_models[representative_model]
    model = model_info['model']
    
    try:
        # Get feature importance scores
        feature_importances = model.feature_importances_
        xgb_feature_names = model_info.get('feature_names', [])
        
        # If feature_names not available or length mismatch, create generic names
        if not xgb_feature_names or len(xgb_feature_names) != len(feature_importances):
            xgb_feature_names = [f"Feature_{i}" for i in range(len(feature_importances))]
        
        # Try to map feature indices to actual names using LightGBM mappings
        if lightgbm_feature_mapping and original_feature_names:
            # Get top 20 important indices
            top_indices = np.argsort(feature_importances)[-20:]
            
            # Create mapped DataFrame
            mapped_data = []
            for idx in reversed(top_indices):
                try:
                    # If the feature is within range of original feature names
                    if idx < len(original_feature_names):
                        # Get the proper name from original feature names
                        feature_name = original_feature_names[idx]
                        importance = feature_importances[idx]
                        mapped_data.append({
                            'Feature': feature_name,
                            'Importance': importance
                        })
                    else:
                        feature_name = f"Feature_{idx}"
                        importance = feature_importances[idx]
                        mapped_data.append({
                            'Feature': feature_name,
                            'Importance': importance
                        })
                except Exception as mapping_error:
                    print(f"Error mapping feature index {idx}: {mapping_error}")
                    feature_name = f"Feature_{idx}"
                    importance = feature_importances[idx]
                    mapped_data.append({
                        'Feature': feature_name,
                        'Importance': importance
                    })
            
            importance_df = pd.DataFrame(mapped_data)
        else:
            # Create DataFrame with generic names
            importance_df = pd.DataFrame({
                'Feature': xgb_feature_names[:len(feature_importances)],
                'Importance': feature_importances
            })
            
            # Sort by importance and take top 20
            importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        # Save to CSV
        csv_path = shap_dir / "xgboost_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        print(f"Saved XGBoost feature importance table to {csv_path}")
        
        # Create the bar chart
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        
        # Add value annotations
        for i, importance in enumerate(importance_df.head(15)['Importance']):
            ax.text(importance + max(importance_df['Importance']) * 0.01, i, f"{importance:.4f}", va='center')
        
        plt.title("XGBoost Feature Importance", fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plot_path = shap_dir / "xgboost_feature_importance_bar.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created XGBoost feature importance bar chart at {plot_path}")
    
    except Exception as e:
        print(f"Error creating XGBoost feature importance chart: {e}")

def create_lightgbm_feature_importance_chart():
    """
    Create a feature importance bar chart for LightGBM model based on existing data.
    """
    try:
        # Set up output directory
        shap_dir = setup_output_directory()
        
        # Load existing feature importance data
        csv_path = shap_dir / "lightgbm_feature_importance_fixed.csv"
        if not csv_path.exists():
            print(f"Feature importance CSV not found: {csv_path}")
            return
        
        # Load existing data
        importance_df = pd.read_csv(csv_path)
        print(f"Loaded LightGBM feature importance data from {csv_path}")
        
        # Create the bar chart (similar format to other models)
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        
        # Add value annotations
        for i, importance in enumerate(importance_df.head(15)['Importance']):
            ax.text(importance + max(importance_df['Importance']) * 0.01, i, f"{importance:.4f}", va='center')
        
        plt.title("LightGBM Feature Importance (Mean |SHAP value|)", fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plot_path = shap_dir / "lightgbm_feature_importance_bar.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created standardized LightGBM feature importance bar chart at {plot_path}")
        
    except Exception as e:
        print(f"Error creating LightGBM feature importance chart: {e}")

def main():
    """Main function to generate consistent feature importance bar charts."""
    print("Generating consistent feature importance bar charts...")
    
    # Create feature importance charts for all model types
    create_catboost_feature_importance_chart()
    create_xgboost_feature_importance_chart()
    create_lightgbm_feature_importance_chart()
    
    print("Feature importance chart generation complete.")

if __name__ == "__main__":
    main()