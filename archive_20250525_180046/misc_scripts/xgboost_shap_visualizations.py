#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Specialized feature importance visualization generator for XGBoost models.

This script provides feature importance visualizations for XGBoost models
using several alternative approaches when SHAP integration fails:

1. Feature importance from model.get_feature_importance()
2. Permutation importance using sklearn's implementation
3. Partial dependence plots for top features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
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

def load_xgboost_models():
    """
    Load XGBoost model data.
    
    Returns:
        dict: Dictionary with model data
    """
    try:
        file_path = Path(settings.MODEL_DIR) / "xgboost_models.pkl"
        if file_path.exists():
            models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
            print(f"Loaded XGBoost model data")
            return models
        else:
            print(f"Model file not found: {file_path}")
            return {}
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        return {}

def create_feature_importance_plot(model_data, shap_dir):
    """
    Create feature importance plot using built-in XGBoost feature importance.
    
    Args:
        model_data: Dictionary with XGBoost model data
        shap_dir: Output directory for plots
    """
    # Choose representative models (prefer optuna models)
    representative_models = {}
    
    for model_name, model_info in model_data.items():
        model_variant = model_name.split('_')[1]  # Base or Yeo
        
        # Skip if not a valid model or doesn't have 'model' key
        if not isinstance(model_info, dict) or 'model' not in model_info:
            continue
            
        # Prefer optuna models
        if '_optuna' in model_name and model_variant not in representative_models:
            representative_models[model_variant] = (model_name, model_info)
    
    # If no optuna models found, use basic models
    if not representative_models:
        for model_name, model_info in model_data.items():
            model_variant = model_name.split('_')[1]  # Base or Yeo
            
            if not isinstance(model_info, dict) or 'model' not in model_info:
                continue
                
            if model_variant not in representative_models:
                representative_models[model_variant] = (model_name, model_info)
    
    # Create feature importance plots for each representative model
    for model_variant, (model_name, model_info) in representative_models.items():
        model = model_info['model']
        
        try:
            # Get feature importance from model
            feature_importances = model.feature_importances_
            feature_names = model_info.get('feature_names', [])
            
            # If feature_names not available, create generic names
            if not feature_names or len(feature_names) != len(feature_importances):
                feature_names = [f"Feature_{i}" for i in range(len(feature_importances))]
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(feature_importances)],
                'Importance': feature_importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Take top 20 features
            importance_df = importance_df.head(20)
            
            # Plot creation removed to avoid creating xgboost_{model_variant.lower()}_feature_importance.png
            print(f"Feature importance plot for XGBoost {model_variant} skipped to avoid file creation")
            
        except Exception as e:
            print(f"Error creating feature importance plot for {model_name}: {e}")

def create_permutation_importance_plot(model_data, shap_dir):
    """
    Create permutation importance plot as an alternative to SHAP.
    
    Args:
        model_data: Dictionary with XGBoost model data
        shap_dir: Output directory for plots
    """
    from sklearn.inspection import permutation_importance
    
    # Choose representative models (prefer optuna models)
    representative_models = {}
    
    for model_name, model_info in model_data.items():
        model_variant = model_name.split('_')[1]  # Base or Yeo
        
        # Skip if not a valid model or doesn't have required keys
        if not isinstance(model_info, dict) or 'model' not in model_info or 'X_test' not in model_info or 'y_test' not in model_info:
            continue
            
        # Prefer optuna models
        if '_optuna' in model_name and model_variant not in representative_models:
            representative_models[model_variant] = (model_name, model_info)
    
    # If no optuna models found, use basic models
    if not representative_models:
        for model_name, model_info in model_data.items():
            model_variant = model_name.split('_')[1]  # Base or Yeo
            
            if not isinstance(model_info, dict) or 'model' not in model_info or 'X_test' not in model_info or 'y_test' not in model_info:
                continue
                
            if model_variant not in representative_models:
                representative_models[model_variant] = (model_name, model_info)
    
    # Create permutation importance plots for each representative model
    for model_variant, (model_name, model_info) in representative_models.items():
        model = model_info['model']
        X_test = model_info['X_test']
        y_test = model_info['y_test']
        
        try:
            # Clean data by converting object columns
            X_clean = X_test.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    X_clean = X_clean.drop(columns=[col])
            
            # Calculate permutation importance
            result = permutation_importance(
                model, X_clean, y_test, 
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Create DataFrame with importance values
            importance_df = pd.DataFrame({
                'Feature': X_clean.columns,
                'Importance': result.importances_mean,
                'Std': result.importances_std
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Take top 20 features
            importance_df = importance_df.head(20)
            
            # Create plot
            plt.figure(figsize=(12, 10))
            ax = sns.barplot(
                data=importance_df, 
                x='Importance', 
                y='Feature',
                xerr=importance_df['Std'],
                error_kw={'elinewidth': 1.5, 'capsize': 3}
            )
            
            # Customize plot
            plt.title(f"XGBoost Permutation Importance - {model_variant}", fontsize=14)
            plt.xlabel('Mean Decrease in Performance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            
            # Save the plot
            plt.tight_layout()
            plot_path = shap_dir / f"xgboost_{model_variant.lower()}_permutation_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created permutation importance plot for XGBoost {model_variant}")
        
        except Exception as e:
            print(f"Error creating permutation importance plot for {model_name}: {e}")

def create_partial_dependence_plots(model_data, shap_dir):
    """
    Create partial dependence plots for top features as an alternative visualization.
    
    Args:
        model_data: Dictionary with XGBoost model data
        shap_dir: Output directory for plots
    """
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    
    # Choose representative models (prefer optuna models)
    representative_models = {}
    
    for model_name, model_info in model_data.items():
        model_variant = model_name.split('_')[1]  # Base or Yeo
        
        # Skip if not a valid model or doesn't have required keys
        if not isinstance(model_info, dict) or 'model' not in model_info or 'X_test' not in model_info:
            continue
            
        # Prefer optuna models
        if '_optuna' in model_name and model_variant not in representative_models:
            representative_models[model_variant] = (model_name, model_info)
    
    # If no optuna models found, use basic models
    if not representative_models:
        for model_name, model_info in model_data.items():
            model_variant = model_name.split('_')[1]  # Base or Yeo
            
            if not isinstance(model_info, dict) or 'model' not in model_info or 'X_test' not in model_info:
                continue
                
            if model_variant not in representative_models:
                representative_models[model_variant] = (model_name, model_info)
    
    # Create partial dependence plots for each representative model
    for model_variant, (model_name, model_info) in representative_models.items():
        model = model_info['model']
        X_test = model_info['X_test']
        
        try:
            # Clean data by converting object columns
            X_clean = X_test.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    X_clean = X_clean.drop(columns=[col])
            
            # Get feature importance
            feature_importances = model.feature_importances_
            
            # Get indices of top 5 features
            top_indices = np.argsort(feature_importances)[-5:]
            top_features = X_clean.columns[top_indices]
            
            # Create partial dependence plots
            fig, ax = plt.subplots(figsize=(12, 10))
            PartialDependenceDisplay.from_estimator(
                model, 
                X_clean, 
                top_features,
                kind='average',
                subsample=100,
                n_jobs=-1,
                random_state=42,
                ax=ax
            )
            
            # Customize plot
            plt.suptitle(f"XGBoost Partial Dependence - {model_variant}", fontsize=14)
            
            # Save the plot
            plt.tight_layout()
            plot_path = shap_dir / f"xgboost_{model_variant.lower()}_partial_dependence.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create individual plots for each top feature
            for feature in top_features:
                fig, ax = plt.subplots(figsize=(8, 6))
                PartialDependenceDisplay.from_estimator(
                    model, 
                    X_clean, 
                    [feature],
                    kind='average',
                    subsample=100,
                    n_jobs=-1,
                    random_state=42,
                    ax=ax
                )
                
                # Customize plot
                plt.suptitle(f"XGBoost Partial Dependence - {feature}", fontsize=14)
                
                # Save the plot
                plt.tight_layout()
                safe_feature_name = feature.replace("/", "_").replace("\\", "_")
                plot_path = shap_dir / f"xgboost_{model_variant.lower()}_pdp_{safe_feature_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Created partial dependence plots for XGBoost {model_variant}")
        
        except Exception as e:
            print(f"Error creating partial dependence plots for {model_name}: {e}")

def main():
    """Main function to generate XGBoost visualizations."""
    print("Generating XGBoost visualizations...")
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Load XGBoost model data
    model_data = load_xgboost_models()
    
    if not model_data:
        print("No XGBoost model data found. Exiting.")
        return

    # Create feature importance plot as a fallback
    print("\n--- Creating built-in feature importance plot ---")
    create_feature_importance_plot(model_data, shap_dir)
    
    # Create permutation importance plot as another alternative
    print("\n--- Creating permutation importance plot ---")
    create_permutation_importance_plot(model_data, shap_dir)
    
    # Create partial dependence plots as another alternative
    print("\n--- Creating partial dependence plots ---")
    create_partial_dependence_plots(model_data, shap_dir)
    
    print("XGBoost visualization generation complete.")
    print(f"Visualizations saved to: {shap_dir}")

if __name__ == "__main__":
    main()