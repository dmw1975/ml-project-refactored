#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ElasticNet SHAP visualizations generator.

This script creates SHAP visualizations for ElasticNet models using the same
format as the other tree-based model SHAP visualizations for consistency.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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

def load_elasticnet_models():
    """
    Load ElasticNet model data.
    
    Returns:
        dict: Dictionary with model data
    """
    try:
        file_path = Path(settings.MODEL_DIR) / "elasticnet_models.pkl"
        if file_path.exists():
            models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
            print(f"Loaded ElasticNet model data")
            return models
        else:
            print(f"Model file not found: {file_path}")
            return {}
    except Exception as e:
        print(f"Error loading ElasticNet model: {e}")
        return {}

def generate_elasticnet_shap_values(model_data):
    """
    Generate SHAP values for ElasticNet models.
    
    Args:
        model_data: Dictionary with ElasticNet model data
        
    Returns:
        dict: Dictionary with SHAP values by dataset
    """
    shap_values_dict = {}
    
    for model_name, model_info in model_data.items():
        # Skip if not a valid model or doesn't have required keys
        if not isinstance(model_info, dict) or 'model' not in model_info or 'X_test' not in model_info:
            continue
        
        model = model_info['model']
        X_test = model_info['X_test']
        feature_names = model_info.get('feature_names', X_test.columns.tolist())
        
        # For efficiency, use a sample of the test data if it's large
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
        
        try:
            print(f"Generating SHAP values for ElasticNet - {model_name}")
            
            # Get the feature names that were used during training
            if hasattr(model, 'feature_names_in_'):
                # Use the exact features that were used during training
                training_features = model.feature_names_in_.tolist()
                print(f"Using {len(training_features)} training features from model.feature_names_in_")
                
                # Check if all training features are in the test data
                missing_features = [f for f in training_features if f not in X_test.columns]
                if missing_features:
                    print(f"Warning: {len(missing_features)} training features are missing in the test data")
                    # Use only features that are available in the test data
                    training_features = [f for f in training_features if f in X_test.columns]
                
                X_sample_filtered = X_sample[training_features]
            else:
                # Fallback: Filter out non-numeric columns and match the size by taking top N features
                print("Model doesn't have feature_names_in_, using alternative approach")
                numeric_cols = X_test.select_dtypes(include=['number']).columns.tolist()
                # Take only the number of columns that matches the coefficient length
                training_features = numeric_cols[:len(model.coef_)]
                X_sample_filtered = X_sample[training_features]
            
            # For linear models like ElasticNet, we can directly use the coefficients
            # as SHAP values (the effect of a feature is its coefficient * feature value)
            coef = model.coef_
            
            # Generate SHAP values manually: coef * (X - X_mean)
            X_mean = X_sample_filtered.mean(axis=0)
            expected_value = model.intercept_
            
            # Calculate SHAP values based on coefficients and mean-centered data
            X_centered = X_sample_filtered - X_mean
            
            # Handle missing features in random models
            if '_Random' in model_name and len(X_centered.columns) != len(coef):
                print(f"Column count mismatch: X_centered has {len(X_centered.columns)} columns but coef has {len(coef)} elements")
                print("Using only the first N coefficients that match the available features")
                # Use only the first N coefficients that match the available features
                shap_values = X_centered.values * coef[:X_centered.shape[1]]
            else:
                shap_values = X_centered.values * coef
            
            print(f"Successfully calculated SHAP values with shape {shap_values.shape} for {X_sample_filtered.shape[1]} features")
            
            # Store SHAP values and related data
            shap_values_dict[model_name] = {
                'shap_values': shap_values,
                'expected_value': expected_value,
                'X_sample': X_sample_filtered,
                'feature_names': training_features,
                'model': model
            }
            
            print(f"Successfully generated SHAP values for ElasticNet - {model_name}")
            
        except Exception as e:
            print(f"Error generating SHAP values for ElasticNet - {model_name}: {e}")
    
    return shap_values_dict

def create_elasticnet_shap_summary_plot(shap_values_dict, shap_dir):
    """
    Create SHAP summary plot for ElasticNet models.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    if not shap_values_dict:
        print("No SHAP values available for ElasticNet models")
        return
    
    # Choose a representative model (prefer models with higher R²)
    best_model_name = None
    best_r2 = -float('inf')
    
    for model_name in shap_values_dict.keys():
        if 'yeo' in model_name.lower():  # Prefer Yeo-Johnson transformed models
            best_model_name = model_name
            break
    
    # If no Yeo model found, take the first available
    if best_model_name is None and shap_values_dict:
        best_model_name = next(iter(shap_values_dict.keys()))
    
    if best_model_name is None:
        print("No models found for ElasticNet SHAP summary plot")
        return
    
    # Get SHAP data for the representative model
    shap_data = shap_values_dict[best_model_name]
    
    # Create the summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_data['shap_values'], 
        shap_data['X_sample'],
        feature_names=shap_data['feature_names'],
        show=False,
        plot_size=(12, 8)
    )
    
    # Customize plot
    plt.title("ElasticNet Feature Impact Distribution", fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plot_path = shap_dir / "elasticnet_shap_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created SHAP summary plot for ElasticNet")

def create_elasticnet_feature_importance_table(shap_values_dict, shap_dir):
    """
    Create ElasticNet feature importance table and bar chart.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    if not shap_values_dict:
        print("No SHAP values available for ElasticNet models")
        return
    
    # Choose a representative model (prefer models with higher R²)
    best_model_name = None
    
    for model_name in shap_values_dict.keys():
        if 'yeo' in model_name.lower():  # Prefer Yeo-Johnson transformed models
            best_model_name = model_name
            break
    
    # If no Yeo model found, take the first available
    if best_model_name is None and shap_values_dict:
        best_model_name = next(iter(shap_values_dict.keys()))
    
    if best_model_name is None:
        print("No models found for ElasticNet feature importance table")
        return
    
    # Get SHAP data for the representative model
    shap_data = shap_values_dict[best_model_name]
    
    # Calculate feature importance based on mean absolute SHAP value
    mean_abs_shap = np.abs(shap_data['shap_values']).mean(0)
    
    # Create DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'Feature': shap_data['feature_names'],
        'Importance': mean_abs_shap
    })
    
    # Sort by importance and take top 20
    importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
    
    # Save to CSV
    csv_path = shap_dir / "elasticnet_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved ElasticNet feature importance table to {csv_path}")
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    
    # Add value annotations
    for i, importance in enumerate(importance_df.head(15)['Importance']):
        ax.text(importance + max(importance_df['Importance']) * 0.01, i, f"{importance:.4f}", va='center')
    
    plt.title("ElasticNet Feature Importance (Mean |SHAP value|)", fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plot_path = shap_dir / "elasticnet_feature_importance_bar.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created ElasticNet feature importance bar chart")

def create_elasticnet_force_plots(shap_values_dict, shap_dir):
    """
    Create SHAP force plots for ElasticNet models.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    if not shap_values_dict:
        print("No SHAP values available for ElasticNet models")
        return
    
    # Choose a representative model (prefer models with higher R²)
    best_model_name = None
    
    for model_name in shap_values_dict.keys():
        if 'yeo' in model_name.lower():  # Prefer Yeo-Johnson transformed models
            best_model_name = model_name
            break
    
    # If no Yeo model found, take the first available
    if best_model_name is None and shap_values_dict:
        best_model_name = next(iter(shap_values_dict.keys()))
    
    if best_model_name is None:
        print("No models found for ElasticNet force plots")
        return
    
    # Get SHAP data for the representative model
    shap_data = shap_values_dict[best_model_name]
    
    # Create force plots for a few examples
    num_samples = min(5, len(shap_data['X_sample']))
    
    for i in range(num_samples):
        plt.figure(figsize=(14, 3))
        
        # Create force plot
        force_plot = shap.force_plot(
            base_value=shap_data['expected_value'],
            shap_values=shap_data['shap_values'][i],
            features=shap_data['X_sample'].iloc[i],
            feature_names=shap_data['feature_names'],
            matplotlib=True,
            show=False
        )
        
        # Customize plot
        plt.title(f"ElasticNet SHAP Force Plot - Sample {i+1}", fontsize=14)
        
        # Save the plot
        plt.tight_layout()
        plot_path = shap_dir / f"elasticnet_force_plot_sample_{i+1}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Created ElasticNet force plots")

def main():
    """Main function to generate ElasticNet SHAP visualizations."""
    print("Generating ElasticNet SHAP visualizations...")
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Load model data
    elasticnet_models = load_elasticnet_models()
    
    if not elasticnet_models:
        print("No ElasticNet model data found. Exiting.")
        return
    
    # Generate SHAP values
    shap_values_dict = generate_elasticnet_shap_values(elasticnet_models)
    
    if not shap_values_dict:
        print("No SHAP values could be calculated for ElasticNet models. Exiting.")
        return
    
    # Create visualizations
    create_elasticnet_shap_summary_plot(shap_values_dict, shap_dir)
    create_elasticnet_feature_importance_table(shap_values_dict, shap_dir)
    create_elasticnet_force_plots(shap_values_dict, shap_dir)
    
    print("ElasticNet SHAP visualization generation complete.")
    print(f"Visualizations saved to: {shap_dir}")

if __name__ == "__main__":
    main()