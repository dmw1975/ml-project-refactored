#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate SHAP (SHapley Additive exPlanations) visualizations for tree-based models
(XGBoost, CatBoost, and LightGBM) to provide unified, consistent model explanations.

This script produces four key SHAP visualizations:
1. SHAP Summary Plot: Overall feature importance distribution
2. SHAP Dependence Plot: How specific features impact predictions
3. SHAP Force Plot: Individual prediction explanations
4. Model Comparison SHAP Plot: Feature importance across all models

All plots are saved to the visualizations/shap directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
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

def load_model_data():
    """
    Load model data for XGBoost, CatBoost, and LightGBM.
    
    Returns:
        dict: Dictionary with model data by model type
    """
    model_files = {
        "xgboost": "xgboost_models.pkl",
        "catboost": "catboost_models.pkl", 
        "lightgbm": "lightgbm_models.pkl"
    }
    
    model_data = {}
    
    for model_type, filename in model_files.items():
        try:
            file_path = Path(settings.MODEL_DIR) / filename
            if file_path.exists():
                models = io.load_model(filename, settings.MODEL_DIR)
                model_data[model_type] = models
                print(f"Loaded {model_type} model data")
            else:
                print(f"Model file not found: {file_path}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    return model_data

def get_shap_explainer(model, model_type, X_sample):
    """
    Get appropriate SHAP explainer for the model type.
    
    Args:
        model: Trained model instance
        model_type: Type of model ('xgboost', 'catboost', or 'lightgbm')
        X_sample: Sample of features for explanation
        
    Returns:
        SHAP explainer object and expected values
    """
    # Clean data by converting object columns to category or dropping them
    X_clean = X_sample.copy()
    
    # Handle string/object columns that cause issues with SHAP
    for col in X_clean.columns:
        if X_clean[col].dtype == 'object':
            # Drop the problematic column - focusing on numerical features
            X_clean = X_clean.drop(columns=[col])
    
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
        return explainer, explainer.expected_value, X_clean
    
    elif model_type == 'catboost':
        # CatBoost typically handles categorical features better
        explainer = shap.TreeExplainer(model)
        return explainer, explainer.expected_value, X_clean
    
    elif model_type == 'lightgbm':
        explainer = shap.TreeExplainer(model)
        return explainer, explainer.expected_value, X_clean
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_shap_values(model_data):
    """
    Generate SHAP values for all models.
    
    Args:
        model_data: Dictionary with model data by model type
        
    Returns:
        dict: Dictionary with SHAP values by model type and dataset
    """
    shap_values_dict = {}
    
    for model_type, models in model_data.items():
        shap_values_dict[model_type] = {}
        
        # Find best model variant and calculate SHAP values
        for model_name, model_info in models.items():
            # Skip if not a valid model or doesn't have the right keys
            if not isinstance(model_info, dict) or 'model' not in model_info or 'X_test' not in model_info:
                continue
                
            model = model_info['model']
            X_test = model_info['X_test']
            dataset = model_info.get('model_name', model_name)
            
            # For efficiency, use a sample of the test data if it's large
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
            
            try:
                print(f"Generating SHAP values for {model_type} - {dataset}")
                explainer, expected_value, X_clean = get_shap_explainer(model, model_type, X_sample)
                shap_values = explainer.shap_values(X_clean)
                
                # Store SHAP values and related data
                shap_values_dict[model_type][dataset] = {
                    'shap_values': shap_values,
                    'expected_value': expected_value,
                    'X_sample': X_clean,  # Use the cleaned data
                    'feature_names': X_clean.columns.tolist(),  # Use the cleaned column names
                    'model': model
                }
                
            except Exception as e:
                print(f"Error generating SHAP values for {model_type} - {dataset}: {e}")
    
    return shap_values_dict

def create_shap_summary_plot(shap_values_dict, shap_dir):
    """
    Create SHAP summary plot for each model and dataset.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    for model_type, datasets in shap_values_dict.items():
        # Choose one representative dataset per model type
        # Prefer optuna models over basic ones
        representative_dataset = None
        for dataset_name in datasets.keys():
            if "_optuna" in dataset_name:
                representative_dataset = dataset_name
                break
        
        # If no optuna model, use the first available
        if representative_dataset is None and datasets:
            representative_dataset = list(datasets.keys())[0]
        
        if representative_dataset is None:
            print(f"No datasets found for {model_type}")
            continue
            
        # Get SHAP data for the representative dataset
        shap_data = datasets[representative_dataset]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        if model_type != 'catboost':  # Normal case
            shap.summary_plot(
                shap_data['shap_values'], 
                shap_data['X_sample'],
                feature_names=shap_data['feature_names'],
                show=False,
                plot_size=(12, 8)
            )
        else:  # CatBoost sometimes returns a list instead of array
            if isinstance(shap_data['shap_values'], list):
                shap.summary_plot(
                    shap_data['shap_values'][0], 
                    shap_data['X_sample'],
                    feature_names=shap_data['feature_names'],
                    show=False,
                    plot_size=(12, 8)
                )
            else:
                shap.summary_plot(
                    shap_data['shap_values'], 
                    shap_data['X_sample'],
                    feature_names=shap_data['feature_names'],
                    show=False,
                    plot_size=(12, 8)
                )
        
        # Customize plot
        plt.title(f"{model_type.upper()} Feature Impact Distribution", fontsize=14)
        
        # Save the plot
        plt.tight_layout()
        plot_path = shap_dir / f"{model_type}_shap_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created SHAP summary plot for {model_type}")

def create_shap_dependence_plots(shap_values_dict, shap_dir):
    """
    Create SHAP dependence plots for top features of each model.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    for model_type, datasets in shap_values_dict.items():
        # Choose one representative dataset per model type (prefer optuna)
        representative_dataset = None
        for dataset_name in datasets.keys():
            if "_optuna" in dataset_name:
                representative_dataset = dataset_name
                break
        
        # If no optuna model, use the first available
        if representative_dataset is None and datasets:
            representative_dataset = list(datasets.keys())[0]
        
        if representative_dataset is None:
            print(f"No datasets found for {model_type}")
            continue
            
        # Get SHAP data for the representative dataset
        shap_data = datasets[representative_dataset]
        
        # Calculate feature importance based on mean absolute SHAP value
        if isinstance(shap_data['shap_values'], list):
            # For multi-output models like CatBoost
            mean_abs_shap = np.abs(shap_data['shap_values'][0]).mean(0)
        else:
            mean_abs_shap = np.abs(shap_data['shap_values']).mean(0)
        
        # Get the indices of the top 5 features
        top_indices = np.argsort(mean_abs_shap)[-5:]
        
        # Create dependence plots for top 5 features
        for index in reversed(top_indices):
            feature_name = shap_data['feature_names'][index]
            plt.figure(figsize=(10, 6))
            
            if isinstance(shap_data['shap_values'], list):
                shap.dependence_plot(
                    index, 
                    shap_data['shap_values'][0], 
                    shap_data['X_sample'],
                    feature_names=shap_data['feature_names'],
                    show=False
                )
            else:
                shap.dependence_plot(
                    index, 
                    shap_data['shap_values'], 
                    shap_data['X_sample'],
                    feature_names=shap_data['feature_names'],
                    show=False
                )
            
            # Customize plot
            plt.title(f"{model_type.upper()} SHAP Dependence: {feature_name}", fontsize=14)
            
            # Save the plot
            plt.tight_layout()
            safe_feature_name = feature_name.replace("/", "_").replace("\\", "_")
            plot_path = shap_dir / f"{model_type}_dependence_{safe_feature_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Created SHAP dependence plots for top features of {model_type}")

def create_shap_force_plots(shap_values_dict, shap_dir):
    """
    Create SHAP force plots for sample predictions of each model.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    for model_type, datasets in shap_values_dict.items():
        # Choose one representative dataset per model type (prefer optuna)
        representative_dataset = None
        for dataset_name in datasets.keys():
            if "_optuna" in dataset_name:
                representative_dataset = dataset_name
                break
        
        # If no optuna model, use the first available
        if representative_dataset is None and datasets:
            representative_dataset = list(datasets.keys())[0]
        
        if representative_dataset is None:
            print(f"No datasets found for {model_type}")
            continue
            
        # Get SHAP data for the representative dataset
        shap_data = datasets[representative_dataset]
        
        # Choose 5 diverse samples for force plots
        if len(shap_data['X_sample']) >= 5:
            # Get the sample with highest and lowest SHAP values, plus 3 from middle range
            if isinstance(shap_data['shap_values'], list):
                shap_sum = np.sum(np.abs(shap_data['shap_values'][0]), axis=1)
            else:
                shap_sum = np.sum(np.abs(shap_data['shap_values']), axis=1)
                
            sorted_indices = np.argsort(shap_sum)
            sample_indices = [
                sorted_indices[0],  # Lowest impact
                sorted_indices[len(sorted_indices)//4],  # 25th percentile
                sorted_indices[len(sorted_indices)//2],  # Median impact
                sorted_indices[3*len(sorted_indices)//4],  # 75th percentile
                sorted_indices[-1]  # Highest impact
            ]
        else:
            # If fewer than 5 samples, use all available
            sample_indices = list(range(len(shap_data['X_sample'])))
        
        # Create force plots for selected samples
        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(14, 3))
            
            # Create the force plot
            if isinstance(shap_data['shap_values'], list):
                shap_values = shap_data['shap_values'][0][idx]
            else:
                shap_values = shap_data['shap_values'][idx]
                
            # Convert to matplotlib plot
            force_plot = shap.force_plot(
                base_value=shap_data['expected_value'] if not isinstance(shap_data['expected_value'], list) else shap_data['expected_value'][0],
                shap_values=shap_values,
                features=shap_data['X_sample'].iloc[idx],
                feature_names=shap_data['feature_names'],
                matplotlib=True,
                show=False
            )
            
            # Customize plot
            plt.title(f"{model_type.upper()} SHAP Force Plot - Sample {i+1}", fontsize=14)
            
            # Save the plot
            plt.tight_layout()
            plot_path = shap_dir / f"{model_type}_force_plot_sample_{i+1}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Created SHAP force plots for {model_type}")

def create_model_comparison_plot(shap_values_dict, shap_dir):
    """
    Create SHAP model comparison plot to compare feature importance across models.
    
    Args:
        shap_values_dict: Dictionary with SHAP values
        shap_dir: Output directory for plots
    """
    # Extract feature importance from each model
    model_importance = {}
    all_features = set()
    
    for model_type, datasets in shap_values_dict.items():
        # Skip if no datasets
        if not datasets:
            continue
            
        # Choose one representative dataset per model type (prefer optuna)
        representative_dataset = None
        for dataset_name in datasets.keys():
            if "_optuna" in dataset_name:
                representative_dataset = dataset_name
                break
        
        # If no optuna model, use the first available
        if representative_dataset is None:
            representative_dataset = list(datasets.keys())[0]
            
        # Get SHAP data for the representative dataset
        shap_data = datasets[representative_dataset]
        
        # Calculate mean absolute SHAP value for each feature
        if isinstance(shap_data['shap_values'], list):
            mean_abs_shap = np.abs(shap_data['shap_values'][0]).mean(0)
        else:
            mean_abs_shap = np.abs(shap_data['shap_values']).mean(0)
        
        # Create dictionary of feature importance
        importance_dict = {}
        for i, feature in enumerate(shap_data['feature_names']):
            importance_dict[feature] = mean_abs_shap[i]
            all_features.add(feature)
        
        model_importance[model_type] = importance_dict
    
    # Convert to DataFrame for plotting
    all_features = sorted(list(all_features))
    comparison_data = []
    
    for feature in all_features:
        for model_type in model_importance.keys():
            importance = model_importance[model_type].get(feature, 0)
            comparison_data.append({
                'Feature': feature,
                'Model': model_type.upper(),
                'Importance': importance
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Get top 15 features by average importance across models
    top_features = comparison_df.groupby('Feature')['Importance'].mean().nlargest(15).index.tolist()
    comparison_df = comparison_df[comparison_df['Feature'].isin(top_features)]
    
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
    plt.title('Feature Importance Comparison Across Models', fontsize=16)
    plt.ylabel('Feature', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plot_path = shap_dir / "model_comparison_shap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created model comparison SHAP plot")

def main():
    """Main function to generate SHAP visualizations."""
    print("Generating SHAP visualizations...")
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Load model data
    model_data = load_model_data()
    
    if not model_data:
        print("No model data found. Exiting.")
        return
    
    # Generate SHAP values
    shap_values_dict = generate_shap_values(model_data)
    
    if not any(datasets for datasets in shap_values_dict.values()):
        print("No SHAP values could be calculated. Exiting.")
        return
    
    # Create visualizations
    create_shap_summary_plot(shap_values_dict, shap_dir)
    create_shap_dependence_plots(shap_values_dict, shap_dir)
    create_shap_force_plots(shap_values_dict, shap_dir)
    create_model_comparison_plot(shap_values_dict, shap_dir)
    
    print("SHAP visualization generation complete.")
    print(f"Visualizations saved to: {shap_dir}")

if __name__ == "__main__":
    main()