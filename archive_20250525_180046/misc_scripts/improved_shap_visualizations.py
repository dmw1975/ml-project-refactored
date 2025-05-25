#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved SHAP (SHapley Additive exPlanations) visualizations for tree-based models
(XGBoost, CatBoost, and LightGBM) with fixes for feature alignment issues.

This script addresses two key issues with the original implementation:
1. XGBoost shape mismatch: "shape.cbegin() == chunksize * rows" error
2. LightGBM feature count mismatch: "number of features in data (388) is not the same as in training data (362)"

Key improvements:
- Uses different approach for XGBoost: shap.Explainer instead of shap.TreeExplainer when appropriate
- Ensures proper feature alignment for LightGBM by using stored X_test_clean
- Provides better handling of data types and categorical variables
- Cleans data consistently with the training procedure
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
import xgboost as xgb
import lightgbm as lgb

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

def get_xgboost_shap_values(model, X_sample):
    """
    Get SHAP values for XGBoost model using appropriate method based on the model version.
    
    Args:
        model: XGBoost model
        X_sample: Sample data for explanation
        
    Returns:
        tuple: (shap_values, expected_value)
    """
    try:
        # First attempt: Use TreeExplainer (works for many XGBoost models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value
        
        return shap_values, expected_value
    except Exception as e:
        print(f"TreeExplainer failed, trying alternative approach: {e}")
        
        # Second attempt: Use shap.Explainer (more compatible with recent XGBoost)
        explainer = shap.Explainer(model)
        shap_explanation = explainer(X_sample)
        
        return shap_explanation.values, shap_explanation.base_values[0]

def get_lightgbm_shap_values(model, X_sample, cleaned_feature_names=None, feature_name_mapping=None):
    """
    Get SHAP values for LightGBM model, ensuring feature alignment.
    
    Args:
        model: LightGBM model
        X_sample: Sample data for explanation
        cleaned_feature_names: List of cleaned feature names used during training
        feature_name_mapping: Mapping from cleaned to original feature names
        
    Returns:
        tuple: (shap_values, expected_value)
    """
    # If we have cleaned feature names and mapping, ensure proper alignment
    if cleaned_feature_names and feature_name_mapping:
        # Create properly aligned data frame with clean column names
        X_clean = pd.DataFrame(index=X_sample.index)
        
        # Only include features that were used during training (by cleaned name)
        for clean_name in cleaned_feature_names:
            # Get original feature name
            orig_name = feature_name_mapping.get(clean_name)
            
            # If original name exists in X_sample, add it with the clean name
            if orig_name in X_sample.columns:
                X_clean[clean_name] = X_sample[orig_name]
            else:
                # Feature not found, create a column of zeros as placeholder
                X_clean[clean_name] = 0
                print(f"Warning: Feature {orig_name} not found, using zeros")
        
        # Now X_clean has exactly the same features as what was used in training
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_clean)
        expected_value = explainer.expected_value
        
        return shap_values, expected_value, X_clean
    else:
        # Fallback to regular approach if we don't have the mapping
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        expected_value = explainer.expected_value
        
        return shap_values, expected_value, X_sample

def get_shap_values_for_model(model, model_type, X_sample, model_info=None):
    """
    Get SHAP values using model-specific approaches.
    
    Args:
        model: Trained model instance
        model_type: Type of model ('xgboost', 'catboost', or 'lightgbm')
        X_sample: Sample of features for explanation
        model_info: Additional model information (for LightGBM feature mapping)
        
    Returns:
        tuple: (shap_values, expected_value, X_clean)
    """
    # Clean data by converting object columns to category or dropping them
    X_clean = X_sample.copy()
    
    # Handle string/object columns that cause issues with SHAP
    for col in X_clean.columns:
        if X_clean[col].dtype == 'object':
            # Drop the problematic column - focusing on numerical features
            X_clean = X_clean.drop(columns=[col])
    
    if model_type == 'xgboost':
        shap_values, expected_value = get_xgboost_shap_values(model, X_clean)
        return shap_values, expected_value, X_clean
    
    elif model_type == 'lightgbm':
        # If we have the stored test data with clean names, use it
        if model_info and 'X_test_clean' in model_info and 'cleaned_feature_names' in model_info:
            # Use a sample from the existing clean test data
            sample_size = min(100, len(model_info['X_test_clean']))
            X_test_clean_sample = model_info['X_test_clean'].sample(sample_size, random_state=42) if len(model_info['X_test_clean']) > sample_size else model_info['X_test_clean']
            
            # Use TreeExplainer directly with the correctly aligned data
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_clean_sample)
            expected_value = explainer.expected_value
            
            return shap_values, expected_value, X_test_clean_sample
        else:
            # Fallback to cleaning the data if we don't have X_test_clean
            cleaned_feature_names = model_info.get('cleaned_feature_names') if model_info else None
            feature_name_mapping = model_info.get('feature_name_mapping') if model_info else None
            
            return get_lightgbm_shap_values(model, X_clean, cleaned_feature_names, feature_name_mapping)
    
    elif model_type == 'catboost':
        # CatBoost typically handles categorical features better
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_clean)
        expected_value = explainer.expected_value
        
        return shap_values, expected_value, X_clean
    
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
            if not isinstance(model_info, dict) or 'model' not in model_info:
                continue
                
            model = model_info['model']
            
            # For LightGBM, use the stored clean test data if available
            if model_type == 'lightgbm' and 'X_test_clean' in model_info:
                X_test = model_info['X_test_clean']
            elif 'X_test' in model_info:
                X_test = model_info['X_test']
            else:
                print(f"Skipping {model_type} - {model_name}: No test data found")
                continue
                
            dataset = model_info.get('model_name', model_name)
            
            # For efficiency, use a sample of the test data if it's large
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42) if len(X_test) > sample_size else X_test
            
            try:
                print(f"Generating SHAP values for {model_type} - {dataset}")
                shap_values, expected_value, X_clean = get_shap_values_for_model(model, model_type, X_sample, model_info)
                
                # Store SHAP values and related data
                shap_values_dict[model_type][dataset] = {
                    'shap_values': shap_values,
                    'expected_value': expected_value,
                    'X_sample': X_clean,  # Use the cleaned/aligned data
                    'feature_names': X_clean.columns.tolist(),  # Use the cleaned column names
                    'model': model
                }
                
                print(f"Successfully generated SHAP values for {model_type} - {dataset}")
                
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
        
        # Check if feature importance calculation succeeded
        if len(mean_abs_shap) != len(shap_data['feature_names']):
            print(f"Warning: Feature importance shape mismatch for {model_type}. "
                  f"Got {len(mean_abs_shap)} values for {len(shap_data['feature_names'])} features.")
            continue
            
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
            try:
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
            except Exception as e:
                print(f"Error creating force plot for {model_type} sample {i+1}: {e}")
        
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
            
        # Skip if shapes don't match
        if len(mean_abs_shap) != len(shap_data['feature_names']):
            print(f"Warning: Shape mismatch for {model_type}, skipping for comparison plot")
            continue
        
        # Create dictionary of feature importance
        importance_dict = {}
        for i, feature in enumerate(shap_data['feature_names']):
            importance_dict[feature] = mean_abs_shap[i]
            all_features.add(feature)
        
        model_importance[model_type] = importance_dict
    
    # Check if we have any models to compare
    if not model_importance:
        print("No valid models for comparison plot")
        return
        
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
    
    # Model comparison plot creation removed to avoid creating model_comparison_shap.png
    print("Model comparison plot generation skipped to avoid creating model_comparison_shap.png")

def main():
    """Main function to generate SHAP visualizations."""
    print("Generating improved SHAP visualizations...")
    
    # Set up output directory
    shap_dir = setup_output_directory()
    
    # Load model data
    model_data = load_model_data()
    
    if not model_data:
        print("No model data found. Exiting.")
        return
    
    # Generate SHAP values with improved approach
    shap_values_dict = generate_shap_values(model_data)
    
    if not any(datasets for datasets in shap_values_dict.values()):
        print("No SHAP values could be calculated. Exiting.")
        return
    
    # Create visualizations (skipping dependence plots to save space)
    create_shap_summary_plot(shap_values_dict, shap_dir)
    # Removed: create_shap_dependence_plots(shap_values_dict, shap_dir)
    create_shap_force_plots(shap_values_dict, shap_dir)
    create_model_comparison_plot(shap_values_dict, shap_dir)
    
    print("SHAP visualization generation complete.")
    print(f"Visualizations saved to: {shap_dir}")

if __name__ == "__main__":
    main()