#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved SHAP visualization for CatBoost with better categorical feature handling.
This addresses the grey dots issue by creating appropriate visualizations for categorical features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
import warnings

warnings.filterwarnings('ignore')

def create_categorical_shap_plot(shap_values, X_sample, feature_name, output_path):
    """
    Create a specialized SHAP plot for categorical features.
    Shows the distribution of SHAP values for each category.
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        feature_name: Name of the categorical feature
        output_path: Path to save the plot
    """
    # Get the feature column index
    feature_idx = list(X_sample.columns).index(feature_name)
    
    # Extract SHAP values for this feature
    feature_shap_values = shap_values[:, feature_idx]
    
    # Get the feature values
    feature_values = X_sample[feature_name].values
    
    # Create a DataFrame for easier plotting
    plot_df = pd.DataFrame({
        'Category': feature_values,
        'SHAP Value': feature_shap_values
    })
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Use violin plot to show distribution of SHAP values per category
    sns.violinplot(data=plot_df, x='Category', y='SHAP Value', inner='box')
    
    # Customize the plot
    plt.title(f'SHAP Value Distribution by {feature_name}', fontsize=14)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('SHAP Value (impact on model output)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add annotation
    plt.text(0.02, 0.98, 'Above 0 = increases prediction\nBelow 0 = decreases prediction', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created categorical SHAP plot for {feature_name}")


def create_mixed_shap_summary(shap_values, X_sample, categorical_features, output_path):
    """
    Create a SHAP summary plot that properly handles both numerical and categorical features.
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data
        categorical_features: List of categorical feature names
        output_path: Path to save the plot
    """
    # Separate features into numerical and categorical
    numerical_features = [col for col in X_sample.columns if col not in categorical_features]
    
    # Calculate mean absolute SHAP values for feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': mean_abs_shap,
        'is_categorical': [col in categorical_features for col in X_sample.columns]
    }).sort_values('importance', ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Traditional SHAP summary for numerical features
    ax1.set_title('Numerical Features - SHAP Summary', fontsize=14)
    numerical_indices = [i for i, col in enumerate(X_sample.columns) if col in numerical_features]
    if numerical_indices:
        # Create subset of SHAP values for numerical features only
        numerical_shap = shap_values[:, numerical_indices]
        numerical_data = X_sample[numerical_features]
        
        # Use SHAP's summary plot
        shap.summary_plot(numerical_shap, numerical_data, show=False, plot_size=None)
        plt.sca(ax1)
    
    # Right plot: Feature importance bar chart with categorical features highlighted
    ax2.set_title('All Features - Mean |SHAP Value|', fontsize=14)
    
    # Create color map
    colors = ['grey' if is_cat else 'steelblue' 
              for is_cat in feature_importance['is_categorical']]
    
    # Create horizontal bar plot
    top_features = feature_importance.head(20)
    ax2.barh(range(len(top_features)), top_features['importance'], color=colors[:20])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('Mean |SHAP Value|')
    ax2.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Numerical'),
        Patch(facecolor='grey', label='Categorical')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.suptitle('CatBoost SHAP Analysis - Numerical vs Categorical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created mixed SHAP summary plot")


def identify_categorical_features(X_sample, model=None):
    """
    Identify categorical features in the dataset.
    
    Args:
        X_sample: Sample data
        model: CatBoost model (optional, to get cat_features info)
        
    Returns:
        List of categorical feature names
    """
    categorical_features = []
    
    # Method 1: Check data types
    for col in X_sample.columns:
        if X_sample[col].dtype == 'object' or X_sample[col].dtype.name == 'category':
            categorical_features.append(col)
    
    # Method 2: Check for integer columns with few unique values (likely categorical)
    for col in X_sample.columns:
        if col not in categorical_features:
            if X_sample[col].dtype in ['int64', 'int32']:
                n_unique = X_sample[col].nunique()
                if n_unique < 20 and n_unique < len(X_sample) * 0.05:  # Less than 20 unique values or 5% of data
                    categorical_features.append(col)
    
    # Method 3: Known categorical columns from your dataset
    known_categorical = ['gics_sector', 'issuer_cntry_domicile', 'moodys_rating', 
                        'sp_rating', 'fitch_rating', 'tier']
    for col in known_categorical:
        if col in X_sample.columns and col not in categorical_features:
            categorical_features.append(col)
    
    return categorical_features


# Example usage
if __name__ == "__main__":
    print("This module provides improved SHAP visualizations for CatBoost with categorical features.")
    print("\nKey functions:")
    print("- create_categorical_shap_plot(): Creates violin plots for categorical features")
    print("- create_mixed_shap_summary(): Creates side-by-side plots for numerical and categorical features")
    print("- identify_categorical_features(): Automatically identifies categorical features")
    print("\nGrey dots in SHAP plots are not a bug - they indicate categorical features!")