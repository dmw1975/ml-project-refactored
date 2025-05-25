#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate CV RMSE distribution plots for XGBoost, CatBoost, and LightGBM models.
Similar to the ElasticNet CV RMSE distribution plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
from scipy import stats
import pickle

from config import settings
from utils import io

def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean and 95% confidence interval for the data."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def save_figure(fig, filename, directory=None):
    """Save figure to specified directory with proper formatting."""
    if directory is None:
        directory = Path(settings.OUTPUT_DIR) / "visualizations" / "performance"
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Save figure
    fig.savefig(directory / f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"Saved figure to {directory / f'{filename}.png'}")
    plt.close(fig)

def extract_cv_results_from_xgboost():
    """Extract CV results from XGBoost models."""
    filename = "xgboost_models.pkl"
    if not (Path(settings.MODEL_DIR) / filename).exists():
        print(f"XGBoost model file not found: {Path(settings.MODEL_DIR) / filename}")
        return None
    
    models = io.load_model(filename, settings.MODEL_DIR)
    
    cv_data = []
    for model_name, model_info in models.items():
        if model_name.endswith("_optuna"):
            # Process only optuna models
            dataset = model_info['model_name']
            study = model_info.get('study')
            
            if study is None:
                continue
                
            # Extract trials data
            for trial in study.trials:
                cv_mse = trial.user_attrs.get('mean_cv_mse')
                cv_mse_std = trial.user_attrs.get('std_cv_mse')
                
                if cv_mse is not None:
                    # Convert MSE to RMSE for consistency with other plots
                    rmse = np.sqrt(cv_mse)
                    
                    # Parameters from trial
                    params = trial.params
                    
                    cv_data.append({
                        'Dataset': dataset,
                        'RMSE': rmse,
                        'Trial': trial.number,
                        'Params': str(params)
                    })
    
    return pd.DataFrame(cv_data) if cv_data else None

def extract_cv_results_from_catboost():
    """Extract CV results from CatBoost models."""
    filename = "catboost_models.pkl"
    if not (Path(settings.MODEL_DIR) / filename).exists():
        print(f"CatBoost model file not found: {Path(settings.MODEL_DIR) / filename}")
        return None
    
    models = io.load_model(filename, settings.MODEL_DIR)
    
    cv_data = []
    for model_name, model_info in models.items():
        if model_name.endswith("_optuna"):
            # Process only optuna models
            dataset = model_info['model_name']
            study = model_info.get('study')
            
            if study is None:
                continue
                
            # Extract trials data
            for trial in study.trials:
                cv_mse = trial.user_attrs.get('mean_cv_mse')
                cv_mse_std = trial.user_attrs.get('std_cv_mse')
                
                if cv_mse is not None:
                    # Convert MSE to RMSE for consistency with other plots
                    rmse = np.sqrt(cv_mse)
                    
                    # Parameters from trial
                    params = trial.params
                    
                    cv_data.append({
                        'Dataset': dataset,
                        'RMSE': rmse,
                        'Trial': trial.number,
                        'Params': str(params)
                    })
    
    return pd.DataFrame(cv_data) if cv_data else None

def extract_cv_results_from_lightgbm():
    """Extract CV results from LightGBM models."""
    filename = "lightgbm_models.pkl"
    if not (Path(settings.MODEL_DIR) / filename).exists():
        print(f"LightGBM model file not found: {Path(settings.MODEL_DIR) / filename}")
        return None
    
    models = io.load_model(filename, settings.MODEL_DIR)
    
    cv_data = []
    for model_name, model_info in models.items():
        if model_name.endswith("_optuna"):
            # Process only optuna models
            dataset = model_info['model_name']
            study = model_info.get('study')
            
            if study is None:
                continue
                
            # Extract trials data
            for trial in study.trials:
                cv_mse = trial.user_attrs.get('mean_cv_mse')
                cv_mse_std = trial.user_attrs.get('std_cv_mse')
                
                if cv_mse is not None:
                    # Convert MSE to RMSE for consistency with other plots
                    rmse = np.sqrt(cv_mse)
                    
                    # Parameters from trial
                    params = trial.params
                    
                    cv_data.append({
                        'Dataset': dataset,
                        'RMSE': rmse,
                        'Trial': trial.number,
                        'Params': str(params)
                    })
    
    return pd.DataFrame(cv_data) if cv_data else None

def plot_cv_rmse_distribution(df, model_name):
    """Create CV RMSE distribution plot for the given model."""
    if df is None or df.empty:
        print(f"No CV data available for {model_name}")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Boxplot for RMSE distribution - hide outliers
    box = sns.boxplot(x='Dataset', y='RMSE', data=df, palette='pastel', ax=ax, 
                     showfliers=False, width=0.6)
    
    # Stripplot for individual trials
    strip = sns.stripplot(x='Dataset', y='RMSE', data=df, color='gray', alpha=0.6,
                        jitter=True, ax=ax, size=5)
    
    # For legend creation
    box_patch = plt.Rectangle((0, 0), 1, 1, fc="lightblue", edgecolor="black", linewidth=1)
    trial_dot = plt.Line2D([], [], marker='o', markerfacecolor='gray', markeredgecolor='gray',
                          markersize=8, alpha=0.6)
    
    # Plot mean as red points without error bars
    for i, dataset in enumerate(df['Dataset'].unique()):
        rmse_vals = df[df['Dataset'] == dataset]['RMSE']
        mean = np.mean(rmse_vals)
        
        # Plot mean value only
        ax.plot(i, mean, 'o', color='red', markersize=6)
    
    # Add title and labels
    ax.set_title(f"{model_name} CV RMSE Distribution by Dataset", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend with no horizontal lines
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        box_patch,
        Line2D([], [], marker='o', linestyle='None', markerfacecolor='gray', 
              markeredgecolor='gray', markersize=8, alpha=0.6),
        Line2D([], [], marker='o', linestyle='None', markerfacecolor='red', 
              markeredgecolor='red', markersize=8)
    ]
    
    # Legend labels
    legend_labels = [
        'RMSE Distribution (Boxplot)', 
        'Individual Trial RMSE',
        'Mean RMSE'
    ]
    
    # Add legend with all elements
    ax.legend(handles=legend_elements, labels=legend_labels, loc='upper right', 
             title=f'{model_name} Cross-Validation Results')
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Save figure to respective directory
    output_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "performance" / model_name.lower()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, f"{model_name.lower()}_cv_rmse_distribution", output_dir)
    
    return fig

def main():
    """Main function to generate CV plots for all models."""
    # XGBoost
    xgb_df = extract_cv_results_from_xgboost()
    plot_cv_rmse_distribution(xgb_df, "XGBoost")
    
    # CatBoost
    catboost_df = extract_cv_results_from_catboost()
    plot_cv_rmse_distribution(catboost_df, "CatBoost")
    
    # LightGBM
    lightgbm_df = extract_cv_results_from_lightgbm()
    plot_cv_rmse_distribution(lightgbm_df, "LightGBM")
    
    print("CV RMSE distribution plots generated successfully.")

if __name__ == "__main__":
    main()