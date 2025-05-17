#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update the ElasticNet CV RMSE distribution plot to remove confidence intervals,
matching the format of the other model plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

def save_figure(fig, filename, directory):
    """Save figure to specified directory with proper formatting."""
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Save figure
    fig.savefig(directory / f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"Saved figure to {directory / f'{filename}.png'}")
    plt.close(fig)

def update_elasticnet_cv_plot():
    """Update the ElasticNet CV RMSE distribution plot to remove confidence intervals."""
    # Define paths
    output_dir = Path(settings.OUTPUT_DIR)
    elasticnet_dir = output_dir / "visualizations" / "performance" / "elasticnet"
    model_path = Path(settings.MODEL_DIR) / "elasticnet_params.pkl"
    
    # Check if model data exists
    if not model_path.exists():
        print(f"ElasticNet model file not found: {model_path}")
        return
    
    # Load ElasticNet CV results
    try:
        cv_results = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
    except Exception as e:
        print(f"Error loading ElasticNet CV results: {e}")
        return
    
    # Prepare data for the plot
    rmse_data = []
    for result in cv_results:
        dataset = result['dataset']
        cv_df = result['cv_results']
        
        # Get fold RMSEs for each parameter combination
        for _, row in cv_df.iterrows():
            rmse_data.append({
                'Dataset': dataset,
                'RMSE': row['mean_rmse'],
                'Alpha': row['alpha'],
                'L1_Ratio': row['l1_ratio']
            })

    rmse_df = pd.DataFrame(rmse_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Boxplot for RMSE distribution - hide outliers
    box = sns.boxplot(x='Dataset', y='RMSE', data=rmse_df, palette='pastel', ax=ax, 
                     showfliers=False, width=0.6)
    
    # Stripplot for individual trials
    strip = sns.stripplot(x='Dataset', y='RMSE', data=rmse_df, color='gray', alpha=0.6,
                        jitter=True, ax=ax, size=5)
    
    # Plot mean as red points without error bars
    for i, dataset in enumerate(rmse_df['Dataset'].unique()):
        rmse_vals = rmse_df[rmse_df['Dataset'] == dataset]['RMSE']
        mean = np.mean(rmse_vals)
        
        # Plot mean value only
        ax.plot(i, mean, 'o', color='red', markersize=6)
    
    # Add title and labels
    ax.set_title("ElasticNet CV RMSE Distribution by Dataset", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend with no horizontal lines
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # For legend creation
    box_patch = plt.Rectangle((0, 0), 1, 1, fc="lightblue", edgecolor="black", linewidth=1)
    
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
             title='ElasticNet Cross-Validation Results')
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Save the updated plot
    save_figure(fig, "elasticnet_cv_rmse_distribution", elasticnet_dir)
    
    print("ElasticNet CV RMSE distribution plot updated successfully.")

if __name__ == "__main__":
    update_elasticnet_cv_plot()