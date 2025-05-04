"""Creative dataset comparison visualizations.

This module provides alternative, more intuitive and attractive visualizations
for comparing model performance across different datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import math

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import ModelViz, ComparativeViz
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.components.annotations import add_value_labels
from visualization_new.components.layouts import create_grid_layout, create_comparison_layout
from visualization_new.components.formats import format_figure_for_export, save_figure
from visualization_new.utils.io import load_all_models, ensure_dir


def extract_model_metrics(models: List[ModelData]) -> pd.DataFrame:
    """
    Extract metrics from models into a DataFrame for easier comparison.
    
    Args:
        models: List of model adapters
        
    Returns:
        pd.DataFrame: DataFrame with model metrics
    """
    # Create DataFrame from model metrics
    model_data = []
    
    for model in models:
        # Get model name and metadata
        metadata = model.get_metadata()
        model_name = metadata.get('model_name', 'Unknown')
        model_type = metadata.get('model_type', 'Unknown')
        
        # Extract metrics
        metrics = model.get_metrics()
        
        # Basic model info
        model_info = {
            'model_name': model_name,
            'model_type': model_type
        }
        
        # Parse dataset from model name
        if 'Base_Random' in model_name:
            dataset = 'Base_Random'
        elif 'Yeo_Random' in model_name:
            dataset = 'Yeo_Random'
        elif 'Base' in model_name:
            dataset = 'Base'
        elif 'Yeo' in model_name:
            dataset = 'Yeo'
        else:
            dataset = 'Unknown'
        
        model_info['dataset'] = dataset
        
        # Determine model family
        if 'LR_' in model_name or 'ElasticNet' in model_name:
            model_family = 'Linear'
        elif 'XGB' in model_name:
            model_family = 'XGBoost'
        elif 'LightGBM' in model_name:
            model_family = 'LightGBM'
        elif 'CatBoost' in model_name:
            model_family = 'CatBoost'
        else:
            model_family = 'Other'
        
        model_info['model_family'] = model_family
        
        # Determine if model is tuned
        is_tuned = False
        if 'ElasticNet' in model_name:
            is_tuned = True  # ElasticNet models are always tuned in this project
        elif 'optuna' in model_name.lower():
            is_tuned = True
        
        model_info['is_tuned'] = is_tuned
        model_info['tuning_status'] = 'Tuned' if is_tuned else 'Basic'
        
        # Add metrics
        model_info.update(metrics)
        
        # Create display name for plots
        model_info['display_name'] = f"{model_family} ({'Tuned' if is_tuned else 'Basic'})"
        
        # Add to list
        model_data.append(model_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(model_data)
    
    return df


def create_radar_chart(models: List[ModelData], config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """
    Create a radar/spider chart visualization for model comparison.
    
    Args:
        models: List of model adapters
        config: Visualization configuration
        
    Returns:
        plt.Figure: Figure with radar chart
    """
    # Process configuration
    if config is None:
        config = VisualizationConfig()
    
    # Extract metrics to DataFrame
    df = extract_model_metrics(models)
    
    # Filter to a specific dataset if provided
    dataset = config.get('dataset', None)
    if dataset is not None:
        df = df[df['dataset'] == dataset]
    
    # If no models for this dataset, return empty figure
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"No models found for dataset: {dataset}", 
                ha='center', va='center', fontsize=14)
        plt.close()
        return fig
    
    # Define metrics to include (and whether lower or higher is better)
    metrics = [
        ('RMSE', False),  # (name, higher_is_better)
        ('MAE', False),
        ('R2', True),
        ('MSE', False)
    ]
    
    # Set up colors for model families
    family_colors = {
        'Linear': '#9b59b6',    # Purple
        'XGBoost': '#3498db',   # Blue
        'LightGBM': '#2ecc71',  # Green
        'CatBoost': '#e74c3c'   # Red
    }
    
    # Set up line styles for tuning status
    tuning_styles = {
        'Basic': 'dashed',
        'Tuned': 'solid'
    }
    
    # Group by model family and tuning status
    grouped = df.groupby(['model_family', 'tuning_status'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    # Create a radar chart for each dataset
    datasets = sorted(df['dataset'].unique())
    
    for i, dataset_name in enumerate(datasets):
        if i >= len(axes):
            break
            
        ax = axes[i]
        dataset_df = df[df['dataset'] == dataset_name]
        
        # If no models for this dataset, skip
        if len(dataset_df) == 0:
            ax.text(0, 0, f"No models for {dataset_name}", ha='center', va='center')
            continue
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        # Close the polygon
        angles += angles[:1]
        
        # Set up labels and normalize values
        labels = [m[0] for m in metrics]
        
        # Add directions to metric labels
        label_texts = []
        for metric, higher_is_better in metrics:
            label_texts.append(f"{metric}\n({'↑' if higher_is_better else '↓'} better)")
        
        # Plot each model group
        for (family, tuning), group in grouped:
            if len(group[group['dataset'] == dataset_name]) == 0:
                continue
                
            # Get model for this dataset
            model_df = group[group['dataset'] == dataset_name]
            if len(model_df) == 0:
                continue
            
            # Get metrics for this model
            values = []
            for metric, higher_is_better in metrics:
                if metric not in model_df.columns:
                    values.append(0)
                    continue
                    
                value = model_df[metric].values[0]
                
                # Normalize and invert values if needed
                if higher_is_better:
                    # For R², higher is better, so use the raw value
                    # R² is already in 0-1 range typically
                    norm_value = max(0, min(1, value))
                else:
                    # For error metrics, lower is better, so invert
                    # Scale the value based on the range in the dataset
                    all_values = df[df['dataset'] == dataset_name][metric]
                    if len(all_values) == 0 or all_values.max() == all_values.min():
                        norm_value = 0.5  # Default if no range
                    else:
                        # Normalize to 0-1 range and invert (1 = best, 0 = worst)
                        norm_value = 1 - (value - all_values.min()) / (all_values.max() - all_values.min())
                
                values.append(norm_value)
            
            # Close the polygon
            values += values[:1]
            
            # Plot the values
            ax.plot(angles, values, 
                    linewidth=2, 
                    linestyle=tuning_styles[tuning], 
                    color=family_colors.get(family, '#7f8c8d'),
                    label=f"{family} ({tuning})")
            ax.fill(angles, values, 
                    alpha=0.1, 
                    color=family_colors.get(family, '#7f8c8d'))
        
        # Set up radar chart layout
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(label_texts)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25', '0.5', '0.75'])
        ax.set_ylim(0, 1)
        
        # Add title
        ax.set_title(f"{dataset_name} Dataset", fontsize=14, pad=15)
    
    # Create a single legend for the entire figure
    handles = []
    for family, color in family_colors.items():
        for tuning, style in tuning_styles.items():
            handles.append(plt.Line2D([0], [0], color=color, linestyle=style, 
                                       label=f"{family} ({tuning})"))
    
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    # Add title to the figure
    plt.suptitle("Model Performance Comparison - Radar Chart", fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure if output directory is provided
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        ensure_dir(output_dir)
        
        dataset_str = f"_{dataset}" if dataset else "_all_datasets"
        filename = f"radar_chart{dataset_str}"
        
        save_figure(fig, filename, output_dir, 
                   dpi=config.get('dpi', 300),
                   format=config.get('format', 'png'))
    
    return fig


def create_performance_delta(models: List[ModelData], config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """
    Create a visualization showing performance improvements from basic to tuned models.
    
    Args:
        models: List of model adapters
        config: Visualization configuration
        
    Returns:
        plt.Figure: Figure with performance delta visualization
    """
    # Process configuration
    if config is None:
        config = VisualizationConfig()
    
    # Extract metrics to DataFrame
    df = extract_model_metrics(models)
    
    # Select specific dataset if provided
    dataset = config.get('dataset', None)
    if dataset is not None:
        df = df[df['dataset'] == dataset]
    
    # Define metrics to include (and whether lower or higher is better)
    metrics = [
        ('RMSE', False),  # (name, higher_is_better)
        ('MAE', False),
        ('R2', True),
        ('MSE', False)
    ]
    
    # Set up colors for model families
    family_colors = {
        'Linear': '#9b59b6',    # Purple
        'XGBoost': '#3498db',   # Blue
        'LightGBM': '#2ecc71',  # Green
        'CatBoost': '#e74c3c'   # Red
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # For each metric, calculate delta between basic and tuned models
    for i, (metric, higher_is_better) in enumerate(metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Group by dataset and model family
        deltas = []
        
        for dataset_name in sorted(df['dataset'].unique()):
            dataset_df = df[df['dataset'] == dataset_name]
            
            for family in sorted(dataset_df['model_family'].unique()):
                family_df = dataset_df[dataset_df['model_family'] == family]
                
                # Get basic and tuned models
                basic = family_df[family_df['tuning_status'] == 'Basic']
                tuned = family_df[family_df['tuning_status'] == 'Tuned']
                
                if len(basic) == 0 or len(tuned) == 0 or metric not in basic.columns or metric not in tuned.columns:
                    continue
                
                basic_value = basic[metric].values[0]
                tuned_value = tuned[metric].values[0]
                
                # Calculate percent improvement
                if higher_is_better:
                    # For R², higher is better
                    if basic_value == 0 or basic_value < 0.00001:
                        # Avoid division by zero
                        pct_change = float('inf') if tuned_value > 0 else float('-inf')
                    else:
                        pct_change = ((tuned_value - basic_value) / abs(basic_value)) * 100
                else:
                    # For error metrics, lower is better
                    if basic_value == 0:
                        # Avoid division by zero
                        pct_change = float('-inf') if tuned_value > 0 else float('inf')
                    else:
                        # Negative means improvement (lower error)
                        pct_change = ((basic_value - tuned_value) / basic_value) * 100
                
                # Cap the percentage improvement for display purposes
                pct_change = max(-100, min(100, pct_change))
                
                deltas.append({
                    'dataset': dataset_name,
                    'model_family': family,
                    'metric': metric,
                    'basic_value': basic_value,
                    'tuned_value': tuned_value,
                    'pct_change': pct_change,
                    'abs_change': tuned_value - basic_value if higher_is_better else basic_value - tuned_value
                })
        
        # Convert to DataFrame
        delta_df = pd.DataFrame(deltas)
        
        if len(delta_df) == 0:
            ax.text(0.5, 0.5, f"No data available for {metric}", 
                    ha='center', va='center', fontsize=14)
            continue
            
        # Filter to the current metric
        metric_df = delta_df[delta_df['metric'] == metric]
        
        if len(metric_df) == 0:
            ax.text(0.5, 0.5, f"No data available for {metric}", 
                    ha='center', va='center', fontsize=14)
            continue
        
        # Set up bar positions
        datasets = sorted(metric_df['dataset'].unique())
        families = sorted(metric_df['model_family'].unique())
        
        bar_width = 0.8 / len(families)
        positions = np.arange(len(datasets))
        
        # Plot bars for each model family
        for j, family in enumerate(families):
            family_data = metric_df[metric_df['model_family'] == family]
            
            # Map dataset names to positions
            values = []
            for dataset_name in datasets:
                dataset_rows = family_data[family_data['dataset'] == dataset_name]
                if len(dataset_rows) > 0:
                    values.append(dataset_rows['pct_change'].values[0])
                else:
                    values.append(0)
            
            # Calculate bar positions
            pos = positions + (j - len(families)/2 + 0.5) * bar_width
            
            # Create bars
            bars = ax.bar(pos, values, width=bar_width, 
                         color=family_colors.get(family, '#7f8c8d'),
                         label=family)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value) and not np.isinf(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2,
                           height + np.sign(height) * 1,
                           f"{value:.1f}%",
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=9)
        
        # Set up axes
        ax.set_xticks(positions)
        ax.set_xticklabels(datasets)
        ax.set_xlabel('Dataset')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Define y-axis based on metric
        if higher_is_better:
            ax.set_ylabel(f'Improvement in {metric} (%)\n(higher is better)')
        else:
            ax.set_ylabel(f'Reduction in {metric} (%)\n(higher is better)')
        
        # Set title
        ax.set_title(f"Performance Improvement: {metric}", fontsize=14)
        
        # Add legend
        ax.legend(title="Model Family")
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
    
    # Add title to the figure
    plt.suptitle("Performance Improvement from Basic to Tuned Models", fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if output directory is provided
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        ensure_dir(output_dir)
        
        dataset_str = f"_{dataset}" if dataset else "_all_datasets"
        filename = f"performance_delta{dataset_str}"
        
        save_figure(fig, filename, output_dir, 
                   dpi=config.get('dpi', 300),
                   format=config.get('format', 'png'))
    
    return fig


def create_parallel_coordinates(models: List[ModelData], config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """
    Create a parallel coordinates plot for comparing models across metrics.
    
    Args:
        models: List of model adapters
        config: Visualization configuration
        
    Returns:
        plt.Figure: Figure with parallel coordinates plot
    """
    # Process configuration
    if config is None:
        config = VisualizationConfig()
    
    # Import pandas plotting
    from pandas.plotting import parallel_coordinates
    
    # Extract metrics to DataFrame
    df = extract_model_metrics(models)
    
    # Select specific dataset if provided
    dataset = config.get('dataset', None)
    if dataset is not None:
        df = df[df['dataset'] == dataset]
    
    # If no models for this dataset, return empty figure
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"No models found for dataset: {dataset}", 
                ha='center', va='center', fontsize=14)
        plt.close()
        return fig
    
    # Define metrics to include
    metrics = ['RMSE', 'MAE', 'R2', 'MSE']
    
    # Filter to only the needed columns
    # Ensure all required metrics are present
    for metric in metrics:
        if metric not in df.columns:
            df[metric] = np.nan
    
    # Create a copy with only needed columns
    plot_df = df[['model_family', 'tuning_status', 'display_name', 'dataset'] + metrics].copy()
    
    # Create figure
    fig, axes = plt.subplots(1, len(df['dataset'].unique()), figsize=(16, 6))
    
    # Handle case where there's only one dataset
    if len(df['dataset'].unique()) == 1:
        axes = [axes]
    
    # Create a parallel coordinates plot for each dataset
    for i, dataset_name in enumerate(sorted(df['dataset'].unique())):
        if i >= len(axes):
            break
            
        ax = axes[i]
        dataset_df = plot_df[plot_df['dataset'] == dataset_name]
        
        # If no models for this dataset, skip
        if len(dataset_df) == 0:
            ax.text(0.5, 0.5, f"No models for {dataset_name}", ha='center', va='center')
            continue
        
        # Normalize the metrics to 0-1 range for easier comparison
        for metric in metrics:
            if df[metric].isna().all():
                continue
                
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            if max_val > min_val:
                # For R^2, higher is better, so normalize directly
                if metric == 'R2':
                    dataset_df[metric] = (dataset_df[metric] - min_val) / (max_val - min_val)
                # For error metrics, lower is better, so invert the normalization
                else:
                    dataset_df[metric] = 1 - (dataset_df[metric] - min_val) / (max_val - min_val)
        
        # Create the parallel coordinates plot
        # We'll do this manually since the pandas version is limited
        
        # Set up colors for model families
        family_colors = {
            'Linear': '#9b59b6',    # Purple
            'XGBoost': '#3498db',   # Blue
            'LightGBM': '#2ecc71',  # Green
            'CatBoost': '#e74c3c'   # Red
        }
        
        # Set up line styles for tuning status
        tuning_styles = {
            'Basic': 'dashed',
            'Tuned': 'solid'
        }
        
        # Set up x positions
        x_pos = np.arange(len(metrics))
        
        # Plot each model
        for idx, row in dataset_df.iterrows():
            family = row['model_family']
            tuning = row['tuning_status']
            
            # Get values to plot
            values = []
            for metric in metrics:
                values.append(row[metric])
            
            # Check if values are valid
            if any(np.isnan(values)):
                continue
            
            # Plot the line
            ax.plot(x_pos, values, 
                   color=family_colors.get(family, '#7f8c8d'),
                   linestyle=tuning_styles.get(tuning, 'solid'),
                   label=row['display_name'],
                   marker='o')
        
        # Set up axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.set_ylim(-0.05, 1.05)  # Allow a bit of padding
        ax.set_ylabel('Normalized Performance (higher is better)')
        
        # Add vertical grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add title
        ax.set_title(f"{dataset_name} Dataset", fontsize=14)
        
    # Create a single legend for the entire figure
    handles = []
    for family, color in family_colors.items():
        for tuning, style in tuning_styles.items():
            handles.append(plt.Line2D([0], [0], color=color, linestyle=style, 
                                     label=f"{family} ({tuning})"))
    
    # Place legend outside the plots
    fig.legend(handles=handles, loc='upper center', ncol=4, 
              bbox_to_anchor=(0.5, 0.05))
    
    # Add overall title
    plt.suptitle("Model Performance - Parallel Coordinates", fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save figure if output directory is provided
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        ensure_dir(output_dir)
        
        dataset_str = f"_{dataset}" if dataset else "_all_datasets"
        filename = f"parallel_coordinates{dataset_str}"
        
        save_figure(fig, filename, output_dir, 
                   dpi=config.get('dpi', 300),
                   format=config.get('format', 'png'))
    
    return fig


def create_visual_leaderboard(models: List[ModelData], config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """
    Create a visual leaderboard for model comparison.
    
    Args:
        models: List of model adapters
        config: Visualization configuration
        
    Returns:
        plt.Figure: Figure with visual leaderboard
    """
    # Process configuration
    if config is None:
        config = VisualizationConfig()
    
    # Extract metrics to DataFrame
    df = extract_model_metrics(models)
    
    # Select specific dataset if provided
    dataset = config.get('dataset', None)
    if dataset is not None:
        df = df[df['dataset'] == dataset]
        
    # Define metrics to include (and whether lower or higher is better)
    metrics = [
        ('RMSE', False),  # (name, higher_is_better)
        ('MAE', False),
        ('R2', True),
        ('MSE', False)
    ]
    
    # Set up colors for model families
    family_colors = {
        'Linear': '#9b59b6',    # Purple
        'XGBoost': '#3498db',   # Blue
        'LightGBM': '#2ecc71',  # Green
        'CatBoost': '#e74c3c'   # Red
    }
    
    # Create figure (one per dataset)
    datasets = sorted(df['dataset'].unique())
    fig_width = 18
    fig_height = 3 + len(datasets) * 5
    
    fig, axes = plt.subplots(len(datasets), 1, figsize=(fig_width, fig_height))
    
    # Handle case where there's only one dataset
    if len(datasets) == 1:
        axes = [axes]
    
    # For each dataset, create a leaderboard
    for i, dataset_name in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset_name]
        
        if len(dataset_df) == 0:
            if i < len(axes):
                axes[i].text(0.5, 0.5, f"No models found for dataset: {dataset_name}", 
                           ha='center', va='center', fontsize=14)
            continue
        
        # Create a grid of bar charts (1 row x 4 columns)
        # Using subplot inside subplot to arrange metrics
        gs = plt.GridSpec(1, len(metrics), wspace=0.35, hspace=0.25, 
                         left=0.05, right=0.95, bottom=0.1, top=0.9)
        
        for j, (metric, higher_is_better) in enumerate(metrics):
            # Create subplot
            sub_ax = fig.add_subplot(gs[i, j])
            
            # Filter out models missing this metric
            metric_df = dataset_df.dropna(subset=[metric]).copy()
            
            if len(metric_df) == 0:
                sub_ax.text(0.5, 0.5, f"No data for {metric}", 
                           ha='center', va='center', fontsize=10)
                continue
            
            # Sort by metric (ascending or descending based on better_is_higher)
            metric_df = metric_df.sort_values(metric, ascending=not higher_is_better)
            
            # Create horizontal bars
            position = np.arange(len(metric_df))
            bars = sub_ax.barh(
                position,
                metric_df[metric],
                height=0.6,
                color=[family_colors.get(family, '#7f8c8d') for family in metric_df['model_family']]
            )
            
            # Add model names as y-tick labels
            sub_ax.set_yticks(position)
            sub_ax.set_yticklabels([f"{row['model_family']} ({'T' if row['is_tuned'] else 'B'})" 
                                  for _, row in metric_df.iterrows()])
            
            # Add value labels to bars
            for idx, bar in enumerate(bars):
                width = bar.get_width()
                sub_ax.text(
                    width * 1.01, 
                    bar.get_y() + bar.get_height()/2,
                    f"{width:.4f}",
                    va='center',
                    fontsize=8
                )
            
            # Add model ranking numbers
            for idx, bar in enumerate(bars):
                sub_ax.text(
                    bar.get_width() * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f"#{idx+1}",
                    va='center',
                    ha='left',
                    color='white',
                    fontweight='bold',
                    fontsize=9
                )
            
            # Add medal emoji for top 3
            medal_emojis = ["🥇", "🥈", "🥉"]
            for idx, bar in enumerate(bars[:3]):
                if idx < len(medal_emojis):
                    sub_ax.text(
                        0, 
                        bar.get_y() - 0.3,
                        medal_emojis[idx],
                        va='center',
                        ha='left',
                        fontsize=12
                    )
            
            # Set title
            direction = "↑" if higher_is_better else "↓"
            sub_ax.set_title(f"{metric} ({direction} better)", fontsize=12)
            
            # Remove y-axis label for all but the first metric
            if j > 0:
                sub_ax.set_yticklabels([])
            
            # Set grid
            sub_ax.grid(axis='x', alpha=0.3)
            
            # Set tight layout
            sub_ax.set_xlim(0, metric_df[metric].max() * 1.15)
        
        # Add dataset title to the first column
        if i < len(axes):
            axes[i].text(0.02, 0.5, f"{dataset_name} Dataset", 
                       rotation=90, ha='center', va='center', 
                       fontsize=16, transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Add overall title
    plt.suptitle("Model Performance Leaderboard", fontsize=18, y=0.99)
    
    # Save figure if output directory is provided
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        ensure_dir(output_dir)
        
        dataset_str = f"_{dataset}" if dataset else "_all_datasets"
        filename = f"visual_leaderboard{dataset_str}"
        
        save_figure(fig, filename, output_dir, 
                   dpi=config.get('dpi', 300),
                   format=config.get('format', 'png'))
    
    return fig


def create_sunburst_chart(models: List[ModelData], config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """
    Create a sunburst chart for hierarchical visualization of model performance.
    
    Args:
        models: List of model adapters
        config: Visualization configuration
        
    Returns:
        plt.Figure: Figure with sunburst chart
    """
    # Import plotly for this visualization (dynamic import because it's optional)
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "This visualization requires plotly to be installed.\nPlease run: pip install plotly", 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Process configuration
    if config is None:
        config = VisualizationConfig()
    
    # Extract metrics to DataFrame
    df = extract_model_metrics(models)
    
    # Select specific dataset if provided
    dataset = config.get('dataset', None)
    if dataset is not None:
        df = df[df['dataset'] == dataset]
    
    # Create lists to build the sunburst chart
    labels = []
    parents = []
    values = []
    colors = []
    hovers = []
    
    # Define a metric to use for the values
    metric = config.get('metric', 'RMSE')
    higher_is_better = metric == 'R2'  # R2 is the only metric where higher is better
    
    # Set up colors for model families
    family_colors = {
        'Linear': '#9b59b6',    # Purple
        'XGBoost': '#3498db',   # Blue
        'LightGBM': '#2ecc71',  # Green
        'CatBoost': '#e74c3c'   # Red
    }
    
    # Create the root node
    labels.append('All Models')
    parents.append('')
    values.append(100)  # Placeholder
    colors.append('#7f8c8d')  # Gray
    hovers.append('All Models')
    
    # Add dataset nodes
    for dataset_name in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset_name]
        
        if len(dataset_df) == 0:
            continue
        
        # Calculate average metric value for this dataset
        if metric in dataset_df.columns:
            avg_value = dataset_df[metric].mean()
        else:
            avg_value = None
        
        # Add dataset node
        labels.append(dataset_name)
        parents.append('All Models')
        values.append(100 // len(df['dataset'].unique()))  # Equal size for all datasets
        colors.append('#bdc3c7')  # Light gray
        
        if avg_value is not None:
            hovers.append(f"{dataset_name}<br>Avg {metric}: {avg_value:.4f}")
        else:
            hovers.append(dataset_name)
        
        # Add model family nodes
        for family in sorted(dataset_df['model_family'].unique()):
            family_df = dataset_df[dataset_df['model_family'] == family]
            
            # Calculate average metric value for this family
            if metric in family_df.columns:
                avg_value = family_df[metric].mean()
            else:
                avg_value = None
            
            # Add family node
            family_id = f"{dataset_name}_{family}"
            labels.append(family_id)
            parents.append(dataset_name)
            values.append(100 // (len(df['dataset'].unique()) * len(dataset_df['model_family'].unique())))
            colors.append(family_colors.get(family, '#7f8c8d'))
            
            if avg_value is not None:
                hovers.append(f"{family}<br>Avg {metric}: {avg_value:.4f}")
            else:
                hovers.append(family)
            
            # Add model nodes
            for _, row in family_df.iterrows():
                model_name = row['model_name']
                is_tuned = row['is_tuned']
                
                # Get model metric value
                if metric in row and not pd.isna(row[metric]):
                    metric_value = row[metric]
                else:
                    metric_value = None
                
                # Create model label
                model_label = f"{model_name} ({'Tuned' if is_tuned else 'Basic'})"
                
                # Add model node
                labels.append(model_label)
                parents.append(family_id)
                
                # Calculate value - for leaf nodes, use actual metric value if available
                if metric_value is not None:
                    if higher_is_better:
                        # For R2, higher is better
                        # Scale to keep node visible (min 1/5 of parent)
                        normalized = max(0.2, min(1.0, metric_value))
                    else:
                        # For error metrics, lower is better
                        # Get min/max for normalization
                        min_val = df[metric].min()
                        max_val = df[metric].max()
                        
                        if max_val > min_val:
                            # Normalize and invert (1 = best, 0 = worst)
                            normalized = 1 - (metric_value - min_val) / (max_val - min_val)
                            # Scale to keep node visible (min 1/5 of parent)
                            normalized = max(0.2, min(1.0, normalized))
                        else:
                            normalized = 0.5
                    
                    # Scale value relative to parent
                    values.append(values[-1] * normalized * 0.8)
                else:
                    # Default size if metric not available
                    values.append(values[-1] / 2)
                
                # Set color - darker for tuned models
                base_color = family_colors.get(family, '#7f8c8d')
                if is_tuned:
                    colors.append(base_color)
                else:
                    # Lighter version of the color
                    rgb = mcolors.to_rgb(base_color)
                    lighter = (min(1, rgb[0] * 1.3), min(1, rgb[1] * 1.3), min(1, rgb[2] * 1.3))
                    colors.append(mcolors.to_hex(lighter))
                
                # Create hover text
                hover_text = f"{model_name}<br>{'Tuned' if is_tuned else 'Basic'}"
                if metric_value is not None:
                    hover_text += f"<br>{metric}: {metric_value:.4f}"
                hovers.append(hover_text)
    
    # Create plotly figure
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        hovertext=hovers,
        hoverinfo="text",
        branchvalues="total"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Hierarchical View of Model Performance<br><sub>{metric} {'(higher is better)' if higher_is_better else '(lower is better)'}</sub>",
        width=1000,
        height=1000,
        margin=dict(t=80, l=0, r=0, b=0)
    )
    
    # Save figure if output directory is provided
    if 'output_dir' in config:
        output_dir = Path(config['output_dir'])
        ensure_dir(output_dir)
        
        dataset_str = f"_{dataset}" if dataset else "_all_datasets"
        metric_str = f"_{metric.lower()}"
        filename = f"sunburst_chart{dataset_str}{metric_str}"
        
        # Save as HTML for interactive version
        html_path = output_dir / f"{filename}.html"
        fig.write_html(str(html_path))
        
        # Save static image as well
        img_path = output_dir / f"{filename}.png"
        fig.write_image(str(img_path))
        
        print(f"Sunburst chart saved to:\n- {html_path} (interactive)\n- {img_path} (static)")
    
    # Create a static version for matplotlib compatibility
    plt_fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, "Sunburst Chart (See HTML output for interactive version)", 
           ha='center', va='center', fontsize=14)
    
    return plt_fig