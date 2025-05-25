"""Dataset-centric model comparison visualizations (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new package instead.",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def load_all_models_for_comparison():
    """Load all models from different algorithms and combine into one DataFrame."""
    all_model_data = {}
    
    # Load linear regression models
    try:
        linear_models = io.load_model("linear_regression_models.pkl", settings.MODEL_DIR)
        for name, model in linear_models.items():
            all_model_data[name] = model
        print(f"Loaded {len(linear_models)} linear regression models")
    except Exception as e:
        print(f"Error loading linear regression models: {e}")
    
    # Load ElasticNet models
    try:
        elastic_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
        for name, model in elastic_models.items():
            all_model_data[name] = model
        print(f"Loaded {len(elastic_models)} ElasticNet models")
    except Exception as e:
        print(f"Error loading ElasticNet models: {e}")
    
    # Load XGBoost models
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
        for name, model in xgboost_models.items():
            all_model_data[name] = model
        print(f"Loaded {len(xgboost_models)} XGBoost models")
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
    
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        for name, model in lightgbm_models.items():
            all_model_data[name] = model
        print(f"Loaded {len(lightgbm_models)} LightGBM models")
    except Exception as e:
        print(f"Error loading LightGBM models: {e}")
    
    # Load CatBoost models
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
        for name, model in catboost_models.items():
            all_model_data[name] = model
        print(f"Loaded {len(catboost_models)} CatBoost models")
    except Exception as e:
        print(f"Error loading CatBoost models: {e}")
    
    return all_model_data

def extract_model_metrics(all_models):
    """Extract metrics from all models into a DataFrame for easier comparison."""
    model_metrics = []
    
    for model_name, model_data in all_models.items():
        # Skip if missing metrics
        if 'RMSE' not in model_data or 'R2' not in model_data or 'MAE' not in model_data:
            print(f"Skipping {model_name}: Missing metrics")
            continue
        
        # Determine dataset type
        if 'Base_Random' in model_name:
            dataset = 'Base_Random'
        elif 'Base' in model_name:
            dataset = 'Base'
        elif 'Yeo_Random' in model_name:
            dataset = 'Yeo_Random'
        elif 'Yeo' in model_name:
            dataset = 'Yeo'
        else:
            dataset = 'Unknown'
        
        # Determine model family and tuning status
        if model_name.startswith('LR_'):
            model_family = 'Linear'
            tuned = False
        elif model_name.startswith('ElasticNet_'):
            model_family = 'Linear'
            tuned = True
        elif 'XGB' in model_name:
            model_family = 'XGBoost'
            tuned = 'optuna' in model_name
        elif 'LightGBM' in model_name:
            model_family = 'LightGBM'
            tuned = 'optuna' in model_name
        elif 'CatBoost' in model_name:
            model_family = 'CatBoost'
            tuned = 'optuna' in model_name
        else:
            model_family = 'Unknown'
            tuned = False
            
        # Calculate MSE from RMSE if not available
        if 'MSE' not in model_data:
            mse = model_data['RMSE'] ** 2
        else:
            mse = model_data['MSE']
            
        # Create a record with all the information
        model_metrics.append({
            'model_name': model_name,
            'dataset': dataset,
            'model_family': model_family,
            'tuned': tuned,
            'RMSE': model_data['RMSE'],
            'MAE': model_data['MAE'],
            'R2': model_data['R2'],
            'MSE': mse
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    return metrics_df

def plot_dataset_comparison(dataset_name, metrics_df, output_dir=None):
    """
    Create a 2x2 grid of plots showing different metrics for all models on a specific dataset,
    grouped by model family with basic and tuned variants shown side-by-side.
    
    Args:
        dataset_name: Name of the dataset to filter by (e.g., 'Base', 'Yeo')
        metrics_df: DataFrame containing model metrics
        output_dir: Directory to save the plot
    """
    # Set up style
    style = setup_visualization_style()
    
    # Filter metrics for the specified dataset
    dataset_metrics = metrics_df[metrics_df['dataset'] == dataset_name]
    
    if dataset_metrics.empty:
        print(f"No models found for dataset {dataset_name}")
        return None
    
    # Define model families and their colors
    model_families = ['Linear', 'XGBoost', 'LightGBM', 'CatBoost']
    family_colors = {
        'Linear': '#9b59b6',   # Purple
        'XGBoost': '#3498db',  # Blue
        'LightGBM': '#2ecc71', # Green
        'CatBoost': '#e74c3c', # Red
        'Unknown': '#95a5a6'   # Gray
    }
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Set up axes for each metric
    metrics = [
        {'name': 'RMSE', 'ax': axes[0, 0], 'title': 'RMSE Comparison', 'ylabel': 'RMSE (lower is better)', 'best': 'min'},
        {'name': 'MAE', 'ax': axes[0, 1], 'title': 'MAE Comparison', 'ylabel': 'MAE (lower is better)', 'best': 'min'},
        {'name': 'R2', 'ax': axes[1, 0], 'title': 'R² Comparison', 'ylabel': 'R² (higher is better)', 'best': 'max'},
        {'name': 'MSE', 'ax': axes[1, 1], 'title': 'MSE Comparison', 'ylabel': 'MSE (lower is better)', 'best': 'min'}
    ]
    
    # Create subplots for each metric
    for metric in metrics:
        ax = metric['ax']
        metric_name = metric['name']
        best_fn = min if metric['best'] == 'min' else max
        
        # Group data by model family
        x_positions = []
        x_labels = []
        bar_colors = []
        bar_values = []
        best_value = best_fn(dataset_metrics[metric_name])
        
        # Add models with consistent spacing between families
        family_gap = 0.5  # Gap between different model families
        bar_width = 0.35  # Width of each bar
        pair_center = 0   # Center position for each basic/tuned pair
        
        for family in model_families:
            # Filter models for this family
            family_models = dataset_metrics[dataset_metrics['model_family'] == family]
            
            if len(family_models) == 0:
                continue
                
            # Get basic and tuned models
            basic_model = family_models[family_models['tuned'] == False]
            tuned_model = family_models[family_models['tuned'] == True]
            
            # Check if we have models
            has_basic = not basic_model.empty
            has_tuned = not tuned_model.empty
            
            # Calculate positions
            if has_basic and has_tuned:
                # Both models exist - place side by side
                x_positions.extend([pair_center - bar_width/2, pair_center + bar_width/2])
                
                # Add values
                bar_values.extend([
                    basic_model.iloc[0][metric_name] if has_basic else 0,
                    tuned_model.iloc[0][metric_name] if has_tuned else 0
                ])
                
                # Add colors
                color = family_colors.get(family, family_colors['Unknown'])
                bar_colors.extend([color, color])
                
                # Labels
                x_labels.extend(['', ''])
                
                # Update center for next family
                pair_center += family_gap + 1
            elif has_basic:
                # Only basic model exists
                x_positions.append(pair_center)
                bar_values.append(basic_model.iloc[0][metric_name])
                color = family_colors.get(family, family_colors['Unknown'])
                bar_colors.append(color)
                x_labels.append('')
                pair_center += family_gap + 1
            elif has_tuned:
                # Only tuned model exists
                x_positions.append(pair_center)
                bar_values.append(tuned_model.iloc[0][metric_name])
                color = family_colors.get(family, family_colors['Unknown'])
                bar_colors.append(color)
                x_labels.append('')
                pair_center += family_gap + 1
        
        # Create bars
        bars = ax.bar(x_positions, bar_values, width=bar_width, color=bar_colors)
        
        # Add value labels
        for bar, value in zip(bars, bar_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Highlight the best model
            if (metric['best'] == 'min' and value == best_fn(bar_values)) or \
               (metric['best'] == 'max' and value == best_fn(bar_values)):
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        # Add family labels below x-axis
        family_positions = []
        shown_families = []
        
        pair_center = 0
        for family in model_families:
            family_models = dataset_metrics[dataset_metrics['model_family'] == family]
            if len(family_models) > 0:
                family_positions.append(pair_center)
                shown_families.append(family)
                pair_center += family_gap + 1
        
        # Set x-axis properties
        ax.set_xticks(family_positions)
        ax.set_xticklabels(shown_families, fontsize=12)
        
        # Add a legend for basic vs tuned
        basic_patch = plt.Rectangle((0, 0), 1, 1, fc='none', ec='none', label='Basic')
        tuned_patch = plt.Rectangle((0, 0), 1, 1, fc='none', ec='none', label='Tuned')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='none', ec='none', label=' '),
            plt.Rectangle((0, 0), 1, 1, fc='none', ec='black', linewidth=2, label='Best Model')
        ]
        
        # Add shapes to indicate basic/tuned positioning
        ax.annotate('Basic', xy=(family_positions[0] - bar_width/2, -0.05), 
                   xycoords=('data', 'axes fraction'), ha='center', fontsize=10)
        ax.annotate('Tuned', xy=(family_positions[0] + bar_width/2, -0.05), 
                   xycoords=('data', 'axes fraction'), ha='center', fontsize=10)
        
        # Set other axes properties
        ax.set_title(f'{metric["title"]} - {dataset_name} Dataset', fontsize=14)
        ax.set_ylabel(metric['ylabel'], fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Set reasonable ylim
        if metric['best'] == 'min':
            ax.set_ylim(0, best_fn(bar_values) * 1.5)
        else:
            # For R², we want a more specific range (usually 0 to 1)
            ax.set_ylim(max(0, min(bar_values) - 0.1), min(1.0, max(bar_values) + 0.1))
    
    # Add overall title
    plt.suptitle(f'Model Performance Metrics on {dataset_name} Dataset', fontsize=16, y=0.98)
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
               "Models are grouped by family with basic and tuned variants side-by-side.\n"
               "Linear: Basic=Linear Regression, Tuned=ElasticNet | Other models: Basic=Default params, Tuned=Optuna optimized", 
               ha='center', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Set up output directory if not provided
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "dataset_comparison"
        io.ensure_dir(output_dir)
    
    # Save figure
    save_figure(fig, f"{dataset_name}_model_family_comparison", output_dir)
    print(f"Saved {dataset_name} dataset comparison to {output_dir}")
    
    return fig

def create_all_dataset_comparisons():
    """Create comparison visualizations for all datasets."""
    # Load all models
    all_models = load_all_models_for_comparison()
    
    # Extract metrics
    metrics_df = extract_model_metrics(all_models)
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "dataset_comparison"
    io.ensure_dir(output_dir)
    
    # Create comparison for each dataset
    datasets = ['Base', 'Base_Random', 'Yeo', 'Yeo_Random']
    figures = {}
    
    for dataset in datasets:
        print(f"Creating comparison for {dataset} dataset...")
        fig = plot_dataset_comparison(dataset, metrics_df, output_dir)
        if fig:
            figures[dataset] = fig
    
    print(f"Dataset comparison visualizations saved to {output_dir}")
    return figures

if __name__ == "__main__":
    create_all_dataset_comparisons()