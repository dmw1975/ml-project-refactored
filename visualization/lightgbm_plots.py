"""Visualization functions for LightGBM models (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new.adapters.lightgbm_adapter instead.",
    DeprecationWarning,
    stacklevel=2
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import optuna.visualization as optuna_vis

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def plot_lightgbm_comparison():
    """Compare basic vs. Optuna-optimized LightGBM models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load LightGBM results
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
    except:
        print("No LightGBM models found. Please train LightGBM models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Create comparison data
    model_data = []
    for model_name, model_results in lightgbm_models.items():
        model_data.append({
            'model_name': model_name,
            'dataset': model_name.split('_')[1],  # Base or Yeo
            'has_random': 'Random' in model_name,
            'model_type': 'Basic' if 'basic' in model_name else 'Optuna',
            'RMSE': model_results['RMSE'],
            'R2': model_results['R2'],
            'MAE': model_results['MAE']
        })
    
    df = pd.DataFrame(model_data)
    
    # Create grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    ax = axes[0]
    bars = sns.barplot(
        data=df, 
        x='dataset', 
        y='RMSE', 
        hue='model_type',
        ax=ax,
        palette={'Basic': '#3498db', 'Optuna': '#e74c3c'}
    )
    ax.set_title('RMSE: Basic vs. Optuna LightGBM', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # R² comparison
    ax = axes[1]
    bars = sns.barplot(
        data=df, 
        x='dataset', 
        y='R2', 
        hue='model_type',
        ax=ax,
        palette={'Basic': '#3498db', 'Optuna': '#e74c3c'}
    )
    ax.set_title('R²: Basic vs. Optuna LightGBM', fontsize=14)
    ax.set_ylabel('R² (higher is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, "lightgbm_basic_vs_optuna", output_dir)
    
    # Plot improvement percentages
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvements
    improvements = []
    for dataset in df['dataset'].unique():
        basic_rmse = df[(df['dataset'] == dataset) & (df['model_type'] == 'Basic')]['RMSE'].values[0]
        optuna_rmse = df[(df['dataset'] == dataset) & (df['model_type'] == 'Optuna')]['RMSE'].values[0]
        improvement = ((basic_rmse - optuna_rmse) / basic_rmse) * 100
        improvements.append({'dataset': dataset, 'improvement': improvement})
    
    imp_df = pd.DataFrame(improvements)
    bars = ax.bar(imp_df['dataset'], imp_df['improvement'], color='#2ecc71')
    
    ax.set_title('RMSE Improvement with Optuna Optimization', fontsize=14)
    ax.set_ylabel('Improvement (%)')
    ax.set_xlabel('Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    save_figure(fig, "lightgbm_optuna_improvement", output_dir)
    
    print(f"LightGBM comparison plots saved to {output_dir}")
    return fig

def plot_optuna_optimization_history():
    """Plot Optuna optimization history for each dataset."""
    # Set up style
    style = setup_visualization_style()
    
    # Load LightGBM results
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
    except:
        print("No LightGBM models found. Please train LightGBM models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Extract Optuna studies
    for model_name, model_results in lightgbm_models.items():
        if 'optuna' in model_name and 'study' in model_results:
            study = model_results['study']
            
            # Create optimization history plot
            fig = optuna_vis.plot_optimization_history(study)
            fig.update_layout(
                title=f'Optimization History: {model_name}',
                xaxis_title='Trial Number',
                yaxis_title='Mean CV MSE',
                template='plotly_white'
            )
            fig.write_image(f"{output_dir}/{model_name}_optimization_history.png", scale=2)
            
            # Create parameter importance plot
            try:
                fig = optuna_vis.plot_param_importances(study)
                fig.update_layout(
                    title=f'Parameter Importance: {model_name}',
                    template='plotly_white'
                )
                fig.write_image(f"{output_dir}/{model_name}_param_importance.png", scale=2)
            except:
                print(f"Could not generate parameter importance plot for {model_name}")
    
    print(f"Optuna optimization history plots saved to {output_dir}")

def plot_lightgbm_hyperparameter_comparison():
    """Create a comparison of hyperparameters across models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load LightGBM results
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
    except:
        print("No LightGBM models found. Please train LightGBM models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Extract hyperparameters
    hyperparams = []
    for model_name, model_results in lightgbm_models.items():
        if 'optuna' in model_name and 'best_params' in model_results:
            for param, value in model_results['best_params'].items():
                # Skip non-optimized parameters
                if param in ['objective', 'metric', 'verbosity', 'boosting_type', 'random_state']:
                    continue
                hyperparams.append({
                    'model': model_name.replace('_optuna', ''),
                    'parameter': param,
                    'value': value
                })
    
    if not hyperparams:
        print("No hyperparameter data found.")
        return None
    
    df = pd.DataFrame(hyperparams)
    
    # Create separate plots for each important hyperparameter
    params_to_plot = ['num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction', 'min_child_samples']
    
    for param in params_to_plot:
        param_df = df[df['parameter'] == param]
        if param_df.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(param_df['model'], param_df['value'], color='#3498db')
        
        ax.set_title(f'Best {param} for Each Model', fontsize=14)
        ax.set_ylabel(param)
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * height,
                    f'{height:.4f}' if param == 'learning_rate' else f'{int(height)}' if param == 'num_leaves' or param == 'min_child_samples' else f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, f"lightgbm_best_{param}_comparison", output_dir)
    
    print(f"LightGBM hyperparameter comparison plots saved to {output_dir}")

def plot_model_comparison_with_lightgbm():
    """Compare LightGBM performance with other models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load metrics from all model types
    metrics = []
    
    # Try to load Linear Regression metrics
    try:
        lr_metrics = pd.read_csv(f"{settings.METRICS_DIR}/linear_regression_metrics.csv")
        metrics.append(lr_metrics)
    except:
        print("No Linear Regression metrics found")
    
    # Try to load ElasticNet metrics
    try:
        en_metrics = pd.read_csv(f"{settings.METRICS_DIR}/elasticnet_metrics.csv")
        metrics.append(en_metrics)
    except:
        print("No ElasticNet metrics found")
    
    # Try to load XGBoost metrics
    try:
        xgb_metrics = pd.read_csv(f"{settings.METRICS_DIR}/xgboost_metrics.csv")
        metrics.append(xgb_metrics)
    except:
        print("No XGBoost metrics found")
    
    # Try to load LightGBM metrics
    try:
        lgb_metrics = pd.read_csv(f"{settings.METRICS_DIR}/lightgbm_metrics.csv")
        metrics.append(lgb_metrics)
    except:
        print("No LightGBM metrics found")
    
    if not metrics:
        print("No metrics found for any model. Please train models first.")
        return None
    
    # Combine all metrics
    all_metrics = pd.concat(metrics)
    
    # Only include the best models (Optuna version) for each model type and dataset
    best_models = []
    
    # Process model names to get consistent naming across different model types
    for i, row in all_metrics.iterrows():
        model_name = row['model_name']
        if 'LR_' in model_name:
            model_family = 'Linear Regression'
            dataset = model_name.replace('LR_', '')
        elif 'ElasticNet_LR_' in model_name:
            model_family = 'ElasticNet'
            dataset = model_name.replace('ElasticNet_LR_', '')
        elif 'XGB_' in model_name:
            model_family = 'XGBoost'
            dataset = model_name.replace('XGB_', '').split('_')[0]
            if not ('optuna' in model_name.lower()):
                continue  # Skip non-optuna XGBoost models
        elif 'LightGBM_' in model_name:
            model_family = 'LightGBM'
            dataset = model_name.replace('LightGBM_', '').split('_')[0]
            if not ('optuna' in model_name.lower()):
                continue  # Skip non-optuna LightGBM models
        else:
            continue
        
        best_models.append({
            'model_family': model_family,
            'dataset': dataset,
            'RMSE': row['RMSE'],
            'R2': row['R2'],
            'MAE': row.get('MAE', np.nan)
        })
    
    best_df = pd.DataFrame(best_models)
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "summary"
    io.ensure_dir(output_dir)
    
    # Plot RMSE comparison across models and datasets
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=best_df, 
        x='dataset', 
        y='RMSE', 
        hue='model_family',
        palette={
            'Linear Regression': '#3498db',
            'ElasticNet': '#e74c3c',
            'XGBoost': '#2ecc71',
            'LightGBM': '#f39c12'
        }
    )
    plt.title('RMSE Comparison Across Models', fontsize=16)
    plt.ylabel('RMSE (lower is better)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Model Type')
    plt.xticks(rotation=45)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=8)
    
    plt.tight_layout()
    save_figure(plt.gcf(), "all_models_rmse_comparison", output_dir)
    
    # Plot R² comparison across models and datasets
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=best_df, 
        x='dataset', 
        y='R2', 
        hue='model_family',
        palette={
            'Linear Regression': '#3498db',
            'ElasticNet': '#e74c3c',
            'XGBoost': '#2ecc71',
            'LightGBM': '#f39c12'
        }
    )
    plt.title('R² Comparison Across Models', fontsize=16)
    plt.ylabel('R² (higher is better)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Model Type')
    plt.xticks(rotation=45)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=8)
    
    plt.tight_layout()
    save_figure(plt.gcf(), "all_models_r2_comparison", output_dir)
    
    print(f"Model comparison plots saved to {output_dir}")
    return plt.gcf()

def visualize_lightgbm_models():
    """Run all LightGBM visualizations."""
    print("Generating LightGBM visualizations...")
    
    plot_lightgbm_comparison()
    plot_optuna_optimization_history()
    plot_lightgbm_hyperparameter_comparison()
    plot_model_comparison_with_lightgbm()
    
    print("LightGBM visualizations completed.")

if __name__ == "__main__":
    visualize_lightgbm_models()