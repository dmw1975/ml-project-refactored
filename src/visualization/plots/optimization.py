"""
Visualization module for optimization-related plots like Optuna results.

This module provides functions to visualize optimization history, parameter importance,
and other hyperparameter-related visualizations.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Import make_subplots for improved contour plotting
from typing import Optional, Dict, Any, List, Union

try:
    import optuna
    import optuna.visualization as optuna_vis
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.visualization.core.interfaces import ModelData, VisualizationConfig
from config import settings


def _process_config(config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
                   output_dir: Optional[Path] = None) -> VisualizationConfig:
    """Process visualization configuration safely.
    
    Args:
        config: Input configuration dictionary or VisualizationConfig object
        output_dir: Default output directory if config doesn't specify one
        
    Returns:
        VisualizationConfig: Properly configured visualization config
    """
    # Start with default configuration
    config_dict = {
        'output_dir': output_dir,
        'format': 'png',
        'dpi': 300,
        'save': True,
        'show': False
    }
    
    # Update with provided config values
    if isinstance(config, dict):
        config_dict.update(config)
    elif isinstance(config, VisualizationConfig):
        # Extract values from VisualizationConfig using get()
        for attr in ['format', 'dpi', 'save', 'show']:
            config_dict[attr] = config.get(attr, config_dict[attr])
        
        # Handle output_dir specially to prioritize the input parameter
        if config.get('output_dir') is not None:
            config_dict['output_dir'] = config.get('output_dir')
    
    # Create and return VisualizationConfig object
    return VisualizationConfig(**config_dict)


def plot_optimization_history(
    study,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_name: str = "model"
) -> Optional[str]:
    """
    Plot the Optuna optimization history.
    
    Args:
        study: Optuna study object with optimization history
        config: Visualization configuration
        model_name: Name of the model for output filename
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna visualization not available. Please install optuna.")
        return None
    
    if study is None:
        print(f"No study object provided for {model_name}")
        return None
    
    # Determine default output directory based on model name
    default_output_dir = None
    if model_name.lower().startswith("xgb"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
    elif model_name.lower().startswith("lightgbm"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "lightgbm"
    elif model_name.lower().startswith("catboost"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
    elif model_name.lower().startswith("elasticnet"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
    else:
        default_output_dir = settings.VISUALIZATION_DIR / "performance"
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Ensure output directory exists
    output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    try:
        fig = optuna_vis.plot_optimization_history(study)
        fig.update_layout(
            title=f'Optimization History: {model_name}',
            xaxis_title='Trial Number',
            yaxis_title='Mean CV MSE',
            template='plotly_white'
        )
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{model_name}_optuna_optimization_history.{config.get('format', 'png')}")
        fig.write_image(output_path, scale=2)
        
        return output_path
    except Exception as e:
        print(f"Error creating optimization history plot for {model_name}: {e}")
        return None


def plot_param_importance(
    study, 
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_name: str = "model"
) -> Optional[str]:
    """
    Plot the Optuna parameter importance.
    
    Args:
        study: Optuna study object with optimization history
        config: Visualization configuration
        model_name: Name of the model for output filename
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna visualization not available. Please install optuna.")
        return None
    
    if study is None:
        print(f"No study object provided for {model_name}")
        return None
    
    # Determine default output directory based on model name
    default_output_dir = None
    if model_name.lower().startswith("xgb"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
    elif model_name.lower().startswith("lightgbm"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "lightgbm"
    elif model_name.lower().startswith("catboost"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
    elif model_name.lower().startswith("elasticnet"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
    else:
        default_output_dir = settings.VISUALIZATION_DIR / "performance"
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Ensure output directory exists
    output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    try:
        fig = optuna_vis.plot_param_importances(study)
        fig.update_layout(
            title=f'Parameter Importance: {model_name}',
            template='plotly_white'
        )
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{model_name}_optuna_param_importance.{config.get('format', 'png')}")
        fig.write_image(output_path, scale=2)
        
        return output_path
    except Exception as e:
        print(f"Error creating parameter importance plot for {model_name}: {e}")
        return None


def plot_contour(
    study, 
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_name: str = "model"
) -> Optional[str]:
    """
    Plot the parameter contour plot.
    
    Args:
        study: Optuna study object with optimization history
        config: Visualization configuration
        model_name: Name of the model for output filename
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna visualization not available. Please install optuna.")
        return None
    
    if study is None:
        print(f"No study object provided for {model_name}")
        return None
    
    # Determine default output directory based on model name
    default_output_dir = None
    if model_name.lower().startswith("xgb"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
    elif model_name.lower().startswith("lightgbm"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "lightgbm"
    elif model_name.lower().startswith("catboost"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
    elif model_name.lower().startswith("elasticnet"):
        default_output_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
    else:
        default_output_dir = settings.VISUALIZATION_DIR / "performance"
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Ensure output directory exists
    output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Always use the improved contour plot for better readability
    return plot_improved_contour(study, config, model_name)


def plot_improved_contour(
    study, 
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_name: str = "model"
) -> Optional[str]:
    """
    Plot an improved parameter contour plot with better readability.
    
    This version:
    1. Selects only the most important parameter pairs
    2. Uses a larger figure size
    3. Improves spacing between subplots
    4. Creates cleaner visualizations of each parameter pair
    
    Args:
        study: Optuna study object with optimization history
        config: Visualization configuration
        model_name: Name of the model for output filename
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    try:
        # Get most important parameters from the study
        param_importances = optuna.importance.get_param_importances(study)
        
        # Select top parameters (up to 5 most important)
        top_params = list(param_importances.keys())[:5]
        
        if len(top_params) < 2:
            print(f"Not enough parameters in study for {model_name} to create contour plot")
            return None
        
        # Calculate the number of pairs (n choose 2)
        n_pairs = len(top_params) * (len(top_params) - 1) // 2
        
        # Determine grid size - want approximately square grid
        grid_size = int(np.ceil(np.sqrt(n_pairs)))
        n_rows = n_pairs // grid_size + (1 if n_pairs % grid_size else 0)
        n_cols = min(grid_size, n_pairs)
        
        # Create figure
        fig = go.Figure()
        
        # Create subplot for each pair of top parameters
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f'{p1} vs {p2}' for i, p1 in enumerate(top_params) for p2 in top_params[i+1:]]
        )
        
        # Add contour plots for each pair
        subplot_idx = 1
        for i, param_i in enumerate(top_params):
            for param_j in top_params[i+1:]:
                # Calculate row and column for this subplot
                row = (subplot_idx - 1) // n_cols + 1
                col = (subplot_idx - 1) % n_cols + 1
                
                # Extract parameter values from trials
                param_values_i = []
                param_values_j = []
                objective_values = []
                
                for trial in study.trials:
                    if trial.state == TrialState.COMPLETE:
                        if param_i in trial.params and param_j in trial.params:
                            param_values_i.append(trial.params[param_i])
                            param_values_j.append(trial.params[param_j])
                            objective_values.append(trial.value)
                
                # Create scatter plot with color based on objective value
                scatter = go.Scatter(
                    x=param_values_i,
                    y=param_values_j,
                    mode='markers',
                    marker=dict(
                        color=objective_values,
                        # Use Plasma colorscale which has better contrast than Viridis
                        colorscale='Plasma',  # Alternative options: 'Jet', 'Rainbow', 'Turbo'
                        # Add black outline to each dot for better separation
                        line=dict(width=1, color='rgba(0,0,0,0.5)'),
                        colorbar=dict(
                            title='Objective Value',
                            titleside='right',
                            thickness=15,
                            len=0.5,
                            outlinewidth=1
                        ) if i == 0 and param_j == top_params[1] else None,
                        showscale=i == 0 and param_j == top_params[1],
                        # Increase dot size for better visibility
                        size=10,
                        # Add some transparency to see overlapping points
                        opacity=0.85
                    ),
                    showlegend=False
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
                # Update axes labels
                fig.update_xaxes(title_text=param_i, row=row, col=col)
                fig.update_yaxes(title_text=param_j, row=row, col=col)
                
                subplot_idx += 1
        
        # Update layout for better readability
        fig.update_layout(
            title=f'Parameter Contour Plot: {model_name}',
            template='plotly_white',
            height=350 * n_rows + 100,   # Even larger height for more readability
            width=450 * n_cols,          # Even larger width for more readability
            font=dict(size=12),          # Keep larger font size
            margin=dict(l=50, r=50, t=80, b=50),  # Increase margins
            title_font=dict(size=18),    # Larger title font
            title_x=0.5,                 # Center title
            showlegend=False,
            # Add grid pattern to better separate the plots
            plot_bgcolor='rgba(240,240,240,0.3)',
            # Add a subtle border around the entire figure
            paper_bgcolor='white',
            # Add annotation highlighting color scale meaning
            annotations=[
                dict(
                    text="Lower values (purple) are better",
                    x=0.99,
                    y=0.01,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
            ]
        )
        
        # Update all x and y axes to make them more readable
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(220,220,220,0.5)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)',
            title_font=dict(size=11),
            tickfont=dict(size=10)
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(220,220,220,0.5)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)',
            title_font=dict(size=11),
            tickfont=dict(size=10)
        )
        
        # Save the figure with a simpler, cleaner name
        output_dir = config.get('output_dir')
        output_path = os.path.join(output_dir, f"{model_name}_contour.{config.get('format', 'png')}")
        fig.write_image(output_path, scale=3)  # Higher scale for better resolution
        
        return output_path
        
    except Exception as e:
        print(f"Error creating improved contour plot for {model_name}: {e}")
        return None


def plot_hyperparameter_comparison(
    model_data_list: List[Dict[str, Any]],
    parameter_name: str,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_family: str = "xgboost"
) -> Optional[str]:
    """
    Compare a specific hyperparameter across different models.
    
    Args:
        model_data_list: List of model data dictionaries
        parameter_name: Name of the hyperparameter to compare
        config: Visualization configuration
        model_family: Model family name (xgboost, lightgbm, catboost)
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    # Determine default output directory
    default_output_dir = settings.VISUALIZATION_DIR / "performance" / model_family
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Ensure output directory exists
    output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract hyperparameter values
    model_names = []
    param_values = []
    
    for model_data in model_data_list:
        # Skip models without best_params
        if 'best_params' not in model_data or parameter_name not in model_data['best_params']:
            continue
        
        model_name = model_data.get('model_name', 'Unknown')
        model_names.append(model_name.replace('_optuna', ''))
        param_values.append(model_data['best_params'][parameter_name])
    
    if not model_names:
        print(f"No models with parameter '{parameter_name}' found")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, param_values, color='#3498db')
    
    ax.set_title(f'Best {parameter_name} for Each Model', fontsize=14)
    ax.set_ylabel(parameter_name)
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01 * height,
            f'{height:.4f}' if parameter_name == 'learning_rate' else f'{int(height)}',
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{model_family}_best_{parameter_name}_comparison.{config.get('format', 'png')}")
    plt.savefig(output_path, dpi=config.get('dpi', 300))
    
    if config.get('show', False):
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def plot_optuna_improvement(
    model_data_list: List[Dict[str, Any]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_family: str = "xgboost"
) -> Optional[str]:
    """
    Plot the improvement gained by Optuna optimization for each dataset.
    
    Args:
        model_data_list: List of model data dictionaries
        config: Visualization configuration
        model_family: Model family name (xgboost, lightgbm, catboost)
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    # Determine default output directory
    default_output_dir = settings.VISUALIZATION_DIR / "performance" / model_family
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Ensure output directory exists
    output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Group models by dataset
    dataset_models = {}
    
    for model_data in model_data_list:
        model_name = model_data.get('model_name', 'Unknown')
        
        # Extract dataset name (e.g., "XGB_Base" from "XGB_Base_optuna" or "XGB_Base_basic")
        parts = model_name.split('_')
        if len(parts) >= 2:
            dataset = '_'.join(parts[:-1])  # Remove the last part (basic/optuna)
            model_type = parts[-1]  # "basic" or "optuna"
            
            if dataset not in dataset_models:
                dataset_models[dataset] = {}
                
            dataset_models[dataset][model_type] = model_data
    
    # Calculate improvements
    improvements = []
    
    for dataset, models in dataset_models.items():
        if 'basic' in models and 'optuna' in models:
            basic_rmse = models['basic']['RMSE']
            optuna_rmse = models['optuna']['RMSE']
            improvement = ((basic_rmse - optuna_rmse) / basic_rmse) * 100
            improvements.append({
                'dataset': dataset,
                'improvement': improvement
            })
    
    if not improvements:
        print("No paired basic/optuna models found to calculate improvements")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [imp['dataset'] for imp in improvements]
    improvement_values = [imp['improvement'] for imp in improvements]
    
    bars = ax.bar(datasets, improvement_values, color='#2ecc71')
    
    ax.set_title(f'RMSE Improvement with Optuna Optimization', fontsize=14)
    ax.set_ylabel('Improvement (%)')
    ax.set_xlabel('Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'{height:.1f}%', 
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{model_family}_optuna_improvement.{config.get('format', 'png')}")
    plt.savefig(output_path, dpi=config.get('dpi', 300))
    
    if config.get('show', False):
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def plot_basic_vs_optuna(
    model_data_list: List[Dict[str, Any]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None,
    model_family: str = "xgboost"
) -> Optional[str]:
    """
    Compare basic vs. Optuna-optimized models using bar charts.
    
    Args:
        model_data_list: List of model data dictionaries
        config: Visualization configuration
        model_family: Model family name (xgboost, lightgbm, catboost)
    
    Returns:
        Path to the saved visualization or None if not saved
    """
    # Determine default output directory
    default_output_dir = settings.VISUALIZATION_DIR / "performance" / model_family
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Ensure output directory exists
    output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier processing
    model_data = []
    for model in model_data_list:
        model_name = model.get('model_name', 'Unknown')
        model_type = 'Basic' if 'basic' in model_name else 'Optuna' if 'optuna' in model_name else 'Unknown'
        
        # Skip if not basic or optuna model
        if model_type == 'Unknown':
            continue
        
        # Extract dataset (e.g., "Base" or "Yeo" from "XGB_Base_basic")
        parts = model_name.split('_')
        if len(parts) >= 2:
            dataset = parts[1]  # Assuming format like XGB_Base_basic
            has_random = 'Random' in model_name
            
            model_data.append({
                'model_name': model_name,
                'dataset': dataset,
                'has_random': has_random,
                'model_type': model_type,
                'RMSE': model.get('RMSE', 0),
                'R2': model.get('R2', 0),
                'MAE': model.get('MAE', 0)
            })
    
    if not model_data:
        print("No basic or optuna models found")
        return None
    
    df = pd.DataFrame(model_data)
    
    # Create grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    ax = axes[0]
    try:
        import seaborn as sns
        sns.barplot(
            data=df, 
            x='dataset', 
            y='RMSE', 
            hue='model_type',
            ax=ax,
            palette={'Basic': '#3498db', 'Optuna': '#e74c3c'}
        )
    except ImportError:
        # Fallback if seaborn not available
        for i, model_type in enumerate(['Basic', 'Optuna']):
            data = df[df['model_type'] == model_type]
            x = np.arange(len(data)) + i * 0.4 - 0.2
            ax.bar(x, data['RMSE'], width=0.4, label=model_type,
                  color='#3498db' if model_type == 'Basic' else '#e74c3c')
            ax.set_xticks(np.arange(len(data)))
            ax.set_xticklabels(data['dataset'])
    
    ax.set_title(f'RMSE: Basic vs. Optuna {model_family.capitalize()}', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # R² comparison
    ax = axes[1]
    try:
        import seaborn as sns
        sns.barplot(
            data=df, 
            x='dataset', 
            y='R2', 
            hue='model_type',
            ax=ax,
            palette={'Basic': '#3498db', 'Optuna': '#e74c3c'}
        )
    except ImportError:
        for i, model_type in enumerate(['Basic', 'Optuna']):
            data = df[df['model_type'] == model_type]
            x = np.arange(len(data)) + i * 0.4 - 0.2
            ax.bar(x, data['R2'], width=0.4, label=model_type,
                  color='#3498db' if model_type == 'Basic' else '#e74c3c')
            ax.set_xticks(np.arange(len(data)))
            ax.set_xticklabels(data['dataset'])
    
    ax.set_title(f'R²: Basic vs. Optuna {model_family.capitalize()}', fontsize=14)
    ax.set_ylabel('R² (higher is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{model_family}_basic_vs_optuna.{config.get('format', 'png')}")
    plt.savefig(output_path, dpi=config.get('dpi', 300))
    
    if config.get('show', False):
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def plot_all_optimization_visualizations(
    model_data: Dict[str, Any],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, str]:
    """
    Create all optimization-related visualizations for a model.
    
    Args:
        model_data: Model data dictionary including study object
        config: Visualization configuration
    
    Returns:
        Dictionary of visualization paths
    """
    # Get model name and study object
    model_name = model_data.get('model_name', 'model')
    study = model_data.get('study', None)
    
    # Determine model family
    model_family = "unknown"
    if 'xgb' in model_name.lower():
        model_family = "xgboost"
    elif 'lightgbm' in model_name.lower():
        model_family = "lightgbm"
    elif 'catboost' in model_name.lower():
        model_family = "catboost"
    elif 'elasticnet' in model_name.lower():
        model_family = "elasticnet"
    
    # Determine default output directory
    default_output_dir = settings.VISUALIZATION_DIR / "performance" / model_family
    
    # Process config using helper function
    config = _process_config(config, default_output_dir)
    
    # Dictionary to store paths to created visualizations
    output_paths = {}
    
    # Create optimization history plot
    if study is not None:
        hist_path = plot_optimization_history(study, config, model_name)
        if hist_path:
            output_paths['optimization_history'] = hist_path
        
        # Create parameter importance plot
        param_path = plot_param_importance(study, config, model_name)
        if param_path:
            output_paths['param_importance'] = param_path
        
        # Create contour plot (only the improved version)
        try:
            contour_path = plot_improved_contour(study, config, model_name)
            if contour_path:
                output_paths['contour'] = contour_path
                print(f"Created contour plot for {model_name} with better visibility")
        except Exception as e:
            print(f"Error creating contour plot for {model_name}: {e}")
    
    return output_paths