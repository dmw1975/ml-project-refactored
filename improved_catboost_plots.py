"""Improved CatBoost visualization functions."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def plot_catboost_comparison():
    """Compare basic vs. Optuna-optimized CatBoost models with improved grouping."""
    # Set up style
    style = setup_visualization_style()
    
    # Load CatBoost results
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No CatBoost models found. Please train CatBoost models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/catboost"
    io.ensure_dir(output_dir)
    
    # Extract performance metrics
    performance_data = []
    for name, model_data in catboost_models.items():
        # Skip if not a valid model
        if 'RMSE' not in model_data:
            continue
            
        # Parse model name parts
        name_parts = name.split("_")
        
        # Determine dataset (Base or Yeo) and if it's random
        dataset = name_parts[1]  # Base or Yeo
        has_random = 'Random' in name
        
        # For display, add Random indicator if needed
        dataset_display = dataset
        if has_random:
            dataset_display = f"{dataset} R"  # Add R for Random
        
        performance_data.append({
            'model_name': name,
            'RMSE': model_data['RMSE'],
            'R2': model_data.get('R2', 0),  # Default to 0 if not present
            'dataset': dataset_display,
            'Type': 'Optuna' if 'optuna' in name else 'Basic'
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Check if we have enough data
    if len(perf_df) < 1:
        print("Not enough CatBoost models found for comparison.")
        return None
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    ax = axes[0]
    bars = sns.barplot(
        data=perf_df, 
        x='dataset', 
        y='RMSE', 
        hue='Type',
        ax=ax,
        palette={'Basic': '#3498db', 'Optuna': '#e74c3c'}
    )
    ax.set_title('RMSE: Basic vs. Optuna CatBoost', fontsize=14)
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
        data=perf_df, 
        x='dataset', 
        y='R2', 
        hue='Type',
        ax=ax,
        palette={'Basic': '#3498db', 'Optuna': '#e74c3c'}
    )
    ax.set_title('R²: Basic vs. Optuna CatBoost', fontsize=14)
    ax.set_ylabel('R² (higher is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, "catboost_basic_vs_optuna", output_dir)
    
    print(f"CatBoost comparison plot saved to {output_dir}")
    return fig

def plot_catboost_optimization_history():
    """Plot optimization history for Optuna-optimized CatBoost models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load CatBoost results
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No CatBoost models found. Please train CatBoost models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/catboost/optimization"
    io.ensure_dir(output_dir)
    
    # Extract Optuna study data
    for name, model_data in catboost_models.items():
        if 'optuna' not in name or 'optuna_study' not in model_data:
            continue
            
        try:
            study = model_data['optuna_study']
            
            # Get optimization history
            fig = plt.figure(figsize=(10, 6))
            
            # Get trials data
            trials = study.trials
            values = [t.value for t in trials]
            iterations = list(range(len(trials)))
            
            # Plot optimization history
            plt.plot(iterations, values, 'o-')
            
            best_trial = study.best_trial
            best_value = best_trial.value
            plt.axhline(y=best_value, color='r', linestyle='--', label=f'Best value: {best_value:.4f}')
            
            plt.title(f'Optuna Optimization History - {name}', fontsize=14)
            plt.xlabel('Trial')
            plt.ylabel('Objective value (RMSE)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            save_figure(fig, f"{name}_optuna_optimization_history", output_dir)
            plt.close(fig)
            
            # Create parameter importance plot if available
            try:
                if hasattr(study, 'get_param_importances'):
                    param_importance = study.get_param_importances()
                    
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame({
                        'Parameter': list(param_importance.keys()),
                        'Importance': list(param_importance.values())
                    })
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot
                    sns.barplot(x='Importance', y='Parameter', data=importance_df, ax=ax, palette='viridis')
                    
                    # Title and labels
                    ax.set_title(f'Parameter Importance - {name}', fontsize=14)
                    ax.set_xlabel('Importance')
                    ax.set_ylabel('Parameter')
                    
                    plt.tight_layout()
                    save_figure(fig, f"{name}_optuna_param_importance", output_dir)
                    plt.close(fig)
            except Exception as e:
                print(f"Error creating parameter importance plot for {name}: {e}")
                
        except Exception as e:
            print(f"Error creating optimization history for {name}: {e}")
    
    return True

def plot_catboost_hyperparameter_comparison():
    """Plot hyperparameter comparison for Optuna-optimized CatBoost models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load CatBoost results
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No CatBoost models found. Please train CatBoost models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/catboost/hyperparameters"
    io.ensure_dir(output_dir)
    
    # Extract hyperparameters
    hyperparams = []
    for model_name, model_results in catboost_models.items():
        if 'optuna' in model_name and 'best_params' in model_results:
            for param, value in model_results['best_params'].items():
                # Skip non-optimized or non-numeric parameters
                if param in ['task_type', 'devices', 'logging_level', 'random_seed', 'thread_count']:
                    continue
                    
                # Parse value if it's a numeric string
                try:
                    value = float(value)
                except:
                    pass
                    
                hyperparams.append({
                    'model': model_name.replace('_optuna', ''),
                    'parameter': param,
                    'value': value
                })
    
    if not hyperparams:
        print("No hyperparameter data found.")
        return None
    
    df = pd.DataFrame(hyperparams)
    
    # Identify common parameters to plot
    common_params = df['parameter'].value_counts()
    params_to_plot = common_params[common_params >= 2].index.tolist()
    
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
            if isinstance(height, (int, float)):
                if param == 'learning_rate' or param == 'l2_leaf_reg':
                    label = f'{height:.4f}'
                elif isinstance(height, int) or height.is_integer():
                    label = f'{int(height)}'
                else:
                    label = f'{height:.2f}'
            else:
                label = str(height)
            
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * height,
                    label, ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, f"catboost_best_{param}_comparison", output_dir)
        plt.close(fig)
    
    # Calculate overall improvement from basic to Optuna models
    improved_df = []
    for model_base in ['Base', 'Base_Random', 'Yeo', 'Yeo_Random']:
        basic_name = f"CatBoost_{model_base}_basic"
        optuna_name = f"CatBoost_{model_base}_optuna"
        
        if basic_name in catboost_models and optuna_name in catboost_models:
            basic_rmse = catboost_models[basic_name]['RMSE']
            optuna_rmse = catboost_models[optuna_name]['RMSE']
            
            improvement = (basic_rmse - optuna_rmse) / basic_rmse * 100
            improved_df.append({
                'model': model_base,
                'basic_rmse': basic_rmse,
                'optuna_rmse': optuna_rmse,
                'improvement': improvement
            })
    
    if improved_df:
        improved_df = pd.DataFrame(improved_df)
        
        # Create improvement plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(improved_df['model'], improved_df['improvement'], color='#2ecc71')
        
        ax.set_title('CatBoost RMSE Improvement from Optuna Optimization', fontsize=14)
        ax.set_ylabel('Improvement (%)')
        ax.set_xlabel('Dataset')
        ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(0.1, abs(height)),
                    f'{height:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, "catboost_optuna_improvement", output_dir)
        plt.close(fig)
    
    print(f"CatBoost hyperparameter comparison plots saved to {output_dir}")
    return True

def improved_catboost_visualizations():
    """Generate all improved CatBoost visualizations."""
    print("Generating improved CatBoost visualizations...")
    
    plot_catboost_comparison()
    plot_catboost_optimization_history()
    plot_catboost_hyperparameter_comparison()
    
    print("CatBoost visualizations completed.")

if __name__ == "__main__":
    improved_catboost_visualizations()