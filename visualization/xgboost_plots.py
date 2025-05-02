"""Visualization functions for XGBoost models (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use visualization_new package instead.
"""

import warnings
from pathlib import Path
import sys

warnings.warn(
    "This module is deprecated. Please use visualization_new.adapters.xgboost_adapter instead.",
    DeprecationWarning,
    stacklevel=2
)

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

def plot_xgboost_comparison():
    """Compare basic vs. Optuna-optimized XGBoost models."""
    # Import required modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from visualization.style import setup_visualization_style, save_figure
    
    # Set up style
    style = setup_visualization_style()
    
    # Load XGBoost results
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No XGBoost models found. Please train XGBoost models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Create comparison data
    model_data = []
    for model_name, model_results in xgboost_models.items():
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
    ax.set_title('RMSE: Basic vs. Optuna XGBoost', fontsize=14)
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
    ax.set_title('R²: Basic vs. Optuna XGBoost', fontsize=14)
    ax.set_ylabel('R² (higher is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, "xgboost_basic_vs_optuna", output_dir)
    
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
    save_figure(fig, "xgboost_optuna_improvement", output_dir)
    
    print(f"XGBoost comparison plots saved to {output_dir}")
    return fig

def plot_optuna_optimization_history():
    """Plot Optuna optimization history for each dataset."""
    # Import required modules
    import optuna.visualization as optuna_vis
    from visualization.style import setup_visualization_style
    
    # Set up style
    style = setup_visualization_style()
    
    # Load XGBoost results
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No XGBoost models found. Please train XGBoost models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Extract Optuna studies
    for model_name, model_results in xgboost_models.items():
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

def plot_xgboost_hyperparameter_comparison():
    """Create a comparison of hyperparameters across models."""
    # Import required modules
    import pandas as pd
    import matplotlib.pyplot as plt
    from visualization.style import setup_visualization_style, save_figure
    
    # Set up style
    style = setup_visualization_style()
    
    # Load XGBoost results
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
    except:
        print("No XGBoost models found. Please train XGBoost models first.")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(output_dir)
    
    # Extract hyperparameters
    hyperparams = []
    for model_name, model_results in xgboost_models.items():
        if 'optuna' in model_name and 'best_params' in model_results:
            for param, value in model_results['best_params'].items():
                hyperparams.append({
                    'model': model_name.replace('_optuna', ''),
                    'parameter': param,
                    'value': value
                })
    
    if not hyperparams:
        print("No hyperparameter data found.")
        return None
    
    df = pd.DataFrame(hyperparams)
    
    # Create heatmap of hyperparameters
    params_to_plot = ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'colsample_bytree']
    
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
                    f'{height:.4f}' if param == 'learning_rate' else f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, f"xgboost_best_{param}_comparison", output_dir)
    
    print(f"XGBoost hyperparameter comparison plots saved to {output_dir}")

def visualize_xgboost_models():
    """Run all XGBoost visualizations."""
    print("Generating XGBoost visualizations...")
    
    plot_xgboost_comparison()
    plot_optuna_optimization_history()
    plot_xgboost_hyperparameter_comparison()
    
    print("XGBoost visualizations completed.")

if __name__ == "__main__":
    visualize_xgboost_models()