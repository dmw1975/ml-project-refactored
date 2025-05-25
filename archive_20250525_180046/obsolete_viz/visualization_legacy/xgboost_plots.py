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
    
    # Set up main output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(perf_dir)
    
    # Create xgboost directory for model-specific performance plots
    output_dir = perf_dir / "xgboost"
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
    
    print(f"XGBoost comparison plots saved to {output_dir} directory")
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
    
    # Set up main output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(perf_dir)
    
    # Create xgboost directory for model-specific performance plots
    output_dir = perf_dir / "xgboost"
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
            # Save directly to the xgboost directory
            fig.write_image(f"{output_dir}/{model_name}_optimization_history.png", scale=2)
            
            # Create parameter importance plot
            try:
                fig = optuna_vis.plot_param_importances(study)
                fig.update_layout(
                    title=f'Parameter Importance: {model_name}',
                    template='plotly_white'
                )
                # Save directly to the xgboost directory
                fig.write_image(f"{output_dir}/{model_name}_param_importance.png", scale=2)
            except:
                print(f"Could not generate parameter importance plot for {model_name}")
    
    print(f"XGBoost optimization history plots saved to {output_dir} directory")

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
    
    # Set up main output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(perf_dir)
    
    # Create xgboost directory for model-specific performance plots
    output_dir = perf_dir / "xgboost"
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
        
        # Save directly to the xgboost directory
        save_figure(fig, f"xgboost_best_{param}_comparison", output_dir)
    
    print(f"XGBoost hyperparameter comparison plots saved to {output_dir} directory")

def plot_xgboost_feature_importance():
    """Plot XGBoost built-in feature importance for each model."""
    # Import required modules
    import pandas as pd
    import numpy as np
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
    
    # Set up output directories
    features_dir = settings.VISUALIZATION_DIR / "features"
    xgboost_dir = features_dir / "xgboost"
    io.ensure_dir(features_dir)
    io.ensure_dir(xgboost_dir)
    
    # Try to load various feature name files
    original_feature_names = None
    feature_name_files = [
        "original_feature_names.pkl",
        "base_columns.pkl",
        "yeo_columns.pkl",
        "feature_names.pkl",
        "xgboost_feature_names.pkl"
    ]
    
    for file_name in feature_name_files:
        try:
            original_feature_names = io.load_pickle(file_name, settings.DATA_PKL_DIR)
            print(f"Successfully loaded feature names from {file_name}")
            break
        except:
            continue
    
    if original_feature_names is None:
        print("Warning: Could not load feature names from any pickle file. Will use model feature names.")
    
    # Plot feature importance for each model
    for name, model_data in xgboost_models.items():
        if 'model' not in model_data or model_data['model'] is None:
            print(f"Skipping {name}: No model object found")
            continue
        
        # Get model
        model = model_data['model']
        
        # Try multiple approaches to get feature names
        feature_names = None
        
        # Option 1: Use original_feature_names mapping if available
        if original_feature_names is not None:
            feature_names = original_feature_names
        
        # Option 2: Check model_data for feature_names
        if feature_names is None and 'feature_names' in model_data:
            feature_names = model_data['feature_names']
        
        # Option 3: Check model_data for original_feature_names
        if feature_names is None and 'original_feature_names' in model_data:
            feature_names = model_data['original_feature_names']
        
        # Option 4: Try model attribute
        if feature_names is None and hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_
        
        # Option 5: Try X_test columns if not generic names
        if feature_names is None and 'X_test' in model_data:
            X_test = model_data['X_test']
            if hasattr(X_test, 'columns'):
                if not all(col.startswith('feature_') for col in X_test.columns):
                    feature_names = X_test.columns.tolist()
        
        # If all else fails, try to get the number of features from importances or use generic names
        if feature_names is None:
            print(f"Warning: No feature names found for {name}. Attempting to determine number of features.")
            
            # Try to get feature importance first, as it will tell us the number of features
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                n_features = len(importance)
                print(f"Using feature importance length: {n_features} features")
                feature_names = [f"feature_{i}" for i in range(n_features)]
            # Try to get from model attributes
            elif hasattr(model, 'n_features_'):
                n_features = model.n_features_
                print(f"Using model.n_features_: {n_features} features")
                feature_names = [f"feature_{i}" for i in range(n_features)]
            # Try to get from X_test shape or columns
            elif 'X_test' in model_data:
                X_test = model_data['X_test']
                if hasattr(X_test, 'shape'):
                    n_features = X_test.shape[1]
                    print(f"Using X_test.shape[1]: {n_features} features")
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                elif hasattr(X_test, 'columns'):
                    feature_names = X_test.columns.tolist()
                    print(f"Using X_test.columns: {len(feature_names)} features")
            
            if feature_names is None:
                print(f"Skipping {name}: Could not determine number of features")
                continue
        
        # Get feature importance
        try:
            # Get feature importance from model
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                print(f"Skipping {name}: No feature_importances_ attribute found")
                continue
            
            # Handle case where importance and feature_names have different lengths
            if len(importance) != len(feature_names):
                print(f"Warning: {name} has {len(importance)} feature importances but {len(feature_names)} feature names")
                # Truncate the longer one to match
                min_len = min(len(importance), len(feature_names))
                importance = importance[:min_len]
                feature_names = feature_names[:min_len]
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            
            # Sort by importance (descending)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Limit to top 20 features
            if len(importance_df) > 20:
                importance_df = importance_df.head(20)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Prepare data for horizontal bar chart
            features = importance_df['Feature'].values
            importance_values = importance_df['Importance'].values
            y_pos = np.arange(len(features))
            
            # Plot horizontal bar chart with standard blue color (#3498db) for consistency
            bars = ax.barh(y_pos, importance_values, align='center', color='#3498db')
            
            # Set y-ticks
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            
            # Title and labels
            ax.set_title(f'XGBoost Feature Importance: {name}', fontsize=14)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            
            # Reverse y-axis to show most important at the top
            ax.invert_yaxis()
            
            plt.tight_layout()
            model_filename = f"xgboost_feature_importance_{name}"
            
            # Save only to xgboost subdirectory
            save_figure(fig, model_filename, xgboost_dir)
            
            print(f"Feature importance plot saved for {name} in features/xgboost/")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating feature importance plot for {name}: {e}")
    
    return True

def visualize_xgboost_models():
    """Run all XGBoost visualizations."""
    print("Generating XGBoost visualizations...")
    
    plot_xgboost_comparison()
    plot_optuna_optimization_history()
    plot_xgboost_hyperparameter_comparison()
    plot_xgboost_feature_importance()
    
    print("XGBoost visualizations completed.")

if __name__ == "__main__":
    visualize_xgboost_models()