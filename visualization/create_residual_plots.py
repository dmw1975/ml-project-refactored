"""
Script to create thesis-quality residual plots for all models (DEPRECATED).
This script creates residual plots for all trained models and saves them to the outputs directory.

This module is deprecated and will be removed in a future version.
Please use visualization_new.plots.residuals instead.
"""

import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new.plots.residuals instead.",
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
from utils import io
from matplotlib import rcParams
import matplotlib.ticker as ticker
from scipy import stats

def load_all_models():
    """Load all trained models from model directory."""
    all_models = {}
    
    # Load linear regression models
    try:
        linear_models = io.load_model("linear_regression_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(linear_models)} linear regression models")
        all_models.update(linear_models)
    except Exception as e:
        print(f"Error loading linear regression models: {e}")
    
    # Load ElasticNet models
    try:
        elastic_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(elastic_models)} ElasticNet models")
        all_models.update(elastic_models)
    except Exception as e:
        print(f"Error loading ElasticNet models: {e}")
    
    # Load XGBoost models
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(xgboost_models)} XGBoost models")
        all_models.update(xgboost_models)
    except Exception as e:
        print(f"Error loading XGBoost models: {e}")
    
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(lightgbm_models)} LightGBM models")
        all_models.update(lightgbm_models)
    except Exception as e:
        print(f"Error loading LightGBM models: {e}")
    
    print(f"Loaded {len(all_models)} models in total")
    return all_models

def create_thesis_residual_plot(model_name, model_data, output_dir):
    """Create a thesis-style residual plot for a specific model."""
    # Extract test set predictions and actual values
    y_test = model_data.get('y_test')
    y_pred = model_data.get('y_pred')
    
    if y_test is None or y_pred is None:
        print(f"Missing y_test or y_pred for {model_name}, skipping")
        return
    
    # Ensure y_test and y_pred have the same format and are aligned
    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
        y_test_values = y_test.values.flatten()
    else:
        y_test_values = np.array(y_test).flatten()
        
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred_values = y_pred.values.flatten()
    else:
        y_pred_values = np.array(y_pred).flatten()
    
    # Calculate residuals
    residuals = y_test_values - y_pred_values
    
    # Set thesis-quality settings
    rcParams.update({'font.size': 14})
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Predicted vs Actual (top left)
    ax = axes[0, 0]
    ax.scatter(y_pred_values, y_test_values, alpha=0.7, color='#3498db')
    min_val = min(y_pred_values.min(), y_test_values.min())
    max_val = max(y_pred_values.max(), y_test_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel('Predicted ESG Score')
    ax.set_ylabel('Actual ESG Score')
    ax.set_title('Predicted vs Actual Values', fontsize=16)
    ax.grid(alpha=0.5)
    
    corr = np.corrcoef(y_pred_values, y_test_values)[0, 1]
    r_squared = corr**2
    ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    
    # 2. Residuals vs Predicted (top right)
    ax = axes[0, 1]
    ax.scatter(y_pred_values, residuals, alpha=0.7, color='#2ecc71')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted ESG Score')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted Values', fontsize=16)
    ax.grid(alpha=0.5)
    
    # 3. Histogram of Residuals (bottom left)
    ax = axes[1, 0]
    sns.histplot(residuals, kde=True, ax=ax, color='#e67e22')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Residuals', fontsize=16)
    
    mean_res = residuals.mean()
    std_res = residuals.std()
    ax.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    
    # 4. Q-Q Plot (bottom right)
    ax = axes[1, 1]
    standardized_residuals = (residuals - mean_res) / std_res
    stats.probplot(standardized_residuals, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot of Standardized Residuals', fontsize=16)
    ax.grid(alpha=0.5)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    metrics_text = (
        f"RMSE: {rmse:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"R²: {r_squared:.4f}\n"
        f"n_samples: {len(y_test_values)}"
    )
    
    # Add metrics to figure
    fig.text(0.5, 0.01, metrics_text, ha='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Tight layout and supertitle
    plt.suptitle(f'Residual Analysis for {model_name}', fontsize=20, y=1.02)
    plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    filename = f"{model_name}_thesis_residuals.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path

def create_all_residual_plots():
    """Create residual plots for all models."""
    # Load models
    all_models = load_all_models()
    
    # Create output directory
    output_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "residuals"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create plots for each model
    created_files = []
    for model_name, model_data in all_models.items():
        try:
            output_path = create_thesis_residual_plot(model_name, model_data, output_dir)
            if output_path:
                created_files.append(output_path)
        except Exception as e:
            print(f"Error creating plot for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Created {len(created_files)} residual plots in {output_dir}")
    return created_files

if __name__ == "__main__":
    create_all_residual_plots()