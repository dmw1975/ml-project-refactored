"""
Comparison script for all trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io
from evaluation.metrics import load_all_models

def compare_all_models():
    """Compare all trained models and visualize the results."""
    # Load all models
    print("Loading all trained models...")
    all_models = load_all_models()
    
    if not all_models:
        print("No models found. Please train models first.")
        return
    
    print(f"Found {len(all_models)} trained models.")
    
    # Create a comparison DataFrame
    model_metrics = []
    for model_name, model_data in all_models.items():
        model_type = model_data.get('model_type', 'Unknown')
        if 'XGB_' in model_name:
            model_family = 'XGBoost'
            optuna_optimized = 'optuna' in model_name
        elif 'LightGBM_' in model_name:
            model_family = 'LightGBM'
            optuna_optimized = 'optuna' in model_name
        elif 'ElasticNet' in model_name:
            model_family = 'ElasticNet'
            optuna_optimized = True  # ElasticNet is always optimized
        else:
            model_family = 'Linear Regression'
            optuna_optimized = False
        
        # Get dataset name
        if 'LR_' in model_name:
            dataset = model_name.replace('LR_', '')
        elif 'ElasticNet_LR_' in model_name:
            dataset = model_name.replace('ElasticNet_LR_', '')
        elif 'XGB_' in model_name:
            dataset = model_name.replace('XGB_', '').split('_')[0]
        elif 'LightGBM_' in model_name:
            dataset = model_name.replace('LightGBM_', '').split('_')[0]
        else:
            dataset = 'Unknown'
        
        metrics = {
            'model_name': model_name,
            'model_family': model_family,
            'model_type': model_type,
            'dataset': dataset,
            'optuna_optimized': optuna_optimized,
            'RMSE': model_data.get('RMSE', np.sqrt(model_data.get('MSE', 0))),
            'MAE': model_data.get('MAE', 0),
            'MSE': model_data.get('MSE', 0),
            'R2': model_data.get('R2', 0),
            'n_companies': model_data.get('n_companies', 0),
            'n_features': model_data.get('n_features', 0)
        }
        model_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    
    # Save to CSV
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/all_models_comparison.csv", index=False)
    
    # Create output directory for plots
    output_dir = settings.VISUALIZATION_DIR / "summary"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    
    # 1. Compare RMSE across model families
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=metrics_df[metrics_df['optuna_optimized']],  # Only optimized models
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
    plt.title('RMSE Comparison Across Models (Optimized Models Only)', fontsize=16)
    plt.ylabel('RMSE (lower is better)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Model Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_comparison.png", dpi=300)
    
    # 2. Compare R² across model families
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=metrics_df[metrics_df['optuna_optimized']],  # Only optimized models
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
    plt.title('R² Comparison Across Models (Optimized Models Only)', fontsize=16)
    plt.ylabel('R² (higher is better)', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.legend(title='Model Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_comparison.png", dpi=300)
    
    # 3. Optuna improvement for XGBoost and LightGBM
    improvement_data = []
    
    for model_family in ['XGBoost', 'LightGBM']:
        family_df = metrics_df[metrics_df['model_family'] == model_family]
        
        for dataset in family_df['dataset'].unique():
            basic = family_df[(family_df['dataset'] == dataset) & (~family_df['optuna_optimized'])]['RMSE'].values
            optuna = family_df[(family_df['dataset'] == dataset) & (family_df['optuna_optimized'])]['RMSE'].values
            
            if len(basic) > 0 and len(optuna) > 0:
                improvement = ((basic[0] - optuna[0]) / basic[0]) * 100
                improvement_data.append({
                    'model_family': model_family,
                    'dataset': dataset,
                    'improvement': improvement
                })
    
    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=imp_df,
            x='dataset',
            y='improvement',
            hue='model_family',
            palette={
                'XGBoost': '#2ecc71',
                'LightGBM': '#f39c12'
            }
        )
        plt.title('RMSE Improvement with Optuna Optimization', fontsize=16)
        plt.ylabel('Improvement (%)', fontsize=12)
        plt.xlabel('Dataset', fontsize=12)
        plt.legend(title='Model Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optuna_improvement_comparison.png", dpi=300)
    
    print("\nModel comparison completed. Results saved to metrics directory and visualization plots saved.")
    return metrics_df

if __name__ == "__main__":
    # Run the comparison
    compare_all_models()