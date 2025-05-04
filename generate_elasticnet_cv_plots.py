"""Script to generate missing ElasticNet CV plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def mean_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval."""
    import scipy.stats as st
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def generate_elasticnet_cv_plots():
    """
    Generate ElasticNet CV RMSE distribution and best parameters plots.
    
    These plots were missing from the performance/linear folder.
    """
    # Import required modules
    import seaborn as sns
    from matplotlib.lines import Line2D
    
    # Set up style
    style = setup_visualization_style()
    
    # Load CV results
    try:
        cv_results = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
        print(f"Loaded ElasticNet CV results")
    except Exception as e:
        print(f"No ElasticNet cross-validation results found: {e}")
        return
    
    # Set up main output directory
    perf_dir = settings.VISUALIZATION_DIR / "performance"
    io.ensure_dir(perf_dir)
    
    # Create elasticnet directory for ElasticNet-specific plots
    elasticnet_dir = perf_dir / "elasticnet"
    io.ensure_dir(elasticnet_dir)
    
    # 1. CV RMSE Distribution Plot
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Boxplot for RMSE distribution
    box = sns.boxplot(x='Dataset', y='RMSE', data=rmse_df, palette='pastel', ax=ax)
    
    # Stripplot for individual fold RMSEs
    strip = sns.stripplot(x='Dataset', y='RMSE', data=rmse_df, color='gray', alpha=0.6, jitter=True, ax=ax)
    
    # Plot mean and 95% CI as red points with error bars
    for i, dataset in enumerate(rmse_df['Dataset'].unique()):
        rmse_vals = rmse_df[rmse_df['Dataset'] == dataset]['RMSE']
        mean = np.mean(rmse_vals)
        ci_low, ci_high = mean_confidence_interval(rmse_vals)[1:]
        err = ax.errorbar(i, mean, yerr=[[mean - ci_low], [ci_high - mean]],
                          fmt='o', color='red', capsize=5, 
                          label='Mean ± 95% CI' if i == 0 else "")
    
    # Title and axes
    ax.set_title('ElasticNet RMSE Distribution per Dataset', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_xlabel('Dataset')
    plt.xticks(rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='RMSE Distribution (Boxplot)',
               markerfacecolor='lightblue', markersize=15),
        Line2D([0], [0], marker='o', color='gray', label='Individual CV Fold RMSE',
               linestyle='None', markersize=8, alpha=0.6),
        Line2D([0], [0], marker='o', color='red', label='Mean ± 95% CI',
               linestyle='None', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save to elasticnet directory only
    save_figure(fig, "elasticnet_cv_rmse_distribution", elasticnet_dir)
    
    print(f"ElasticNet CV distribution plot saved to {elasticnet_dir}")
    
    # 2. Best Parameters Plot
    # Extract best parameters for visualization
    best_params = []
    for result in cv_results:
        dataset = result['dataset']
        best_alpha, best_l1 = result['best_params']
        best_cv_mse = result['best_cv_mse']
        best_cv_rmse = np.sqrt(best_cv_mse) if 'best_cv_mse' == result else result.get('best_cv_rmse', 0)
        
        best_params.append({
            'Dataset': dataset,
            'Alpha': best_alpha,
            'L1_Ratio': best_l1,
            'RMSE': best_cv_rmse
        })
    
    params_df = pd.DataFrame(best_params)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Alpha plot
    ax = axes[0]
    bars = ax.bar(params_df['Dataset'], params_df['Alpha'], color='#3498db')
    ax.set_title('Optimal Alpha (Regularization Strength)', fontsize=14)
    ax.set_ylabel('Alpha')
    ax.set_xlabel('Dataset')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # L1 ratio plot
    ax = axes[1]
    bars = ax.bar(params_df['Dataset'], params_df['L1_Ratio'], color='#e74c3c')
    ax.set_title('Optimal L1 Ratio (0=Ridge, 1=Lasso)', fontsize=14)
    ax.set_ylabel('L1 Ratio')
    ax.set_xlabel('Dataset')
    
    # Add reference lines for Ridge (0.0), Elastic (0.5), and Lasso (1.0)
    ax.axhline(y=0.0, color='blue', linestyle='--', alpha=0.5, label='Ridge (L2 Only)')
    ax.axhline(y=0.5, color='purple', linestyle='--', alpha=0.5, label='Equal Mix (L1 + L2)')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Lasso (L1 Only)')
    
    ax.legend()
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save to elasticnet directory only
    save_figure(fig, "elasticnet_best_parameters", elasticnet_dir)
    
    print(f"ElasticNet best parameters plot saved to {elasticnet_dir}")
    
    return True

if __name__ == "__main__":
    generate_elasticnet_cv_plots()