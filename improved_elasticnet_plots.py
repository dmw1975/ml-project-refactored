"""Improved ElasticNet visualization functions."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np
from scipy import stats
from sklearn.linear_model import ElasticNetCV

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def plot_elasticnet_comparison():
    """Compare basic vs. optimized ElasticNet models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load ElasticNet results
    try:
        en_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
    except Exception as e:
        print(f"No ElasticNet models found. Please train ElasticNet models first. Error: {e}")
        return None
    
    # Load Linear Regression models for comparison
    try:
        lr_models = io.load_model("linear_regression_models.pkl", settings.MODEL_DIR)
        print(f"Successfully loaded Linear Regression models for comparison")
    except Exception as e:
        print(f"Warning: Could not load Linear Regression models: {e}")
        lr_models = {}
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/elasticnet"
    io.ensure_dir(output_dir)
    
    # Extract performance metrics
    performance_data = []
    
    # Add ElasticNet models
    for name, model_data in en_models.items():
        # Skip if not a valid model
        if 'RMSE' not in model_data:
            continue
            
        # Parse dataset
        if 'Base_Random' in name:
            dataset = 'Base R'
        elif 'Yeo_Random' in name:
            dataset = 'Yeo R'
        elif 'Base' in name:
            dataset = 'Base'
        elif 'Yeo' in name:
            dataset = 'Yeo'
        else:
            dataset = 'Unknown'
        
        performance_data.append({
            'model_name': name,
            'RMSE': model_data['RMSE'],
            'R2': model_data.get('R2', 0),  # Default to 0 if not present
            'MAE': model_data.get('MAE', 0),
            'dataset': dataset,
            'Type': 'ElasticNet'
        })
    
    # Add Linear Regression models
    for name, model_data in lr_models.items():
        # Skip if not a valid model
        if 'RMSE' not in model_data:
            continue
            
        # Parse dataset
        if 'Base_Random' in name:
            dataset = 'Base R'
        elif 'Yeo_Random' in name:
            dataset = 'Yeo R'
        elif 'Base' in name:
            dataset = 'Base'
        elif 'Yeo' in name:
            dataset = 'Yeo'
        else:
            dataset = 'Unknown'
        
        performance_data.append({
            'model_name': name,
            'RMSE': model_data['RMSE'],
            'R2': model_data.get('R2', 0),  # Default to 0 if not present
            'MAE': model_data.get('MAE', 0),
            'dataset': dataset,
            'Type': 'LinearRegression'
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Check if we have enough data
    if len(perf_df) < 1:
        print("Not enough models found for comparison.")
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
        palette={'ElasticNet': '#e74c3c', 'LinearRegression': '#3498db'}
    )
    ax.set_title('RMSE: Linear Regression vs. ElasticNet', fontsize=14)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # R² comparison
    ax = axes[1]
    bars = sns.barplot(
        data=perf_df, 
        x='dataset', 
        y='R2', 
        hue='Type',
        ax=ax,
        palette={'ElasticNet': '#e74c3c', 'LinearRegression': '#3498db'}
    )
    ax.set_title('R²: Linear Regression vs. ElasticNet', fontsize=14)
    ax.set_ylabel('R² (higher is better)')
    ax.set_xlabel('Dataset')
    ax.legend(title='Model Type')
    
    # Add value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, "elasticnet_vs_linear_regression", output_dir)
    
    print(f"ElasticNet vs. Linear Regression comparison plot saved to {output_dir}")
    
    # Calculate improvement percentage
    improvement_data = []
    for dataset in perf_df['dataset'].unique():
        lr_df = perf_df[(perf_df['dataset'] == dataset) & (perf_df['Type'] == 'LinearRegression')]
        en_df = perf_df[(perf_df['dataset'] == dataset) & (perf_df['Type'] == 'ElasticNet')]
        
        if not lr_df.empty and not en_df.empty:
            lr_rmse = lr_df['RMSE'].values[0]
            en_rmse = en_df['RMSE'].values[0]
            
            improvement = (lr_rmse - en_rmse) / lr_rmse * 100
            
            improvement_data.append({
                'dataset': dataset,
                'improvement': improvement
            })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        
        # Create improvement plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(improvement_df['dataset'], improvement_df['improvement'], color='#2ecc71')
        
        ax.set_title('RMSE Improvement from ElasticNet vs. Linear Regression', fontsize=14)
        ax.set_ylabel('Improvement (%)')
        ax.set_xlabel('Dataset')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(0.1, abs(height)),
                    f'{height:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, "elasticnet_improvement", output_dir)
    
    return fig

def plot_elasticnet_hyperparameters():
    """Visualize important hyperparameters for ElasticNet models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load ElasticNet results and params
    try:
        en_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
        en_params = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
    except Exception as e:
        print(f"ElasticNet models or parameters not found. Error: {e}")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/elasticnet/hyperparameters"
    io.ensure_dir(output_dir)
    
    # 1. Alpha Comparison
    alpha_data = []
    for model_name, model_data in en_models.items():
        if 'ElasticNet' in model_name and 'best_params' in model_data:
            alpha = model_data['best_params'][0]  # First value is alpha
            
            # Parse dataset
            if 'Base_Random' in model_name:
                dataset = 'Base R'
            elif 'Yeo_Random' in model_name:
                dataset = 'Yeo R'
            elif 'Base' in model_name:
                dataset = 'Base'
            elif 'Yeo' in model_name:
                dataset = 'Yeo'
            else:
                dataset = 'Unknown'
                
            alpha_data.append({
                'model': model_name,
                'dataset': dataset,
                'alpha': alpha
            })
    
    if alpha_data:
        alpha_df = pd.DataFrame(alpha_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(alpha_df['dataset'], alpha_df['alpha'], color='#3498db')
        
        ax.set_title('Optimal Alpha (Regularization Strength) by Dataset', fontsize=14)
        ax.set_ylabel('Alpha')
        ax.set_xlabel('Dataset')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(0.1, abs(height)),
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, "elasticnet_alpha_comparison", output_dir)
        print(f"ElasticNet alpha comparison plot saved to {output_dir}")
    
    # 2. L1 Ratio Comparison
    l1_ratio_data = []
    for model_name, model_data in en_models.items():
        if 'ElasticNet' in model_name and 'best_params' in model_data:
            l1_ratio = model_data['best_params'][1]  # Second value is l1_ratio
            
            # Parse dataset
            if 'Base_Random' in model_name:
                dataset = 'Base R'
            elif 'Yeo_Random' in model_name:
                dataset = 'Yeo R'
            elif 'Base' in model_name:
                dataset = 'Base'
            elif 'Yeo' in model_name:
                dataset = 'Yeo'
            else:
                dataset = 'Unknown'
                
            l1_ratio_data.append({
                'model': model_name,
                'dataset': dataset,
                'l1_ratio': l1_ratio
            })
    
    if l1_ratio_data:
        l1_ratio_df = pd.DataFrame(l1_ratio_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(l1_ratio_df['dataset'], l1_ratio_df['l1_ratio'], color='#e74c3c')
        
        ax.set_title('Optimal L1 Ratio by Dataset', fontsize=14)
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
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(0.1, abs(height)),
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, "elasticnet_l1_ratio_comparison", output_dir)
        print(f"ElasticNet L1 ratio comparison plot saved to {output_dir}")
    
    # 3. Feature Count Comparison (non-zero coefficients)
    feature_count_data = []
    for model_name, model_data in en_models.items():
        if 'ElasticNet' in model_name and 'model' in model_data:
            model = model_data['model']
            
            # Count non-zero coefficients
            if hasattr(model, 'coef_'):
                non_zero_count = np.sum(model.coef_ != 0)
                
                # Parse dataset
                if 'Base_Random' in model_name:
                    dataset = 'Base R'
                elif 'Yeo_Random' in model_name:
                    dataset = 'Yeo R'
                elif 'Base' in model_name:
                    dataset = 'Base'
                elif 'Yeo' in model_name:
                    dataset = 'Yeo'
                else:
                    dataset = 'Unknown'
                    
                feature_count_data.append({
                    'model': model_name,
                    'dataset': dataset,
                    'non_zero_features': non_zero_count
                })
    
    if feature_count_data:
        feature_count_df = pd.DataFrame(feature_count_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(feature_count_df['dataset'], feature_count_df['non_zero_features'], color='#9b59b6')
        
        ax.set_title('Number of Non-Zero Coefficients by Dataset', fontsize=14)
        ax.set_ylabel('Non-Zero Feature Count')
        ax.set_xlabel('Dataset')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(0.1, abs(height)),
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, "elasticnet_feature_count", output_dir)
        print(f"ElasticNet feature count plot saved to {output_dir}")
    
    # 4. Regularization Path if CV data available
    for result in en_params:
        dataset = result['dataset']
        cv_results = result.get('cv_results', None)
        
        if cv_results is not None and isinstance(cv_results, pd.DataFrame) and not cv_results.empty:
            try:
                # Get alphas and create corresponding L1 ratios
                if 'alpha' in cv_results.columns and 'l1_ratio' in cv_results.columns:
                    # Create plot of RMSE vs alpha by l1_ratio
                    alphas = sorted(cv_results['alpha'].unique())
                    l1_ratios = sorted(cv_results['l1_ratio'].unique())
                    
                    if len(alphas) > 1 and len(l1_ratios) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        for l1_ratio in l1_ratios:
                            subset = cv_results[cv_results['l1_ratio'] == l1_ratio]
                            if len(subset) > 0:
                                # Group by alpha and aggregate
                                rmse_by_alpha = subset.groupby('alpha')['mean_rmse'].mean().reset_index()
                                rmse_by_alpha = rmse_by_alpha.sort_values('alpha')
                                
                                ax.plot(rmse_by_alpha['alpha'], rmse_by_alpha['mean_rmse'], 
                                        'o-', label=f'L1 Ratio = {l1_ratio:.2f}')
                        
                        # Find best combination
                        best_row = cv_results.loc[cv_results['mean_rmse'].idxmin()]
                        best_alpha = best_row['alpha']
                        best_l1 = best_row['l1_ratio']
                        best_rmse = best_row['mean_rmse']
                        
                        # Mark best point
                        ax.scatter([best_alpha], [best_rmse], color='red', s=100, zorder=10, 
                                label=f'Best: α={best_alpha:.4f}, L1={best_l1:.2f}')
                        
                        ax.set_title(f'RMSE vs. Alpha by L1 Ratio - {dataset}', fontsize=14)
                        ax.set_xlabel('Alpha (Regularization Strength)')
                        ax.set_ylabel('Mean RMSE (Cross-Validation)')
                        ax.set_xscale('log')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        plt.tight_layout()
                        save_figure(fig, f"elasticnet_regularization_path_{dataset}", output_dir)
                        print(f"ElasticNet regularization path plot saved for {dataset}")
            except Exception as e:
                print(f"Error creating regularization path plot for {dataset}: {e}")
    
    # 5. Cross-validation RMSE distribution
    for result in en_params:
        dataset = result['dataset']
        cv_results = result.get('cv_results', None)
        
        if cv_results is not None and isinstance(cv_results, pd.DataFrame) and not cv_results.empty:
            try:
                # Get CV scores for all combinations
                if 'alpha' in cv_results.columns and 'l1_ratio' in cv_results.columns:
                    # Find best combination
                    best_row = cv_results.loc[cv_results['mean_rmse'].idxmin()]
                    best_alpha = best_row['alpha']
                    best_l1 = best_row['l1_ratio']
                    
                    # Get CV scores for all combinations
                    rmse_data = []
                    
                    for _, row in cv_results.iterrows():
                        alpha = row['alpha']
                        l1_ratio = row['l1_ratio']
                        mean_rmse = row['mean_rmse']
                        std_rmse = row.get('std_rmse', 0)
                        
                        is_best = (alpha == best_alpha) and (l1_ratio == best_l1)
                        
                        rmse_data.append({
                            'alpha': alpha,
                            'l1_ratio': l1_ratio, 
                            'mean_rmse': mean_rmse,
                            'std_rmse': std_rmse,
                            'is_best': is_best,
                            'label': f'α={alpha:.4f}, L1={l1_ratio:.2f}'
                        })
                    
                    rmse_df = pd.DataFrame(rmse_data)
                    
                    # Sort by mean RMSE (ascending)
                    rmse_df = rmse_df.sort_values('mean_rmse')
                    
                    # Take top 10 combinations
                    top_combinations = min(10, len(rmse_df))
                    rmse_df = rmse_df.head(top_combinations)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot mean RMSE with error bars for top combinations
                    bars = ax.bar(
                        range(len(rmse_df)), 
                        rmse_df['mean_rmse'],
                        yerr=rmse_df['std_rmse'],
                        color=[('#2ecc71' if is_best else '#3498db') for is_best in rmse_df['is_best']],
                        capsize=5
                    )
                    
                    # Set x-tick labels
                    ax.set_xticks(range(len(rmse_df)))
                    ax.set_xticklabels(rmse_df['label'], rotation=45, ha='right')
                    
                    ax.set_title(f'Top {top_combinations} ElasticNet Hyperparameter Combinations - {dataset}', 
                                fontsize=14)
                    ax.set_ylabel('Mean RMSE (Cross-Validation)')
                    ax.set_xlabel('Hyperparameter Combination')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    save_figure(fig, f"elasticnet_cv_performance_{dataset}", output_dir)
                    print(f"ElasticNet CV performance plot saved for {dataset}")
            except Exception as e:
                print(f"Error creating CV performance plot for {dataset}: {e}")
    
    return True


def plot_elasticnet_optimization_history():
    """Visualize the optimization process for ElasticNet models."""
    # Set up style
    style = setup_visualization_style()
    
    # Load ElasticNet params
    try:
        en_params = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
    except Exception as e:
        print(f"ElasticNet parameters not found. Error: {e}")
        return None
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "performance/elasticnet/optimization"
    io.ensure_dir(output_dir)
    
    # Loop through datasets
    for result in en_params:
        dataset = result['dataset']
        cv_results = result.get('cv_results', None)
        
        if cv_results is not None and isinstance(cv_results, pd.DataFrame) and not cv_results.empty:
            try:
                # Sort by alpha and l1_ratio to simulate optimization history
                cv_results = cv_results.sort_values(['alpha', 'l1_ratio'])
                
                # Add iteration number
                cv_results['iteration'] = range(1, len(cv_results) + 1)
                
                # Find best iteration
                best_idx = cv_results['mean_rmse'].idxmin()
                best_iter = cv_results.loc[best_idx, 'iteration']
                best_rmse = cv_results.loc[best_idx, 'mean_rmse']
                best_alpha = cv_results.loc[best_idx, 'alpha']
                best_l1 = cv_results.loc[best_idx, 'l1_ratio']
                
                # Create optimization history plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(cv_results['iteration'], cv_results['mean_rmse'], 'o-', alpha=0.6)
                
                # Mark best point
                ax.scatter([best_iter], [best_rmse], color='red', s=100, zorder=10,
                          label=f'Best: α={best_alpha:.4f}, L1={best_l1:.2f}, RMSE={best_rmse:.4f}')
                
                # Add running best line
                running_best = cv_results['mean_rmse'].cummin()
                ax.step(cv_results['iteration'], running_best, 'r--', alpha=0.7, 
                       where='post', label='Running Best')
                
                ax.set_title(f'ElasticNet Optimization Progress - {dataset}', fontsize=14)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Mean RMSE (Cross-Validation)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plt.tight_layout()
                save_figure(fig, f"elasticnet_optimization_history_{dataset}", output_dir)
                print(f"ElasticNet optimization history plot saved for {dataset}")
            except Exception as e:
                print(f"Error creating optimization history plot for {dataset}: {e}")
    
    return True


def improved_elasticnet_visualizations():
    """Generate all improved ElasticNet visualizations."""
    print("Generating improved ElasticNet visualizations...")
    
    plot_elasticnet_comparison()
    plot_elasticnet_hyperparameters()
    plot_elasticnet_optimization_history()
    
    print("ElasticNet visualizations completed.")

if __name__ == "__main__":
    improved_elasticnet_visualizations()