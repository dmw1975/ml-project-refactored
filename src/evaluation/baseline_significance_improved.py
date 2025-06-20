"""
Improved statistical significance testing using test set predictions for proper statistical power.

This module implements:
1. Single model test design using 441 test samples (instead of 5 CV folds)
2. Paired Wilcoxon signed-rank test on absolute residuals
3. Proper Holm-Bonferroni correction for 3 comparisons per model
4. Best model selection based on test set RMSE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import sys
from tqdm import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.evaluation.baselines import generate_random_baseline, generate_mean_baseline, generate_median_baseline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_best_model_per_type(models_dict):
    """
    Select the best performing version of each model type based on test set RMSE.
    
    For models with multiple versions (e.g., XGBoost_Base_categorical_optuna,
    XGBoost_Yeo_categorical_optuna), select the one with lowest test RMSE.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of all model data
        
    Returns
    -------
    dict
        Dictionary with best model per type
    """
    # Group models by base type
    model_groups = {}
    
    for model_name, model_data in models_dict.items():
        if not isinstance(model_data, dict):
            continue
            
        # Skip baseline models
        if any(baseline in model_name for baseline in ['_Random', '_Mean', '_Median']):
            continue
            
        # Extract base model type (e.g., "XGBoost", "LightGBM", etc.)
        if 'XGBoost' in model_name:
            base_type = 'XGBoost'
        elif 'LightGBM' in model_name:
            base_type = 'LightGBM'
        elif 'CatBoost' in model_name:
            base_type = 'CatBoost'
        elif 'ElasticNet' in model_name:
            base_type = 'ElasticNet'
        elif 'LR_' in model_name or model_name.startswith('lr_'):
            base_type = 'LinearRegression'
        else:
            base_type = 'Other'
            
        # Get test RMSE
        test_rmse = model_data.get('test_rmse')
        if test_rmse is None and 'rmse' in model_data:
            test_rmse = model_data['rmse']
        if test_rmse is None and 'test_mse' in model_data:
            test_rmse = np.sqrt(model_data['test_mse'])
            
        if test_rmse is not None:
            if base_type not in model_groups:
                model_groups[base_type] = []
            model_groups[base_type].append({
                'name': model_name,
                'data': model_data,
                'rmse': test_rmse
            })
    
    # Select best model per group
    best_models = {}
    for base_type, models in model_groups.items():
        if models:
            # Sort by RMSE and take the best
            best_model = min(models, key=lambda x: x['rmse'])
            best_models[best_model['name']] = best_model['data']
            logger.info(f"Selected {best_model['name']} as best {base_type} model (RMSE: {best_model['rmse']:.4f})")
    
    return best_models


def perform_wilcoxon_test_on_residuals(y_true, y_pred_model, y_pred_baseline):
    """
    Perform paired Wilcoxon signed-rank test on absolute residuals.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred_model : array-like
        Model predictions
    y_pred_baseline : array-like
        Baseline predictions
        
    Returns
    -------
    float
        p-value from Wilcoxon test
    """
    # Calculate absolute residuals
    model_residuals = np.abs(y_true - y_pred_model)
    baseline_residuals = np.abs(y_true - y_pred_baseline)
    
    # Perform paired Wilcoxon signed-rank test
    try:
        stat, p_value = stats.wilcoxon(baseline_residuals, model_residuals, alternative='greater')
        return p_value
    except Exception as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        # Fall back to paired t-test
        stat, p_value = stats.ttest_rel(baseline_residuals, model_residuals)
        return p_value / 2  # One-sided test


def apply_holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni correction to p-values.
    
    Parameters
    ----------
    p_values : list or array
        Raw p-values from multiple comparisons
    alpha : float
        Significance level (default: 0.05)
        
    Returns
    -------
    tuple
        (adjusted_p_values, significant_flags)
    """
    n = len(p_values)
    if n == 0:
        return [], []
    
    # Sort p-values with their indices
    p_with_idx = [(p, i) for i, p in enumerate(p_values)]
    p_sorted = sorted(p_with_idx, key=lambda x: x[0])
    
    # Apply Holm-Bonferroni correction
    adjusted_p = []
    significant = []
    
    for rank, (p, original_idx) in enumerate(p_sorted):
        # Holm correction: multiply by (n - rank)
        adj_p = min(p * (n - rank), 1.0)
        
        # Enforce monotonicity
        if rank > 0 and adj_p < adjusted_p[rank-1]:
            adj_p = adjusted_p[rank-1]
            
        adjusted_p.append(adj_p)
        significant.append(adj_p < alpha)
    
    # Restore original order
    result_p = [None] * n
    result_sig = [None] * n
    
    for rank, (p, original_idx) in enumerate(p_sorted):
        result_p[original_idx] = adjusted_p[rank]
        result_sig[original_idx] = significant[rank]
    
    return result_p, result_sig


def run_improved_baseline_significance_tests(models_dict, output_path=None):
    """
    Run improved statistical significance tests using test set predictions.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of all model data
    output_path : str or Path, optional
        Path to save results CSV
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with significance tests
    """
    if output_path is None:
        output_path = settings.VISUALIZATION_DIR / "statistical_tests" / "baseline_significance_tests_improved.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select best models
    logger.info("Selecting best model per type based on test RMSE...")
    best_models = select_best_model_per_type(models_dict)
    
    if not best_models:
        logger.error("No valid models found for testing")
        return pd.DataFrame()
    
    results = []
    
    for model_name, model_data in tqdm(best_models.items(), desc="Testing models"):
        # Get test data
        y_test = model_data.get('y_test')
        y_pred = model_data.get('y_pred')
        if y_pred is None:
            y_pred = model_data.get('y_test_pred')
            
        y_train = model_data.get('y_train')
        
        if y_test is None or y_pred is None:
            logger.warning(f"Skipping {model_name}: missing test data")
            continue
            
        # Convert to arrays
        if hasattr(y_test, 'values'):
            y_test = y_test.values.flatten()
        else:
            y_test = np.array(y_test).flatten()
            
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values.flatten()
        else:
            y_pred = np.array(y_pred).flatten()
            
        # Log sample size
        n_samples = len(y_test)
        logger.info(f"Testing {model_name} with {n_samples} test samples")
        
        # Calculate model RMSE
        model_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        # Generate baselines
        # 1. Random baseline - uses test data range
        random_baseline = generate_random_baseline(
            y_test,
            min_val=float(y_test.min()),
            max_val=float(y_test.max()),
            seed=42
        )
        random_rmse = np.sqrt(np.mean((y_test - random_baseline) ** 2))
        
        # 2. Mean baseline - uses training data
        if y_train is not None:
            if hasattr(y_train, 'values'):
                y_train = y_train.values.flatten()
            else:
                y_train = np.array(y_train).flatten()
            mean_val, _ = generate_mean_baseline(y_train)
            mean_baseline = np.full(len(y_test), mean_val)
            mean_rmse = np.sqrt(np.mean((y_test - mean_baseline) ** 2))
            
            # 3. Median baseline - uses training data
            median_val, _ = generate_median_baseline(y_train)
            median_baseline = np.full(len(y_test), median_val)
            median_rmse = np.sqrt(np.mean((y_test - median_baseline) ** 2))
        else:
            logger.warning(f"{model_name}: No training data, skipping mean/median baselines")
            mean_baseline = None
            median_baseline = None
            mean_rmse = None
            median_rmse = None
        
        # Perform Wilcoxon tests
        p_values = []
        baseline_types = []
        baseline_rmses = []
        improvements = []
        
        # Test vs Random
        p_random = perform_wilcoxon_test_on_residuals(y_test, y_pred, random_baseline)
        p_values.append(p_random)
        baseline_types.append('Random')
        baseline_rmses.append(random_rmse)
        improvements.append((random_rmse - model_rmse) / random_rmse * 100)
        
        # Test vs Mean
        if mean_baseline is not None:
            p_mean = perform_wilcoxon_test_on_residuals(y_test, y_pred, mean_baseline)
            p_values.append(p_mean)
            baseline_types.append('Mean')
            baseline_rmses.append(mean_rmse)
            improvements.append((mean_rmse - model_rmse) / mean_rmse * 100)
        
        # Test vs Median
        if median_baseline is not None:
            p_median = perform_wilcoxon_test_on_residuals(y_test, y_pred, median_baseline)
            p_values.append(p_median)
            baseline_types.append('Median')
            baseline_rmses.append(median_rmse)
            improvements.append((median_rmse - model_rmse) / median_rmse * 100)
        
        # Apply Holm-Bonferroni correction
        adjusted_p_values, significant_flags = apply_holm_bonferroni_correction(p_values)
        
        # Log raw and adjusted p-values
        logger.info(f"{model_name} p-values:")
        for i, baseline_type in enumerate(baseline_types):
            logger.info(f"  vs {baseline_type}: raw={p_values[i]:.6f}, adjusted={adjusted_p_values[i]:.6f}, significant={significant_flags[i]}")
        
        # Store results for each baseline
        for i, baseline_type in enumerate(baseline_types):
            results.append({
                'Model': model_name,
                'Baseline': baseline_type,
                'Model Mean RMSE': model_rmse,
                'Baseline Mean RMSE': baseline_rmses[i],
                'Improvement (%)': improvements[i],
                'p-value': p_values[i],
                'Test Type': 'Wilcoxon',
                'p-value-adjusted': adjusted_p_values[i],
                'Significant': significant_flags[i],
                'Test Samples': n_samples
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by baseline type and improvement
        results_df = results_df.sort_values(['Baseline', 'Improvement (%)'], ascending=[True, False])
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TEST SUMMARY")
        print("="*80)
        
        for baseline in ['Random', 'Mean', 'Median']:
            baseline_df = results_df[results_df['Baseline'] == baseline]
            if not baseline_df.empty:
                n_significant = baseline_df['Significant'].sum()
                n_total = len(baseline_df)
                print(f"\nvs {baseline} Baseline:")
                print(f"  Significant improvements: {n_significant}/{n_total} models")
                
                # Show top 3 models
                top_models = baseline_df.nlargest(3, 'Improvement (%)')
                for _, row in top_models.iterrows():
                    sig_marker = "✓" if row['Significant'] else "✗"
                    print(f"  {sig_marker} {row['Model']}: {row['Improvement (%)']:.1f}% improvement (p={row['p-value']:.6f}, p_adj={row['p-value-adjusted']:.6f})")
    
    return results_df


def create_improved_significance_visualization(results_df, output_dir=None):
    """
    Create improved visualization showing significance test results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from improved significance tests
    output_dir : str or Path, optional
        Directory to save visualizations
    """
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create separate plots for each baseline type
    for baseline in ['Random', 'Mean', 'Median']:
        baseline_df = results_df[results_df['Baseline'] == baseline].copy()
        
        if baseline_df.empty:
            continue
            
        # Sort by improvement
        baseline_df = baseline_df.sort_values('Improvement (%)', ascending=True)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(baseline_df) * 0.5)))
        
        # Plot 1: RMSE comparison
        y_pos = np.arange(len(baseline_df))
        
        # Model RMSE
        ax1.barh(y_pos - 0.2, baseline_df['Model Mean RMSE'], 0.4, 
                label='Model RMSE', color='#2196F3', alpha=0.8)
        
        # Baseline RMSE
        ax1.barh(y_pos + 0.2, baseline_df['Baseline Mean RMSE'], 0.4,
                label=f'{baseline} Baseline RMSE', color='#F44336', alpha=0.8)
        
        # Labels
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(baseline_df['Model'].str.replace('_', ' ', regex=False))
        ax1.set_xlabel('RMSE')
        ax1.set_title(f'Model vs {baseline} Baseline RMSE (Test Set, n={baseline_df.iloc[0]["Test Samples"]})')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Improvement with significance
        colors = ['#4CAF50' if sig else '#FFC107' for sig in baseline_df['Significant']]
        bars = ax2.barh(y_pos, baseline_df['Improvement (%)'], color=colors, alpha=0.8)
        
        # Add p-value annotations
        for i, (_, row) in enumerate(baseline_df.iterrows()):
            # Add significance marker
            sig_marker = "***" if row['p-value'] < 0.001 else ("**" if row['p-value'] < 0.01 else ("*" if row['p-value'] < 0.05 else ""))
            
            # Position text
            x_pos = row['Improvement (%)'] + 0.5
            text = f"{row['Improvement (%)']:.1f}%{sig_marker}"
            
            ax2.text(x_pos, i, text, va='center', fontsize=9,
                    weight='bold' if row['Significant'] else 'normal')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title(f'Improvement vs {baseline} Baseline (Holm-Bonferroni Corrected)')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label='Significant (p_adj < 0.05)', alpha=0.8),
            Patch(facecolor='#FFC107', label='Not Significant', alpha=0.8)
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        # Overall title
        fig.suptitle(f'Statistical Significance Test Results vs {baseline} Baseline\n'
                    f'Using Test Set Predictions ({baseline_df.iloc[0]["Test Samples"]} samples)',
                    fontsize=14)
        
        # Add note
        fig.text(0.5, 0.02, 
                '*** p < 0.001, ** p < 0.01, * p < 0.05 (raw p-values)\n'
                'Significance based on Holm-Bonferroni adjusted p-values',
                ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        # Save
        output_path = output_dir / f"improved_significance_{baseline.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        plt.close()


if __name__ == "__main__":
    # Load all models
    from src.utils import io
    
    logger.info("Loading all models...")
    all_models = io.load_all_models()
    logger.info(f"Loaded {len(all_models)} models")
    
    # Run improved tests
    results_df = run_improved_baseline_significance_tests(all_models)
    
    # Create visualizations
    if not results_df.empty:
        create_improved_significance_visualization(results_df)
        
    logger.info("Improved statistical significance testing complete")