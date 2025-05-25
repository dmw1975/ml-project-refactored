#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate complete baseline comparisons for all models.
Creates visualizations showing how all models compare to mean, median, and random baselines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

from config import settings
from utils import io

def generate_baseline_predictions(y_true, baseline_type='mean', y_train=None):
    """Generate baseline predictions."""
    if baseline_type == 'mean':
        if y_train is not None:
            baseline_value = np.mean(y_train)
        else:
            baseline_value = np.mean(y_true)
        return np.full_like(y_true, baseline_value)
    
    elif baseline_type == 'median':
        if y_train is not None:
            baseline_value = np.median(y_train)
        else:
            baseline_value = np.median(y_true)
        return np.full_like(y_true, baseline_value)
    
    elif baseline_type == 'random':
        np.random.seed(42)
        min_val = np.min(y_true)
        max_val = np.max(y_true)
        return np.random.uniform(min_val, max_val, size=len(y_true))
    
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

def calculate_improvement(y_true, y_pred, baseline_pred):
    """Calculate improvement of model over baseline."""
    model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    
    improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    
    # Statistical test
    model_residuals = y_true - y_pred
    baseline_residuals = y_true - baseline_pred
    
    # Use paired t-test since we're comparing on the same test set
    t_stat, p_value = stats.ttest_rel(np.abs(baseline_residuals), np.abs(model_residuals))
    
    return {
        'model_rmse': model_rmse,
        'baseline_rmse': baseline_rmse,
        'improvement': improvement,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def extract_model_info(model_name):
    """Extract model family and dataset info from model name."""
    # Model family
    if 'XGB' in model_name or 'XGBoost' in model_name:
        family = 'XGBoost'
    elif 'LightGBM' in model_name:
        family = 'LightGBM'
    elif 'CatBoost' in model_name:
        family = 'CatBoost'
    elif 'ElasticNet' in model_name:
        family = 'ElasticNet'
    elif 'LR_' in model_name:
        family = 'Linear Regression'
    else:
        family = 'Unknown'
    
    # Dataset
    if 'Base_Random' in model_name:
        dataset = 'Base_Random'
    elif 'Yeo_Random' in model_name:
        dataset = 'Yeo_Random'
    elif 'Base' in model_name:
        dataset = 'Base'
    elif 'Yeo' in model_name:
        dataset = 'Yeo'
    else:
        dataset = 'Unknown'
    
    # Optimization
    is_optimized = 'optuna' in model_name or 'ElasticNet' in model_name
    
    return family, dataset, is_optimized

def create_baseline_comparison_plot(results_df, baseline_type, output_dir):
    """Create visualization for baseline comparison."""
    # Filter for this baseline type
    df = results_df[results_df['baseline_type'] == baseline_type].copy()
    
    # Sort by improvement
    df = df.sort_values('improvement', ascending=False)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(df) * 0.3)))
    
    # Left plot: RMSE comparison
    y_pos = np.arange(len(df))
    bar_width = 0.35
    
    # Model RMSE bars
    bars1 = ax1.barh(y_pos - bar_width/2, df['model_rmse'], bar_width, 
                      label='Model RMSE', color='#2980b9', alpha=0.7)
    
    # Baseline RMSE bars
    bars2 = ax1.barh(y_pos + bar_width/2, df['baseline_rmse'], bar_width,
                      label=f'{baseline_type.capitalize()} Baseline RMSE', color='#c0392b', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['display_name'])
    ax1.set_xlabel('RMSE')
    ax1.set_title(f'Model vs {baseline_type.capitalize()} Baseline RMSE')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Right plot: Improvement percentage
    colors = ['#27ae60' if sig else '#e74c3c' for sig in df['significant']]
    bars3 = ax2.barh(y_pos, df['improvement'], color=colors, alpha=0.7)
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        text = f"{row['improvement']:.1f}%"
        if row['significant']:
            text += " *"
        ax2.text(row['improvement'] + 1, i, text, va='center', fontsize=9,
                fontweight='bold' if row['significant'] else 'normal')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title(f'Performance Improvement vs {baseline_type.capitalize()} Baseline')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Significant (p < 0.05)', alpha=0.7),
        Patch(facecolor='#e74c3c', label='Not Significant', alpha=0.7)
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    # Overall title
    fig.suptitle(f'All Models vs {baseline_type.capitalize()} Baseline', fontsize=14)
    
    # Add note
    fig.text(0.5, 0.01, '* indicates statistical significance (p < 0.05)', 
             ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'complete_baseline_comparison_{baseline_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {baseline_type} baseline comparison to {output_path}")
    
    return fig

def main():
    """Generate complete baseline comparisons for all models."""
    print("Loading all models...")
    all_models = io.load_all_models()
    print(f"Found {len(all_models)} models total")
    
    # Prepare results
    results = []
    
    # Process each model
    for model_name, model_data in all_models.items():
        # Skip if not a dictionary
        if not isinstance(model_data, dict):
            print(f"Skipping {model_name} - not in expected format")
            continue
        
        # Get predictions
        y_test = model_data.get('y_test')
        y_pred = model_data.get('y_pred', model_data.get('y_test_pred'))
        
        if y_test is None or y_pred is None:
            print(f"Skipping {model_name} - missing test data or predictions")
            continue
        
        # Convert to numpy arrays
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Get training data if available
        y_train = model_data.get('y_train')
        if y_train is not None:
            y_train = np.array(y_train).flatten()
        
        print(f"Processing {model_name}...")
        
        # Extract model info
        family, dataset, is_optimized = extract_model_info(model_name)
        
        # Create display name
        opt_suffix = " (opt)" if is_optimized else ""
        display_name = f"{family}{opt_suffix} - {dataset}"
        
        # Compare against each baseline
        for baseline_type in ['mean', 'median', 'random']:
            # Generate baseline predictions
            baseline_pred = generate_baseline_predictions(y_test, baseline_type, y_train)
            
            # Calculate metrics
            metrics = calculate_improvement(y_test, y_pred, baseline_pred)
            
            results.append({
                'model': model_name,
                'display_name': display_name,
                'family': family,
                'dataset': dataset,
                'optimized': is_optimized,
                'baseline_type': baseline_type,
                **metrics
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = settings.VISUALIZATION_DIR / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "complete_baseline_comparisons.csv", index=False)
    print(f"\nSaved results to {output_dir / 'complete_baseline_comparisons.csv'}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    for baseline_type in ['mean', 'median', 'random']:
        create_baseline_comparison_plot(results_df, baseline_type, output_dir)
    
    # Print summary
    print("\nSummary of baseline comparisons:")
    for baseline_type in ['mean', 'median', 'random']:
        baseline_df = results_df[results_df['baseline_type'] == baseline_type]
        
        print(f"\n{baseline_type.capitalize()} Baseline:")
        print(f"  Total models: {len(baseline_df)}")
        print(f"  Average improvement: {baseline_df['improvement'].mean():.2f}%")
        print(f"  Best model: {baseline_df.iloc[0]['display_name']} ({baseline_df.iloc[0]['improvement']:.2f}%)")
        print(f"  Significant improvements: {baseline_df['significant'].sum()} / {len(baseline_df)}")
    
    return results_df

if __name__ == "__main__":
    main()