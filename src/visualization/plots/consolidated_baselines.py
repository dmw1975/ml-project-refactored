import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parents[2].absolute()
sys.path.append(str(project_root))

from src.config import settings


def create_consolidated_baseline_comparison(
    baseline_data_path, 
    output_dir, 
    metric='RMSE',
    figsize=(16, 10), 
    dpi=300
):
    """
    Create a consolidated visualization comparing model performance against all baseline types.
    Shows all three baselines (mean, median, random) as horizontal lines with the model
    performance as bars. Improvement is calculated relative to the best (hardest to beat) baseline.
    
    Parameters
    ----------
    baseline_data_path : str
        Path to the CSV file containing baseline comparison data
    output_dir : str
        Directory to save the output visualization
    metric : str, optional
        Metric to visualize (default: 'RMSE', can be 'RMSE', 'MAE', 'MSE', 'R²')
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the output image
        
    Returns
    -------
    str
        Path to the saved visualization
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    df = pd.read_csv(baseline_data_path)
    
    # Handle R² vs R2 naming
    if metric == 'R²' and 'R²' not in df.columns and 'R2' in df.columns:
        metric = 'R2'
    
    # Extract unique models (without baseline type suffix)
    def get_base_model_name(model_name):
        # Remove baseline type suffixes
        for suffix in ['_mean', '_median', '_random']:
            if model_name.endswith(suffix):
                return model_name[:-len(suffix)]
        return model_name
    
    df['Base Model'] = df['Model'].apply(get_base_model_name)
    
    # Get unique models
    unique_models = df['Base Model'].unique()
    
    # Create a DataFrame with consolidated data
    consolidated_data = []
    
    for model in unique_models:
        model_data = df[df['Base Model'] == model]
        
        # Get model performance (should be same across baseline types)
        model_perf = model_data[metric].iloc[0]
        
        # Get baseline performances
        baseline_perfs = {}
        for _, row in model_data.iterrows():
            baseline_type = row['Baseline Type']
            baseline_metric = f'Baseline {metric}'
            if baseline_metric in row:
                baseline_perfs[baseline_type] = row[baseline_metric]
        
        # Calculate improvement relative to best (hardest to beat) baseline
        lower_is_better = metric not in ['R²', 'R2']
        
        if baseline_perfs:
            if lower_is_better:
                # For metrics where lower is better, best baseline has lowest value
                best_baseline_type = min(baseline_perfs.keys(), key=lambda k: baseline_perfs[k])
                best_baseline_value = baseline_perfs[best_baseline_type]
                improvement = ((best_baseline_value - model_perf) / best_baseline_value) * 100
            else:
                # For metrics where higher is better, best baseline has highest value
                best_baseline_type = max(baseline_perfs.keys(), key=lambda k: baseline_perfs[k])
                best_baseline_value = baseline_perfs[best_baseline_type]
                # Special handling for R² where baseline is typically 0
                if best_baseline_value == 0 and metric in ['R²', 'R2']:
                    # For R², when baseline is 0, the improvement is essentially the model R² value itself
                    improvement = model_perf * 100  # Show as percentage points above baseline
                else:
                    improvement = ((model_perf - best_baseline_value) / abs(best_baseline_value)) * 100 if best_baseline_value != 0 else 0
        else:
            best_baseline_type = 'Unknown'
            best_baseline_value = 0
            improvement = 0
        
        consolidated_data.append({
            'Model': model,
            'Model Performance': model_perf,
            'Mean Baseline': baseline_perfs.get('Mean', 0),
            'Median Baseline': baseline_perfs.get('Median', 0),
            'Random Baseline': baseline_perfs.get('Random', 0),
            'Best Baseline Type': best_baseline_type,
            'Best Baseline Value': best_baseline_value,
            'Improvement (%)': improvement
        })
    
    # Create DataFrame
    plot_df = pd.DataFrame(consolidated_data)
    
    # Sort by improvement (best first)
    plot_df = plot_df.sort_values('Improvement (%)', ascending=False)
    
    # Extract model type for coloring
    def extract_model_type(model_name):
        if model_name.startswith('XGBoost'):
            return 'XGBoost'
        elif model_name.startswith('LightGBM'):
            return 'LightGBM'
        elif model_name.startswith('CatBoost'):
            return 'CatBoost'
        elif model_name.startswith('ElasticNet'):
            return 'ElasticNet'
        elif model_name.startswith('LR'):
            return 'Linear Regression'
        else:
            return 'Other'
    
    plot_df['Model Type'] = plot_df['Model'].apply(extract_model_type)
    
    # Define colors for each model type
    color_map = {
        'XGBoost': '#0173B2',
        'LightGBM': '#029E73', 
        'CatBoost': '#D55E00',
        'ElasticNet': '#CC78BC',
        'Linear Regression': '#ECE133'
    }
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot parameters
    bar_height = 0.6
    y_positions = np.arange(len(plot_df))
    
    # Plot model metric bars
    model_bars = ax.barh(
        y_positions,
        plot_df['Model Performance'],
        height=bar_height,
        color=[color_map.get(t, '#999999') for t in plot_df['Model Type']],
        alpha=0.8,
        label=f'Model {metric}'
    )
    
    # Define baseline colors and styles
    baseline_styles = {
        'Mean': {'color': '#FF6B6B', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8},
        'Median': {'color': '#4ECDC4', 'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.8},
        'Random': {'color': '#95A5A6', 'linestyle': ':', 'linewidth': 3, 'alpha': 0.8}
    }
    
    # Plot baseline lines for each model
    for i, (_, row) in enumerate(plot_df.iterrows()):
        for baseline_type, style in baseline_styles.items():
            baseline_value = row[f'{baseline_type} Baseline']
            # For R², baseline values can be 0 which is valid
            if metric in ['R²', 'R2'] or baseline_value > 0:
                ax.plot(
                    [baseline_value, baseline_value], 
                    [i - bar_height/2, i + bar_height/2],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    alpha=style['alpha']
                )
    
    # Add improvement percentage text
    for i, (_, row) in enumerate(plot_df.iterrows()):
        # Calculate position for text
        if lower_is_better:
            text_x = max(row['Model Performance'], row['Best Baseline Value']) * 1.02
        else:
            text_x = max(row['Model Performance'], row['Best Baseline Value']) * 1.02
        
        # Add improvement text with best baseline info
        if metric in ['R²', 'R2'] and row['Best Baseline Value'] == 0:
            # For R² with baseline of 0, show the actual R² value as percentage points
            improvement_text = f"R²={row['Model Performance']:.3f} (baseline=0)"
        else:
            improvement_text = f"{row['Improvement (%)']:.1f}% vs {row['Best Baseline Type']}"
        ax.text(
            text_x,
            i,
            improvement_text,
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # Process model names for y-axis labels
    enhanced_names = []
    for name in plot_df['Model']:
        # Find dataset and optimization information
        dataset_info = ""
        opt_info = ""
        
        if "_Base_Random" in name:
            dataset_info = "BaseRand"
        elif "_Yeo_Random" in name:
            dataset_info = "YeoRand"
        elif "_Base" in name:
            dataset_info = "Base"
        elif "_Yeo" in name:
            dataset_info = "Yeo"
        else:
            dataset_info = "Unknown"
        
        # Check for optuna optimization
        if "_optuna" in name:
            opt_info = " (opt)"
        
        # Create enhanced name
        enhanced_name = f"{dataset_info}{opt_info}"
        enhanced_names.append(enhanced_name)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(enhanced_names)
    
    # Create legends
    # Model type legend - create fresh artists for each legend
    model_handles = [plt.Rectangle((0,0), 1, 1, color=color_map[t]) 
                     for t in color_map if t in plot_df['Model Type'].values]
    model_labels = [t for t in color_map.keys() if t in plot_df['Model Type'].values]
    
    # Baseline legend - create fresh artists
    baseline_handles = []
    baseline_labels = []
    for baseline_type, style in baseline_styles.items():
        baseline_handles.append(
            plt.Line2D([0], [0], 
                      color=style['color'], 
                      linestyle=style['linestyle'], 
                      linewidth=style['linewidth'])
        )
        baseline_labels.append(f'{baseline_type} Baseline')
    
    # Create first legend (model types)
    first_legend = ax.legend(
        model_handles, 
        model_labels, 
        loc='lower right', 
        title='Model Type', 
        frameon=True,
        framealpha=0.9
    )
    
    # Add first legend explicitly before creating second
    ax.add_artist(first_legend)
    
    # Create second legend (baseline types) - this will be the primary legend
    ax.legend(
        handles=baseline_handles,
        labels=baseline_labels,
        loc='upper right',
        title='Baseline Types',
        frameon=True,
        framealpha=0.9
    )
    
    # Set plot title and labels
    metric_display = metric.replace('R2', 'R²')
    ax.set_title(f'Model Performance vs All Baselines ({metric_display})', fontsize=16, pad=20)
    
    if lower_is_better:
        ax.set_xlabel(f'{metric_display} (lower is better)', fontsize=12)
    else:
        ax.set_xlabel(f'{metric_display} (higher is better)', fontsize=12)
    
    # Set x-axis limits dynamically
    all_values = []
    all_values.extend(plot_df['Model Performance'].values)
    all_values.extend(plot_df['Mean Baseline'].values)
    all_values.extend(plot_df['Median Baseline'].values)
    all_values.extend(plot_df['Random Baseline'].values)
    
    # For R², include 0 in valid values since baselines are at 0
    if metric in ['R²', 'R2']:
        valid_values = [v for v in all_values]
    else:
        valid_values = [v for v in all_values if v > 0]
    
    if valid_values:
        if lower_is_better:
            x_min = max(0, min(valid_values) * 0.9)
            x_max = max(valid_values) * 1.15
        else:
            # For R² and other "higher is better" metrics
            x_min = min(valid_values) - 0.05  # Add some padding to show 0
            x_max = max(1.0, max(valid_values) * 1.1)
        
        ax.set_xlim(x_min, x_max)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Tight layout and save
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{metric_display}_consolidated_baseline_comparison.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Consolidated baseline comparison saved to {output_path}")
    return output_path


def create_consolidated_baseline_visualizations(
    baseline_data_path=None, 
    output_dir=None, 
    metrics=['RMSE', 'MAE', 'R²'],
    figsize=(16, 10),
    dpi=300
):
    """
    Create consolidated baseline comparison visualizations for all specified metrics.
    
    Parameters
    ----------
    baseline_data_path : str, optional
        Path to the CSV file containing baseline comparison data. 
        If None, uses the default path from settings.
    output_dir : str, optional
        Directory to save visualizations. If None, uses default from settings.
    metrics : list, optional
        List of metrics to visualize (default: ['RMSE', 'MAE', 'R²'])
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of output images (default: 300)
        
    Returns
    -------
    dict
        Dictionary mapping metric names to output paths
    """
    if baseline_data_path is None:
        # Check if adapted CSV exists, otherwise use original
        adapted_path = settings.METRICS_DIR / "baseline_comparison_adapted.csv"
        original_path = settings.METRICS_DIR / "baseline_comparison.csv"
        
        if adapted_path.exists():
            baseline_data_path = adapted_path
        else:
            baseline_data_path = original_path
    
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "baselines"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if baseline data exists
    if not os.path.exists(baseline_data_path):
        print(f"Baseline comparison data not found at {baseline_data_path}")
        print("Run baseline evaluation first.")
        return {}
    
    # Dictionary to store output paths
    output_paths = {}
    
    # Create consolidated plot for each metric
    for metric in metrics:
        try:
            path = create_consolidated_baseline_comparison(
                baseline_data_path,
                output_dir,
                metric=metric,
                figsize=figsize,
                dpi=dpi
            )
            if path:
                output_paths[metric] = path
        except Exception as e:
            print(f"Error creating consolidated plot for {metric}: {e}")
    
    return output_paths


if __name__ == "__main__":
    # Test the consolidated visualization
    baseline_data_path = Path("/mnt/d/ml_project_refactored/outputs/metrics/baseline_comparison.csv")
    output_dir = Path("/mnt/d/ml_project_refactored/outputs/visualizations/baselines")
    
    # Create consolidated visualizations
    paths = create_consolidated_baseline_visualizations(
        baseline_data_path, 
        output_dir,
        metrics=['RMSE', 'MAE', 'R²']
    )
    
    print(f"\nCreated {len(paths)} consolidated baseline visualizations:")
    for metric, path in paths.items():
        print(f"  {metric}: {path}")