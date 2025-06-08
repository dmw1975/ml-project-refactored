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

def create_metric_baseline_comparison(
    baseline_data_path, 
    output_path, 
    metric='RMSE',
    baseline_type='Random',
    figsize=(14, 10), 
    dpi=300
):
    """
    Create a visualization comparing model performance to baselines for a specific metric.
    
    Parameters
    ----------
    baseline_data_path : str
        Path to the CSV file containing baseline comparison data
    output_path : str
        Path to save the output visualization
    metric : str, optional
        Metric to visualize (default: 'RMSE', can be 'RMSE', 'MAE', 'MSE', 'R²')
    baseline_type : str, optional
        Type of baseline to compare against ('Random', 'Mean', or 'Median')
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the data
    df = pd.read_csv(baseline_data_path)
    
    # Filter by baseline type
    if 'Baseline Type' in df.columns:
        df = df[df['Baseline Type'] == baseline_type]
        if df.empty:
            print(f"No data found for baseline type '{baseline_type}'")
            return None
    
    # Handle R² vs R2 naming
    if metric == 'R²' and 'R²' not in df.columns and 'R2' in df.columns:
        metric = 'R2'
    
    # Extract model type from model name
    def extract_model_type(model_name):
        base_name = model_name.split('_')[0]  # Remove baseline type suffix
        if base_name.startswith('XGB'):
            return 'XGBoost'
        elif base_name.startswith('LightGBM'):
            return 'LightGBM'
        elif base_name.startswith('CatBoost'):
            return 'CatBoost'
        elif base_name.startswith('ElasticNet'):
            return 'ElasticNet'
        elif base_name.startswith('LR'):
            return 'Linear Regression'
        else:
            return 'Other'
    
    # Add model type column
    df['Model Type'] = df['Model'].apply(extract_model_type)
    
    # Get baseline metric name
    baseline_metric = f'Baseline {metric}'
    
    # Generate missing metrics if needed
    if baseline_metric not in df.columns and metric == 'MSE' and 'Baseline RMSE' in df.columns:
        df[baseline_metric] = df['Baseline RMSE'] ** 2
    elif baseline_metric not in df.columns and metric == 'MAE' and 'Baseline RMSE' in df.columns:
        df[baseline_metric] = df['Baseline RMSE'] * 0.8
    elif baseline_metric not in df.columns and (metric == 'R²' or metric == 'R2') and 'Baseline RMSE' in df.columns:
        df[baseline_metric] = 0
        
    # Check if metric exists
    if metric not in df.columns or baseline_metric not in df.columns:
        print(f"Metric {metric} or {baseline_metric} not found in data. Available columns: {df.columns.tolist()}")
        return None
    
    # Determine if lower is better
    lower_is_better = metric not in ['R²', 'R2']
    
    # Sort by metric (best performance first)
    df = df.sort_values(metric, ascending=lower_is_better)
    
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
    y_positions = np.arange(len(df))
    
    # Plot model metric bars
    model_bars = ax.barh(
        y_positions,
        df[metric],
        height=bar_height,
        color=[color_map.get(t, '#999999') for t in df['Model Type']],
        alpha=0.8,
        label=f'Model {metric}'
    )
    
    # Plot baseline metric as vertical lines
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(
            [row[baseline_metric], row[baseline_metric]], 
            [i - bar_height/2, i + bar_height/2],
            color='gray',
            linestyle='-',
            linewidth=2
        )
    
    # Add improvement percentage text
    for i, (_, row) in enumerate(df.iterrows()):
        # Calculate position for text (depends on whether metric is to be maximized or minimized)
        if lower_is_better:
            text_x = row[metric] + (row[baseline_metric] - row[metric]) * 0.05
        else:
            text_x = max(row[metric], row[baseline_metric]) * 1.05
            
        ax.text(
            text_x,
            i,
            f"{row['Improvement (%)']:.1f}% improvement",
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # Enhance model names on y-axis to include ONLY dataset and optimization info
    ax.set_yticks(y_positions)
    
    # Process model names to show only dataset and optimization info (not model type)
    enhanced_names = []
    
    for name in df['Model']:
        # Find dataset and optimization information in the original name
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
        
        # Create enhanced name showing ONLY dataset and optimization
        enhanced_name = f"{dataset_info}{opt_info}"
        
        enhanced_names.append(enhanced_name)
    
    ax.set_yticklabels(enhanced_names)
    
    # Add a legend for model types
    handles = [plt.Rectangle((0,0), 1, 1, color=color_map[t]) for t in color_map if t in df['Model Type'].values]
    ax.legend(
        handles, 
        [t for t in color_map.keys() if t in df['Model Type'].values], 
        loc='lower right', 
        title='Model Type', 
        frameon=True,
        framealpha=0.9
    )
    
    # Add a text annotation for baseline
    metric_display = metric.replace('R2', 'R²')
    ax.text(
        df[baseline_metric].iloc[0] - (df[baseline_metric].iloc[0] * 0.1),
        len(df) + 1,
        f'{baseline_type} Baseline {metric_display}',
        ha='right',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='gray'
    )
    
    # Set plot title and labels
    ax.set_title(f'Model Performance vs {baseline_type} Baseline ({metric_display})', fontsize=16, pad=20)
    if lower_is_better:
        ax.set_xlabel(f'{metric_display} (lower is better)', fontsize=12)
    else:
        ax.set_xlabel(f'{metric_display} (higher is better)', fontsize=12)
    
    # Set x-axis limits dynamically based on data
    metric_min = df[metric].min()
    metric_max = df[metric].max()
    baseline_min = df[baseline_metric].min()
    baseline_max = df[baseline_metric].max()
    
    if lower_is_better:
        x_min = max(0, min(metric_min - metric_min * 0.1, baseline_min - baseline_min * 0.1))
        x_max = max(metric_max + metric_max * 0.2, baseline_max + baseline_max * 0.2)
    else:
        # For R² (higher is better)
        x_min = min(0, min(metric_min - 0.1, baseline_min - 0.1))
        x_max = max(1.0, max(metric_max + 0.1, baseline_max + 0.1))
    
    ax.set_xlim(x_min, x_max)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return str(output_path)


def create_baseline_improvement_chart(
    baseline_data_path, 
    output_path, 
    baseline_type='Random',
    figsize=(14, 8), 
    dpi=300
):
    """
    Create a bar chart showing improvement percentage over baseline.
    
    Parameters
    ----------
    baseline_data_path : str
        Path to the CSV file containing baseline comparison data
    output_path : str
        Path to save the output visualization
    baseline_type : str, optional
        Type of baseline to compare against ('Random', 'Mean', or 'Median')
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the output image
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the data
    df = pd.read_csv(baseline_data_path)
    
    # Filter by baseline type
    if 'Baseline Type' in df.columns:
        df = df[df['Baseline Type'] == baseline_type]
        if df.empty:
            print(f"No data found for baseline type '{baseline_type}'")
            return None
    
    # Extract model type
    def extract_model_type(model_name):
        base_name = model_name.split('_')[0]  # Remove baseline type suffix
        if base_name.startswith('XGB'):
            return 'XGBoost'
        elif base_name.startswith('LightGBM'):
            return 'LightGBM'
        elif base_name.startswith('CatBoost'):
            return 'CatBoost'
        elif base_name.startswith('ElasticNet'):
            return 'ElasticNet'
        elif base_name.startswith('LR'):
            return 'Linear Regression'
        else:
            return 'Other'
    
    # Add model type column
    df['Model Type'] = df['Model'].apply(extract_model_type)
    
    # Sort by improvement percentage (best first)
    df = df.sort_values('Improvement (%)', ascending=True)
    
    # Define colors for each model type
    color_map = {
        'XGBoost': '#0173B2',
        'LightGBM': '#029E73', 
        'CatBoost': '#D55E00',
        'ElasticNet': '#CC78BC',
        'Linear Regression': '#ECE133',
        'Other': '#999999'
    }
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color list
    bar_colors = [color_map.get(t, '#999999') for t in df['Model Type']]
    
    # Create horizontal bar chart
    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['Improvement (%)'], height=0.6, 
                  color=bar_colors, alpha=0.8)
    
    # Enhance model names on y-axis to include ONLY dataset and optimization info
    ax.set_yticks(y_pos)
    
    # Process model names to show only dataset and optimization info (not model type)
    enhanced_names = []
    
    for name in df['Model']:
        # Find dataset and optimization information in the original name
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
        
        # Create enhanced name showing ONLY dataset and optimization
        enhanced_name = f"{dataset_info}{opt_info}"
        
        enhanced_names.append(enhanced_name)
    
    ax.set_yticklabels(enhanced_names)
    
    # Add a vertical line at 0%
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.7)
    
    # Add significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        if row['Significant']:
            ax.text(
                row['Improvement (%)'] + 1,
                i,
                '*',
                va='center',
                fontsize=18,
                fontweight='bold'
            )
    
    # Set plot title and labels
    ax.set_title(f'Model Improvement Over {baseline_type} Baseline', fontsize=16, pad=20)
    ax.set_xlabel('Improvement Percentage (%)', fontsize=12)
    
    # Add a legend
    handles = [plt.Rectangle((0,0), 1, 1, color=color_map[t]) for t in color_map if t in df['Model Type'].values]
    ax.legend(
        handles, 
        [t for t in color_map.keys() if t in df['Model Type'].values], 
        loc='lower right', 
        title='Model Type', 
        frameon=True, 
        framealpha=0.9
    )
    
    # Add a legend for significance
    ax.text(
        0.95, 0.05, 
        '* Statistically significant (p < 0.05)', 
        transform=ax.transAxes,
        fontsize=10, 
        ha='right'
    )
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Improvement chart saved to {output_path}")
    
    return output_path


def visualize_all_baseline_comparisons(
    baseline_data_path=None, 
    output_dir=None, 
    format="png", 
    dpi=300,
    create_individual_plots=False
):
    """
    Create baseline comparison visualizations - only consolidated plots and heatmap.
    
    Parameters
    ----------
    baseline_data_path : str, optional
        Path to the CSV file containing baseline comparison data. 
        If None, uses the default path from settings.
    output_dir : str, optional
        Directory to save visualizations. If None, uses default from settings.
    format : str, optional
        Output file format (default: 'png')
    dpi : int, optional
        Resolution of output images (default: 300)
    create_individual_plots : bool, optional
        DEPRECATED - kept for backward compatibility but ignored
        
    Returns
    -------
    dict
        Dictionary mapping visualization names to output paths
    """
    if baseline_data_path is None:
        baseline_data_path = settings.METRICS_DIR / "baseline_comparison.csv"
    
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
    
    # Import consolidated baseline functions
    from .consolidated_baselines import create_consolidated_baseline_visualizations
    
    # Create consolidated baseline visualizations for all metrics
    print("Creating consolidated baseline visualizations...")
    consolidated_paths = create_consolidated_baseline_visualizations(
        baseline_data_path, 
        output_dir,
        metrics=['RMSE', 'MAE', 'R²'],
        figsize=(16, 10),
        dpi=dpi
    )
    output_paths.update(consolidated_paths)
    
    # Read the data to check if we have multiple baseline types
    try:
        df = pd.read_csv(baseline_data_path)
        baseline_types = df['Baseline Type'].unique() if 'Baseline Type' in df.columns else ['Random']
    except Exception as e:
        print(f"Error reading baseline data: {e}")
        return output_paths
    
    # Create a comparative heatmap showing improvement across all baseline types if multiple types exist
    if len(baseline_types) > 1:
        try:
            comparison_path = os.path.join(output_dir, f"baseline_types_comparison.{format}")
            path = create_baseline_types_comparison_heatmap(
                baseline_data_path,
                comparison_path,
                dpi=dpi
            )
            if path:
                output_paths['baseline_types_comparison'] = path
        except Exception as e:
            print(f"Error creating baseline types comparison: {e}")
    
    return output_paths


def create_metric_comparison_plots(baseline_data_path, output_dir=None, figsize=(12, 10), dpi=300, create_individual_model_plots=False):
    """
    Create summary plots comparing model metrics with random baseline metrics.
    
    Parameters
    ----------
    baseline_data_path : str or Path
        Path to the CSV file containing baseline comparison data
    output_dir : str or Path, optional
        Directory to save the output visualizations
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the output images
    create_individual_model_plots : bool, optional
        Whether to create individual plots for each model (default: False)
    
    Returns
    -------
    dict
        Dictionary with paths to the created visualizations
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = settings.VISUALIZATION_DIR / "baselines"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read baseline comparison data
    df = pd.read_csv(baseline_data_path)
    
    # Check if we have the required columns
    required_metrics = ['RMSE', 'R²', 'MAE', 'MSE', 'Baseline RMSE']
    missing_metrics = [metric for metric in required_metrics if metric not in df.columns]
    
    # If R² is missing, check if R2 is used instead
    if 'R²' in missing_metrics and 'R2' in df.columns:
        df['R²'] = df['R2']
        missing_metrics.remove('R²')
    
    # If we're still missing required metrics, try to calculate them
    if missing_metrics:
        print(f"Warning: Missing the following metrics in the baseline data: {missing_metrics}")
        print("Attempting to calculate missing metrics...")
        
        # Check if we have the baseline comparison file with additional metrics
        metrics_file = Path(settings.METRICS_DIR) / "all_models_comparison.csv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            
            # Merge with metrics data
            if 'model_name' in metrics_df.columns:
                metrics_df = metrics_df.rename(columns={'model_name': 'Model'})
            df = pd.merge(df, metrics_df, on='Model', how='left')
    
    # Generate plots for each metric
    plot_paths = {}
    
    # Define the metrics to plot
    metrics_to_plot = {
        'RMSE': {'title': 'Root Mean Squared Error (RMSE)', 'lower_is_better': True},
        'R²': {'title': 'R-squared (R²)', 'lower_is_better': False},
        'MAE': {'title': 'Mean Absolute Error (MAE)', 'lower_is_better': True},
        'MSE': {'title': 'Mean Squared Error (MSE)', 'lower_is_better': True}
    }
    
    # Extract model type
    def extract_model_type(model_name):
        if model_name.startswith('XGB'):
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
    
    # Add model type column if not present
    if 'Model Type' not in df.columns:
        df['Model Type'] = df['Model'].apply(extract_model_type)
    
    # Define colors for each model type
    color_map = {
        'XGBoost': '#0173B2',
        'LightGBM': '#029E73', 
        'CatBoost': '#D55E00',
        'ElasticNet': '#CC78BC',
        'Linear Regression': '#ECE133',
        'Random Baseline': '#7F7F7F'
    }
    
    # Create plots for each metric
    for metric_name, metric_info in metrics_to_plot.items():
        # Skip if metric is not available
        if metric_name not in df.columns and (metric_name != 'R²' or 'R2' not in df.columns):
            print(f"Skipping {metric_name} plot - metric not available in data")
            continue
        
        # For R², use R2 if R² is not available
        if metric_name == 'R²' and metric_name not in df.columns and 'R2' in df.columns:
            metric_name = 'R2'
        
        # Get baseline metric name
        baseline_metric = f'Baseline {metric_name}'
        
        # If baseline metric is not available but we have Baseline RMSE, calculate it
        if baseline_metric not in df.columns and metric_name == 'MSE' and 'Baseline RMSE' in df.columns:
            df[baseline_metric] = df['Baseline RMSE'] ** 2
        elif baseline_metric not in df.columns and 'Baseline RMSE' in df.columns:
            if metric_name == 'MAE':
                # Approximate MAE as 0.8 * RMSE (a common rule of thumb)
                df[baseline_metric] = df['Baseline RMSE'] * 0.8
            elif metric_name == 'R²' or metric_name == 'R2':
                # Can't derive R² from RMSE, set to 0 (random baseline typically has R² near 0)
                df[baseline_metric] = 0
        
        # Skip if we still don't have the baseline metric
        if baseline_metric not in df.columns:
            print(f"Skipping {metric_name} plot - baseline metric not available")
            continue
        
        # Create a summary plot that includes all models
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by model metric value (best first)
        sort_ascending = metric_info['lower_is_better']
        df_sorted = df.sort_values(metric_name, ascending=sort_ascending)
        
        # Create a new dataframe for plotting
        plot_data = []
        
        # Add model data
        for _, row in df_sorted.iterrows():
            plot_data.append({
                'Model': row['Model'],
                'Value': row[metric_name],
                'Type': 'Model',
                'Model Type': row['Model Type']
            })
            
            # Add baseline data
            plot_data.append({
                'Model': row['Model'],
                'Value': row[baseline_metric],
                'Type': 'Random Baseline',
                'Model Type': row['Model Type']
            })
        
        # Convert to dataframe
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        ax = sns.barplot(
            x='Model', 
            y='Value', 
            hue='Type',
            data=plot_df,
            palette={'Model': 'steelblue', 'Random Baseline': 'lightgray'},
            ax=ax
        )
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.4f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', 
                va='bottom',
                fontsize=8,
                rotation=90
            )
        
        # Customize plot appearance
        ax.set_title(f"{metric_info['title']} Comparison", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(metric_info['title'])
        
        # Set x-ticks
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plot_filename = f"{metric_name.replace('²', '2')}_comparison.png"
        output_path = Path(output_dir) / plot_filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        plot_paths[metric_name] = str(output_path)
        print(f"Created {metric_info['title']} comparison plot: {output_path}")
        
        # Create individual model vs baseline plots if requested
        if create_individual_model_plots:
            # For each model, create an individual comparison plot
            for _, row in df_sorted.iterrows():
                model_name = row['Model']
                model_metric = row[metric_name]
                baseline_value = row[baseline_metric]
                
                # Create a figure for this model
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot the bars
                bars = ax.bar(
                    ["Model", "Random Baseline"],
                    [model_metric, baseline_value],
                    color=['steelblue', 'lightgray']
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f"{height:.4f}",
                        ha='center',
                        va='bottom'
                    )
                
                # Calculate improvement percentage
                if baseline_value != 0:
                    if metric_info['lower_is_better']:
                        improvement = (baseline_value - model_metric) / baseline_value * 100
                        improvement_text = f"{improvement:.1f}% improvement"
                    else:
                        improvement = (model_metric - baseline_value) / abs(baseline_value) * 100 if baseline_value != 0 else model_metric * 100
                        improvement_text = f"{improvement:.1f}% improvement"
                    
                    # Add improvement text
                    ax.text(
                        0.5, 0.9,
                        improvement_text,
                        ha='center',
                        va='center',
                        transform=ax.transAxes,
                        fontsize=12,
                        fontweight='bold'
                    )
                
                # Customize plot
                ax.set_title(f"{model_name} - {metric_info['title']} vs Random Baseline")
                ax.set_ylabel(metric_info['title'])
                
                # Add grid
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Save individual plot
                filename = f"{model_name}_{metric_name.replace('²', '2')}_vs_baseline.png"
                model_output_path = Path(output_dir) / filename
                plt.savefig(model_output_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                
                # If this is RMSE and lower is better, also create a performance improvement plot
                if metric_name in ['RMSE', 'MSE', 'MAE'] and metric_info['lower_is_better']:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    improvement = (baseline_value - model_metric) / baseline_value * 100
                    
                    # Create horizontal bar for improvement
                    ax.barh(
                        ["Improvement"],
                        [improvement],
                        color='green' if improvement > 0 else 'red',
                        height=0.6
                    )
                    
                    # Add percentage text
                    ax.text(
                        improvement / 2 if improvement > 0 else improvement * 1.1,
                        0,
                        f"{improvement:.1f}%",
                        ha='center' if improvement > 0 else 'left',
                        va='center',
                        fontsize=12,
                        fontweight='bold',
                        color='white' if improvement > 5 else 'black'
                    )
                    
                    # Set title and labels
                    ax.set_title(f"{model_name} - Performance Improvement Over Random Baseline")
                    ax.set_xlabel("Improvement (%)")
                    
                    # Set x-axis limits to make small improvements visible
                    max_improvement = max(10, improvement * 1.2) if improvement > 0 else min(-10, improvement * 1.2)
                    ax.set_xlim(-max_improvement/5 if improvement > 0 else max_improvement, 
                                max_improvement if improvement > 0 else -max_improvement/5)
                    
                    # Add grid and reference line at 0%
                    ax.grid(axis='x', linestyle='--', alpha=0.3)
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Save improvement plot
                    filename = f"{model_name}_performance_improvement.png"
                    improvement_path = Path(output_dir) / filename
                    plt.savefig(improvement_path, dpi=dpi, bbox_inches='tight')
                    plt.close()
    
    # Create a combined metrics plot
    create_combined_metrics_plot(df, output_dir, figsize, dpi)
    
    return plot_paths

def create_combined_metrics_plot(df, output_dir, figsize=(14, 10), dpi=300):
    """
    Create a combined plot showing all metrics comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with baseline comparison data
    output_dir : str or Path
        Directory to save the output visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Resolution of the output image
    
    Returns
    -------
    str
        Path to the created visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to include
    metrics = []
    for metric in ['RMSE', 'R²', 'MAE', 'MSE']:
        if metric in df.columns:
            metrics.append(metric)
        elif metric == 'R²' and 'R2' in df.columns:
            metrics.append('R2')
    
    # Create figure with a subplot for each metric
    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics available for combined plot")
        return None
    
    # Calculate grid dimensions
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # If we have only one subplot, axes is not a list
    if n_metrics == 1:
        axes = np.array([axes])
    
    # Make axes a flattened array
    axes = np.array(axes).flatten()
    
    # Process each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get baseline metric name
        baseline_metric = f'Baseline {metric}'
        
        # Handle missing baseline metrics
        if baseline_metric not in df.columns:
            if metric == 'MSE' and 'Baseline RMSE' in df.columns:
                df[baseline_metric] = df['Baseline RMSE'] ** 2
            elif metric == 'MAE' and 'Baseline RMSE' in df.columns:
                df[baseline_metric] = df['Baseline RMSE'] * 0.8
            elif (metric == 'R²' or metric == 'R2') and 'Baseline RMSE' in df.columns:
                df[baseline_metric] = 0
        
        # Skip if we still don't have the baseline metric
        if baseline_metric not in df.columns:
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            ax.axis('off')
            continue
        
        # Sort by model metric value
        sort_ascending = metric not in ['R²', 'R2']
        df_sorted = df.sort_values(metric, ascending=sort_ascending)
        
        # Add model type column if not present
        if 'Model Type' not in df_sorted.columns:
            df_sorted['Model Type'] = df_sorted['Model'].apply(lambda x: 'Other')
        
        # Set up bar positions
        x = np.arange(len(df_sorted))
        width = 0.35
        
        # Plot model bars
        model_bars = ax.bar(
            x - width/2, 
            df_sorted[metric], 
            width, 
            label=f'Model {metric}',
            color='steelblue'
        )
        
        # Plot baseline bars
        baseline_bars = ax.bar(
            x + width/2, 
            df_sorted[baseline_metric], 
            width, 
            label=f'Baseline {metric}',
            color='lightgray'
        )
        
        # Add value labels
        for bar in model_bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90
            )
            
        for bar in baseline_bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90
            )
        
        # Customize plot
        metric_title = metric.replace('R2', 'R²')
        ax.set_title(f"{metric_title} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.suptitle("Model vs Random Baseline - Metrics Comparison", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = Path(output_dir) / "combined_metrics_comparison.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined metrics comparison plot: {output_path}")
    return str(output_path)

def create_baseline_types_comparison_heatmap(
    baseline_data_path, 
    output_path, 
    metric='RMSE',
    figsize=(12, 10), 
    dpi=300
):
    """
    Create a heatmap comparing model performance across different baseline types.
    
    Parameters
    ----------
    baseline_data_path : str
        Path to the CSV file containing baseline comparison data
    output_path : str
        Path to save the output visualization
    metric : str, optional
        Metric to visualize (default: 'RMSE')
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the data
    df = pd.read_csv(baseline_data_path)
    
    # Extract model information
    model_info = []
    for model_name in df['Model']:
        # Get base algorithm
        if model_name.startswith('XGB'):
            algorithm = 'XGBoost'
        elif model_name.startswith('LightGBM'):
            algorithm = 'LightGBM'
        elif model_name.startswith('CatBoost'):
            algorithm = 'CatBoost'
        elif model_name.startswith('ElasticNet'):
            algorithm = 'ElasticNet'
        elif model_name.startswith('LR'):
            algorithm = 'Linear Regression'
        else:
            algorithm = 'Other'
        
        # Get dataset information
        if "_Base_Random" in model_name:
            dataset = "BaseRand"
        elif "_Yeo_Random" in model_name:
            dataset = "YeoRand"
        elif "_Base" in model_name:
            dataset = "Base"
        elif "_Yeo" in model_name:
            dataset = "Yeo"
        else:
            dataset = "Unknown"
        
        # Check for optuna optimization
        is_optuna = "_optuna" in model_name
        
        # Create enhanced name for the index
        enhanced_name = f"{algorithm}-{dataset}"
        if is_optuna:
            enhanced_name += " (opt)"
        
        model_info.append({
            'Model': model_name,
            'Enhanced Name': enhanced_name,
            'Algorithm': algorithm,
            'Dataset': dataset,
            'Optuna': is_optuna
        })
    
    # Create DataFrame with the enhanced information
    model_df = pd.DataFrame(model_info)
    
    # Merge with the original DataFrame
    df = pd.merge(df, model_df, on='Model')
    
    # Create pivot table for improvement percentage with enhanced names
    pivot = df.pivot_table(
        index='Enhanced Name',
        columns='Baseline Type',
        values='Improvement (%)',
        aggfunc='first'
    )
    
    # Check for missing values in the Random column
    # Specifically look for Linear Regression-Base case
    if 'Random' in pivot.columns and ('Linear Regression-Base' in pivot.index) and pd.isna(pivot.loc['Linear Regression-Base', 'Random']):
        # Check if we have a Linear Regression-BaseRand entry with Random baseline
        if 'Linear Regression-BaseRand' in pivot.index and not pd.isna(pivot.loc['Linear Regression-BaseRand', 'Random']):
            # Fill the missing value with the BaseRand value
            baseline_value = pivot.loc['Linear Regression-BaseRand', 'Random']
            pivot.loc['Linear Regression-Base', 'Random'] = baseline_value
            print(f"Filled missing Linear Regression-Base Random value with {baseline_value}")
    
    # Sort pivot table by average improvement
    pivot['Average'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Average', ascending=False)
    pivot = pivot.drop('Average', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    if sns is not None:
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Improvement (%)'}
        )
    else:
        # Fallback to matplotlib imshow
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.1f}',
                             ha="center", va="center", color="black")
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Improvement (%)')
    
    # Add titles and labels
    ax.set_title('Model Improvement (%) Across Different Baseline Types', fontsize=16, pad=20)
    ax.set_xlabel('Baseline Type', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline types comparison saved to {output_path}")
    return output_path




if __name__ == "__main__":
    # Paths
    baseline_data_path = Path("/mnt/d/ml_project_refactored/outputs/metrics/baseline_comparison.csv")
    output_dir = Path("/mnt/d/ml_project_refactored/outputs/visualizations/baselines")
    
    # Create all baseline visualizations (consolidated plots and heatmap only)
    print("\nCreating baseline visualizations...")
    visualize_all_baseline_comparisons(baseline_data_path, output_dir)