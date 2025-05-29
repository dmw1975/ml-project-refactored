"""ElasticNet-specific CV distribution visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pickle

from visualization_new.core.interfaces import VisualizationConfig
from visualization_new.components.formats import save_figure
from visualization_new.utils.io import ensure_dir


def plot_elasticnet_cv_distribution(
    params_file: Optional[Union[str, Path]] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Optional[plt.Figure]:
    """
    Create ElasticNet CV RMSE distribution plot from hyperparameter search results.
    
    This plot shows the distribution of RMSE values across different hyperparameter
    combinations tested during cross-validation grid search.
    
    Args:
        params_file: Path to elasticnet_params.pkl file (if None, uses default location)
        config: Visualization configuration
        
    Returns:
        plt.Figure: Figure with ElasticNet CV distribution plot, or None if data not found
    """
    # Handle configuration
    if config is None:
        config = VisualizationConfig()
    elif isinstance(config, dict):
        config = VisualizationConfig(**config)
    
    # Set default params file path if not provided
    if params_file is None:
        # Import settings
        import sys
        from pathlib import Path
        
        # Add project root to path if needed
        project_root = Path(__file__).parent.parent.parent.absolute()
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            
        # Import settings
        from config import settings
        
        params_file = settings.MODEL_DIR / 'elasticnet_params.pkl'
    
    params_file = Path(params_file)
    
    # Check if file exists
    if not params_file.exists():
        print(f"ElasticNet parameters file not found: {params_file}")
        return None
    
    # Load CV results
    try:
        with open(params_file, 'rb') as f:
            cv_results = pickle.load(f)
        print(f"Loaded ElasticNet CV results from {params_file}")
    except Exception as e:
        print(f"Error loading ElasticNet CV results: {e}")
        return None
    
    # Verify data structure
    if not isinstance(cv_results, list) or len(cv_results) == 0:
        print("CV results are empty or in unexpected format.")
        return None
    
    # Prepare data for the plot
    rmse_data = []
    for result in cv_results:
        dataset = result.get('dataset', 'Unknown')
        cv_df = result.get('cv_results')
        
        if cv_df is None or not hasattr(cv_df, 'iterrows'):
            continue
            
        # Get RMSE for each parameter combination
        for _, row in cv_df.iterrows():
            rmse_data.append({
                'Dataset': dataset,
                'RMSE': row['mean_rmse'],
                'Alpha': row['alpha'],
                'L1_Ratio': row['l1_ratio']
            })
    
    if not rmse_data:
        print("No valid RMSE data found in CV results.")
        return None
    
    rmse_df = pd.DataFrame(rmse_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define color palette
    palette = sns.color_palette("pastel", n_colors=len(rmse_df['Dataset'].unique()))
    
    # Boxplot for RMSE distribution - hide outliers to avoid clutter
    box = sns.boxplot(
        x='Dataset', 
        y='RMSE', 
        data=rmse_df, 
        palette=palette, 
        ax=ax,
        showfliers=False,
        width=0.6
    )
    
    # Stripplot for individual parameter combinations
    strip = sns.stripplot(
        x='Dataset', 
        y='RMSE', 
        data=rmse_df, 
        color='gray', 
        alpha=0.6,
        jitter=True, 
        ax=ax, 
        size=5
    )
    
    # Plot mean as red points
    for i, dataset in enumerate(rmse_df['Dataset'].unique()):
        rmse_vals = rmse_df[rmse_df['Dataset'] == dataset]['RMSE']
        mean = np.mean(rmse_vals)
        
        # Plot mean value
        ax.plot(i, mean, 'o', color='red', markersize=8, zorder=10)
    
    # Add title and labels
    ax.set_title("ElasticNet CV RMSE Distribution by Dataset", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("RMSE (lower is better)", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend
    # For legend creation
    box_patch = plt.Rectangle((0, 0), 1, 1, fc="lightblue", edgecolor="black", linewidth=1)
    
    legend_elements = [
        box_patch,
        Line2D([], [], marker='o', linestyle='None', markerfacecolor='gray', 
              markeredgecolor='gray', markersize=8, alpha=0.6),
        Line2D([], [], marker='o', linestyle='None', markerfacecolor='red', 
              markeredgecolor='red', markersize=8)
    ]
    
    # Legend labels
    legend_labels = [
        'RMSE Distribution (Boxplot)', 
        'Individual Parameter Combination',
        'Mean RMSE'
    ]
    
    # Add legend with all elements
    ax.legend(
        handles=legend_elements, 
        labels=legend_labels, 
        loc='upper right',
        title='ElasticNet Grid Search Results'
    )
    
    # Add text annotation about number of combinations
    n_combinations = len(rmse_df[rmse_df['Dataset'] == rmse_df['Dataset'].unique()[0]])
    ax.text(
        0.02, 0.98, 
        f'Each dataset tested {n_combinations} hyperparameter combinations',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Save figure if requested
    if config.get('save', True):
        output_dir = config.get('output_dir')
        if output_dir is None:
            # Use default output directory
            from config import settings
            output_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
        
        # Ensure directory exists
        ensure_dir(output_dir)
        
        # Save figure
        save_figure(
            fig=fig,
            filename="elasticnet_cv_rmse_distribution",
            output_dir=output_dir,
            dpi=config.get('dpi', 300),
            format=config.get('format', 'png')
        )
    
    return fig