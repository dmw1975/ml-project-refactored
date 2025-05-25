"""Test script to generate improved performance visualizations."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization_new.components.formats import save_figure
from utils import io

def improve_metrics_summary_table():
    """Create an improved metrics summary table with better column sizing."""
    # Load metrics data
    metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
    if not metrics_file.exists():
        print(f"Error: Metrics file not found at {metrics_file}")
        return
    
    metrics_df = pd.read_csv(metrics_file)
    
    # Make sure we have the right model name column
    if 'model_name' in metrics_df.columns:
        # Create better model names
        display_names = []
        for name in metrics_df['model_name']:
            # Extract dataset information first
            if 'Base_Random' in name:
                dataset = 'Base R'
            elif 'Yeo_Random' in name:
                dataset = 'Yeo R'
            elif 'Base' in name:
                dataset = 'Base'
            elif 'Yeo' in name:
                dataset = 'Yeo'
            else:
                dataset = ''
                
            # Now identify model type with proper prefix
            if 'XGB' in name and 'optuna' in name:
                model_prefix = 'XGB Opt'
            elif 'XGB' in name:
                model_prefix = 'XGB Base'
            elif 'LightGBM' in name and 'optuna' in name:
                model_prefix = 'LGBM Opt'
            elif 'LightGBM' in name:
                model_prefix = 'LGBM Base'
            elif 'CatBoost' in name and 'optuna' in name:
                model_prefix = 'CB Opt'
            elif 'CatBoost' in name:
                model_prefix = 'CB Base'
            elif 'ElasticNet' in name:
                model_prefix = 'EN'  # ElasticNet
            elif name.startswith('LR_'):
                model_prefix = 'LR'  # Linear Regression
            else:
                model_prefix = name.split('_')[0]
                
            # Combine with proper spacing    
            display_name = f"{model_prefix} {dataset}"
            display_names.append(display_name)
            
        # Replace model names
        metrics_df['Model'] = display_names
    elif 'index' in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'index': 'Model'})
    
    # Select columns for the table
    table_columns = ['Model', 'RMSE', 'MAE', 'R²', 'MSE']
    
    # Fix R² column name if needed
    if 'R2' in metrics_df.columns and 'R²' not in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={'R2': 'R²'})
    
    # Filter columns that exist in the DataFrame
    available_columns = [col for col in table_columns if col in metrics_df.columns]
    table_data = metrics_df[available_columns].copy()
    
    # Create figure with improved sizing
    fig = plt.figure(figsize=(10, len(table_data) * 0.5 + 1))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Initialize cell colors
    colors = [['white' for _ in range(len(table_data.columns))] for _ in range(len(table_data))]
    
    # Highlight best values
    best_indices = {}
    for col in table_data.columns:
        if col == 'Model':
            continue
        
        if table_data[col].dtype in [np.float64, np.int64, float, int]:
            if col in ['MSE', 'MAE', 'RMSE']:
                best_indices[col] = table_data[col].idxmin()
            elif col in ['R²']:
                best_indices[col] = table_data[col].idxmax()
    
    # Apply highlighting colors
    for col, idx in best_indices.items():
        col_idx = list(table_data.columns).index(col)
        colors[table_data.index.get_loc(idx)][col_idx] = '#d9ead3'  # Light green
    
    # Format values as strings
    cell_text = []
    for row in table_data.values:
        row_text = []
        for i, val in enumerate(row):
            if i == 0:  # Model name
                row_text.append(str(val))
            else:  # Numeric value
                if isinstance(val, (int, float, np.number)):
                    row_text.append(f"{val:.4f}")
                else:
                    row_text.append(str(val))
        cell_text.append(row_text)
    
    # Create table with improved column widths
    table = plt.table(
        cellText=cell_text,
        colLabels=table_data.columns,
        cellColours=colors,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.15, 0.15, 0.15, 0.15]  # Wider first column, narrower metric columns
    )
    
    # Improve table styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add column header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472c4')  # Blue header background
    
    # Set title
    plt.title('Model Performance Metrics Summary', fontsize=14, pad=20)
    
    # Save the improved visualization to the new comparison directory
    output_dir = settings.VISUALIZATION_DIR / "performance/comparison"
    io.ensure_dir(output_dir)
    save_figure(fig, "metrics_summary_table", output_dir, dpi=300)
    
    print(f"Improved metrics summary table saved to {output_dir}")
    return fig

def improve_model_comparison_rmse():
    """Create an improved model comparison RMSE visualization with better readability."""
    # Load metrics data
    metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
    if not metrics_file.exists():
        print(f"Error: Metrics file not found at {metrics_file}")
        return
    
    metrics_df = pd.read_csv(metrics_file)
    
    # Sort models by RMSE ascending
    sorted_df = metrics_df.sort_values('RMSE', ascending=True)
    
    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create cleaner model names for display
    display_names = []
    for name in sorted_df['model_name']:
        # Extract dataset information first
        if 'Base_Random' in name:
            dataset = 'Base R'
        elif 'Yeo_Random' in name:
            dataset = 'Yeo R'
        elif 'Base' in name:
            dataset = 'Base'
        elif 'Yeo' in name:
            dataset = 'Yeo'
        else:
            dataset = ''
            
        # Now identify model type with proper prefix
        if 'XGB' in name and 'optuna' in name:
            model_prefix = 'XGB Opt'
        elif 'XGB' in name:
            model_prefix = 'XGB Base'
        elif 'LightGBM' in name and 'optuna' in name:
            model_prefix = 'LGBM Opt'
        elif 'LightGBM' in name:
            model_prefix = 'LGBM Base'
        elif 'CatBoost' in name and 'optuna' in name:
            model_prefix = 'CB Opt'
        elif 'CatBoost' in name:
            model_prefix = 'CB Base'
        elif 'ElasticNet' in name:
            model_prefix = 'EN'  # ElasticNet
        elif name.startswith('LR_'):
            model_prefix = 'LR'  # Linear Regression
        else:
            model_prefix = name.split('_')[0]
            
        # Combine with proper spacing    
        display_name = f"{model_prefix} {dataset}"
        display_names.append(display_name)
    
    # Plot with improved colors
    bars = ax.bar(display_names, sorted_df['RMSE'], color='#5b9bd5', alpha=0.8)
    
    # Add value labels with better positioning - reduced to 3 decimal places
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.001,  # Small offset
            f'{height:.3f}',  # Changed from 4 to 3 decimal places
            ha='center',
            va='bottom',
            fontsize=10,
            rotation=0
        )
    
    # Title and labels with better sizing
    ax.set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=16, pad=15)
    ax.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    
    # Improve x-tick readability
    plt.xticks(fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Add alternating background for better readability
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
    
    # Tighten y-axis to focus
    min_rmse = sorted_df['RMSE'].min() * 0.95
    max_rmse = sorted_df['RMSE'].max() * 1.05
    ax.set_ylim(min_rmse, max_rmse)
    
    # Format y-axis to 2 decimals
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # Highlight best model
    min_val = sorted_df['RMSE'].min()
    for i, val in enumerate(sorted_df['RMSE']):
        if val == min_val:
            bars[i].set_color('#70ad47')  # Green for best model
            # Add star annotation
            ax.text(
                bars[i].get_x() + bars[i].get_width()/2,
                val + 0.01,
                '★',
                ha='center',
                va='bottom',
                fontsize=16,
                color='red'
            )
    
    # Add legend for star
    ax.text(
        0.98, 0.98, 
        '★ Best performing model',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f2f2f2", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save the improved visualization to the new comparison directory
    output_dir = settings.VISUALIZATION_DIR / "performance/comparison"
    io.ensure_dir(output_dir)
    save_figure(fig, "thesis_model_comparison_rmse", output_dir, dpi=300)
    
    print(f"Improved model comparison RMSE chart saved to {output_dir}")
    return fig

if __name__ == "__main__":
    print("Generating improved visualizations...")
    improve_metrics_summary_table()
    improve_model_comparison_rmse()
    print("Done!")