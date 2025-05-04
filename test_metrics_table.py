"""Test script to generate a metrics table with left-aligned model names."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.absolute()
import sys
sys.path.append(str(project_root))

from config import settings
from visualization_new.core.style import setup_visualization_style
from visualization_new.components.formats import save_figure

# Create test metrics data without using adapters
metrics_df = pd.DataFrame([
    {
        'Model': 'XGBoost (LR_Base)',
        'RMSE': 0.2345,
        'MAE': 0.1234,
        'R2': 0.8765,
        'MSE': 0.0550
    },
    {
        'Model': 'CatBoost (LR_Base) with Experimental Settings',
        'RMSE': 0.2123,
        'MAE': 0.1122,
        'R2': 0.8899,
        'MSE': 0.0451
    },
    {
        'Model': 'LightGBM (LR_Base)',
        'RMSE': 0.2267,
        'MAE': 0.1198,
        'R2': 0.8845,
        'MSE': 0.0514
    },
    {
        'Model': 'ElasticNet (LR_Base)',
        'RMSE': 0.2876,
        'MAE': 0.1456,
        'R2': 0.8234,
        'MSE': 0.0827
    }
])

def create_metrics_table(metrics_df):
    """Recreate the metrics table with left-aligned model names."""
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up style
    style = setup_visualization_style()
    
    # Determine metrics to show (excluding 'Model')
    metrics_columns = [col for col in metrics_df.columns if col != 'Model']
    
    # Sort by RMSE (ascending) to put best models first
    table_data = metrics_df.sort_values('RMSE', ascending=True).copy()
    
    # Create figure
    fig_height = max(6, min(20, len(table_data) * 0.3 + 1.5))
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Get colors for each cell
    colors = []
    
    # Initialize with white for standard rows and light gray for header
    header_color = '#f2f2f2'  # Light gray for header
    for i in range(len(table_data)):
        row_colors = ['white'] * len(table_data.columns)
        colors.append(row_colors)
    
    # Highlight best values for each metric
    for i, metric in enumerate(metrics_columns):
        # Get column index
        col_idx = table_data.columns.get_loc(metric)
        
        # Determine best value
        if metric in ['RMSE', 'MAE', 'MSE']:  # Lower is better
            best_idx = table_data[metric].idxmin()
            colors[best_idx][col_idx] = '#d9ead3'  # Light green
        elif metric in ['R2']:  # Higher is better
            best_idx = table_data[metric].idxmax()
            colors[best_idx][col_idx] = '#d9ead3'  # Light green
    
    # Calculate column widths
    col_widths = [0.6]  # Much wider for model names
    metric_width = 0.4 / len(metrics_columns)
    for _ in metrics_columns:
        col_widths.append(metric_width)
    
    # Format values as strings
    cell_text = []
    for _, row in table_data.iterrows():
        row_text = []
        for j, val in enumerate(row):
            if j == 0:  # Model name
                row_text.append(str(val))
            else:  # Metric
                if isinstance(val, (int, float, np.number)):
                    row_text.append(f"{val:.3f}")
                else:
                    row_text.append(str(val))
        cell_text.append(row_text)
    
    # Add a title row
    title_row = ['Model Performance Metrics Summary'] + [''] * (len(table_data.columns) - 1)
    cell_text.insert(0, title_row)
    
    # Add a color row for the title
    title_row_colors = ['#d4e6f1'] * len(table_data.columns)  # Light blue for title
    colors.insert(0, title_row_colors)
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        cellColours=colors,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    # Set cell properties
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Title row
            # Make title larger and bold
            cell.set_text_props(weight='bold', color='black', fontsize=11)
            
            # For the first cell (which contains the title)
            if col == 0:
                # Make title span all columns
                cell.visible_edges = 'open'  # No borders
            else:
                # Hide all other cells in title row
                cell.set_text_props(alpha=0)
                cell.visible_edges = 'open'  # No borders
                
        elif row == 1:  # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(header_color)
            
            # Set column labels (first row is title, second row is header)
            if col == 0:
                cell.get_text().set_text('Model')
                # Also left-align the 'Model' header
                cell.get_text().set_horizontalalignment('left')
            elif col < len(table_data.columns):
                cell.get_text().set_text(table_data.columns[col])
        
        # Left-align text in the Model column (first column)
        if col == 0 and row > 0:
            cell.get_text().set_horizontalalignment('left')
        
        # Add borders except for title row
        if row > 0:
            cell.set_edgecolor('gray')
    
    # Scale table
    table.scale(1.0, 1.05)
    
    # Tight layout
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    
    # Save figure
    file_path = output_dir / 'test_metrics_table_left_aligned.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    print(f"Test metrics table saved to {file_path}")
    
    plt.close(fig)
    
    return file_path

def main():
    """Run test to generate metrics table with left-aligned model names."""
    # Create the metrics table
    create_metrics_table(metrics_df)
    
if __name__ == "__main__":
    main()