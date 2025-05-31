"""Performance metrics plots for all model types."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from src.visualization.core.interfaces import ModelData, VisualizationConfig
from src.visualization.core.base import ModelViz, ComparativeViz
from src.visualization.core.registry import get_adapter_for_model
from src.visualization.components.annotations import add_value_labels
from src.visualization.components.layouts import create_grid_layout, create_comparison_layout
from src.visualization.components.formats import format_figure_for_export, save_figure


class MetricsTable(ComparativeViz):
    """Metrics summary table for multiple models."""
    
    def __init__(
        self, 
        models: List[Union[ModelData, Dict[str, Any]]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize metrics table.
        
        Args:
            models: List of model data or adapters
            config: Visualization configuration
        """
        # Convert model data to adapters if needed
        model_adapters = []
        for model_data in models:
            if not isinstance(model_data, ModelData):
                model_adapters.append(get_adapter_for_model(model_data))
            else:
                model_adapters.append(model_data)
            
        # Call parent constructor
        super().__init__(model_adapters, config)
        
    def plot(self) -> plt.Figure:
        """
        Create metrics summary table.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Close any existing figures to avoid conflicts
        plt.close('all')
        # Get metrics for each model
        metrics_data = []
        
        for model in self.models:
            # Get model metadata
            metadata = model.get_metadata()
            model_name = metadata.get('model_name', 'Unknown Model')
            
            # Get metrics
            metrics = model.get_metrics()
            
            # Add model name and metrics
            model_metrics = {'Model': model_name}
            model_metrics.update(metrics)
            
            # Add to data
            metrics_data.append(model_metrics)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Determine metrics to show
        metrics_to_show = self.config.get('metrics', ['RMSE', 'MAE', 'R2', 'MSE'])
        
        # Filter metrics
        available_metrics = [col for col in metrics_to_show if col in metrics_df.columns]
        
        # Create table
        table_data = metrics_df[['Model'] + available_metrics].copy()
        
        # Create figure with better proportions - calculate proper height based on number of rows
        # Ensure adequate height for all rows - increase factor for many models
        # With 36 models + header + title, we need about 0.5 units per row
        fig_height = max(10, len(table_data) * 0.5 + 3)  # Increased height factor for 36+ models
        fig_width = 14  # Fixed reasonable width
        
        # Create figure with minimal margins
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
        metric_indices = []
        
        for i, metric in enumerate(available_metrics):
            # Get column index
            col_idx = table_data.columns.get_loc(metric)
            metric_indices.append(col_idx)
            
            # Determine best value
            if metric in ['RMSE', 'MAE', 'MSE']:  # Lower is better
                best_idx = table_data[metric].idxmin()
                colors[best_idx][col_idx] = '#d9ead3'  # Light green
            elif metric in ['R2']:  # Higher is better
                best_idx = table_data[metric].idxmax()
                colors[best_idx][col_idx] = '#d9ead3'  # Light green
                
            # Also handle R² if present (legacy format)
            elif metric == 'R²':  # Higher is better
                best_idx = table_data[metric].idxmax()
                colors[best_idx][col_idx] = '#d9ead3'  # Light green
        
        # Format values as strings with appropriate precision
        cell_text = []
        
        # Calculate ideal column widths - give model names much more space
        model_name_width = max(len(str(name)) for name in table_data['Model'])
        
        # The model column gets 60% of the width, metrics share the remaining 40%
        col_widths = [0.6]  # Much wider for model names
        
        # Calculate width for metric columns - smaller and equal width
        metric_width = 0.4 / len(available_metrics) 
        
        for i, row in table_data.iterrows():
            row_text = []
            
            for j, val in enumerate(row):
                if j == 0:  # Model name
                    # Don't truncate model names since we're giving them more space
                    row_text.append(str(val))
                else:  # Metric
                    if isinstance(val, (int, float, np.number)):
                        # Use fewer decimal places to save space
                        row_text.append(f"{val:.3f}")
                    else:
                        row_text.append(str(val))
            
            cell_text.append(row_text)
            
            # Add column widths for metric columns (all equal)
            if i == 0:
                # Use the calculated metric width for all metric columns
                for j in range(len(row) - 1):
                    col_widths.append(metric_width)
        
        # Add a title row above the table - by adding a temporary row to cell_text
        title_row = ['Model Performance Metrics Summary'] + [''] * (len(table_data.columns) - 1)
        cell_text.insert(0, title_row)
        
        # Add a color row for the title (title gets a different background)
        title_row_colors = ['#d4e6f1'] * len(table_data.columns)  # Light blue for title
        colors.insert(0, title_row_colors)
        
        # Create table with improved formatting - without using colLabels (we'll set them manually)
        table = ax.table(
            cellText=cell_text,
            cellColours=colors,
            cellLoc='center',  # Default cell alignment
            loc='center',
            colWidths=col_widths
        )
        
        # Format table with better appearance
        table.auto_set_font_size(False)
        table.set_fontsize(9)  # Slightly smaller font for better readability
        
        # Set cell properties - title and header rows
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
        
        # Scale table to be more compact and fill the figure appropriately
        # Use appropriate scale to fit all rows - increased for better spacing
        table.scale(1.0, 1.3)
        
        # No separate title needed as it's part of the table
        
        # Adjust layout with better margins to prevent cutoff
        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
        
        # Save figure if requested
        if self.config.get('save', True):
            output_dir = self.config.get('output_dir')
            if output_dir is None:
                # Default output directory
                from pathlib import Path
                import sys
                
                # Add project root to path if needed
                project_root = Path(__file__).parent.parent.parent.absolute()
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                    
                # Import settings
                from config import settings
                
                # Use performance directory instead of metrics
                output_dir = settings.VISUALIZATION_DIR / "performance"
            
            # Save figure
            save_figure(
                fig=fig,
                filename="metrics_summary_table",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig


class MetricsComparisonPlot(ComparativeViz):
    """Metrics comparison plot for multiple models."""
    
    def __init__(
        self, 
        models: List[Union[ModelData, Dict[str, Any]]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize metrics comparison plot.
        
        Args:
            models: List of model data or adapters
            config: Visualization configuration
        """
        # Convert model data to adapters if needed
        model_adapters = []
        for model_data in models:
            if not isinstance(model_data, ModelData):
                model_adapters.append(get_adapter_for_model(model_data))
            else:
                model_adapters.append(model_data)
            
        # Call parent constructor
        super().__init__(model_adapters, config)
        
    def plot(self) -> plt.Figure:
        """
        Create metrics comparison plot.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Get metrics for each model
        metrics_data = []
        
        for model in self.models:
            # Get model metadata
            metadata = model.get_metadata()
            model_name = metadata.get('model_name', 'Unknown Model')
            
            # Get metrics
            metrics = model.get_metrics()
            
            # Add model name and metrics
            model_metrics = {'model_name': model_name}
            model_metrics.update(metrics)
            
            # Add to data
            metrics_data.append(model_metrics)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Determine metrics to show
        metrics_to_show = self.config.get('metrics', ['RMSE', 'MAE', 'R2'])
        
        # Filter metrics
        available_metrics = [col for col in metrics_to_show if col in metrics_df.columns]
        
        # Create figure and axes for each metric
        fig, axes = create_comparison_layout(
            n_items=len(metrics_df),
            n_metrics=len(available_metrics),
            figsize=self.config.get('figsize', (5 * len(available_metrics), 6)),
            title=self.config.get('title', 'Model Performance Comparison'),
            title_fontsize=self.config.get('title_fontsize', 16)
        )
        
        # Create plots for each metric
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Sort by metric
            if metric in ['RMSE', 'MAE', 'MSE']:  # Lower is better
                sorted_df = metrics_df.sort_values(metric, ascending=True)
            elif metric in ['R2']:  # Higher is better
                sorted_df = metrics_df.sort_values(metric, ascending=False)
            else:
                sorted_df = metrics_df
            
            # Get colors
            palette = self.style.get('colors', {}).get('primary', '#3498db')
            
            # Create bar chart
            bars = ax.bar(
                sorted_df['model_name'],
                sorted_df[metric],
                color=palette,
                alpha=0.7
            )
            
            # Add value labels
            add_value_labels(
                ax=ax,
                precision=4,
                fontsize=self.config.get('annotation_fontsize', 8),
                color='black',
                vertical_offset=0.01
            )
            
            # Set axis labels
            ax.set_xlabel('Model', fontsize=self.config.get('label_fontsize', 12))
            ax.set_ylabel(metric, fontsize=self.config.get('label_fontsize', 12))
            
            # Set title
            ax.set_title(f'{metric} Comparison', fontsize=self.config.get('title_fontsize', 14))
            
            # Rotate x-tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add grid
            if self.config.get('grid', True):
                ax.grid(axis='y', alpha=self.config.get('grid_alpha', 0.3))
            
            # Format y-axis labels
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            
            # Highlight best model
            if metric in ['RMSE', 'MAE', 'MSE']:  # Lower is better
                best_idx = sorted_df[metric].idxmin()
                best_model = sorted_df.loc[best_idx, 'model_name']
            elif metric in ['R2']:  # Higher is better
                best_idx = sorted_df[metric].idxmax()
                best_model = sorted_df.loc[best_idx, 'model_name']
            else:
                best_model = None
            
            if best_model is not None and self.config.get('highlight_best', True):
                for j, bar in enumerate(bars):
                    if sorted_df.iloc[j]['model_name'] == best_model:
                        bar.set_color(self.style.get('colors', {}).get('success', '#2ecc71'))
                        
                        # Add star annotation
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.1,
                            '*',
                            ha='center',
                            va='center',
                            fontsize=16,
                            color='red'
                        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            output_dir = self.config.get('output_dir')
            if output_dir is None:
                # Default output directory
                from pathlib import Path
                import sys
                
                # Add project root to path if needed
                project_root = Path(__file__).parent.parent.parent.absolute()
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                    
                # Import settings
                from config import settings
                
                # Use performance directory instead of metrics
                output_dir = settings.VISUALIZATION_DIR / "performance"
            
            # Save figure - only if explicitly requested through a config parameter
            # This prevents the model_metrics_comparison.png file from being created by default
            if self.config.get('create_model_metrics_plot', False):
                save_figure(
                    fig=fig,
                    filename="model_metrics_comparison",
                    output_dir=output_dir,
                    dpi=self.config.get('dpi', 300),
                    format=self.config.get('format', 'png')
                )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig


def plot_metrics(
    models: List[Union[ModelData, Dict[str, Any]]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> plt.Figure:
    """
    Create metrics comparison plot.
    
    Args:
        models: List of model data or adapters
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot = MetricsComparisonPlot(models, config)
    return plot.plot()


def plot_metrics_table(
    models: List[Union[ModelData, Dict[str, Any]]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> plt.Figure:
    """
    Create metrics summary table.
    
    Args:
        models: List of model data or adapters
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot = MetricsTable(models, config)
    return plot.plot()


# Alias for compatibility
plot_model_comparison = plot_metrics