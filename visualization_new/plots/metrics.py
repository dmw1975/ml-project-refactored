"""Performance metrics plots for all model types."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import ModelViz, ComparativeViz
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.components.annotations import add_value_labels
from visualization_new.components.layouts import create_grid_layout, create_comparison_layout
from visualization_new.components.formats import format_figure_for_export, save_figure


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
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, len(table_data) * 0.5 + 1)))
        
        # Hide axes
        ax.axis('off')
        ax.axis('tight')
        
        # Get colors for each cell
        colors = []
        
        # Initialize with white
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
        
        # Format values as strings
        cell_text = []
        
        for i, row in table_data.iterrows():
            row_text = []
            
            for j, val in enumerate(row):
                if j == 0:  # Model name
                    row_text.append(str(val))
                else:  # Metric
                    if isinstance(val, (int, float, np.number)):
                        row_text.append(f"{val:.4f}")
                    else:
                        row_text.append(str(val))
            
            cell_text.append(row_text)
        
        # Create table
        table = ax.table(
            cellText=cell_text,
            colLabels=table_data.columns,
            cellColours=colors,
            cellLoc='center',
            loc='center'
        )
        
        # Format table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Set title
        plt.title('Model Performance Metrics Summary', fontsize=self.config.get('title_fontsize', 16), pad=20)
        
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
                
                # Use default from settings
                output_dir = settings.VISUALIZATION_DIR / "metrics"
            
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
                
                # Use default from settings
                output_dir = settings.VISUALIZATION_DIR / "metrics"
            
            # Save figure
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