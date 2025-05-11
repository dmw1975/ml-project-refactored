"""Sector-specific visualizations for model analysis.

This module provides topic-based visualizations for sector models,
using the new model-agnostic visualization architecture.
"""

import sys
import os
from pathlib import Path

# Add project root to the path so Python can find the modules
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Union, List, Tuple

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import BaseViz, ComparativeViz
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.components.annotations import add_value_labels
from visualization_new.components.formats import save_figure
from visualization_new.utils.io import ensure_dir

# Import project settings
import sys

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
from config import settings


class SectorPerformanceComparison(BaseViz):
    """Sector-specific performance comparison visualization."""
    
    def __init__(
        self, 
        metrics_df: Optional[pd.DataFrame] = None,
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize sector performance comparison.
        
        Args:
            metrics_df: DataFrame with sector metrics
            config: Visualization configuration
        """
        super().__init__(config)
        
        # Load metrics if not provided
        if metrics_df is None:
            metrics_file = settings.METRICS_DIR / "sector_models_metrics.csv"
            if metrics_file.exists():
                self.metrics_df = pd.read_csv(metrics_file)
            else:
                raise ValueError("No sector model metrics found. Please run sector model evaluation first.")
        else:
            self.metrics_df = metrics_df
        
        # Check if sector column exists
        if 'sector' not in self.metrics_df.columns:
            print("Warning: 'sector' column not found in metrics file.")
            # Try to extract sector from model_name
            if 'model_name' in self.metrics_df.columns:
                print("Extracting sector information from model names...")
                # Extract sector from model names (format: "Sector_SectorName_...")
                self.metrics_df['sector'] = self.metrics_df['model_name'].apply(
                    lambda x: x.split('_')[1] if x.startswith('Sector_') and len(x.split('_')) > 2 else 'Unknown'
                )
            else:
                raise ValueError("Cannot create sector visualizations: Missing required data.")
    
    def plot_sector_performance(self) -> plt.Figure:
        """
        Create sector performance comparison plot.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Group by sector and get average metrics
        sector_perf = self.metrics_df.groupby('sector').agg({
            'RMSE': 'mean',
            'R2': 'mean',
            'n_companies': 'mean'  # All models for a sector have the same count
        }).reset_index()
        
        # Sort by RMSE
        sector_perf = sector_perf.sort_values('RMSE')
        
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 8)))
        
        # Create bar chart
        bars = ax.bar(sector_perf['sector'], sector_perf['RMSE'], 
                     color=self.style.get('colors', {}).get('primary', '#3498db'), alpha=0.7)
        
        # Add value labels
        for bar, r2, count in zip(bars, sector_perf['R2'], sector_perf['n_companies']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'RMSE: {height:.4f}\nR²: {r2:.4f}\n(n={int(count)})', 
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Average Model Performance by Sector', 
                    fontsize=self.config.get('title_fontsize', 14))
        ax.set_ylabel('Mean RMSE (lower is better)')
        ax.set_xlabel('Sector')
        
        # Set rotation for x-axis tick labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            output_dir = self.config.get('output_dir')
            if output_dir is None:
                output_dir = settings.VISUALIZATION_DIR / "sectors"
            
            save_figure(
                fig=fig,
                filename="sector_performance_comparison",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def plot_model_type_heatmap(self) -> Optional[plt.Figure]:
        """
        Create sector model type performance heatmap.
        
        Returns:
            matplotlib.figure.Figure: The created figure or None if not applicable
        """
        if 'type' not in self.metrics_df.columns:
            print("Warning: 'type' column not found. Skipping model type heatmap.")
            return None
        
        try:
            # Pivot to create sector x model_type grid with RMSE values
            pivot_df = self.metrics_df.pivot_table(
                index='sector', 
                columns='type', 
                values='RMSE',
                aggfunc='mean'
            )
            
            # Sort sectors by overall performance
            sector_perf = self.metrics_df.groupby('sector')['RMSE'].mean().sort_values().index.tolist()
            pivot_df = pivot_df.reindex(sector_perf)
            
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 10)))
            
            # Create heatmap
            sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlGnBu_r',
                       linewidths=0.5, ax=ax)
            
            ax.set_title('Model Type Performance by Sector (RMSE)', 
                        fontsize=self.config.get('title_fontsize', 14))
            
            plt.tight_layout()
            
            # Save figure if requested
            if self.config.get('save', True):
                output_dir = self.config.get('output_dir')
                if output_dir is None:
                    output_dir = settings.VISUALIZATION_DIR / "sectors"
                
                save_figure(
                    fig=fig,
                    filename="sector_model_type_heatmap",
                    output_dir=output_dir,
                    dpi=self.config.get('dpi', 300),
                    format=self.config.get('format', 'png')
                )
            
            # Show figure if requested
            if self.config.get('show', False):
                plt.show()
            
            return fig
        except Exception as e:
            print(f"Error creating model type heatmap: {e}")
            return None
    
    def plot_sector_vs_overall(self) -> Optional[plt.Figure]:
        """
        Create sector vs overall model comparison.
        
        Returns:
            matplotlib.figure.Figure: The created figure or None if not applicable
        """
        # Load main metrics
        main_metrics_file = settings.METRICS_DIR / "all_models_comparison.csv"
        if not main_metrics_file.exists():
            print("Main metrics file not found. Skipping overall vs. sector comparison.")
            return None
        
        try:
            main_metrics_df = pd.read_csv(main_metrics_file)
            
            # Filter for relevant model types to compare
            if 'model_type' in main_metrics_df.columns:
                main_models = main_metrics_df[main_metrics_df['model_type'] == 'Linear Regression']
            else:
                main_models = main_metrics_df
            
            # Calculate average metrics for each approach
            main_avg = main_models.mean()
            sector_avg = self.metrics_df.mean()
            
            # Prepare comparison data
            comparison_data = {
                'Approach': ['Overall Models', 'Sector-Specific Models'],
                'RMSE': [main_avg['RMSE'], sector_avg['RMSE']],
                'MAE': [main_avg['MAE'], sector_avg['MAE']],
                'R2': [main_avg['R2'], sector_avg['R2']]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=self.config.get('figsize', (14, 6)))
            
            # RMSE comparison
            ax = axes[0]
            bars = ax.bar(comparison_df['Approach'], comparison_df['RMSE'], 
                         color=['#3498db', '#e74c3c'])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            ax.set_title('RMSE Comparison', fontsize=self.config.get('title_fontsize', 14))
            ax.set_ylabel('RMSE (lower is better)')
            ax.grid(axis='y', alpha=0.3)
            
            # R2 comparison
            ax = axes[1]
            bars = ax.bar(comparison_df['Approach'], comparison_df['R2'], 
                         color=['#3498db', '#e74c3c'])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            ax.set_title('R² Comparison', fontsize=self.config.get('title_fontsize', 14))
            ax.set_ylabel('R² (higher is better)')
            ax.grid(axis='y', alpha=0.3)
            
            plt.suptitle('Overall vs. Sector-Specific Model Performance', 
                        fontsize=self.config.get('title_fontsize', 16))
            plt.tight_layout()
            
            # Save figure if requested
            if self.config.get('save', True):
                output_dir = self.config.get('output_dir')
                if output_dir is None:
                    output_dir = settings.VISUALIZATION_DIR / "sectors"
                
                save_figure(
                    fig=fig,
                    filename="overall_vs_sector_comparison",
                    output_dir=output_dir,
                    dpi=self.config.get('dpi', 300),
                    format=self.config.get('format', 'png')
                )
            
            # Show figure if requested
            if self.config.get('show', False):
                plt.show()
            
            return fig
        except Exception as e:
            print(f"Error creating overall vs. sector comparison: {e}")
            return None
    
    def plot_sector_boxplots(self) -> plt.Figure:
        """
        Create sector performance boxplots.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config.get('figsize', (16, 8)))
        
        # RMSE by sector
        ax = axes[0]
        sns.boxplot(x='sector', y='RMSE', data=self.metrics_df, ax=ax, palette='Blues')
        ax.set_title('RMSE Distribution by Sector', fontsize=self.config.get('title_fontsize', 14))
        ax.set_xlabel('Sector')
        ax.set_ylabel('RMSE (lower is better)')
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # R2 by sector
        ax = axes[1]
        sns.boxplot(x='sector', y='R2', data=self.metrics_df, ax=ax, palette='Blues')
        ax.set_title('R² Distribution by Sector', fontsize=self.config.get('title_fontsize', 14))
        ax.set_xlabel('Sector')
        ax.set_ylabel('R² (higher is better)')
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            output_dir = self.config.get('output_dir')
            if output_dir is None:
                output_dir = settings.VISUALIZATION_DIR / "sectors"
            
            save_figure(
                fig=fig,
                filename="sector_performance_boxplots",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def plot(self) -> Dict[str, plt.Figure]:
        """
        Create all sector performance visualizations.
        
        Returns:
            Dict[str, matplotlib.figure.Figure]: Dictionary of created figures
        """
        figures = {}
        
        # 1. Sector Performance Comparison
        figures['sector_performance'] = self.plot_sector_performance()
        
        # 2. Model Type Heatmap
        model_type_fig = self.plot_model_type_heatmap()
        if model_type_fig is not None:
            figures['model_type_heatmap'] = model_type_fig
        
        # 3. Sector vs Overall Comparison
        sector_vs_overall_fig = self.plot_sector_vs_overall()
        if sector_vs_overall_fig is not None:
            figures['sector_vs_overall'] = sector_vs_overall_fig
        
        # 4. Sector Performance Boxplots
        figures['sector_boxplots'] = self.plot_sector_boxplots()
        
        return figures


class SectorMetricsTable(BaseViz):
    """Sector metrics summary table visualization."""
    
    def __init__(
        self, 
        metrics_df: Optional[pd.DataFrame] = None,
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize sector metrics table.
        
        Args:
            metrics_df: DataFrame with sector metrics
            config: Visualization configuration
        """
        super().__init__(config)
        
        # Load metrics if not provided
        if metrics_df is None:
            metrics_file = settings.METRICS_DIR / "sector_models_metrics.csv"
            if metrics_file.exists():
                self.metrics_df = pd.read_csv(metrics_file)
            else:
                raise ValueError("No sector metrics data found. Please run sector model evaluation first.")
        else:
            self.metrics_df = metrics_df
    
    def plot(self) -> plt.Figure:
        """
        Create sector metrics summary table.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Select and rename columns for the table
        if 'model_name' in self.metrics_df.columns:
            metrics_df = self.metrics_df.rename(columns={'model_name': 'Model'})
        elif 'index' in self.metrics_df.columns:
            metrics_df = self.metrics_df.rename(columns={'index': 'Model'})
        else:
            metrics_df = self.metrics_df.copy()
        
        # Make sure R² is spelled correctly
        if 'R2' in metrics_df.columns and 'R²' not in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={'R2': 'R²'})
        
        # Select only the required columns (excluding sector and type)
        table_columns = ['Model', 'MSE', 'MAE', 'RMSE', 'R²', 'n_companies']
        
        # Filter columns that exist in the DataFrame
        available_columns = [col for col in table_columns if col in metrics_df.columns]
        table_data = metrics_df[available_columns].copy()
        
        # Convert n_companies to integer
        if 'n_companies' in table_data.columns:
            table_data['n_companies'] = table_data['n_companies'].astype(int)
        
        # Create figure - increase height for more rows and width for model names
        fig = plt.figure(figsize=self.config.get('figsize', (16, len(table_data) * 0.5 + 1)))
        
        # Create a table with no cells, just the data
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Create cell colors array - initialize with white
        colors = [['white' for _ in range(len(table_data.columns))] for _ in range(len(table_data))]
        
        # Highlight cells with positive R² values
        if 'R²' in table_data.columns:
            r2_col_idx = list(table_data.columns).index('R²')
            for i, row in enumerate(table_data.values):
                r2_value = row[r2_col_idx]
                if r2_value > 0:
                    colors[i][r2_col_idx] = '#d9ead3'  # Light green for positive R²
        
        # Convert values to formatted strings
        cell_text = []
        for row in table_data.values:
            row_text = []
            for i, val in enumerate(row):
                col_name = table_data.columns[i]
                if col_name == 'Model':
                    # Just use the string value for model name
                    row_text.append(str(val))
                elif col_name == 'n_companies':
                    # Format as integer
                    row_text.append(f"{int(val)}")
                elif isinstance(val, (int, float, np.number)):
                    # Format other numeric columns
                    row_text.append(f"{val:.4f}")
                else:
                    row_text.append(str(val))
            cell_text.append(row_text)
        
        # Set column widths, making Model column wider
        col_widths = [0.4 if table_data.columns[i] == 'Model' else 0.12 for i in range(len(table_data.columns))]
        
        # Create the table with adjusted column widths
        table = plt.table(
            cellText=cell_text,
            colLabels=table_data.columns,
            cellColours=colors,
            cellLoc='center',
            loc='center',
            colWidths=col_widths
        )
        
        # Adjust font size and spacing
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        plt.title('Sector Models Performance Metrics Summary', 
                 fontsize=self.config.get('title_fontsize', 16), pad=20)
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            output_dir = self.config.get('output_dir')
            if output_dir is None:
                output_dir = settings.VISUALIZATION_DIR / "sectors"
            
            save_figure(
                fig=fig,
                filename="sector_metrics_summary_table",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig


def plot_sector_performance(
    metrics_df: Optional[pd.DataFrame] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    Create sector performance visualizations.
    
    Args:
        metrics_df: DataFrame with sector metrics
        config: Visualization configuration
        
    Returns:
        Dict[str, matplotlib.figure.Figure]: Dictionary of created figures
    """
    plot = SectorPerformanceComparison(metrics_df, config)
    return plot.plot()


def plot_sector_metrics_table(
    metrics_df: Optional[pd.DataFrame] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> plt.Figure:
    """
    Create sector metrics summary table.
    
    Args:
        metrics_df: DataFrame with sector metrics
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot = SectorMetricsTable(metrics_df, config)
    return plot.plot()


def visualize_all_sector_plots(
    metrics_df: Optional[pd.DataFrame] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    Create all sector visualizations.
    
    Args:
        metrics_df: DataFrame with sector metrics
        config: Visualization configuration
        
    Returns:
        Dict[str, matplotlib.figure.Figure]: Dictionary of created figures
    """
    # Set output directory
    if config is None:
        config = VisualizationConfig()
    elif isinstance(config, dict):
        config = VisualizationConfig(**config)
    
    if config.output_dir is None:
        config.output_dir = settings.VISUALIZATION_DIR / "sectors"
    
    ensure_dir(config.output_dir)
    
    # Create all visualizations
    figures = {}
    
    # 1. Sector Performance
    perf_figures = plot_sector_performance(metrics_df, config)
    figures.update(perf_figures)
    
    # 2. Sector Metrics Table
    figures['metrics_table'] = plot_sector_metrics_table(metrics_df, config)
    
    print(f"All sector visualizations saved to {config.output_dir}")
    return figures