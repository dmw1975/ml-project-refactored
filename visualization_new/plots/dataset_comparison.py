"""Dataset-centric model comparison visualizations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import ModelViz, ComparativeViz
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.components.annotations import add_value_labels
from visualization_new.components.layouts import create_grid_layout, create_comparison_layout
from visualization_new.components.formats import format_figure_for_export, save_figure
from visualization_new.utils.io import load_all_models, ensure_dir


class DatasetModelComparisonPlot(ComparativeViz):
    """Dataset-centric comparison plot for multiple models grouped by model family."""
    
    def __init__(
        self, 
        models: List[Union[ModelData, Dict[str, Any]]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize dataset-centric model comparison plot.
        
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

    def extract_model_metrics(self) -> pd.DataFrame:
        """
        Extract metrics from models into a DataFrame for easier comparison.
        
        Returns:
            pd.DataFrame: DataFrame with model metrics
        """
        model_metrics = []
        
        for model in self.models:
            # Get model metadata
            metadata = model.get_metadata()
            model_name = metadata.get('model_name', 'Unknown')
            
            # Skip if missing metrics
            metrics = model.get_metrics()
            if 'RMSE' not in metrics or 'R2' not in metrics or 'MAE' not in metrics:
                print(f"Skipping {model_name}: Missing metrics")
                continue
            
            # Determine dataset type
            if 'Base_Random' in model_name:
                dataset = 'Base_Random'
            elif 'Base' in model_name:
                dataset = 'Base'
            elif 'Yeo_Random' in model_name:
                dataset = 'Yeo_Random'
            elif 'Yeo' in model_name:
                dataset = 'Yeo'
            else:
                dataset = 'Unknown'
            
            # Determine model family and tuning status
            if model_name.startswith('LR_'):
                model_family = 'Linear'
                tuned = False
            elif model_name.startswith('ElasticNet_'):
                model_family = 'Linear'
                tuned = True
            elif 'XGB' in model_name:
                model_family = 'XGBoost'
                tuned = 'optuna' in model_name
            elif 'LightGBM' in model_name:
                model_family = 'LightGBM'
                tuned = 'optuna' in model_name
            elif 'CatBoost' in model_name:
                model_family = 'CatBoost'
                tuned = 'optuna' in model_name
            else:
                model_family = 'Unknown'
                tuned = False
                
            # Calculate MSE from RMSE if not available
            if 'MSE' not in metrics:
                mse = metrics['RMSE'] ** 2
            else:
                mse = metrics['MSE']
                
            # Create a record with all the information
            model_metrics.append({
                'model_name': model_name,
                'dataset': dataset,
                'model_family': model_family,
                'tuned': tuned,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MSE': mse
            })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(model_metrics)
        return metrics_df
    
    def plot_dataset_comparison(self, dataset_name: str) -> plt.Figure:
        """
        Create plots showing different metrics for all models on a specific dataset,
        grouped by model family with basic and tuned variants shown side-by-side.
        
        Args:
            dataset_name: Name of the dataset to filter by (e.g., 'Base', 'Yeo')
            
        Returns:
            plt.Figure: Figure object
        """
        # Extract metrics
        metrics_df = self.extract_model_metrics()
        
        # Filter metrics for the specified dataset
        dataset_metrics = metrics_df[metrics_df['dataset'] == dataset_name]
        
        if dataset_metrics.empty:
            print(f"No models found for dataset {dataset_name}")
            return None
        
        # Define model families and their colors
        model_families = ['Linear', 'XGBoost', 'LightGBM', 'CatBoost']
        family_colors = {
            'Linear': '#9b59b6',   # Purple
            'XGBoost': '#3498db',  # Blue
            'LightGBM': '#2ecc71', # Green
            'CatBoost': '#e74c3c', # Red
            'Unknown': '#95a5a6'   # Gray
        }
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Set up axes for each metric
        metrics = [
            {'name': 'RMSE', 'ax': axes[0, 0], 'title': 'RMSE Comparison', 'ylabel': 'RMSE (lower is better)', 'best': 'min'},
            {'name': 'MAE', 'ax': axes[0, 1], 'title': 'MAE Comparison', 'ylabel': 'MAE (lower is better)', 'best': 'min'},
            {'name': 'R2', 'ax': axes[1, 0], 'title': 'R² Comparison', 'ylabel': 'R² (higher is better)', 'best': 'max'},
            {'name': 'MSE', 'ax': axes[1, 1], 'title': 'MSE Comparison', 'ylabel': 'MSE (lower is better)', 'best': 'min'}
        ]
        
        # Create subplots for each metric
        for metric in metrics:
            ax = metric['ax']
            metric_name = metric['name']
            best_fn = min if metric['best'] == 'min' else max
            
            # Group data by model family
            x_positions = []
            x_labels = []
            bar_colors = []
            bar_values = []
            best_value = best_fn(dataset_metrics[metric_name])
            
            # Add models with consistent spacing between families
            family_gap = 0.5  # Gap between different model families
            bar_width = 0.35  # Width of each bar
            pair_center = 0   # Center position for each basic/tuned pair
            
            for family in model_families:
                # Filter models for this family
                family_models = dataset_metrics[dataset_metrics['model_family'] == family]
                
                if len(family_models) == 0:
                    continue
                    
                # Get basic and tuned models
                basic_model = family_models[family_models['tuned'] == False]
                tuned_model = family_models[family_models['tuned'] == True]
                
                # Check if we have models
                has_basic = not basic_model.empty
                has_tuned = not tuned_model.empty
                
                # Calculate positions
                if has_basic and has_tuned:
                    # Both models exist - place side by side
                    x_positions.extend([pair_center - bar_width/2, pair_center + bar_width/2])
                    
                    # Add values
                    bar_values.extend([
                        basic_model.iloc[0][metric_name] if has_basic else 0,
                        tuned_model.iloc[0][metric_name] if has_tuned else 0
                    ])
                    
                    # Add colors
                    color = family_colors.get(family, family_colors['Unknown'])
                    bar_colors.extend([color, color])
                    
                    # Labels
                    x_labels.extend(['', ''])
                    
                    # Update center for next family
                    pair_center += family_gap + 1
                elif has_basic:
                    # Only basic model exists
                    x_positions.append(pair_center)
                    bar_values.append(basic_model.iloc[0][metric_name])
                    color = family_colors.get(family, family_colors['Unknown'])
                    bar_colors.append(color)
                    x_labels.append('')
                    pair_center += family_gap + 1
                elif has_tuned:
                    # Only tuned model exists
                    x_positions.append(pair_center)
                    bar_values.append(tuned_model.iloc[0][metric_name])
                    color = family_colors.get(family, family_colors['Unknown'])
                    bar_colors.append(color)
                    x_labels.append('')
                    pair_center += family_gap + 1
            
            # Create bars
            bars = ax.bar(x_positions, bar_values, width=bar_width, color=bar_colors)
            
            # Add value labels
            for bar, value in zip(bars, bar_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                
                # Highlight the best model
                if (metric['best'] == 'min' and value == best_fn(bar_values)) or \
                   (metric['best'] == 'max' and value == best_fn(bar_values)):
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)
            
            # Add family labels below x-axis
            family_positions = []
            shown_families = []
            
            pair_center = 0
            for family in model_families:
                family_models = dataset_metrics[dataset_metrics['model_family'] == family]
                if len(family_models) > 0:
                    family_positions.append(pair_center)
                    shown_families.append(family)
                    pair_center += family_gap + 1
            
            # Set x-axis properties
            ax.set_xticks(family_positions)
            ax.set_xticklabels(shown_families, fontsize=12)
            
            # Add a legend for basic vs tuned
            basic_patch = plt.Rectangle((0, 0), 1, 1, fc='none', ec='none', label='Basic')
            tuned_patch = plt.Rectangle((0, 0), 1, 1, fc='none', ec='none', label='Tuned')
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc='none', ec='none', label=' '),
                plt.Rectangle((0, 0), 1, 1, fc='none', ec='black', linewidth=2, label='Best Model')
            ]
            
            # Add shapes to indicate basic/tuned positioning
            ax.annotate('Basic', xy=(family_positions[0] - bar_width/2, -0.05), 
                       xycoords=('data', 'axes fraction'), ha='center', fontsize=10)
            ax.annotate('Tuned', xy=(family_positions[0] + bar_width/2, -0.05), 
                       xycoords=('data', 'axes fraction'), ha='center', fontsize=10)
            
            # Set other axes properties
            ax.set_title(f'{metric["title"]} - {dataset_name} Dataset', fontsize=14)
            ax.set_ylabel(metric['ylabel'], fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            # Set reasonable ylim
            if metric['best'] == 'min':
                ax.set_ylim(0, best_fn(bar_values) * 1.5)
            else:
                # For R², we want a more specific range (usually 0 to 1)
                ax.set_ylim(max(0, min(bar_values) - 0.1), min(1.0, max(bar_values) + 0.1))
        
        # Add overall title
        plt.suptitle(f'Model Performance Metrics on {dataset_name} Dataset', fontsize=16, y=0.98)
        
        # Add explanation
        plt.figtext(0.5, 0.01, 
                   "Models are grouped by family with basic and tuned variants side-by-side.\n"
                   "Linear: Basic=Linear Regression, Tuned=ElasticNet | Other models: Basic=Default params, Tuned=Optuna optimized", 
                   ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        return fig
    
    def plot(self) -> Dict[str, plt.Figure]:
        """
        Create comparison visualizations for all datasets.
        
        Returns:
            Dict[str, plt.Figure]: Dictionary of created figures by dataset
        """
        # Extract metrics
        metrics_df = self.extract_model_metrics()
        
        # Get unique datasets
        datasets = list(metrics_df['dataset'].unique())
        
        # Create comparison for each dataset
        figures = {}
        
        for dataset in datasets:
            print(f"Creating comparison for {dataset} dataset...")
            fig = self.plot_dataset_comparison(dataset)
            if fig:
                figures[dataset] = fig
                
                # Save figure if requested
                if self.config.get('save', True):
                    output_dir = self.config.get('output_dir')
                    if output_dir is None:
                        # Import settings
                        from pathlib import Path
                        import sys
                        
                        # Add project root to path if needed
                        project_root = Path(__file__).parent.parent.parent.absolute()
                        if str(project_root) not in sys.path:
                            sys.path.append(str(project_root))
                            
                        # Import settings
                        from config import settings
                        
                        # Use default output directory
                        output_dir = settings.VISUALIZATION_DIR / "dataset_comparison"
                    
                    # Ensure directory exists
                    ensure_dir(output_dir)
                    
                    # Save figure
                    save_figure(
                        fig=fig,
                        filename=f"{dataset}_model_family_comparison",
                        output_dir=output_dir,
                        dpi=self.config.get('dpi', 300),
                        format=self.config.get('format', 'png')
                    )
                    
                    print(f"Saved {dataset} dataset comparison to {output_dir}")
        
        return figures
        

def plot_dataset_comparison(
    models: List[Union[ModelData, Dict[str, Any]]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    Create dataset-centric model comparison visualizations.
    
    Args:
        models: List of model data or adapters
        config: Visualization configuration
        
    Returns:
        Dict[str, plt.Figure]: Dictionary of created figures
    """
    plot = DatasetModelComparisonPlot(models, config)
    return plot.plot()


def create_all_dataset_comparisons(
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    Create dataset-centric model comparison visualizations for all models.
    
    Args:
        config: Visualization configuration
        
    Returns:
        Dict[str, plt.Figure]: Dictionary of created figures
    """
    # Load all models
    all_models = load_all_models()
    model_list = list(all_models.values())
    
    # Create comparison
    return plot_dataset_comparison(model_list, config)