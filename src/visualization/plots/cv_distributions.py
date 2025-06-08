"""Cross-validation distribution visualizations for model comparison."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import scipy.stats as st

from src.visualization.core.interfaces import ModelData, VisualizationConfig
from src.visualization.core.base import ModelViz, ComparativeViz
from src.visualization.core.registry import get_adapter_for_model
from src.visualization.components.annotations import add_value_labels
from src.visualization.components.layouts import create_grid_layout, create_comparison_layout
from src.visualization.components.formats import format_figure_for_export, save_figure
from src.visualization.utils.io import ensure_dir


class CVDistributionPlot(ComparativeViz):
    """Visualizations for cross-validation RMSE distributions across models."""
    
    def __init__(
        self, 
        models: List[ModelData], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize CV distribution visualizations.
        
        Args:
            models: List of ModelData objects with CV results (can be adapters or raw data)
            config: Visualization configuration
        """
        # Don't call parent constructor if models might be adapters
        self.models = models
        self.config = config if isinstance(config, VisualizationConfig) else VisualizationConfig(**(config or {}))
        
    def _extract_cv_metrics(self, model: ModelData) -> Optional[Dict[str, Any]]:
        """
        Extract cross-validation metrics from a model.
        
        Args:
            model: ModelData object or adapter
            
        Returns:
            Dict with CV metrics or None if not available
        """
        # Handle both raw model data and adapters
        if hasattr(model, 'get_raw_model_data'):
            # It's already an adapter
            adapter = model
        else:
            # It's raw model data, get adapter
            adapter = get_adapter_for_model(model)
        
        # Get raw model data
        raw_data = adapter.get_raw_model_data()
        
        # Initialize rmse_scores as None
        rmse_scores = None
        
        # Try different ways to extract CV scores
        if isinstance(raw_data, dict):
            # Tree models and ElasticNet store cv_scores directly
            if 'cv_scores' in raw_data and raw_data['cv_scores'] is not None:
                cv_scores = raw_data['cv_scores']
                # Check if these are already RMSE or need conversion from MSE
                if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
                    # For tree models, cv_scores are already RMSE values
                    rmse_scores = np.array(cv_scores)
            elif 'cv_mse' in raw_data and raw_data['cv_mse'] is not None:
                # Convert MSE to RMSE
                rmse_scores = np.sqrt(np.array(raw_data['cv_mse']))
        
        # Try adapter's cv_scores attribute if available
        if rmse_scores is None and hasattr(model, 'data'):
            if hasattr(model.data, 'cv_scores') and model.data.cv_scores is not None:
                cv_scores = model.data.cv_scores
                if isinstance(cv_scores, (list, np.ndarray)) and len(cv_scores) > 0:
                    rmse_scores = np.array(cv_scores)
            
            # Try adapter's cv_mse attribute if available
            elif hasattr(model.data, 'cv_mse') and model.data.cv_mse is not None:
                rmse_scores = np.sqrt(np.array(model.data.cv_mse))
        
        # For linear models, check if they have cv_results stored
        if rmse_scores is None and adapter.get_model_type() in ['Linear Regression', 'ElasticNet']:
            if hasattr(model.data, 'cv_results') and model.data.cv_results is not None:
                # Linear models might store full CV results
                cv_results = model.data.cv_results
                if isinstance(cv_results, dict) and 'test_score' in cv_results:
                    # These are typically negative MSE scores from sklearn
                    rmse_scores = np.sqrt(-np.array(cv_results['test_score']))
        
        if rmse_scores is None:
            return None
        
        # Ensure we have valid RMSE scores
        rmse_scores = np.array(rmse_scores)
        if len(rmse_scores) == 0:
            return None
        
        # Get model name
        if hasattr(model, 'name'):
            model_name = model.name
        elif hasattr(adapter, 'model_name'):
            model_name = adapter.model_name
        else:
            model_name = 'Unknown'
            
        return {
            'model_name': model_name,
            'model_type': adapter.get_model_type(),
            'dataset': adapter.get_dataset_name(),
            'rmse_scores': rmse_scores,
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'n_folds': len(rmse_scores)
        }
    
    def _mean_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval.
        
        Args:
            data: Array of values
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        n = len(data)
        m, se = np.mean(data), st.sem(data)
        h = se * st.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h
    
    def plot_cv_rmse_distribution(self, model_types: Optional[List[str]] = None) -> plt.Figure:
        """
        Create box plots showing CV RMSE distribution for all models.
        
        Args:
            model_types: Optional list of model types to include
            
        Returns:
            plt.Figure: Figure with CV RMSE distribution plots
        """
        # Extract CV metrics from all models
        cv_data = []
        for model in self.models:
            metrics = self._extract_cv_metrics(model)
            if metrics:
                cv_data.append(metrics)
        
        if not cv_data:
            print("No cross-validation metrics found in any model.")
            return None
        
        # Filter by model types if specified
        if model_types:
            cv_data = [m for m in cv_data if m['model_type'] in model_types]
        
        if not cv_data:
            print(f"No models found for types: {model_types}")
            return None
        
        # Prepare data for plotting
        plot_data = []
        for metrics in cv_data:
            for rmse in metrics['rmse_scores']:
                plot_data.append({
                    'Model': metrics['model_name'],
                    'Model Type': metrics['model_type'],
                    'Dataset': metrics['dataset'],
                    'RMSE': rmse
                })
        
        df = pd.DataFrame(plot_data)
        
        # Determine plot layout based on number of model types
        unique_types = df['Model Type'].unique()
        n_types = len(unique_types)
        
        if n_types == 1:
            # Single model type - group by dataset
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create box plot
            box = sns.boxplot(
                x='Dataset', 
                y='RMSE', 
                data=df, 
                palette='pastel', 
                ax=ax
            )
            
            # Add strip plot for individual fold RMSEs
            strip = sns.stripplot(
                x='Dataset', 
                y='RMSE', 
                data=df, 
                color='gray', 
                alpha=0.6, 
                jitter=True, 
                ax=ax
            )
            
            # Add mean as red dot (without CI)
            for i, dataset in enumerate(df['Dataset'].unique()):
                dataset_data = df[df['Dataset'] == dataset]['RMSE']
                mean = np.mean(dataset_data.values)
                
                # Plot mean value only
                ax.plot(i, mean, 'o', color='red', markersize=8, zorder=10)
            
            ax.set_title(f'{unique_types[0]} CV RMSE Distribution by Dataset', fontsize=14)
            ax.set_ylabel('RMSE (lower is better)')
            ax.set_xlabel('Dataset')
            
        else:
            # Multiple model types - create subplots
            fig, axes = plt.subplots(
                nrows=(n_types + 1) // 2, 
                ncols=2, 
                figsize=(16, 6 * ((n_types + 1) // 2))
            )
            
            if n_types == 2:
                axes = axes.reshape(-1)
            else:
                axes = axes.flatten()
            
            for idx, model_type in enumerate(unique_types):
                ax = axes[idx]
                type_data = df[df['Model Type'] == model_type]
                
                # Create box plot
                sns.boxplot(
                    x='Dataset', 
                    y='RMSE', 
                    data=type_data, 
                    palette='pastel', 
                    ax=ax
                )
                
                # Add strip plot
                sns.stripplot(
                    x='Dataset', 
                    y='RMSE', 
                    data=type_data, 
                    color='gray', 
                    alpha=0.6, 
                    jitter=True, 
                    ax=ax
                )
                
                # Add mean as red dot (without CI)
                for i, dataset in enumerate(type_data['Dataset'].unique()):
                    dataset_data = type_data[type_data['Dataset'] == dataset]['RMSE']
                    if len(dataset_data) > 0:
                        mean = np.mean(dataset_data.values)
                        
                        # Plot mean value only
                        ax.plot(i, mean, 'o', color='red', markersize=8, zorder=10)
                
                ax.set_title(f'{model_type} CV RMSE Distribution', fontsize=12)
                ax.set_ylabel('RMSE (lower is better)')
                ax.set_xlabel('Dataset')
                ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Hide extra subplots if odd number of model types
            for idx in range(n_types, len(axes)):
                axes[idx].set_visible(False)
            
            fig.suptitle('Cross-Validation RMSE Distribution by Model Type', fontsize=16)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='RMSE Distribution (Boxplot)',
                   markerfacecolor='lightblue', markersize=15),
            Line2D([0], [0], marker='o', color='gray', label='Individual CV Fold RMSE',
                   linestyle='None', markersize=8, alpha=0.6),
            Line2D([0], [0], marker='o', color='red', label='Mean RMSE',
                   linestyle='None', markersize=8)
        ]
        
        if n_types == 1:
            ax.legend(handles=legend_elements, loc='upper right')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            # Add legend to the figure
            fig.legend(
                handles=legend_elements, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.98),
                ncol=3
            )
        
        plt.tight_layout()
        
        return fig
    
    def plot_cv_rmse_comparison(self) -> plt.Figure:
        """
        Create a comparative plot showing all models' CV RMSE distributions together.
        
        Returns:
            plt.Figure: Figure with comparative CV RMSE plot
        """
        # Extract CV metrics from all models
        cv_data = []
        for model in self.models:
            metrics = self._extract_cv_metrics(model)
            if metrics:
                cv_data.append(metrics)
        
        if not cv_data:
            print("No cross-validation metrics found in any model.")
            return None
        
        # Sort by mean RMSE
        cv_data.sort(key=lambda x: x['mean_rmse'])
        
        # Prepare data for plotting
        model_names = []
        rmse_values = []
        colors = []
        
        # Define color mapping for model types
        color_map = {
            'Linear Regression': '#9b59b6',
            'ElasticNet': '#8e44ad',
            'XGBoost': '#3498db',
            'LightGBM': '#2ecc71',
            'CatBoost': '#e74c3c'
        }
        
        for metrics in cv_data:
            model_names.append(metrics['model_name'])
            rmse_values.append(metrics['rmse_scores'])
            colors.append(color_map.get(metrics['model_type'], '#95a5a6'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create box plot
        positions = np.arange(len(model_names))
        bp = ax.boxplot(
            rmse_values, 
            positions=positions,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )
        
        # Color boxes by model type
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('CV RMSE (lower is better)', fontsize=12)
        ax.set_title('Cross-Validation RMSE Comparison Across All Models', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add legend for model types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=model_type, alpha=0.7)
            for model_type, color in color_map.items()
            if any(m['model_type'] == model_type for m in cv_data)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add horizontal line at best mean RMSE
        best_mean = cv_data[0]['mean_rmse']
        ax.axhline(y=best_mean, color='green', linestyle='--', alpha=0.5)
        ax.text(
            len(model_names) - 1, best_mean + 0.01,
            f'Best mean RMSE: {best_mean:.4f}',
            ha='right', va='bottom', color='green'
        )
        
        plt.tight_layout()
        
        return fig
    
    def plot(self) -> Dict[str, plt.Figure]:
        """
        Create all CV distribution visualizations.
        
        Returns:
            Dict[str, plt.Figure]: Dictionary of created figures
        """
        figures = {}
        
        # Get unique model types
        model_types = set()
        for model in self.models:
            # Handle both raw model data and adapters
            if hasattr(model, 'get_model_type'):
                # It's already an adapter
                adapter = model
            else:
                # It's raw model data, get adapter
                adapter = get_adapter_for_model(model)
            model_types.add(adapter.get_model_type())
        
        # Create individual plots for each model type
        for model_type in sorted(model_types):
            print(f"Creating CV RMSE distribution plot for {model_type}...")
            fig = self.plot_cv_rmse_distribution(model_types=[model_type])
            if fig:
                figures[f'{model_type.lower().replace(" ", "_")}_cv_distribution'] = fig
        
        # Save figures if requested
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
                from src.config import settings
                
                # Use default output directory
                output_dir = settings.VISUALIZATION_DIR / "performance" / "cv_distributions"
            
            # Ensure directory exists
            ensure_dir(output_dir)
            
            # Save each figure
            for name, fig in figures.items():
                save_figure(
                    fig=fig,
                    filename=name,
                    output_dir=output_dir,
                    dpi=self.config.get('dpi', 300),
                    format=self.config.get('format', 'png')
                )
        
        return figures


def plot_cv_distributions(
    models: List[ModelData],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    Create CV distribution visualizations for given models.
    
    Args:
        models: List of ModelData objects
        config: Visualization configuration
        
    Returns:
        Dict[str, plt.Figure]: Dictionary of created figures
    """
    plot = CVDistributionPlot(models, config)
    return plot.plot()