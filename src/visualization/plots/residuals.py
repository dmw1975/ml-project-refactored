"""Residual analysis plots for all model types."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
from pathlib import Path
from scipy import stats
from typing import Dict, Any, Optional, Union, List, Tuple

from src.visualization.core.interfaces import ModelData, VisualizationConfig
from src.visualization.core.base import ModelViz
from src.visualization.core.registry import get_adapter_for_model
from src.visualization.components.annotations import add_statistics_text, add_metrics_text
from src.visualization.components.layouts import create_grid_layout
from src.visualization.components.formats import format_figure_for_export, save_figure
from src.visualization.utils.statistics import calculate_residual_statistics


class ResidualPlot(ModelViz):
    """Residual analysis plot for a model."""
    
    def __init__(
        self, 
        model_data: Union[ModelData, Dict[str, Any]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize residual plot.
        
        Args:
            model_data: Model data or adapter
            config: Visualization configuration
        """
        # Convert model data to adapter if needed
        if not isinstance(model_data, ModelData):
            model_data = get_adapter_for_model(model_data)
            
        # Call parent constructor
        super().__init__(model_data, config)
        
    def plot(self) -> plt.Figure:
        """
        Create residual analysis plot.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Close any existing figures to avoid conflicts
        plt.close('all')
        
        # Get prediction data
        y_true, y_pred = self.model_data.get_predictions()
        residuals = self.model_data.get_residuals()
        
        # Get model metadata
        metadata = self.model_data.get_metadata()
        model_name = metadata.get('model_name', 'Unknown Model')
        
        # Calculate residual statistics
        residual_stats = calculate_residual_statistics(residuals)
        
        # Get metrics
        metrics = self.model_data.get_metrics()
        
        # Create figure
        fig, axes = create_grid_layout(
            nrows=2,
            ncols=2,
            figsize=self.config.get('figsize', (16, 12)),
            suptitle=f"Residual Analysis for {model_name}",
            suptitle_fontsize=self.config.get('title_fontsize', 16) + 2
        )
        
        # 1. Predicted vs Actual (top left)
        ax = axes[0]
        ax.scatter(y_pred, y_true, alpha=0.7, color=self.style.get('colors', {}).get('primary', '#3498db'))
        min_val = min(y_pred.min(), y_true.min())
        max_val = max(y_pred.max(), y_true.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_xlabel('Predicted Value', fontsize=self.config.get('label_fontsize', 12))
        ax.set_ylabel('Actual Value', fontsize=self.config.get('label_fontsize', 12))
        ax.set_title('Predicted vs Actual Values', fontsize=self.config.get('title_fontsize', 14))
        
        if self.config.get('grid', True):
            ax.grid(alpha=self.config.get('grid_alpha', 0.3))
        
        # Add R² annotation
        r2 = metrics.get('R2', np.corrcoef(y_pred, y_true)[0, 1] ** 2)
        add_statistics_text(
            ax=ax,
            statistics={'R²': r2},
            position=(0.05, 0.95),
            fontsize=self.config.get('annotation_fontsize', 10)
        )
        
        # 2. Residuals vs Predicted (top right)
        ax = axes[1]
        ax.scatter(y_pred, residuals, alpha=0.7, color=self.style.get('colors', {}).get('secondary', '#2ecc71'))
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted Value', fontsize=self.config.get('label_fontsize', 12))
        ax.set_ylabel('Residual', fontsize=self.config.get('label_fontsize', 12))
        ax.set_title('Residuals vs Predicted Values', fontsize=self.config.get('title_fontsize', 14))
        
        if self.config.get('grid', True):
            ax.grid(alpha=self.config.get('grid_alpha', 0.3))
        
        # 3. Histogram of Residuals (bottom left)
        ax = axes[2]
        if sns is not None:
            sns.histplot(residuals, kde=True, ax=ax, color=self.style.get('colors', {}).get('tertiary', '#e67e22'))
        else:
            # Fallback to matplotlib histogram
            n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, 
                                     color=self.style.get('colors', {}).get('tertiary', '#e67e22'))
            # Add KDE line manually
            kde = stats.gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            ax.plot(x_range, kde(x_range), 'k-', lw=2)
            
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Residual', fontsize=self.config.get('label_fontsize', 12))
        ax.set_ylabel('Density', fontsize=self.config.get('label_fontsize', 12))
        ax.set_title('Distribution of Residuals', fontsize=self.config.get('title_fontsize', 14))
        
        # Add mean and std annotation
        mean_res = residual_stats['mean']
        std_res = residual_stats['std']
        add_statistics_text(
            ax=ax,
            statistics={'Mean': mean_res, 'Std': std_res},
            position=(0.05, 0.95),
            fontsize=self.config.get('annotation_fontsize', 10)
        )
        
        # 4. Q-Q Plot (bottom right)
        ax = axes[3]
        standardized_residuals = (residuals - mean_res) / std_res
        stats.probplot(standardized_residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot of Standardized Residuals', fontsize=self.config.get('title_fontsize', 14))
        
        if self.config.get('grid', True):
            ax.grid(alpha=self.config.get('grid_alpha', 0.3))
        
        # Add metrics to figure
        metrics_text = {
            'RMSE': metrics.get('RMSE', np.sqrt(residual_stats['mse'])),
            'MAE': metrics.get('MAE', residual_stats['mae']),
            'R²': r2,
            'n_samples': len(y_true)
        }
        
        # Add metrics annotation
        add_metrics_text(
            fig=fig,
            metrics=metrics_text,
            position=(0.5, 0.01),
            fontsize=self.config.get('annotation_fontsize', 10)
        )
        
        # Format figure for export if requested
        if self.config.get('format_for_export', False):
            fig = format_figure_for_export(
                fig=fig,
                tight_layout=True,
                pad=2.0,
                theme=self.config.get('theme', 'default')
            )
        else:
            # Adjust layout
            plt.tight_layout(pad=2.0, rect=[0, 0.05, 1, 0.95])
        
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
                output_dir = settings.VISUALIZATION_DIR / "residuals"
            
            # Save figure
            save_figure(
                fig=fig,
                filename=f"{model_name}_residuals",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig


def plot_residuals(
    model_data: Union[ModelData, Dict[str, Any]], 
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> plt.Figure:
    """
    Create residual analysis plot for a model.
    
    Args:
        model_data: Model data or adapter
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Wrap in adapter if needed
    if isinstance(model_data, dict):
        from src.visualization.core.registry import get_adapter_for_model
        model_data = get_adapter_for_model(model_data)
    
    plot = ResidualPlot(model_data, config)
    return plot.plot()


def plot_all_residuals(
    models: Optional[Union[List[Union[ModelData, Dict[str, Any]]], Dict[str, Union[ModelData, Dict[str, Any]]]]] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> List[plt.Figure]:
    """
    Create residual analysis plots for all models.
    
    Args:
        models: List or dictionary of models or adapters (if None, load all models)
        config: Visualization configuration
        
    Returns:
        List[matplotlib.figure.Figure]: List of created figures
    """
    # Load all models if None
    if models is None:
        from src.visualization.utils.io import load_all_models
        models = load_all_models()
    
    # Convert list to dictionary if needed
    if isinstance(models, list):
        models_dict = {}
        for i, model in enumerate(models):
            if isinstance(model, ModelData):
                model_name = model.get_metadata().get('model_name', f'Model_{i+1}')
                models_dict[model_name] = model
            else:
                models_dict[model.get('model_name', f'Model_{i+1}')] = model
        models = models_dict
    
    # Create plots
    figures = []
    for model_name, model_data in models.items():
        try:
            # Create plot
            fig = plot_residuals(model_data, config)
            figures.append(fig)
        except Exception as e:
            print(f"Error creating residual plot for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return figures