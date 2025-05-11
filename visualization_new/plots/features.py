"""Feature importance plots for all model types."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import ModelViz, ComparativeViz
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.components.annotations import add_statistics_text
from visualization_new.components.layouts import create_grid_layout
from visualization_new.components.formats import format_figure_for_export, save_figure


class FeatureImportancePlot(ModelViz):
    """Feature importance plot for a model."""
    
    def __init__(
        self, 
        model_data: Union[ModelData, Dict[str, Any]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize feature importance plot.
        
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
        Create feature importance plot.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Get feature importance data
        importance_df = self.model_data.get_feature_importance()
        
        # Get model metadata
        metadata = self.model_data.get_metadata()
        model_name = metadata.get('model_name', 'Unknown Model')
        
        # Determine number of features to show
        top_n = self.config.get('top_n', 15)
        
        # Select top features
        top_features = importance_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (10, 8)))
        
        # Plot horizontal bar chart
        bars = ax.barh(
            top_features['Feature'][::-1],  # Reverse to have highest at the top
            top_features['Importance'][::-1],
            xerr=top_features['Std'][::-1] if self.config.get('show_error', True) else None,
            color=self.style.get('colors', {}).get('primary', '#3498db'),  # Use standard blue color
            alpha=0.8,  # Slightly increased alpha for better visibility
            error_kw={'ecolor': 'gray', 'capsize': 5} if self.config.get('show_error', True) else None
        )
        
        # Add value labels if requested
        if self.config.get('show_values', True):
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x = max(width + 0.01, 0.01)  # Handle negative importances
                ax.text(
                    label_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.4f}",
                    va='center',
                    fontsize=self.config.get('annotation_fontsize', 10)
                )
        
        # Set plot properties
        ax.set_xlabel('Importance', fontsize=self.config.get('label_fontsize', 12))
        ax.set_title(f"Top {top_n} Features for {model_name}", fontsize=self.config.get('title_fontsize', 14))
        
        # Add grid
        if self.config.get('grid', True):
            ax.grid(axis='x', alpha=self.config.get('grid_alpha', 0.3))
        
        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Format figure for export if requested
        if self.config.get('format_for_export', False):
            fig = format_figure_for_export(
                fig=fig,
                tight_layout=True,
                theme=self.config.get('theme', 'default')
            )
        else:
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
                output_dir = settings.VISUALIZATION_DIR / "features"
            
            # Save figure
            save_figure(
                fig=fig,
                filename=f"{model_name}_top_features",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig


class FeatureImportanceComparisonPlot(ComparativeViz):
    """Feature importance comparison plot for multiple models."""
    
    def __init__(
        self, 
        models: List[Union[ModelData, Dict[str, Any]]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize feature importance comparison plot.
        
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
        Create feature importance comparison plot.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Determine number of features to show
        top_n = self.config.get('top_n', 20)
        
        # Get all feature importance data and combine
        all_features = set()
        importance_by_model = {}
        
        for model in self.models:
            # Get model metadata
            metadata = model.get_metadata()
            model_name = metadata.get('model_name', 'Unknown Model')
            
            # Get feature importance
            importance_df = model.get_feature_importance()
            
            # Add to importance by model
            importance_by_model[model_name] = importance_df
            
            # Add to all features
            all_features.update(importance_df['Feature'])
        
        # Create consolidated DataFrame
        consolidated = pd.DataFrame(index=list(all_features))
        
        # Add importance for each model
        for model_name, importance_df in importance_by_model.items():
            # Convert to dictionary for easier lookup
            importance_dict = dict(zip(importance_df['Feature'], importance_df['Importance']))
            
            # Add to consolidated DataFrame
            consolidated[model_name] = consolidated.index.map(lambda x: importance_dict.get(x, 0))
        
        # Add average importance
        consolidated['avg_importance'] = consolidated.mean(axis=1)
        
        # Sort by average importance
        consolidated = consolidated.sort_values('avg_importance', ascending=False)
        
        # Select top features
        top_df = consolidated.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 10)))
        
        # Plot horizontal bar chart for average importance
        bars = ax.barh(
            top_df.index[::-1],  # Reverse to have highest at the top
            top_df['avg_importance'][::-1],
            color=self.style.get('colors', {}).get('primary', '#3498db'),
            alpha=0.7
        )
        
        # Add value labels if requested
        if self.config.get('show_values', True):
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x = max(width + 0.01, 0.01)  # Handle negative importances
                ax.text(
                    label_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.4f}",
                    va='center',
                    fontsize=self.config.get('annotation_fontsize', 10)
                )
        
        # Set plot properties
        ax.set_xlabel('Average Importance', fontsize=self.config.get('label_fontsize', 12))
        ax.set_title(f"Top {top_n} Features by Average Importance Across Models", fontsize=self.config.get('title_fontsize', 14))
        
        # Add grid
        if self.config.get('grid', True):
            ax.grid(axis='x', alpha=self.config.get('grid_alpha', 0.3))
        
        # Format figure for export if requested
        if self.config.get('format_for_export', False):
            fig = format_figure_for_export(
                fig=fig,
                tight_layout=True,
                theme=self.config.get('theme', 'default')
            )
        else:
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

                # Use default from settings - ALWAYS save to main features directory only
                output_dir = settings.VISUALIZATION_DIR / "features"
            else:
                # Check if this is a model-specific subdirectory
                output_dir_str = str(output_dir).lower()
                for model_dir in ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear']:
                    if model_dir in output_dir_str:
                        # If we're in a model-specific subdirectory, use the parent directory
                        # to avoid saving top_N_features_avg_importance.png in model subdirectories
                        from config import settings
                        output_dir = settings.VISUALIZATION_DIR / "features"
                        print(f"Redirecting top_{top_n}_features_avg_importance.png to main features directory")
                        break

            # Save figure
            save_figure(
                fig=fig,
                filename=f"top_{top_n}_features_avg_importance",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Check if we should create heatmap
        # For LightGBM models, we'll set create_heatmap to False by default
        # This can be overridden by explicitly setting create_heatmap in the config
        create_heatmap = self.config.get('create_heatmap')
        if create_heatmap is None:  # Not explicitly set
            # Check if this is for LightGBM models based on output directory
            output_dir = self.config.get('output_dir')
            if output_dir and 'lightgbm' in str(output_dir).lower():
                # Skip heatmap for LightGBM by default
                create_heatmap = False
            else:
                # Default to True for other models
                create_heatmap = True
        
        # No longer creating heatmap visualization
        print("Skipping heatmap creation - this output is no longer needed")
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def _create_heatmap(self, feature_df: pd.DataFrame) -> plt.Figure:
        """
        Create heatmap of feature importance across models.
        
        Args:
            feature_df: DataFrame of feature importance by model
            
        Returns:
            matplotlib.figure.Figure: Heatmap figure
        """
        # Check if this is for LightGBM models based on output directory
        output_dir = self.config.get('output_dir')
        if output_dir:
            # Convert to string to handle both string and Path objects
            output_dir_str = str(output_dir)
            # Skip heatmap creation for LightGBM models
            if 'lightgbm' in output_dir_str.lower():
                print("Skipping heatmap creation for LightGBM models as it's not needed")
                # Return empty figure to maintain API compatibility
                return plt.figure()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 12)))
        
        # Create heatmap
        sns.heatmap(
            feature_df,
            cmap=self.style.get('color_maps', {}).get('blue_gradient', 'Blues'),
            annot=self.config.get('show_values', True),
            fmt='.3f',
            linewidths=0.5,
            ax=ax
        )
        
        # Set plot properties
        plt.title('Top Features Importance Across Models', fontsize=self.config.get('title_fontsize', 14))
        plt.ylabel('Features')
        plt.xlabel('Models')
        plt.xticks(rotation=45, ha='right')
        
        # Format figure for export if requested
        if self.config.get('format_for_export', False):
            fig = format_figure_for_export(
                fig=fig,
                tight_layout=True,
                theme=self.config.get('theme', 'default')
            )
        else:
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

                # Use default from settings - ALWAYS save to main features directory
                output_dir = settings.VISUALIZATION_DIR / "features"
            else:
                # Check if this is a model-specific subdirectory
                output_dir_str = str(output_dir).lower()
                for model_dir in ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear']:
                    if model_dir in output_dir_str:
                        # If we're in a model-specific subdirectory, use the parent directory
                        # to avoid saving top_features_heatmap.png in model subdirectories
                        from config import settings
                        output_dir = settings.VISUALIZATION_DIR / "features"
                        print(f"Redirecting top_features_heatmap.png to main features directory")
                        break

            # Save figure
            save_figure(
                fig=fig,
                filename="top_features_heatmap",
                output_dir=output_dir,
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig


def plot_feature_importance(
    model_data: Union[ModelData, Dict[str, Any]], 
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> plt.Figure:
    """
    Create feature importance plot for a model.
    
    Args:
        model_data: Model data or adapter
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot = FeatureImportancePlot(model_data, config)
    return plot.plot()


def plot_feature_importance_comparison(
    models: List[Union[ModelData, Dict[str, Any]]],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> plt.Figure:
    """
    Create feature importance comparison plot for multiple models.
    
    Args:
        models: List of model data or adapters
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    plot = FeatureImportanceComparisonPlot(models, config)
    return plot.plot()