# Visualization Architecture Example Code

This document provides concrete examples of the code structure for the new visualization architecture.

## 1. Core Interfaces

```python
# visualization/core/interfaces.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union

class ModelData(ABC):
    """Abstract interface for model data extraction."""
    
    @abstractmethod
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test set predictions and actual values.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_true, y_pred)
        """
        pass
    
    @abstractmethod
    def get_residuals(self) -> np.ndarray:
        """
        Get model residuals.
        
        Returns:
            np.ndarray: Residuals array (y_true - y_pred)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance data.
        
        Returns:
            pd.DataFrame: DataFrame with feature importance data
                          (columns: Feature, Importance, Std)
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.
        
        Returns:
            Dict[str, float]: Dictionary of metrics (RMSE, MAE, R2, etc.)
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameters
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dict[str, Any]: Dictionary of metadata (model_name, dataset, etc.)
        """
        pass


class VisualizationConfig:
    """Configuration for visualizations."""
    
    def __init__(self, **kwargs):
        """
        Initialize visualization configuration.
        
        Args:
            **kwargs: Configuration parameters
        """
        self.config = {
            # Default configuration
            "figsize": (10, 6),
            "dpi": 300,
            "format": "png",
            "style": "whitegrid",
            "palette": "default",
            "output_dir": None,
            "show": False,
            "save": True,
            "title_fontsize": 14,
            "label_fontsize": 12,
            "tick_fontsize": 10,
            "legend_fontsize": 10,
            "annotation_fontsize": 10,
            "grid": True,
            "grid_alpha": 0.3,
        }
        # Update with provided configuration
        self.config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key (str): Configuration key
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def update(self, **kwargs) -> None:
        """
        Update configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
```

## 2. Model Adapters

```python
# visualization/adapters/xgboost_adapter.py

import numpy as np
import pandas as pd
from visualization.core.interfaces import ModelData

class XGBoostAdapter(ModelData):
    """Adapter for XGBoost models."""
    
    def __init__(self, model_data: dict):
        """
        Initialize XGBoost adapter.
        
        Args:
            model_data (dict): XGBoost model data dictionary
        """
        self.model_data = model_data
        self.model_name = model_data.get('model_name', 'Unknown')
        self.model = model_data.get('model', None)
    
    def get_predictions(self) -> tuple:
        """Get test set predictions and actual values."""
        y_test = self.model_data.get('y_test')
        y_pred = self.model_data.get('y_pred')
        
        # Convert to numpy arrays
        if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.flatten()
        else:
            y_test = np.array(y_test).flatten()
            
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values.flatten()
        else:
            y_pred = np.array(y_pred).flatten()
            
        return y_test, y_pred
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals."""
        y_test, y_pred = self.get_predictions()
        return y_test - y_pred
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance data."""
        if self.model is None:
            return pd.DataFrame(columns=['Feature', 'Importance', 'Std'])
        
        # Check if precomputed feature importance exists
        if 'feature_importance' in self.model_data:
            return self.model_data['feature_importance']
        
        # Extract feature importance from model
        importance = self.model.feature_importances_
        std = np.zeros_like(importance)
        
        # Get feature names
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Std': std
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        return df
    
    def get_metrics(self) -> dict:
        """Get model performance metrics."""
        metrics = {}
        
        for metric in ['RMSE', 'MAE', 'MSE', 'R2']:
            if metric in self.model_data:
                metrics[metric] = self.model_data[metric]
        
        # Calculate any missing metrics
        if 'RMSE' not in metrics and 'MSE' in metrics:
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            
        if 'MSE' not in metrics and 'RMSE' in metrics:
            metrics['MSE'] = metrics['RMSE'] ** 2
            
        if 'R2' not in metrics or 'MAE' not in metrics:
            y_test, y_pred = self.get_predictions()
            
            if 'R2' not in metrics:
                from sklearn.metrics import r2_score
                metrics['R2'] = r2_score(y_test, y_pred)
                
            if 'MAE' not in metrics:
                from sklearn.metrics import mean_absolute_error
                metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        
        return metrics
    
    def get_hyperparameters(self) -> dict:
        """Get model hyperparameters."""
        if self.model is None:
            return {}
            
        # Check if best_params exists (for Optuna)
        if 'best_params' in self.model_data:
            return self.model_data['best_params']
            
        # Extract hyperparameters from model
        params = self.model.get_params()
        return params
    
    def get_metadata(self) -> dict:
        """Get model metadata."""
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_data.get('model_type', 'XGBoost'),
        }
        
        # Add additional metadata if available
        for key in ['dataset', 'n_features', 'n_companies', 'n_companies_train', 'n_companies_test']:
            if key in self.model_data:
                metadata[key] = self.model_data[key]
        
        return metadata
```

## 3. Topic-Based Plot Module

```python
# visualization/plots/residuals.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from visualization.core.interfaces import ModelData, VisualizationConfig
from visualization.core.registry import get_adapter_for_model
from visualization.core.style import setup_visualization_style
from visualization.components.annotations import add_metrics_text
from visualization.utils.io import ensure_dir

def plot_residuals(model_data, config=None):
    """
    Create residual analysis plots for a model.
    
    Args:
        model_data: Model data dictionary or ModelData object
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle configuration
    if config is None:
        config = VisualizationConfig()
    elif not isinstance(config, VisualizationConfig):
        config = VisualizationConfig(**config)
    
    # Get model adapter
    if not isinstance(model_data, ModelData):
        model_data = get_adapter_for_model(model_data)
    
    # Extract prediction data
    y_true, y_pred = model_data.get_predictions()
    residuals = model_data.get_residuals()
    metrics = model_data.get_metrics()
    metadata = model_data.get_metadata()
    
    # Set up style
    style = setup_visualization_style(config.get('style'))
    plt.rcParams.update({'font.size': config.get('label_fontsize')})
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=config.get('figsize', (16, 12)))
    
    # 1. Predicted vs Actual (top left)
    ax = axes[0, 0]
    ax.scatter(y_pred, y_true, alpha=0.7, color=style.get('colors', {}).get('primary', '#3498db'))
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel('Predicted Value', fontsize=config.get('label_fontsize'))
    ax.set_ylabel('Actual Value', fontsize=config.get('label_fontsize'))
    ax.set_title('Predicted vs Actual Values', fontsize=config.get('title_fontsize'))
    
    if config.get('grid', True):
        ax.grid(alpha=config.get('grid_alpha', 0.3))
    
    # Add R² annotation
    r2 = metrics.get('R2', np.corrcoef(y_pred, y_true)[0, 1] ** 2)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=config.get('annotation_fontsize'), verticalalignment='top')
    
    # 2. Residuals vs Predicted (top right)
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.7, color=style.get('colors', {}).get('secondary', '#2ecc71'))
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Value', fontsize=config.get('label_fontsize'))
    ax.set_ylabel('Residual', fontsize=config.get('label_fontsize'))
    ax.set_title('Residuals vs Predicted Values', fontsize=config.get('title_fontsize'))
    
    if config.get('grid', True):
        ax.grid(alpha=config.get('grid_alpha', 0.3))
    
    # 3. Histogram of Residuals (bottom left)
    ax = axes[1, 0]
    sns.histplot(residuals, kde=True, ax=ax, color=style.get('colors', {}).get('tertiary', '#e67e22'))
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('Residual', fontsize=config.get('label_fontsize'))
    ax.set_ylabel('Density', fontsize=config.get('label_fontsize'))
    ax.set_title('Distribution of Residuals', fontsize=config.get('title_fontsize'))
    
    # Add mean and std annotation
    mean_res = residuals.mean()
    std_res = residuals.std()
    ax.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}', transform=ax.transAxes,
            fontsize=config.get('annotation_fontsize'), verticalalignment='top')
    
    # 4. Q-Q Plot (bottom right)
    ax = axes[1, 1]
    standardized_residuals = (residuals - mean_res) / std_res
    stats.probplot(standardized_residuals, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot of Standardized Residuals', fontsize=config.get('title_fontsize'))
    
    if config.get('grid', True):
        ax.grid(alpha=config.get('grid_alpha', 0.3))
    
    # Add metrics to figure
    metrics_text = (
        f"RMSE: {metrics.get('RMSE', np.sqrt(np.mean(residuals**2))):.4f}\n"
        f"MAE: {metrics.get('MAE', np.mean(np.abs(residuals))):.4f}\n"
        f"R²: {r2:.4f}\n"
        f"n_samples: {len(y_true)}"
    )
    
    # Add metrics to figure
    fig.text(0.5, 0.01, metrics_text, ha='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add title
    model_name = metadata.get('model_name', 'Unknown Model')
    plt.suptitle(f'Residual Analysis for {model_name}', fontsize=config.get('title_fontsize', 14) + 2, y=1.02)
    
    # Adjust layout
    plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.97])
    
    # Save if requested
    if config.get('save', True):
        output_dir = config.get('output_dir', Path('outputs/visualizations/residuals'))
        ensure_dir(output_dir)
        
        filename = f"{model_name}_residuals.{config.get('format', 'png')}"
        filepath = Path(output_dir) / filename
        
        fig.savefig(filepath, dpi=config.get('dpi', 300), bbox_inches='tight')
        print(f"Saved {filepath}")
    
    # Show if requested
    if config.get('show', False):
        plt.show()
        
    return fig
```

## 4. Factory Module for Easy Access

```python
# visualization/viz_factory.py

from pathlib import Path
from typing import Dict, List, Union, Any, Optional

from visualization.core.interfaces import ModelData, VisualizationConfig
from visualization.core.registry import get_adapter_for_model, load_model
from visualization.plots import residuals, features, metrics, comparative, optimization

def create_residual_plot(model_data, config=None):
    """
    Create residual analysis plot for a model.
    
    Args:
        model_data: Model data, name, or ModelData object
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
    
    return residuals.plot_residuals(model_data, config)

def create_feature_importance_plot(model_data, config=None):
    """
    Create feature importance plot for a model.
    
    Args:
        model_data: Model data, name, or ModelData object
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model name
    if isinstance(model_data, str):
        model_data = load_model(model_data)
        
    return features.plot_feature_importance(model_data, config)

def create_model_comparison_plot(models, metric='RMSE', config=None):
    """
    Create model comparison plot.
    
    Args:
        models: List of model data, names, or ModelData objects
        metric: Metric to compare by
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Handle string model names
    model_list = []
    for model in models:
        if isinstance(model, str):
            model_list.append(load_model(model))
        else:
            model_list.append(model)
            
    return comparative.plot_model_comparison(model_list, metric, config)

def create_all_residual_plots(config=None):
    """
    Create residual plots for all models.
    
    Args:
        config: Visualization configuration
        
    Returns:
        List[matplotlib.figure.Figure]: List of created figures
    """
    from visualization.utils.io import load_all_models
    
    # Load all models
    all_models = load_all_models()
    
    # Create plots
    figures = []
    for model_name, model_data in all_models.items():
        try:
            fig = create_residual_plot(model_data, config)
            figures.append(fig)
        except Exception as e:
            print(f"Error creating plot for {model_name}: {e}")
            
    return figures

def create_comparative_dashboard(models=None, metrics=None, config=None):
    """
    Create comparative dashboard with multiple plots.
    
    Args:
        models: List of model data, names, or ModelData objects
        metrics: List of metrics to include
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Use all models if not specified
    if models is None:
        from visualization.utils.io import load_all_models
        all_models = load_all_models()
        models = list(all_models.values())
    
    # Use default metrics if not specified
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'R2']
        
    return comparative.plot_dashboard(models, metrics, config)
```

## 5. Usage Examples

### Basic Usage

```python
from visualization import viz_factory

# Create residual plot for a specific model
viz_factory.create_residual_plot("XGB_Base_optuna")

# Create feature importance plot
viz_factory.create_feature_importance_plot("XGB_Base_optuna", config={
    "top_n": 15,
    "figsize": (12, 8)
})

# Create model comparison
viz_factory.create_model_comparison_plot([
    "XGB_Base_optuna",
    "LightGBM_Base_optuna",
    "CatBoost_Base_optuna"
], metric="RMSE")

# Create all residual plots
viz_factory.create_all_residual_plots()

# Create a comprehensive dashboard
viz_factory.create_comparative_dashboard()
```

### Advanced Usage with Configuration

```python
from visualization.core.interfaces import VisualizationConfig
from visualization.plots import residuals, features, comparative
from visualization.core.registry import load_model

# Create custom configuration
config = VisualizationConfig(
    figsize=(14, 10),
    dpi=600,
    format="pdf",
    style="darkgrid",
    palette="viridis",
    output_dir="outputs/thesis_figures",
    title_fontsize=16,
    label_fontsize=14,
    annotation_fontsize=12,
    grid_alpha=0.2
)

# Load model
model = load_model("XGB_Base_optuna")

# Create highly customized residual plot
fig = residuals.plot_residuals(model, config)

# Create feature importance with custom parameters
config.update(top_n=20, sort_by="importance")
fig = features.plot_feature_importance(model, config)

# Create custom comparison
models = [
    load_model("XGB_Base_optuna"),
    load_model("LightGBM_Base_optuna"),
    load_model("CatBoost_Base_optuna")
]
config.update(
    metrics=["RMSE", "MAE", "R2"],
    group_by="dataset",
    highlight_best=True,
    show_improvement=True
)
fig = comparative.plot_model_comparison(models, config)
```

These examples demonstrate the flexibility and power of the new visualization architecture, showing how it can be used for both simple and complex visualization tasks.