"""Core interfaces for the visualization package."""

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