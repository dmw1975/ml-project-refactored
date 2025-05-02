"""Data preparation utilities for visualization."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple

def standardize_array(arr: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
    """
    Standardize array to numpy array.
    
    Args:
        arr: Array to standardize
        
    Returns:
        np.ndarray: Standardized array
    """
    if isinstance(arr, pd.Series):
        return arr.values
    elif isinstance(arr, list):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        try:
            return np.array(arr)
        except:
            raise ValueError(f"Cannot convert {type(arr)} to numpy array")

def prepare_prediction_data(y_true: Union[np.ndarray, pd.Series, List], 
                           y_pred: Union[np.ndarray, pd.Series, List]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare prediction data for visualization.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Standardized true and predicted values
    """
    # Standardize arrays
    y_true = standardize_array(y_true)
    y_pred = standardize_array(y_pred)
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs. y_pred {y_pred.shape}")
    
    return y_true, y_pred

def prepare_residuals(y_true: Union[np.ndarray, pd.Series, List], 
                     y_pred: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
    """
    Prepare residuals for visualization.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        np.ndarray: Residuals
    """
    # Standardize data
    y_true, y_pred = prepare_prediction_data(y_true, y_pred)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    return residuals

def prepare_feature_importance(feature_importance: Union[Dict[str, float], pd.DataFrame, np.ndarray],
                              feature_names: Optional[List[str]] = None,
                              top_n: Optional[int] = None,
                              sort_by: str = 'importance') -> pd.DataFrame:
    """
    Prepare feature importance data for visualization.
    
    Args:
        feature_importance: Feature importance data
        feature_names: Feature names (required if feature_importance is array)
        top_n: Number of top features to include
        sort_by: Sort by column
        
    Returns:
        pd.DataFrame: Standardized feature importance DataFrame
    """
    # Convert to DataFrame
    if isinstance(feature_importance, pd.DataFrame):
        df = feature_importance.copy()
    elif isinstance(feature_importance, dict):
        df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        })
    elif isinstance(feature_importance, np.ndarray):
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
    else:
        raise ValueError(f"Unsupported feature importance type: {type(feature_importance)}")
    
    # Ensure required columns
    required_columns = ['Feature', 'Importance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add Std column if not present
    if 'Std' not in df.columns:
        df['Std'] = 0.0
    
    # Sort by importance
    if sort_by == 'importance':
        df = df.sort_values('Importance', ascending=False)
    elif sort_by == 'feature':
        df = df.sort_values('Feature')
    
    # Select top_n
    if top_n is not None and top_n > 0:
        df = df.head(top_n)
    
    return df

def prepare_metrics_for_comparison(metrics_list: List[Dict[str, float]], 
                                 model_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare metrics for comparison visualization.
    
    Args:
        metrics_list: List of metrics dictionaries
        model_names: Model names
        
    Returns:
        pd.DataFrame: Metrics DataFrame for comparison
    """
    # Create model names if not provided
    if model_names is None:
        model_names = [f'Model_{i+1}' for i in range(len(metrics_list))]
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Add model_name column
    metrics_df['model_name'] = model_names
    
    return metrics_df