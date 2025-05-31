"""Statistical utilities for visualization."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from scipy import stats

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for data.
    
    Args:
        data: Data array
        
    Returns:
        Dict[str, float]: Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75)
    }

def calculate_residual_statistics(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for residuals.
    
    Args:
        residuals: Residuals array
        
    Returns:
        Dict[str, Any]: Dictionary of residual statistics
    """
    stats_dict = calculate_statistics(residuals)
    
    # Add additional residual statistics
    stats_dict.update({
        'mse': np.mean(residuals ** 2),
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'mae': np.mean(np.abs(residuals)),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'normality_test': stats.shapiro(residuals)
    })
    
    return stats_dict

def calculate_confidence_intervals(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for data.
    
    Args:
        data: Data array
        confidence: Confidence level
        
    Returns:
        Tuple[float, float]: Lower and upper bounds of confidence interval
    """
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2, n-1)
    return m - h, m + h

def perform_statistical_tests(model1_predictions: np.ndarray, 
                                model2_predictions: np.ndarray,
                                true_values: np.ndarray,
                                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Statistically compare two models.
    
    Args:
        model1_predictions: Predictions from model 1
        model2_predictions: Predictions from model 2
        true_values: True values
        alpha: Significance level
        
    Returns:
        Dict[str, Any]: Dictionary of comparison results
    """
    # Calculate residuals
    residuals1 = true_values - model1_predictions
    residuals2 = true_values - model2_predictions
    
    # Calculate squared errors for each prediction
    squared_errors1 = residuals1 ** 2
    squared_errors2 = residuals2 ** 2
    
    # Paired t-test on squared errors
    t_stat, p_value = stats.ttest_rel(squared_errors1, squared_errors2)
    
    # Determine which model is better
    mean_sq_error1 = np.mean(squared_errors1)
    mean_sq_error2 = np.mean(squared_errors2)
    
    better_model = 1 if mean_sq_error1 < mean_sq_error2 else 2
    
    # Determine if difference is significant
    significant = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'better_model': better_model,
        'mse_model1': mean_sq_error1,
        'mse_model2': mean_sq_error2,
        'mean_difference': mean_sq_error1 - mean_sq_error2,
        'percent_improvement': (1 - mean_sq_error1 / mean_sq_error2) * 100 if better_model == 1 else (1 - mean_sq_error2 / mean_sq_error1) * 100
    }