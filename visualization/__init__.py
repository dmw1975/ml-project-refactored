"""Visualization module for ML models and results (DEPRECATED).

This module is deprecated and will be removed in a future version.
Please use the visualization_new package instead.
"""

import warnings

warnings.warn(
    "The visualization module is deprecated. Please use visualization_new instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from old modules for backward compatibility
from visualization.metrics_plots import plot_model_comparison, plot_residuals, plot_statistical_tests_filtered
from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model, plot_feature_correlations
from visualization.style import setup_visualization_style, save_figure

# Define what this package exports
__all__ = [
    'plot_model_comparison', 'plot_residuals', 'plot_statistical_tests_filtered',
    'plot_top_features', 'plot_feature_importance_by_model', 'plot_feature_correlations',
    'setup_visualization_style', 'save_figure'
]

# Import modules for backward compatibility
from visualization import metrics_plots, feature_plots, style

# Note: Future versions will import from visualization_new for full compatibility