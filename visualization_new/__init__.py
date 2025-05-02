"""Unified visualization framework for ML project.

This package provides a model-agnostic visualization framework
for creating standardized plots across different ML model types.
"""

from visualization_new.viz_factory import (
    create_residual_plot,
    create_feature_importance_plot,
    create_model_comparison_plot,
    create_all_residual_plots,
    create_comparative_dashboard,
    create_metrics_table
)

# Import sector visualization functions
from visualization_new.plots.sectors import (
    plot_sector_performance,
    plot_sector_metrics_table,
    visualize_all_sector_plots
)

__all__ = [
    'create_residual_plot',
    'create_feature_importance_plot',
    'create_model_comparison_plot',
    'create_all_residual_plots',
    'create_comparative_dashboard',
    'create_metrics_table',
    'plot_sector_performance',
    'plot_sector_metrics_table',
    'visualize_all_sector_plots'
]
