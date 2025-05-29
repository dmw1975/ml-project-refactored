"""Topic-based visualization modules.

Each module focuses on a specific visualization topic
(residuals, features, metrics, etc.) rather than a specific model type.
"""

from visualization_new.plots.residuals import (
    plot_residuals,
    plot_all_residuals
)

from visualization_new.plots.features import (
    plot_feature_importance,
    plot_feature_importance_comparison
)

from visualization_new.plots.metrics import (
    plot_metrics,
    plot_metrics_table,
    plot_model_comparison
)

from visualization_new.plots.cv_distributions import (
    plot_cv_distributions,
    CVDistributionPlot
)

__all__ = [
    'plot_residuals',
    'plot_all_residuals',
    'plot_feature_importance',
    'plot_feature_importance_comparison',
    'plot_metrics',
    'plot_metrics_table',
    'plot_model_comparison',
    'plot_cv_distributions',
    'CVDistributionPlot'
]
