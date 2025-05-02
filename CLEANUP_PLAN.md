# Visualization Code Cleanup Plan

This document outlines the plan to clean up redundant visualization code after implementing the new model-agnostic visualization architecture.

## Goals

1. Remove redundant code to improve maintainability
2. Reduce confusion by clearly deprecating old modules
3. Ensure backward compatibility during transition
4. Provide clear guidelines for using the new architecture

## Phase 1: Deprecation

### 1. Deprecate old visualization modules

Add deprecation warnings to the old visualization modules:

- `visualization/metrics_plots.py`
- `visualization/feature_plots.py`
- `visualization/xgboost_plots.py`
- `visualization/lightgbm_plots.py`
- `visualization/sector_plots.py`
- `visualization/create_residual_plots.py`

Example deprecation notice:
```python
import warnings

warnings.warn(
    "This module is deprecated. Please use the new visualization_new package instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 2. Update old visualization module imports

Modify the `visualization/__init__.py` to import from the new architecture:

```python
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

# Import from new architecture for backward compatibility
from visualization_new import (
    create_residual_plot,
    create_feature_importance_plot,
    create_model_comparison_plot,
    create_all_residual_plots,
    create_comparative_dashboard,
    create_metrics_table
)

# Re-export with old names for backward compatibility
plot_residuals = create_residual_plot
plot_model_comparison = create_model_comparison_plot
plot_feature_importance_by_model = create_feature_importance_plot
```

## Phase 2: Cleanup

### 1. Update main.py to use new architecture

Remove old visualization imports and use the new architecture:

```python
# Old
# from visualization.metrics_plots import plot_model_comparison, plot_residuals
# from visualization.feature_plots import plot_top_features

# New
from visualization_new import (
    create_model_comparison_plot as plot_model_comparison,
    create_residual_plot as plot_residuals,
    create_feature_importance_plot as plot_top_features
)
```

### 2. Remove unused imports from modules

Check for unused imports in all modules and remove them:

- `import numpy as np` if numpy is not used
- `import pandas as pd` if pandas is not used
- `import matplotlib.pyplot as plt` if plt is not used

### 3. Identify and remove dead code

Identify functions that are no longer called:

- Use static analysis tools to find unused functions
- Remove functions that are completely replaced by the new architecture

## Phase 3: Documentation and Transition

### 1. Update documentation

Create clear documentation on how to use the new visualization architecture:

- Add examples for each visualization type
- Document the model adapter system
- Explain the benefits of the new architecture

### 2. Create transition guide

Create a transition guide for developers still using the old architecture:

- Map old functions to new functions
- Explain how to adapt existing code
- Provide examples of before and after

## Phase 4: Final Removal (Future)

After a transition period (e.g., 2-3 months):

1. Remove the deprecated modules
2. Remove backward compatibility shims
3. Update all references in the codebase to use new architecture

## Implementation Plan

1. Start with adding deprecation warnings
2. Update main.py and test scripts to use new architecture
3. Document the transition process
4. Schedule future removal of deprecated code

## Timeline

- **Week 1-2**: Implement Phase 1 (Deprecation)
- **Week 3-4**: Implement Phase 2 (Cleanup)
- **Week 4-5**: Implement Phase 3 (Documentation)
- **3 Months Later**: Implement Phase 4 (Final Removal)