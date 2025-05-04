# Visualization Code Cleanup Plan

This document outlines the plan to clean up redundant visualization code after implementing the new model-agnostic visualization architecture, with special attention to directory structure issues and removing unwanted CatBoost_* directories.

## Goals

1. Remove redundant code to improve maintainability
2. Eliminate CatBoost_* directories while preserving the preferred type-based directory structure
3. Standardize on the new visualization architecture (`visualization_new`) 
4. Reduce confusion by clearly deprecating old modules
5. Ensure backward compatibility during transition
6. Provide clear guidelines for using the new architecture
7. Ensure future enhancements only need to be implemented once

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

## Phase 0: Directory Structure Cleanup (Immediate Priority)

### 1. Fix CatBoost_* Directory Generation

The current issue is in `visualization_new/viz_factory.py` lines 290-291, which creates model-name based directories:

```python
# Use default from settings
output_dir = settings.VISUALIZATION_DIR / model_name
```

This needs to be updated to use type-based directories instead:

```python
# Add helper function for consistent directory structure
def get_visualization_dir(model_name, plot_type):
    """Return standardized directory path for visualizations."""
    from config import settings
    # Convert model name to lowercase for consistency
    return settings.VISUALIZATION_DIR / plot_type / model_name.lower()

# Then use this function for different visualization types:
residuals_dir = get_visualization_dir(model_name, "residuals")
features_dir = get_visualization_dir(model_name, "features")
```

### 2. Modify `visualize_all_models()` Function

In the `visualize_all_models()` function (line 353), change:

```python
# Current code
plots = visualize_model(
    model_data=model_data,
    output_dir=output_dir / model_name if output_dir else None,
    format=format,
    dpi=dpi,
    show=show
)
```

To:

```python
# Updated code - let visualize_model handle directory structure
plots = visualize_model(
    model_data=model_data,
    output_dir=None,  # Let visualize_model use type-based directories
    format=format,
    dpi=dpi,
    show=show
)
```

### 3. Remove Existing CatBoost_* Directories

Create a cleanup script to safely remove CatBoost_* directories:

```python
import os
import shutil
from pathlib import Path
from config import settings

# Directory to clean
viz_dir = settings.VISUALIZATION_DIR

# Find all CatBoost_* directories
catboost_dirs = list(viz_dir.glob("CatBoost_*"))

# Output information
print(f"Found {len(catboost_dirs)} CatBoost directories to remove:")
for d in catboost_dirs:
    print(f" - {d}")

# Ask for confirmation
if input("Proceed with removal? (y/n): ").lower() == 'y':
    for d in catboost_dirs:
        shutil.rmtree(d)
        print(f"Removed {d}")
else:
    print("Operation cancelled.")
```

## Implementation Plan

1. Start with fixing directory structure (Phase 0)
2. Add deprecation warnings to legacy code (Phase 1)
3. Update main.py and test scripts to use new architecture
4. Document the transition process
5. Schedule future removal of deprecated code

## Timeline

- **Immediate (Day 1-2)**: Implement Phase 0 (Directory Structure Cleanup)
- **Week 1-2**: Implement Phase 1 (Deprecation)
- **Week 3-4**: Implement Phase 2 (Cleanup)
- **Week 4-5**: Implement Phase 3 (Documentation)
- **3 Months Later**: Implement Phase 4 (Final Removal)