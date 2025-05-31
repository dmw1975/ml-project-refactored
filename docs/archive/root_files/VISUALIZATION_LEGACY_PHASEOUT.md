# Legacy Visualization Phase-Out Plan

This document outlines the plan for completely phasing out the legacy visualization code in favor of the new architecture.

## Timeline

| Phase | Timeframe | Actions |
| ----- | --------- | ------- |
| 1. Deprecation | Current | Mark all legacy modules as deprecated with warnings |
| 2. Redirection | Current - 1 month | Implement import redirection to new architecture |
| 3. Legacy Mode | 1-2 months | Add `--use-legacy-viz` flag option only for emergencies |
| 4. Full Removal | 3 months | Remove legacy visualization code entirely |

## Phase 1: Deprecation (Current)

**Status: Complete**

All legacy visualization modules now include deprecation warnings:

```python
import warnings

warnings.warn(
    "This module is deprecated. Please use visualization_new package instead.",
    DeprecationWarning,
    stacklevel=2
)
```

## Phase 2: Redirection (Current - 1 month)

**Status: In Progress**

1. Update `visualization/__init__.py` to redirect imports to the new architecture:
   ```python
   from visualization_new import *
   
   # Re-export with legacy names
   plot_residuals = create_residual_plot
   plot_model_comparison = create_model_comparison_plot
   plot_feature_importance_by_model = create_feature_importance_plot
   ```

2. Update all legacy visualization modules to call their equivalents in the new architecture.

3. Update `main.py` to prioritize new visualization architecture for all flags.

## Phase 3: Legacy Mode (1-2 months)

**Status: Planned**

Add a `--use-legacy-viz` flag to `main.py` that explicitly opts in to using the legacy code:

```python
parser.add_argument('--use-legacy-viz', action='store_true',
                   help='Use legacy visualization code (DEPRECATED)')

if args.use_legacy_viz:
    warnings.warn(
        "Using legacy visualization code. This option will be removed in a future update.",
        DeprecationWarning,
        stacklevel=2
    )
    # Call legacy code
else:
    # Use new architecture (default)
```

This flag should only be used for emergency compatibility cases.

## Phase 4: Full Removal (3 months)

**Status: Planned**

1. Remove the `visualization/` directory entirely
2. Remove the `--use-legacy-viz` flag option
3. Update all scripts and notebooks to use only the new architecture
4. Update documentation to remove references to legacy visualization

## Transition Guidelines for Users

### How to Migrate

1. Replace imports:
   ```python
   # Old
   from visualization.feature_plots import plot_feature_importance_by_model
   
   # New
   from visualization_new import create_feature_importance_plot
   ```

2. Update function calls:
   ```python
   # Old
   plot_feature_importance_by_model(importance_results)
   
   # New
   create_feature_importance_plot(model_data)
   ```

3. Review directory structure:
   - Be aware that visualizations are now saved in type-based directories 
   - For example, feature importance plots are in `outputs/visualizations/features/`

### Command-Line Migration

1. Replace `--visualize` with `--visualize` (both now use the new architecture)
2. Replace model-specific visualization flags (these will continue to work but now use the new architecture)

## Implementation Notes

1. All visualizations should now be created using the `visualization_new` package.
2. The `visualization/` directory is maintained only for backward compatibility.
3. New features should only be added to the `visualization_new` package.
4. Documentation should focus exclusively on the new architecture.

## Impact Assessment

For the full phase-out to be successful, we need to ensure:

1. All features in the legacy code have equivalents in the new architecture
2. All scripts and notebooks using the legacy code are updated
3. All documentation referencing the legacy code is updated
4. Users are notified of the timeline through documentation and deprecation warnings