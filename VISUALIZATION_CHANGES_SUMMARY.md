# Visualization Changes Summary

This document summarizes the changes made to fix the directory structure issue with CatBoost_* directories.

## Problem

The visualization_new architecture was creating two types of directories:
1. Type-based structure: `features/catboost/`, `performance/catboost/`, etc. (preferred)
2. Model-name directories: `CatBoost_Base_basic/`, `CatBoost_Base_optuna/`, etc. (unwanted)

## Solution

We modified the visualization_new architecture to consistently use the type-based directory structure and avoid creating model-name directories.

### Key Changes

1. Added a helper function to provide consistent directory paths:
   ```python
   def get_visualization_dir(model_name: str, plot_type: str) -> Path:
       """
       Return standardized directory path for visualizations.
       
       Args:
           model_name: Name of the model
           plot_type: Type of visualization (features, residuals, performance, etc.)
           
       Returns:
           Path: Path to the visualization directory
       """
       # ...
       # Convert model name to lowercase for consistency
       model_name = model_name.lower()
       
       # Create and return the path
       output_dir = settings.VISUALIZATION_DIR / plot_type / model_name
       output_dir.mkdir(parents=True, exist_ok=True)
       
       return output_dir
   ```

2. Modified the `visualize_model` function to use type-based directories:
   ```python
   # Instead of this (old code):
   output_dir = settings.VISUALIZATION_DIR / model_name  # Creates CatBoost_* directories
   
   # Now using plot-type specific directories:
   residuals_dir = get_visualization_dir(model_name, "residuals")
   features_dir = get_visualization_dir(model_name, "features")
   ```

3. Updated the `visualize_all_models` function to use consistent directories:
   ```python
   # Instead of this (old code):
   plots = visualize_model(
       model_data=model_data,
       output_dir=output_dir / model_name if output_dir else None,
       format=format,
       dpi=dpi,
       show=show
   )
   
   # Now using:
   plots = visualize_model(
       model_data=model_data,
       output_dir=None,  # Let visualize_model use type-based directories
       format=format,
       dpi=dpi,
       show=show
   )
   ```

4. Created a cleanup script to remove existing CatBoost_* directories: `cleanup_catboost_directories.py`

## Directory Structure

### Before

```
/outputs/visualizations/
├── CatBoost_Base_basic/       # Unwanted
├── CatBoost_Base_optuna/      # Unwanted
├── features/
│   ├── catboost/
│   ├── lightgbm/
│   └── xgboost/
└── performance/
    ├── catboost/
    ├── lightgbm/
    └── xgboost/
```

### After

```
/outputs/visualizations/
├── features/
│   ├── catboost/
│   ├── lightgbm/
│   └── xgboost/
├── residuals/
│   ├── catboost/
│   ├── lightgbm/
│   └── xgboost/
└── performance/
    ├── catboost/
    ├── comparison/
    ├── lightgbm/
    └── xgboost/
```

## Next Steps

1. Run the cleanup script to remove existing CatBoost_* directories:
   ```
   python cleanup_catboost_directories.py
   ```

2. Run tests to ensure visualizations are still working correctly:
   ```
   python test_visualization.py
   ```

3. Continue with the broader visualization cleanup plan outlined in CLEANUP_PLAN.md