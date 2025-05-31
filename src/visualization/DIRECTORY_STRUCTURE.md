# Visualization Directory Structure

This document describes the standardized directory structure used by the visualization_new package.

## Overview

The visualization system organizes outputs using a **type-based directory structure**. This means visualizations are first categorized by type (e.g., features, residuals, performance), then by model type (e.g., catboost, xgboost, lightgbm).

## Directory Structure

```
/outputs/visualizations/
├── features/              # Feature importance visualizations
│   ├── catboost/          # CatBoost feature importance plots
│   ├── lightgbm/          # LightGBM feature importance plots
│   ├── linear/            # Linear models feature importance plots
│   └── xgboost/           # XGBoost feature importance plots
│
├── residuals/             # Residual analysis visualizations
│   ├── catboost/          # CatBoost residual plots
│   ├── lightgbm/          # LightGBM residual plots
│   ├── linear/            # Linear models residual plots
│   └── xgboost/           # XGBoost residual plots
│
├── performance/           # Performance metrics visualizations
│   ├── catboost/          # CatBoost performance plots
│   ├── comparison/        # Cross-model comparison plots
│   ├── lightgbm/          # LightGBM performance plots
│   ├── linear/            # Linear models performance plots
│   └── xgboost/           # XGBoost performance plots
│
├── sectors/               # Sector-specific visualizations
│
└── statistical_tests/     # Statistical test visualizations
```

## File Naming Conventions

Within each directory, files follow these naming conventions:

1. **Feature Importance**:
   - `top_features_base_basic.png` - Top features for base dataset, basic model
   - `top_features_yeo_optuna.png` - Top features for Yeo dataset, Optuna-optimized model

2. **Residuals**:
   - `residuals_base_basic.png` - Residual plot for base dataset, basic model
   - `residuals_yeo_optuna.png` - Residual plot for Yeo dataset, Optuna-optimized model

3. **Performance**:
   - `comparison_model_metrics.png` - Comparison of all models' performance metrics
   - `comparison_metrics_table.png` - Summary table of all models' metrics
   - `comparison_top_features.png` - Comparison of top features across models

## Usage in Code

The visualization system uses a helper function to ensure consistent directory structure:

```python
def get_visualization_dir(model_name: str, plot_type: str) -> Path:
    """
    Return standardized directory path for visualizations.
    
    Args:
        model_name: Name of the model (e.g., 'CatBoost_Base_basic')
        plot_type: Type of visualization (e.g., 'features', 'residuals', 'performance')
        
    Returns:
        Path: Path to the visualization directory
    """
    from config import settings
    
    # Extract base model type (e.g., "catboost" from "CatBoost_Base_basic")
    model_type = model_name.lower().split('_')[0]
    
    # Create and return the path
    output_dir = settings.VISUALIZATION_DIR / plot_type / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir
```

## Advantages of This Structure

1. **Consistency**: All visualizations follow the same organization pattern
2. **Intuitive Navigation**: Users can find visualizations by type first, then model
3. **Scalability**: New model types can be added without changing the structure
4. **Type-Based Organization**: Easier to compare similar visualizations across models
5. **Flatter Hierarchy**: Avoids nested model-specific subdirectories for cleaner structure

## Implementation Notes

- When adding new visualization types, extend the structure by adding a new top-level directory
- Use lowercase model names for directories to maintain consistency
- Always use the `get_visualization_dir` helper function to ensure consistent directory structure

## Legacy Compatibility

The legacy visualization system in the `visualization/` package has been deprecated and redirects to this new structure. All functions in the legacy system will continue to work, but they now save files to the new directory structure.

Users should migrate to the `visualization_new` package for all new code.