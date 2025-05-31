# Feature Importance Scaling and Visualization Structure

This document explains how feature importance values are scaled across different model types and how visualizations are organized in the visualization system.

## Consolidated Linear Model Visualizations

All linear model types (both Linear Regression and ElasticNet) are now consolidated in a single directory:

- **Folder**: `features/linear/`
- **Models included**: Both standard Linear Regression (LR_*) and ElasticNet (ElasticNet_LR_*) 
- **Consistent scaling**: All linear models use the same 100x scaling factor

This consolidation ensures better comparability between different linear model variations and maintains a cleaner, more intuitive visualization structure.

## Why Feature Importance Scaling is Needed

Different model types calculate feature importance using fundamentally different approaches, leading to values with vastly different scales:

- **Tree-based models** (CatBoost, LightGBM, XGBoost): Calculate importance based on information gain or prediction change
  - Typically produce larger values (often in the tens, hundreds, or thousands)
  - No upper bound on importance values

- **Linear models** (ElasticNet, Linear Regression): Use coefficient magnitudes
  - Typically produce small values (often < 1)
  - Values depend on feature scale and regularization

This makes direct visual comparison between different model types difficult without scaling.

## Standard Scaling Approach

To address this issue, our visualization system applies a consistent scaling approach:

1. **Base Model Scaling (at adapter level)**
   - ElasticNet coefficients are scaled by 100x in the ElasticNetAdapter
   - This ensures that individual ElasticNet feature plots are readable
   - Applied in both the new system (adapters) and legacy visualization code

2. **Cross-Model Comparison Scaling (at visualization level)**
   - Additional scaling (100,000x) is applied in viz_factory.py for cross-model comparisons
   - Only applied when ElasticNet models are compared with other model types
   - Includes a minimum importance floor value of 0.5

## Implementation Details

- **ElasticNetAdapter**: `base_scale = 100`
- **viz_factory.py (cross-model)**: Additional `scale_factor = 100000`
- **Legacy visualization code**: Now consistent with the adapter scaling

## Guidelines for Interpreting Feature Importance

- **Single Model Analysis**: Feature importance values show relative importance within a model
  - Compare the relative ranking of features rather than absolute values
  - ElasticNet values displayed are 100x the raw coefficient magnitudes

- **Cross-Model Comparison**: Additional scaling is applied for better visualization
  - The focus should be on comparing feature rankings across models
  - Absolute values are not directly comparable between different model types

## Technical Details

The scaling is implemented in the following files:

1. `visualization_new/adapters/elasticnet_adapter.py`: Base 100x scaling
2. `visualization_new/adapters/linear_regression_adapter.py`: Base 100x scaling (matching ElasticNet)
3. `visualization/elasticnet_plots.py`: Now includes the same 100x scaling
4. `visualization_legacy/elasticnet_plots.py`: Now includes the same 100x scaling
5. `visualization_new/viz_factory.py`: 
   - Additional 100,000x scaling for cross-model comparisons
   - Routes all ElasticNet visualizations to the same `features/linear` folder as Linear Regression

## Visualization Directory Structure

The visualization system organizes feature importance plots as follows:

1. **Tree-based models**: Each has its dedicated folder
   - `features/catboost/`
   - `features/lightgbm/`
   - `features/xgboost/`

2. **Linear models**: All consolidated in a single folder
   - `features/linear/`: Contains both Linear Regression and ElasticNet visualizations
   - Naming convention preserved: `LR_*.png` for Linear Regression, `ElasticNet_LR_*.png` for ElasticNet

### XGBoost Folder Standardization

For XGBoost models, we've standardized all visualizations to use the `features/xgboost/` folder:

- **Previous situation**: 
  - XGBoost visualizations were split between `features/xgboost/` and `features/xgb/` folders
  - The two folders contained different feature importance calculations and rankings

- **Current solution**:
  - All XGBoost visualizations are now saved to the `features/xgboost/` folder only
  - The visualizations use permutation importance (model-agnostic method)
  - This importance type is more comparable with other models
  - No scaling is applied since tree-based models produce appropriately scaled importance values

This standardization provides cleaner organization while ensuring consistent and comparable feature importance visualization across all model types.