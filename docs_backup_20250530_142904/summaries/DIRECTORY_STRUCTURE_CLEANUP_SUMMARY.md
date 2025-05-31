# Directory Structure Cleanup Summary

## The Problem

We identified two issues with the visualization directory structure:

1. **Unwanted Directory Generation**: The visualization_new architecture was creating model-specific directories like `CatBoost_Base_basic/` in the root visualizations folder.

2. **Inconsistent Structure**: The code was creating both model-specific subdirectories under features/residuals and root model directories.

## The Solution

We implemented a comprehensive solution to standardize the directory structure:

1. **Modified `get_visualization_dir` Function**:
   - Changed to use base model type (e.g., "catboost") rather than full model name
   - Ensured consistent type-based directory structure (features/catboost/, residuals/catboost/)

2. **Updated File Naming in `visualize_model`**:
   - For CatBoost models, files now use standardized naming patterns:
     - `features/catboost/top_features_base_basic.png`
     - `residuals/catboost/residuals_base_optuna.png`
   - This keeps files in the same directory but with unique names

3. **Created Cleanup Script**:
   - Developed `cleanup_catboost_directories.py` to remove unwanted directories
   - The script preserved important files by moving them to the new structure
   - Removed all `CatBoost_*` directories from the root visualization folder
   - Removed model-specific directories under features/catboost_* and residuals/catboost_*

## The New Directory Structure

The directory structure is now simpler and more consistent:

```
/outputs/visualizations/
├── features/
│   ├── catboost/       # All CatBoost feature importance files
│   ├── lightgbm/
│   └── xgboost/
├── residuals/
│   ├── catboost/       # All CatBoost residual plots
│   └── other models...
└── performance/
    ├── catboost/       # CatBoost performance metrics
    ├── lightgbm/
    └── xgboost/
```

## Benefits

1. **Simpler Navigation**: Users can find visualizations by type first, then model
2. **Reduced Clutter**: No more proliferation of model-specific directories
3. **Consistent Organization**: All visualizations follow the same pattern
4. **Future-proof**: The structure works for all model types and variants

## Technical Changes Made

1. Fixed `visualization_new/viz_factory.py`:
   - Added `get_visualization_dir()` helper function to standardize paths
   - Modified `visualize_model()` to use type-based directories
   - Updated file naming to include variant information while maintaining uniqueness

2. Created cleanup script:
   - Identifies and removes unwanted directories
   - Preserves files by moving them to the correct locations
   - Handles both CatBoost_* root directories and model-specific subdirectories

## Testing

We verified our changes by:
1. Running CatBoost visualizations to confirm they use the correct directory structure
2. Checking that both visualize_model() and visualize_all_models() functions use the standardized structure
3. Confirming that no unwanted directories are being created

## Next Steps

1. **Complete Phase 1 of the Cleanup Plan**: Add deprecation warnings to legacy code
2. **Update Documentation**: Document the new directory structure for users
3. **Continue with Phase 2-4 of Cleanup Plan**: Gradually phase out legacy visualization code