# Visualization Migration Plan

This document outlines a plan to fully migrate from the legacy `visualization` module to the new `visualization_new` architecture to ensure code consistency and maintainability.

## Current Status

The project currently maintains two parallel visualization systems:

1. **Legacy `visualization/` module**
   - Organized by model type (e.g., `xgboost_plots.py`, `lightgbm_plots.py`)
   - Used with `--visualize` flag in main.py
   - Shows deprecation warnings
   - Still actively used for certain model-specific visualizations

2. **New `visualization_new/` architecture**
   - Organized by plot type (e.g., `plots/metrics.py`, `plots/features.py`)
   - Uses adapters for different model types
   - Centralized with factory pattern
   - Used with `--visualize-new` flag in main.py

The project is in a transition state with both systems active. This causes maintenance issues as changes might be implemented in one system but not the other.

## Migration Goals

1. Ensure all visualization functionality exists in `visualization_new`
2. Redirect all visualization calls to use only `visualization_new`
3. Keep legacy code available but inactive for reference
4. Make sure there's a single source of truth for visualization code

## Action Plan

### 1. Verify Feature Parity

Ensure that all visualization features from the legacy module exist in the new architecture:

- [ ] Create a comprehensive list of all visualizations in the legacy module
- [ ] Map each legacy visualization to its equivalent in the new architecture
- [ ] Identify any missing features in the new architecture
- [ ] Implement any missing features in `visualization_new`

### 2. Update Main Entry Points

Modify the main.py file to use only the new visualization architecture:

- [ ] Replace direct calls to legacy visualization with new architecture
- [ ] Update the `--visualize` flag to use the new architecture (maintain backward compatibility)
- [ ] Keep `--visualize-new` flag temporarily, eventually merge both
- [ ] Add robust fallback mechanisms as needed during transition

### 3. Update Model-Specific Visualization Calls

Update all model-specific visualization code:

- [ ] Update XGBoost visualization using the pattern implemented for CatBoost
- [ ] Update LightGBM visualization using the same pattern
- [ ] Update ElasticNet/Linear Regression visualization
- [ ] Update any other model-specific visualization code

### 4. Comprehensive Testing

Test all visualization functionality with both flags:

- [ ] Run with `--visualize` flag and verify all plots generated correctly
- [ ] Run with `--visualize-new` flag and verify all plots generated correctly
- [ ] Compare outputs to ensure consistency

### 5. Deprecate Legacy Code

Once all functionality is properly implemented in the new architecture:

- [ ] Add stronger deprecation warnings to all legacy visualization modules
- [ ] Add redirection comments in legacy files pointing to the new implementations
- [ ] Move legacy visualization code to a `visualization_legacy` folder for reference
- [ ] Remove legacy visualization import paths from main.py
- [ ] Update any documentation referring to the legacy visualization

### 6. Documentation

Update documentation to reflect the new visualization approach:

- [ ] Create documentation on how to use the new visualization architecture
- [ ] Document the adapter pattern and how to add new model adapters
- [ ] Update any existing documentation to remove references to legacy visualization

## Implementation Timeline

1. **Phase 1: Feature Parity (Week 1)**
   - Complete the feature mapping
   - Implement missing features

2. **Phase 2: Code Updates (Week 2)**
   - Update main.py
   - Update model-specific visualization code

3. **Phase 3: Testing and Finalization (Week 3)**
   - Test all visualization functionality
   - Fix any issues found during testing

4. **Phase 4: Legacy Removal (Week 4)**
   - Move legacy code to reference folder
   - Update documentation
   - Final testing

## Immediate Next Steps

1. Create a detailed mapping of all visualization functions between old and new architectures
2. Identify any gaps in the new architecture
3. Update the CatBoost implementation (already done)
4. Continue with XGBoost and LightGBM implementations following the same pattern