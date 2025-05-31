# Implementation Guide

All implementation guides, migration plans, and how-to documentation

## Table of Contents

1. [Shap Visualization Implementation](#shap-visualization-implementation)
2. [Visualization Migration Plan](#visualization-migration-plan)
3. [Enhanced Archiving Guide](#enhanced-archiving-guide)
4. [Migration Implementation Guide](#migration-implementation-guide)
5. [Sector Analysis Guide](#sector-analysis-guide)

---

## Shap Visualization Implementation

_Source: SHAP_VISUALIZATION_IMPLEMENTATION.md (root)_

# SHAP Visualization Implementation Summary

## Overview
Successfully implemented SHAP (SHapley Additive exPlanations) visualizations as part of the clean architecture migration, enabling model interpretability across all model types.

## Implementation Details

### 1. SHAP Plot Module (`src/visualization/plots/shap_plots.py`)

Created comprehensive SHAP visualization support with the following plot types:

#### Available SHAP Plots

1. **SHAP Summary Plot** (`shap_summary`)
   - Shows feature importance and impact direction
   - Supports multiple models side-by-side
   - Configurable max features display

2. **SHAP Dependence Plot** (`shap_dependence`)
   - Shows relationship between feature values and SHAP values
   - Supports multiple features in grid layout
   - Automatic interaction detection

3. **SHAP Waterfall Plot** (`shap_waterfall`)
   - Explains individual predictions
   - Shows contribution of each feature
   - Includes actual vs predicted comparison

4. **SHAP Importance Plot** (`shap_importance`)
   - Cross-model feature importance comparison
   - Based on mean absolute SHAP values
   - Horizontal bar chart format

5. **SHAP Interaction Plot** (`shap_interaction`)
   - Shows interaction effects between two features
   - Color-coded by SHAP values
   - Scatter plot visualization

### 2. Integration with Clean Architecture

#### Plot Registry Integration
- All SHAP plots self-register using `@register_plot` decorator
- Automatic discovery through module imports
- Consistent parameter interface: `(data: Dict, config: Dict) -> Figure`

#### Parameter Structure
```python
# Example for SHAP summary plot
data = {
    'models_data': {
        'ModelName': {
            'model': trained_model,
            'X_test': test_features,
            'y_test': test_targets  # optional
        }
    }
}

config = {
    'max_features': 20,
    'save_path': 'path/to/save.png',
    'figsize': (12, 8)
}
```

### 3. Model Support

Implemented smart SHAP explainer selection based on model type:
- **Tree models** (XGBoost, LightGBM, CatBoost): TreeExplainer (fast)
- **Linear models** (ElasticNet, LinearRegression): KernelExplainer
- **Other models**: KernelExplainer with background sampling

### 4. Key Features

1. **Error Handling**
   - Graceful fallback when SHAP not installed
   - Informative error messages
   - Handles missing data/features

2. **Performance Optimization**
   - Uses TreeExplainer for tree models (100x faster)
   - Configurable background samples for KernelExplainer
   - Subset selection for large datasets

3. **Visualization Quality**
   - High DPI output (300 DPI default)
   - Proper labeling and titles
   - Consistent color schemes
   - Grid layouts for multiple plots

### 5. Test Results

Successfully tested with:
- ElasticNet model
- XGBoost model  
- LightGBM model

All plots generated correctly with proper SHAP value calculations.

### 6. Usage Examples

```python
# Using plot manager
from src.visualization.plot_manager import PlotManager

plot_manager = PlotManager(config)

# Create SHAP summary plot
plot_manager.create_plot(
    plot_type='shap_summary',
    data={'models_data': models_data},
    plot_config={'max_features': 15}
)

# Create SHAP dependence plots
plot_manager.create_plot(
    plot_type='shap_dependence',
    data={
        'model_data': models_data['XGBoost'],
        'feature_names': ['feature1', 'feature2']
    }
)

# Create SHAP waterfall for specific prediction
plot_manager.create_plot(
    plot_type='shap_waterfall',
    data={'model_data': models_data['XGBoost']},
    plot_config={'sample_idx': 42}
)
```

## Files Created/Modified

1. **New Files**:
   - `src/visualization/plots/shap_plots.py` - SHAP visualization implementations
   - `test_shap_visualizations.py` - Comprehensive test script
   - `test_shap_simple.py` - Basic SHAP functionality test

2. **Modified Files**:
   - `src/visualization/plots/__init__.py` - Added SHAP module import
   - `CLEAN_ARCHITECTURE_MIGRATION_STATUS.md` - Updated progress

## Dependencies

- **shap**: Version 0.47.2 (verified installed)
- **matplotlib**: For plot generation
- **numpy/pandas**: For data handling

## Future Enhancements

1. **Additional SHAP Plots**:
   - Force plots
   - Decision plots
   - Beeswarm plots (new SHAP feature)

2. **Performance Improvements**:
   - Parallel SHAP calculation for multiple models
   - Caching SHAP values for reuse
   - GPU acceleration for large datasets

3. **Interactive Features**:
   - Plotly/Bokeh backends for interactivity
   - Dashboard integration
   - Real-time SHAP analysis

## Conclusion

The SHAP visualization implementation provides comprehensive model interpretability within the clean architecture framework. All major SHAP plot types are supported with proper error handling, performance optimization, and seamless integration with the existing visualization system.
---

## Visualization Migration Plan

_Source: VISUALIZATION_MIGRATION_PLAN.md (root)_

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
---

## Enhanced Archiving Guide

_Source: docs/guides/ENHANCED_ARCHIVING_GUIDE.md_


---

## Migration Implementation Guide

_Source: docs/guides/MIGRATION_IMPLEMENTATION_GUIDE.md_


---

## Sector Analysis Guide

_Source: docs/guides/SECTOR_ANALYSIS_GUIDE.md_


---

