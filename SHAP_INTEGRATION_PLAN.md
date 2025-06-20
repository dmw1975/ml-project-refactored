# SHAP Module Integration Plan

## Overview
This document provides a detailed plan to merge safety features from `generate_shap_visualizations_safe.py` and additional plot types from `xgboost_feature_removal_proper.py` into the existing `shap_plots.py` module.

## Analysis of Key Components

### 1. Safety Features from generate_shap_visualizations_safe.py

#### Memory Management
- **Function**: `check_memory_usage()` (lines 26-31)
  - Uses psutil to monitor memory in GB
  - Called before and after SHAP computation
  - Essential for preventing OOM errors

#### Resource Limits
- **Parameter**: `max_samples=30` in `compute_shap_for_model()` (line 65)
  - Limits SHAP computation to subset of test data
  - Prevents excessive computation time
  - Uses random sampling when dataset is large

#### Error Handling
- **Pattern**: Per-plot try/except blocks (lines 142-204)
  - Each visualization type wrapped in individual try/except
  - Allows partial success (some plots may fail, others succeed)
  - Detailed error messages for debugging

#### Skip Existing Files
- **Logic**: Lines 243-248
  - Checks if output directory exists and contains PNG files
  - Skips processing if visualizations already exist
  - Prevents redundant computation

#### Model Type Handling
- **CatBoost Special Case**: Lines 88-96, 250-254
  - Attempts TreeExplainer with feature_perturbation
  - Falls back to skipping if generic Explainer needed (too slow)
  - Safe mode skips CatBoost entirely

#### Cleanup
- **Memory cleanup**: Lines 273-276
  - Explicitly deletes SHAP values and samples
  - Calls gc.collect() after each model
  - Also closes all matplotlib figures (line 214)

### 2. Additional Plot Types from xgboost_feature_removal_proper.py

#### Summary Plot Variations
- **Bar plot**: Lines 803-810
  - Standard feature importance bar chart
- **Dot plot**: Lines 812-818
  - Shows feature value distributions with SHAP values

#### Force Plots
- **Single instance**: Lines 825-830
- **Multiple instances**: Lines 832-839
  - Creates separate force plots for multiple test instances
  - Saves each as individual file

#### Interaction Plots
- **Implementation**: Lines 857-867
  - Shows interaction between top 2 features
  - Uses `shap.dependence_plot` with interaction_index

#### Custom Comparison Heatmap
- **Function**: `_create_feature_removal_shap_comparison()` (lines 885-971)
  - Collects SHAP importance across models
  - Creates normalized heatmap with annotations
  - Specific to feature removal analysis

### 3. Current shap_plots.py Capabilities

#### Existing Safety Features
- Basic try/except in `create_shap_visualizations()` (lines 256-344)
- Sample size limiting (line 275)
- Model type detection (lines 280-295)

#### Missing Safety Features
- No memory monitoring
- No per-plot error handling
- No file existence checking
- No explicit cleanup
- No special CatBoost handling

#### Existing Plot Types
- Summary plot (bar only)
- Waterfall plot
- Dependence plots
- Categorical plots
- Model comparison heatmap

## Integration Plan

### Phase 1: Add Core Safety Infrastructure

#### 1.1 Add Memory Management
```python
# Add at module level
import psutil
import gc

def check_memory_usage():
    """Check current memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024
```

#### 1.2 Enhance create_shap_visualizations()
```python
def create_shap_visualizations(model_data: Dict[str, Any], output_dir: Path, 
                              sample_size: int = 100, max_samples: int = 100,
                              skip_existing: bool = True) -> List[Path]:
    """
    Enhanced with safety features:
    - max_samples: Hard limit on SHAP computation
    - skip_existing: Check for existing files
    - Memory monitoring and cleanup
    """
```

#### 1.3 Add Per-Plot Error Handling
Wrap each visualization creation in individual try/except blocks within SHAPVisualizer methods.

### Phase 2: Add New Plot Types to SHAPVisualizer

#### 2.1 Add Summary Dot Plot Method
```python
def create_shap_summary_dot_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame, 
                                 output_path: Path) -> None:
    """Create SHAP summary plot with dot visualization."""
    plt.figure(figsize=self.style['figure_size'])
    shap.summary_plot(shap_values, X_sample, show=False)  # No plot_type="bar"
    # ... rest of implementation
```

#### 2.2 Add Force Plot Methods
```python
def create_shap_force_plot(self, explainer, shap_values: np.ndarray, 
                          X_sample: pd.DataFrame, instance_idx: int, 
                          output_path: Path) -> None:
    """Create SHAP force plot for a single instance."""
    
def create_shap_force_plots_multiple(self, explainer, shap_values: np.ndarray,
                                    X_sample: pd.DataFrame, n_instances: int,
                                    output_dir: Path) -> List[Path]:
    """Create SHAP force plots for multiple instances."""
```

#### 2.3 Add Interaction Plot Method
```python
def create_shap_interaction_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                feature1_idx: int, feature2_idx: int, 
                                output_path: Path) -> None:
    """Create SHAP interaction plot between two features."""
```

### Phase 3: Enhance Main Functions

#### 3.1 Update create_shap_visualizations()
Add new plot types to the main visualization function:
- Check for existing files before processing
- Add memory monitoring
- Create all plot types with proper error handling
- Add cleanup after each model

#### 3.2 Add Special Handlers
```python
def handle_catboost_shap(model, X_sample):
    """Special handling for CatBoost models."""
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        return explainer
    except Exception as e:
        logger.warning(f"CatBoost TreeExplainer failed: {e}")
        return None
```

### Phase 4: Add Specialized Functions

#### 4.1 Feature Removal Comparison
```python
def create_feature_removal_shap_comparison(models: Dict[str, Any], 
                                          excluded_feature: str,
                                          output_dir: Path) -> Optional[Path]:
    """Create specialized SHAP comparison for feature removal analysis."""
```

#### 4.2 Enhanced Model Comparison
Update existing `create_model_comparison_shap_plot()` to include:
- Memory monitoring
- Better error handling
- Support for more model types

### Implementation Steps

1. **Backup current shap_plots.py**
   ```bash
   cp src/visualization/plots/shap_plots.py src/visualization/plots/shap_plots_backup.py
   ```

2. **Add imports and utility functions**
   - Import psutil, gc
   - Add check_memory_usage()
   - Add logging setup

3. **Enhance SHAPVisualizer class**
   - Add new plot methods
   - Add per-plot error handling to existing methods
   - Add memory monitoring

4. **Update main functions**
   - Add safety parameters to create_shap_visualizations()
   - Add file existence checking
   - Add cleanup logic

5. **Add specialized functions**
   - Feature removal comparison
   - Enhanced model comparison

6. **Test with different model types**
   - XGBoost (standard case)
   - LightGBM (standard case)
   - CatBoost (special handling)
   - Linear models (different explainer)

### Code Structure After Integration

```
shap_plots.py
├── Imports (with psutil, gc)
├── Utility Functions
│   ├── check_memory_usage()
│   ├── handle_catboost_shap()
│   └── safe_cleanup()
├── SHAPVisualizer Class
│   ├── Existing Methods (enhanced with safety)
│   ├── create_shap_summary_dot_plot()
│   ├── create_shap_force_plot()
│   ├── create_shap_force_plots_multiple()
│   ├── create_shap_interaction_plot()
│   └── create_shap_comparison_matrix()
├── Main Functions
│   ├── create_shap_visualizations() (enhanced)
│   ├── create_all_shap_visualizations() (enhanced)
│   └── create_model_comparison_shap_plot() (enhanced)
└── Specialized Functions
    ├── create_feature_removal_shap_comparison()
    └── create_batch_shap_analysis()
```

### Testing Plan

1. **Unit Tests**
   - Test each new plot method individually
   - Test error handling with invalid inputs
   - Test memory cleanup

2. **Integration Tests**
   - Run on small dataset first
   - Test with all model types
   - Verify file skipping works
   - Check memory usage stays reasonable

3. **Performance Tests**
   - Large dataset handling
   - CatBoost special cases
   - Memory monitoring effectiveness

### Rollback Plan

If issues arise:
1. Restore from backup: `cp src/visualization/plots/shap_plots_backup.py src/visualization/plots/shap_plots.py`
2. Use generate_shap_visualizations_safe.py as standalone script
3. Document any discovered incompatibilities

## Summary

This integration will:
1. **Preserve all existing functionality** in shap_plots.py
2. **Add robust safety features** from generate_shap_visualizations_safe.py
3. **Include additional plot types** from xgboost_feature_removal_proper.py
4. **Maintain backward compatibility** with existing code
5. **Improve performance and reliability** for large-scale SHAP analysis

The key is to integrate incrementally, testing after each phase to ensure stability.