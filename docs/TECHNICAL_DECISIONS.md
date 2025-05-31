# Technical Decisions

Technical decisions, solutions, and fix documentation

## Table of Contents

1. [Cleanup Plan](#cleanup-plan)
2. [Clean Pipeline Example](#clean-pipeline-example)
3. [Dataset Consistency Solution](#dataset-consistency-solution)
4. [Repo Cleanup Plan](#repo-cleanup-plan)

---

## Cleanup Plan

_Source: CLEANUP_PLAN.md (root)_

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
---

## Clean Pipeline Example

_Source: CLEAN_PIPELINE_EXAMPLE.md (root)_

# Clean Pipeline Example: Before vs After

## Current Messy Approach

### Running the pipeline now:
```bash
# Confusing, unclear what it does
python main.py --all

# Or with many flags
python main.py --train --evaluate --visualize --optimize_xgboost 100 --elasticnet_grid

# Which plots are generated? Who knows!
# Which models are trained? Check 1379 lines of code!
```

### Current Issues:
1. No idea what `--all` actually does without reading code
2. Plots missing? Add more code to main.py!
3. Want to add a new model? Edit multiple files
4. Failed halfway? Start over from beginning

## Clean Pipeline Approach

### 1. Simple, Clear Commands

```bash
# Train all models
esg-ml train

# Train specific model with optimization
esg-ml train --model xgboost --optimize

# Evaluate and visualize
esg-ml evaluate
esg-ml visualize

# Generate report
esg-ml report
```

### 2. Configuration-Driven Workflow

```yaml
# configs/experiments/full_analysis.yaml
name: "Full ESG Analysis"

data:
  features: "all"
  test_size: 0.2
  stratify: "sector"

models:
  - name: linear_regression
    datasets: ["base", "yeo"]
    
  - name: elasticnet
    datasets: ["base", "yeo"]
    optimize:
      method: "optuna"
      trials: 100
      
  - name: xgboost
    datasets: ["base", "yeo"]
    variants: ["basic", "optimized"]
    
  - name: lightgbm
    datasets: ["base", "yeo"]
    variants: ["basic", "optimized"]

evaluation:
  metrics: ["rmse", "mae", "r2"]
  baselines: ["random", "mean", "median"]
  statistical_tests: true

visualization:
  performance:
    - metrics_table
    - model_comparison
    - cv_distributions
  
  features:
    - importance_plots
    - shap_analysis
    
  diagnostics:
    - residual_plots
    - qq_plots
    
  comparisons:
    - baseline_comparisons
    - optimization_impact
```

### 3. Run Specific Experiment

```bash
# Run full analysis
esg-ml run --config configs/experiments/full_analysis.yaml

# Run only certain stages
esg-ml run --config configs/experiments/full_analysis.yaml --stages training,evaluation

# Resume from checkpoint
esg-ml run --config configs/experiments/full_analysis.yaml --resume
```

### 4. Clear Output Structure

```
outputs/
├── 2025-01-29_full_analysis/
│   ├── models/
│   │   ├── linear_regression_base.pkl
│   │   ├── elasticnet_base_optuna.pkl
│   │   ├── xgboost_base_basic.pkl
│   │   └── ...
│   │
│   ├── figures/
│   │   ├── performance/
│   │   │   ├── metrics_table.png
│   │   │   ├── model_comparison.png
│   │   │   └── cv_distributions/
│   │   ├── features/
│   │   │   ├── importance/
│   │   │   └── shap/
│   │   └── diagnostics/
│   │
│   ├── reports/
│   │   ├── summary.html
│   │   ├── detailed_analysis.pdf
│   │   └── metrics.csv
│   │
│   └── logs/
│       ├── pipeline.log
│       └── checkpoints/
```

### 5. Easy Debugging

```python
# src/pipeline/runner.py
class PipelineRunner:
    def run(self):
        for stage in self.stages:
            try:
                self.logger.info(f"Starting {stage}")
                self.run_stage(stage)
                self.save_checkpoint(stage)
                self.logger.info(f"Completed {stage}")
            except Exception as e:
                self.logger.error(f"Failed at {stage}: {e}")
                if self.config.fail_fast:
                    raise
                else:
                    self.logger.warning(f"Continuing despite error in {stage}")
```

### 6. Adding New Features

#### Before (Messy):
1. Edit main.py (find the right place among 1379 lines)
2. Add new if statement
3. Import your code
4. Hope it doesn't break something else
5. No easy way to test in isolation

#### After (Clean):
```python
# src/models/my_new_model.py
from src.models.base import BaseModel

class MyNewModel(BaseModel):
    """My new model implementation."""
    
    def fit(self, X, y):
        # Implementation
        return self
    
    def predict(self, X):
        # Implementation
        return predictions

# Register it
from src.models.registry import ModelRegistry
ModelRegistry.register("my_new_model", MyNewModel)
```

Then just add to config:
```yaml
models:
  - name: my_new_model
    datasets: ["base"]
```

### 7. Interactive Development

```python
# notebooks/experiment.ipynb
from src.pipeline import load_config, create_pipeline

# Load and modify config
config = load_config("configs/default.yaml")
config.models = ["xgboost"]  # Test single model

# Create and run pipeline
pipeline = create_pipeline(config)
results = pipeline.run()

# Analyze results interactively
results.plot_performance()
results.get_best_model()
```

### 8. Automated Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_models/test_xgboost.py

# Run with coverage
pytest --cov=src tests/
```

### 9. Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: make install
      - name: Run tests
        run: make test
      - name: Run linting
        run: make lint
      - name: Test pipeline
        run: esg-ml run --config configs/test.yaml
```

### 10. Clear Documentation

```bash
# Generate API docs
make docs

# Serves at http://localhost:8000
make serve-docs
```

## Summary: Why This Is Better

### Current (Messy):
- 🚫 1379-line main.py
- 🚫 Hidden dependencies
- 🚫 Unclear what runs when
- 🚫 Hard to test
- 🚫 Difficult to extend

### New (Clean):
- ✅ Modular components
- ✅ Clear configuration
- ✅ Easy to understand
- ✅ Fully testable
- ✅ Simple to extend
- ✅ Production ready

### For New Developers:
```bash
# Old way: "Good luck understanding main.py!"

# New way:
git clone repo
make install
make test        # Everything works!
esg-ml --help   # Clear commands
make docs       # Comprehensive documentation
```

This clean architecture makes the codebase:
- **Understandable**: New developers productive in hours, not weeks
- **Maintainable**: Clear where to make changes
- **Extensible**: Easy to add new features
- **Reliable**: Comprehensive testing
- **Professional**: Ready for production use
---

## Dataset Consistency Solution

_Source: DATASET_CONSISTENCY_SOLUTION.md (root)_

# Dataset Consistency Solution

## Summary

This document describes how we fixed the data consistency issue between the old and new ML pipelines to ensure fair comparison and identical results.

## Problem

The initial comparison between old and new pipelines showed significant differences:
- Old pipeline: Test RMSE = 3.4712, Test R2 = -2.2744
- New pipeline: Test RMSE = 1.8645, Test R2 = 0.1066

Despite both pipelines using the same LinearRegression model from sklearn.

## Root Causes Identified

1. **Data Loading**: The new pipeline was using fallback methods to load data from the old pipeline, but the data wasn't being saved in the expected locations.

2. **Train/Test Split Strategy**: The new pipeline defaulted to stratified splitting by sector, while the old pipeline used simple random splitting.

## Solution Implemented

### 1. Data Migration Script (`fix_data_consistency.py`)

Created a script that:
- Loads data using the old pipeline's exact methods
- Saves the processed datasets to the new pipeline's expected locations
- Creates a feature mapping file for reference
- Verifies data consistency after saving

Key files created:
- `esg_ml_clean/data/processed/features_base.csv` - Base features (362 columns)
- `esg_ml_clean/data/processed/features_yeo.csv` - Yeo-transformed features (26 columns)
- `esg_ml_clean/data/processed/targets.csv` - ESG scores
- `esg_ml_clean/data/processed/feature_mapping.json` - Feature metadata

### 2. DataLoader Updates

Updated the new pipeline's DataLoader to:
- First check for migrated data files
- Use the migrated data when available
- Fall back to legacy loading only if needed

### 3. Comparison Script Fix

Updated `compare_old_vs_new.py` to:
- Use the same train/test split strategy (no stratification)
- Match the exact parameters of the old pipeline

## Verification

After implementing these fixes:
- Both pipelines use identical data (2202 samples, 362 features)
- Both pipelines produce identical train/test splits (1761/441 samples)
- Both pipelines produce identical model results (RMSE: 3.4712, R2: -2.2744)

## Usage

1. Run the data consistency fix (only needed once):
   ```bash
   python fix_data_consistency.py
   ```

2. Verify pipeline equivalence:
   ```bash
   cd esg_ml_clean
   python compare_old_vs_new.py
   ```

## Important Notes

- The new pipeline defaults to stratified splitting by sector, which is generally better for model generalization
- For exact comparison with old results, use `stratify_by=None` in the data configuration
- The migrated data files are now the source of truth for both pipelines
- All feature transformations and preprocessing are preserved exactly as in the old pipeline

## Next Steps

With data consistency verified, you can now:
1. Migrate trained models from the old pipeline
2. Run full pipeline comparisons with all model types
3. Gradually transition to using the new pipeline's improved features (like stratified splitting)
---

## Repo Cleanup Plan

_Source: REPO_CLEANUP_PLAN.md (root)_

# Repository Cleanup Plan

## Current State
- **Total Python files**: ~150+
- **Temporary/diagnostic files**: ~120
- **Essential core files**: ~30

## Files to Keep (Used by main.py)

### Core Infrastructure
- `main.py` - Main pipeline entry point
- `config/settings.py` - Configuration
- `data_categorical.py` - Data loading
- `create_categorical_datasets.py` - Dataset creation

### Model Implementations (models/)
- `linear_regression.py`
- `elastic_net.py`
- `xgboost_categorical.py` (keep categorical version)
- `lightgbm_categorical.py` (keep categorical version)
- `catboost_categorical.py` (keep categorical version)
- `sector_models.py`
- `sector_lightgbm_models.py`

### Evaluation (evaluation/)
- `metrics.py`
- `importance.py`
- `multicollinearity.py`
- `baselines.py`

### Utilities (utils/)
- `io.py`
- `helpers.py`

### Visualization (visualization_new/)
Keep the entire `visualization_new/` directory as it's the current architecture

## Files to Remove

### Temporary Scripts (60+ files)
- All `test_*.py` files (except core tests if needed)
- All `fix_*.py` files
- All `check_*.py` files
- All `debug_*.py` files
- All `cleanup_*.py` files
- All `generate_*.py` files
- Miscellaneous diagnostic scripts

### Duplicate Implementations
- `models/*_model.py` (keep only categorical versions)
- `models/*_original.py` backup files
- `backup_integration_*/` directory

### Obsolete Visualizations
- `visualization/` directory (replaced by visualization_new)
- `visualization_legacy/` directory
- Individual visualization scripts (e.g., `*_shap_visualizations.py`)

### Temporary Data Creation Scripts
- `create_unified_data_pipeline.py`
- `train_all_models_unified.py`
- `demonstrate_unified_pipeline.py`
- Other one-off data scripts

## Recommended Actions

1. **Create archive directory**: Move temporary files to `archive/` before deletion
2. **Update imports**: Ensure main.py uses only categorical model versions
3. **Consolidate documentation**: Merge multiple MD files into comprehensive docs
4. **Clean data directories**: Remove temporary pickle files and CSVs
5. **Standardize naming**: Ensure consistent file naming conventions

## Implementation Priority

1. **High Priority**: Remove obvious temporary/diagnostic scripts
2. **Medium Priority**: Consolidate duplicate model implementations
3. **Low Priority**: Clean up documentation and organize remaining files
---

