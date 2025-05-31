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