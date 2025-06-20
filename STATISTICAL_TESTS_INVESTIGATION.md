# Statistical Test Plots Investigation Report

## Investigation Summary

### Question 1: Why are non-optimized models included in statistical test plots?

**Root Cause**: The model filtering logic in `src/evaluation/metrics.py` was checking for model TYPE (xgboost, lightgbm, etc.) rather than optimization status.

**Original Filter**:
```python
allowed_model_types = ['elasticnet', 'xgb', 'catboost', 'lightgbm', 'lr_', 'linear']

def is_allowed(name):
    name = name.lower()
    return any(allowed_type in name for allowed_type in allowed_model_types)
```

This filter included ALL models of these types, including basic (non-optimized) versions.

**Fix Applied**:
```python
def is_allowed(name):
    """Check if model should be included in statistical tests.
    
    Requirements:
    - Include all models with 'optuna' in the name (optimized tree models)
    - Include all ElasticNet models (they are always optimized)
    - Exclude basic (non-optimized) tree models
    """
    return 'optuna' in name or 'ElasticNet' in name
```

Now only optimized models are included in statistical tests.

### Question 2: Metrics Discrepancy for XGBoost_Yeo_categorical_optuna

**Investigation Results**: There is NO actual discrepancy.

**Evidence**:
- CSV file shows: 1.6579915976063044
- Baseline tests show: 1.6579915976063044  
- Model file shows: 1.6579915976063044
- PNG table shows: 1.658 (rounded to 3 decimal places)

The reported value of 1.642 appears to be a misreading. All sources consistently show the RMSE as 1.6579915976063044, which rounds to 1.658 when displayed with 3 decimal places.

## Code Changes Made

### 1. Updated Model Filtering (`src/evaluation/metrics.py`)
- Modified `perform_statistical_tests()` function to only include optimized models
- Updated documentation to clarify the filtering requirements
- Enhanced logging to show filtered models grouped by type

### 2. Key Findings
- **Model Selection**: Fixed - now only includes optimized tree models and ElasticNet models
- **Metrics Consistency**: No issue found - all metrics are consistent across sources
- **Display Rounding**: PNG displays round to 3 decimal places (1.658)

## Verification

After running `python main.py --evaluate`, the statistical tests will now only include:
- All models with "_optuna" suffix (optimized tree models)
- All ElasticNet models (which are always optimized)
- Excludes all "_basic" tree models

The statistical test plots in `outputs/visualizations/statistical_tests/` will be regenerated with only the optimized models.

## Logging Improvements

The evaluation pipeline now logs:
1. Total models after filtering
2. Grouped display of:
   - Optuna-optimized tree models
   - ElasticNet models
3. Clear indication if insufficient models for testing

This makes it easier to verify that the correct models are being compared.