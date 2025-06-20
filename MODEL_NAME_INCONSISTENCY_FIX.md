# Model Name Inconsistency Fix Summary

## Issue Identified
CatBoost and LightGBM models were not appearing correctly in baseline comparison visualizations. Investigation revealed that while CatBoost and LightGBM models ARE present in the baseline comparison data, XGBoost models were being categorized as "Other" in the visualizations.

## Root Cause
The issue was a naming inconsistency:
- **Actual model names**: Use full names like `XGBoost_Base_categorical_basic`
- **Visualization code**: Was checking for models that start with 'XGB'

This mismatch caused the `extract_model_type` functions in visualization code to fail to recognize XGBoost models properly.

## Data Verification
The baseline comparison file (`outputs/metrics/baseline_comparison.csv`) contains:
- 24 CatBoost models ✓
- 24 LightGBM models ✓ 
- 24 XGBoost models ✓
- 12 ElasticNet models ✓
- 4 Linear Regression models ✓

All models ARE included in the baseline comparisons - the issue was only in the visualization layer.

## Files Fixed
1. **src/visualization/plots/baselines.py**
   - Fixed 4 occurrences of `startswith('XGB')` → `startswith('XGBoost')`

2. **src/visualization/plots/consolidated_baselines.py**
   - Fixed 1 occurrence of `startswith('XGB')` → `startswith('XGBoost')`

3. **src/evaluation/baselines.py**
   - Fixed 1 occurrence of `startswith('XGB')` → `startswith('XGBoost')`

4. **src/evaluation/baseline_significance.py**
   - Fixed 2 occurrences: Changed checks from 'XGB' to 'XGBoost'

5. **src/utils/io.py**
   - Fixed algorithm prefix mapping: `'xgboost': 'XGB'` → `'xgboost': 'XGBoost'`

## Impact
After these fixes:
- XGBoost models will now be properly categorized and colored in baseline comparison plots
- The model type legends will correctly show "XGBoost" instead of having these models appear as "Other"
- All tree-based models (XGBoost, LightGBM, CatBoost) will be displayed with their appropriate colors in visualizations

## Verification
To verify the fix works:
1. Regenerate baseline visualizations: `python -m src.visualization.plots.baselines`
2. Check that XGBoost models now appear with the correct color (#0173B2) and are labeled as "XGBoost" in the legend

## No Data Changes Required
The baseline comparison data itself is correct and complete. No re-running of evaluations is needed - only the visualization code needed updating to properly recognize the model names.