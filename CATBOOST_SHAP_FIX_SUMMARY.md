# CatBoost SHAP Visualization Fix Summary

## Issue
Multiple CatBoost model folders under `outputs/visualizations/shap/` were empty:
- CatBoost_Base_Random_categorical_basic
- CatBoost_Base_Random_categorical_optuna
- CatBoost_Yeo_Random_categorical_basic
- CatBoost_Yeo_Random_categorical_optuna
- CatBoost_Yeo_categorical_basic
- CatBoost_Yeo_categorical_optuna

## Root Cause
The SHAP computation was failing with the error:
> "Currently TreeExplainer can only handle models with categorical splits when feature_perturbation="tree_path_dependent" and no background data is passed."

This occurred because:
1. CatBoost models trained with categorical features require special handling in SHAP
2. The generic `shap.Explainer` was using TreeExplainer internally without the required parameters
3. Models with "Random" features and "Yeo" transformed features had categorical variables that triggered this issue

## Solution
Updated the SHAP computation logic in `generate_shap_visualizations.py`:

```python
if "CatBoost" in model_name:
    # For CatBoost with categorical features, use TreeExplainer with specific parameters
    try:
        # First try with tree_path_dependent for categorical support
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    except Exception as e:
        # If that fails, try generic Explainer
        print(f"    TreeExplainer with tree_path_dependent failed: {e}")
        print("    Trying generic Explainer...")
        explainer = shap.Explainer(model, X_sample)
```

## Results
All CatBoost models now have SHAP visualizations:
- **CatBoost_Base_Random_categorical_basic**: 7 plots ✓
- **CatBoost_Base_Random_categorical_optuna**: 7 plots ✓
- **CatBoost_Base_categorical_basic**: 10 plots ✓
- **CatBoost_Base_categorical_optuna**: 10 plots ✓
- **CatBoost_Yeo_Random_categorical_basic**: 7 plots ✓
- **CatBoost_Yeo_Random_categorical_optuna**: 7 plots ✓
- **CatBoost_Yeo_categorical_basic**: 7 plots ✓
- **CatBoost_Yeo_categorical_optuna**: 7 plots ✓

## Generated Plots
Each model now has:
1. **Summary plot** - Feature importance ranking
2. **Waterfall plot** - Individual prediction explanation
3. **Dependence plots** - Top 3 feature relationships
4. **Categorical plots** - Top 2 categorical feature impacts

## Pipeline Integration
The fix is integrated into the main pipeline and will automatically handle CatBoost models with categorical features when running:
```bash
python main.py --visualize
python main.py --all
```

## Key Insights
- Models with "Base" features (non-transformed) generated more plots (10) than models with "Yeo" or "Random" features (7)
- The tree_path_dependent parameter is crucial for CatBoost models with categorical features
- The fix ensures robust SHAP computation that handles both standard and categorical feature scenarios