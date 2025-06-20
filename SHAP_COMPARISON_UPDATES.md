# SHAP Comparison Plot Updates

## Overview
Modified SHAP comparison plots to create separate visualizations for Base and Yeo datasets, using 0-1 normalization and including all Optuna-optimized models.

## Changes Implemented

### 1. Main Pipeline SHAP Comparisons (src/visualization/plots/shap_plots.py)

#### Modified Functions:
- **`create_model_comparison_shap_plot()`**: Now calls the new separated plot function
- **`create_separated_model_comparison_shap_plots()`**: New function that creates separate Base and Yeo plots
- **`_create_dataset_specific_shap_plot()`**: New function that creates dataset-specific plots

#### Key Changes:
- **Dataset Selection**: Now includes ALL Optuna-optimized models (not just first valid)
- **Excludes**: Random models are still excluded from main pipeline plots
- **Output Files**:
  - `model_comparison_shap_base.png` - Base Decision models only
  - `model_comparison_shap_yeo.png` - Yeo Decision models only
- **Normalization**: Maintains 0-1 scale per model
- **Features**: Shows top 15 features based on maximum importance

### 2. Feature Removal SHAP Comparisons (xgboost_feature_removal_proper.py)

#### Modified Functions:
- **`_create_feature_removal_shap_comparison()`**: Now creates separate Base_Random and Yeo_Random plots
- **`_create_dataset_specific_removal_plot()`**: New function for dataset-specific removal plots

#### Key Changes:
- **Dataset Selection**: Separates Base_Random and Yeo_Random models
- **Output Files**:
  - `feature_removal_shap_comparison_base_random.png`
  - `feature_removal_shap_comparison_yeo_random.png`
- **Normalization**: Changed from relative importance (sum to 1) to 0-1 scale per model
- **Features**: Limited to top 15 features (was showing all features)
- **Formatting**: Matches main pipeline heatmap styling

## Results

### Main Pipeline Plots Created:
1. **Base Models** (`model_comparison_shap_base.png`):
   - Shows XGBoost_Base, LightGBM_Base, CatBoost_Base
   - Top feature: `top_3_shareholder_percentage` (1.000 across all models)
   - Other important features: `cntry_of_risk`, `issuer_cntry_domicile`, `gics_sub_ind`

2. **Yeo Models** (`model_comparison_shap_yeo.png`):
   - Shows XGBoost_Yeo, LightGBM_Yeo, CatBoost_Yeo
   - Top feature: `yeo_joh_top_3_shareholder_percentage` (1.000 across all models)
   - Other important features: `cntry_of_risk`, `issuer_cntry_domicile`, `gics_sub_ind`

### Feature Removal Plots:
- Separate plots for Base_Random and Yeo_Random datasets
- Compares "With Feature" vs "Without Feature" models
- Uses same 0-1 normalization and formatting as main pipeline

## Technical Details

### Normalization Method:
```python
# 0-1 scale normalization per model
for col in shap_df_normalized.columns:
    max_val = shap_df_normalized[col].max()
    if max_val > 0:
        shap_df_normalized[col] = shap_df_normalized[col] / max_val
```

### Feature Selection:
```python
# Select top 15 features based on maximum importance across models
max_importance = shap_df.max(axis=1)
top_features = max_importance.nlargest(15).index
```

### Model Filtering:
```python
# Main pipeline: Include only Optuna-optimized non-Random models
if 'optuna' in model_name.lower() and 'Random' not in model_name:
    # Include model
```

## Usage

To regenerate the updated SHAP comparison plots:

```bash
python run_updated_shap_comparisons.py
```

This script will:
1. Create separated Base and Yeo SHAP comparison plots for the main pipeline
2. Update feature removal SHAP comparisons with proper normalization

## Benefits

1. **Clearer Comparisons**: Separate plots for Base and Yeo avoid mixing different feature scales
2. **Consistent Normalization**: 0-1 scale preserves relative importance magnitude
3. **All Models Included**: Shows all Optuna-optimized models, not just first per type
4. **Better Feature Selection**: Top 15 features makes plots more readable
5. **Unified Formatting**: Consistent heatmap styling across all SHAP comparisons