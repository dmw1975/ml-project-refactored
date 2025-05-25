# Statistical Tests Visualization Summary

## Current State

The statistical test visualizations in `outputs/visualizations/statistical_tests/` have been updated and now include:

### 1. **Complete Baseline Comparisons** ✅
- Created new baseline comparison plots that include ALL 28 models
- `baseline_comparison_mean.png` - All models vs mean baseline
- `baseline_comparison_median.png` - All models vs median baseline  
- `baseline_comparison_random.png` - All models vs random baseline
- Results show:
  - Linear Regression models perform best overall (smallest RMSE)
  - All models significantly outperform random baseline (avg 51.78% improvement)
  - Most models outperform mean/median baselines (avg 10.18%/10.57% improvement)
  - 28/28 models show significant improvement over mean baseline
  - 26/28 models show significant improvement over median baseline

### 2. **Pairwise Statistical Tests**
- The `model_comparison_tests.csv` includes pairwise comparisons between models
- However, comparisons between linear and tree models fail due to different test set sizes:
  - Linear models: 2202 test samples
  - Tree models: 1982 test samples
- ElasticNet models are missing from pairwise tests (not in model pickle files)

### 3. **Enhanced Significance Matrices** ✅
- `enhanced_significance_matrix.png` - Shows all available model comparisons
- `enhanced_significance_matrix_Base.png` - Base dataset models only
- `enhanced_significance_matrix_Yeo.png` - Yeo dataset models only
- These visualizations show which models significantly outperform others using Holm-Bonferroni correction

### 4. **Issues Still Present**
- CatBoost models have missing metrics in comparison tables
- Cross-validation metrics not stored, preventing CV-based baseline significance tests
- ElasticNet models not included in pairwise statistical tests
- Test set size mismatch prevents full model comparisons

## Files Generated

Successfully created visualizations:
- `baseline_comparison_mean.png` - Complete comparison of all 28 models vs mean baseline
- `baseline_comparison_median.png` - Complete comparison of all 28 models vs median baseline
- `baseline_comparison_random.png` - Complete comparison of all 28 models vs random baseline
- `enhanced_significance_matrix.png` - Pairwise significance matrix (limited by test set issues)
- `enhanced_significance_matrix_Base.png` - Base dataset pairwise comparisons
- `enhanced_significance_matrix_Yeo.png` - Yeo dataset pairwise comparisons
- `complete_baseline_comparisons.csv` - Full baseline comparison data for all models

## Root Causes

1. **Data Pipeline Inconsistency**: Linear models and tree models use different data pipelines, resulting in different test set sizes
2. **Model Storage Format**: Different model types store metrics in different formats, making unified comparison difficult
3. **Missing Cross-Validation Data**: Models don't store cross-validation fold results needed for baseline significance tests
4. **ElasticNet Model Location**: ElasticNet models aren't being saved to the expected location

## Recommendations

1. **Standardize Test Sets**: Ensure all models use the same test set for fair comparison
2. **Store CV Metrics**: Modify training scripts to save cross-validation metrics with each model
3. **Fix ElasticNet Storage**: Ensure ElasticNet models are saved to the standard model pickle file
4. **Standardize Metric Format**: Use consistent metric storage across all model types