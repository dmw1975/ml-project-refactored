# Statistical Test Validation Report - Holm-Bonferroni Implementation

## Executive Summary

The statistical testing implementation has several critical issues:
1. **All p-values are identical (0.0625)** due to limited statistical power with only 5 CV folds
2. **Holm-Bonferroni correction is properly implemented** but ineffective due to high p-values
3. **Baseline calculations have a bug** - using RMSE values instead of actual predictions for mean/median
4. **56% improvement appears non-significant** due to the limitations of Wilcoxon test with n=5

## 1. Code Location Analysis

### Statistical Test Implementation
- **Main implementation**: `/src/evaluation/baseline_significance.py`
- **Key functions**:
  - `test_models_against_baselines()` (lines 156-267): Main statistical testing
  - `generate_baseline_cv_metrics()` (lines 90-154): Baseline metric generation
  - `collect_cv_metrics_from_models()` (lines 28-87): CV metric collection
  - `run_baseline_significance_analysis()` (lines 427-523): Pipeline orchestration

### Statistical Test Output Generation
- **Visualization**: `/src/visualization/comprehensive.py`
- **Output directory**: `/outputs/visualizations/statistical_tests/`
- **Generated files**:
  - `baseline_significance_tests.csv`: Test results with p-values
  - `baseline_comparison_*.png`: Visualization plots
  - `baseline_significance_table.png`: Summary table

## 2. Holm-Bonferroni Implementation

### Current Implementation (CORRECTLY IMPLEMENTED)
```python
# Lines 252-255 in baseline_significance.py
reject, p_adjusted, _, _ = multipletests(
    results_df.loc[mask, 'p-value'].values,
    method='holm'
)
```

### Implementation Details
- Uses `statsmodels.stats.multitest.multipletests` with `method='holm'`
- Applied separately for each baseline type (Random, Mean, Median)
- Correctly updates `p-value-adjusted` and `Significant` columns
- The implementation is mathematically correct

## 3. P-Value Anomaly Investigation

### Why All P-Values are 0.0625

The issue stems from using **Wilcoxon signed-rank test with only 5 samples** (CV folds):

```python
# Lines 221-228 in baseline_significance.py
if n_folds >= 8:
    # Use t-test for larger sample sizes
    t_stat, p_value = stats.ttest_rel(baseline_cv_metrics, model_cv_metrics)
    test_type = "t-test"
else:
    # Use Wilcoxon for smaller sample sizes
    stat, p_value = stats.wilcoxon(baseline_cv_metrics, model_cv_metrics)
    test_type = "Wilcoxon"
```

**Mathematical Explanation**:
- With n=5 samples, Wilcoxon test has only 2^5 = 32 possible rank sum combinations
- The minimum possible p-value is 2*(0.5)^5 = 0.0625 (two-tailed test)
- This occurs when all 5 differences have the same sign (all positive or all negative)
- Since all models outperform baselines, all differences are positive, yielding p=0.0625

### Why P-Values are Identical Across Baseline Types

The p-values are identical because:
1. All models consistently outperform all three baseline types across all 5 folds
2. The Wilcoxon test only considers the sign and rank of differences, not magnitudes
3. With consistent improvement across folds, all tests yield the minimum p-value

## 4. Statistical Test Setup Validation

### Baseline Calculation Issues

**Critical Bug Found** (Line 531 in baselines.py):
```python
# INCORRECT - Using RMSE value as prediction!
baseline_pred = np.full(len(y_test_arr), baseline_val)
```

This creates constant predictions equal to the baseline RMSE value (e.g., 1.897), not actual mean/median predictions.

### Correct Baseline Generation (in baseline_significance.py)
```python
# Lines 142-150 - Correctly generates baselines
elif baseline_type == "mean":
    mean_pred = np.mean(y)
    mse = np.mean((y - mean_pred) ** 2)
    rmse = np.sqrt(mse)
    return np.full(n_folds, rmse)
```

## 5. Significance Threshold Analysis

### XGBoost_Yeo_Random_Categorical_Optuna Analysis

From the CSV data:
- **Raw p-value**: 0.0625
- **Adjusted p-value**: 1.0 (after Holm-Bonferroni)
- **Improvement**: 56.45% over random baseline
- **Significance threshold**: 0.05

### Why 56% Improvement is Not Significant

1. **Multiple Comparison Correction**:
   - 28 models tested against Random baseline
   - First comparison needs p < 0.05/28 = 0.00179
   - Raw p-value (0.0625) >> threshold (0.00179)

2. **Limited Statistical Power**:
   - Only 5 CV folds limits test sensitivity
   - Minimum possible p-value (0.0625) > significance threshold (0.05)
   - **Impossible to achieve significance with current setup**

## 6. Root Cause Analysis

### Issue 1: Identical P-Values
**Cause**: Limited statistical power with n=5 samples
**Solution**: Increase CV folds (recommended: 10-fold CV)

### Issue 2: Baseline Prediction Bug
**Cause**: Using RMSE values as predictions in baselines.py
**Solution**: Generate actual mean/median predictions

### Issue 3: Non-Significant Large Improvements
**Cause**: Conservative multiple comparison correction + limited power
**Solution**: 
- Increase sample size (more CV folds)
- Consider less conservative corrections (FDR instead of FWER)
- Use bootstrap methods for more robust p-values

## 7. Recommendations

1. **Increase CV Folds**: Use 10-fold CV for adequate statistical power
2. **Fix Baseline Bug**: Generate proper mean/median predictions
3. **Consider Alternative Tests**: 
   - Bootstrap hypothesis testing
   - Permutation tests
   - Bayesian methods
4. **Adjust Correction Method**: 
   - Consider FDR control (Benjamini-Hochberg) for less conservative correction
   - Group models by type before correction
5. **Add Power Analysis**: Calculate required sample size for detecting meaningful differences

## 8. Statistical Validity Assessment

### Current Implementation
- **Mathematically Sound**: ✓ (Holm-Bonferroni correctly implemented)
- **Appropriate Test Choice**: ✓ (Wilcoxon for small samples)
- **Multiple Comparison Correction**: ✓ (Properly applied)

### Practical Issues
- **Statistical Power**: ✗ (Too low with n=5)
- **Baseline Implementation**: ✗ (Bug in baselines.py)
- **Meaningful Comparisons**: ✓ (Comparing against relevant baselines)

## Conclusion

The statistical framework is correctly implemented but suffers from:
1. Insufficient statistical power (n=5 too small)
2. Implementation bug in baseline predictions
3. Conservative correction making significance nearly impossible

The 56% improvement being non-significant is a direct result of these limitations, not a flaw in the statistical methodology itself.