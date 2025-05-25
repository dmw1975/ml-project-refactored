# Model Metrics Discrepancy Analysis

This document explains the discrepancy observed between the baseline comparison visualizations and the metrics summary table regarding model performance rankings, particularly for ElasticNet models.

## Summary of Findings

1. **Different Metrics Sources**:
   - **Metrics Summary Table**: Uses test set RMSE (from model predictions on the held-out test set)
   - **Baseline Significance Analysis**: Uses cross-validation RMSE (from k-fold cross-validation during training)

2. **Key Discrepancy**:
   - ElasticNet models show excellent performance in cross-validation (CV) metrics
   - However, they perform significantly worse on the test set compared to their CV results
   - Tree-based models (XGBoost, LightGBM, CatBoost) show much more consistent performance between CV and test set

## Detailed Analysis

### ElasticNet Models

| Model | Test RMSE | CV RMSE | Difference | % Increase |
|-------|-----------|---------|------------|------------|
| ElasticNet_LR_Base | 1.9610 | 1.3843 | 0.5767 | +41.7% |
| ElasticNet_LR_Yeo | 1.7783 | 1.3084 | 0.4699 | +35.9% |
| ElasticNet_LR_Base_Random | 1.9610 | 1.3843 | 0.5767 | +41.7% |
| ElasticNet_LR_Yeo_Random | 1.7774 | 1.3085 | 0.4689 | +35.8% |

### Tree-based Models (Sample)

| Model | Test RMSE | CV RMSE | Difference | % Increase |
|-------|-----------|---------|------------|------------|
| XGBoost_Base_optuna | 1.6908 | 1.6475 | 0.0433 | +2.6% |
| LightGBM_Base_optuna | 1.7001 | 1.6516 | 0.0486 | +2.9% |
| CatBoost_Base_optuna | 1.7161 | 1.6402 | 0.0760 | +4.6% |

## Explanation

The discrepancy in ElasticNet model rankings can be explained by **overfitting**:

1. **During Cross-Validation**:
   - ElasticNet models appear to be the best performers, with CV RMSE values around 1.30-1.38
   - This is what's reflected in the baseline significance analysis visualizations

2. **On Test Set**:
   - ElasticNet models perform much worse with test RMSE values of 1.78-1.96
   - This represents a 35-42% performance degradation from CV to test
   - Tree-based models only show a 2-5% degradation between CV and test

3. **Root Cause**:
   - ElasticNet models are likely overfitting to the training data
   - While they show excellent in-sample performance (during CV), they generalize poorly to unseen data
   - Tree-based models (XGBoost, LightGBM, CatBoost) demonstrate much better generalization

## Recommendations

1. **Use Test Set Metrics for Final Comparisons**:
   - Test set performance is a more reliable indicator of a model's ability to generalize
   - Based on test set RMSE, tree-based models (particularly XGBoost and LightGBM with optuna tuning) are superior

2. **Address ElasticNet Overfitting**:
   - Consider stronger regularization for ElasticNet models
   - Review feature selection for ElasticNet models
   - Consider using more robust cross-validation strategies

3. **Visualization Improvements**:
   - Ensure visualizations clearly indicate which metrics are being used (CV vs test)
   - Consider adding generalization gap metrics (test-CV difference) to model comparisons

## Technical Details

This analysis was performed by examining the model pickle files and comparing their stored metrics with the values used in different visualizations. The discrepancy was confirmed by directly calculating the square root of the CV MSE values stored in the model files and comparing them to the final RMSE values reported in the metrics summary.