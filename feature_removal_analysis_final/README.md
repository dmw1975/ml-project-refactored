# XGBoost Feature Removal Analysis Results

## Summary
This analysis evaluates the impact of removing the `top_3_shareholder_percentage` feature from the XGBoost model.

## Key Findings

### Model Performance
- **Optimal Baseline RMSE**: 1.6134 (XGBoost_Yeo_Random_categorical_optuna)
- **Model WITH feature**: RMSE = 1.6645, R² = 0.2879
- **Model WITHOUT feature**: RMSE = 1.6738, R² = 0.2800

### Impact of Removing Feature
- **RMSE Change**: +0.0092 (+0.55%)
- **MAE Change**: +0.0088 (+0.67%)
- **R² Change**: -0.0079

### Interpretation
Removing `top_3_shareholder_percentage` results in a **0.55% increase in RMSE**, 
indicating that this feature contributes to model performance, though the effect is relatively small.

## Visualizations Generated
1. **Residual Analysis**: Comprehensive residual plots for both models
2. **SHAP Analysis**: Feature importance and dependence plots
3. **Feature Importance Comparison**: Side-by-side comparison
4. **Metrics Summary Table**: Tabular summary of all metrics

## Optimal Hyperparameters Used
- max_depth: 10
- learning_rate: 0.011682
- n_estimators: 326
- subsample: 0.802249
- colsample_bytree: 0.764521
