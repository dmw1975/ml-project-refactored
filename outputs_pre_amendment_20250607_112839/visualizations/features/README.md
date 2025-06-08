# Feature Plots Organization

This directory contains feature importance visualizations organized by model type.

## Directory Structure

- **catboost/**: CatBoost model feature importance plots
- **lightgbm/**: LightGBM model feature importance plots
- **xgboost/**: XGBoost model feature importance plots
- **elasticnet/**: ElasticNet model feature importance plots
- **linear/**: Linear Regression model feature importance plots
- **comparisons/**: Cross-model comparison plots and aggregated visualizations

## File Naming Convention

Individual model plots follow this pattern:
`{ModelType}_{DataType}_{RandomFeature?}_{Categorical?}_{Optimization?}_top_features.png`

Where:
- ModelType: CatBoost, LightGBM, XGBoost, ElasticNet_LR, LR
- DataType: Base or Yeo
- RandomFeature: "Random" if random feature included
- Categorical: "categorical" for tree models
- Optimization: "basic" or "optuna" for optimized models

Comparison plots include:
- average_feature_rank_{data_type}_{categorical?}.png
- feature_rank_heatmap_{data_type}_{categorical?}.png
- top_N_features_avg_importance.png
