# Performance Plots Organization

This directory contains performance-related visualizations organized by model type.

## Directory Structure

- **catboost/**: CatBoost model optimization plots
- **lightgbm/**: LightGBM model optimization plots
- **xgboost/**: XGBoost model optimization plots
- **elasticnet/**: ElasticNet model optimization plots
- **linear/**: Linear Regression optimization plots (if any)
- **cv_distributions/**: Cross-validation distribution plots for all models

## File Types

### Optimization Plots (in model subdirectories)
- `*_optuna_optimization_history.png`: Optimization history showing how the objective improved over trials
- `*_optuna_param_importance.png`: Parameter importance analysis from Optuna
- `*_contour.png`: Contour plots showing parameter interactions

### Comparison Plots (in model subdirectories)
- `*_basic_vs_optuna.png`: Comparison between basic and Optuna-optimized models
- `*_best_*_comparison.png`: Comparison of best hyperparameter values

### Other Plots (in root)
- `metrics_summary_table.png`: Overall metrics summary table

## Naming Convention

Optimization plots follow this pattern:
`{ModelType}_{DataType}_{RandomFeature?}_{Categorical?}_optuna_{PlotType}.png`

Where:
- ModelType: CatBoost, LightGBM, XGBoost, ElasticNet_LR
- DataType: Base or Yeo
- RandomFeature: "Random" if random feature included
- Categorical: "categorical" for tree models
- PlotType: optimization_history, param_importance, or contour
