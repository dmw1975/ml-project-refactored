# Comprehensive Optuna Configuration Audit

## Executive Summary

This audit reveals a well-structured Optuna implementation across all tree-based models (XGBoost, LightGBM, CatBoost) and ElasticNet, with both centralized configuration and distributed implementations.

## 1. Repository-Wide Optuna File Locations

### A. Configuration Files
```
src/config/
├── settings.py                    # Basic settings (n_trials=50)
└── hyperparameters.py            # NEW: Centralized parameter definitions
```

### B. Enhanced Model Implementations
```
scripts/archive/
├── enhanced_xgboost_categorical.py    # XGBoost with Optuna
├── enhanced_lightgbm_categorical.py   # LightGBM with Optuna
├── enhanced_catboost_categorical.py   # CatBoost with Optuna
└── enhanced_elasticnet_optuna.py      # ElasticNet with Optuna
```

### C. Model Wrappers
```
src/models/
├── xgboost_categorical.py    # Imports enhanced version
├── lightgbm_categorical.py   # Imports enhanced version
├── catboost_categorical.py   # Imports enhanced version
└── elastic_net.py           # Has both grid search and Optuna
```

### D. Pipeline Integration
```
src/pipelines/
├── training.py              # Handles Optuna parameter passing
└── visualization.py         # Visualizes Optuna results
```

### E. Visualization Support
```
src/visualization/plots/
└── optimization.py          # Optuna-specific visualizations
```

### F. Utility Scripts
```
scripts/utilities/
├── add_elasticnet_study_objects.py    # Grid search to Optuna conversion
├── check_existing_studies.py          # Study inspection utilities
└── generate_optuna_plots.py          # Standalone Optuna visualization
```

## 2. Optuna Implementation Details

### A. XGBoost Configuration

**File**: `scripts/archive/enhanced_xgboost_categorical.py`

```python
# Optuna Study Creation
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=random_state)
)
study.optimize(objective, n_trials=n_trials, n_jobs=1)

# Hyperparameter Space
params = {
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True)
}
```

### B. LightGBM Configuration

**File**: `scripts/archive/enhanced_lightgbm_categorical.py`

```python
# Hyperparameter Space
params = {
    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    'num_boost_round': trial.suggest_int('num_boost_round', 50, 500)
}
```

### C. CatBoost Configuration

**File**: `scripts/archive/enhanced_catboost_categorical.py`

```python
# Hyperparameter Space
params = {
    'iterations': trial.suggest_int('iterations', 100, 1000),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'depth': trial.suggest_int('depth', 4, 10),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
    'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
    'border_count': trial.suggest_int('border_count', 32, 255)
}
```

### D. ElasticNet Configuration

**File**: `scripts/archive/enhanced_elasticnet_optuna.py`

```python
# Hyperparameter Space
alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
max_iter = trial.suggest_int('max_iter', 5000, 20000, step=5000)
tol = trial.suggest_float('tol', 1e-5, 1e-3, log=True)

# Special Feature: Random Feature Penalty
if abs(full_model.coef_[random_idx]) > 1e-10:
    trial.set_user_attr('uses_random_feature', True)
    return np.mean(cv_scores) * 1.01  # 1% penalty
```

## 3. Cross-Validation Strategy

All models use consistent CV strategy:
- **Method**: StratifiedKFold
- **Folds**: 5 (default)
- **Stratification**: 
  - Tree models: First categorical column or target quantiles
  - ElasticNet: Sector labels

## 4. Trial Tracking and Attributes

Each trial stores:
```python
trial.set_user_attr('cv_scores', cv_scores)      # Individual fold scores
trial.set_user_attr('cv_mean', np.mean(cv_scores))  # Mean CV score
trial.set_user_attr('cv_std', np.std(cv_scores))    # CV standard deviation
```

ElasticNet additionally stores:
```python
trial.set_user_attr('uses_random_feature', True/False)  # Random feature usage
```

## 5. Command-Line Integration

**File**: `main.py`

```bash
# XGBoost
--optimize-xgboost N      # N trials (default: 50)
--force-retune           # Force new optimization

# LightGBM  
--optimize-lightgbm N     # N trials (default: 50)

# CatBoost
--optimize-catboost N     # N trials (default: 50)

# ElasticNet
--optimize-elasticnet N   # N trials (default: 100)
--elasticnet-grid        # Use grid search instead
```

## 6. Pipeline Integration

**File**: `src/pipelines/training.py`

```python
# Example: XGBoost training
n_trials = kwargs.get('optimize_xgboost', settings.XGBOOST_PARAMS.get('n_trials', 50))
xgboost_models = train_xgboost_categorical_models(
    datasets=datasets,
    n_trials=n_trials,
    force_retune=kwargs.get('force_retune', False)
)
```

## 7. Visualization Support

**File**: `src/visualization/plots/optimization.py`

Optuna visualizations include:
- Optimization history plots
- Parameter importance plots  
- Contour plots (2D parameter relationships)
- CV distribution plots

## 8. Model Output Structure

Each enhanced model saves:
```python
{
    f"{model_name}_basic": {           # Basic model with default params
        'model': model_object,
        'metrics': {...},
        'cv_scores': [...],
        ...
    },
    f"{model_name}_optuna": {          # Optuna-optimized model
        'model': model_object,
        'metrics': {...},
        'cv_scores': [...],
        'study': optuna_study_object,  # Full study for analysis
        'best_params': {...},
        ...
    }
}
```

## 9. Consistency Features

### A. Random State Management
- All studies use `random_state=42`
- TPESampler initialized with same seed
- Ensures reproducible optimization

### B. Error Handling
- Failed trials return `float('inf')`
- Convergence failures handled gracefully
- CV failures logged but don't crash optimization

### C. Progress Tracking
- ElasticNet shows progress bar
- All models log trial progress
- Best parameters printed after optimization

## 10. Special Features

### A. ElasticNet Grid Search Conversion
Script to convert existing grid search results to Optuna format:
```python
scripts/utilities/add_elasticnet_study_objects.py
```

### B. Study Inspection
Check existing studies without retraining:
```bash
python main.py --check-studies
```

### C. Force Retuning
Override existing studies:
```bash
python main.py --train --force-retune
```

## 11. Key Findings

1. **Consistent Implementation**: All models follow the same Optuna pattern
2. **Centralized Config**: New `hyperparameters.py` provides single source of truth
3. **Flexible N_Trials**: Can be overridden via command line
4. **CV Integration**: All models use proper cross-validation
5. **Study Persistence**: Studies saved with models for later analysis
6. **Visualization Ready**: Full visualization pipeline for Optuna results

## 12. Recommendations

1. **Use Centralized Config**: All new code should import from `hyperparameters.py`
2. **Consistent Naming**: Follow pattern `{Model}_{Dataset}_categorical_{basic|optuna}`
3. **Always Save Studies**: Include study object in model output for analysis
4. **Document Ranges**: Add rationale for parameter ranges in config
5. **Monitor Convergence**: Check if 50 trials is sufficient for convergence

## Conclusion

The repository has a mature, well-implemented Optuna integration across all models. The recent addition of centralized configuration (`hyperparameters.py`) provides a path toward more maintainable parameter management while preserving the flexibility of the existing implementation.