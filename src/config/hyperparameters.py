"""
Centralized Hyperparameter Configuration
========================================

This module provides centralized hyperparameter definitions for all models
to ensure consistency across the repository.
"""

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    'basic': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'enable_categorical': True,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    },
    'optuna_space': {
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        'gamma': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
        'alpha': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
        'lambda': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True}
    },
    'optuna_config': {
        'n_trials': 50,
        'cv_folds': 5,
        'direction': 'minimize',
        'random_state': 42
    }
}

# LightGBM Hyperparameters
LIGHTGBM_PARAMS = {
    'basic': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbosity': -1
    },
    'optuna_space': {
        'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
        'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
        'bagging_freq': {'type': 'int', 'low': 1, 'high': 7},
        'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
        'lambda_l1': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'lambda_l2': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'num_boost_round': {'type': 'int', 'low': 50, 'high': 500}
    },
    'optuna_config': {
        'n_trials': 50,
        'cv_folds': 5,
        'direction': 'minimize',
        'random_state': 42
    }
}

# CatBoost Hyperparameters
CATBOOST_PARAMS = {
    'basic': {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    },
    'optuna_space': {
        'iterations': {'type': 'int', 'low': 100, 'high': 1000},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'depth': {'type': 'int', 'low': 4, 'high': 10},
        'l2_leaf_reg': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'bagging_temperature': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'random_strength': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'border_count': {'type': 'int', 'low': 32, 'high': 255}
    },
    'optuna_config': {
        'n_trials': 50,
        'cv_folds': 5,
        'direction': 'minimize',
        'random_state': 42
    }
}

# ElasticNet Hyperparameters
ELASTICNET_PARAMS = {
    'basic': {
        'max_iter': 10000,
        'tol': 0.0001,
        'random_state': 42
    },
    'optuna_space': {
        'alpha': {'type': 'float', 'low': 1e-4, 'high': 10.0, 'log': True},
        'l1_ratio': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'max_iter': {'type': 'int', 'low': 5000, 'high': 20000, 'step': 5000},
        'tol': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True}
    },
    'optuna_config': {
        'n_trials': 50,
        'cv_folds': 5,
        'direction': 'minimize',
        'random_state': 42
    }
}

def get_optuna_params(trial, model_type='xgboost'):
    """
    Get Optuna hyperparameters for a given model type.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    model_type : str
        Model type: 'xgboost', 'lightgbm', 'catboost', 'elasticnet'
    
    Returns
    -------
    dict
        Dictionary of hyperparameters
    """
    if model_type == 'xgboost':
        param_space = XGBOOST_PARAMS['optuna_space']
    elif model_type == 'lightgbm':
        param_space = LIGHTGBM_PARAMS['optuna_space']
    elif model_type == 'catboost':
        param_space = CATBOOST_PARAMS['optuna_space']
    elif model_type == 'elasticnet':
        param_space = ELASTICNET_PARAMS['optuna_space']
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    params = {}
    for param_name, config in param_space.items():
        if config['type'] == 'int':
            if 'step' in config:
                params[param_name] = trial.suggest_int(
                    param_name, config['low'], config['high'], step=config['step']
                )
            else:
                params[param_name] = trial.suggest_int(
                    param_name, config['low'], config['high']
                )
        elif config['type'] == 'float':
            log = config.get('log', False)
            params[param_name] = trial.suggest_float(
                param_name, config['low'], config['high'], log=log
            )
    
    return params

def get_basic_params(model_type='xgboost'):
    """
    Get basic (non-optimized) hyperparameters for a given model type.
    
    Parameters
    ----------
    model_type : str
        Model type: 'xgboost', 'lightgbm', 'catboost', 'elasticnet'
    
    Returns
    -------
    dict
        Dictionary of basic hyperparameters
    """
    if model_type == 'xgboost':
        return XGBOOST_PARAMS['basic'].copy()
    elif model_type == 'lightgbm':
        return LIGHTGBM_PARAMS['basic'].copy()
    elif model_type == 'catboost':
        return CATBOOST_PARAMS['basic'].copy()
    elif model_type == 'elasticnet':
        return ELASTICNET_PARAMS['basic'].copy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Parameter rationale documentation
PARAMETER_RATIONALE = {
    'xgboost': {
        'max_depth': "3-10: Balances model complexity and overfitting risk",
        'learning_rate': "0.01-0.3: Standard range for gradient boosting",
        'n_estimators': "50-500: Sufficient for most datasets without excessive training time",
        'subsample': "0.5-1.0: Prevents overfitting while maintaining performance",
        'colsample_bytree': "0.5-1.0: Feature sampling for regularization",
        'min_child_weight': "1-10: Controls minimum sum of instance weight in child",
        'gamma': "1e-8-1.0: Minimum loss reduction for split",
        'alpha': "1e-8-1.0: L1 regularization term",
        'lambda': "1e-8-1.0: L2 regularization term"
    },
    'lightgbm': {
        'num_leaves': "10-300: Key parameter for model complexity",
        'learning_rate': "0.01-0.3: Standard range for gradient boosting",
        'feature_fraction': "0.4-1.0: Feature sampling ratio",
        'bagging_fraction': "0.4-1.0: Data sampling ratio",
        'min_child_samples': "5-100: Minimum data in leaf",
        'lambda_l1': "1e-8-10.0: L1 regularization",
        'lambda_l2': "1e-8-10.0: L2 regularization"
    }
}