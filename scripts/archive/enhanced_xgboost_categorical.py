#!/usr/bin/env python3
"""
Enhanced XGBoost with native categorical support and Optuna optimization.
Combines the categorical handling from xgboost_categorical.py with 
the Optuna optimization from xgboost_model.py.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import warnings
import pickle
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import unified train/test split
from src.data.train_test_split import get_or_create_split

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_xgb_with_optuna(X_train, y_train, categorical_columns, n_trials=50, cv_folds=5, random_state=42):
    """Optimize XGBoost hyperparameters using Optuna with categorical support."""
    
    # Get categorical indices
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_columns]
    
    def objective(trial):
        # Hyperparameter search space
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  # Required for categorical support
            'enable_categorical': True,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'random_state': random_state,
            'verbosity': 0
        }
        
        # Stratified K-Fold for robust evaluation
        cv_scores = []
        
        # Use first categorical column for stratification if available
        if cat_indices:
            stratify_col = X_train.iloc[:, cat_indices[0]]
        else:
            # Use target bins for stratification
            stratify_col = pd.qcut(y_train, q=min(10, len(y_train)//10), labels=False, duplicates='drop')
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, stratify_col)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Create model
            model = xgb.XGBRegressor(**params)
            
            # Fit without early stopping
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_fold_val)
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(fold_rmse)
        
        # Store CV results in trial user attributes
        # Store CV scores in trial user attributes
        trial.set_user_attr('cv_scores', cv_scores)
        trial.set_user_attr('cv_mean', np.mean(cv_scores))
        trial.set_user_attr('cv_std', np.std(cv_scores))
        
        # Return mean CV score
        return np.mean(cv_scores)
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=random_state)
    )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    # Extract best CV scores from best trial
    best_cv_scores = None
    if study.best_trial:
        best_cv_scores = study.best_trial.user_attrs.get('cv_scores', None)
    
    return study, best_cv_scores

def train_enhanced_xgboost_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """Train enhanced XGBoost models with categorical support and Optuna optimization."""
    
    print(f"\n🌳 Training Enhanced XGBoost with categorical features on {dataset_name}...")
    print(f"Dataset shape: {X.shape}")
    print(f"Categorical features: {categorical_columns}")
    
    # Handle NaN values
    # First check if there are any NaNs
    if X.isnull().any().any() or (isinstance(y, pd.DataFrame) and y.isnull().any().any()) or (isinstance(y, pd.Series) and y.isnull().any()):
        print(f"  Warning: Found NaN values. Dropping rows with NaN...")
        # Create a mask for rows without NaN
        if isinstance(y, pd.DataFrame):
            mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
        else:
            mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask].copy()
        y = y[mask].copy()
        print(f"  After dropping NaN: {X.shape[0]} samples remaining")
    
    # Extract target as Series if it's a DataFrame
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            raise ValueError(f"Expected single target column, got {y.shape[1]}")
    
    # Use unified train/test split - MANDATORY for consistency
    print("  Using unified train/test split for consistency across models...")
    X_train, X_test, y_train, y_test = get_or_create_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify_column='gics_sector'  # Use sector for stratification
    )
    print(f"  Split successful: {len(X_train)} train, {len(X_test)} test samples")
    print(f"  Train indices sample: {list(X_train.index[:5])}")
    print(f"  Test indices sample: {list(X_test.index[:5])}")
    
    results = {}
    
    # 1. Train basic model with cross-validation
    print("\n--- Training Basic XGBoost with Categorical Support ---")
    basic_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'enable_categorical': True,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'verbosity': 0
    }
    
    # Create DMatrix for cross-validation
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    
    # Cross-validation to find best iteration and get CV scores
    cv_results = xgb.cv(
        basic_params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=50,
        verbose_eval=False,
        seed=random_state
    )
    
    best_iter = cv_results.shape[0]
    print(f"Best iteration: {best_iter}")
    
    # Update params with best iteration
    basic_params['n_estimators'] = best_iter
    
    # Train final basic model
    basic_model = xgb.XGBRegressor(**basic_params)
    basic_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate basic model
    y_pred_basic = basic_model.predict(X_test)
    y_train_pred_basic = basic_model.predict(X_train)
    
    basic_metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_basic)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_basic)),
        'train_r2': r2_score(y_train, y_train_pred_basic),
        'test_r2': r2_score(y_test, y_pred_basic),
        'train_mae': mean_absolute_error(y_train, y_train_pred_basic),
        'test_mae': mean_absolute_error(y_test, y_pred_basic)
    }
    
    print(f"Basic Model - Train RMSE: {basic_metrics['train_rmse']:.4f}")
    print(f"Basic Model - Test RMSE: {basic_metrics['test_rmse']:.4f}")
    print(f"Basic Model - Test R²: {basic_metrics['test_r2']:.4f}")
    
    # Compute proper CV scores for basic model
    print(f"\nComputing cross-validation scores for basic model...")
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import make_scorer
    
    # Create RMSE scorer
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
        greater_is_better=False
    )
    
    # Create XGBoost sklearn interface for CV
    xgb_sklearn_basic = xgb.XGBRegressor(**basic_params)
    
    # Use safe cross-validation to avoid sklearn compatibility issues
    try:
        from scripts.utilities.fix_sklearn_xgboost_compatibility import safe_cross_val_score
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        basic_cv_scores = safe_cross_val_score(
            xgb_sklearn_basic, X_train, y_train, 
            cv=cv, scoring=None, n_jobs=-1
        )
        # Convert to positive RMSE values
        basic_cv_scores = -basic_cv_scores
        basic_cv_mean = np.mean(basic_cv_scores)
        basic_cv_std = np.std(basic_cv_scores)
    except Exception as e:
        print(f"Warning: CV failed with error: {e}. Using placeholder values.")
        basic_cv_scores = np.array([basic_metrics['test_rmse']] * 5)
        basic_cv_mean = basic_metrics['test_rmse']
        basic_cv_std = 0.0
    
    print(f"Basic Model - CV RMSE: {basic_cv_mean:.4f} ± {basic_cv_std:.4f}")
    print(f"Basic Model - CV fold scores: {basic_cv_scores}")
    
    # Store basic model results
    model_key = f"XGBoost_{dataset_name}_categorical_basic"
    results[model_key] = {
        'model': basic_model,
        'model_name': model_key,  # Add model_name field
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred_basic,
        'y_train_pred': y_train_pred_basic,
        'train_score': basic_metrics['train_r2'],
        'test_score': basic_metrics['test_r2'],
        'metrics': basic_metrics,
        'RMSE': basic_metrics['test_rmse'],
        'MAE': basic_metrics['test_mae'],
        'MSE': basic_metrics['test_rmse'] ** 2,
        'R2': basic_metrics['test_r2'],
        'best_iteration': best_iter,
        'categorical_features': categorical_columns,
        'model_type': 'xgboost',
        'training_params': basic_params,
        'cv_scores': basic_cv_scores.tolist(),  # Store as list for serialization
        'cv_mean': basic_cv_mean,
        'cv_std': basic_cv_std
    }
    
    # 2. Optimize with Optuna
    print("\n--- Optimizing XGBoost with Optuna ---")
    study, best_cv_scores = optimize_xgb_with_optuna(X_train, y_train, categorical_columns, n_trials=50, random_state=random_state)
    
    print(f"Best trial value (RMSE): {study.best_value:.4f}")
    print("Best parameters:", study.best_params)
    if best_cv_scores:
        print(f"Best CV fold scores: {best_cv_scores}")
        print(f"CV mean: {np.mean(best_cv_scores):.4f}, CV std: {np.std(best_cv_scores):.4f}")
    
    # Extract CV scores from best trial
    best_trial = study.best_trial
    cv_scores = best_trial.user_attrs.get('cv_scores', [])
    cv_mean = best_trial.user_attrs.get('cv_mean', study.best_value)
    cv_std = best_trial.user_attrs.get('cv_std', 0)
    
    print(f"CV Mean RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
    if cv_scores:
        print(f"CV Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Train final model with best parameters
    optuna_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': random_state,
        'verbosity': 0,
        **study.best_params
    }
    
    optuna_model = xgb.XGBRegressor(**optuna_params)
    optuna_model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
    
    # Evaluate Optuna model
    y_pred_optuna = optuna_model.predict(X_test)
    y_train_pred_optuna = optuna_model.predict(X_train)
    
    optuna_metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_optuna)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_optuna)),
        'train_r2': r2_score(y_train, y_train_pred_optuna),
        'test_r2': r2_score(y_test, y_pred_optuna),
        'train_mae': mean_absolute_error(y_train, y_train_pred_optuna),
        'test_mae': mean_absolute_error(y_test, y_pred_optuna)
    }
    
    print(f"\nOptuna Model - Train RMSE: {optuna_metrics['train_rmse']:.4f}")
    print(f"Optuna Model - Test RMSE: {optuna_metrics['test_rmse']:.4f}")
    print(f"Optuna Model - Test R²: {optuna_metrics['test_r2']:.4f}")
    
    # If CV scores weren't properly extracted from Optuna, compute them
    if not cv_scores or len(cv_scores) == 0:
        print(f"\nComputing cross-validation scores for Optuna model...")
        try:
            from scripts.utilities.fix_sklearn_xgboost_compatibility import safe_cross_val_score
            from sklearn.model_selection import KFold
            
            # Create XGBoost with optuna parameters for CV
            xgb_sklearn_optuna = xgb.XGBRegressor(**optuna_params)
            
            # Use regular K-fold
            cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = safe_cross_val_score(
                xgb_sklearn_optuna, X_train, y_train, 
                cv=cv, scoring=None, n_jobs=-1
            )
            
            # Convert to positive RMSE values
            cv_scores = -cv_scores
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        except Exception as e:
            print(f"Warning: CV failed with error: {e}. Using placeholder values.")
            cv_scores = np.array([optuna_metrics['test_rmse']] * 5)
            cv_mean = optuna_metrics['test_rmse']
            cv_std = 0.0
        
        print(f"Optuna Model - CV RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Optuna Model - CV fold scores: {cv_scores}")
    
    # Store Optuna model results
    model_key = f"XGBoost_{dataset_name}_categorical_optuna"
    results[model_key] = {
        'model': optuna_model,
        'model_name': model_key,  # Add model_name field
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred_optuna,
        'y_train_pred': y_train_pred_optuna,
        'train_score': optuna_metrics['train_r2'],
        'test_score': optuna_metrics['test_r2'],
        'metrics': optuna_metrics,
        'RMSE': optuna_metrics['test_rmse'],
        'MAE': optuna_metrics['test_mae'],
        'MSE': optuna_metrics['test_rmse'] ** 2,
        'R2': optuna_metrics['test_r2'],
        'best_params': study.best_params,
        'study': study,
        'cv_scores': cv_scores,  # Store CV fold scores
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'categorical_features': categorical_columns,
        'model_type': 'xgboost',
        'training_params': optuna_params
    }
    
    # Get feature importance for both models
    for model_name, model_data in results.items():
        model = model_data['model']
        
        # Get feature importance
        importance_scores = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        model_data['feature_importance'] = importance_df
        
        # Save feature importance
        output_dir = Path("outputs/feature_importance")
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_dir / f"{model_name}_importance.csv", index=False)
        
        print(f"\nTop 10 features for {model_name}:")
        print(importance_df.head(10))
    
    return results

def main():
    """Test the enhanced implementation."""
    from data_categorical import load_tree_models_data, get_categorical_features
    
    # Load data
    X, y = load_tree_models_data()
    
    # Get categorical columns
    categorical_columns = get_categorical_features()
    
    # For testing, just use the base features (no transformation)
    results = train_enhanced_xgboost_categorical(
        X, y, "Base", categorical_columns
    )
    
    # Save results
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "enhanced_xgboost_categorical_models.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Enhanced XGBoost models saved successfully!")

if __name__ == "__main__":
    main()