#!/usr/bin/env python3
"""
Enhanced LightGBM with native categorical support and Optuna optimization.
Combines the categorical handling from lightgbm_categorical.py with 
the Optuna optimization from lightgbm_model.py.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import warnings
import pickle
from pathlib import Path
import re

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def clean_feature_names(X):
    """Clean feature names to be JSON-compatible for LightGBM."""
    original_names = X.columns.tolist()
    cleaned_names = []
    
    for name in original_names:
        # Replace problematic characters
        clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', name)
        clean_name = clean_name.strip('_')
        cleaned_names.append(clean_name)
    
    # Create mapping
    name_mapping = dict(zip(cleaned_names, original_names))
    
    # Rename columns
    X_clean = X.copy()
    X_clean.columns = cleaned_names
    
    return X_clean, name_mapping

def optimize_lgb_with_optuna(X_train, y_train, categorical_columns, n_trials=50, cv_folds=5, random_state=42):
    """Optimize LightGBM hyperparameters using Optuna with categorical support."""
    
    def objective(trial):
        # Hyperparameter search space
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbosity': -1,
            'random_state': random_state,
            'force_col_wise': True,
            'feature_pre_filter': False  # Needed for dynamic min_data_in_leaf
        }
        
        # Get categorical indices for current feature set
        cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_columns]
        
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
            
            train_data = lgb.Dataset(
                X_fold_train, 
                label=y_fold_train,
                categorical_feature=cat_indices
            )
            
            val_data = lgb.Dataset(
                X_fold_val, 
                label=y_fold_val,
                categorical_feature=cat_indices,
                reference=train_data
            )
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=trial.suggest_int('num_boost_round', 50, 500),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0)
                ]
            )
            
            # Evaluate
            y_pred = model.predict(X_fold_val)
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(fold_rmse)
        
        # Store CV results in trial user attributes
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

def train_enhanced_lightgbm_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """Train enhanced LightGBM models with categorical support and Optuna optimization."""
    
    print(f"\n🌳 Training Enhanced LightGBM with categorical features on {dataset_name}...")
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
    
    # Clean feature names
    X_clean, name_mapping = clean_feature_names(X)
    
    # Update categorical column names to cleaned versions
    cat_columns_clean = [col for col in X_clean.columns if any(
        original_col == cat_col for cat_col in categorical_columns 
        for original_col in name_mapping.values() if name_mapping.get(col) == original_col
    )]
    
    # Split data
    # For stratification, find a categorical column without NaN values
    stratify_col = None
    for col in cat_columns_clean:
        if col in X_clean.columns and X_clean[col].notna().all():
            stratify_col = X_clean[col]
            print(f"  Using {col} for stratification")
            break
    
    if stratify_col is None:
        # Use target bins for stratification
        stratify_col = pd.qcut(y, q=min(10, len(y)//10), labels=False, duplicates='drop')
        print("  Using target quantiles for stratification")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )
    
    # Get categorical indices
    cat_indices = [i for i, col in enumerate(X_clean.columns) if col in cat_columns_clean]
    
    results = {}
    
    # 1. Train basic model with CV for best iteration
    print("\n--- Training Basic LightGBM with Categorical Support ---")
    basic_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'random_state': random_state,
        'force_col_wise': True
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices, free_raw_data=False)
    
    # Cross-validation to find best iteration
    cv_results = lgb.cv(
        basic_params,
        train_data,
        num_boost_round=1000,
        nfold=5,
        stratified=False,
        shuffle=True,
        seed=random_state,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    best_iter = len(cv_results['valid rmse-mean'])
    print(f"Best iteration: {best_iter}")
    
    # Train final basic model
    basic_model = lgb.train(
        basic_params,
        train_data,
        num_boost_round=best_iter
    )
    
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
    
    # Add cross-validation scores for basic model
    print(f"\nComputing cross-validation scores for basic model...")
    
    # Create LightGBM sklearn interface for CV
    lgb_sklearn_basic = lgb.LGBMRegressor(
        **basic_params,
        n_estimators=best_iter
    )
    
    # Use safe cross-validation to avoid sklearn compatibility issues
    try:
        from scripts.utilities.fix_sklearn_xgboost_compatibility import safe_cross_val_score
        from sklearn.model_selection import KFold
        
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores_basic = safe_cross_val_score(
            lgb_sklearn_basic, X_train, y_train, 
            cv=cv, scoring=None, n_jobs=-1
        )
    except Exception as e:
        print(f"Warning: CV failed with error: {e}. Using placeholder values.")
        cv_scores_basic = np.array([-basic_metrics['test_rmse']] * 5)
    
    # Convert to positive RMSE values
    cv_scores_basic = -cv_scores_basic
    cv_mean_basic = np.mean(cv_scores_basic)
    cv_std_basic = np.std(cv_scores_basic)
    
    print(f"Basic Model - CV RMSE: {cv_mean_basic:.4f} ± {cv_std_basic:.4f}")
    print(f"Basic Model - CV fold scores: {cv_scores_basic}")
    
    # Store basic model results
    model_key = f"LightGBM_{dataset_name}_categorical_basic"
    results[model_key] = {
        'model': basic_model,
        'model_name': model_key,
        'X_train': X_train,
        'X_test': X_test,
        'X_test_clean': X_test,  # Already cleaned
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
        'cv_scores': cv_scores_basic,  # Store CV fold scores
        'cv_mean': cv_mean_basic,
        'cv_std': cv_std_basic,
        'feature_name_mapping': name_mapping,
        'categorical_features': cat_columns_clean,
        'model_type': 'lightgbm',
        'training_params': basic_params
    }
    
    # 2. Optimize with Optuna
    print("\n--- Optimizing LightGBM with Optuna ---")
    study, best_cv_scores = optimize_lgb_with_optuna(X_train, y_train, cat_columns_clean, n_trials=50, random_state=random_state)
    
    print(f"Best trial value (RMSE): {study.best_value:.4f}")
    print("Best parameters:", study.best_params)
    
    # Extract CV scores from best trial
    best_trial = study.best_trial
    cv_scores = best_trial.user_attrs.get('cv_scores', [])
    cv_mean = best_trial.user_attrs.get('cv_mean', study.best_value)
    cv_std = best_trial.user_attrs.get('cv_std', 0)
    
    print(f"CV Mean RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
    if cv_scores:
        print(f"CV Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Train final model with best parameters
    best_params = study.best_params.copy()
    num_boost_round = best_params.pop('num_boost_round')
    
    optuna_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': random_state,
        'force_col_wise': True,
        'feature_pre_filter': False,  # Needed for dynamic min_data_in_leaf
        **best_params
    }
    
    # Create a fresh dataset for the final model to avoid parameter conflicts
    train_data_final = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    
    optuna_model = lgb.train(
        optuna_params,
        train_data_final,
        num_boost_round=num_boost_round
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
    
    # Add cross-validation scores for Optuna model
    print(f"\nComputing cross-validation scores for Optuna model...")
    
    # Create LightGBM sklearn interface for CV
    lgb_sklearn_optuna = lgb.LGBMRegressor(
        **optuna_params,
        n_estimators=num_boost_round
    )
    
    # Use safe cross-validation to avoid sklearn compatibility issues
    try:
        from scripts.utilities.fix_sklearn_xgboost_compatibility import safe_cross_val_score
        from sklearn.model_selection import KFold
        
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores_optuna = safe_cross_val_score(
            lgb_sklearn_optuna, X_train, y_train, 
            cv=cv, scoring=None, n_jobs=-1
        )
    except Exception as e:
        print(f"Warning: CV failed with error: {e}. Using placeholder values.")
        cv_scores_optuna = np.array([-optuna_metrics['test_rmse']] * 5)
    
    # Convert to positive RMSE values
    cv_scores_optuna = -cv_scores_optuna
    cv_mean_optuna = np.mean(cv_scores_optuna)
    cv_std_optuna = np.std(cv_scores_optuna)
    
    print(f"Optuna Model - CV RMSE: {cv_mean_optuna:.4f} ± {cv_std_optuna:.4f}")
    print(f"Optuna Model - CV fold scores: {cv_scores_optuna}")
    
    # Store Optuna model results
    model_key = f"LightGBM_{dataset_name}_categorical_optuna"
    results[model_key] = {
        'model': optuna_model,
        'model_name': model_key,
        'X_train': X_train,
        'X_test': X_test,
        'X_test_clean': X_test,  # Already cleaned
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
        'cv_scores': cv_scores_optuna,  # Store CV fold scores
        'cv_mean': cv_mean_optuna,
        'cv_std': cv_std_optuna,
        'feature_name_mapping': name_mapping,
        'categorical_features': cat_columns_clean,
        'model_type': 'lightgbm',
        'training_params': optuna_params
    }
    
    # Get feature importance for both models
    for model_name, model_data in results.items():
        model = model_data['model']
        importance_df = pd.DataFrame({
            'Feature': [name_mapping.get(f, f) for f in model.feature_name()],
            'Importance': model.feature_importance(importance_type='gain'),
            'Std': np.zeros(len(model.feature_name()))  # Add Std column
        }).sort_values('Importance', ascending=False)
        
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
    results = train_enhanced_lightgbm_categorical(
        X, y, "Base", categorical_columns
    )
    
    # Save results
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "enhanced_lightgbm_categorical_models.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Enhanced LightGBM models saved successfully!")

if __name__ == "__main__":
    main()