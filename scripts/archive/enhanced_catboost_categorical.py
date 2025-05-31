#!/usr/bin/env python3
"""
Enhanced CatBoost with native categorical support and Optuna optimization.
Stores CV fold scores during optimization for consistency with other tree models.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import warnings
import pickle
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_catboost_with_optuna(X_train, y_train, categorical_columns, n_trials=50, cv_folds=5, random_state=42):
    """Optimize CatBoost hyperparameters using Optuna with categorical support."""
    
    # Get categorical indices
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_columns]
    
    def objective(trial):
        # Hyperparameter search space
        params = {
            'objective': 'RMSE',
            'eval_metric': 'RMSE',
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': random_state,
            'verbose': False,
            'thread_count': -1,
            'cat_features': cat_indices
        }
        
        # Stratified K-Fold for robust evaluation
        cv_scores = []
        fold_predictions = []
        fold_indices = []
        
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
            
            # Create Pool objects
            train_pool = Pool(
                data=X_fold_train,
                label=y_fold_train,
                cat_features=cat_indices
            )
            
            val_pool = Pool(
                data=X_fold_val,
                label=y_fold_val,
                cat_features=cat_indices
            )
            
            # Create and train model
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_fold_val)
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(fold_rmse)
            
            # Store fold predictions and indices for later analysis
            fold_predictions.append(y_pred)
            fold_indices.append(val_idx)
        
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
    
    return study

def train_enhanced_catboost_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """Train enhanced CatBoost models with categorical support and Optuna optimization."""
    
    print(f"\n🌳 Training Enhanced CatBoost with categorical features on {dataset_name}...")
    print(f"Dataset shape: {X.shape}")
    print(f"Categorical features: {categorical_columns}")
    
    # Handle NaN values
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
    
    # Handle missing values in categorical features
    X_clean = X.copy()
    for cat_feature in categorical_columns:
        if cat_feature in X_clean.columns:
            # Handle categorical columns properly
            if X_clean[cat_feature].dtype.name == 'category':
                # Add 'Unknown' category if needed
                if 'Unknown' not in X_clean[cat_feature].cat.categories:
                    X_clean[cat_feature] = X_clean[cat_feature].cat.add_categories(['Unknown'])
                X_clean[cat_feature] = X_clean[cat_feature].fillna('Unknown')
            else:
                # For non-categorical columns, just fill with 'Unknown'
                X_clean[cat_feature] = X_clean[cat_feature].fillna('Unknown')
    
    # Split data
    # For stratification, find a categorical column without NaN values
    stratify_col = None
    for col in categorical_columns:
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
    cat_indices = [i for i, col in enumerate(X_clean.columns) if col in categorical_columns]
    
    results = {}
    
    # 1. Train basic model with cross-validation
    print("\n--- Training Basic CatBoost with Categorical Support ---")
    basic_params = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_state': random_state,
        'verbose': False,
        'thread_count': -1,
        'cat_features': cat_indices,
        'early_stopping_rounds': 50
    }
    
    # Create Pool objects
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_indices
    )
    
    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=cat_indices
    )
    
    # Train basic model
    basic_model = CatBoostRegressor(**basic_params)
    basic_model.fit(train_pool, eval_set=test_pool, verbose=False)
    
    # Get best iteration
    best_iter = basic_model.get_best_iteration()
    print(f"Best iteration: {best_iter}")
    
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
    
    # Store basic model results
    model_key = f"CatBoost_{dataset_name}_categorical_basic"
    results[model_key] = {
        'model': basic_model,
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
        'model_type': 'catboost',
        'training_params': basic_params
    }
    
    # 2. Optimize with Optuna
    print("\n--- Optimizing CatBoost with Optuna ---")
    study = optimize_catboost_with_optuna(X_train, y_train, categorical_columns, n_trials=50, random_state=random_state)
    
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
    optuna_params = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'random_state': random_state,
        'verbose': False,
        'thread_count': -1,
        'cat_features': cat_indices,
        **study.best_params
    }
    
    optuna_model = CatBoostRegressor(**optuna_params)
    optuna_model.fit(train_pool, eval_set=test_pool, verbose=False)
    
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
    
    # Store Optuna model results
    model_key = f"CatBoost_{dataset_name}_categorical_optuna"
    results[model_key] = {
        'model': optuna_model,
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
        'model_type': 'catboost',
        'training_params': optuna_params
    }
    
    # Get feature importance for both models
    for model_name, model_data in results.items():
        model = model_data['model']
        
        # Get feature importance
        importance_scores = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': X_clean.columns,
            'Importance': importance_scores
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
    results = train_enhanced_catboost_categorical(
        X, y, "Base", categorical_columns
    )
    
    # Save results
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "enhanced_catboost_categorical_models.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Enhanced CatBoost models saved successfully!")

if __name__ == "__main__":
    main()