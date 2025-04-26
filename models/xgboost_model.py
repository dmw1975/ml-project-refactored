"""XGBoost models for ESG score prediction with Optuna optimization."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from utils import io
from models.linear_regression import perform_stratified_split_by_sector

def train_basic_xgb(X, y, dataset_name):
    """Train a basic XGBoost model with default parameters."""
    print(f"\n--- Training Basic XGBoost on {dataset_name} ---")
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
        X, y, test_size=settings.XGBOOST_PARAMS.get('test_size', 0.2), 
        random_state=settings.XGBOOST_PARAMS.get('random_state', 42)
    )
    
    model = XGBRegressor(random_state=settings.XGBOOST_PARAMS.get('random_state', 42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print results
    print(f"Model: {dataset_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  R²  : {r2:.4f}")
    
    # Return results in standard format
    return {
        'model_name': dataset_name,
        'model': model,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'n_companies': len(X),
        'n_companies_train': len(X_train),
        'n_companies_test': len(X_test),
        'y_test': y_test,
        'y_pred': y_pred,
        'n_features': X.shape[1],
        'model_type': 'XGBoost Basic'
    }

def optimize_xgb_with_optuna(X, y, dataset_name, n_trials=50):
    """Optimize XGBoost hyperparameters using Optuna."""
    print(f"\n--- Optimizing XGBoost on {dataset_name} with Optuna ---")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': settings.XGBOOST_PARAMS.get('random_state', 42)
        }
        
        # Use 5-fold cross-validation
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=settings.XGBOOST_PARAMS.get('random_state', 42))
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            cv_scores.append(mse)
            
        return np.mean(cv_scores)
    
    # Create a study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\nBest Params for {dataset_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MSE: {best_score:.4f}")
    
    # Train final model with best parameters
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
        X, y, test_size=settings.XGBOOST_PARAMS.get('test_size', 0.2),
        random_state=settings.XGBOOST_PARAMS.get('random_state', 42)
    )
    
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Final evaluation metrics for {dataset_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  R²  : {r2:.4f}")
    
    # Return results in standard format
    return {
        'model_name': dataset_name,
        'model': best_model,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'best_params': best_params,
        'cv_mse': best_score,
        'n_companies': len(X),
        'n_companies_train': len(X_train),
        'n_companies_test': len(X_test),
        'y_test': y_test,
        'y_pred': y_pred,
        'n_features': X.shape[1],
        'model_type': 'XGBoost Optuna',
        'study': study  # Include the study object for visualization
    }

def train_xgboost_models(datasets=None, n_trials=50):
    """
    Train and optimize XGBoost models for all datasets.
    
    Parameters:
    -----------
    datasets : list, optional
        List of dataset names to process. If None, all datasets are processed.
    n_trials : int, default=50
        Number of Optuna trials for optimization
    """
    print("Loading data...")
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature sets
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # Create versions with random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Target variable
    y = score_df
    
    # Define all available datasets
    all_datasets = [
        {'data': LR_Base, 'name': 'XGB_Base'},
        {'data': LR_Yeo, 'name': 'XGB_Yeo'},
        {'data': LR_Base_random, 'name': 'XGB_Base_Random'},
        {'data': LR_Yeo_random, 'name': 'XGB_Yeo_Random'}
    ]
    
    # Filter datasets if specified
    if datasets and 'all' not in datasets:
        selected_datasets = [d for d in all_datasets if d['name'] in datasets]
    else:
        selected_datasets = all_datasets
    
    print("\n" + "="*50)
    print(f"Training XGBoost Models for {len(selected_datasets)} Datasets")
    print("="*50)
    
    # Dictionary to store all results
    model_results = {}
    
    # Train basic models and optimized models
    for config in selected_datasets:
        # Basic model
        basic_name = f"{config['name']}_basic"
        basic_results = train_basic_xgb(config['data'], y, basic_name)
        model_results[basic_name] = basic_results
        
        # Optuna optimized model
        optuna_name = f"{config['name']}_optuna"
        optuna_results = optimize_xgb_with_optuna(config['data'], y, optuna_name, n_trials=n_trials)
        model_results[optuna_name] = optuna_results
    
    # Save results
    io.save_model(model_results, "xgboost_models.pkl", settings.MODEL_DIR)
    
    # Save summary to CSV
    metrics_df = pd.DataFrame([
        {
            'model_name': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'n_companies': metrics['n_companies'],
            'n_features': metrics['n_features'],
            'model_type': metrics['model_type']
        }
        for name, metrics in model_results.items()
    ])
    
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/xgboost_metrics.csv", index=False)
    
    print("\nXGBoost models trained and saved successfully.")
    return model_results

if __name__ == "__main__":
    # Run this file directly to train all models
    train_xgboost_models()