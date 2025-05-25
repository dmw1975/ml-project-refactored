"""LightGBM models for ESG score prediction with Optuna optimization."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from data_tree_models import get_tree_model_datasets, perform_stratified_split_for_tree_models
from utils import io
# Removed - using tree model specific stratified split

# Cleanup old results
def cleanup_old_results():
    model_file = settings.MODEL_DIR / "lightgbm_models.pkl"
    metrics_file = settings.METRICS_DIR / "lightgbm_metrics.csv"

    # Remove old model file if exists
    if model_file.exists():
        print(f"Removing old model file: {model_file}")
        model_file.unlink()

    # Remove old metrics file if exists
    if metrics_file.exists():
        print(f"Removing old metrics file: {metrics_file}")
        metrics_file.unlink()

# Call the cleanup
cleanup_old_results()


def train_basic_lgb(X, y, dataset_name):
    """Train a basic LightGBM model with default parameters, ensuring sector columns exist."""
    print(f"\n--- Training Basic LightGBM on {dataset_name} ---")

    # Use tree model specific stratified split
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
        X, y, test_size=settings.LIGHTGBM_PARAMS.get('test_size', 0.2), 
        random_state=settings.LIGHTGBM_PARAMS.get('random_state', 42)
    )
    
    # Clean feature names for LightGBM (it doesn't support special JSON characters)
    # Create a mapping of original to cleaned names
    feature_names = X_train.columns.tolist()
    cleaned_feature_names = []
    
    # Create simple, numeric feature names to avoid any issues
    for i in range(len(feature_names)):
        cleaned_feature_names.append(f"feature_{i}")
    
    # Create a copy with cleaned column names
    X_train_clean = X_train.copy()
    X_train_clean.columns = cleaned_feature_names
    X_test_clean = X_test.copy()
    X_test_clean.columns = cleaned_feature_names
    
    # Store the mapping for later reference
    feature_name_mapping = dict(zip(cleaned_feature_names, feature_names))
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train_clean, label=y_train)
    
    # Default parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': settings.LIGHTGBM_PARAMS.get('random_state', 42)
    }
    
    # Train model - using a larger number of trees similar to other models
    num_round = 500  # Increased from 100 to better match XGBoost and CatBoost
    model = lgb.train(params, train_data, num_round)
    
    # Make predictions on cleaned X_test
    y_pred = model.predict(X_test_clean)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Model: {dataset_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  R²  : {r2:.4f}")
    
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
        'feature_names': X.columns.tolist(),
        'feature_name_mapping': feature_name_mapping,
        'cleaned_feature_names': cleaned_feature_names,
        'X_test_clean': X_test_clean,  # Store the cleaned test data for importance calculation
        'model_type': 'LightGBM Basic'
    }


def optimize_lgb_with_optuna(X, y, dataset_name, n_trials=50):
    """Optimize LightGBM hyperparameters using Optuna, reporting mean and std of CV scores."""
    print(f"\n--- Optimizing LightGBM on {dataset_name} with Optuna, using {n_trials} trials ---")
    
    # Create a cleaned version of X for all operations
    feature_names = X.columns.tolist()
    cleaned_feature_names = []
    
    # Create simple, numeric feature names to avoid any issues
    for i in range(len(feature_names)):
        cleaned_feature_names.append(f"feature_{i}")
    
    # Create a copy with cleaned column names
    X_clean = X.copy()
    X_clean.columns = cleaned_feature_names
    
    # Store the mapping for later reference
    feature_name_mapping = dict(zip(cleaned_feature_names, feature_names))
    
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'random_state': settings.LIGHTGBM_PARAMS.get('random_state', 42)
        }
        
        # Also optimize the number of iterations (trees)
        num_trees = trial.suggest_int('num_trees', 100, 1000, step=100)
        
        # Use 5-fold cross-validation
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=settings.LIGHTGBM_PARAMS.get('random_state', 42))
        
        for train_idx, val_idx in kf.split(X_clean):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Train model using the optimized number of trees
            model = lgb.train(params, train_data, num_trees)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            cv_scores.append(mse)
        
        mean_cv_mse = np.mean(cv_scores)
        std_cv_mse = np.std(cv_scores)
        
        # Save extra info into trial attributes
        trial.set_user_attr("mean_cv_mse", mean_cv_mse)
        trial.set_user_attr("std_cv_mse", std_cv_mse)
        
        return mean_cv_mse
    
    # Create a study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best parameters and CV results
    best_params = study.best_params
    best_trial = study.best_trial
    mean_cv_mse = best_trial.user_attrs["mean_cv_mse"]
    std_cv_mse = best_trial.user_attrs["std_cv_mse"]
    
    # Extract optimal number of trees
    best_num_trees = best_params.pop('num_trees', 500)  # Default to 500 if not found
    
    # Add default parameters that weren't optimized
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['verbosity'] = -1
    best_params['boosting_type'] = 'gbdt'
    best_params['random_state'] = settings.LIGHTGBM_PARAMS.get('random_state', 42)
    
    print(f"\nBest Params for {dataset_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MSE: {mean_cv_mse:.4f} ± {std_cv_mse:.4f}")
    
    # Train final model with best parameters
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
        X, y, test_size=settings.LIGHTGBM_PARAMS.get('test_size', 0.2),
        random_state=settings.LIGHTGBM_PARAMS.get('random_state', 42)
    )
    
    # Create clean versions for training and testing
    X_train_clean = X_train.copy()
    X_train_clean.columns = cleaned_feature_names
    X_test_clean = X_test.copy()
    X_test_clean.columns = cleaned_feature_names
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train_clean, label=y_train)
    
    # Train model with the optimal number of trees
    print(f"  Number of trees: {best_num_trees}")
    best_model = lgb.train(best_params, train_data, best_num_trees)
    
    # Predict and evaluate
    y_pred = best_model.predict(X_test_clean)
    
    # Calculate test metrics
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
        'cv_mse': mean_cv_mse,
        'cv_mse_std': std_cv_mse,
        'n_companies': len(X),
        'n_companies_train': len(X_train),
        'n_companies_test': len(X_test),
        'y_test': y_test,
        'y_pred': y_pred,
        'n_features': X.shape[1],
        'feature_names': X.columns.tolist(),
        'feature_name_mapping': feature_name_mapping,
        'cleaned_feature_names': cleaned_feature_names,
        'X_test_clean': X_test_clean,  # Store the cleaned test data for importance calculation
        'X_test': X_test,  # Store original test data with sector information for distribution analysis
        'model_type': 'LightGBM Optuna',
        'study': study  # Include study object for potential further analysis
    }


def train_lightgbm_models(datasets=None, n_trials=50):
    """
    Train and optimize LightGBM models for all datasets.
    """
    print("Loading tree model data...")
    datasets_dict, y = get_tree_model_datasets()
    
    # Define all available datasets
    all_datasets = [
        {'data': datasets_dict['Base'], 'name': 'LightGBM_Base'},
        {'data': datasets_dict['Yeo'], 'name': 'LightGBM_Yeo'},
        {'data': datasets_dict['Base_Random'], 'name': 'LightGBM_Base_Random'},
        {'data': datasets_dict['Yeo_Random'], 'name': 'LightGBM_Yeo_Random'}
    ]
    
    # Filter datasets if specified
    if datasets and 'all' not in datasets:
        selected_datasets = [d for d in all_datasets if d['name'] in datasets]
    else:
        selected_datasets = all_datasets
    
    print("\n" + "="*50)
    print(f"Training LightGBM Models for {len(selected_datasets)} Datasets")
    print("="*50)
    
    model_results = {}
    
    for config in selected_datasets:
        basic_name = f"{config['name']}_basic"
        basic_results = train_basic_lgb(config['data'], y, basic_name)
        model_results[basic_name] = basic_results
        
        optuna_name = f"{config['name']}_optuna"
        optuna_results = optimize_lgb_with_optuna(config['data'], y, optuna_name, n_trials=n_trials)
        model_results[optuna_name] = optuna_results
    
    # Save results
    io.save_model(model_results, "lightgbm_models.pkl", settings.MODEL_DIR)
    
    # Save summary to CSV
    metrics_df = pd.DataFrame([
        {
            'model_name': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'cv_mse': metrics.get('cv_mse', np.nan),
            'cv_mse_std': metrics.get('cv_mse_std', np.nan),
            'n_companies': metrics['n_companies'],
            'n_features': metrics['n_features'],
            'model_type': metrics['model_type']
        }
        for name, metrics in model_results.items()
    ])

    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/lightgbm_metrics.csv", index=False)
    
    print("\nLightGBM models trained and saved successfully.")
    return model_results


if __name__ == "__main__":
    # Run this file directly to train all models
    train_lightgbm_models()