"""LightGBM models for ESG score prediction with Optuna optimization."""

import numpy as np
import pandas as pd
import lightgbm as lgb
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

import os

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

    # Check and ensure sector columns exist
    sector_cols = [col for col in X.columns if col.startswith('gics_sector_')]
    if not sector_cols:
        print("Warning: No sector columns found in X, adding them from full feature data...")
        full_features = load_features_data()
        sector_cols = [col for col in full_features.columns if col.startswith('gics_sector_')]

        for col in sector_cols:
            if col not in X.columns:
                X[col] = full_features[col]
        print(f"Added {len(sector_cols)} sector columns back to dataset.")

    # Now safe to stratify
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
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
    
    # Train model
    num_round = 100
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
    print(f"\n--- Optimizing LightGBM on {dataset_name} with Optuna ---")
    
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
        
        # Use 5-fold cross-validation
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=settings.LIGHTGBM_PARAMS.get('random_state', 42))
        
        for train_idx, val_idx in kf.split(X_clean):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Train model
            num_round = 100
            model = lgb.train(params, train_data, num_round)
            
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
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
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
    
    # Train model
    num_round = 100
    best_model = lgb.train(best_params, train_data, num_round)
    
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
        'model_type': 'LightGBM Optuna',
        'study': study  # Include study object for potential further analysis
    }


def train_lightgbm_models(datasets=None, n_trials=50):
    """
    Train and optimize LightGBM models for all datasets.
    """
    print("Loading data...")
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature sets
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # IMPORTANT: Verify and ensure all expected Yeo features are included
    # Check if the loaded yeo_columns pickle file has features that aren't in the LR_Yeo DataFrame
    with open("data/pkl/yeo_columns.pkl", 'rb') as f:
        import pickle
        expected_yeo_columns = pickle.load(f)
    
    # Count how many features we have vs how many we should have
    print(f"Expected Yeo columns from pickle: {len(expected_yeo_columns)}")
    print(f"Actual Yeo columns in DataFrame: {len(LR_Yeo.columns)}")
    
    # Check for missing columns and add them if available in the feature_df
    missing_yeo_cols = [col for col in expected_yeo_columns if col not in LR_Yeo.columns]
    if missing_yeo_cols:
        print(f"Warning: {len(missing_yeo_cols)} columns from yeo_columns pickle are missing in LR_Yeo dataset")
        print(f"First few missing: {missing_yeo_cols[:5] if len(missing_yeo_cols) > 5 else missing_yeo_cols}")
        
        # Create a copy first to avoid SettingWithCopyWarning
        LR_Yeo = LR_Yeo.copy()
        
        # Create a dictionary of columns to add for better performance
        columns_to_add = {}
        added_cols = 0
        
        for col in missing_yeo_cols:
            if col in feature_df.columns:
                columns_to_add[col] = feature_df[col]
                added_cols += 1
        
        # Add all columns at once (much more efficient)
        if added_cols > 0:
            LR_Yeo = pd.concat([LR_Yeo, pd.DataFrame(columns_to_add)], axis=1)
            print(f"Added {added_cols} missing columns to LR_Yeo dataset")
    
    # Find sector columns
    sector_columns = [col for col in feature_df.columns if col.startswith('gics_sector_')]
    print(f"Sector columns found: {sector_columns}")

    # Add random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Ensure random datasets also contain sector columns
    for sector_col in sector_columns:
        if sector_col not in LR_Base_random.columns:
            LR_Base_random[sector_col] = feature_df[sector_col]
        if sector_col not in LR_Yeo_random.columns:
            LR_Yeo_random[sector_col] = feature_df[sector_col]
    
    # Target variable
    y = score_df
    
    # Define all available datasets
    all_datasets = [
        {'data': LR_Base, 'name': 'LightGBM_Base'},
        {'data': LR_Yeo, 'name': 'LightGBM_Yeo'},
        {'data': LR_Base_random, 'name': 'LightGBM_Base_Random'},
        {'data': LR_Yeo_random, 'name': 'LightGBM_Yeo_Random'}
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