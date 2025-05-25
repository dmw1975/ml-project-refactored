"""CatBoost models for ESG score prediction with Optuna optimization."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
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
    model_file = settings.MODEL_DIR / "catboost_models.pkl"
    metrics_file = settings.METRICS_DIR / "catboost_metrics.csv"

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


def train_basic_catboost(X, y, dataset_name):
    """Train a basic CatBoost model with default parameters, ensuring sector columns exist."""
    print(f"\n--- Training Basic CatBoost on {dataset_name} ---")

    # Use tree model specific stratified split
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
        X, y, test_size=settings.CATBOOST_PARAMS.get('test_size', 0.2), 
        random_state=settings.CATBOOST_PARAMS.get('random_state', 42)
    )
    
    # Mark categorical features (in case there are any)
    cat_features = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
            cat_features.append(col)
    
    # Create CatBoost model with default parameters
    model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=settings.CATBOOST_PARAMS.get('random_state', 42),
        verbose=False  # Set to False to reduce output
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        cat_features=cat_features if cat_features else None,
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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
        'X_test': X_test,  # Store test data for importance calculation
        'model_type': 'CatBoost Basic'
    }


def optimize_catboost_with_optuna(X, y, dataset_name, n_trials=50):
    """Optimize CatBoost hyperparameters using Optuna, reporting mean and std of CV scores."""
    print(f"\n--- Optimizing CatBoost on {dataset_name} with Optuna ---")
    
    # Identify categorical features (if any)
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            cat_features.append(col)
    
    # Define optimization objective
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'loss_function': 'RMSE',
            'random_seed': settings.CATBOOST_PARAMS.get('random_state', 42),
            'verbose': False
        }
        
        # Use 5-fold cross-validation
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=settings.CATBOOST_PARAMS.get('random_state', 42))
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and train model
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                cat_features=cat_features if cat_features else None,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
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
    best_params['loss_function'] = 'RMSE'
    best_params['random_seed'] = settings.CATBOOST_PARAMS.get('random_state', 42)
    best_params['verbose'] = False
    
    print(f"\nBest Params for {dataset_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MSE: {mean_cv_mse:.4f} ± {std_cv_mse:.4f}")
    
    # Train final model with best parameters
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
        X, y, test_size=settings.CATBOOST_PARAMS.get('test_size', 0.2),
        random_state=settings.CATBOOST_PARAMS.get('random_state', 42)
    )

    # Create and train best model
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(
        X_train, y_train,
        cat_features=cat_features if cat_features else None,
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    
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
        'X_test': X_test,  # Store test data for importance calculation
        'model_type': 'CatBoost Optuna',
        'study': study  # Include study object for potential further analysis
    }


def train_catboost_models(datasets=None, n_trials=settings.CATBOOST_PARAMS.get('n_trials', 50)):
    """
    Train and optimize CatBoost models for all datasets.
    """
    print("Loading tree model data...")
    datasets_dict, y = get_tree_model_datasets()
    
    # Define all available datasets
    all_datasets = [
        {'data': datasets_dict['Base'], 'name': 'CatBoost_Base'},
        {'data': datasets_dict['Yeo'], 'name': 'CatBoost_Yeo'},
        {'data': datasets_dict['Base_Random'], 'name': 'CatBoost_Base_Random'},
        {'data': datasets_dict['Yeo_Random'], 'name': 'CatBoost_Yeo_Random'}
    ]
    
    # Filter datasets if specified
    if datasets and 'all' not in datasets:
        selected_datasets = [d for d in all_datasets if d['name'] in datasets]
    else:
        selected_datasets = all_datasets
    
    print("\n" + "="*50)
    print(f"Training CatBoost Models for {len(selected_datasets)} Datasets")
    print("="*50)
    
    model_results = {}
    
    for config in selected_datasets:
        basic_name = f"{config['name']}_basic"
        basic_results = train_basic_catboost(config['data'], y, basic_name)
        model_results[basic_name] = basic_results
        
        optuna_name = f"{config['name']}_optuna"
        optuna_results = optimize_catboost_with_optuna(config['data'], y, optuna_name, n_trials=n_trials)
        model_results[optuna_name] = optuna_results
    
    # Save results
    io.save_model(model_results, "catboost_models.pkl", settings.MODEL_DIR)
    
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
    metrics_df.to_csv(f"{settings.METRICS_DIR}/catboost_metrics.csv", index=False)
    
    print("\nCatBoost models trained and saved successfully.")
    return model_results


if __name__ == "__main__":
    # Run this file directly to train all models
    train_catboost_models()