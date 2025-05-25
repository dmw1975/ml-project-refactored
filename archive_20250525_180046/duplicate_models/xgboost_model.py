"""XGBoost models for ESG score prediction with Optuna optimization."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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
    model_file = settings.MODEL_DIR / "xgboost_models.pkl"
    metrics_file = settings.METRICS_DIR / "xgboost_metrics.csv"

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


def train_basic_xgb(X, y, dataset_name):
    """Train a basic XGBoost model with default parameters, ensuring sector columns exist."""
    print(f"\n--- Training Basic XGBoost on {dataset_name} ---")

    # Use tree model specific stratified split
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
        X, y, test_size=settings.XGBOOST_PARAMS.get('test_size', 0.2), 
        random_state=settings.XGBOOST_PARAMS.get('random_state', 42)
    )
    
    model = XGBRegressor(
        random_state=settings.XGBOOST_PARAMS.get('random_state', 42),
        enable_categorical=True
    )
    model.fit(X_train, y_train)
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
        'model_type': 'XGBoost Basic'
    }


def optimize_xgb_with_optuna(X, y, dataset_name, n_trials=50):
    """Optimize XGBoost hyperparameters using Optuna, reporting mean and std of CV scores."""
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
            
            model = XGBRegressor(**params, enable_categorical=True)
            model.fit(X_train, y_train)
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
    
    print(f"\nBest Params for {dataset_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MSE: {mean_cv_mse:.4f} ± {std_cv_mse:.4f}")
    
    # Train final model with best parameters
    X_train, X_test, y_train, y_test = perform_stratified_split_for_tree_models(
        X, y, test_size=settings.XGBOOST_PARAMS.get('test_size', 0.2),
        random_state=settings.XGBOOST_PARAMS.get('random_state', 42)
    )

    best_model = XGBRegressor(**best_params, enable_categorical=True)
    best_model.fit(X_train, y_train)
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
        'X_test': X_test,  # Store test data for sector distribution analysis
        'model_type': 'XGBoost Optuna',
        'study': study  # Include study object for potential further analysis
    }


def train_xgboost_models(datasets=None, n_trials=50, force_retune=False):
    """
    Train and optimize XGBoost models for all datasets.
    
    Args:
        datasets: List of dataset names to train on
        n_trials: Number of Optuna trials for hyperparameter optimization
        force_retune: If True, skip existing study checks and retrain
    """
    # Check for existing studies unless force_retune is True
    if not force_retune:
        from utils.io import check_existing_studies_for_algorithm, prompt_study_override
        
        existing_studies = check_existing_studies_for_algorithm('xgboost', datasets, n_trials)
        if existing_studies:
            print(f"\n⚠️  Found {len(existing_studies)} existing XGBoost studies.")
            print("💡 Use force_retune=True parameter to override.")
            
            # Show brief summary
            for model_name, study_info in existing_studies.items():
                print(f"   - {model_name}: {study_info['n_trials']} trials, RMSE: {study_info['best_value']:.4f}")
            
            # Load and return existing models instead of retraining
            from utils.io import load_model
            existing_models = load_model("xgboost_models.pkl", settings.MODEL_DIR)
            if existing_models:
                print("✅ Returning existing XGBoost models.")
                return existing_models
    
    print("Loading tree model data...")
    datasets_dict, y = get_tree_model_datasets()
    
    # Define all available datasets
    all_datasets = [
        {'data': datasets_dict['Base'], 'name': 'XGB_Base'},
        {'data': datasets_dict['Yeo'], 'name': 'XGB_Yeo'},
        {'data': datasets_dict['Base_Random'], 'name': 'XGB_Base_Random'},
        {'data': datasets_dict['Yeo_Random'], 'name': 'XGB_Yeo_Random'}
    ]
    
    # Filter datasets if specified
    if datasets and 'all' not in datasets:
        selected_datasets = [d for d in all_datasets if d['name'] in datasets]
    else:
        selected_datasets = all_datasets
    
    print("\n" + "="*50)
    print(f"Training XGBoost Models for {len(selected_datasets)} Datasets")
    print("="*50)
    
    model_results = {}
    
    for config in selected_datasets:
        basic_name = f"{config['name']}_basic"
        basic_results = train_basic_xgb(config['data'], y, basic_name)
        model_results[basic_name] = basic_results
        
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
            'cv_mse': metrics.get('cv_mse', np.nan),
            'cv_mse_std': metrics.get('cv_mse_std', np.nan),
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