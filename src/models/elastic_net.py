"""
ElasticNet models for ESG score prediction.
Now includes Optuna optimization for better hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.data.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from src.utils import io
from src.pipelines.state_manager import get_state_manager
from src.models.linear_regression import perform_stratified_split_by_sector
from scripts.archive.enhanced_elasticnet_optuna import train_enhanced_elasticnet

import os

# Cleanup old results
def cleanup_old_results():
    model_file = settings.MODEL_DIR / "elasticnet_models.pkl"
    metrics_file = settings.METRICS_DIR / "elasticnet_metrics.csv"
    params_file = settings.MODEL_DIR / "elasticnet_params.pkl"

    if model_file.exists():
        print(f"Removing old model file: {model_file}")
        model_file.unlink()

    if metrics_file.exists():
        print(f"Removing old metrics file: {metrics_file}")
        metrics_file.unlink()
        
    if params_file.exists():
        print(f"Removing old params file: {params_file}")
        params_file.unlink()

# Call the cleanup
cleanup_old_results()

def run_elasticnet_model(X_data, y_data, model_name, alpha, l1_ratio, 
                         random_state=42, test_size=0.2,
                         cv_mse=None, cv_mse_std=None):
    """
    Run ElasticNet regression with the provided parameters.
    
    Returns a dictionary with model performance metrics.
    """
    # Use stratified split by sector
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    
    # Initialize and fit model
    model = ElasticNet(
        alpha=alpha, 
        l1_ratio=l1_ratio,
        random_state=random_state,
        max_iter=10000
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Count non-zero coefficients
    n_features_used = np.sum(model.coef_ != 0)
    
    # Print results
    print(f"\nModel: {model_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  R²  : {r2:.4f}")
    print(f"  Parameters: alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}")
    print(f"  Features used: {n_features_used} out of {len(X_data.columns)}")
    
    # Return metrics and model
    return {
        'model_name': model_name,
        'model': model,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'n_features_used': n_features_used,
        'n_companies': len(X_data),
        'n_companies_train': len(X_train),
        'n_companies_test': len(X_test),
        'y_train': y_train,  # Add y_train for baseline comparisons
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': X_data.columns.tolist(),
        'X_test': X_test,  # Store test data for sector distribution analysis
        'cv_mse': cv_mse,             # Store if available
        'cv_mse_std': cv_mse_std       # Store if available
    }

def find_optimal_parameters(X_data, y_data, param_key,
                           alphas=None, l1_ratios=None,
                           random_state=42, test_size=0.2):
    """
    Find optimal ElasticNet parameters using grid search with stratified cross-validation.
    
    Returns the best parameters and cross-validation results.
    """
    # Set more granular hyperparameter grid (similar to notebook)
    if alphas is None:
        alphas = 10 ** np.linspace(-1, 0.2, 15)  # Regularization strengths
    if l1_ratios is None:
        l1_ratios = np.linspace(0, 1, 10)  # L1/L2 mixing ratios
    
    # Extract sector information for stratification
    sector_columns = [col for col in X_data.columns if col.startswith('gics_sector_')]
    sector_data = X_data[sector_columns].copy()
    sector_labels = np.zeros(len(X_data), dtype=int)
    
    for i, col in enumerate(sector_columns):
        sector_labels[sector_data[col] == 1] = i
    
    # Use stratified split
    X_train, X_test, y_train, y_test, sector_train, sector_test = train_test_split(
        X_data, y_data, sector_labels, test_size=test_size, random_state=random_state, stratify=sector_labels
    )
    
    # Verify sector distribution
    print("Sector distribution in train/test sets:")
    for i, col in enumerate(sector_columns):
        sector_name = col.replace('gics_sector_', '')
        train_pct = np.mean(sector_train == i) * 100
        test_pct = np.mean(sector_test == i) * 100
        print(f"  {sector_name}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")
    
    # Set up stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    best_params = None
    best_score = float('inf')
    
    results = []
    
    # Grid search over parameters
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            try:
                cv_scores = []
                
                # Cross-validation with stratification
                for train_idx, val_idx in skf.split(X_train, sector_train):
                    # Create train/validation splits while preserving sector proportions
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Train and evaluate model
                    model = ElasticNet(
                        alpha=alpha, 
                        l1_ratio=l1_ratio,
                        random_state=random_state,
                        max_iter=10000
                    )
                    model.fit(X_cv_train, y_cv_train)
                    y_pred = model.predict(X_cv_val)
                    
                    # Calculate RMSE
                    mse = mean_squared_error(y_cv_val, y_pred)
                    rmse = np.sqrt(mse)
                    cv_scores.append(rmse)
                
                # Average score across folds
                mean_rmse = np.mean(cv_scores)
                
                # Save result
                results.append({
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'mean_rmse': mean_rmse,
                    'std_rmse': np.std(cv_scores),
                    'rmse_folds': cv_scores
                })
                
                # Update best parameters if better
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = (alpha, l1_ratio)
                
            except Exception as e:
                print(f"  Warning: Error with alpha={alpha}, l1_ratio={l1_ratio}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    print(f"Best parameters for {param_key}:")
    print(f"  alpha = {best_params[0]}, l1_ratio = {best_params[1]}")
    print(f"  CV RMSE = {best_score:.4f}")
    
    return best_params, results_df

def train_elasticnet_models(datasets=None, use_optuna=True, n_trials=100):
    """
    Train ElasticNet models with either grid search or Optuna optimization.
    
    Parameters:
    -----------
    datasets : list, optional
        List of dataset names to process. If None, all datasets are processed.
    use_optuna : bool, default=True
        Whether to use Optuna optimization instead of grid search
    n_trials : int, default=100
        Number of Optuna trials (ignored if use_optuna=False)
    """
    # Force reload data module to ensure latest version
    import importlib
    import src.data as data
    importlib.reload(data)
    from src.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    
    print("Loading data...")
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature sets
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # Direct feature count check before continuing
    print(f"\nDIRECT FEATURE COUNT CHECK (AFTER LOADING):")
    print(f"LR_Base column count: {len(LR_Base.columns)}")
    print(f"LR_Yeo column count: {len(LR_Yeo.columns)}")
    
    # If LR_Yeo has fewer features, fix it
    if len(LR_Yeo.columns) < len(LR_Base.columns):
        print(f"WARNING: LR_Yeo has fewer columns than expected, forcing fix...")
        yeo_prefix = 'yeo_joh_'
        yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
        original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
        categorical_columns = [col for col in LR_Base.columns if col not in original_numerical_columns]
        complete_yeo_columns = yeo_transformed_columns + categorical_columns
        LR_Yeo = feature_df[complete_yeo_columns].copy()
        print(f"Fixed LR_Yeo column count: {len(LR_Yeo.columns)}")
    
    # Create versions with random feature
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Target variable
    y = score_df
    
    # Define all available datasets
    all_datasets = [
        {'data': LR_Base, 'name': 'LR_Base'},
        {'data': LR_Yeo, 'name': 'LR_Yeo'},
        {'data': LR_Base_random, 'name': 'LR_Base_Random'},
        {'data': LR_Yeo_random, 'name': 'LR_Yeo_Random'}
    ]
    
    # Filter datasets if specified
    if datasets and 'all' not in datasets:
        selected_datasets = [d for d in all_datasets if d['name'] in datasets]
    else:
        selected_datasets = all_datasets

    if use_optuna:
        print("\n" + "="*50)
        print(f"Training ElasticNet with Optuna Optimization ({n_trials} trials)")
        print("="*50)
        
        model_results = {}
        
        # Train both basic and Optuna models for each dataset
        for config in selected_datasets:
            print(f"\nProcessing dataset: {config['name']}...")
            
            # Use enhanced ElasticNet with Optuna
            results = train_enhanced_elasticnet(
                config['data'],
                y,
                dataset_name=config['name'],
                test_size=settings.ELASTICNET_PARAMS['test_size'],
                random_state=settings.ELASTICNET_PARAMS['random_state'],
                n_trials=n_trials
            )
            
            # Add results to model_results
            for model_name, model_data in results.items():
                model_results[model_name] = model_data
        
        # Save results
        io.save_model(model_results, "elasticnet_models.pkl", settings.MODEL_DIR)
        # Report model completion
        get_state_manager().increment_completed_models('elasticnet')
        
        # Also save parameter results for compatibility
        param_results = []
        for config in selected_datasets:
            # Extract Optuna results
            optuna_key = f"ElasticNet_{config['name']}_optuna"
            if optuna_key in model_results:
                optuna_data = model_results[optuna_key]
                param_results.append({
                    'dataset': config['name'],
                    'best_params': (optuna_data['alpha'], optuna_data['l1_ratio']),
                    'cv_results': None,  # Optuna doesn't use DataFrame results
                    'best_cv_mse': optuna_data.get('cv_mean', optuna_data['MSE']),
                    'best_cv_mse_std': optuna_data.get('cv_std', 0)
                })
        
        io.save_model(param_results, "elasticnet_params.pkl", settings.MODEL_DIR)
        
    else:
        # Original grid search implementation
        print("\n" + "="*50)
        print(f"Finding Optimal ElasticNet Parameters for {len(selected_datasets)} Datasets")
        print("="*50)

        param_results = []

        # First: find optimal parameters
        for config in selected_datasets:
            print(f"\nProcessing dataset: {config['name']}...")
            best_params, results_df = find_optimal_parameters(
                config['data'],
                y,
                param_key=config['name'],
                random_state=settings.ELASTICNET_PARAMS['random_state'],
                test_size=settings.ELASTICNET_PARAMS['test_size']
            )
            
            best_cv_mse = results_df['mean_rmse'].min()
            best_cv_mse_std = results_df.loc[results_df['mean_rmse'].idxmin(), 'std_rmse']
            
            param_results.append({
                'dataset': config['name'],
                'best_params': best_params,
                'cv_results': results_df,
                'best_cv_mse': best_cv_mse,
                'best_cv_mse_std': best_cv_mse_std
            })
        
        # Save parameter results
        io.save_model(param_results, "elasticnet_params.pkl", settings.MODEL_DIR)

        print("\n" + "="*50)
        print("Training ElasticNet Models with Optimal Parameters")
        print("="*50)

        model_results = {}

        # Then: train models using optimal parameters
        for config, param_result in zip(selected_datasets, param_results):
            alpha, l1_ratio = param_result['best_params']
            best_cv_mse = param_result['best_cv_mse']
            best_cv_mse_std = param_result['best_cv_mse_std']
            
            model_name = f"ElasticNet_{config['name']}"
            print(f"\nTraining {model_name}...")

            results = run_elasticnet_model(
                config['data'],
                y,
                model_name=model_name,
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=settings.ELASTICNET_PARAMS['random_state'],
                test_size=settings.ELASTICNET_PARAMS['test_size'],
                cv_mse=best_cv_mse,
                cv_mse_std=best_cv_mse_std
            )

            model_results[model_name] = results

        # Save trained model results
        io.save_model(model_results, "elasticnet_models.pkl", settings.MODEL_DIR)

    # Save summary metrics
    # Removed metrics DataFrame and CSV generation - Analysis showed elasticnet_metrics.csv is NEVER READ
    # Metrics are already stored in model PKL files - Date: 2025-01-15
    # The DataFrame was only used for CSV export which has been removed

    print("\nElasticNet models trained and saved successfully.")
    return model_results

if __name__ == "__main__":
    # Run this file directly to train all models
    train_elasticnet_models()