#!/usr/bin/env python3
"""
Enhanced ElasticNet with Optuna optimization.
Provides both basic and Optuna-optimized ElasticNet models for comparison.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import warnings
import pickle
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.models.linear_regression import perform_stratified_split_by_sector

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_elasticnet_with_optuna(X_train, y_train, sector_labels, n_trials=100, cv_folds=5, random_state=42):
    """
    Optimize ElasticNet hyperparameters using Optuna with stratified cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        sector_labels: Sector labels for stratification
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        study: Optuna study object
        best_cv_scores: Cross-validation scores from best trial
    """
    
    def objective(trial):
        # Hyperparameter search space
        alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
        # Additional parameters that grid search doesn't explore
        max_iter = trial.suggest_int('max_iter', 5000, 20000, step=5000)
        tol = trial.suggest_float('tol', 1e-5, 1e-3, log=True)
        
        # Stratified K-Fold for robust evaluation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, sector_labels)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Train model
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state
            )
            
            try:
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                mse = mean_squared_error(y_fold_val, y_pred)
                cv_scores.append(mse)
            except Exception as e:
                # If model fails to converge, return high penalty
                return float('inf')
        
        # Store CV results in trial
        trial.set_user_attr('cv_scores', cv_scores)
        trial.set_user_attr('cv_mean', np.mean(cv_scores))
        trial.set_user_attr('cv_std', np.std(cv_scores))
        
        # Check if random feature is being used (if it exists)
        if 'random_feature' in X_train.columns:
            random_idx = X_train.columns.get_loc('random_feature')
            # Train on full training set to check coefficients
            full_model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, 
                tol=tol, random_state=random_state
            )
            full_model.fit(X_train, y_train)
            
            # Penalize if random feature has non-zero coefficient
            if abs(full_model.coef_[random_idx]) > 1e-10:
                trial.set_user_attr('uses_random_feature', True)
                # Add small penalty to discourage but not prohibit
                return np.mean(cv_scores) * 1.01
            else:
                trial.set_user_attr('uses_random_feature', False)
        
        return np.mean(cv_scores)
    
    # Create study with TPE sampler for efficient search
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=random_state)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    # Extract best CV scores
    best_cv_scores = None
    if study.best_trial:
        best_cv_scores = study.best_trial.user_attrs.get('cv_scores', None)
    
    return study, best_cv_scores


def train_enhanced_elasticnet(X, y, dataset_name, test_size=0.2, random_state=42, n_trials=100):
    """
    Train both basic and Optuna-optimized ElasticNet models.
    
    Args:
        X: Features DataFrame
        y: Target Series
        dataset_name: Name of the dataset
        test_size: Test set proportion
        random_state: Random state
        n_trials: Number of Optuna trials
        
    Returns:
        dict: Results for both basic and Optuna models
    """
    
    print(f"\n📈 Training Enhanced ElasticNet on {dataset_name}...")
    print(f"Dataset shape: {X.shape}")
    
    # Get sector labels for stratification
    sector_columns = [col for col in X.columns if col.startswith('gics_sector_')]
    sector_data = X[sector_columns].copy()
    sector_labels = np.zeros(len(X), dtype=int)
    
    for i, col in enumerate(sector_columns):
        sector_labels[sector_data[col] == 1] = i
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Get train sector labels
    # Map from company names back to numeric indices
    company_to_idx = {company: idx for idx, company in enumerate(X.index)}
    train_numeric_indices = [company_to_idx[company] for company in X_train.index]
    train_sector_labels = sector_labels[train_numeric_indices]
    
    results = {}
    
    # Optimize with Optuna only (no basic models)
    print(f"\n--- Optimizing ElasticNet with Optuna ({n_trials} trials) ---")
    
    study, best_cv_scores = optimize_elasticnet_with_optuna(
        X_train, y_train, train_sector_labels, n_trials=n_trials, random_state=random_state
    )
    
    print(f"Best trial value (MSE): {study.best_value:.4f}")
    print("Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Extract CV information
    best_trial = study.best_trial
    cv_scores = best_trial.user_attrs.get('cv_scores', [])
    cv_mean = best_trial.user_attrs.get('cv_mean', study.best_value)
    cv_std = best_trial.user_attrs.get('cv_std', 0)
    uses_random = best_trial.user_attrs.get('uses_random_feature', False)
    
    print(f"CV Mean MSE: {cv_mean:.4f} ± {cv_std:.4f}")
    if 'random_feature' in X.columns:
        print(f"Uses random feature: {uses_random}")
    
    # Train final model with best parameters
    best_params = study.best_params
    optuna_model = ElasticNet(
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        max_iter=best_params.get('max_iter', 10000),
        tol=best_params.get('tol', 1e-4),
        random_state=random_state
    )
    
    # Fit and evaluate
    optuna_model.fit(X_train, y_train)
    y_pred_optuna = optuna_model.predict(X_test)
    y_train_pred_optuna = optuna_model.predict(X_train)
    
    # Calculate metrics
    optuna_metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_optuna)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_optuna)),
        'train_r2': r2_score(y_train, y_train_pred_optuna),
        'test_r2': r2_score(y_test, y_pred_optuna),
        'train_mae': mean_absolute_error(y_train, y_train_pred_optuna),
        'test_mae': mean_absolute_error(y_test, y_pred_optuna),
        'n_features_used': np.sum(optuna_model.coef_ != 0)
    }
    
    print(f"\nOptuna Model - Test RMSE: {optuna_metrics['test_rmse']:.4f}")
    print(f"Optuna Model - Test R²: {optuna_metrics['test_r2']:.4f}")
    print(f"Optuna Model - Features used: {optuna_metrics['n_features_used']} / {X.shape[1]}")
    
    # Store Optuna model results
    model_key = f"ElasticNet_{dataset_name}_optuna"
    results[model_key] = {
        'model': optuna_model,
        'model_name': model_key,
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
        'alpha': best_params['alpha'],
        'l1_ratio': best_params['l1_ratio'],
        'n_features_used': optuna_metrics['n_features_used'],
        'best_params': study.best_params,
        'study': study,
        'cv_scores': cv_scores,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'feature_names': X.columns.tolist(),
        'model_type': 'elasticnet'
    }
    
    # Get feature importance (coefficients)
    for model_name, model_data in results.items():
        model = model_data['model']
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_),  # Use absolute values for importance
            'Coefficient': model.coef_,  # Keep actual coefficients too
            'Std': np.zeros(len(model.coef_))  # No std for single model
        }).sort_values('Importance', ascending=False)
        
        model_data['feature_importance'] = importance_df
        
        # Save feature importance
        output_dir = Path("outputs/feature_importance")
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_dir / f"{model_name}_importance.csv", index=False)
        
        print(f"\nTop 10 features for {model_name}:")
        print(importance_df.head(10)[['Feature', 'Importance', 'Coefficient']])
    
    return results


def main():
    """Test the enhanced implementation."""
    from data import load_features_data, load_scores_data, get_base_and_yeo_features
    
    # Load data
    print("Loading data...")
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get base features for testing
    LR_Base, _, _, _ = get_base_and_yeo_features(feature_df)
    
    # Train on base dataset
    results = train_enhanced_elasticnet(
        LR_Base, score_df, "LR_Base", n_trials=50
    )
    
    # Save results
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "enhanced_elasticnet_models.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Enhanced ElasticNet models saved successfully!")


if __name__ == "__main__":
    main()