"""Linear regression models for ESG score prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from utils import io

def perform_stratified_split_by_sector(X, y, test_size=0.2, random_state=42):
    """
    Performs a stratified train-test split based on GICS sectors.
    """
    # Extract sector columns
    sector_columns = [col for col in X.columns if col.startswith('gics_sector_')]
    
    # Create a sector label for each company (convert one-hot to single label)
    sector_data = X[sector_columns].copy()
    sector_labels = np.zeros(len(X), dtype=int)
    
    for i, col in enumerate(sector_columns):
        # Assign a unique integer to each sector
        sector_labels[sector_data[col] == 1] = i
    
    # Get train and test indices preserving sector proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=sector_labels
    )
    
    # Print sector distributions for verification
    print("Sector distribution check:")
    for i, col in enumerate(sector_columns):
        train_pct = X_train[col].mean() * 100
        test_pct = X_test[col].mean() * 100
        print(f"{col.replace('gics_sector_', '')}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")
    
    return X_train, X_test, y_train, y_test

def run_regression_model(X_data, y_data, model_name, random_state=42, test_size=0.2):
    """
    Run standard linear regression on the provided data and evaluate performance.
    
    Returns a dictionary with model performance metrics.
    """
    # Use stratified split by sector
    X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    
    # Initialize and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print results
    print(f"\nModel: {model_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  R²  : {r2:.4f}")
    
    # Return metrics and model
    return {
        'model_name': model_name,
        'model': model,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'n_companies': len(X_data),
        'n_companies_train': len(X_train),
        'n_companies_test': len(X_test),
        'y_test': y_test,
        'y_pred': y_pred
    }

def train_all_models():
    """
    Train all linear regression model variants and save results.
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
    
    print("\n==================================================")
    print("Running Linear Regression on All Feature Sets")
    print("==================================================")
    
    # Dictionary to store all model results
    model_results = {}
    
    # Define datasets to process
    datasets = [
        {'data': LR_Base, 'name': 'LR_Base'},
        {'data': LR_Yeo, 'name': 'LR_Yeo'},
        {'data': LR_Base_random, 'name': 'LR_Base_Random'},
        {'data': LR_Yeo_random, 'name': 'LR_Yeo_Random'}
    ]
    
    # Train all models
    for config in datasets:
        print(f"\nProcessing dataset: {config['name']}")
        results = run_regression_model(
            config['data'], 
            y, 
            model_name=config['name'],
            random_state=settings.LINEAR_REGRESSION_PARAMS['random_state'],
            test_size=settings.LINEAR_REGRESSION_PARAMS['test_size']
        )
        model_results[config['name']] = results
    
    # Save results
    io.ensure_dir(settings.MODEL_DIR)
    io.save_model(model_results, "linear_regression_models.pkl", settings.MODEL_DIR)
    
    # Save summary to CSV
    metrics_df = pd.DataFrame([
        {
            'model_name': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'n_companies': metrics['n_companies']
        }
        for name, metrics in model_results.items()
    ])
    
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/linear_regression_metrics.csv", index=False)
    
    print("\nLinear regression models trained and saved successfully.")
    return model_results

def train_linear_with_elasticnet_params():
    """Train linear models using the optimal ElasticNet parameters already found."""
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
    
    # Load optimal ElasticNet parameters
    try:
        param_results = io.load_model("elasticnet_params.pkl", settings.MODEL_DIR)
        print("Loaded optimal ElasticNet parameters")
    except Exception as e:
        print(f"Error loading optimal parameters: {e}")
        print("Run ElasticNet parameter search first")
        return
    
    print("\n==================================================")
    print("Training Linear Models with Optimal ElasticNet Parameters")
    print("==================================================")
    
    # Dictionary to store model results
    model_results = {}
    
    # Map of dataset names
    dataset_map = {
        'LR_Base': LR_Base,
        'LR_Yeo': LR_Yeo,
        'LR_Base_Random': LR_Base_random,
        'LR_Yeo_Random': LR_Yeo_random
    }
    
    # Train models with optimal parameters
    for param_result in param_results:
        dataset_name = param_result['dataset']
        alpha, l1_ratio = param_result['best_params']
        
        # Get the corresponding dataset
        X_data = dataset_map.get(dataset_name)
        if X_data is None:
            print(f"Dataset {dataset_name} not found, skipping...")
            continue
        
        model_name = f"{dataset_name}_elasticnet"
        print(f"\nTraining {model_name} with optimal parameters:")
        print(f"  alpha = {alpha}, l1_ratio = {l1_ratio}")
        
        # Use ElasticNet with the optimal parameters
        from sklearn.linear_model import ElasticNet
        
        # Split data
        X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
            X_data, y, 
            test_size=settings.LINEAR_REGRESSION_PARAMS['test_size'],
            random_state=settings.LINEAR_REGRESSION_PARAMS['random_state']
        )
        
        # Train model
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=settings.LINEAR_REGRESSION_PARAMS['random_state'],
            max_iter=10000
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        n_features_used = np.sum(model.coef_ != 0)
        
        # Print results
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE : {mae:.4f}")
        print(f"  MSE : {mse:.4f}")
        print(f"  R²  : {r2:.4f}")
        print(f"  Features used: {n_features_used} out of {len(X_data.columns)}")
        
        # Store results
        model_results[model_name] = {
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
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    # Save results
    io.save_model(model_results, "linear_elasticnet_models.pkl", settings.MODEL_DIR)
    
    # Save summary to CSV
    metrics_df = pd.DataFrame([
        {
            'model_name': name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'alpha': metrics['alpha'],
            'l1_ratio': metrics['l1_ratio'],
            'n_features_used': metrics['n_features_used'],
            'n_companies': metrics['n_companies'],
            'model_type': 'Linear Regression with ElasticNet'
        }
        for name, metrics in model_results.items()
    ])
    
    metrics_df.to_csv(f"{settings.METRICS_DIR}/linear_elasticnet_metrics.csv", index=False)
    
    print("\nLinear models with ElasticNet parameters trained and saved successfully.")
    return model_results

if __name__ == "__main__":
    # Run this file directly to train all models
    train_all_models()