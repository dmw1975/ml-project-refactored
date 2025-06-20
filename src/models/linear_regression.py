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

from src.config import settings
from src.data.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
from src.utils import io

from sklearn.model_selection import train_test_split
import numpy as np

def perform_stratified_split_by_sector(X, y, test_size=0.2, random_state=42):
    """
    Performs a stratified train-test split based on GICS sectors, 
    but preserves ALL original features for model training.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature data with sector one-hot encoding and other KPIs.
    y : pandas.Series
        Target variable (e.g., ESG score).
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : split data
    """
    # Step 1: Identify sector columns
    sector_columns = [col for col in X.columns if col.startswith('gics_sector_') or col.startswith('sector_')]

    if len(sector_columns) == 0:
        raise ValueError("No sector columns found in X!")

    # Step 2: Create sector labels for stratification
    sector_labels = np.zeros(len(X), dtype=int)
    for i, col in enumerate(sector_columns):
        sector_labels[X[col] == 1] = i

    # Step 3: Perform stratified split (only use labels for splitting guidance)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        stratify=sector_labels
    )

    # Step 4: Split the FULL feature set
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Step 5: Print sector distribution for verification
    print("Sector distribution check:")
    for i, col in enumerate(sector_columns):
        train_pct = X_train[col].mean() * 100
        test_pct = X_test[col].mean() * 100
        print(f"{col.replace('gics_sector_', '').replace('sector_', '')}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")

    return X_train, X_test, y_train, y_test



# Cleanup old results
def cleanup_old_results():
    model_file = settings.MODEL_DIR / "linear_regression_models.pkl"
    metrics_file = settings.METRICS_DIR / "linear_regression_metrics.csv"

    if model_file.exists():
        print(f"Removing old model file: {model_file}")
        model_file.unlink()

    if metrics_file.exists():
        print(f"Removing old metrics file: {metrics_file}")
        metrics_file.unlink()

# Call the cleanup
cleanup_old_results()

from src.pipelines.state_manager import get_state_manager



def run_regression_model(X_data, y_data, model_name, random_state=42, test_size=0.2):
    """
    Run standard linear regression on the provided data and evaluate performance.
    
    Returns a dictionary with model performance metrics.
    """
    # Print number of features being used
    print(f"  Features for {model_name}: {X_data.shape[1]}")
    
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
    # Handle both 1D and 2D coefficient arrays
    n_features = model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
    print(f"  Features used: {n_features}")
    
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
        'y_train': y_train,  # Add y_train for baseline comparisons
        'y_test': y_test,
        'y_pred': y_pred,
        'n_features': model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
    }

def train_all_models():
    """
    Train all linear regression model variants and save results.
    """
    # Force reload data module to ensure latest version
    import importlib
    import src.data as data
    importlib.reload(data)
    from src.data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    
    print("Loading data...")
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature sets with direct debug
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # Direct feature count check before continuing
    print(f"\nDIRECT FEATURE COUNT CHECK (AFTER LOADING):")
    print(f"LR_Base column count: {len(LR_Base.columns)}")
    print(f"LR_Yeo column count: {len(LR_Yeo.columns)}")
    
    # If LR_Yeo has less features, fix it directly here
    if len(LR_Yeo.columns) < len(LR_Base.columns):
        print(f"WARNING: LR_Yeo has fewer columns than expected, forcing fix...")
        # Identify all Yeo-transformed columns
        yeo_prefix = 'yeo_joh_'
        yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
        original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
        categorical_columns = [col for col in LR_Base.columns if col not in original_numerical_columns]
        complete_yeo_columns = yeo_transformed_columns + categorical_columns
        LR_Yeo = feature_df[complete_yeo_columns].copy()
        print(f"Fixed LR_Yeo column count: {len(LR_Yeo.columns)}")
    
    # FEATURE COUNT VALIDATION
    print("\n" + "="*50)
    print("FEATURE COUNT VALIDATION")
    print("="*50)
    print(f"LR_Base: {LR_Base.shape[1]} features")
    print(f"LR_Yeo: {LR_Yeo.shape[1]} features")
    
    # Print the first few feature names for each dataset to verify content
    print("\nSample LR_Base features:")
    print(", ".join(LR_Base.columns[:5]))
    print("\nSample LR_Yeo features:")
    print(", ".join(LR_Yeo.columns[:5]))
    
    # Check for Yeo-transformed features
    yeo_transformed_count = sum(1 for col in LR_Yeo.columns if col.startswith('yeo_joh_'))
    print(f"\nYeo-transformed features in LR_Yeo: {yeo_transformed_count}")
    
    # Create versions with random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Check counts after adding random feature
    print(f"\nLR_Base_random: {LR_Base_random.shape[1]} features")
    print(f"LR_Yeo_random: {LR_Yeo_random.shape[1]} features")
    print("="*50)
    
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
    
    # Print feature counts for each dataset before training
    for config in datasets:
        print(f"Dataset {config['name']} has {config['data'].shape[1]} features")
    
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
    # Report model completion
    get_state_manager().increment_completed_models('linear_regression')
    
    # Removed metrics DataFrame and CSV generation - Analysis showed linear_regression_metrics.csv is NEVER READ
    # Metrics are already stored in model PKL files - Date: 2025-01-15
    # The DataFrame was only used for CSV export which has been removed
    
    # Print final feature counts used by each model
    print("\nFeature counts used by each model:")
    for name, metrics in model_results.items():
        n_features = metrics['model'].coef_.shape[-1] if metrics['model'].coef_.ndim > 1 else len(metrics['model'].coef_)
        print(f"{name}: {n_features} features")
    
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
    
    # Removed metrics DataFrame and CSV generation - Analysis showed linear_elasticnet_metrics.csv is NEVER READ
    # Metrics are already stored in model PKL files - Date: 2025-01-15
    # The DataFrame was only used for CSV export which has been removed
    
    print("\nLinear models with ElasticNet parameters trained and saved successfully.")
    return model_results

if __name__ == "__main__":
    # Run this file directly to train all models
    train_all_models()