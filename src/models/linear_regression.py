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
from src.data.train_test_split import get_or_create_split

def perform_stratified_split_by_sector(X, y, test_size=0.2, random_state=42):
    """
    Performs a stratified train-test split based on GICS sectors, 
    but preserves ALL original features for model training.
    
    This function now uses the unified train/test split mechanism to ensure
    consistency across all model types.
    
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
    # Use unified split mechanism
    try:
        # Try to use the unified split
        # Note: X should have issuer_name as index from load_linear_models_data()
        X_train, X_test, y_train, y_test = get_or_create_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify_column='gics_sector'  # Changed from 'sector' to match one-hot column pattern
        )
        
        # Print sector distribution for verification
        sector_columns = [col for col in X.columns if col.startswith('gics_sector_') or col.startswith('sector_')]
        if sector_columns:
            print("Sector distribution check:")
            for col in sector_columns:
                train_pct = X_train[col].mean() * 100
                test_pct = X_test[col].mean() * 100
                print(f"{col.replace('gics_sector_', '').replace('sector_', '')}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")
                
    except Exception as e:
        # No fallback - we want to ensure unified split is used
        print(f"ERROR: Failed to use unified split: {e}")
        raise RuntimeError(f"Must use unified train/test split for consistency. Error: {e}")

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
    # Use the new data loading with issuer_name indices
    import importlib
    import src.data.data_categorical as data_cat
    import src.data as data
    importlib.reload(data_cat)
    importlib.reload(data)
    from src.data.data_categorical import load_linear_models_data
    from src.data import add_random_feature
    
    print("Loading data with issuer_name indices...")
    feature_df, y = load_linear_models_data()
    
    print(f"Loaded data: {feature_df.shape[0]} companies, {feature_df.shape[1]} features")
    print(f"Index type: {type(feature_df.index)}")
    print(f"Index name: {feature_df.index.name}")
    print(f"Sample indices: {list(feature_df.index[:5])}")
    
    # Create feature sets directly to preserve index
    # Get base features (non-Yeo columns)
    base_columns = [col for col in feature_df.columns if not col.startswith('yeo_joh_')]
    LR_Base = feature_df[base_columns].copy()
    
    # Get Yeo features (Yeo columns + categorical columns) - using ElasticNet's correct approach
    yeo_prefix = 'yeo_joh_'
    yeo_transformed_columns = [col for col in feature_df.columns if col.startswith(yeo_prefix)]
    original_numerical_columns = [col.replace(yeo_prefix, '') for col in yeo_transformed_columns]
    categorical_columns = [col for col in base_columns if col not in original_numerical_columns]
    
    # Create LR_Yeo with yeo-transformed numerical + categorical columns
    complete_yeo_columns = yeo_transformed_columns + categorical_columns
    LR_Yeo = feature_df[complete_yeo_columns].copy()
    
    print(f"Created LR_Base with {len(base_columns)} columns, index: {LR_Base.index.name}")
    print(f"Created LR_Yeo with {len(complete_yeo_columns)} columns, index: {LR_Yeo.index.name}")
    
    # Categorical columns are already included in the feature sets
    
    # Direct feature count check before continuing
    print(f"\nDIRECT FEATURE COUNT CHECK (AFTER LOADING):")
    print(f"LR_Base column count: {len(LR_Base.columns)}")
    print(f"LR_Yeo column count: {len(LR_Yeo.columns)}")
    
    
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
    
    # Target variable already loaded with correct indices
    # y is already loaded from load_linear_models_data()
    
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

if __name__ == "__main__":
    # Run this file directly to train all models
    train_all_models()