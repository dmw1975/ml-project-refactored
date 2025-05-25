#!/usr/bin/env python3
"""Fix linear models to store training data for proper baseline calculation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils import io
from config import settings

def fix_linear_regression_models():
    """Re-train linear regression models and ensure training data is stored."""
    print("Fixing Linear Regression models to include training data...")
    
    # Import the modified function
    from models.linear_regression import run_regression_model, perform_stratified_split_by_sector
    from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    
    # Load data
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
    
    # Dictionary to store all model results
    model_results = {}
    
    # Define datasets to process
    datasets = [
        {'data': LR_Base, 'name': 'LR_Base'},
        {'data': LR_Yeo, 'name': 'LR_Yeo'},
        {'data': LR_Base_random, 'name': 'LR_Base_Random'},
        {'data': LR_Yeo_random, 'name': 'LR_Yeo_Random'}
    ]
    
    # Train all models with modified function to store training data
    for config in datasets:
        print(f"\nProcessing dataset: {config['name']}")
        
        # Use stratified split
        X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
            config['data'], 
            y, 
            test_size=settings.LINEAR_REGRESSION_PARAMS['test_size'],
            random_state=settings.LINEAR_REGRESSION_PARAMS['random_state']
        )
        
        # Train model
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store results WITH training data
        results = {
            'model_name': config['name'],
            'model': model,
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'n_companies': len(config['data']),
            'n_companies_train': len(X_train),
            'n_companies_test': len(X_test),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train': y_train,  # ADD THIS
            'X_train': X_train,  # ADD THIS (optional but useful)
            'n_features': model.coef_.shape[0]
        }
        
        model_results[config['name']] = results
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Train samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
    
    # Save updated results
    io.ensure_dir(settings.MODEL_DIR)
    io.save_model(model_results, "linear_regression_models.pkl", settings.MODEL_DIR)
    print("\nLinear regression models updated with training data.")
    return model_results

def fix_elastic_net_models():
    """Fix ElasticNet models to include training data."""
    print("\nFixing ElasticNet models to include training data...")
    
    # Load existing models
    try:
        elastic_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
    except:
        print("ElasticNet models not found. Please train them first.")
        return None
    
    # For each model, we need to re-split the data to get training data
    from data import load_features_data, load_scores_data, get_base_and_yeo_features, add_random_feature
    from models.linear_regression import perform_stratified_split_by_sector
    
    # Load data
    feature_df = load_features_data()
    score_df = load_scores_data()
    
    # Get feature sets
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # Create versions with random features
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Target variable
    y = score_df
    
    # Dataset mapping
    dataset_map = {
        'ElasticNet_LR_Base': LR_Base,
        'ElasticNet_LR_Yeo': LR_Yeo,
        'ElasticNet_LR_Base_Random': LR_Base_random,
        'ElasticNet_LR_Yeo_Random': LR_Yeo_random
    }
    
    # Update each model with training data
    for model_name, model_data in elastic_models.items():
        if model_name in dataset_map:
            X_data = dataset_map[model_name]
            
            # Re-split to get training data
            X_train, X_test, y_train, y_test = perform_stratified_split_by_sector(
                X_data, 
                y, 
                test_size=settings.ELASTICNET_PARAMS['test_size'],
                random_state=settings.ELASTICNET_PARAMS['random_state']
            )
            
            # Add training data to model dictionary
            model_data['y_train'] = y_train
            model_data['X_train'] = X_train
            model_data['n_companies_train'] = len(X_train)
            
            print(f"Updated {model_name} with {len(y_train)} training samples")
    
    # Save updated models
    io.save_model(elastic_models, "elasticnet_models.pkl", settings.MODEL_DIR)
    print("\nElasticNet models updated with training data.")
    return elastic_models

def verify_consistent_datasets():
    """Verify that all models use consistent test set sizes."""
    print("\n" + "="*60)
    print("Verifying dataset consistency across all models")
    print("="*60)
    
    all_models = io.load_all_models()
    
    # Group by test size
    test_sizes = {}
    for model_name, model_data in all_models.items():
        if 'y_test' in model_data and model_data['y_test'] is not None:
            test_size = len(model_data['y_test'])
            if test_size not in test_sizes:
                test_sizes[test_size] = []
            test_sizes[test_size].append(model_name)
    
    print("\nModels grouped by test set size:")
    for size, models in sorted(test_sizes.items()):
        print(f"\nTest size {size}:")
        for model in sorted(models):
            print(f"  - {model}")

if __name__ == "__main__":
    # Fix linear regression models
    fix_linear_regression_models()
    
    # Fix elastic net models
    fix_elastic_net_models()
    
    # Verify consistency
    verify_consistent_datasets()
    
    print("\n✅ Linear models have been updated with training data.")
    print("   You can now re-run the baseline evaluation for accurate comparisons.")