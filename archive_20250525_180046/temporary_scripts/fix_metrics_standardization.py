#!/usr/bin/env python3
"""
Fix metrics standardization across all models.
This script updates all model files to ensure consistent metric storage.
"""

import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def standardize_model_metrics(model_data):
    """
    Standardize metrics in a model dictionary to ensure all models have
    RMSE, MAE, MSE, and R2 in the same format.
    """
    # If model_data is not a dict, return as is
    if not isinstance(model_data, dict):
        return model_data
    
    # Check if this is a model with metrics
    if 'y_test' not in model_data or 'y_pred' not in model_data:
        return model_data
    
    # Calculate standard metrics
    y_test = model_data['y_test']
    y_pred = model_data['y_pred']
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Update with standard keys
    model_data['RMSE'] = rmse
    model_data['MAE'] = mae
    model_data['MSE'] = mse
    model_data['R2'] = r2
    
    # If there's a nested metrics dict, update it too
    if 'metrics' in model_data and isinstance(model_data['metrics'], dict):
        model_data['metrics']['RMSE'] = rmse
        model_data['metrics']['MAE'] = mae
        model_data['metrics']['MSE'] = mse
        model_data['metrics']['R2'] = r2
        
        # Also keep the old format for backward compatibility
        model_data['metrics']['test_rmse'] = rmse
        model_data['metrics']['test_mae'] = mae
        model_data['metrics']['test_mse'] = mse
        model_data['metrics']['test_r2'] = r2
    
    # Handle CV metrics if available
    if 'cv_mse' in model_data and not np.isnan(model_data.get('cv_mse', np.nan)):
        model_data['cv_rmse'] = np.sqrt(model_data['cv_mse'])
        if 'cv_mse_std' in model_data:
            # Approximate std of RMSE from std of MSE
            model_data['cv_rmse_std'] = model_data['cv_mse_std'] / (2 * np.sqrt(model_data['cv_mse']))
    
    return model_data


def fix_model_file(filepath):
    """Fix metrics in a model file."""
    if not filepath.exists():
        return False
        
    try:
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        
        # Handle single model or dict of models
        if isinstance(models, dict):
            for key in models:
                models[key] = standardize_model_metrics(models[key])
        else:
            models = standardize_model_metrics(models)
        
        # Save back
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"✅ Fixed metrics in {filepath.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {filepath.name}: {e}")
        return False


def main():
    """Fix all model files in the outputs/models directory."""
    from config import settings
    
    model_dir = settings.MODEL_DIR
    
    # List of model files to fix
    model_files = [
        'linear_regression_models.pkl',
        'elasticnet_models.pkl',
        'xgboost_models.pkl',
        'lightgbm_models.pkl',
        'catboost_models.pkl',
        'lightgbm_categorical_models.pkl',
        'xgboost_categorical_models.pkl',
        'catboost_categorical_models.pkl',
    ]
    
    # Also check for individual model files
    for file in model_dir.glob('*.pkl'):
        if file.name not in model_files:
            model_files.append(file.name)
    
    print("Fixing metrics standardization in all model files...")
    print("=" * 60)
    
    fixed_count = 0
    for model_file in model_files:
        filepath = model_dir / model_file
        if fix_model_file(filepath):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} model files")
    
    # Now regenerate the metrics comparison CSV
    print("\nRegenerating metrics comparison CSV...")
    from evaluation.metrics import evaluate_models
    evaluate_models()
    
    print("\n✅ Metrics standardization complete!")


if __name__ == "__main__":
    main()