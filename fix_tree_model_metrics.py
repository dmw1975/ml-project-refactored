#!/usr/bin/env python3
"""
Fix tree model metrics by adding top-level metric keys that visualization expects.
"""

import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def fix_model_metrics(model_path):
    """Fix metrics in a model file by adding top-level keys."""
    
    # Load models
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    updated = False
    
    for model_name, model_data in models.items():
        # Add model_name if missing
        if 'model_name' not in model_data:
            model_data['model_name'] = model_name
            updated = True
            
        # Check if we need to add top-level metrics
        if 'RMSE' not in model_data and 'metrics' in model_data:
            # Extract from nested metrics
            metrics = model_data['metrics']
            
            # Add top-level metrics that visualization expects
            if 'test_rmse' in metrics:
                model_data['RMSE'] = metrics['test_rmse']
                model_data['MAE'] = metrics['test_mae']
                # Calculate MSE from RMSE if not present
                if 'test_mse' in metrics:
                    model_data['MSE'] = metrics['test_mse']
                else:
                    model_data['MSE'] = metrics['test_rmse'] ** 2
                model_data['R2'] = metrics['test_r2']
                updated = True
            
            # If y_test and y_pred exist but metrics don't, calculate them
            elif 'y_test' in model_data and 'y_pred' in model_data:
                y_test = model_data['y_test']
                y_pred = model_data['y_pred']
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                model_data['RMSE'] = rmse
                model_data['MAE'] = mae
                model_data['MSE'] = mse
                model_data['R2'] = r2
                updated = True
                
                print(f"  Calculated metrics for {model_name}: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # Save updated models
    if updated:
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)
        print(f"✅ Updated {model_path}")
    else:
        print(f"ℹ️  No updates needed for {model_path}")
    
    return updated

def main():
    """Fix metrics for all tree models."""
    model_dir = Path("outputs/models")
    
    model_files = [
        "xgboost_models.pkl",
        "lightgbm_models.pkl",
        "catboost_models.pkl"
    ]
    
    print("Fixing tree model metrics...")
    
    for model_file in model_files:
        model_path = model_dir / model_file
        if model_path.exists():
            print(f"\nProcessing {model_file}...")
            fix_model_metrics(model_path)
        else:
            print(f"\n⚠️  {model_file} not found")
    
    print("\n✅ Tree model metrics fix complete!")

if __name__ == "__main__":
    main()