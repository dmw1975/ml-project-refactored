#!/usr/bin/env python3
"""
Debug script to check why model names are missing in metrics table.
"""

import pickle
from pathlib import Path
import pandas as pd

def check_model_names():
    """Check if models have proper names."""
    
    model_files = {
        'Linear Regression': 'outputs/models/linear_regression_models.pkl',
        'ElasticNet': 'outputs/models/elasticnet_models.pkl',
        'XGBoost': 'outputs/models/xgboost_models.pkl',
        'LightGBM': 'outputs/models/lightgbm_models.pkl',
        'CatBoost': 'outputs/models/catboost_models.pkl'
    }
    
    all_model_names = []
    
    for model_type, filepath in model_files.items():
        print(f"\n{model_type} Models:")
        try:
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
            
            for name, data in models.items():
                # Check if model_name is in the data
                if 'model_name' in data:
                    print(f"  {name}: model_name = '{data['model_name']}'")
                else:
                    print(f"  {name}: NO model_name field!")
                    # Add it if missing
                    data['model_name'] = name
                
                all_model_names.append(name)
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Create a sample metrics table to debug
    print("\n\nSample metrics data structure:")
    metrics_data = []
    
    # Simulate what MetricsTable does
    for model_name in all_model_names[:5]:  # Just first 5 for testing
        model_metrics = {
            'Model': model_name,
            'RMSE': 1.5,
            'MAE': 1.2,
            'R2': 0.85,
            'MSE': 2.25
        }
        metrics_data.append(model_metrics)
        print(f"  {model_metrics}")
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    print("\nDataFrame:")
    print(df)
    
    print("\nDataFrame columns:", df.columns.tolist())
    print("Model column values:", df['Model'].tolist())

if __name__ == "__main__":
    check_model_names()