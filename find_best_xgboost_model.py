#!/usr/bin/env python3
"""Find the best XGBoost model and its parameters."""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load models
with open('outputs/models/xgboost_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Calculate metrics for each model
results = []
for name, model_data in models.items():
    if isinstance(model_data, dict) and 'y_test' in model_data and 'y_pred' in model_data:
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Extract parameters if it's an optuna model
        params = {}
        if 'study' in model_data and model_data['study'] is not None:
            params = model_data['study'].best_params
        elif 'best_params' in model_data:
            params = model_data['best_params']
        
        results.append({
            'model': name,
            'RMSE': rmse,
            'R2': r2,
            'has_params': bool(params),
            'params': params
        })

# Create dataframe and sort by RMSE
df = pd.DataFrame(results).sort_values('RMSE')
print('XGBoost Model Performance Ranking:')
print('='*60)
for idx, row in df.iterrows():
    print(f"{row['model']:45} RMSE: {row['RMSE']:.4f} R²: {row['R2']:.4f}")

print('\n' + '='*60)
print(f"BEST MODEL: {df.iloc[0]['model']}")
print(f"RMSE: {df.iloc[0]['RMSE']:.4f}")
print(f"R²: {df.iloc[0]['R2']:.4f}")

# Get best model parameters
best_model_name = df.iloc[0]['model']
best_model_data = models[best_model_name]

print('\n' + '='*60)
print('OPTIMAL HYPERPARAMETERS:')
print('='*60)

if df.iloc[0]['has_params']:
    params = df.iloc[0]['params']
    for param, value in params.items():
        if isinstance(value, float):
            print(f"{param:20} = {value:.6f}")
        else:
            print(f"{param:20} = {value}")
else:
    # Try to extract from the model object
    model = best_model_data.get('model')
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for param in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 
                     'colsample_bytree', 'min_child_weight', 'gamma', 'alpha', 'lambda']:
            if param in params:
                print(f"{param:20} = {params[param]}")

# Save best parameters for reuse
best_params = {
    'model_name': best_model_name,
    'rmse': df.iloc[0]['RMSE'],
    'r2': df.iloc[0]['R2'],
    'params': df.iloc[0]['params'] if df.iloc[0]['has_params'] else {}
}

with open('outputs/best_xgboost_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print(f"\nBest parameters saved to: outputs/best_xgboost_params.pkl")