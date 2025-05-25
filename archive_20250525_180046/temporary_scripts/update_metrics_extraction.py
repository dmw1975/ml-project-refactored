#!/usr/bin/env python3
"""
Update metrics extraction to handle all model formats consistently.
"""

import numpy as np
from pathlib import Path

def create_updated_metrics_extraction():
    """Create updated version of create_comparison_table function."""
    
    updated_code = '''def create_comparison_table(all_models):
    """Create a comparison table of all model metrics."""
    
    # Create DataFrame with metrics
    model_metrics = []
    for model_name, model_data in all_models.items():
        # Skip if model_data is not a dictionary
        if not isinstance(model_data, dict):
            print(f"WARNING: Skipping {model_name} in comparison table - not in expected format")
            continue
        
        # Initialize metrics with defaults
        metrics = {
            'Model': model_name,
            'RMSE': np.nan,
            'MAE': np.nan,
            'MSE': np.nan,
            'R2': np.nan,
            'n_companies': 0
        }
        
        # First priority: Check for standard metric keys
        if 'RMSE' in model_data:
            metrics['RMSE'] = model_data['RMSE']
            metrics['MAE'] = model_data.get('MAE', np.nan)
            metrics['MSE'] = model_data.get('MSE', metrics['RMSE']**2)
            metrics['R2'] = model_data.get('R2', np.nan)
        
        # Second priority: Check nested metrics dictionary
        elif 'metrics' in model_data and isinstance(model_data['metrics'], dict):
            m = model_data['metrics']
            # Check for standard keys in metrics
            if 'RMSE' in m:
                metrics['RMSE'] = m['RMSE']
                metrics['MAE'] = m.get('MAE', np.nan)
                metrics['MSE'] = m.get('MSE', metrics['RMSE']**2)
                metrics['R2'] = m.get('R2', np.nan)
            # Check for test_ prefixed keys
            elif 'test_rmse' in m:
                metrics['RMSE'] = m['test_rmse']
                metrics['MAE'] = m.get('test_mae', np.nan)
                metrics['MSE'] = m.get('test_mse', metrics['RMSE']**2)
                metrics['R2'] = m.get('test_r2', np.nan)
        
        # Third priority: Check for test_ prefixed keys at top level
        elif 'test_rmse' in model_data:
            metrics['RMSE'] = model_data['test_rmse']
            metrics['MAE'] = model_data.get('test_mae', np.nan)
            metrics['MSE'] = model_data.get('test_mse', metrics['RMSE']**2)
            metrics['R2'] = model_data.get('test_r2', np.nan)
        
        # Fourth priority: Calculate from y_test and y_pred if available
        elif 'y_test' in model_data and 'y_pred' in model_data:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            
            mse = mean_squared_error(y_test, y_pred)
            metrics['MSE'] = mse
            metrics['RMSE'] = np.sqrt(mse)
            metrics['MAE'] = mean_absolute_error(y_test, y_pred)
            metrics['R2'] = r2_score(y_test, y_pred)
        
        # Get n_companies
        if 'n_companies' in model_data:
            metrics['n_companies'] = model_data['n_companies']
        elif 'y_test' in model_data:
            try:
                metrics['n_companies'] = len(model_data['y_test'])
            except:
                metrics['n_companies'] = 0
        
        # Only add if we have at least RMSE
        if not np.isnan(metrics['RMSE']):
            model_metrics.append(metrics)
        else:
            print(f"WARNING: No metrics found for {model_name}")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(model_metrics)
    
    # Sort by RMSE
    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df
'''
    
    return updated_code


def update_metrics_file():
    """Update the metrics.py file with improved extraction logic."""
    metrics_file = Path("evaluation/metrics.py")
    
    # Read the current file
    with open(metrics_file, 'r') as f:
        content = f.read()
    
    # Find the create_comparison_table function
    import re
    pattern = r'def create_comparison_table\(all_models\):.*?(?=\ndef|\nif __name__|$)'
    
    # Get the updated function
    updated_function = create_updated_metrics_extraction()
    
    # Replace the function
    new_content = re.sub(pattern, updated_function.strip() + '\n\n', content, flags=re.DOTALL)
    
    # Write back
    with open(metrics_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Updated metrics.py with improved extraction logic")


if __name__ == "__main__":
    update_metrics_file()