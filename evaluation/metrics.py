"""Model evaluation metrics and comparison utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io

def load_all_models():
    """Load all trained models from model directory."""
    # Load linear regression models
    try:
        linear_models = io.load_model("linear_regression_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(linear_models)} linear regression models")
    except:
        print("No linear regression models found")
        linear_models = {}
    
    # Load ElasticNet models
    try:
        elastic_models = io.load_model("elasticnet_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(elastic_models)} ElasticNet models")
    except:
        print("No ElasticNet models found")
        elastic_models = {}

     # Load XGBoost models
    try:
        xgboost_models = io.load_model("xgboost_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(xgboost_models)} XGBoost models")
    except:
        print("No XGBoost models found")
        xgboost_models = {}
    
    # Load LightGBM models
    try:
        lightgbm_models = io.load_model("lightgbm_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(lightgbm_models)} LightGBM models")
    except:
        print("No LightGBM models found")
        lightgbm_models = {}
    
    # Combine all models
    all_models = {**linear_models, **elastic_models, **xgboost_models, **lightgbm_models}
    
       
    return all_models

def calculate_residuals(all_models):
    """Calculate and save residuals for all models."""
    residuals = {}
    
    for model_name, model_data in all_models.items():
        print(f"Processing residuals for {model_name}...")
        # Extract test set predictions and actual values
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        # Ensure y_test and y_pred have the same format and are aligned
        # Convert both to numpy arrays if they aren't already
        if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            y_test_values = y_test.values
        else:
            y_test_values = np.array(y_test)
            
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_pred_values = y_pred.values
        else:
            y_pred_values = np.array(y_pred)
        
        # Ensure they're flattened
        y_test_values = y_test_values.flatten()
        y_pred_values = y_pred_values.flatten()
        
        # Check lengths match
        if len(y_test_values) != len(y_pred_values):
            print(f"WARNING: Length mismatch for {model_name}: y_test={len(y_test_values)}, y_pred={len(y_pred_values)}")
            print("Skipping residual calculation for this model")
            continue
        
        # Calculate residuals using numpy arrays
        res = y_test_values - y_pred_values
        
        # Calculate standardized residuals
        std_res = res / np.std(res)
        
        # Store residuals
        residuals[model_name] = {
            'residuals': res,
            'std_residuals': std_res,
            'abs_residuals': np.abs(res),
            'squared_residuals': res**2,
            'y_test': y_test_values,
            'y_pred': y_pred_values
        }
    
    # Save residuals
    io.save_model(residuals, "model_residuals.pkl", settings.METRICS_DIR)
    
    return residuals

def create_comparison_table(all_models):
    """Create a comparison table of all model metrics."""
    
    # Create DataFrame with metrics
    model_metrics = []
    for model_name, model_data in all_models.items():
        metrics = {
            'model_name': model_name,
            'RMSE': model_data.get('RMSE', np.sqrt(model_data.get('MSE', 0))),
            'MAE': model_data.get('MAE', 0),
            'MSE': model_data.get('MSE', 0),
            'R2': model_data.get('R2', 0),
            'n_companies': model_data.get('n_companies', 0),
            'n_features_used': model_data.get('n_features_used', None),
            'alpha': model_data.get('alpha', None),
            'l1_ratio': model_data.get('l1_ratio', None),
            'model_type': model_data.get('model_type', 
                        'ElasticNet' if 'ElasticNet' in model_name 
                        else 'XGBoost' if 'XGB_' in model_name
                        else 'LightGBM' if 'LightGBM_' in model_name
                        else 'Linear Regression')
        }
        model_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    
    # Save to CSV
    io.ensure_dir(settings.METRICS_DIR)
    metrics_df.to_csv(f"{settings.METRICS_DIR}/all_models_comparison.csv", index=False)
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("=========================")
    print(metrics_df[['model_name', 'RMSE', 'R2', 'model_type']].sort_values('RMSE'))
    
    return metrics_df

def perform_statistical_tests(all_models):
    """Perform statistical tests to compare serious models with Holm-Bonferroni correction."""
    import scipy.stats as stats
    import numpy as np
    import pandas as pd

    # Define allowed serious model types
    allowed_model_types = ['elasticnet', 'xgb', 'catboost', 'lightgbm']

    def is_allowed(name):
        name = name.lower()
        return any(allowed_type in name for allowed_type in allowed_model_types)

    # Filter models
    filtered_models = {name: model for name, model in all_models.items() if is_allowed(name)}
    model_names = list(filtered_models.keys())
    n_models = len(model_names)

    print("\nFiltered serious models for statistical testing:")
    for m in model_names:
        print(f"  - {m}")

    if n_models < 2:
        print("Not enough serious models for statistical testing after filtering.")
        return

    all_tests = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a = model_names[i]
            model_b = model_names[j]

            # Extract and flatten y_test and y_pred for model_a
            y_test_a = np.array(filtered_models[model_a]['y_test']).flatten()
            y_pred_a = np.array(filtered_models[model_a]['y_pred']).flatten()

            # Extract and flatten y_test and y_pred for model_b
            y_test_b = np.array(filtered_models[model_b]['y_test']).flatten()
            y_pred_b = np.array(filtered_models[model_b]['y_pred']).flatten()

            # Calculate squared errors
            se_a = (y_test_a - y_pred_a) ** 2
            se_b = (y_test_b - y_pred_b) ** 2

            if len(se_a) != len(se_b):
                print(f"Warning: Cannot compare {model_a} and {model_b} (different test set sizes)")
                continue

            t_stat, p_value = stats.ttest_rel(se_a, se_b)

            better_model = model_a if t_stat < 0 else model_b

            all_tests.append({
                'model_a': model_a,
                'model_b': model_b,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'better_model': better_model
            })

    # Holm-Bonferroni correction
    all_tests = sorted(all_tests, key=lambda x: x['p_value'])
    m = len(all_tests)

    test_results = []
    for i, test in enumerate(all_tests):
        adj_threshold = 0.05 / (m - i)
        test_results.append({
            'model_a': test['model_a'],
            'model_b': test['model_b'],
            't_statistic': test['t_statistic'],
            'p_value': test['p_value'],
            'adjusted_threshold': adj_threshold,
            'significant': test['p_value'] < adj_threshold,
            'better_model': test['better_model']
        })

    tests_df = pd.DataFrame(test_results)
    tests_df.to_csv(f"{settings.METRICS_DIR}/model_comparison_tests.csv", index=False)

    print("\nStatistical tests completed and saved to model_comparison_tests.csv")
    return tests_df



def evaluate_models():
    """Run all evaluation steps on trained models."""
    print("Loading trained models...")
    all_models = load_all_models()
    
    if not all_models:
        print("No models found. Please train models first.")
        return
    
    print(f"Found {len(all_models)} trained models.")
    
    print("\nCalculating model residuals...")
    residuals = calculate_residuals(all_models)
    
    print("\nCreating model comparison table...")
    metrics_df = create_comparison_table(all_models)
    
    print("\nPerforming statistical tests...")
    tests_df = perform_statistical_tests(all_models)
    
    print("\nEvaluation complete. Results saved to metrics directory.")
    return {
        'all_models': all_models,
        'residuals': residuals,
        'metrics_df': metrics_df,
        'tests_df': tests_df
    }

if __name__ == "__main__":
    # Run the evaluation
    evaluate_models()