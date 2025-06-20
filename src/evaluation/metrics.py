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

from src.config import settings
from src.utils import io

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
    
    # Load CatBoost models
    try:
        catboost_models = io.load_model("catboost_models.pkl", settings.MODEL_DIR)
        print(f"Loaded {len(catboost_models)} CatBoost models")
    except:
        print("No CatBoost models found")
        catboost_models = {}
    
    # Combine all models
    all_models = {**linear_models, **elastic_models, **xgboost_models, **lightgbm_models, **catboost_models}
    
       
    return all_models

def calculate_residuals(all_models):
    """Calculate and save residuals for all models."""
    residuals = {}
    
    for model_name, model_data in all_models.items():
        print(f"Processing residuals for {model_name}...")
        
        # Skip if model_data is not a dictionary (e.g., raw Booster objects)
        if not isinstance(model_data, dict):
            print(f"WARNING: Skipping {model_name} - not in expected dictionary format")
            continue
            
        # Check if required fields exist
        if 'y_test' not in model_data:
            print(f"WARNING: Skipping {model_name} - missing y_test data")
            continue
            
        # Extract test set predictions and actual values
        y_test = model_data['y_test']
        # Handle both 'y_pred' and 'y_test_pred' keys for compatibility
        y_pred = model_data.get('y_pred', model_data.get('y_test_pred'))
        
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
    
    # Removed model_residuals.pkl generation - Analysis showed this file is NEVER READ
    # Residuals are calculated on-the-fly from y_test/y_pred in model PKL files - Date: 2025-01-15
    # io.save_model(residuals, "model_residuals.pkl", settings.METRICS_DIR)
    
    return residuals

def create_comparison_table(all_models):
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


def perform_statistical_tests(all_models):
    """Perform statistical tests to compare optimized models with Holm-Bonferroni correction.
    
    Only includes:
    - Optuna-optimized tree models (models with "_optuna" suffix)
    - ElasticNet models (which are always optimized)
    """
    import scipy.stats as stats
    import numpy as np
    import pandas as pd

    def is_allowed(name):
        """Check if model should be included in statistical tests.
        
        Requirements:
        - Include all models with 'optuna' in the name (optimized tree models)
        - Include all ElasticNet models (they are always optimized)
        - Exclude basic (non-optimized) tree models
        """
        # Include optimized tree models and ElasticNet models
        return 'optuna' in name or 'ElasticNet' in name

    # Filter models - also check if they are dictionaries
    filtered_models = {}
    for name, model in all_models.items():
        if is_allowed(name) and isinstance(model, dict) and 'y_test' in model and 'y_pred' in model:
            filtered_models[name] = model
        elif is_allowed(name) and isinstance(model, dict) and 'y_test' in model and 'y_test_pred' in model:
            # Handle alternative key name
            model['y_pred'] = model['y_test_pred']
            filtered_models[name] = model
            
    model_names = list(filtered_models.keys())
    n_models = len(model_names)

    print("\nFiltered optimized models for statistical testing:")
    print(f"Total models after filtering: {n_models}")
    
    # Group by type for better display
    optuna_models = [m for m in model_names if 'optuna' in m]
    elasticnet_models = [m for m in model_names if 'ElasticNet' in m and 'optuna' not in m]
    
    if optuna_models:
        print("\nOptuna-optimized tree models:")
        for m in sorted(optuna_models):
            print(f"  - {m}")
    
    if elasticnet_models:
        print("\nElasticNet models:")
        for m in sorted(elasticnet_models):
            print(f"  - {m}")

    if n_models < 2:
        print("Not enough optimized models for statistical testing after filtering.")
        return

    all_tests = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a = model_names[i]
            model_b = model_names[j]

            # Extract and flatten y_test and y_pred for model_a
            y_test_a = np.array(filtered_models[model_a]['y_test']).flatten()
            y_pred_a = np.array(filtered_models[model_a].get('y_pred', filtered_models[model_a].get('y_test_pred'))).flatten()

            # Extract and flatten y_test and y_pred for model_b
            y_test_b = np.array(filtered_models[model_b]['y_test']).flatten()
            y_pred_b = np.array(filtered_models[model_b].get('y_pred', filtered_models[model_b].get('y_test_pred'))).flatten()

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
    
    # Run baseline comparison if there are models
    print("\nRunning baseline comparisons against random models...")
    try:
        from evaluation.baselines import run_baseline_evaluation
        baseline_comparison, baseline_summary = run_baseline_evaluation(all_models)
        print(f"Baseline comparison complete. Generated {len(baseline_summary)} model comparisons.")
        
        # Print a preview of the results
        if not baseline_summary.empty:
            print("\nBaseline Comparison Preview (Top 5 Models):")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(baseline_summary[['Model', 'RMSE', 'Baseline RMSE', 'Improvement (%)', 'Significant']].head())
    except Exception as e:
        print(f"Error in baseline evaluation: {e}")
        baseline_comparison = None
        baseline_summary = None
    
    print("\nEvaluation complete. Results saved to metrics directory.")
    return {
        'all_models': all_models,
        'residuals': residuals,
        'metrics_df': metrics_df,
        'tests_df': tests_df,
        'baseline_comparison': baseline_comparison,
        'baseline_summary': baseline_summary
    }

if __name__ == "__main__":
    # Run the evaluation
    evaluate_models()