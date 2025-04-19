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
    
    # Combine all models
    all_models = {**linear_models, **elastic_models}
    
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
            'model_type': 'ElasticNet' if 'ElasticNet' in model_name else 'Linear Regression'
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
    """Perform statistical tests to compare models with Holm-Bonferroni correction."""
    import scipy.stats as stats
    import numpy as np
    
    # We'll compare models using paired t-tests on squared errors
    model_names = list(all_models.keys())
    n_models = len(model_names)
    
    # Collect all p-values and corresponding test information
    all_tests = []
    
    # Perform pairwise comparisons
    for i in range(n_models):
        for j in range(i+1, n_models):
            model_a = model_names[i]
            model_b = model_names[j]
            
            # Get predictions for both models
            y_test_a = all_models[model_a]['y_test']
            y_pred_a = all_models[model_a]['y_pred']
            y_test_b = all_models[model_b]['y_test']
            y_pred_b = all_models[model_b]['y_pred']
            
            # Convert to numpy arrays for consistent handling
            if isinstance(y_test_a, pd.Series) or isinstance(y_test_a, pd.DataFrame):
                y_test_a = y_test_a.values
            if isinstance(y_pred_a, pd.Series) or isinstance(y_pred_a, pd.DataFrame):
                y_pred_a = y_pred_a.values
            if isinstance(y_test_b, pd.Series) or isinstance(y_test_b, pd.DataFrame):
                y_test_b = y_test_b.values
            if isinstance(y_pred_b, pd.Series) or isinstance(y_pred_b, pd.DataFrame):
                y_pred_b = y_pred_b.values
            
            # Make sure all arrays are 1D
            y_test_a = y_test_a.flatten()
            y_pred_a = y_pred_a.flatten()
            y_test_b = y_test_b.flatten()
            y_pred_b = y_pred_b.flatten()
            
            # Calculate squared errors
            se_a = (y_test_a - y_pred_a)**2
            se_b = (y_test_b - y_pred_b)**2
            
            # Now we need to make sure we're comparing the same test samples
            # If models were trained on different splits, this will require realignment
            if len(se_a) != len(se_b):
                print(f"Warning: Models {model_a} and {model_b} have different test set sizes.")
                print(f"  {model_a}: {len(se_a)}, {model_b}: {len(se_b)}")
                print(f"Cannot perform statistical comparison between them.")
                continue
            
            # Debug output
            print(f"Comparing {model_a} vs {model_b}: test set size = {len(se_a)}")
            
            # Paired t-test on squared errors
            try:
                t_stat, p_value = stats.ttest_rel(se_a, se_b)
                
                # For non-scalar results (which should not happen with properly flattened arrays)
                if hasattr(t_stat, 'size') and t_stat.size > 1:
                    print(f"Warning: t_stat has size {t_stat.size}, expected scalar. Using mean.")
                    t_stat_float = float(np.mean(t_stat))
                    p_value_float = float(np.mean(p_value))
                else:
                    # Normal scalar handling
                    t_stat_float = float(t_stat)
                    p_value_float = float(p_value)
                
                # Determine which model is better
                better_model = model_a if t_stat_float < 0 else model_b if t_stat_float > 0 else 'Equal'
                
                # Add to all tests
                all_tests.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    't_statistic': t_stat_float,
                    'p_value': p_value_float,
                    'better_model': better_model
                })
                
                print(f"  Result: t={t_stat_float:.4f}, p={p_value_float:.4f}")
                
            except Exception as e:
                print(f"Error comparing {model_a} and {model_b}: {str(e)}")
                # Print more debugging info
                print(f"  Shapes: se_a={se_a.shape}, se_b={se_b.shape}")
                print(f"  Types: se_a={type(se_a)}, se_b={type(se_b)}")
                print(f"  Values: se_a[0]={se_a[0] if len(se_a) > 0 else 'empty'}, se_b[0]={se_b[0] if len(se_b) > 0 else 'empty'}")
                continue
    
    # If we have no valid tests, return empty DataFrame
    if not all_tests:
        print("No valid statistical tests could be performed.")
        return pd.DataFrame()
    
    # Apply Holm-Bonferroni correction
    # Sort tests by p-value (ascending)
    all_tests = sorted(all_tests, key=lambda x: x['p_value'])
    m = len(all_tests)  # Total number of tests
    
    # Store final test results
    test_results = []
    
    # Apply sequential correction
    for i, test in enumerate(all_tests):
        # Holm-Bonferroni adjusted threshold
        adj_threshold = 0.05 / (m - i)
        
        # Store in test results with adjusted significance
        test_results.append({
            'model_a': test['model_a'],
            'model_b': test['model_b'],
            't_statistic': test['t_statistic'],
            'p_value': test['p_value'],
            'adjusted_threshold': adj_threshold,
            'significant': test['p_value'] < adj_threshold,
            'better_model': test['better_model']
        })
    
    # Convert to DataFrame
    tests_df = pd.DataFrame(test_results)
    
    # Save to CSV
    tests_df.to_csv(f"{settings.METRICS_DIR}/model_comparison_tests.csv", index=False)
    
    # Print summary of significant differences
    print("\nSignificant Model Differences (with Holm-Bonferroni correction):")
    print("=============================================================")
    sig_tests = tests_df[tests_df['significant'] == True] if not tests_df.empty else tests_df
    if not sig_tests.empty:
        for _, row in sig_tests.iterrows():
            better = row['better_model']
            worse = row['model_a'] if better == row['model_b'] else row['model_b']
            p_val = row['p_value']
            adj_threshold = row['adjusted_threshold']
            print(f"{better} is significantly better than {worse} (p={p_val:.4f}, threshold={adj_threshold:.4f})")
    else:
        print("No significant differences found between models after correction.")
    
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