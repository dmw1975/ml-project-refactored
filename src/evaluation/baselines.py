"""
Module for handling baseline model comparisons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings


def generate_random_baseline(actual_values, min_val=0, max_val=10, seed=42):
    """
    Generate random baseline predictions within a specified range.
    
    Parameters
    ----------
    actual_values : array-like
        The actual target values, used to determine the number of predictions
    min_val : float, optional
        Minimum value for random predictions (default: 0)
    max_val : float, optional
        Maximum value for random predictions (default: 10)
    seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    np.ndarray
        Random predictions with same shape as actual_values
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Convert actual_values to numpy array if it's not already
    if isinstance(actual_values, pd.Series) or isinstance(actual_values, pd.DataFrame):
        actual_values = actual_values.values
    
    # Get the length of the actual values
    n_samples = len(actual_values)
    
    # Generate random predictions within the specified range
    random_preds = np.random.uniform(min_val, max_val, n_samples)
    
    return random_preds


def generate_mean_baseline(train_values):
    """
    Generate mean baseline predictions using training set mean.
    
    Parameters
    ----------
    train_values : array-like
        The training set target values, used to calculate the mean
        
    Returns
    -------
    np.ndarray
        Mean predictions with same shape as actual_values (test set)
    """
    # Convert train_values to numpy array if it's not already
    if isinstance(train_values, pd.Series) or isinstance(train_values, pd.DataFrame):
        train_values = train_values.values
    
    # Calculate mean of the training values
    mean_value = np.mean(train_values)
    
    # Generate predictions with the mean value
    # (same length as train_values for now, will be adjusted in the calling function)
    mean_preds = np.full(len(train_values), mean_value)
    
    return mean_value, mean_preds


def generate_median_baseline(train_values):
    """
    Generate median baseline predictions using training set median.
    
    Parameters
    ----------
    train_values : array-like
        The training set target values, used to calculate the median
        
    Returns
    -------
    np.ndarray
        Median predictions with same shape as actual_values (test set)
    """
    # Convert train_values to numpy array if it's not already
    if isinstance(train_values, pd.Series) or isinstance(train_values, pd.DataFrame):
        train_values = train_values.values
    
    # Calculate median of the training values
    median_value = np.median(train_values)
    
    # Generate predictions with the median value
    # (same length as train_values for now, will be adjusted in the calling function)
    median_preds = np.full(len(train_values), median_value)
    
    return median_value, median_preds


def calculate_baseline_comparison(
    actual,
    model_predictions,
    baseline_predictions,
    model_name,
    output_path=None,
    baseline_type="Random"
):
    """
    Calculate and compare model performance against a baseline.
    
    Parameters
    ----------
    actual : array-like
        The actual target values
    model_predictions : array-like
        The model's predictions
    baseline_predictions : array-like
        The baseline model's predictions (random, mean, or median)
    model_name : str
        Name of the model
    output_path : str, optional
        Path to save the results to CSV
    baseline_type : str, optional
        Type of baseline being compared (Random, Mean, or Median)
        
    Returns
    -------
    dict
        Dictionary containing performance metrics
    """
    # Calculate metrics
    model_rmse = np.sqrt(mean_squared_error(actual, model_predictions))
    baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_predictions))
    
    # Calculate MAE
    model_mae = np.mean(np.abs(actual - model_predictions))
    baseline_mae = np.mean(np.abs(actual - baseline_predictions))
    
    # Calculate improvement percentage
    improvement_pct = (baseline_rmse - model_rmse) / baseline_rmse * 100
    
    # Calculate R²
    model_r2 = r2_score(actual, model_predictions)
    baseline_r2 = r2_score(actual, baseline_predictions)
    
    # Calculate predictive power (normalized improvement)
    predictive_power = (baseline_rmse - model_rmse) / baseline_rmse
    
    # Statistical significance (t-test of residuals)
    model_residuals = actual - model_predictions
    baseline_residuals = actual - baseline_predictions
    t_test = sm.stats.ttest_ind(model_residuals, baseline_residuals)
    p_value = t_test[1]
    significant = p_value < 0.05
    
    # Prepare results
    results = {
        'Model': model_name,
        'Baseline Type': baseline_type,
        'RMSE': model_rmse,
        'Baseline RMSE': baseline_rmse,
        'MAE': model_mae,
        'Baseline MAE': baseline_mae,
        'Improvement (%)': improvement_pct,
        'p-value': p_value,
        'Significant': significant,
        'R²': model_r2,
        'Baseline R²': baseline_r2,
        'Predictive Power': predictive_power
    }
    
    # Save to CSV if output path provided
    if output_path:
        # Check if file exists to append or create new
        try:
            df = pd.read_csv(output_path)
            
            # Check if the file has the new format with 'Baseline Type' column
            if 'Baseline Type' in df.columns:
                # Update or append with Baseline Type
                existing_row = (df['Model'] == model_name) & (df['Baseline Type'] == baseline_type)
                if existing_row.any():
                    # Update existing row with new values
                    for col, val in results.items():
                        df.loc[existing_row, col] = val
                else:
                    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
            else:
                # Handle legacy format without 'Baseline Type' column
                # For backward compatibility, only update/append if this is a Random baseline
                if baseline_type == "Random":
                    if model_name in df['Model'].values:
                        # Update existing row
                        for col, val in results.items():
                            if col != 'Baseline Type' and col in df.columns:
                                df.loc[df['Model'] == model_name, col] = val
                    else:
                        # Create a copy without Baseline Type for legacy format
                        legacy_results = {k: v for k, v in results.items() if k != 'Baseline Type'}
                        df = pd.concat([df, pd.DataFrame([legacy_results])], ignore_index=True)
                else:
                    # For non-Random baselines, create a new file with the new format
                    print(f"Creating new baseline comparison file with updated format at {output_path}")
                    df = pd.DataFrame([results])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame([results])
            
        # Sort by performance (best first)
        if 'Baseline Type' in df.columns:
            df = df.sort_values(['Baseline Type', 'RMSE'])
        else:
            df = df.sort_values('RMSE')
        
        # Save
        df.to_csv(output_path, index=False)
    
    return results


def generate_baseline_comparison_report(csv_path, output_path=None):
    """
    Generate a text report summarizing baseline comparisons.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file with baseline comparison data
    output_path : str, optional
        Path to save the report
    
    Returns
    -------
    str
        Report text
    """
    df = pd.read_csv(csv_path)
    
    # Sort by improvement percentage (best first)
    df = df.sort_values('Improvement (%)', ascending=False)
    
    # Generate report text
    report = ["# Model Performance vs Baselines Report\n"]
    report.append("## Summary\n")
    
    # Count unique models (excluding baseline variations)
    unique_models = len(set([name.split('_')[0] for name in df['Model']]))
    report.append(f"Total unique models analyzed: {unique_models}\n")
    
    # Get best models by baseline type
    baseline_types = df['Baseline Type'].unique()
    
    for baseline_type in baseline_types:
        baseline_df = df[df['Baseline Type'] == baseline_type]
        if not baseline_df.empty:
            best_model = baseline_df.iloc[0]
            report.append(f"Best model vs {baseline_type} baseline: {best_model['Model']} "
                          f"({best_model['Improvement (%)']:.2f}% improvement)\n")
            report.append(f"Average improvement vs {baseline_type} baseline: "
                          f"{baseline_df['Improvement (%)'].mean():.2f}%\n")
    
    # Add baseline type comparison
    report.append("\n## Comparison Across Baseline Types\n")
    
    # Get model algorithms (XGBoost, LightGBM, etc)
    def get_model_algorithm(model_name):
        # Extract base model name without baseline suffix
        base_name = model_name.split('_')[0]
        if base_name.startswith('XGB'):
            return 'XGBoost'
        elif base_name.startswith('LightGBM'):
            return 'LightGBM'
        elif base_name.startswith('CatBoost'):
            return 'CatBoost'
        elif base_name.startswith('ElasticNet'):
            return 'ElasticNet'
        elif base_name.startswith('LR'):
            return 'Linear Regression'
        else:
            return 'Other'
    
    df['Algorithm'] = df['Model'].apply(get_model_algorithm)
    
    # Compare baseline types for each algorithm
    for algorithm, group in df.groupby('Algorithm'):
        report.append(f"### {algorithm}\n")
        
        # Sort by baseline type
        pivot_table = group.pivot_table(
            index=['Model'], 
            columns=['Baseline Type'], 
            values=['RMSE', 'Improvement (%)'],
            aggfunc='first'
        )
        
        # Find best model for each baseline type
        for baseline_type in baseline_types:
            if ('Improvement (%)', baseline_type) in pivot_table.columns:
                # Get the model with the best improvement for this baseline type
                best_idx = pivot_table[('Improvement (%)', baseline_type)].idxmax()
                if pd.notna(best_idx):
                    best_rmse = pivot_table.loc[best_idx, ('RMSE', baseline_type)]
                    best_improve = pivot_table.loc[best_idx, ('Improvement (%)', baseline_type)]
                    
                    report.append(f"Best vs {baseline_type} baseline: {best_idx}\n")
                    report.append(f"RMSE: {best_rmse:.4f}\n")
                    report.append(f"Improvement over {baseline_type.lower()} baseline: {best_improve:.2f}%\n")
        
        report.append(f"Average improvement across baseline types: "
                     f"{group['Improvement (%)'].mean():.2f}%\n\n")
    
    # Add per-baseline type summaries
    for baseline_type in baseline_types:
        baseline_df = df[df['Baseline Type'] == baseline_type]
        if not baseline_df.empty:
            report.append(f"\n## Performance vs {baseline_type} Baseline\n")
            
            for algorithm, alg_group in baseline_df.groupby('Algorithm'):
                best_model = alg_group.iloc[0]
                report.append(f"### {algorithm}\n")
                report.append(f"Best model: {best_model['Model']}\n")
                report.append(f"RMSE: {best_model['RMSE']:.4f}\n")
                report.append(f"Improvement over {baseline_type.lower()} baseline: {best_model['Improvement (%)']:.2f}%\n")
                report.append(f"Average improvement for type: {alg_group['Improvement (%)'].mean():.2f}%\n\n")
    
    # Save report if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
    
    return '\n'.join(report)


def run_baseline_evaluation(all_models, output_path=None, min_val=0, max_val=10, seed=42, 
                         include_mean=True, include_median=True):
    """
    Run baseline evaluation for all models, comparing each against different baselines.
    
    Parameters
    ----------
    all_models : dict
        Dictionary of all trained models with their metadata
    output_path : str, optional
        Path to save baseline comparison results
    min_val : float, optional
        Minimum value for random baseline predictions (default: 0)
    max_val : float, optional
        Maximum value for random baseline predictions (default: 10)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    include_mean : bool, optional
        Whether to include mean baseline comparison (default: True)
    include_median : bool, optional
        Whether to include median baseline comparison (default: True)
        
    Returns
    -------
    dict, pd.DataFrame
        Dictionary of baseline comparison results for each model,
        and a DataFrame with the summary of all comparisons
    """
    if output_path is None:
        # Use default path
        output_path = settings.METRICS_DIR / "baseline_comparison.csv"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store results
    baseline_comparisons = {}
    
    # Process each model
    for model_name, model_data in all_models.items():
        print(f"Running baseline evaluation for {model_name}...")
        
        # Extract actual values (y_test)
        y_test = model_data.get('y_test', None)
        if y_test is None:
            print(f"Warning: No test data found for model {model_name}, skipping.")
            continue
        
        # Extract model predictions
        y_pred = model_data.get('y_pred', None)
        if y_pred is None:
            print(f"Warning: No predictions found for model {model_name}, skipping.")
            continue
        
        # Extract training values if available for mean/median calculation
        y_train = model_data.get('y_train', None)
        if y_train is None and (include_mean or include_median):
            print(f"Warning: No training data found for model {model_name}, will use test data for mean/median baselines.")
            y_train = y_test
        
        # Ensure y_test is numpy array
        if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            y_test_values = y_test.values.flatten()
        else:
            y_test_values = np.array(y_test).flatten()
        
        # Ensure y_pred is numpy array
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_pred_values = y_pred.values.flatten()
        else:
            y_pred_values = np.array(y_pred).flatten()
        
        # Ensure y_train is numpy array (if available)
        if y_train is not None:
            if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
                y_train_values = y_train.values.flatten()
            else:
                y_train_values = np.array(y_train).flatten()
        
        # Generate random baseline predictions
        random_preds = generate_random_baseline(y_test_values, min_val, max_val, seed)
        
        # Calculate random baseline comparison metrics
        random_comparison = calculate_baseline_comparison(
            y_test_values, 
            y_pred_values, 
            random_preds, 
            f"{model_name}_random", 
            output_path,
            baseline_type="Random"
        )
        
        # Store in dictionary
        baseline_comparisons[f"{model_name}_random"] = random_comparison
        
        # Generate mean baseline predictions if requested
        if include_mean and y_train is not None:
            # Generate mean baseline
            mean_value, _ = generate_mean_baseline(y_train_values)
            mean_preds = np.full(len(y_test_values), mean_value)
            
            # Calculate mean baseline comparison metrics
            mean_comparison = calculate_baseline_comparison(
                y_test_values,
                y_pred_values,
                mean_preds,
                f"{model_name}_mean",
                output_path,
                baseline_type="Mean"
            )
            
            # Store in dictionary
            baseline_comparisons[f"{model_name}_mean"] = mean_comparison
        
        # Generate median baseline predictions if requested
        if include_median and y_train is not None:
            # Generate median baseline
            median_value, _ = generate_median_baseline(y_train_values)
            median_preds = np.full(len(y_test_values), median_value)
            
            # Calculate median baseline comparison metrics
            median_comparison = calculate_baseline_comparison(
                y_test_values,
                y_pred_values,
                median_preds,
                f"{model_name}_median",
                output_path,
                baseline_type="Median"
            )
            
            # Store in dictionary
            baseline_comparisons[f"{model_name}_median"] = median_comparison
    
    # Combine all results into a DataFrame
    all_results_df = pd.DataFrame([result for result in baseline_comparisons.values()])
    
    # Sort by improvement percentage (best first)
    if not all_results_df.empty:
        all_results_df = all_results_df.sort_values('Improvement (%)', ascending=False)
    
    # Generate summary report
    report_path = settings.METRICS_DIR / "baseline_comparison_report.md"
    report_text = generate_baseline_comparison_report(output_path, report_path)
    
    print(f"Baseline evaluation complete. Results saved to {output_path}")
    print(f"Summary report saved to {report_path}")
    
    return baseline_comparisons, all_results_df