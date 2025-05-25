#!/usr/bin/env python3
"""
Simple baseline evaluation for all models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import settings
from utils.io import load_model
from data import load_features_data, load_scores_data


def evaluate_model_against_baselines(model_data, scores_df):
    """Evaluate a single model against simple baseline methods."""
    if not isinstance(model_data, dict) or 'y_test' not in model_data:
        return None
    
    y_test = model_data['y_test']
    y_pred = model_data.get('y_pred', model_data.get('y_test_pred'))
    
    if y_pred is None:
        return None
    
    # Get training data for baseline calculations
    y_train = model_data.get('y_train', scores_df)
    
    # Calculate simple baselines
    mean_baseline = np.full_like(y_test, np.mean(y_train))
    median_baseline = np.full_like(y_test, np.median(y_train))
    
    # Random baseline
    np.random.seed(42)
    random_baseline = np.random.uniform(0, 10, size=len(y_test))
    
    # Calculate metrics for model
    model_metrics = {
        'model_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'model_mae': mean_absolute_error(y_test, y_pred),
        'model_r2': r2_score(y_test, y_pred),
    }
    
    # Calculate metrics for baselines
    baseline_metrics = {
        'mean_baseline_rmse': np.sqrt(mean_squared_error(y_test, mean_baseline)),
        'mean_baseline_mae': mean_absolute_error(y_test, mean_baseline),
        'mean_baseline_r2': r2_score(y_test, mean_baseline),
        'median_baseline_rmse': np.sqrt(mean_squared_error(y_test, median_baseline)),
        'median_baseline_mae': mean_absolute_error(y_test, median_baseline),
        'median_baseline_r2': r2_score(y_test, median_baseline),
        'random_baseline_rmse': np.sqrt(mean_squared_error(y_test, random_baseline)),
        'random_baseline_mae': mean_absolute_error(y_test, random_baseline),
        'random_baseline_r2': r2_score(y_test, random_baseline),
    }
    
    # Calculate improvements
    improvements = {
        'improvement_over_mean': ((baseline_metrics['mean_baseline_rmse'] - model_metrics['model_rmse']) / 
                                  baseline_metrics['mean_baseline_rmse'] * 100),
        'improvement_over_median': ((baseline_metrics['median_baseline_rmse'] - model_metrics['model_rmse']) / 
                                    baseline_metrics['median_baseline_rmse'] * 100),
        'improvement_over_random': ((baseline_metrics['random_baseline_rmse'] - model_metrics['model_rmse']) / 
                                    baseline_metrics['random_baseline_rmse'] * 100),
    }
    
    return {**model_metrics, **baseline_metrics, **improvements}


def evaluate_all_models_against_baselines():
    """Evaluate all models against baselines."""
    print("Evaluating all models against baselines...")
    
    # Load scores for baseline calculations
    scores_df = load_scores_data()
    
    # Model files to check
    model_files = [
        'linear_regression_models.pkl',
        'elasticnet_models.pkl',
        'xgboost_models.pkl',
        'lightgbm_models.pkl',
        'catboost_models.pkl',
    ]
    
    all_results = []
    
    for model_file in model_files:
        models = load_model(model_file, settings.MODEL_DIR)
        if not models:
            continue
            
        if isinstance(models, dict):
            for model_name, model_data in models.items():
                print(f"  Evaluating {model_name}...")
                results = evaluate_model_against_baselines(model_data, scores_df)
                if results:
                    results['model_name'] = model_name
                    results['model_type'] = model_file.replace('_models.pkl', '')
                    all_results.append(results)
    
    # Also check individual categorical model files
    categorical_files = [
        'lightgbm_base_categorical_basic.pkl',
        'lightgbm_base_categorical_optuna.pkl',
        'lightgbm_yeo_categorical_basic.pkl',
        'lightgbm_yeo_categorical_optuna.pkl',
        'lightgbm_base_random_categorical_basic.pkl',
        'lightgbm_base_random_categorical_optuna.pkl',
        'lightgbm_yeo_random_categorical_basic.pkl',
        'lightgbm_yeo_random_categorical_optuna.pkl',
        'catboost_categorical_models.pkl',
        'xgboost_base_categorical_basic.pkl',
        'xgboost_base_categorical_optuna.pkl',
        'xgboost_yeo_categorical_basic.pkl',
        'xgboost_yeo_categorical_optuna.pkl',
        'xgboost_base_random_categorical_basic.pkl',
        'xgboost_base_random_categorical_optuna.pkl',
        'xgboost_yeo_random_categorical_basic.pkl',
        'xgboost_yeo_random_categorical_optuna.pkl',
    ]
    
    for cat_file in categorical_files:
        pkl_file = settings.MODEL_DIR / cat_file
        if pkl_file.exists():
            try:
                with open(pkl_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    # Handle nested model dictionaries
                    if 'y_test' in model_data:
                        model_name = pkl_file.stem
                        print(f"  Evaluating {model_name}...")
                        results = evaluate_model_against_baselines(model_data, scores_df)
                        if results:
                            results['model_name'] = model_name
                            results['model_type'] = 'categorical'
                            all_results.append(results)
                    else:
                        # Check if it's a dict of models
                        for sub_name, sub_data in model_data.items():
                            if isinstance(sub_data, dict) and 'y_test' in sub_data:
                                print(f"  Evaluating {sub_name}...")
                                results = evaluate_model_against_baselines(sub_data, scores_df)
                                if results:
                                    results['model_name'] = sub_name
                                    results['model_type'] = 'categorical'
                                    all_results.append(results)
            except Exception as e:
                print(f"  Error loading {cat_file}: {e}")
    
    # Create DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort by model RMSE
        results_df = results_df.sort_values('model_rmse')
        
        # Save full results
        output_path = settings.METRICS_DIR / 'baseline_comparison_complete.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Baseline comparison saved to {output_path}")
        
        # Create summary table
        summary_columns = [
            'model_name', 'model_rmse', 'mean_baseline_rmse', 
            'improvement_over_mean', 'improvement_over_random'
        ]
        summary_df = results_df[summary_columns].copy()
        summary_df.columns = ['Model', 'RMSE', 'Mean Baseline RMSE', 
                              'Improvement vs Mean (%)', 'Improvement vs Random (%)']
        
        # Save summary
        summary_path = settings.METRICS_DIR / 'baseline_comparison_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        print("\nBaseline Comparison Summary (Top 10):")
        print("=" * 80)
        print(summary_df.head(10).to_string(index=False))
        
        print(f"\n✅ Found and evaluated {len(all_results)} models")
    else:
        print("❌ No models found to evaluate")


if __name__ == "__main__":
    evaluate_all_models_against_baselines()