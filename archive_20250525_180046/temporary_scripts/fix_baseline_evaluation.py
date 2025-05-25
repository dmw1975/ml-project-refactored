#!/usr/bin/env python3
"""
Fix baseline evaluation to include all model types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import settings
from evaluation.baselines import (
    generate_mean_baseline, 
    generate_median_baseline,
    generate_random_baseline,
    generate_sector_mean_baseline
)
from utils.io import load_model
from data import load_features_data, load_scores_data


def evaluate_model_against_baselines(model_data, features_df, scores_df):
    """Evaluate a single model against all baseline methods."""
    if not isinstance(model_data, dict) or 'y_test' not in model_data:
        return None
    
    y_test = model_data['y_test']
    y_pred = model_data['y_pred']
    
    # Get test indices (if available) or use y_test index
    if hasattr(y_test, 'index'):
        test_indices = y_test.index
    else:
        # If no index, we can't do sector baseline
        test_indices = None
    
    # Calculate baselines
    y_train = model_data.get('y_train', scores_df)  # Use all data if y_train not available
    
    mean_baseline = generate_mean_baseline(y_train, y_test)
    median_baseline = generate_median_baseline(y_train, y_test)
    random_baseline = generate_random_baseline(y_test)
    
    # Try sector baseline if we have the necessary data
    sector_baseline = None
    if test_indices is not None and 'X_test' in model_data:
        try:
            X_test = model_data['X_test']
            # Check if we have sector information
            if 'gics_sector' in X_test.columns:
                # Use categorical sector column
                sector_baseline = generate_sector_mean_baseline(
                    model_data.get('X_train', features_df),
                    X_test,
                    y_train,
                    y_test,
                    sector_col='gics_sector'
                )
            else:
                # Look for one-hot encoded sectors
                sector_cols = [col for col in X_test.columns if col.startswith('gics_sector_')]
                if sector_cols:
                    sector_baseline = generate_sector_mean_baseline(
                        model_data.get('X_train', features_df),
                        X_test,
                        y_train,
                        y_test
                    )
        except Exception as e:
            print(f"  Warning: Could not calculate sector baseline: {e}")
    
    # Calculate metrics for each baseline
    results = {
        'model_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'model_mae': mean_absolute_error(y_test, y_pred),
        'model_r2': r2_score(y_test, y_pred),
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
    
    if sector_baseline is not None:
        results.update({
            'sector_baseline_rmse': np.sqrt(mean_squared_error(y_test, sector_baseline)),
            'sector_baseline_mae': mean_absolute_error(y_test, sector_baseline),
            'sector_baseline_r2': r2_score(y_test, sector_baseline),
        })
    
    return results


def evaluate_all_models_against_baselines():
    """Evaluate all models against baselines."""
    print("Evaluating all models against baselines...")
    
    # Load features and scores for baseline calculations
    features_df = load_features_data()
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
                print(f"Evaluating {model_name}...")
                results = evaluate_model_against_baselines(model_data, features_df, scores_df)
                if results:
                    results['model_name'] = model_name
                    results['model_type'] = model_file.replace('_models.pkl', '')
                    all_results.append(results)
    
    # Also check individual model files for categorical models
    for pkl_file in settings.MODEL_DIR.glob('*_categorical_*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict) and 'y_test' in model_data:
                model_name = pkl_file.stem
                print(f"Evaluating {model_name}...")
                results = evaluate_model_against_baselines(model_data, features_df, scores_df)
                if results:
                    results['model_name'] = model_name
                    results['model_type'] = 'categorical'
                    all_results.append(results)
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
    
    # Create DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = settings.METRICS_DIR / 'baseline_comparison_complete.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Baseline comparison saved to {output_path}")
        
        # Print summary
        print("\nBaseline Comparison Summary:")
        print("=" * 60)
        for _, row in results_df.iterrows():
            print(f"\n{row['model_name']}:")
            print(f"  Model RMSE: {row['model_rmse']:.4f}")
            print(f"  Mean Baseline RMSE: {row['mean_baseline_rmse']:.4f}")
            print(f"  Improvement over mean: {(row['mean_baseline_rmse'] - row['model_rmse']) / row['mean_baseline_rmse'] * 100:.1f}%")
    else:
        print("❌ No models found to evaluate")


if __name__ == "__main__":
    evaluate_all_models_against_baselines()