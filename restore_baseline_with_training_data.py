#!/usr/bin/env python3
"""Restore baseline comparisons using proper training data methodology."""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.evaluation.baselines import (
    generate_random_baseline, 
    calculate_baseline_comparison
)
from src.utils.io import load_all_models

def get_training_statistics():
    """
    Get the training set statistics for baseline calculations.
    
    Returns:
        dict: Dictionary with 'mean' and 'median' of training scores
    """
    # Load the original score data
    score_df = pd.read_csv('data/raw/score.csv')
    
    # Load the train/test split indices
    split_path = Path('data/processed/unified/train_test_split.pkl')
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    # Get training indices
    train_indices = split_data['train_indices']
    
    # Get training scores
    train_scores = score_df.iloc[train_indices]['esg_score']
    
    return {
        'mean': train_scores.mean(),
        'median': train_scores.median(),
        'min': train_scores.min(),
        'max': train_scores.max()
    }

def restore_baseline_comparisons_properly():
    """Restore baseline comparisons using proper training data for baselines."""
    
    print("=" * 80)
    print("RESTORING BASELINE COMPARISONS WITH PROPER METHODOLOGY")
    print("=" * 80)
    
    # Get training statistics
    train_stats = get_training_statistics()
    
    print("\nTRAINING DATA STATISTICS:")
    print("-" * 40)
    print(f"Mean: {train_stats['mean']:.6f}")
    print(f"Median: {train_stats['median']:.6f}")
    print(f"Min: {train_stats['min']:.6f}")
    print(f"Max: {train_stats['max']:.6f}")
    
    # Load ALL models
    all_models = load_all_models()
    
    # Track which models we process
    processed_models = {
        'Linear Regression': [],
        'ElasticNet': [],
        'XGBoost': [],
        'LightGBM': [],
        'CatBoost': []
    }
    
    # Collect all baseline comparisons
    all_comparisons = []
    
    # Process each model
    for model_name, model_data in all_models.items():
        if not isinstance(model_data, dict):
            print(f"Skipping {model_name}: not a dictionary")
            continue
            
        # Get y_test and y_pred
        y_test = model_data.get('y_test')
        y_pred = model_data.get('y_pred')
        if y_pred is None:
            y_pred = model_data.get('y_test_pred')
        
        if y_test is None or y_pred is None:
            print(f"Skipping {model_name}: missing y_test or y_pred")
            continue
        
        # Determine model type
        if 'LR_' in model_name or model_name.startswith('LR') or model_name.startswith('lr_'):
            model_type = 'Linear Regression'
        elif 'ElasticNet' in model_name:
            model_type = 'ElasticNet'
        elif 'XGBoost' in model_name:
            model_type = 'XGBoost'
        elif 'LightGBM' in model_name:
            model_type = 'LightGBM'
        elif 'CatBoost' in model_name:
            model_type = 'CatBoost'
        else:
            model_type = 'Unknown'
            
        processed_models.setdefault(model_type, []).append(model_name)
        
        # Generate baselines using PROPER methodology
        
        # 1. MEAN baseline - use training mean
        mean_baseline = np.full(len(y_test), train_stats['mean'])
        
        result = calculate_baseline_comparison(
            actual=y_test,
            model_predictions=y_pred,
            baseline_predictions=mean_baseline,
            model_name=model_name,
            baseline_type='Mean',
            output_path=None
        )
        result['Model'] = f"{model_name}_mean"
        all_comparisons.append(result)
        
        # 2. MEDIAN baseline - use training median  
        median_baseline = np.full(len(y_test), train_stats['median'])
        
        result = calculate_baseline_comparison(
            actual=y_test,
            model_predictions=y_pred,
            baseline_predictions=median_baseline,
            model_name=model_name,
            baseline_type='Median',
            output_path=None
        )
        result['Model'] = f"{model_name}_median"
        all_comparisons.append(result)
        
        # 3. RANDOM baseline - use test data range (original logic)
        random_baseline = generate_random_baseline(
            y_test, 
            min_val=float(y_test.min()), 
            max_val=float(y_test.max()),
            seed=42
        )
        
        result = calculate_baseline_comparison(
            actual=y_test,
            model_predictions=y_pred,
            baseline_predictions=random_baseline,
            model_name=model_name,
            baseline_type='Random',
            output_path=None
        )
        result['Model'] = f"{model_name}_random"
        all_comparisons.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Sort by baseline type and RMSE
    df = df.sort_values(['Baseline Type', 'RMSE'])
    
    # Save to CSV
    output_path = settings.METRICS_DIR / "baseline_comparison.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved baseline comparison to: {output_path}")
    
    # Summary
    print("\nMODELS PROCESSED:")
    print("-" * 40)
    total_models = 0
    for model_type, models in processed_models.items():
        if models:
            print(f"{model_type}: {len(models)} models")
            total_models += len(models)
    
    print(f"\nTotal models processed: {total_models}")
    print(f"Total comparisons generated: {len(df)}")
    
    # Verify all model types included
    print("\nVERIFICATION:")
    print("-" * 40)
    
    # Check which model types are in the CSV
    model_types_in_csv = set()
    for model in df['Model'].unique():
        if 'LightGBM' in model:
            model_types_in_csv.add('LightGBM')
        elif 'CatBoost' in model:
            model_types_in_csv.add('CatBoost')
        elif 'XGBoost' in model:
            model_types_in_csv.add('XGBoost')
        elif 'ElasticNet' in model:
            model_types_in_csv.add('ElasticNet')
        elif 'LR_' in model or model.startswith('LR') or 'lr_' in model:
            model_types_in_csv.add('Linear Regression')
    
    expected_types = ['Linear Regression', 'ElasticNet', 'XGBoost', 'LightGBM', 'CatBoost']
    for model_type in expected_types:
        if model_type in model_types_in_csv:
            print(f"✓ {model_type} included")
        else:
            print(f"✗ {model_type} MISSING")
    
    # Check baseline values
    print("\nBASELINE VALUES:")
    print("-" * 40)
    
    for baseline_type in ['Mean', 'Median', 'Random']:
        baseline_df = df[df['Baseline Type'] == baseline_type]
        if not baseline_df.empty:
            # Get unique baseline RMSE values (should be consistent except Random)
            unique_baseline_rmse = baseline_df['Baseline RMSE'].unique()
            if len(unique_baseline_rmse) == 1 or baseline_type == 'Random':
                if baseline_type != 'Random':
                    print(f"{baseline_type}: {unique_baseline_rmse[0]:.6f} (consistent across all models)")
                else:
                    print(f"{baseline_type}: varies by model (as expected)")
            else:
                print(f"{baseline_type}: WARNING - inconsistent values found")
    
    # Compare with backup if exists
    backup_path = settings.METRICS_DIR / "baseline_comparison_backup.csv"
    if backup_path.exists():
        print("\nCOMPARING WITH ORIGINAL VALUES:")
        print("-" * 40)
        
        backup_df = pd.read_csv(backup_path)
        
        # Compare baseline RMSE values for a sample model
        sample_models = ['XGBoost_Base_Random_categorical_optuna']
        
        for model_base in sample_models:
            for baseline_type in ['Mean', 'Median']:
                # Original
                orig_row = backup_df[
                    (backup_df['Model'] == f"{model_base}_{baseline_type.lower()}") & 
                    (backup_df['Baseline Type'] == baseline_type)
                ]
                
                # New
                new_row = df[
                    (df['Model'] == f"{model_base}_{baseline_type.lower()}") & 
                    (df['Baseline Type'] == baseline_type)
                ]
                
                if not orig_row.empty and not new_row.empty:
                    orig_baseline = orig_row['Baseline RMSE'].iloc[0]
                    new_baseline = new_row['Baseline RMSE'].iloc[0]
                    
                    print(f"{baseline_type} baseline RMSE: {new_baseline:.6f} (was {orig_baseline:.6f})")
    
    return df

if __name__ == "__main__":
    restore_baseline_comparisons_properly()