#!/usr/bin/env python3
"""Fix baseline comparison to include ALL models including LightGBM and CatBoost."""

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
    generate_mean_baseline, 
    generate_median_baseline,
    calculate_baseline_comparison
)
from src.utils.io import load_all_models

def regenerate_baseline_comparisons():
    """Regenerate baseline comparison CSV with ALL models."""
    
    print("=" * 80)
    print("REGENERATING BASELINE COMPARISONS WITH ALL MODELS")
    print("=" * 80)
    
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
        if 'LR_' in model_name or model_name.startswith('LR'):
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
        
        # Generate baselines
        # Note: mean_baseline and median_baseline expect training data, 
        # but we'll use y_test to generate appropriate baseline values
        mean_val = float(np.mean(y_test))
        median_val = float(np.median(y_test))
        
        # Create baseline predictions with same shape as y_test
        mean_baseline = np.full_like(y_test, mean_val, dtype=np.float64)
        median_baseline = np.full_like(y_test, median_val, dtype=np.float64)
        random_baseline = generate_random_baseline(y_test, 
                                                  min_val=float(y_test.min()), 
                                                  max_val=float(y_test.max()))
        
        # Compare to each baseline type
        for baseline_type, baseline_pred in [
            ('Mean', mean_baseline),
            ('Median', median_baseline),
            ('Random', random_baseline)
        ]:
            result = calculate_baseline_comparison(
                actual=y_test,
                model_predictions=y_pred,
                baseline_predictions=baseline_pred,
                model_name=model_name,
                baseline_type=baseline_type,
                output_path=None  # Don't save individual results
            )
            
            # Add model name with baseline type suffix for identification
            result['Model'] = f"{model_name}_{baseline_type.lower()}"
            all_comparisons.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Sort by baseline type and RMSE
    df = df.sort_values(['Baseline Type', 'RMSE'])
    
    # Save to CSV
    output_path = settings.METRICS_DIR / "baseline_comparison_fixed.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved fixed baseline comparison to: {output_path}")
    
    # Summary
    print("\nMODELS PROCESSED:")
    print("-" * 40)
    total_models = 0
    for model_type, models in processed_models.items():
        if models:
            print(f"{model_type}: {len(models)} models")
            for m in models:
                print(f"  - {m}")
            total_models += len(models)
    
    print(f"\nTotal models processed: {total_models}")
    print(f"Total comparisons generated: {len(df)}")
    
    # Check for missing model types
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
        elif 'LR_' in model or model.startswith('LR'):
            model_types_in_csv.add('Linear Regression')
    
    expected_types = ['Linear Regression', 'ElasticNet', 'XGBoost', 'LightGBM', 'CatBoost']
    for model_type in expected_types:
        if model_type in model_types_in_csv:
            print(f"✓ {model_type} included")
        else:
            print(f"✗ {model_type} MISSING")
    
    # Replace the original file
    import shutil
    original_path = settings.METRICS_DIR / "baseline_comparison.csv"
    
    # Backup original
    if original_path.exists():
        backup_path = settings.METRICS_DIR / "baseline_comparison_backup.csv"
        shutil.copy(original_path, backup_path)
        print(f"\n✓ Backed up original to: {backup_path}")
    
    # Replace with fixed version
    shutil.copy(output_path, original_path)
    print(f"✓ Replaced baseline_comparison.csv with fixed version")
    
    return df

if __name__ == "__main__":
    regenerate_baseline_comparisons()