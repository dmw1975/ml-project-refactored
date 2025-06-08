#!/usr/bin/env python3
"""Fix baseline comparison to include ALL models using ORIGINAL baseline methodology."""

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

def regenerate_baseline_comparisons_with_original_logic():
    """Regenerate baseline comparison CSV with ALL models using ORIGINAL methodology."""
    
    print("=" * 80)
    print("REGENERATING BASELINE COMPARISONS WITH ORIGINAL METHODOLOGY")
    print("=" * 80)
    print("\nIMPORTANT: Using TRAINING data for mean/median baselines as per original logic")
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
    
    # Track models skipped due to missing y_train
    skipped_models = []
    
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
        
        # CRITICAL: Get y_train for baseline calculations
        y_train = model_data.get('y_train')
        if y_train is None:
            print(f"WARNING: {model_name} missing y_train - will skip mean/median baselines")
            skipped_models.append(model_name)
            # Still process random baseline
        
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
        
        # Generate baselines using ORIGINAL methodology
        
        # 1. RANDOM baseline - uses test data range (original logic)
        random_baseline = generate_random_baseline(y_test, 
                                                  min_val=float(y_test.min()), 
                                                  max_val=float(y_test.max()))
        
        # Compare to random baseline
        result = calculate_baseline_comparison(
            actual=y_test,
            model_predictions=y_pred,
            baseline_predictions=random_baseline,
            model_name=model_name,
            baseline_type='Random',
            output_path=None  # Don't save individual results
        )
        result['Model'] = f"{model_name}_random"
        all_comparisons.append(result)
        
        # 2. MEAN baseline - uses TRAINING data (original logic)
        if y_train is not None:
            # Use the original function that expects training data
            mean_val, _ = generate_mean_baseline(y_train)
            # Create predictions of the right length for test set
            mean_baseline = np.full(len(y_test), mean_val)
            
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
        
        # 3. MEDIAN baseline - uses TRAINING data (original logic)
        if y_train is not None:
            # Use the original function that expects training data
            median_val, _ = generate_median_baseline(y_train)
            # Create predictions of the right length for test set
            median_baseline = np.full(len(y_test), median_val)
            
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
    
    # Create DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Sort by baseline type and RMSE
    df = df.sort_values(['Baseline Type', 'RMSE'])
    
    # Save to CSV
    output_path = settings.METRICS_DIR / "baseline_comparison_corrected.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved corrected baseline comparison to: {output_path}")
    
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
    
    if skipped_models:
        print(f"\nWARNING: {len(skipped_models)} models skipped for mean/median baselines due to missing y_train:")
        for model in skipped_models[:5]:  # Show first 5
            print(f"  - {model}")
        if len(skipped_models) > 5:
            print(f"  ... and {len(skipped_models) - 5} more")
    
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
    
    # Compare baseline values with backup (if exists)
    backup_path = settings.METRICS_DIR / "baseline_comparison_backup.csv"
    if backup_path.exists():
        print("\nCOMPARING BASELINE VALUES WITH ORIGINAL:")
        print("-" * 40)
        
        backup_df = pd.read_csv(backup_path)
        
        # Get a sample model that exists in both
        common_models = set(df['Model'].str.replace('_(mean|median|random)$', '', regex=True)) & \
                       set(backup_df['Model'].str.replace('_(mean|median|random)$', '', regex=True))
        
        if common_models:
            sample_model = list(common_models)[0]
            print(f"\nSample comparison for model: {sample_model}")
            
            for baseline_type in ['Mean', 'Median', 'Random']:
                # Get baseline RMSE from backup
                backup_row = backup_df[
                    (backup_df['Model'].str.contains(sample_model)) & 
                    (backup_df['Baseline Type'] == baseline_type)
                ]
                
                # Get baseline RMSE from new
                new_row = df[
                    (df['Model'].str.contains(sample_model)) & 
                    (df['Baseline Type'] == baseline_type)
                ]
                
                if not backup_row.empty and not new_row.empty:
                    old_baseline = backup_row['Baseline RMSE'].iloc[0]
                    new_baseline = new_row['Baseline RMSE'].iloc[0]
                    
                    if abs(old_baseline - new_baseline) < 0.0001:
                        print(f"  {baseline_type} baseline: {old_baseline:.6f} ✓ (matches original)")
                    else:
                        print(f"  {baseline_type} baseline: {old_baseline:.6f} → {new_baseline:.6f} ✗ (changed)")
    
    # Replace the original file
    import shutil
    original_path = settings.METRICS_DIR / "baseline_comparison.csv"
    
    # Replace with corrected version
    shutil.copy(output_path, original_path)
    print(f"\n✓ Replaced baseline_comparison.csv with corrected version using original methodology")
    
    return df

if __name__ == "__main__":
    regenerate_baseline_comparisons_with_original_logic()