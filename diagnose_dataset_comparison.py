#!/usr/bin/env python3
"""Diagnose why LightGBM and CatBoost are missing from dataset comparisons."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.visualization.plots.dataset_comparison import DatasetModelComparisonPlot
from src.utils.io import load_all_models
from src.visualization.core.registry import get_adapter_for_model

def diagnose_dataset_comparison():
    """Diagnose the dataset comparison issue."""
    
    print("=" * 80)
    print("DATASET COMPARISON DIAGNOSTIC")
    print("=" * 80)
    
    # Load all models
    all_models = load_all_models()
    
    # Add model names to the model data if missing
    model_list = []
    for model_name, model_data in all_models.items():
        model_data_copy = model_data.copy()
        if 'model_name' not in model_data_copy:
            model_data_copy['model_name'] = model_name
        model_list.append(model_data_copy)
    
    # Create comparison object
    comparison = DatasetModelComparisonPlot(model_list)
    
    # Extract metrics
    print("\nExtracting metrics...")
    metrics_df = comparison.extract_model_metrics()
    
    print(f"\nTotal models with metrics: {len(metrics_df)}")
    
    # Show model families
    print("\nModel families found:")
    family_counts = metrics_df['model_family'].value_counts()
    for family, count in family_counts.items():
        print(f"  {family}: {count} models")
    
    # Check for missing model types
    print("\nChecking for missing models...")
    
    # Check which models were skipped
    all_model_names = set(all_models.keys())
    extracted_model_names = set(metrics_df['model_name'].unique())
    skipped_models = all_model_names - extracted_model_names
    
    if skipped_models:
        print(f"\nSkipped {len(skipped_models)} models:")
        
        # Categorize skipped models
        skipped_by_type = {'LightGBM': [], 'CatBoost': [], 'XGBoost': [], 'Other': []}
        
        for model_name in sorted(skipped_models):
            if 'LightGBM' in model_name:
                skipped_by_type['LightGBM'].append(model_name)
            elif 'CatBoost' in model_name:
                skipped_by_type['CatBoost'].append(model_name)
            elif 'XGBoost' in model_name:
                skipped_by_type['XGBoost'].append(model_name)
            else:
                skipped_by_type['Other'].append(model_name)
        
        for model_type, models in skipped_by_type.items():
            if models:
                print(f"\n  {model_type} ({len(models)} models):")
                for model in models:
                    # Check why it was skipped
                    model_data = all_models[model]
                    adapter = get_adapter_for_model(model_data)
                    metrics = adapter.get_metrics()
                    
                    missing_metrics = []
                    for metric in ['RMSE', 'R2', 'MAE']:
                        if metric not in metrics:
                            missing_metrics.append(metric)
                    
                    print(f"    - {model}: Missing {missing_metrics}")
                    print(f"      Available metrics: {list(metrics.keys())}")
    else:
        print("\nNo models were skipped!")
    
    # Show sample of what's included
    print("\nSample of included models:")
    for idx, row in metrics_df.head(10).iterrows():
        print(f"  {row['model_name']} - Family: {row['model_family']}, Dataset: {row['dataset']}")

if __name__ == "__main__":
    diagnose_dataset_comparison()