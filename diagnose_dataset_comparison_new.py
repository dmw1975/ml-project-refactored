#!/usr/bin/env python3
"""Diagnose why LightGBM and CatBoost are missing from dataset comparison plots."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_all_models
from src.visualization.core.registry import get_adapter

def check_model_metrics():
    """Check which models have metrics and adapters."""
    
    print("=== DIAGNOSTIC: Dataset Comparison Model Analysis ===\n")
    
    # Load all models
    all_models = load_all_models()
    print(f"Total models loaded: {len(all_models)}\n")
    
    # Check each model
    models_with_metrics = []
    models_without_metrics = []
    models_with_adapter_issues = []
    
    for model_name, model_data in all_models.items():
        print(f"Checking {model_name}:")
        
        # Check if model has adapter
        try:
            adapter = get_adapter(model_name)
            print(f"  ✓ Adapter found: {adapter.__class__.__name__}")
            
            # Check if adapter has data
            if hasattr(adapter, 'data'):
                print(f"  ✓ Adapter has data attribute")
            else:
                print(f"  ✗ Adapter missing data attribute")
                models_with_adapter_issues.append(model_name)
                continue
                
            # Check if model has metrics
            try:
                metrics = adapter.get_metrics()
                if metrics and all(key in metrics for key in ['RMSE', 'R2', 'MAE']):
                    print(f"  ✓ Has required metrics: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
                    models_with_metrics.append(model_name)
                else:
                    print(f"  ✗ Missing required metrics. Has: {list(metrics.keys()) if metrics else 'None'}")
                    models_without_metrics.append(model_name)
            except Exception as e:
                print(f"  ✗ Error getting metrics: {e}")
                models_without_metrics.append(model_name)
                
        except Exception as e:
            print(f"  ✗ No adapter found or error: {e}")
            models_with_adapter_issues.append(model_name)
        
        print()
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Models with metrics: {len(models_with_metrics)}")
    print(f"Models without metrics: {len(models_without_metrics)}")
    print(f"Models with adapter issues: {len(models_with_adapter_issues)}")
    
    if models_without_metrics:
        print(f"\nModels WITHOUT metrics:")
        for m in models_without_metrics:
            print(f"  - {m}")
    
    if models_with_adapter_issues:
        print(f"\nModels with adapter issues:")
        for m in models_with_adapter_issues:
            print(f"  - {m}")

if __name__ == "__main__":
    check_model_metrics()