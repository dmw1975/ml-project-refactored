#!/usr/bin/env python3
"""Diagnose why LightGBM and CatBoost are missing from visualizations."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_all_models, load_model
from src.config import settings
from src.evaluation.baseline_significance import run_baseline_significance_analysis
import pickle

def diagnose_models():
    """Diagnose model loading and visualization issues."""
    
    print("=== DIAGNOSTIC: Model Loading Analysis ===\n")
    
    # 1. Check raw model files
    print("1. Checking raw model files in outputs/models/:")
    model_files = [
        "linear_regression_models.pkl",
        "elasticnet_models.pkl", 
        "xgboost_models.pkl",
        "lightgbm_models.pkl",
        "catboost_models.pkl"
    ]
    
    model_counts = {}
    all_model_names = []
    
    for model_file in model_files:
        filepath = settings.MODEL_DIR / model_file
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    models = pickle.load(f)
                if isinstance(models, dict):
                    count = len(models)
                    model_counts[model_file] = count
                    print(f"  ✓ {model_file}: {count} models")
                    for name in models.keys():
                        print(f"    - {name}")
                        all_model_names.append(name)
                else:
                    print(f"  ✗ {model_file}: Not a dict")
            except Exception as e:
                print(f"  ✗ {model_file}: Error loading - {e}")
        else:
            print(f"  ✗ {model_file}: File not found")
    
    print(f"\nTotal models found in files: {sum(model_counts.values())}")
    
    # 2. Check load_all_models()
    print("\n2. Testing load_all_models() function:")
    all_models = load_all_models()
    print(f"  - Models loaded: {len(all_models)}")
    
    # Check which models are missing
    missing_models = set(all_model_names) - set(all_models.keys())
    if missing_models:
        print(f"  - MISSING MODELS: {missing_models}")
    
    # 3. Count models by type
    print("\n3. Model type analysis:")
    
    # Count models by type
    model_types = {}
    for model_name in all_models.keys():
        if 'LightGBM' in model_name:
            model_types['LightGBM'] = model_types.get('LightGBM', 0) + 1
        elif 'CatBoost' in model_name:
            model_types['CatBoost'] = model_types.get('CatBoost', 0) + 1
        elif 'XGBoost' in model_name or 'XGB' in model_name:
            model_types['XGBoost'] = model_types.get('XGBoost', 0) + 1
        elif 'ElasticNet' in model_name:
            model_types['ElasticNet'] = model_types.get('ElasticNet', 0) + 1
        elif 'LR_' in model_name:
            model_types['LinearRegression'] = model_types.get('LinearRegression', 0) + 1
    
    print("  Model counts by type:")
    for mtype, count in model_types.items():
        print(f"    - {mtype}: {count}")
    
    # 4. Check baseline_comparison.csv
    print("\n4. Checking baseline_comparison.csv:")
    baseline_path = settings.OUTPUT_DIR / "baseline_comparison.csv"
    if baseline_path.exists():
        import pandas as pd
        df = pd.read_csv(baseline_path)
        print(f"  - Total rows: {len(df)}")
        print(f"  - Unique models: {df['Model'].nunique()}")
        
        # Check for each model type
        for mtype in ['LightGBM', 'CatBoost', 'XGBoost', 'ElasticNet', 'LR_']:
            count = df['Model'].str.contains(mtype).sum()
            print(f"    - {mtype}: {count} entries")
    else:
        print("  - File not found!")
    
    # 5. Test creating visualizations
    print("\n5. Testing visualization creation:")
    from src.visualization.plots.dataset_comparison import DatasetComparisonVisualizer
    
    viz = DatasetComparisonVisualizer(all_models)
    metrics_df = viz.extract_model_metrics()
    
    print(f"  - Extracted metrics for {len(metrics_df)} models")
    print(f"  - Model families found: {metrics_df['model_family'].unique()}")
    
    # Check which models have missing metrics
    models_with_metrics = set(metrics_df['model_name'].unique())
    models_without_metrics = set(all_models.keys()) - models_with_metrics
    
    if models_without_metrics:
        print(f"\n  CRITICAL: Models without metrics:")
        for model_name in sorted(models_without_metrics):
            model = all_models[model_name]
            # Check what's in the model
            if hasattr(model, 'get_metrics'):
                try:
                    metrics = model.get_metrics()
                    print(f"    - {model_name}: metrics = {list(metrics.keys()) if metrics else 'None'}")
                except Exception as e:
                    print(f"    - {model_name}: Error getting metrics - {e}")
            else:
                print(f"    - {model_name}: No get_metrics method!")

if __name__ == "__main__":
    diagnose_models()