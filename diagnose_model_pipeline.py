#!/usr/bin/env python
"""Diagnostic script to trace model loading and visualization pipeline."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.absolute()))

import pickle
from src.utils.io import load_all_models
from src.config import settings
from src.visualization.utils.io import load_all_models as viz_load_all_models
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def diagnose_model_loading():
    """Diagnose issues with model loading pipeline."""
    
    print("\n" + "="*80)
    print("MODEL LOADING DIAGNOSTIC")
    print("="*80)
    
    # Step 1: Check what model files exist
    print("\n1. CHECKING MODEL FILES:")
    print("-" * 40)
    model_files = [
        "linear_regression_models.pkl",
        "elasticnet_models.pkl", 
        "xgboost_models.pkl",
        "lightgbm_models.pkl",
        "catboost_models.pkl"
    ]
    
    existing_files = {}
    for model_file in model_files:
        path = settings.MODEL_DIR / model_file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {model_file}: {size_mb:.2f} MB")
            existing_files[model_file] = path
        else:
            print(f"✗ {model_file}: NOT FOUND")
    
    # Step 2: Load models directly from files
    print("\n2. LOADING MODELS DIRECTLY:")
    print("-" * 40)
    direct_models = {}
    model_counts = {}
    
    for model_file, path in existing_files.items():
        try:
            with open(path, 'rb') as f:
                models = pickle.load(f)
            if isinstance(models, dict):
                count = len(models)
                model_counts[model_file] = count
                direct_models.update(models)
                print(f"✓ {model_file}: {count} models loaded")
                # List model names
                for name in sorted(models.keys()):
                    print(f"   - {name}")
            else:
                print(f"⚠️  {model_file}: Invalid format (not a dict)")
        except Exception as e:
            print(f"✗ {model_file}: Error loading - {e}")
    
    print(f"\nTotal models loaded directly: {len(direct_models)}")
    
    # Step 3: Load models using io.load_all_models()
    print("\n3. LOADING MODELS USING io.load_all_models():")
    print("-" * 40)
    io_models = load_all_models()
    print(f"Models loaded by io.load_all_models(): {len(io_models)}")
    
    # Compare
    missing_in_io = set(direct_models.keys()) - set(io_models.keys())
    if missing_in_io:
        print(f"\n⚠️  Models missing in io.load_all_models(): {len(missing_in_io)}")
        for name in sorted(missing_in_io):
            print(f"   - {name}")
    
    # Step 4: Check visualization load function
    print("\n4. LOADING MODELS USING visualization.utils.io:")
    print("-" * 40)
    try:
        viz_models = viz_load_all_models()
        print(f"Models loaded by viz load_all_models: {len(viz_models)}")
        
        missing_in_viz = set(direct_models.keys()) - set(viz_models.keys())
        if missing_in_viz:
            print(f"\n⚠️  Models missing in viz load: {len(missing_in_viz)}")
            for name in sorted(missing_in_viz):
                print(f"   - {name}")
    except Exception as e:
        print(f"✗ Error loading with viz utils: {e}")
        viz_models = {}
    
    # Step 5: Check model types distribution
    print("\n5. MODEL TYPE DISTRIBUTION:")
    print("-" * 40)
    model_types = {}
    for name in direct_models.keys():
        model_type = name.split('_')[0]
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    for model_type, count in sorted(model_types.items()):
        print(f"{model_type}: {count} models")
    
    # Step 6: Check CV scores availability
    print("\n6. CV SCORES AVAILABILITY:")
    print("-" * 40)
    models_with_cv = 0
    models_without_cv = []
    
    for name, model_data in direct_models.items():
        if isinstance(model_data, dict):
            has_cv = any(key in model_data for key in ['cv_scores', 'cv_mean', 'cv_mse'])
            if has_cv:
                models_with_cv += 1
            else:
                models_without_cv.append(name)
    
    print(f"Models with CV scores: {models_with_cv}/{len(direct_models)}")
    if models_without_cv:
        print("\nModels WITHOUT CV scores:")
        for name in sorted(models_without_cv):
            print(f"   - {name}")
    
    # Step 7: Check model structure for each type
    print("\n7. MODEL STRUCTURE ANALYSIS:")
    print("-" * 40)
    
    # Check one model from each type
    sample_models = {}
    for model_type in ['LR', 'ElasticNet', 'XGBoost', 'LightGBM', 'CatBoost']:
        for name, data in direct_models.items():
            if name.startswith(model_type):
                sample_models[model_type] = (name, data)
                break
    
    for model_type, (name, data) in sample_models.items():
        print(f"\n{model_type} sample ({name}):")
        if isinstance(data, dict):
            print(f"   Keys: {sorted(data.keys())}")
            # Check for critical fields
            has_model = 'model' in data
            has_metrics = all(m in data for m in ['rmse', 'mse', 'mae', 'r2'])
            has_test_data = 'X_test' in data and 'y_test' in data
            has_predictions = 'y_pred' in data or 'predictions' in data
            
            print(f"   ✓ Has model object: {has_model}")
            print(f"   ✓ Has metrics: {has_metrics}")
            print(f"   ✓ Has test data: {has_test_data}")
            print(f"   ✓ Has predictions: {has_predictions}")
    
    return direct_models, io_models, viz_models

def check_visualization_adapters():
    """Check if visualization adapters exist for all model types."""
    print("\n8. VISUALIZATION ADAPTER CHECK:")
    print("-" * 40)
    
    try:
        from src.visualization.core.registry import _adapter_registry
        print("Registered adapters:")
        for key in sorted(_adapter_registry.keys()):
            print(f"   - {key}")
    except Exception as e:
        print(f"Error checking adapters: {e}")
    
    # Check if adapters exist for our model types
    expected_adapters = ['linear_regression', 'elasticnet', 'xgboost', 'lightgbm', 'catboost']
    adapter_files = {
        'linear_regression': 'linear_adapter.py',
        'elasticnet': 'elasticnet_adapter.py',
        'xgboost': 'xgboost_adapter.py',
        'lightgbm': 'lightgbm_adapter.py',
        'catboost': 'catboost_adapter.py'
    }
    
    print("\nAdapter files:")
    adapters_dir = Path("src/visualization/adapters")
    for adapter_type, filename in adapter_files.items():
        path = adapters_dir / filename
        exists = path.exists()
        print(f"   {adapter_type}: {filename} - {'✓ EXISTS' if exists else '✗ MISSING'}")

def trace_cv_distribution_pipeline():
    """Trace why CV distributions are missing for some models."""
    print("\n9. CV DISTRIBUTION PIPELINE TRACE:")
    print("-" * 40)
    
    from src.visualization.comprehensive import create_comprehensive_visualizations
    from src.visualization.plots.cv_distributions import plot_cv_distributions
    
    # Load models
    models = load_all_models()
    
    # Filter models with CV data (mimicking comprehensive.py logic)
    cv_models = []
    for model_data in models.values():
        if isinstance(model_data, dict) and any(key in model_data for key in 
                                               ['cv_scores', 'cv_fold_scores', 'cv_mean']):
            cv_models.append(model_data)
    
    print(f"Total models: {len(models)}")
    print(f"Models with CV data: {len(cv_models)}")
    
    # Check which model types have CV data
    cv_model_types = {}
    for model_data in cv_models:
        if 'model_name' in model_data:
            model_type = model_data['model_name'].split('_')[0]
            cv_model_types[model_type] = cv_model_types.get(model_type, 0) + 1
    
    print("\nCV models by type:")
    for model_type, count in sorted(cv_model_types.items()):
        print(f"   {model_type}: {count}")
    
    # Check what's missing
    all_model_types = {}
    for name in models.keys():
        model_type = name.split('_')[0]
        all_model_types[model_type] = all_model_types.get(model_type, 0) + 1
    
    print("\nModels without CV data by type:")
    for model_type, total_count in all_model_types.items():
        cv_count = cv_model_types.get(model_type, 0)
        if cv_count < total_count:
            print(f"   {model_type}: {total_count - cv_count} models missing CV data")

if __name__ == "__main__":
    direct_models, io_models, viz_models = diagnose_model_loading()
    check_visualization_adapters()
    trace_cv_distribution_pipeline()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)