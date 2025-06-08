#!/usr/bin/env python3
"""Diagnose why CatBoost and LightGBM are excluded from CV distribution plots."""

import pickle
import os
from pathlib import Path
import numpy as np
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualization.core.registry import get_adapter_for_model
from src.visualization.plots.cv_distributions import CVDistributionPlot

def diagnose_model_cv_data():
    """Diagnose CV data availability for all models."""
    
    model_dir = Path("/mnt/d/ml_project_refactored/outputs/models")
    
    # Load all model files
    model_files = {
        'CatBoost': 'catboost_models.pkl',
        'LightGBM': 'lightgbm_models.pkl',
        'XGBoost': 'xgboost_models.pkl',
        'ElasticNet': 'elasticnet_models.pkl',
        'Linear Regression': 'linear_regression_models.pkl'
    }
    
    all_models = []
    model_data_summary = {}
    
    for model_type, filename in model_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            print(f"\n{'='*60}")
            print(f"Loading {model_type} models from {filename}")
            print('='*60)
            
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
            
            model_data_summary[model_type] = []
            
            for model_name, model_data in models.items():
                print(f"\n--- {model_name} ---")
                
                # Get adapter
                try:
                    adapter = get_adapter_for_model(model_data)
                    
                    # Try to extract CV metrics using the same logic as CVDistributionPlot
                    cv_plot = CVDistributionPlot([], {})
                    cv_metrics = cv_plot._extract_cv_metrics(adapter)
                    
                    if cv_metrics:
                        print(f"✓ CV metrics extracted successfully:")
                        print(f"  - Model type: {cv_metrics['model_type']}")
                        print(f"  - Number of folds: {cv_metrics['n_folds']}")
                        print(f"  - Mean RMSE: {cv_metrics['mean_rmse']:.4f}")
                        print(f"  - Std RMSE: {cv_metrics['std_rmse']:.4f}")
                        print(f"  - RMSE scores: {cv_metrics['rmse_scores']}")
                        all_models.append(adapter)
                        model_data_summary[model_type].append({
                            'model_name': model_name,
                            'has_cv': True,
                            'n_folds': cv_metrics['n_folds']
                        })
                    else:
                        print(f"✗ No CV metrics found")
                        
                        # Debug what's available
                        raw_data = adapter.get_raw_model_data()
                        if isinstance(raw_data, dict):
                            print(f"  Available keys: {list(raw_data.keys())}")
                            if 'cv_scores' in raw_data:
                                print(f"  cv_scores present: {raw_data['cv_scores'] is not None}")
                                if raw_data['cv_scores'] is not None:
                                    print(f"  cv_scores type: {type(raw_data['cv_scores'])}")
                                    print(f"  cv_scores: {raw_data['cv_scores']}")
                        
                        model_data_summary[model_type].append({
                            'model_name': model_name,
                            'has_cv': False,
                            'n_folds': 0
                        })
                        
                except Exception as e:
                    print(f"✗ Error getting adapter: {e}")
                    import traceback
                    traceback.print_exc()
                    model_data_summary[model_type].append({
                        'model_name': model_name,
                        'has_cv': False,
                        'error': str(e)
                    })
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for model_type, models in model_data_summary.items():
        total = len(models)
        with_cv = sum(1 for m in models if m.get('has_cv', False))
        print(f"\n{model_type}:")
        print(f"  Total models: {total}")
        print(f"  Models with CV data: {with_cv}")
        if with_cv < total:
            print(f"  Models without CV data:")
            for m in models:
                if not m.get('has_cv', False):
                    print(f"    - {m['model_name']}")
    
    # Test CV distribution plot creation
    print(f"\n\n{'='*60}")
    print("TESTING CV DISTRIBUTION PLOT CREATION")
    print('='*60)
    
    if all_models:
        print(f"Creating CV distribution plot with {len(all_models)} models...")
        cv_plot = CVDistributionPlot(all_models, {'save': False})
        
        # Get unique model types that have CV data
        model_types_with_cv = set()
        for model in all_models:
            metrics = cv_plot._extract_cv_metrics(model)
            if metrics:
                model_types_with_cv.add(metrics['model_type'])
        
        print(f"Model types with CV data: {sorted(model_types_with_cv)}")
        
        # Try to create plots
        try:
            figures = cv_plot.plot()
            print(f"Successfully created {len(figures)} figures:")
            for name in figures.keys():
                print(f"  - {name}")
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No models with CV data found!")

if __name__ == "__main__":
    diagnose_model_cv_data()