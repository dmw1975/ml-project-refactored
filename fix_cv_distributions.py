#!/usr/bin/env python3
"""Fix missing CV distribution plots for CatBoost and LightGBM models."""

import sys
from pathlib import Path
import pickle
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.plots.cv_distributions import plot_cv_distributions
from src.visualization.core.style import setup_visualization_style

def diagnose_cv_data():
    """Diagnose CV data availability for all models."""
    models_dir = settings.MODEL_DIR
    
    print("=" * 80)
    print("CV DATA DIAGNOSTIC REPORT")
    print("=" * 80)
    
    all_models = {}
    cv_model_count = 0
    
    # Load all model files
    model_files = {
        'catboost': 'catboost_models.pkl',
        'lightgbm': 'lightgbm_models.pkl',
        'xgboost': 'xgboost_models.pkl',
        'elasticnet': 'elasticnet_models.pkl'
    }
    
    for model_type, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models = pickle.load(f)
                all_models.update(models)
                
                print(f"\n{model_type.upper()} Models ({len(models)} total):")
                for name, data in models.items():
                    has_cv = False
                    cv_info = []
                    
                    if isinstance(data, dict):
                        if 'cv_scores' in data and data['cv_scores'] is not None:
                            has_cv = True
                            cv_info.append(f"cv_scores[{len(data['cv_scores'])}]")
                        if 'cv_fold_scores' in data and data['cv_fold_scores'] is not None:
                            has_cv = True
                            cv_info.append(f"cv_fold_scores[{len(data['cv_fold_scores'])}]")
                        if 'cv_mean' in data and data['cv_mean'] is not None:
                            has_cv = True
                            cv_info.append(f"cv_mean={data['cv_mean']:.4f}")
                        if 'cv_std' in data and data['cv_std'] is not None:
                            cv_info.append(f"cv_std={data['cv_std']:.4f}")
                    
                    if has_cv:
                        cv_model_count += 1
                        print(f"  ✓ {name}: {', '.join(cv_info)}")
                    else:
                        print(f"  ✗ {name}: NO CV DATA")
    
    print(f"\n\nSUMMARY: {cv_model_count} out of {len(all_models)} models have CV data")
    return all_models

def generate_cv_distributions():
    """Generate CV distribution plots for models with CV data."""
    print("\n" + "=" * 80)
    print("GENERATING CV DISTRIBUTION PLOTS")
    print("=" * 80)
    
    # Load all models
    all_models = diagnose_cv_data()
    
    # Filter models with CV data
    cv_models = {}
    for name, data in all_models.items():
        if isinstance(data, dict):
            if any(key in data for key in ['cv_scores', 'cv_fold_scores', 'cv_mean']):
                cv_models[name] = data
    
    print(f"\nModels with CV data: {len(cv_models)}")
    
    if not cv_models:
        print("No models with CV data found!")
        return
    
    # Setup visualization style
    setup_visualization_style()
    
    # Generate CV distribution plots by model type
    model_types = {
        'catboost': [name for name in cv_models.keys() if 'CatBoost' in name],
        'lightgbm': [name for name in cv_models.keys() if 'LightGBM' in name],
        'xgboost': [name for name in cv_models.keys() if 'XGBoost' in name],
        'elasticnet': [name for name in cv_models.keys() if 'ElasticNet' in name]
    }
    
    output_dir = settings.VISUALIZATION_DIR / 'performance' / 'cv_distributions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type, model_names in model_types.items():
        if model_names:
            print(f"\nGenerating {model_type} CV distribution plot...")
            type_models = {name: cv_models[name] for name in model_names}
            
            try:
                output_path = output_dir / f"{model_type}_cv_distribution.png"
                # Convert to list format expected by plot_cv_distributions
                model_list = list(type_models.values())
                plot_cv_distributions(model_list, save_path=output_path)
                print(f"  ✓ Saved to: {output_path}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

def verify_outputs():
    """Verify that CV distribution plots were created."""
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    output_dir = settings.VISUALIZATION_DIR / 'performance' / 'cv_distributions'
    
    expected_files = [
        'catboost_cv_distribution.png',
        'lightgbm_cv_distribution.png',
        'xgboost_cv_distribution.png',
        'elasticnet_cv_distribution.png'
    ]
    
    print("CV Distribution Files:")
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")

if __name__ == "__main__":
    generate_cv_distributions()
    verify_outputs()