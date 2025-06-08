#!/usr/bin/env python3
"""Diagnose why CV distribution plots show 'Unknown' model names."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.visualization.utils.io import load_all_models
from src.visualization.core.registry import get_adapter_for_model


def diagnose_model_names():
    """Check how model names are extracted through adapters."""
    print("Loading all models...")
    models = load_all_models()
    
    print(f"\nChecking model name extraction for models with CV data:")
    print("=" * 80)
    
    for key, model_data in models.items():
        if isinstance(model_data, dict):
            # Check for CV data
            has_cv = any(k in model_data for k in ['cv_scores', 'cv_fold_scores', 'cv_mean', 'cv_mse'])
            
            if has_cv:
                print(f"\nModel key: {key}")
                print(f"  - Has 'model_name' field: {'model_name' in model_data}")
                if 'model_name' in model_data:
                    print(f"  - model_name value: {model_data['model_name']}")
                
                # Try to get adapter
                try:
                    adapter = get_adapter_for_model(model_data)
                    print(f"  - Adapter type: {type(adapter).__name__}")
                    
                    # Check adapter's model_name
                    if hasattr(adapter, 'model_name'):
                        print(f"  - Adapter model_name: {adapter.model_name}")
                    else:
                        print(f"  - Adapter has no model_name attribute")
                    
                    # Check what the CV distribution plot would extract
                    if hasattr(adapter, 'name'):
                        print(f"  - Adapter name attribute: {adapter.name}")
                    else:
                        print(f"  - Adapter has no name attribute")
                        
                except Exception as e:
                    print(f"  - Error getting adapter: {e}")


def check_cv_extraction():
    """Check how CV metrics are extracted."""
    from src.visualization.plots.cv_distributions import CVDistributionPlot
    
    print("\n\nChecking CV metric extraction:")
    print("=" * 80)
    
    models = load_all_models()
    
    # Get models with CV data
    cv_models = []
    for key, model_data in models.items():
        if isinstance(model_data, dict):
            if any(k in model_data for k in ['cv_scores', 'cv_fold_scores', 'cv_mean', 'cv_mse']):
                cv_models.append(model_data)
    
    # Create CVDistributionPlot instance
    cv_plot = CVDistributionPlot(cv_models)
    
    # Check what _extract_cv_metrics returns
    for i, model in enumerate(cv_models[:3]):  # Check first 3 models
        print(f"\nExtracting CV metrics for model {i}:")
        metrics = cv_plot._extract_cv_metrics(model)
        if metrics:
            print(f"  - Model name extracted: {metrics['model_name']}")
            print(f"  - Model type: {metrics['model_type']}")
            print(f"  - Dataset: {metrics['dataset']}")
            print(f"  - Number of folds: {metrics['n_folds']}")
        else:
            print("  - No CV metrics extracted!")


if __name__ == "__main__":
    diagnose_model_names()
    check_cv_extraction()