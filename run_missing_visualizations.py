#!/usr/bin/env python3
"""Run missing visualizations for CatBoost and LightGBM models."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io import load_all_models
from src.visualization.pipeline_orchestrator import create_model_visualizations
from src.config import settings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_missing_visualizations():
    """Run visualizations for CatBoost and LightGBM models."""
    
    print("RUNNING MISSING VISUALIZATIONS FOR CATBOOST AND LIGHTGBM")
    print("=" * 80)
    
    # Load all models
    all_models = load_all_models()
    
    # Filter CatBoost and LightGBM models
    target_models = {
        name: data for name, data in all_models.items()
        if 'catboost' in name.lower() or 'lightgbm' in name.lower()
    }
    
    print(f"Found {len(target_models)} CatBoost/LightGBM models")
    
    # Configuration for visualizations
    viz_config = {
        'base_dir': settings.VISUALIZATION_DIR,
        'save': True,
        'show': False,
        'format': 'png',
        'dpi': 300,
        'n_samples': 100,  # For SHAP
        'max_display': 15  # For SHAP
    }
    
    # Track results
    results = {
        'cv_success': 0,
        'shap_success': 0,
        'total': len(target_models)
    }
    
    # Process each model
    for model_name, model_data in target_models.items():
        print(f"\nProcessing {model_name}...")
        
        try:
            viz_results = create_model_visualizations(model_name, model_data, viz_config)
            
            if viz_results['cv_dist']:
                results['cv_success'] += 1
                print(f"  ✓ CV distribution created")
            else:
                print(f"  ✗ CV distribution failed")
                
            if viz_results['shap']:
                results['shap_success'] += 1
                print(f"  ✓ SHAP visualizations created")
            else:
                print(f"  ✗ SHAP visualizations failed")
                
        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Total models processed: {results['total']}")
    print(f"CV distributions created: {results['cv_success']}/{results['total']}")
    print(f"SHAP visualizations created: {results['shap_success']}/{results['total']}")
    
    # Check if files were actually created
    print("\nVERIFYING FILE CREATION:")
    
    # Check CV distributions
    for model_type in ['catboost', 'lightgbm']:
        cv_dir = settings.VISUALIZATION_DIR / 'cv_distributions' / model_type
        if cv_dir.exists():
            cv_files = list(cv_dir.glob('*.png'))
            print(f"  {model_type} CV distributions: {len(cv_files)} files")
        else:
            print(f"  {model_type} CV distributions: Directory not found")
    
    # Check SHAP
    for model_type in ['catboost', 'lightgbm']:
        shap_dir = settings.VISUALIZATION_DIR / 'shap' / model_type
        if shap_dir.exists():
            # Count all subdirectories with plots
            shap_count = 0
            for subdir in shap_dir.iterdir():
                if subdir.is_dir():
                    shap_count += len(list(subdir.glob('*.png')))
            print(f"  {model_type} SHAP plots: {shap_count} files")
        else:
            print(f"  {model_type} SHAP: Directory not found")

if __name__ == "__main__":
    run_missing_visualizations()