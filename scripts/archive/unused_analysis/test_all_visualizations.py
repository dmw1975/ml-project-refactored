#!/usr/bin/env python3
"""Test all visualization functions to find any remaining array comparison errors."""

import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization_new.viz_factory import VisualizationFactory
from visualization_new.utils.io import load_all_models

def test_visualization(viz_type, models, config=None):
    """Test a specific visualization type."""
    try:
        if config is None:
            config = {
                'save': False,
                'show': False,
                'format_for_export': False
            }
        
        factory = VisualizationFactory()
        figs = factory.create_visualization(viz_type, models, config)
        print(f"✓ {viz_type}: Successfully created {len(figs) if isinstance(figs, list) else 1} plot(s)")
        return True
    except Exception as e:
        print(f"✗ {viz_type}: Error - {type(e).__name__}: {str(e)}")
        if "truth value" in str(e).lower() and "array" in str(e).lower():
            print("  ^ This is the array comparison error we're looking for!")
            traceback.print_exc()
        return False

def main():
    """Test all visualization types."""
    print("Loading all models...")
    models = load_all_models()
    
    # Get CatBoost models for focused testing
    catboost_models = {k: v for k, v in models.items() if 'catboost' in k.lower()}
    
    print(f"\nLoaded {len(models)} models total, {len(catboost_models)} CatBoost models")
    
    # List of visualization types to test
    viz_types = [
        'residuals',
        'metrics_comparison',
        'feature_importance',
        'predictions',
        'cv_distributions',
        'optimization',
        'baseline_comparison',
        'consolidated_baseline',
        'dataset_comparison',
        'stratification',
        'statistical_tests',
        'sector_analysis'
    ]
    
    print("\nTesting all visualization types with CatBoost models:")
    print("=" * 60)
    
    error_count = 0
    for viz_type in viz_types:
        if not test_visualization(viz_type, catboost_models):
            error_count += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(viz_types) - error_count}/{len(viz_types)} visualization types passed")
    
    if error_count > 0:
        print(f"\n{error_count} visualization type(s) had errors")

if __name__ == "__main__":
    main()