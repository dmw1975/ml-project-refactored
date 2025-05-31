#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify comprehensive visualization pipeline creates all expected visualizations.
"""

import sys
sys.path.append('.')

from pathlib import Path
from src.visualization import run_comprehensive_visualization_pipeline
from src.config import settings

def main():
    print("="*60)
    print("TESTING COMPREHENSIVE VISUALIZATION PIPELINE")
    print("="*60)
    
    # Check existing visualizations before
    viz_dir = settings.VISUALIZATION_DIR
    before_count = {}
    for subdir in viz_dir.iterdir():
        if subdir.is_dir():
            count = len(list(subdir.glob("**/*.png")))
            if count > 0:
                before_count[subdir.name] = count
    
    print("\nExisting visualizations:")
    for name, count in sorted(before_count.items()):
        print(f"  {name}: {count} files")
    print(f"  Total: {sum(before_count.values())} files")
    
    # Run the comprehensive pipeline
    print("\n" + "="*60)
    print("Running comprehensive visualization pipeline...")
    print("="*60)
    
    try:
        result = run_comprehensive_visualization_pipeline()
        
        # Check created visualizations
        print("\n" + "="*60)
        print("VERIFICATION OF CREATED VISUALIZATIONS")
        print("="*60)
        
        # Count files after
        after_count = {}
        for subdir in viz_dir.iterdir():
            if subdir.is_dir():
                count = len(list(subdir.glob("**/*.png")))
                if count > 0:
                    after_count[subdir.name] = count
        
        print("\nVisualization files by directory:")
        for name, count in sorted(after_count.items()):
            diff = count - before_count.get(name, 0)
            diff_str = f" (+{diff})" if diff > 0 else ""
            print(f"  {name}: {count} files{diff_str}")
        print(f"  Total: {sum(after_count.values())} files")
        
        # Check the result dictionary
        print("\nVisualization types in result:")
        total_in_result = 0
        for viz_type, paths in result.items():
            if isinstance(paths, dict):
                count = sum(len(p) for p in paths.values())
            elif isinstance(paths, list):
                count = len(paths)
            else:
                count = 1 if paths else 0
            total_in_result += count
            print(f"  {viz_type}: {count} items")
        print(f"  Total in result: {total_in_result} items")
        
        # Check if all expected types are present
        expected_types = [
            'residual_plots', 'feature_importance', 'cv_distributions', 'shap',
            'model_comparison', 'metrics_table', 'sector_plots', 'dataset_comparison',
            'statistical_tests', 'baseline_comparison', 'sector_weights', 
            'optimization', 'dashboard'
        ]
        
        missing_types = [t for t in expected_types if t not in result]
        if missing_types:
            print(f"\nWARNING: Missing visualization types: {missing_types}")
        else:
            print(f"\n✓ All {len(expected_types)} expected visualization types were created!")
        
        # Check for empty results
        empty_types = [t for t, v in result.items() if not v or (isinstance(v, list) and len(v) == 0)]
        if empty_types:
            print(f"\nWARNING: Empty visualization types: {empty_types}")
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()