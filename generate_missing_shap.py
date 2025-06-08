#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate missing SHAP visualizations for CatBoost and LightGBM models."""

import sys
sys.path.append('.')

from pathlib import Path
from src.visualization.plots.shap_plots import create_shap_visualizations
from src.visualization.utils.io import load_all_models

# Configuration
output_base = Path('outputs/visualizations/SHAP')
sample_size = 100

# Load all models
print("Loading all models...")
models = load_all_models()

# Track what we create
created_count = 0
failed_models = []

# Process CatBoost and LightGBM models only
for model_name, model_data in models.items():
    if 'CatBoost' in model_name or 'LightGBM' in model_name:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        
        # Check if visualizations already exist
        model_output_dir = output_base / model_name.replace(' ', '_')
        existing_files = list(model_output_dir.glob('*.png')) if model_output_dir.exists() else []
        
        if existing_files:
            print(f"  Already has {len(existing_files)} visualizations - SKIPPING")
            continue
        
        # Create visualizations
        try:
            print(f"  Creating SHAP visualizations...")
            paths = create_shap_visualizations(model_data, output_base, sample_size=sample_size)
            
            if paths:
                created_count += len(paths)
                print(f"  SUCCESS: Created {len(paths)} visualizations")
                
                # Verify files were created
                actual_files = list(model_output_dir.glob('*.png'))
                if len(actual_files) != len(paths):
                    print(f"  WARNING: Expected {len(paths)} files but found {len(actual_files)}")
            else:
                print(f"  FAILED: No visualizations created")
                failed_models.append(model_name)
                
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_models.append(model_name)
            import traceback
            traceback.print_exc()

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Total visualizations created: {created_count}")
print(f"Models with failures: {len(failed_models)}")
if failed_models:
    print("\nFailed models:")
    for model in failed_models:
        print(f"  - {model}")

# Clean up duplicate directories if they exist
print("\nCleaning up duplicate directories...")
lowercase_dir = Path('outputs/visualizations/shap')
if lowercase_dir.exists() and output_base.exists():
    print(f"  Found both {lowercase_dir} and {output_base}")
    print("  The uppercase 'SHAP' directory is the correct one")