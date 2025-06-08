#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate SHAP visualizations for one model type at a time with smaller samples."""

import sys
sys.path.append('.')

import pickle
from pathlib import Path
from src.visualization.plots.shap_plots import create_shap_visualizations

# Configuration
output_base = Path('outputs/visualizations/SHAP')
sample_size = 50  # Reduced sample size for faster processing

# Process just CatBoost first
model_file = Path('outputs/models/catboost_models.pkl')
print(f"Loading CatBoost models from {model_file}...")

with open(model_file, 'rb') as f:
    models = pickle.load(f)

# Process first 2 CatBoost models as a test
test_models = list(models.items())[:2]

for model_name, model_data in test_models:
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    
    # Check if already exists
    model_output_dir = output_base / model_name.replace(' ', '_')
    if model_output_dir.exists() and list(model_output_dir.glob('*.png')):
        print(f"  Already has visualizations - SKIPPING")
        continue
    
    try:
        print(f"  Creating SHAP visualizations with {sample_size} samples...")
        paths = create_shap_visualizations(model_data, output_base, sample_size=sample_size)
        
        if paths:
            print(f"  SUCCESS: Created {len(paths)} visualizations")
            # List first 3 files
            for p in paths[:3]:
                print(f"    - {p.name}")
        else:
            print(f"  FAILED: No visualizations created")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\nDone! Run this script again with different model selections to process more models.")