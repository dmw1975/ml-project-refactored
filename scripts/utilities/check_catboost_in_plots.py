#!/usr/bin/env python3
"""
Check if CatBoost is included in the baseline comparison plots.
"""

import os
from PIL import Image
import numpy as np

# Check if plots exist
baselines_dir = "/mnt/d/ml_project_refactored/outputs/visualizations/baselines"
plots = ["RMSE_consolidated_baseline_comparison.png", 
         "MAE_consolidated_baseline_comparison.png",
         "R²_consolidated_baseline_comparison.png"]

print("Checking baseline comparison plots...")
print("=" * 50)

for plot_name in plots:
    plot_path = os.path.join(baselines_dir, plot_name)
    if os.path.exists(plot_path):
        # Get file info
        size = os.path.getsize(plot_path)
        print(f"\n✓ {plot_name}")
        print(f"  Size: {size:,} bytes")
        
        # Open and check image dimensions
        try:
            with Image.open(plot_path) as img:
                print(f"  Dimensions: {img.size[0]} x {img.size[1]} pixels")
                print(f"  Mode: {img.mode}")
        except Exception as e:
            print(f"  Error reading image: {e}")
    else:
        print(f"\n✗ {plot_name} - NOT FOUND")

# Also check the baseline comparison CSV to confirm CatBoost is there
print("\n" + "=" * 50)
print("Checking baseline comparison data...")

import pandas as pd
baseline_csv = "/mnt/d/ml_project_refactored/outputs/metrics/baseline_comparison.csv"

if os.path.exists(baseline_csv):
    df = pd.read_csv(baseline_csv)
    
    # Count models by type
    model_counts = {}
    for model in df['Model'].unique():
        if 'CatBoost' in model:
            model_type = 'CatBoost'
        elif 'XGBoost' in model or 'XGB' in model:
            model_type = 'XGBoost'
        elif 'LightGBM' in model:
            model_type = 'LightGBM'
        elif 'ElasticNet' in model:
            model_type = 'ElasticNet'
        elif 'LR_' in model:
            model_type = 'LinearRegression'
        else:
            model_type = 'Other'
        
        model_counts[model_type] = model_counts.get(model_type, 0) + 1
    
    print("\nModels in baseline comparison:")
    for model_type, count in sorted(model_counts.items()):
        print(f"  {model_type}: {count} models")
    
    # Show best performing models
    print("\nTop 5 best performing models (by RMSE):")
    top_models = df.nsmallest(5, 'RMSE')[['Model', 'RMSE', 'Baseline RMSE', 'Improvement (%)']]
    for idx, row in top_models.iterrows():
        model_type = 'CatBoost' if 'CatBoost' in row['Model'] else row['Model'].split('_')[0]
        print(f"  {model_type}: RMSE={row['RMSE']:.4f}, Improvement={row['Improvement (%)']: .2f}%")
else:
    print("Baseline comparison CSV not found!")

print("\n✅ Analysis complete!")