#!/usr/bin/env python3
"""Reorganize performance plots with consistent directory structure."""

import os
import shutil
from pathlib import Path
import re

# Define the source directory
SOURCE_DIR = Path("/mnt/d/ml_project_refactored/outputs/visualizations/performance")

# Define the model type patterns
MODEL_PATTERNS = {
    'catboost': re.compile(r'^CatBoost_', re.IGNORECASE),
    'lightgbm': re.compile(r'^LightGBM_', re.IGNORECASE),
    'xgboost': re.compile(r'^XGBoost_', re.IGNORECASE),
    'elasticnet': re.compile(r'^ElasticNet_', re.IGNORECASE),
    'linear': re.compile(r'^LR_', re.IGNORECASE)
}

# Special files that should stay in root
KEEP_IN_ROOT = [
    'metrics_summary_table.png',
    # Add any other files that should stay in root
]

def get_model_type(filename):
    """Determine the model type from filename."""
    for model_type, pattern in MODEL_PATTERNS.items():
        if pattern.match(filename):
            return model_type
    return None

def reorganize_plots():
    """Reorganize performance plots into proper directory structure."""
    
    # Create subdirectories if they don't exist
    subdirs = ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear']
    for subdir in subdirs:
        (SOURCE_DIR / subdir).mkdir(exist_ok=True)
    
    # Also ensure cv_distributions exists
    (SOURCE_DIR / 'cv_distributions').mkdir(exist_ok=True)
    
    # Track files to move
    moves = []
    
    # Process all PNG files in the root directory
    for file_path in SOURCE_DIR.glob("*.png"):
        filename = file_path.name
        
        # Skip files that should stay in root
        if filename in KEEP_IN_ROOT:
            continue
        
        # Check model type
        model_type = get_model_type(filename)
        if model_type:
            dest_dir = SOURCE_DIR / model_type
            dest_path = dest_dir / filename
            
            # Check if file already exists in destination
            if dest_path.exists():
                print(f"  Warning: {filename} already exists in {model_type}/, keeping newer version")
                # Compare modification times
                if file_path.stat().st_mtime > dest_path.stat().st_mtime:
                    moves.append((file_path, dest_path))
                else:
                    # Remove the older file from root
                    file_path.unlink()
            else:
                moves.append((file_path, dest_path))
    
    # Execute moves
    print(f"Reorganizing {len(moves)} performance plots...")
    for src, dst in moves:
        print(f"  Moving {src.name} to {dst.parent.name}/")
        shutil.move(str(src), str(dst))
    
    print(f"\nReorganization complete!")
    print(f"  Files moved: {len(moves)}")
    
    # Print final structure
    print("\nFinal structure:")
    for subdir in subdirs + ['cv_distributions']:
        subdir_path = SOURCE_DIR / subdir
        if subdir_path.exists():
            count = len(list(subdir_path.glob("*.png")))
            if count > 0:
                print(f"  {subdir}/: {count} plots")
    
    # Count root files
    root_count = len(list(SOURCE_DIR.glob("*.png")))
    print(f"  root: {root_count} plots")

def create_readme():
    """Create a README explaining the organization."""
    readme_content = """# Performance Plots Organization

This directory contains performance-related visualizations organized by model type.

## Directory Structure

- **catboost/**: CatBoost model optimization plots
- **lightgbm/**: LightGBM model optimization plots
- **xgboost/**: XGBoost model optimization plots
- **elasticnet/**: ElasticNet model optimization plots
- **linear/**: Linear Regression optimization plots (if any)
- **cv_distributions/**: Cross-validation distribution plots for all models

## File Types

### Optimization Plots (in model subdirectories)
- `*_optuna_optimization_history.png`: Optimization history showing how the objective improved over trials
- `*_optuna_param_importance.png`: Parameter importance analysis from Optuna
- `*_contour.png`: Contour plots showing parameter interactions

### Comparison Plots (in model subdirectories)
- `*_basic_vs_optuna.png`: Comparison between basic and Optuna-optimized models
- `*_best_*_comparison.png`: Comparison of best hyperparameter values

### Other Plots (in root)
- `metrics_summary_table.png`: Overall metrics summary table

## Naming Convention

Optimization plots follow this pattern:
`{ModelType}_{DataType}_{RandomFeature?}_{Categorical?}_optuna_{PlotType}.png`

Where:
- ModelType: CatBoost, LightGBM, XGBoost, ElasticNet_LR
- DataType: Base or Yeo
- RandomFeature: "Random" if random feature included
- Categorical: "categorical" for tree models
- PlotType: optimization_history, param_importance, or contour
"""
    
    readme_path = SOURCE_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"\nCreated README at {readme_path}")

if __name__ == "__main__":
    print("Performance Plots Reorganization Script")
    print("=" * 50)
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        exit(1)
    
    # Perform reorganization
    reorganize_plots()
    
    # Create README
    create_readme()
    
    print("\nDone!")