#!/usr/bin/env python3
"""Reorganize feature plots with consistent naming and folder structure."""

import os
import shutil
from pathlib import Path
import re

# Define the source directory
SOURCE_DIR = Path("/mnt/d/ml_project_refactored/outputs/visualizations/features")

# Define the model type patterns
MODEL_PATTERNS = {
    'catboost': re.compile(r'^CatBoost_', re.IGNORECASE),
    'lightgbm': re.compile(r'^LightGBM_', re.IGNORECASE),
    'xgboost': re.compile(r'^XGBoost_', re.IGNORECASE),
    'elasticnet': re.compile(r'^ElasticNet_', re.IGNORECASE),
    'linear': re.compile(r'^LR_', re.IGNORECASE)
}

# Define comparison/aggregate plot patterns
COMPARISON_PATTERNS = [
    re.compile(r'^average_feature_rank_'),
    re.compile(r'^feature_rank_heatmap_'),
    re.compile(r'^top_\d+_features_avg_importance')
]

def get_model_type(filename):
    """Determine the model type from filename."""
    for model_type, pattern in MODEL_PATTERNS.items():
        if pattern.match(filename):
            return model_type
    return None

def is_comparison_plot(filename):
    """Check if the file is a comparison/aggregate plot."""
    for pattern in COMPARISON_PATTERNS:
        if pattern.match(filename):
            return True
    return False

def reorganize_plots():
    """Reorganize feature plots into proper directory structure."""
    
    # Create subdirectories
    subdirs = ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear', 'comparisons']
    for subdir in subdirs:
        (SOURCE_DIR / subdir).mkdir(exist_ok=True)
    
    # Track files to move
    moves = []
    
    # Process all PNG files in the root directory
    for file_path in SOURCE_DIR.glob("*.png"):
        filename = file_path.name
        
        # Check if it's a comparison plot
        if is_comparison_plot(filename):
            dest_dir = SOURCE_DIR / 'comparisons'
            moves.append((file_path, dest_dir / filename))
        else:
            # Check if it's a model-specific plot
            model_type = get_model_type(filename)
            if model_type:
                dest_dir = SOURCE_DIR / model_type
                moves.append((file_path, dest_dir / filename))
    
    # Execute moves
    print(f"Reorganizing {len(moves)} feature plots...")
    for src, dst in moves:
        print(f"  Moving {src.name} to {dst.parent.name}/")
        shutil.move(str(src), str(dst))
    
    # Remove duplicates in subdirectories (keep only the subdirectory version)
    print("\nRemoving duplicates...")
    removed_count = 0
    
    for subdir in ['catboost', 'lightgbm', 'xgboost', 'elasticnet', 'linear']:
        subdir_path = SOURCE_DIR / subdir
        if not subdir_path.exists():
            continue
            
        for file_path in subdir_path.glob("*.png"):
            # Check if this file already exists (was just moved)
            # If it was in the moves list, it means we had a duplicate
            filename = file_path.name
            was_moved = any(dst.name == filename and dst.parent.name == subdir for _, dst in moves)
            
            if not was_moved:
                # This is an original file in the subdirectory, keep it
                continue
    
    print(f"\nReorganization complete!")
    print(f"  Files moved: {len(moves)}")
    
    # Print final structure
    print("\nFinal structure:")
    for subdir in subdirs:
        subdir_path = SOURCE_DIR / subdir
        if subdir_path.exists():
            count = len(list(subdir_path.glob("*.png")))
            if count > 0:
                print(f"  {subdir}/: {count} plots")

def create_readme():
    """Create a README explaining the organization."""
    readme_content = """# Feature Plots Organization

This directory contains feature importance visualizations organized by model type.

## Directory Structure

- **catboost/**: CatBoost model feature importance plots
- **lightgbm/**: LightGBM model feature importance plots
- **xgboost/**: XGBoost model feature importance plots
- **elasticnet/**: ElasticNet model feature importance plots
- **linear/**: Linear Regression model feature importance plots
- **comparisons/**: Cross-model comparison plots and aggregated visualizations

## File Naming Convention

Individual model plots follow this pattern:
`{ModelType}_{DataType}_{RandomFeature?}_{Categorical?}_{Optimization?}_top_features.png`

Where:
- ModelType: CatBoost, LightGBM, XGBoost, ElasticNet_LR, LR
- DataType: Base or Yeo
- RandomFeature: "Random" if random feature included
- Categorical: "categorical" for tree models
- Optimization: "basic" or "optuna" for optimized models

Comparison plots include:
- average_feature_rank_{data_type}_{categorical?}.png
- feature_rank_heatmap_{data_type}_{categorical?}.png
- top_N_features_avg_importance.png
"""
    
    readme_path = SOURCE_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"\nCreated README at {readme_path}")

if __name__ == "__main__":
    print("Feature Plots Reorganization Script")
    print("=" * 50)
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        exit(1)
    
    # Perform reorganization
    reorganize_plots()
    
    # Create README
    create_readme()
    
    print("\nDone!")