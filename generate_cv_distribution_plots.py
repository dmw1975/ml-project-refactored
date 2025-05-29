#!/usr/bin/env python3
"""Generate CV distribution plots for models that have cross-validation data."""

import os
from pathlib import Path
from visualization_new.utils.io import load_all_models
from visualization_new.plots.cv_distributions import plot_cv_distributions

def generate_cv_plots():
    """Generate CV distribution plots."""
    print("Loading all models...")
    all_models = load_all_models()
    
    # Create output directory
    cv_dir = Path("outputs/visualizations/performance/cv_distributions")
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter models that have CV data
    cv_models = []
    for model_name, model_data in all_models.items():
        if isinstance(model_data, dict):
            # Check for CV scores
            if 'cv_scores' in model_data or 'cv_fold_scores' in model_data:
                cv_models.append(model_data)
                print(f"  Found CV data in: {model_name}")
            elif 'cv_mean' in model_data and 'cv_std' in model_data:
                # Some models might just have summary stats
                print(f"  Found CV summary stats in: {model_name}")
    
    if not cv_models:
        print("No models with CV data found!")
        return
        
    print(f"\nFound {len(cv_models)} models with CV data")
    
    # Create CV distribution plots
    print("\nGenerating CV distribution plots...")
    try:
        # Create config to save plots
        config = {
            'save': True,
            'output_dir': cv_dir,
            'dpi': 300,
            'format': 'png'
        }
        figures = plot_cv_distributions(cv_models, config)
        print(f"✓ CV distribution plots created successfully: {len(figures)} plots")
    except Exception as e:
        print(f"✗ Error creating CV distribution plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_cv_plots()