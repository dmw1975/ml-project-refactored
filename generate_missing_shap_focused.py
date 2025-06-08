#!/usr/bin/env python3
"""Focused script to generate missing SHAP visualizations for CatBoost and LightGBM models."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.visualization.plots.shap_plots import SHAPVisualizer
from src.visualization.core.style import setup_visualization_style


def check_memory_usage():
    """Check current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # GB


def load_specific_models(model_types=['catboost', 'lightgbm']):
    """Load only CatBoost and LightGBM models."""
    models_dir = settings.MODEL_DIR
    all_models = {}
    
    if 'catboost' in model_types:
        catboost_file = models_dir / "catboost_models.pkl"
        if catboost_file.exists():
            print(f"Loading CatBoost models from {catboost_file}")
            with open(catboost_file, 'rb') as f:
                catboost_models = pickle.load(f)
                all_models.update(catboost_models)
                print(f"  Loaded {len(catboost_models)} CatBoost models")
    
    if 'lightgbm' in model_types:
        lightgbm_file = models_dir / "lightgbm_models.pkl"
        if lightgbm_file.exists():
            print(f"Loading LightGBM models from {lightgbm_file}")
            with open(lightgbm_file, 'rb') as f:
                lightgbm_models = pickle.load(f)
                all_models.update(lightgbm_models)
                print(f"  Loaded {len(lightgbm_models)} LightGBM models")
    
    return all_models


def compute_shap_safe(model_name, model_data, max_samples=20):
    """Compute SHAP values with extra safety measures for problematic models."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Memory before: {check_memory_usage():.2f} GB")
    
    model = model_data.get('model')
    X_test = model_data.get('X_test')
    
    if model is None or X_test is None:
        print(f"  ✗ Missing model or test data")
        return None, None
    
    # Check if outputs already exist
    output_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
    if output_dir.exists() and len(list(output_dir.glob("*.png"))) >= 5:
        print(f"  ⚠ SHAP plots already exist, skipping...")
        return None, None
    
    # Sample data for safety
    print(f"  Test data shape: {X_test.shape}")
    if len(X_test) > max_samples:
        print(f"  Sampling {max_samples} from {len(X_test)} samples for safety")
        sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
    else:
        X_sample = X_test
    
    try:
        print(f"  Creating SHAP explainer...")
        
        if "CatBoost" in model_name:
            # CatBoost specific handling
            print("  Using CatBoost-specific SHAP configuration...")
            try:
                # Try TreeExplainer first
                explainer = shap.TreeExplainer(model)
                print("  ✓ TreeExplainer created successfully")
            except Exception as e:
                print(f"  ✗ TreeExplainer failed: {str(e)[:100]}")
                print("  Skipping CatBoost model due to compatibility issues")
                return None, None
                
        elif "LightGBM" in model_name:
            # LightGBM specific handling
            print("  Using LightGBM TreeExplainer...")
            explainer = shap.TreeExplainer(model)
            print("  ✓ TreeExplainer created successfully")
        else:
            print(f"  ✗ Unexpected model type: {model_name}")
            return None, None
        
        # Compute SHAP values
        print(f"  Computing SHAP values for {X_sample.shape[0]} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multiple outputs
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        print(f"  ✓ SHAP values computed: shape {shap_values.shape}")
        print(f"  Memory after SHAP: {check_memory_usage():.2f} GB")
        
        return shap_values, X_sample
        
    except Exception as e:
        print(f"  ✗ Error computing SHAP: {str(e)[:200]}")
        return None, None


def create_minimal_shap_plots(model_name, model_data, shap_values, X_sample):
    """Create essential SHAP plots with error handling."""
    print(f"  Creating SHAP visualizations...")
    
    output_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model_name is in model_data
    model_data['model_name'] = model_name
    
    plots_created = 0
    
    # 1. Summary plot (most important)
    try:
        print("    Creating summary plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        plots_created += 1
        print("    ✓ Summary plot created")
    except Exception as e:
        print(f"    ✗ Summary plot failed: {str(e)[:100]}")
    
    # 2. Feature importance bar plot
    try:
        print("    Creating feature importance plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_shap_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        plots_created += 1
        print("    ✓ Feature importance plot created")
    except Exception as e:
        print(f"    ✗ Feature importance plot failed: {str(e)[:100]}")
    
    # 3. Top 3 feature dependence plots
    try:
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-3:][::-1]
        
        for idx in top_indices:
            feature_name = X_sample.columns[idx]
            print(f"    Creating dependence plot for {feature_name}...")
            
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(idx, shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_shap_dependence_{feature_name}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            plots_created += 1
    except Exception as e:
        print(f"    ✗ Dependence plots failed: {str(e)[:100]}")
    
    print(f"  Total plots created: {plots_created}")
    
    # Clean up
    plt.close('all')
    gc.collect()
    
    return plots_created > 0


def main():
    """Generate missing SHAP visualizations for CatBoost and LightGBM."""
    print("="*80)
    print("FOCUSED SHAP GENERATION FOR MISSING MODELS")
    print("="*80)
    print(f"Initial memory: {check_memory_usage():.2f} GB")
    
    # Setup
    setup_visualization_style()
    
    # Load only the models we need
    print("\nLoading models...")
    models = load_specific_models(['catboost', 'lightgbm'])
    
    if not models:
        print("No models found to process!")
        return
    
    # Process each model
    successful = 0
    failed = 0
    
    for model_name, model_data in models.items():
        # Compute SHAP
        shap_values, X_sample = compute_shap_safe(model_name, model_data)
        
        if shap_values is None:
            failed += 1
            continue
        
        # Create plots
        success = create_minimal_shap_plots(model_name, model_data, shap_values, X_sample)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Aggressive cleanup
        del shap_values
        if 'X_sample' in locals():
            del X_sample
        gc.collect()
        
        print(f"Memory after cleanup: {check_memory_usage():.2f} GB\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Final memory: {check_memory_usage():.2f} GB")
    
    # Run validation
    print("\nRunning validation...")
    os.system("./validate_pipeline_outputs.sh | grep -E '(SHAP|CRITICAL)'")


if __name__ == "__main__":
    # Set matplotlib to use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    main()