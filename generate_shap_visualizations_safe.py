#!/usr/bin/env python3
"""Generate SHAP visualizations for all tree-based models with resource limits."""

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
from src.visualization.plots.shap_plots import SHAPVisualizer, create_model_comparison_shap_plot
from src.visualization.core.style import setup_visualization_style


def check_memory_usage():
    """Check current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # GB


def load_grouped_models():
    """Load all tree-based models from grouped pickle files."""
    models_dir = settings.MODEL_DIR
    all_models = {}
    
    # Load CatBoost models
    catboost_file = models_dir / "catboost_models.pkl"
    if catboost_file.exists():
        with open(catboost_file, 'rb') as f:
            catboost_models = pickle.load(f)
            all_models.update(catboost_models)
            print(f"Loaded {len(catboost_models)} CatBoost models")
    
    # Load LightGBM models
    lightgbm_file = models_dir / "lightgbm_models.pkl"
    if lightgbm_file.exists():
        with open(lightgbm_file, 'rb') as f:
            lightgbm_models = pickle.load(f)
            all_models.update(lightgbm_models)
            print(f"Loaded {len(lightgbm_models)} LightGBM models")
    
    # Load XGBoost models
    xgboost_file = models_dir / "xgboost_models.pkl"
    if xgboost_file.exists():
        with open(xgboost_file, 'rb') as f:
            xgboost_models = pickle.load(f)
            all_models.update(xgboost_models)
            print(f"Loaded {len(xgboost_models)} XGBoost models")
    
    return all_models


def compute_shap_for_model(model_name, model_data, max_samples=30):
    """Compute SHAP values for a single model with resource limits."""
    print(f"\n  Computing SHAP values for {model_name}...")
    print(f"  Memory usage: {check_memory_usage():.2f} GB")
    sys.stdout.flush()
    
    model = model_data.get('model')
    X_test = model_data.get('X_test')
    
    if model is None or X_test is None:
        print(f"    ✗ Missing model or X_test data")
        return None, None
    
    # Limit samples for performance
    if len(X_test) > max_samples:
        sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
        print(f"  Sampling {max_samples} from {len(X_test)} test samples")
    else:
        X_sample = X_test
    
    try:
        # Create appropriate explainer based on model type
        if "CatBoost" in model_name:
            # For CatBoost, skip if it would use generic Explainer (too slow)
            try:
                # First try with tree_path_dependent for categorical support
                explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            except Exception as e:
                print(f"    ✗ CatBoost TreeExplainer failed: {e}")
                print("    Skipping CatBoost SHAP (generic Explainer too slow)")
                return None, None
        elif "LightGBM" in model_name:
            # For LightGBM, use TreeExplainer
            explainer = shap.TreeExplainer(model)
        elif "XGBoost" in model_name:
            # For XGBoost, use TreeExplainer
            explainer = shap.TreeExplainer(model)
        else:
            print(f"    ✗ Unknown model type")
            return None, None
        
        # Compute SHAP values
        print("  Computing SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Handle potential multiple outputs
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        print(f"    ✓ SHAP values computed: shape {shap_values.shape}")
        print(f"  Memory usage after SHAP: {check_memory_usage():.2f} GB")
        
        return shap_values, X_sample
        
    except Exception as e:
        print(f"    ✗ Error computing SHAP values: {str(e)}")
        return None, None


def create_shap_plots(model_name, model_data, shap_values, X_sample):
    """Create SHAP visualizations for a model."""
    print(f"  Generating SHAP plots...")
    sys.stdout.flush()
    
    # Create output directory
    output_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model_name is in model_data
    model_data['model_name'] = model_name
    
    # Initialize visualizer
    visualizer = SHAPVisualizer(model_data)
    
    plots_created = 0
    
    try:
        # 1. Summary plot
        try:
            visualizer.create_shap_summary_plot(
                shap_values, 
                X_sample,
                output_dir / f"{model_name}_shap_summary.png"
            )
            plots_created += 1
            print(f"    ✓ Summary plot created")
        except Exception as e:
            print(f"    ✗ Summary plot failed: {e}")
        
        # 2. Waterfall plot (for first instance)
        try:
            visualizer.create_shap_waterfall_plot(
                shap_values,
                X_sample,
                instance_idx=0,
                output_path=output_dir / f"{model_name}_shap_waterfall.png"
            )
            plots_created += 1
            print(f"    ✓ Waterfall plot created")
        except Exception as e:
            print(f"    ✗ Waterfall plot failed: {e}")
        
        # 3. Dependence plots for top features (limit to 5)
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(5)['feature'].tolist()
        
        for feature in top_features:
            try:
                visualizer.create_shap_dependence_plot(
                    shap_values,
                    X_sample,
                    feature,
                    output_dir / f"{model_name}_shap_dependence_{feature}.png"
                )
                plots_created += 1
            except Exception as e:
                print(f"    ✗ Dependence plot for {feature} failed: {e}")
        
        # 4. Categorical feature plots (if any)
        categorical_features = [col for col in X_sample.columns if 
                              'gics_sector' in col or 
                              'gics_sub_ind' in col or 
                              'cntry_domicile' in col]
        
        for cat_feature in categorical_features[:3]:  # Limit to 3
            try:
                visualizer.create_shap_categorical_plot(
                    shap_values,
                    X_sample,
                    cat_feature,
                    output_dir / f"{model_name}_shap_categorical_{cat_feature}.png"
                )
                plots_created += 1
            except Exception as e:
                print(f"    ✗ Categorical plot for {cat_feature} failed: {e}")
        
        print(f"    Total plots created: {plots_created}")
        return plots_created > 0
        
    except Exception as e:
        print(f"    ✗ Error creating plots: {e}")
        return False
    finally:
        # Clean up matplotlib
        plt.close('all')
        gc.collect()


def main():
    """Main function to generate SHAP visualizations."""
    print("=" * 80)
    print("SHAP Visualization Generation (Safe Mode)")
    print("=" * 80)
    print(f"Initial memory usage: {check_memory_usage():.2f} GB")
    
    # Setup style
    setup_visualization_style()
    
    # Load models
    print("\nLoading models...")
    models = load_grouped_models()
    print(f"Total models loaded: {len(models)}")
    
    # Process each model
    successful = 0
    failed = 0
    skipped = 0
    
    for model_name, model_data in models.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        # Skip if SHAP plots already exist
        output_dir = settings.VISUALIZATION_DIR / 'shap' / model_name
        if output_dir.exists() and len(list(output_dir.glob("*.png"))) > 0:
            print(f"  ⚠ SHAP plots already exist, skipping...")
            skipped += 1
            continue
        
        # Skip CatBoost models in safe mode
        if "CatBoost" in model_name:
            print(f"  ⚠ Skipping CatBoost model (too resource intensive)")
            skipped += 1
            continue
        
        # Compute SHAP values
        shap_values, X_sample = compute_shap_for_model(model_name, model_data)
        
        if shap_values is None:
            print(f"  ✗ Failed to compute SHAP values")
            failed += 1
            continue
        
        # Create plots
        success = create_shap_plots(model_name, model_data, shap_values, X_sample)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Clean up memory after each model
        del shap_values
        if 'X_sample' in locals():
            del X_sample
        gc.collect()
        
        print(f"  Memory usage after cleanup: {check_memory_usage():.2f} GB")
    
    # Create comparison plot
    print("\n" + "="*80)
    print("Creating model comparison plot...")
    try:
        create_model_comparison_shap_plot(models)
        print("✓ Model comparison plot created")
    except Exception as e:
        print(f"✗ Model comparison plot failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SHAP Visualization Generation Complete")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Final memory usage: {check_memory_usage():.2f} GB")
    print("="*80)


if __name__ == "__main__":
    main()