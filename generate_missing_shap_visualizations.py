#!/usr/bin/env python3
"""Generate missing SHAP visualizations for LightGBM and CatBoost models."""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.io import load_all_models
from src.config import settings

def check_existing_shap_folders():
    """Check which SHAP folders already exist."""
    shap_dir = settings.VISUALIZATION_DIR / "shap"
    existing_folders = []
    
    if shap_dir.exists():
        for folder in shap_dir.iterdir():
            if folder.is_dir():
                existing_folders.append(folder.name)
    
    return existing_folders, shap_dir

def create_shap_visualizations_for_model(model_name, model_data, output_dir):
    """Create all SHAP visualizations for a single model."""
    
    print(f"\nGenerating SHAP visualizations for {model_name}...")
    
    # Extract model and data
    model = model_data.get('model')
    X_test = model_data.get('X_test')
    X_train = model_data.get('X_train')
    
    if model is None or X_test is None:
        print(f"  ✗ Missing model or test data")
        return False
    
    # Create output directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Handle different model types
        if 'LightGBM' in model_name:
            # For LightGBM Booster objects
            if hasattr(model, 'predict'):
                # Create SHAP explainer for LightGBM
                explainer = shap.TreeExplainer(model)
                
                # Get feature names
                if hasattr(model, 'feature_name'):
                    feature_names = model.feature_name()
                else:
                    feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_test)
                
                # If binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                print(f"  ✗ LightGBM model doesn't have predict method")
                return False
                
        elif 'CatBoost' in model_name:
            # For CatBoost models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Get feature names
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            else:
                feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
        else:
            print(f"  ✗ Unknown model type")
            return False
        
        # 1. Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(model_dir / f'{model_name}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated summary plot")
        
        # 2. Waterfall Plot (for first instance)
        plt.figure(figsize=(10, 6))
        shap_exp = shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value,
                                    data=X_test[0],
                                    feature_names=feature_names)
        shap.waterfall_plot(shap_exp, show=False)
        plt.title(f'SHAP Waterfall Plot - {model_name} (First Instance)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(model_dir / f'{model_name}_shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated waterfall plot")
        
        # 3. Feature Importance Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(model_dir / f'{model_name}_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated feature importance plot")
        
        # 4. Dependence Plots for top 5 features
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        
        for idx in top_features_idx:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False)
            feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
            plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Clean filename
            safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
            plt.savefig(model_dir / f'{model_name}_shap_dependence_{safe_feature_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Generated dependence plots for top 5 features")
        
        # 5. Force Plot (save as HTML)
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test[0], 
                                    feature_names=feature_names)
        shap.save_html(model_dir / f'{model_name}_shap_force.html', force_plot)
        print(f"  ✓ Generated force plot (HTML)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error generating SHAP visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Generate missing SHAP visualizations for LightGBM and CatBoost."""
    
    print("=== Generating Missing SHAP Visualizations ===\n")
    
    # Check existing folders
    existing_folders, shap_dir = check_existing_shap_folders()
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(existing_folders)} existing SHAP folders:")
    for folder in existing_folders:
        print(f"  - {folder}")
    
    # Load all models
    print("\nLoading all models...")
    all_models = load_all_models()
    print(f"Loaded {len(all_models)} models")
    
    # Process LightGBM and CatBoost models
    lightgbm_count = 0
    catboost_count = 0
    generated_count = 0
    
    for model_name, model_data in all_models.items():
        if 'LightGBM' in model_name:
            lightgbm_count += 1
            if model_name not in existing_folders:
                if create_shap_visualizations_for_model(model_name, model_data, shap_dir):
                    generated_count += 1
        elif 'CatBoost' in model_name:
            catboost_count += 1
            if model_name not in existing_folders:
                if create_shap_visualizations_for_model(model_name, model_data, shap_dir):
                    generated_count += 1
    
    print(f"\n=== Summary ===")
    print(f"LightGBM models found: {lightgbm_count}")
    print(f"CatBoost models found: {catboost_count}")
    print(f"SHAP visualizations generated: {generated_count}")
    
    # Verify folders created
    final_folders, _ = check_existing_shap_folders()
    print(f"\nFinal SHAP folders count: {len(final_folders)}")
    
    # Check for specific model types
    lightgbm_folders = [f for f in final_folders if 'LightGBM' in f]
    catboost_folders = [f for f in final_folders if 'CatBoost' in f]
    
    print(f"LightGBM SHAP folders: {len(lightgbm_folders)}")
    print(f"CatBoost SHAP folders: {len(catboost_folders)}")

if __name__ == "__main__":
    main()