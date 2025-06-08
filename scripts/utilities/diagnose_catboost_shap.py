#!/usr/bin/env python3
"""Diagnose why CatBoost SHAP plots are missing."""

import sys
import pickle
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings
from scripts.utilities.generate_shap_visualizations import load_grouped_models, compute_shap_for_model


def diagnose_catboost_shap():
    """Diagnose issues with CatBoost SHAP generation."""
    print("Diagnosing CatBoost SHAP generation issues...\n")
    
    # Load all models
    all_models = load_grouped_models()
    
    # Get CatBoost models that are missing plots
    missing_plots_models = [
        "CatBoost_Base_Random_categorical_basic",
        "CatBoost_Base_Random_categorical_optuna",
        "CatBoost_Yeo_Random_categorical_basic",
        "CatBoost_Yeo_Random_categorical_optuna",
        "CatBoost_Yeo_categorical_basic",
        "CatBoost_Yeo_categorical_optuna"
    ]
    
    # Models that have plots for comparison
    working_models = [
        "CatBoost_Base_categorical_basic",
        "CatBoost_Base_categorical_optuna"
    ]
    
    print("Checking models with missing plots:")
    print("=" * 60)
    
    for model_name in missing_plots_models:
        print(f"\nModel: {model_name}")
        
        if model_name not in all_models:
            print(f"  ✗ Model not found in loaded models!")
            continue
        
        model_data = all_models[model_name]
        
        # Check model components
        print(f"  Model object: {'✓' if model_data.get('model') is not None else '✗'}")
        print(f"  X_test data: {'✓' if model_data.get('X_test') is not None else '✗'}")
        
        if model_data.get('X_test') is not None:
            X_test = model_data['X_test']
            print(f"  X_test shape: {X_test.shape}")
            print(f"  X_test type: {type(X_test)}")
            
            # Check for data issues
            if hasattr(X_test, 'isnull'):
                null_count = X_test.isnull().sum().sum()
                print(f"  Null values in X_test: {null_count}")
            
            # Try to compute SHAP values
            print("  Attempting SHAP computation...")
            try:
                shap_values, X_sample = compute_shap_for_model(model_name, model_data, max_samples=10)
                if shap_values is not None:
                    print(f"    ✓ SHAP computation successful! Shape: {shap_values.shape}")
                else:
                    print(f"    ✗ SHAP computation returned None")
            except Exception as e:
                print(f"    ✗ SHAP computation failed: {str(e)}")
    
    print("\n\nChecking working models for comparison:")
    print("=" * 60)
    
    for model_name in working_models:
        print(f"\nModel: {model_name}")
        
        if model_name not in all_models:
            print(f"  ✗ Model not found!")
            continue
        
        model_data = all_models[model_name]
        print(f"  Model object: {'✓' if model_data.get('model') is not None else '✗'}")
        print(f"  X_test data: {'✓' if model_data.get('X_test') is not None else '✗'}")
        
        if model_data.get('X_test') is not None:
            X_test = model_data['X_test']
            print(f"  X_test shape: {X_test.shape}")
            print(f"  X_test type: {type(X_test)}")
    
    # Check if there's a pattern in the data
    print("\n\nAnalyzing patterns:")
    print("=" * 60)
    
    # Check all CatBoost models
    catboost_models = {name: data for name, data in all_models.items() if "CatBoost" in name}
    
    for model_name, model_data in catboost_models.items():
        has_plots = model_name in working_models
        has_random = "Random" in model_name
        has_yeo = "Yeo" in model_name
        
        print(f"\n{model_name}:")
        print(f"  Has plots: {has_plots}")
        print(f"  Has 'Random': {has_random}")
        print(f"  Has 'Yeo': {has_yeo}")
        
        if model_data.get('X_test') is not None:
            X_test = model_data['X_test']
            # Check feature names
            if hasattr(X_test, 'columns'):
                sample_features = list(X_test.columns)[:5]
                print(f"  Sample features: {sample_features}")
                
                # Check for Yeo-Johnson transformed features
                yeo_features = [col for col in X_test.columns if 'yeo' in col.lower()]
                print(f"  Yeo-Johnson features: {len(yeo_features)}")


if __name__ == "__main__":
    diagnose_catboost_shap()