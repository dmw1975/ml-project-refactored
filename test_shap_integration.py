#!/usr/bin/env python3
"""Test SHAP integration with enhanced models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import pickle
from config import settings

def test_shap_with_enhanced_models():
    """Test SHAP visualization with enhanced tree models."""
    
    print("Testing SHAP integration with enhanced models...")
    print("-" * 50)
    
    # Test each model type
    model_types = ["xgboost", "lightgbm", "catboost"]
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type.upper()}...")
        try:
            # Load model
            model_file = Path(settings.MODEL_DIR) / f"{model_type}_models.pkl"
            if not model_file.exists():
                print(f"  ❌ Model file not found: {model_file}")
                results[model_type] = "Model not found"
                continue
                
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            
            # Get first model (usually the base model)
            if not models:
                print(f"  ❌ No models found in {model_file}")
                results[model_type] = "No models in file"
                continue
                
            model_name = list(models.keys())[0]
            model_data = models[model_name]
            print(f"  ✓ Loaded model: {model_name}")
            
            # Check if it's from enhanced implementation
            if 'cv_scores' in model_data:
                print(f"  ✓ Enhanced model detected (has CV scores)")
            else:
                print(f"  ⚠️  Standard model (no CV scores)")
            
            # Get model and test data
            model = model_data.get('model')
            X_test = model_data.get('X_test')
            
            if model is None or X_test is None:
                print(f"  ❌ Missing model or test data")
                results[model_type] = "Missing data"
                continue
            
            # Test SHAP
            try:
                # Create explainer
                if model_type == "lightgbm":
                    explainer = shap.TreeExplainer(model)
                elif model_type == "xgboost":
                    explainer = shap.TreeExplainer(model)
                elif model_type == "catboost":
                    explainer = shap.TreeExplainer(model)
                
                # Calculate SHAP values for a small sample
                sample_size = min(100, len(X_test))
                X_sample = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]
                
                shap_values = explainer.shap_values(X_sample)
                
                print(f"  ✓ SHAP values calculated successfully")
                print(f"    Shape: {shap_values.shape}")
                print(f"    Mean absolute SHAP: {np.abs(shap_values).mean():.4f}")
                
                # Create a simple summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 6))
                
                # Save plot
                output_dir = Path(settings.OUTPUT_DIR) / "visualizations" / "shap" / "test"
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_path = output_dir / f"{model_type}_shap_test.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ SHAP plot saved to: {plot_path}")
                results[model_type] = "Success"
                
            except Exception as e:
                print(f"  ❌ SHAP error: {str(e)}")
                results[model_type] = f"SHAP error: {str(e)}"
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[model_type] = f"Error: {str(e)}"
    
    # Summary
    print("\n" + "=" * 50)
    print("SHAP Integration Test Summary:")
    print("-" * 50)
    for model_type, result in results.items():
        status = "✓" if result == "Success" else "✗"
        print(f"{status} {model_type.upper()}: {result}")
    
    # Overall result
    success_count = sum(1 for r in results.values() if r == "Success")
    print(f"\nOverall: {success_count}/{len(model_types)} models passed SHAP test")
    
    return results

if __name__ == "__main__":
    test_shap_with_enhanced_models()